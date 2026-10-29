"""
Background agent runner for the ThermoML browser.

Manages launching analysis-agent runs in background threads and polling
their progress via the session directory's ``_working_memory.md`` file
(which is synced to disk after every tool call by the agent).
"""

import datetime as dt
import csv
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Optional
from zipfile import BadZipFile

from ThermoML_results_browser.helpers.benchmark_zip import (
    ZipPath, as_path, is_within, path_identity,
)

# Data and agent code are siblings of the browser in this checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE = _REPO_ROOT / "ThermoML_research_agent"

from NIST_ThermoML_agents.general_db_query_engine.general_hooks_management_helpers.general_memory_management_tools_hooks_helpers.session_manager_output_storage import (
    ensure_directory,
)
from NIST_ThermoML_agents.general_db_query_engine.general_hooks_management_helpers.general_tracking_hooks_output_helpers.output_writers import (
    prepare_result_answer,
    render_answer_envelope_markdown,
    structured_envelope_section_lines,
    write_result_bundle,
)
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"})
_ARTIFACT_SUFFIXES = {
    "plots": _IMAGE_SUFFIXES,
    "data": frozenset({".csv"}),
}


def discover_run_artifacts(artifact_base: Path | str) -> tuple[list[str], list[str]]:
    """Return browser-safe image and CSV keys below one run directory.

    Analysis runs write directly to ``plots/`` and ``data/``. Main runs keep
    delegated Analysis sessions below ``analysis_runs/run_N/``, so limiting
    discovery to the run root makes completed Main runs incorrectly look as
    though they generated no files. Root-level files retain their historical
    basename-only keys; nested files use a POSIX relative path so duplicate
    filenames from different delegated runs remain addressable.
    """
    base = as_path(artifact_base)
    images: list[str] = []
    csv_files: list[str] = []
    if not base.is_dir():
        return images, csv_files

    if isinstance(base, ZipPath):
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(base)
            parts = relative.parts
            if len(parts) < 2 or any(part in _ARTIFACT_SUFFIXES for part in parts[:-2]):
                continue
            kind = parts[-2]
            allowed = _ARTIFACT_SUFFIXES.get(kind)
            if allowed is None or path.suffix.lower() not in allowed:
                continue
            key = path.name if len(parts) == 2 else relative.as_posix()
            (images if kind == "plots" else csv_files).append(key)
        return sorted(images, key=str.casefold), sorted(csv_files, key=str.casefold)

    try:
        for current_root, dir_names, file_names in os.walk(base, followlinks=False):
            current = Path(current_root)
            kind = current.name
            allowed = _ARTIFACT_SUFFIXES.get(kind)
            if allowed is None:
                continue

            # Generated artefact folders are flat. Avoid scanning anything a
            # tool may have placed beneath one, and never follow symlink dirs.
            dir_names[:] = []
            target = images if kind == "plots" else csv_files
            for name in file_names:
                path = current / name
                if path.suffix.lower() not in allowed:
                    continue
                relative = path.relative_to(base)
                key = name if relative.parent == Path(kind) else relative.as_posix()
                target.append(key)
    except OSError as exc:
        log.debug("Failed to discover generated artefacts below %s: %s", base, exc)

    images.sort(key=str.casefold)
    csv_files.sort(key=str.casefold)
    return images, csv_files


def resolve_artifact_file(
    artifact_base: Path | str,
    artifact_key: str,
    kind: str,
) -> Optional[Path]:
    """Resolve a discovery key without allowing access outside its run.

    ``artifact_key`` may be a legacy root-level basename or the nested POSIX
    path returned by :func:`discover_run_artifacts`.
    """
    allowed = _ARTIFACT_SUFFIXES.get(kind)
    if allowed is None or not artifact_key:
        return None

    key = PurePosixPath(str(artifact_key).replace("\\", "/"))
    if key.is_absolute() or any(part in {"", ".", ".."} for part in key.parts):
        return None
    if len(key.parts) == 1:
        relative = PurePosixPath(kind) / key
    elif len(key.parts) >= 2 and key.parts[-2] == kind:
        relative = key
    else:
        return None

    base = as_path(artifact_base)
    try:
        candidate = base.joinpath(*relative.parts)
        if (candidate.suffix.lower() not in allowed or not candidate.is_file()
                or not is_within(candidate, base)):
            return None
    except (OSError, ValueError):
        return None
    return candidate




# ---------------------------------------------------------------------------
# Global run registry
# ---------------------------------------------------------------------------

_runs: dict = {}
_lock = threading.Lock()

# Only one analysis can run at a time (module-level session singleton)
_active_run_id: Optional[str] = None

log = logging.getLogger("browser-agent-runner")


# ---------------------------------------------------------------------------
# Log-capture handler — funnels agent log lines into an AgentRun buffer
# ---------------------------------------------------------------------------

class _RunLogHandler(logging.Handler):
    """Captures log records into an :class:`AgentRun` browser buffer."""

    def __init__(self, run_obj: "AgentRun"):
        super().__init__(level=logging.DEBUG)
        self._run_obj = run_obj
        self.setFormatter(logging.Formatter(
            "%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord):
        try:
            self._run_obj.append_log(self.format(record))
        except Exception:
            pass


AGENT_MODEL_OPTIONS = [
    {"value": "", "label": "Default (agent config)",
     "description": "Use each ThermoML agent's configured default model."},
    {"value": "claudeopus47", "label": "claudeopus47",
     "description": "Claude Opus model, when available on Argo."},
    {"value": "claudeopus46", "label": "claudeopus46",
     "description": "Current ThermoML default."},
    {"value": "claudesonnet46", "label": "claudesonnet46",
     "description": "Lower-cost Claude Sonnet option."},
    {"value": "gpt54", "label": "gpt54",
     "description": "Argo GPT-5.4 alias."},
    {"value": "gpt5", "label": "gpt5",
     "description": "Argo GPT-5 alias."},
    {"value": "gpt41", "label": "gpt41",
     "description": "Argo GPT-4.1 alias."},
]

_ALLOWED_AGENT_MODELS = {
    option["value"] for option in AGENT_MODEL_OPTIONS if option["value"]
}


def _latest_result_md_path(directory: Path, prefix: str = "") -> Path:
    """Select the highest numeric continuation result in one run directory."""
    directory = as_path(directory)
    fallback = directory / f"{prefix}result.md"
    best_path = fallback
    best_round = 1 if fallback.is_file() else -1
    if not directory.is_dir():
        return fallback
    name_re = re.compile(
        rf"^{re.escape(prefix)}result(?:_R([1-9][0-9]*))?\.md$"
    )
    try:
        candidates = directory.iterdir()
        for candidate in candidates:
            match = name_re.fullmatch(candidate.name)
            if match is None or match.group(1) is None or not candidate.is_file():
                continue
            round_number = int(match.group(1))
            if round_number > best_round:
                best_path = candidate
                best_round = round_number
    except OSError:
        return fallback
    return best_path

_MODEL_FIELDS_BY_AGENT = {
    "query": ("MODEL", "L1_MODEL", "L2_MODEL", "VERDICT_MODEL", "PLANNER_MODEL"),
    "analysis": ("MODEL", "L1_MODEL", "VERDICT_MODEL", "PLANNER_MODEL"),
    "main": ("MODEL", "VERDICT_MODEL", "PLANNER_MODEL", "MENU_PLANNER_MODEL"),
}


def normalize_model_choice(value: str | None) -> str | None:
    """Return a supported model alias or ``None`` for the agent default."""
    model = (value or "").strip()
    if not model:
        return None
    if model not in _ALLOWED_AGENT_MODELS:
        raise ValueError(f"Unsupported ThermoML agent model: {model}")
    return model


class AgentRun:
    """Tracks the state of a single agent run."""

    def __init__(self, run_id: str, question: str, agent_type: str = "analysis",
                 run_verdict: bool = True, time_overrides: dict | None = None,
                 model_override: str | None = None,
                 api_user: str | None = None):
        self.run_id = run_id
        self.question = question
        self.agent_type = agent_type
        self.run_verdict = run_verdict
        self.time_overrides = time_overrides or {}
        self.model_override = normalize_model_choice(model_override)
        self.api_user = (api_user or "").strip() or None
        self.status = "starting"        # starting | running | completed | error
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.result = None               # AnalysisRunResult when done
        self.error: Optional[str] = None
        self.session_dir: Optional[str] = None
        self.progress_log: list[dict] = []
        self.reasoning_snapshots: list[dict] = []  # {iteration, snippet}
        self._last_wm_content = ""
        self._last_wm_mtime: float = 0
        self._last_history_mtime: float = 0
        self._seen_tool_keys: set = set()
        self._thread: Optional[threading.Thread] = None
        # Incremental tracking for SSE streaming
        self._sent_history_len: int = 0
        self._sent_reasoning_len: int = 0
        # Terminal log capture. Sequence numbers make SSE reconnects and
        # multiple viewers independent instead of sharing a destructive cursor.
        self._log_lock = threading.Lock()
        self._log_buf: deque[tuple[int, str]] = deque(maxlen=20_000)
        self._log_seq: int = 0

    # -- computed properties ------------------------------------------------

    @property
    def elapsed(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    def append_log(self, line: str) -> None:
        """Append one terminal line and assign it a monotonic sequence ID."""
        with self._log_lock:
            self._log_seq += 1
            self._log_buf.append((self._log_seq, line))

    def logs_since(self, last_seq: int) -> tuple[int, list[str]]:
        """Return log lines newer than *last_seq* without mutating run state."""
        with self._log_lock:
            new_last = last_seq
            lines: list[str] = []
            for seq, line in self._log_buf:
                if seq > last_seq:
                    lines.append(line)
                    new_last = seq
            return new_last, lines

    @property
    def images(self) -> list[str]:
        if not self.session_dir:
            return []
        return discover_run_artifacts(self.session_dir)[0]

    @property
    def data_files(self) -> list[str]:
        if not self.session_dir:
            return []
        return discover_run_artifacts(self.session_dir)[1]

    def artifacts(self) -> tuple[list[str], list[str]]:
        """Discover both artefact kinds in one filesystem walk."""
        if not self.session_dir:
            return [], []
        return discover_run_artifacts(self.session_dir)

    # -- working-memory polling --------------------------------------------

    def _discover_session_dir(self):
        """Try to find the session directory from the running agent."""
        if self.session_dir:
            return
        try:
            from NIST_ThermoML_agents.general_db_query_engine.general_hooks_management_helpers.general_memory_management_tools_hooks_helpers import (
                session_manager_output_storage as sm,
            )
            sess = sm.get_session()
            if sess and sess.session_dir and sess.session_dir.exists():
                self.session_dir = str(sess.session_dir)
        except Exception as exc:
            log.debug("Failed to discover active session directory: %s", exc)

    def check_working_memory(self) -> Optional[str]:
        """Return the current working memory content (or None if unchanged)."""
        self._discover_session_dir()
        if not self.session_dir:
            return None
        # Query agent uses "working_memory.md"; analysis/main use "_working_memory.md"
        wm_path = None
        for name in ("_working_memory.md", "working_memory.md"):
            p = Path(self.session_dir) / name
            if p.exists():
                wm_path = p
                break
        if wm_path is None:
            return None
        try:
            mtime = wm_path.stat().st_mtime
            if mtime <= self._last_wm_mtime:
                return None
            self._last_wm_mtime = mtime
            content = wm_path.read_text(encoding="utf-8", errors="replace")
            if content != self._last_wm_content:
                self._last_wm_content = content
                return content
        except OSError as exc:
            log.debug("Failed to poll working memory from %s: %s", wm_path, exc)
        return None

    def get_working_memory(self) -> str:
        self._discover_session_dir()
        if not self.session_dir:
            return ""
        for name in ("_working_memory.md", "working_memory.md"):
            wm_path = Path(self.session_dir) / name
            if wm_path.exists():
                try:
                    return wm_path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    log.debug("Failed to read working memory from %s: %s", wm_path, exc)
        return ""

    def get_result_md(self) -> str:
        self._discover_session_dir()
        if not self.session_dir:
            return ""
        p = _latest_result_md_path(Path(self.session_dir))
        if not p.exists():
            return ""
        text = p.read_text(encoding="utf-8", errors="replace")
        return render_ledger_tables(_render_postanswer_envelope(text))

    def get_reasoning_md(self) -> str:
        """Return the live-updated reasoning_tokens_stripped.md content."""
        self._discover_session_dir()
        if not self.session_dir:
            return ""
        p = Path(self.session_dir) / "reasoning_tokens_stripped.md"
        if not p.exists():
            return ""
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.debug("Failed to read reasoning log from %s: %s", p, exc)
            return ""

    def get_reference_stats_md(self) -> str:
        """Return the live-updated reference_stats.md content."""
        self._discover_session_dir()
        if not self.session_dir:
            return ""
        p = Path(self.session_dir) / "reference_stats.md"
        if not p.exists():
            return ""
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.debug("Failed to read reference stats from %s: %s", p, exc)
            return ""

    def _history_path(self) -> Optional[Path]:
        if not self.session_dir:
            return None
        base = Path(self.session_dir)
        return base / "run_history.md"

    def get_history_md(self) -> str:
        p = self._history_path()
        if not p:
            return ""
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

    def check_history_for_activity(self) -> None:
        """Parse run_history.md for new tool calls and add reasoning snapshots."""
        self._discover_session_dir()
        if not self.session_dir:
            return
        hist_path = self._history_path()
        if not hist_path:
            return
        if not hist_path.exists():
            return
        try:
            mtime = hist_path.stat().st_mtime
            if mtime <= self._last_history_mtime:
                return
            self._last_history_mtime = mtime
            content = hist_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.debug("Failed to inspect history file %s: %s", hist_path, exc)
            return
        # Parse step headers: "### Step 3: `search_compounds`"
        # followed by "- **Result:** 1,234 chars  |  **Time:** 2.1s"
        current_iter = None
        current_tool = None
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("### Step "):
                # e.g.  "### Step 3: `search_compounds`"
                #  or   "### Step 3 [L1-query #1 · c9]: `search_compounds`"
                rest = stripped[9:]  # after "### Step "
                colon_pos = rest.find(":")
                if colon_pos > 0:
                    head = rest[:colon_pos].split("[")[0].strip()
                    try:
                        current_iter = int(head)
                    except ValueError:
                        current_iter = None
                        continue
                    # Extract tool name from backticks
                    bt1 = rest.find("`", colon_pos)
                    bt2 = rest.find("`", bt1 + 1) if bt1 >= 0 else -1
                    current_tool = rest[bt1 + 1:bt2] if bt2 > bt1 >= 0 else rest[colon_pos + 1:].strip()
            elif stripped.startswith("- **Result:**") and current_iter is not None and current_tool:
                key = f"{current_iter}_{current_tool}"
                if key not in self._seen_tool_keys:
                    self._seen_tool_keys.add(key)
                    # Extract chars and time from "- **Result:** 1,234 chars  |  **Time:** 2.1s"
                    detail = stripped.replace("- **Result:**", "").strip()
                    snippet = f"{current_tool}  —  {detail}"
                    self.reasoning_snapshots.append({
                        "iteration": current_iter,
                        "snippet": snippet,
                    })
                current_iter = None
                current_tool = None

    # -- incremental getters for SSE streaming ----------------------------

    def get_new_history_lines(self) -> list[str]:
        """Return only the new lines from run_history.md since the last call."""
        self._discover_session_dir()
        if not self.session_dir:
            return []
        hist_path = self._history_path()
        if not hist_path or not hist_path.exists():
            return []
        try:
            content = hist_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        if len(content) <= self._sent_history_len:
            return []
        new_text = content[self._sent_history_len:]
        self._sent_history_len = len(content)
        return [ln for ln in new_text.splitlines() if ln.strip()]

    def get_new_reasoning_text(self) -> str:
        """Return only the new portion of reasoning_tokens_stripped.md."""
        self._discover_session_dir()
        if not self.session_dir:
            return ""
        p = Path(self.session_dir) / "reasoning_tokens_stripped.md"
        if not p.exists():
            return ""
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        if len(content) <= self._sent_reasoning_len:
            return ""
        new_text = content[self._sent_reasoning_len:]
        self._sent_reasoning_len = len(content)
        return new_text

# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def start_run(question: str, agent_type: str = "analysis",
              run_verdict: bool = True,
              time_overrides: dict | None = None,
              model_override: str | None = None) -> str:
    """Start an agent run in a background thread.  Returns the run_id."""
    global _active_run_id

    with _lock:
        if _active_run_id and _active_run_id in _runs:
            prev = _runs[_active_run_id]
            if prev.status in ("starting", "running"):
                raise RuntimeError("An agent run is already in progress.")

    run_id = uuid.uuid4().hex[:12]
    api_user = _current_default_api_user()
    run_obj = AgentRun(run_id, question, agent_type, run_verdict,
                       time_overrides, model_override, api_user=api_user)
    with _lock:
        _runs[run_id] = run_obj
        _active_run_id = run_id

    thread = threading.Thread(target=_execute_run, args=(run_obj,), daemon=True)
    run_obj._thread = thread
    thread.start()
    return run_id


def _apply_time_overrides(overrides: dict) -> None:
    """Apply user time-limit overrides to the active EngineConfig singleton."""
    if not overrides:
        return
    from NIST_ThermoML_agents.general_db_query_engine.general_argo_engine_helpers import get_config
    try:
        cfg = get_config()
    except RuntimeError:
        # Config not yet loaded — agent hasn't initialised yet; skip.
        return
    for key, val in overrides.items():
        if hasattr(cfg, key) and isinstance(val, (int, float)):
            setattr(cfg, key, int(val))


def _agent_config(agent_type: str):
    """Return the mutable config singleton for one ThermoML agent."""
    if agent_type == "query":
        from NIST_ThermoML_agents.NIST_ThermoML_query_agent.ThermoML_query_argo_config import (
            AGENT_CONFIG,
        )
        return AGENT_CONFIG
    if agent_type == "analysis":
        from NIST_ThermoML_agents.NIST_ThermoML_analysis_agent.ThermoML_analysis_argo_config import (
            AGENT_CONFIG,
        )
        return AGENT_CONFIG
    if agent_type == "main":
        from NIST_ThermoML_agents._NIST_ThermoML_main_agent.ThermoML_main_argo_config import (
            AGENT_CONFIG,
        )
        return AGENT_CONFIG
    raise ValueError(f"Unknown agent type: {agent_type}")


def _current_default_api_user() -> str:
    """Use the provider identity configured for the server process."""
    return os.environ.get("ARGO_API_USER", "").strip()



def _apply_api_user_override(api_user: str | None) -> dict:
    """Apply an Argo username to all ThermoML config singletons."""
    api_user = (api_user or "").strip()
    if not api_user:
        return {}
    previous = {"env": os.environ.get("ARGO_API_USER")}
    os.environ["ARGO_API_USER"] = api_user
    for agent_type in ("query", "analysis", "main"):
        try:
            cfg = _agent_config(agent_type)
            previous[agent_type] = getattr(cfg, "API_USER", None)
            cfg.API_USER = api_user
        except Exception:
            pass
    log.info("Applied ThermoML Argo API user: %s", api_user)
    return previous


def _restore_api_user_override(previous: dict) -> None:
    if not previous:
        return
    env_user = previous.get("env")
    if env_user is None:
        os.environ.pop("ARGO_API_USER", None)
    else:
        os.environ["ARGO_API_USER"] = env_user
    for agent_type in ("query", "analysis", "main"):
        if agent_type not in previous:
            continue
        try:
            cfg = _agent_config(agent_type)
            cfg.API_USER = previous[agent_type]
        except Exception:
            pass


def _apply_model_override(agent_type: str, model: str | None) -> dict:
    """Apply a per-run model override and return fields to restore later."""
    model = normalize_model_choice(model)
    if not model:
        return {}
    cfg = _agent_config(agent_type)
    fields = tuple(
        field for field in _MODEL_FIELDS_BY_AGENT.get(agent_type, ())
        if hasattr(cfg, field)
    )
    previous = {field: getattr(cfg, field) for field in fields}
    for field in fields:
        setattr(cfg, field, model)
    log.info("Applied %s model override %s to %s", agent_type, model, fields)
    return previous


def _restore_model_override(agent_type: str, previous: dict) -> None:
    """Restore model fields changed by ``_apply_model_override``."""
    if not previous:
        return
    cfg = _agent_config(agent_type)
    for field, value in previous.items():
        setattr(cfg, field, value)
    log.info("Restored %s model configuration", agent_type)


def _write_query_result_md(session_dir: Path, question: str, result) -> None:
    """Write a result.md for a browser-launched query run."""
    try:
        result_path = session_dir / "result.md"
        display_answer, structured_sections = prepare_result_answer(
            result_path, result.answer)
        lines = [
            f"# Query Run — {dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "",
            f"**Question:** {question}",
            "",
            f"**Time:** {result.elapsed_seconds:.1f}s | "
            f"**Iterations:** {result.iterations} | "
            f"**Tools:** {len(result.tool_history)}",
            "",
            "---",
            "",
            "## Answer",
            "",
            display_answer,
        ]
        lines += structured_sections
        write_result_bundle(result_path, "\n".join(lines), result.answer)
    except Exception as exc:
        log.debug("Failed to write query result.md: %s", exc)


def _extract_reasoning_snapshots(result, run_obj: AgentRun) -> None:
    """Pull reasoning snippets from tool_history into run_obj.reasoning_snapshots."""
    import re as _re
    history = getattr(result, "tool_history", None) or []
    for entry in history:
        raw = entry.get("reasoning", "") or ""
        if not raw.strip():
            continue
        # Strip XML-style utility markers: <tool_call>, </tool_call>, <reasoning>, <wait/>, etc.
        cleaned = _re.sub(r"</?[a-zA-Z_][a-zA-Z0-9_]*[^>]*/?>", "", raw)
        cleaned = cleaned.strip()
        if len(cleaned) > 30:
            # Truncate long reasoning to ~500 chars for display
            snippet = cleaned[:500] + ("…" if len(cleaned) > 500 else "")
            run_obj.reasoning_snapshots.append({
                "iteration": entry.get("iteration", "?"),
                "snippet": snippet,
            })


def _execute_run(run_obj: AgentRun):
    """Execute the agent run (called in background thread)."""
    global _active_run_id

    # Attach a log handler to capture all agent log output
    handler = _RunLogHandler(run_obj)
    root_logger = logging.getLogger()
    previous_root_level = root_logger.level
    if previous_root_level == logging.NOTSET or previous_root_level > logging.DEBUG:
        root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    model_restore: dict = {}
    api_user_restore: dict = {}

    try:
        run_obj.status = "running"
        run_obj.progress_log.append({
            "time": time.time(),
            "message": f"Starting {run_obj.agent_type} agent…",
        })
        api_user_restore = _apply_api_user_override(run_obj.api_user)
        run_obj.append_log(f"[browser] Argo API user set to: {run_obj.api_user}")

        if run_obj.agent_type == "analysis":
            from NIST_ThermoML_agents.NIST_ThermoML_analysis_agent.ThermoML_analysis_api import (
                ThermoML_analysis_run,
            )
            model_restore = _apply_model_override(
                "analysis", run_obj.model_override)
            _apply_time_overrides(run_obj.time_overrides)
            output_root = _apply_output_dir("analysis")
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", run_obj.question[:40]).strip("_").lower()
            live_session = output_root / f"run_{ts}_{slug or 'analysis'}"
            ensure_directory(live_session, exist_ok=False)
            run_obj.session_dir = str(live_session)
            result = ThermoML_analysis_run(
                run_obj.question, run_verdict=run_obj.run_verdict,
                session_dir=live_session,
            )
            run_obj.result = result
            run_obj.session_dir = result.session_dir

        elif run_obj.agent_type == "query":
            from NIST_ThermoML_agents.NIST_ThermoML_query_agent.ThermoML_query_api import (
                ThermoML_query_run,
            )
            model_restore = _apply_model_override("query", run_obj.model_override)
            _apply_time_overrides(run_obj.time_overrides)
            # Create a dedicated output directory (mirrors run_tests.py layout)
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = re.sub(r"[^a-zA-Z0-9]+", "_",
                          run_obj.question[:40]).strip("_").lower()
            q_session = _new_run_dir("query") / f"run_{ts}_{slug}"
            ensure_directory(q_session)
            run_obj.session_dir = str(q_session)
            mem_path = q_session / "working_memory.md"
            result = ThermoML_query_run(
                run_obj.question, memory_path=mem_path, session_dir=q_session,
            )
            run_obj.result = result
            # Write result.md (query engine doesn't do this automatically)
            _write_query_result_md(q_session, run_obj.question, result)

        elif run_obj.agent_type == "main":
            from NIST_ThermoML_agents._NIST_ThermoML_main_agent.ThermoML_main_api import (
                ThermoML_main_run,
            )
            model_restore = _apply_model_override("main", run_obj.model_override)
            _apply_time_overrides(run_obj.time_overrides)
            output_root = _apply_output_dir("main")
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", run_obj.question[:40]).strip("_").lower()
            live_session = output_root / f"run_{ts}_{slug or 'main'}"
            ensure_directory(live_session, exist_ok=False)
            run_obj.session_dir = str(live_session)
            result = ThermoML_main_run(
                run_obj.question, run_verdict=run_obj.run_verdict,
                session_dir=live_session,
            )
            run_obj.result = result
            run_obj.session_dir = getattr(result, "session_dir", None)

        else:
            run_obj.status = "error"
            run_obj.error = f"Unknown agent type: {run_obj.agent_type}"
            return

        # Extract reasoning snapshots from tool_history
        _extract_reasoning_snapshots(result, run_obj)

        run_obj.status = "completed"
        run_obj.progress_log.append({
            "time": time.time(),
            "message": (f"Completed — {result.iterations} iterations, "
                        f"{result.elapsed_seconds:.0f}s"),
        })

    except Exception as exc:
        import traceback
        run_obj.status = "error"
        run_obj.error = str(exc)
        run_obj.progress_log.append({
            "time": time.time(),
            "message": f"Error: {exc}",
        })
        traceback.print_exc()

    finally:
        _restore_model_override(run_obj.agent_type, model_restore)
        _restore_api_user_override(api_user_restore)
        root_logger.removeHandler(handler)
        if root_logger.level == logging.DEBUG:
            root_logger.setLevel(previous_root_level)
        run_obj.end_time = time.time()
        with _lock:
            if _active_run_id == run_obj.run_id:
                _active_run_id = None


def get_run(run_id: str) -> Optional[AgentRun]:
    with _lock:
        return _runs.get(run_id)


def list_runs() -> list[AgentRun]:
    with _lock:
        return list(_runs.values())


def is_busy() -> bool:
    """True if an agent run is currently in progress."""
    with _lock:
        if _active_run_id and _active_run_id in _runs:
            return _runs[_active_run_id].status in ("starting", "running")
    return False


# ---------------------------------------------------------------------------
# Past-run discovery — load runs from disk
# ---------------------------------------------------------------------------

AGENT_TYPES = ("main", "analysis", "query")
AGENT_OUTPUT_DIRS: dict[str, Path] = {
    kind: _REPO_ROOT / "_output" / kind.title() for kind in AGENT_TYPES
}
BENCHMARK_OUTPUT_DIRS: dict[str, Path] = {
    kind: _REPO_ROOT / "_benchmark" / kind.title() for kind in AGENT_TYPES
}


def _new_run_dir(agent_type: str) -> Path:
    """Shared persistent output directory for a new freeform run."""
    directory = AGENT_OUTPUT_DIRS[agent_type]
    directory.mkdir(parents=True, exist_ok=True)
    return directory



def _apply_output_dir(agent_type: str) -> Path:
    """Point the engine config's ``OUTPUT_DIR`` at the shared target.

    Analysis/Main build their own session directory from the engine config's
    ``OUTPUT_DIR``; override it before the run.
    """
    target = _new_run_dir(agent_type)
    try:
        from NIST_ThermoML_agents.general_db_query_engine.general_argo_engine_helpers import (
            get_config,
        )
        cfg = get_config()
        if hasattr(cfg, "OUTPUT_DIR"):
            cfg.OUTPUT_DIR = target
    except Exception:
        pass
    return target




_PROMPT_RUN_NAME = re.compile(
    r"(?P<question_id>Q\d+(?:\.\d+)*)_run_\d{8}_\d{6}(?:_\d{6})?"
)


def _question_id_from_run_name(name: str) -> str | None:
    """Read the current prompt ID carried by a renamed benchmark example."""
    match = _PROMPT_RUN_NAME.fullmatch(name)
    return match.group("question_id") if match else None


def _is_session_directory_name(name: str) -> bool:
    return name.startswith("run_") or _question_id_from_run_name(name) is not None


def _stored_run_directories(directory) -> list:
    """List ordinary run folders and read-only ZIPs without extraction."""
    entries = list(directory.iterdir())
    directories = [entry for entry in entries
                   if entry.is_dir() and not entry.is_symlink()]
    folder_names = {entry.name for entry in directories}
    archives = []
    if not isinstance(directory, ZipPath):
        for entry in entries:
            if (entry.is_file() and entry.suffix.lower() == ".zip"
                    and entry.stem not in folder_names and not entry.is_symlink()):
                try:
                    archive = ZipPath(entry)
                    if archive.is_dir():
                        archives.append(archive)
                except (OSError, ValueError, BadZipFile) as exc:
                    log.warning("Could not read output archive %s: %s", entry, exc)
    # A continued ordinary session takes precedence over its archived copy.
    return directories + archives


def _run_directory_child(root, name):
    """Resolve an actual directory or a collection's ZIP with the same stem."""
    try:
        child = root / name
        if child.is_dir():
            return child
        if not isinstance(root, ZipPath):
            archive_file = root / (name if name.lower().endswith(".zip") else name + ".zip")
            if archive_file.is_file() and not archive_file.is_symlink():
                archive = ZipPath(archive_file)
                if archive.is_dir():
                    return archive
    except (OSError, ValueError, BadZipFile):
        return None
    return None


def _is_benchmark_collection(path: Path) -> bool:
    """True for a folder that holds benchmark answers itself: per-prompt
    ``Q*`` folders, flat ``Q*_result.md`` files or ``TEST_TRACE*.json``
    manifests, whatever the folder is called."""
    try:
        for child in path.iterdir():
            name = child.name
            if child.is_dir():
                if re.match(r"Q\d", name) and not _is_session_directory_name(name):
                    return True
            elif ((name.startswith("TEST_TRACE") and name.endswith(".json"))
                  or re.fullmatch(r"Q[^_]+_result\.md", name)):
                return True
    except OSError:
        pass
    return False


def _benchmark_answer_parents(folder: Path, depth: int = 0) -> list[Path]:
    """Folders at any depth below a benchmark root that directly hold run
    sessions or answer collections, shallowest first. Sessions, ``Q*``
    answers, any folder holding result.md (an answer: its workers are not
    answers) and ``_``/``.`` folders are never descended into."""
    try:
        children = sorted(
            child for child in _stored_run_directories(folder)
            if child.is_dir() and not child.is_symlink()
            and not child.name.startswith(("_", ".")))
    except OSError:
        return []
    holds, nested = False, []
    for child in children:
        if _is_session_directory_name(child.name):
            holds = True
        elif re.match(r"Q\d", child.name) or (child / "result.md").is_file():
            continue
        else:
            if _is_benchmark_collection(child):
                holds = True
            if depth < 8:
                nested.extend(_benchmark_answer_parents(child, depth + 1))
    return ([folder] if holds and depth else []) + nested


# One page load or answer click asks for the roots several times; walking a
# network share each time is slow, so a walk is reused for a few seconds.
_ANSWER_ROOTS_CACHE: dict = {}
_ANSWER_ROOTS_TTL = 10.0


def _benchmark_answer_roots(folder: Path) -> list[Path]:
    """``_benchmark_answer_parents`` of one benchmark root, reused for
    ``_ANSWER_ROOTS_TTL`` seconds."""
    key = path_identity(folder)
    now = time.monotonic()
    hit = _ANSWER_ROOTS_CACHE.get(key)
    if hit and now - hit[0] < _ANSWER_ROOTS_TTL:
        return list(hit[1])
    found = _benchmark_answer_parents(folder)
    _ANSWER_ROOTS_CACHE[key] = (now, found)
    return list(found)


def _scan_roots(agent_type: str, scope: str = "all") -> list[dict]:
    """Existing directories to scan for past runs, most-relevant first.

    ``scope='user'`` is the compatibility name for shared freeform runs.
    ``scope='benchmark'`` contains read-only benchmark records.
    ``all`` merges both scopes.
    """
    if scope not in {"all", "user", "benchmark"}:
        raise ValueError(f"Unknown run scope: {scope}")
    roots: list[dict] = []
    seen: set[str] = set()

    def _add(path, read_only, temp, bucket="", source_scope="user"):
        if not path:
            return
        try:
            rp = path_identity(path)
        except Exception:
            return
        if rp in seen or not as_path(path).is_dir():
            return
        seen.add(rp)
        roots.append({"path": as_path(path), "read_only": read_only,
                      "temp": temp, "bucket": bucket,
                      "scope": source_scope})

    if agent_type not in AGENT_OUTPUT_DIRS:
        return roots
    if scope in {"all", "user"}:
        _add(AGENT_OUTPUT_DIRS[agent_type], False, False, "Shared", "user")
        # Existing migrated Main sessions were saved directly under _output.
        if agent_type == "main":
            _add(_REPO_ROOT / "_output", False, False, "Shared", "user")
    if scope in {"all", "benchmark"}:
        _add(BENCHMARK_OUTPUT_DIRS[agent_type], True, False, "Benchmark", "benchmark")
    # Walk nested campaigns and multipart ZIPs as read-only directories.
    # Stop at answer/session folders so workers are never indexed as answers.
    for src in list(roots):
        if src["scope"] == "benchmark":
            for sub in _benchmark_answer_roots(src["path"]):
                _add(sub, True, False,
                     f"{src['bucket']}/{sub.relative_to(src['path']).as_posix()}",
                     "benchmark")
            continue
        # Freeform archives keep the existing shared output behavior.
        try:
            subs = _stored_run_directories(src["path"])
        except OSError:
            continue
        for sub in subs:
            if not isinstance(sub, ZipPath) or _is_session_directory_name(sub.name):
                continue
            try:
                has_runs = any(c.is_dir() and _is_session_directory_name(c.name)
                               for c in sub.iterdir())
            except OSError:
                continue
            if has_runs:
                _add(sub, True, False, f"{src['bucket']}/{sub.name}", "user")
    return roots


def resolve_run_dir(agent_type: str, dir_name: str,
                    scope: str = "all") -> Optional[Path]:
    """Find an existing run directory named *dir_name* across all scan roots."""
    safe = os.path.basename(dir_name)
    if not safe or safe != dir_name or safe in {".", ".."}:
        return None
    for src in _scan_roots(agent_type, scope=scope):
        cand = _run_directory_child(src["path"], safe)
        if cand is not None and is_within(cand, src["path"]):
            return cand
    return None


def _question_from_result_md(result_md: Path) -> Optional[str]:
    try:
        for line in result_md.read_text(encoding="utf-8",
                                         errors="replace").splitlines():
            if line.startswith("**Question:**") or line.startswith("**Prompt:**"):
                return line.split(":**", 1)[1].strip()
    except OSError as exc:
        log.debug("Failed to read question text from %s: %s", result_md, exc)
    return None


def _extract_question(run_dir: Path) -> str:
    """Try to extract the question text from result.md in a run directory."""
    # Root result.md first; the first Q*.* sub-run is only consulted when the
    # root yields nothing.  Match the name before touching the file system:
    # a session directory holds hundreds of artefacts and one stat per entry
    # was ~70 % of the benchmark registry build time.
    root_md = run_dir / "result.md"
    if root_md.is_file():
        question = _question_from_result_md(root_md)
        if question is not None:
            return question
    try:
        if isinstance(run_dir, ZipPath):
            sub_runs = sorted(entry.name for entry in run_dir.iterdir()
                              if re.match(r"Q\d", entry.name) and entry.is_dir())
        else:
            with os.scandir(run_dir) as entries:
                sub_runs = sorted(
                    entry.name for entry in entries
                    if re.match(r"Q\d", entry.name) and entry.is_dir()
                )
    except OSError:
        sub_runs = []
    if sub_runs:
        sub_md = run_dir / sub_runs[0] / "result.md"
        if sub_md.is_file():
            question = _question_from_result_md(sub_md)
            if question is not None:
                return question
    return run_dir.name


_BENCHMARK_PROMPT_PATHS = {
    "main": (
        "_NIST_ThermoML_main_agent",
        "_DEBUG_script",
        "test_prompts.md",
    ),
    "analysis": (
        "NIST_ThermoML_analysis_agent",
        "_DEBUG_script",
        "test_prompts.md",
    ),
    "query": (
        "NIST_ThermoML_query_agent",
        "_DEBUG_script",
        "test_prompts.md",
    ),
}
_BENCHMARK_PROMPT_ROW_RE = re.compile(
    r"^\|\s*(?P<id>\d+(?:\.\d+)+)\s*\|\s*(?P<prompt>.+?)\s*\|\s*$"
)


def _benchmark_prompt_table(agent_type: str) -> dict[str, str]:
    """Return canonical ``Q<id> -> prompt`` entries for one agent."""
    parts = _BENCHMARK_PROMPT_PATHS.get(agent_type)
    if not parts:
        return {}
    path = Path(_WORKSPACE) / "NIST_ThermoML_agents"
    path = path.joinpath(*parts)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        log.warning("Could not read benchmark prompts %s: %s", path, exc)
        return {}

    prompts: dict[str, str] = {}
    for line in lines:
        match = _BENCHMARK_PROMPT_ROW_RE.match(line)
        if match:
            prompts[f"Q{match.group('id')}"] = match.group("prompt").strip()
    return prompts


def _prompt_match_key(prompt: str) -> str:
    """Normalize prompt typography and spacing for archive-to-ID matching."""
    return re.sub(r"[^a-z0-9]+", "", (prompt or "").casefold())


def _benchmark_time_label(name: str) -> str:
    """Format the first timestamp embedded in a run/trace name for the UI."""
    match = re.search(r"(\d{8}_\d{6})", name)
    if not match:
        return name
    try:
        return dt.datetime.strptime(
            match.group(1), "%Y%m%d_%H%M%S"
        ).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return name


def _benchmark_result_metadata(run_dir: Path) -> tuple[Optional[float], Optional[int]]:
    """Read elapsed time and iteration count from a standalone result.md."""
    path = run_dir / "result.md"
    if not path.is_file():
        return None, None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:5000]
    except OSError:
        return None, None
    elapsed_match = re.search(r"\*\*Time:\*\*\s*([\d.]+)s", text)
    iteration_match = re.search(r"\*\*Iterations:\*\*\s*(\d+)", text)
    elapsed = float(elapsed_match.group(1)) if elapsed_match else None
    iterations = int(iteration_match.group(1)) if iteration_match else None
    return elapsed, iterations


def list_past_runs(limit: int = 20) -> list[dict]:
    """List recent session directories on disk for the history tab."""
    return list_agent_runs("analysis", limit=limit)


def _question_sort_key(question_id: str) -> tuple:
    """Sort Q1.2 before Q1.10 while tolerating non-numeric IDs."""
    parts = re.split(r"(\d+)", question_id)
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in parts)


def _benchmark_models(run_dir: Path) -> list[str]:
    """Extract the actual model names recorded for one prompt session."""
    stats_path = run_dir / "reference_stats.md"
    if not stats_path.is_file():
        return []
    try:
        lines = stats_path.read_text(
            encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    models: list[str] = []
    model_column: Optional[int] = None
    for line in lines:
        if not line.lstrip().startswith("|"):
            model_column = None
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if "Model(s)" in cells:
            model_column = cells.index("Model(s)")
            continue
        if "Model" in cells:
            model_column = cells.index("Model")
            continue
        if model_column is None or model_column >= len(cells):
            continue
        value = cells[model_column].strip()
        if not value or re.fullmatch(r":?-+:?", value):
            continue
        for model in re.split(r"\s*(?:,|<br\s*/?>)\s*", value,
                              flags=re.IGNORECASE):
            if model and model not in models:
                models.append(model)
    return models


def _benchmark_mode(collection: Path) -> str:
    """Identify the execution mode from its actual collection directory."""
    # Campaigns may contain an older test_run_<timestamp> collection.
    # Its enclosing mode directory remains the execution's mode.
    for parent in (collection, *collection.parents):
        name = parent.name.lower()
        for prefix, mode in (
            ("test_run_no_ledger_", "no_ledger"),
            ("test_run_bare_model_", "bare_model"),
            ("test_run_flat_tools_", "flat_tools"),
            ("test_run_only_sql_no_prefix_", "sql_no_prefix"),
            ("test_run_only_sql_rawdb_", "sql_raw"),
            ("test_run_only_sql_", "sql_parsed"),
        ):
            if name == prefix.rstrip("_") or name.startswith(prefix):
                return mode
    return "full_framework"


def _benchmark_mode_columns(modes: set[str]) -> list[dict]:
    """Keep a stable column order using only this agent's saved modes."""
    return [
        {"key": key, "label": label}
        for key, label in (
            ("full_framework", "Full framework"),
            ("no_ledger", "No ledger"),
            ("bare_model", "Bare model"),
            ("flat_tools", "Flat tools"),
            ("sql_parsed", "SQL (parsed cards)"),
            ("sql_no_prefix", "SQL (no ID prefixes)"),
            ("sql_raw", "SQL (raw ThermoML)"),
        )
        if key in modes
    ]


def list_benchmark_questions(agent_type: str) -> tuple[list[dict], dict]:
    """Register benchmark prompts and group their available executions.

    Prompt Markdown is the canonical question registry.  Every TEST_TRACE
    manifest in an answer collection contributes executions; standalone
    ``Q<id>_run_<timestamp>`` archives use the ID in their name; legacy
    ``run_*`` archives are matched back to the registry by prompt text.  Query stores answers in ``Q*`` directories, while older
    Analysis collections use flat ``Q*_result.md`` files.

    Session folders on disk are the authority for what counts as an
    execution.  A collection's flat ``Q*_result.md`` is rewritten by every
    rerun of that question, so it can stand in only for the newest trace
    record of the question in that collection; older records whose session
    folder is gone have no artifact of their own and are not listed.
    """
    prompt_table = _benchmark_prompt_table(agent_type)
    grouped: dict[str, dict] = {
        question_id: {
            "question_id": question_id,
            "prompt_text": prompt,
            "executions": [],
        }
        for question_id, prompt in prompt_table.items()
    }
    prompt_ids_by_text = {
        _prompt_match_key(prompt): question_id
        for question_id, prompt in prompt_table.items()
    }
    seen_collections: set[str] = set()
    seen_runs: set[str] = set()
    seen_executions: set[tuple[str, str, str]] = set()
    traced_questions: set[tuple[str, str]] = set()  # (collection, question)
    collections: list[Path] = []
    traced_collections: set[str] = set()
    available_modes = {"full_framework"}
    batch_count = 0
    execution_count = 0
    unmatched_runs: list[str] = []

    for src in _scan_roots(agent_type, scope="benchmark"):
        try:
            batches = sorted(
                (path for path in _stored_run_directories(src["path"])
                 if not _is_session_directory_name(path.name)
                 and _is_benchmark_collection(path)),
                key=lambda path: path.name,
                reverse=True,
            )
        except OSError:
            continue

        for batch_dir in batches:
            # Higher-priority roots shadow an identically named collection.
            if batch_dir.name in seen_collections:
                continue
            seen_collections.add(batch_dir.name)
            collections.append(batch_dir)
            mode = _benchmark_mode(batch_dir)
            available_modes.add(mode)
            traces = sorted(batch_dir.glob("TEST_TRACE*.json"), reverse=True)
            for trace_path in traces:
                try:
                    records = json.loads(trace_path.read_text(
                        encoding="utf-8", errors="replace"))
                except (OSError, json.JSONDecodeError) as exc:
                    log.warning("Could not import benchmark trace %s: %s",
                                trace_path, exc)
                    continue
                if not isinstance(records, list):
                    continue

                imported_from_trace = False
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    question_id = _trace_question_id(record, prompt_table)
                    prompt = str(
                        record.get("prompt_text") or record.get("prompt") or ""
                    ).strip()
                    if not question_id:
                        continue
                    # Traces are visited newest first, so the first record
                    # seen for a question owns the collection's flat files.
                    collection_key = (batch_dir.name, question_id)
                    newest_in_collection = collection_key not in traced_questions
                    traced_questions.add(collection_key)

                    # Prefer the complete Analysis/Main session named in the
                    # trace. Query traces have no session_dir and use Q folders.
                    session_leaf = _trace_session_leaf(record)
                    session_dir = _trace_session_dir(record, trace_path, agent_type)
                    query_dir = batch_dir / question_id
                    flat_result = batch_dir / f"{question_id}_result.md"

                    if session_dir is not None and session_dir.is_dir():
                        detail_dir = session_dir
                        dir_name = session_dir.name
                        sub_run = None
                        has_result = (session_dir / "result.md").is_file()
                        display_stamp = _benchmark_time_label(session_dir.name)
                    elif query_dir.is_dir():
                        detail_dir = query_dir
                        dir_name = batch_dir.name
                        sub_run = question_id
                        has_result = (query_dir / "result.md").is_file()
                        display_stamp = _benchmark_time_label(trace_path.stem)
                    elif not newest_in_collection:
                        log.info("benchmark trace %s/%s %s: session folder %s "
                                 "is gone and the flat result was rewritten "
                                 "by a newer rerun; not listed",
                                 batch_dir.name, trace_path.name, question_id,
                                 session_leaf or "(unrecorded)")
                        continue
                    else:
                        # Old Analysis layout: Q1.1_result.md at batch root.
                        detail_dir = batch_dir
                        dir_name = batch_dir.name
                        sub_run = question_id
                        has_result = flat_result.is_file()
                        display_stamp = _benchmark_time_label(trace_path.stem)

                    execution_key = (question_id, dir_name, sub_run or "")
                    if execution_key in seen_executions:
                        continue
                    seen_executions.add(execution_key)

                    # Asset tabs resolve their files lazily when the user
                    # opens an execution; recursively walking every archived
                    # session here makes the question index needlessly slow.
                    n_images = n_csv = 0
                    models = _benchmark_models(detail_dir)
                    error = str(record.get("error") or "").strip()
                    timed_out = bool(record.get("timed_out"))
                    if error:
                        status = "error"
                    elif timed_out:
                        status = "timed_out"
                    elif has_result or record.get("answer"):
                        status = "completed"
                    else:
                        status = "incomplete"

                    row = grouped.setdefault(question_id, {
                        "question_id": question_id,
                        "prompt_text": prompt,
                        "executions": [],
                    })
                    if prompt and not row["prompt_text"]:
                        row["prompt_text"] = prompt
                    row["executions"].append({
                        "mode": mode,
                        "dir_name": dir_name,
                        "sub_run": sub_run,
                        "batch_name": trace_path.stem,
                        "batch_label": display_stamp,
                        "status": status,
                        "has_result": has_result,
                        "elapsed_s": record.get("elapsed_s"),
                        "iterations": record.get("iterations"),
                        "n_images": n_images,
                        "n_csv": n_csv,
                        "models": models,
                        "model_label": ", ".join(models) if models else "unknown",
                    })
                    execution_count += 1
                    imported_from_trace = True

                if imported_from_trace:
                    batch_count += 1
                    traced_collections.add(batch_dir.name)

    # Main has no TEST_TRACE collection, and imported Analysis archives may
    # also contain standalone reruns. Match those run directories to the
    # canonical registry by their explicit name prefix or recorded question text.
    for src in _scan_roots(agent_type, scope="benchmark"):
        try:
            run_dirs = sorted(
                (path for path in src["path"].iterdir()
                 if path.is_dir() and _is_session_directory_name(path.name)),
                key=lambda path: path.name,
                reverse=True,
            )
        except OSError:
            continue
        for run_dir in run_dirs:
            if run_dir.name in seen_runs:
                continue
            seen_runs.add(run_dir.name)
            question = _extract_question(run_dir)
            prefixed_id = _question_id_from_run_name(run_dir.name)
            question_id = (
                prefixed_id if prefixed_id in prompt_table else None
            ) if prefixed_id else prompt_ids_by_text.get(_prompt_match_key(question))
            if not question_id:
                # Not a defect of the run: ad-hoc or renamed-prompt sessions
                # live here too.  Count them so a registry/prompt drift that
                # silently empties a question is visible in the header.
                unmatched_runs.append(run_dir.name)
                log.info("benchmark run %s not matched to the %s prompt "
                         "registry (question: %.80s)", run_dir.name,
                         agent_type, question)
                continue
            execution_key = (question_id, run_dir.name, "")
            if execution_key in seen_executions:
                continue
            seen_executions.add(execution_key)

            has_result = (run_dir / "result.md").is_file()
            elapsed_s, iterations = _benchmark_result_metadata(run_dir)
            n_images = n_csv = 0
            models = _benchmark_models(run_dir)
            mode = _benchmark_mode(run_dir.parent)
            available_modes.add(mode)
            grouped[question_id]["executions"].append({
                "mode": mode,
                "dir_name": run_dir.name,
                "sub_run": None,
                "batch_name": run_dir.name,
                "batch_label": _benchmark_time_label(run_dir.name),
                "status": "completed" if has_result else "incomplete",
                "has_result": has_result,
                "elapsed_s": elapsed_s,
                "iterations": iterations,
                "n_images": n_images,
                "n_csv": n_csv,
                "models": models,
                "model_label": ", ".join(models) if models else "unknown",
            })
            execution_count += 1

    # Legacy collections without usable trace records still expose their
    # answer once per registered question. They remain an index source, not a
    # campaign-run card in the browser sidebar.
    for batch_dir in collections:
        imported_from_collection = False
        candidates: dict[str, tuple[Path, Optional[str]]] = {}
        try:
            for path in batch_dir.iterdir():
                if path.is_dir() and re.fullmatch(r"Q\d+(?:\.\d+)+", path.name):
                    if (path / "result.md").is_file():
                        candidates[path.name] = (path, path.name)
                elif path.is_file():
                    match = re.fullmatch(r"(Q\d+(?:\.\d+)+)_result\.md", path.name)
                    if match:
                        candidates[match.group(1)] = (batch_dir, match.group(1))
        except OSError:
            continue

        for question_id, (detail_dir, sub_run) in candidates.items():
            row = grouped.setdefault(question_id, {
                "question_id": question_id,
                "prompt_text": "",
                "executions": [],
            })
            if (batch_dir.name, question_id) in traced_questions:
                continue
            execution_key = (question_id, batch_dir.name, sub_run or "")
            if execution_key in seen_executions:
                continue
            seen_executions.add(execution_key)
            n_images = n_csv = 0
            models = _benchmark_models(detail_dir)
            row["executions"].append({
                "mode": _benchmark_mode(batch_dir),
                "dir_name": batch_dir.name,
                "sub_run": sub_run,
                "batch_name": batch_dir.name,
                "batch_label": _benchmark_time_label(batch_dir.name),
                "status": "completed",
                "has_result": True,
                "elapsed_s": None,
                "iterations": None,
                "n_images": n_images,
                "n_csv": n_csv,
                "models": models,
                "model_label": ", ".join(models) if models else "unknown",
            })
            execution_count += 1
            imported_from_collection = True
        if imported_from_collection and batch_dir.name not in traced_collections:
            batch_count += 1

    rows = sorted(grouped.values(), key=lambda row: _question_sort_key(
        row["question_id"]))
    mode_columns = _benchmark_mode_columns(available_modes)
    for row in rows:
        row["executions"].sort(
            key=lambda execution: execution["batch_name"], reverse=True)
        row["batch_count"] = len(row["executions"])
        row["executions_by_mode"] = {
            mode["key"]: [execution for execution in row["executions"]
                          if execution["mode"] == mode["key"]]
            for mode in mode_columns
        }
    return rows, {
        "modes": mode_columns,
        "questions": len(rows),
        "executions": execution_count,
        "batches": batch_count,
        "collections": len(seen_collections),
        "unmatched_runs": len(unmatched_runs),
    }


def _trace_question_id(record: dict, prompt_table: dict | None = None) -> str:
    prefixed_id = _question_id_from_run_name(_trace_session_leaf(record))
    if prefixed_id and prompt_table is not None and prefixed_id in prompt_table:
        return prefixed_id
    raw_id = str(record.get("prompt_id") or record.get("id") or "").strip()
    if not raw_id:
        return ""
    return raw_id if raw_id.upper().startswith("Q") else f"Q{raw_id}"


def _trace_session_leaf(record: dict) -> str:
    """Return the final component of a trace session path on any OS."""
    value = str(record.get("session_dir") or "").rstrip("/\\")
    return re.split(r"[/\\]", value)[-1] if value else ""


def _trace_session_dir(
    record: dict,
    trace_path: Path,
    agent_type: str | None = None,
) -> Optional[Path]:
    """Resolve a trace session directory under current Benchmark roots.

    Historical TEST_TRACE files store absolute Windows paths in
    ``session_dir``.  Treat those paths as metadata only: the portable part is
    the final session directory name, which should be resolved under the
    ``_benchmark/<Agent>/`` tree.
    """
    leaf = _trace_session_leaf(record)
    if not leaf:
        return None
    roots = [trace_path.parent.parent]
    if agent_type:
        roots.extend(src["path"] for src in _scan_roots(
            agent_type, scope="benchmark"))
    seen: set[str] = set()
    for root in roots:
        key = path_identity(root)
        if key in seen:
            continue
        seen.add(key)
        candidate = root / leaf
        if candidate.is_dir():
            return candidate
    return None


def _find_benchmark_trace_record(agent_type: str, dir_name: str,
                                 sub_run: Optional[str]) -> tuple[dict, Path] | None:
    """Find the TEST_TRACE record backing one browser benchmark result."""
    safe_dir = os.path.basename(dir_name)
    safe_sub = os.path.basename(sub_run or "")
    prompt_table = _benchmark_prompt_table(agent_type)
    for src in _scan_roots(agent_type, scope="benchmark"):
        try:
            if safe_sub:
                candidate = _run_directory_child(src["path"], safe_dir)
                candidates = [candidate] if candidate is not None else []
            else:
                candidates = sorted(
                    (path for path in _stored_run_directories(src["path"])
                     if not _is_session_directory_name(path.name)
                     and _is_benchmark_collection(path)),
                    key=lambda path: path.name,
                    reverse=True,
                )
        except OSError:
            continue

        for batch_dir in candidates:
            if not batch_dir.is_dir():
                continue
            traces = sorted(batch_dir.glob("TEST_TRACE*.json"), reverse=True)
            if not traces:
                continue
            for trace_path in traces:
                try:
                    records = json.loads(trace_path.read_text(
                        encoding="utf-8", errors="replace"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(records, list):
                    continue
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    if safe_sub:
                        if safe_sub and _trace_question_id(record, prompt_table) == safe_sub:
                            return record, trace_path
                    elif _trace_session_leaf(record) == safe_dir:
                        return record, trace_path
    return None


def resolve_run_artifact_dir(agent_type: str, dir_name: str,
                             sub_run: Optional[str] = None,
                             scope: str = "all") -> Optional[Path]:
    """Resolve the directory that holds plots/data/stats for a run detail."""
    base = resolve_run_dir(agent_type, dir_name, scope=scope)
    if base is None:
        return None

    safe_sub = os.path.basename(sub_run or "")
    if not safe_sub:
        return base

    nested = base / safe_sub
    if nested.is_dir():
        return nested

    # Older benchmark pages address Analysis/Main examples as
    # test_run_*/Q*.md flat files. The trace points to the real sibling run_*
    # directory that holds generated plots, CSVs, and reference_stats.md.
    if scope == "benchmark" and (base / f"{safe_sub}_result.md").is_file():
        trace_match = _find_benchmark_trace_record(agent_type, dir_name, safe_sub)
        if trace_match:
            record, trace_path = trace_match
            session_dir = _trace_session_dir(record, trace_path, agent_type)
            if session_dir is not None:
                return session_dir
        return base

    return None


def delete_user_run(agent_type: str, dir_name: str) -> bool:
    """Delete one shared freeform run; benchmark roots are excluded."""
    if agent_type not in AGENT_OUTPUT_DIRS:
        return False
    safe = os.path.basename(dir_name)
    if not safe or safe != dir_name:
        return False
    for src in _scan_roots(agent_type, scope="user"):
        if src.get("read_only"):
            continue
        root = src["path"]
        candidate = root / safe
        if not candidate.is_dir():
            continue
        try:
            real_root = os.path.realpath(str(root))
            real_candidate = os.path.realpath(str(candidate))
            if os.path.commonpath((real_root, real_candidate)) != real_root:
                return False
            shutil.rmtree(candidate)
            return not candidate.exists()
        except (OSError, ValueError):
            return False
    return False


def _markdown_fence(value: str) -> str:
    """Choose a code fence longer than any backtick run in *value*."""
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", value)),
                  default=0)
    return "`" * max(3, longest + 1)


def _render_full_tool_history(agent_type: str, record: dict,
                              trace_path: Path) -> str:
    """Render every stored tool result without the compact-history preview."""
    question_id = _trace_question_id(record, _benchmark_prompt_table(agent_type))
    prompt = str(record.get("prompt_text") or record.get("prompt") or "").strip()
    history = record.get("tool_history") or []
    lines = [
        f"# Full Tool History — {agent_type.title()} Agent",
        "",
        f"**Question:** {question_id}",
        "",
        f"**Prompt:** {prompt}",
        "",
        f"**Trace:** `{trace_path.name}`",
        "",
        f"**Tool calls:** {len(history)}",
        "",
        "---",
        "",
    ]
    if not history:
        lines.extend(["*No tool calls were recorded.*", ""])
        return "\n".join(lines)

    for index, entry in enumerate(history, start=1):
        if not isinstance(entry, dict):
            continue
        iteration = entry.get("iteration", index)
        tool_name = str(entry.get("tool") or "unknown")
        result_full = str(entry.get("result_full") or "")
        result_chars = entry.get("result_chars", len(result_full))
        elapsed = entry.get("elapsed_s")
        lines.extend([
            f"## Step {iteration}: `{tool_name}`",
            "",
            f"**Full result size:** {result_chars:,} characters"
            + (f" · **Elapsed:** {elapsed}s" if elapsed is not None else ""),
            "",
        ])

        arguments = entry.get("arguments")
        if arguments:
            args_text = json.dumps(arguments, indent=2, default=str,
                                   ensure_ascii=False)
            args_fence = _markdown_fence(args_text)
            lines.extend([
                "<details><summary>Arguments</summary>",
                "",
                f"{args_fence}json",
                args_text,
                args_fence,
                "",
                "</details>",
                "",
            ])

        reasoning = str(entry.get("reasoning") or "").strip()
        if reasoning:
            reasoning_fence = _markdown_fence(reasoning)
            lines.extend([
                "<details><summary>Agent reasoning recorded for this call</summary>",
                "",
                reasoning_fence,
                reasoning,
                reasoning_fence,
                "",
                "</details>",
                "",
            ])

        result_fence = _markdown_fence(result_full)
        lines.extend([
            f"<details open><summary>Full tool result ({result_chars:,} characters)</summary>",
            "",
            result_fence,
            result_full or "(empty result)",
            result_fence,
            "",
            "</details>",
            "",
            "---",
            "",
        ])
    return "\n".join(lines)


def list_agent_runs(agent_type: str, limit: int = 30,
                    scope: str = "all", offset: int = 0) -> list[dict]:
    """List past runs for an agent type across all scan roots.

    ``scope`` selects shared freeform, benchmark, or merged roots. Within a scope,
    higher-priority roots shadow duplicate run names. ``offset`` and ``limit``
    paginate before per-run metadata files are opened.
    """
    pairs: list[tuple] = []          # (dir_path, source)
    seen_names: set[str] = set()
    for src in _scan_roots(agent_type, scope=scope):
        try:
            entries = _stored_run_directories(src["path"])
        except Exception:
            continue
        for d in entries:
            if not d.is_dir():
                continue
            if not (_is_session_directory_name(d.name) or d.name.startswith("test_run_")):
                continue
            # A benchmark test_run_* folder is an answer collection consumed
            # by list_benchmark_questions(), not an independently browsable
            # campaign execution.
            if (src.get("scope") == "benchmark"
                    and d.name.startswith("test_run_")):
                continue
            if d.name in seen_names:
                continue
            seen_names.add(d.name)
            pairs.append((d, src))
    pairs.sort(key=lambda t: t[0].stat().st_mtime, reverse=True)
    offset = max(0, int(offset or 0))
    pairs = pairs[offset:offset + limit]

    result = []
    for d, src in pairs:
        result_md = d / "result.md"
        has_result = result_md.exists()

        # For query agent: check for Q*.* sub-directories
        sub_runs = sorted(
            s.name for s in d.iterdir()
            if s.is_dir() and re.match(r"Q\d", s.name)
        )
        if sub_runs and not has_result:
            has_result = any((d / sub_run / "result.md").exists()
                             for sub_run in sub_runs)

        question = _extract_question(d)

        images, csv_files = discover_run_artifacts(d)

        result.append({
            "dir_name": d.name,
            "path": str(d),
            "question": question,
            "has_result": has_result,
            "n_images": len(images),
            "n_csv": len(csv_files),
            "sub_runs": sub_runs,
            "mtime": d.stat().st_mtime,
            "read_only": src["read_only"] or isinstance(d, ZipPath),
            "bucket": src.get("bucket", ""),
            "temp": src.get("temp", False),
            "scope": src.get("scope", scope),
        })
    return result


_ID_CATALOG_HEADING = "## ID Catalog Snapshot (structured ledger)"
_CORE_ID_UPDATES_HEADING = "## Core ID Updates (deterministic ledger)"
_CORE_BLOCKS_HEADING = "## Core Blocks Found (deterministic ledger)"
_LEDGER_HEADING = "## Data Inspections (deterministic evidence ledger)"

_LEDGER_META_KEYS = (
    "inspection_id", "lit_num_id", "doi", "block_number", "BLKsubsys_id",
    "table_mode", "columns", "rows_shown",
)


def _ledger_entry_table(item: dict) -> list[str]:
    """Render one data_inspections entry as a captioned markdown table.

    The caption puts the GLOBlit id immediately before the block id so the
    client-side linkifier resolves the block against the correct paper.
    """
    lit = str(item.get("lit_num_id") or "").strip()
    block = str(item.get("block_number") or "").strip()
    anchor = f"{lit}::{block}" if lit and block else (lit or block or "?")
    bits = [f"**{item.get('inspection_id', '?')} — {anchor}**"]
    if item.get("BLKsubsys_id"):
        bits.append(f"`{item['BLKsubsys_id']}`")
    if item.get("table_mode"):
        bits.append(f"mode `{item['table_mode']}`")
    if item.get("doi"):
        bits.append(f"DOI {item['doi']}")
    lines = [" · ".join(bits), ""]

    rows = item.get("rows_shown")
    rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
    columns = item.get("columns")
    if not (isinstance(columns, list) and columns):
        seen: dict = {}
        for row in rows:
            seen.update(dict.fromkeys(row))
        columns = list(seen)
    if rows and columns:
        def _cell(value) -> str:
            return str(value if value is not None else "").replace("|", "\\|")
        lines.append("| " + " | ".join(_cell(c) for c in columns) + " |")
        lines.append("|" + "---|" * len(columns))
        for row in rows:
            lines.append(
                "| " + " | ".join(_cell(row.get(c, "")) for c in columns) + " |")
    else:
        lines.append("*No verbatim rows recorded for this inspection.*")

    extras = {k: v for k, v in item.items() if k not in _LEDGER_META_KEYS}
    if extras:
        lines += [
            "",
            "<details><summary>Inspection metadata</summary>",
            "",
        ]
        lines += [
            f"- **{key.replace('_', ' ').title()}:** {_browser_cell(value)}"
            for key, value in extras.items()
        ]
        lines += ["", "</details>"]
    return lines


def _ledger_tables_md(ledger: list) -> str:
    """Render a data_inspections list as markdown tables (one per entry)."""
    parts: list[str] = []
    for item in ledger:
        if isinstance(item, dict):
            parts.extend(_ledger_entry_table(item))
            parts.append("")
    return "\n".join(parts).strip()


def _browser_cell(value) -> str:
    """Flatten structured values for Markdown without exposing raw JSON."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        text = "; ".join(
            f"{key}={_browser_cell(item)}" for key, item in value.items())
    elif isinstance(value, list):
        text = "; ".join(_browser_cell(item) for item in value)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _records_table_md(value, preferred: tuple[str, ...] = ()) -> str:
    """Render a structured list/dict as complete Markdown records."""
    records = value if isinstance(value, list) else [value]
    records = [record for record in records if isinstance(record, dict)]
    if not records:
        return "*No structured records were supplied.*"
    columns = [key for key in preferred if any(key in row for row in records)]
    columns += sorted(
        {key for row in records for key in row} - set(columns),
        key=str.casefold,
    )
    if not columns:
        return "*No structured records were supplied.*"
    lines = [
        "| " + " | ".join(key.replace("_", " ") for key in columns) + " |",
        "|" + "---|" * len(columns),
    ]
    lines += [
        "| " + " | ".join(_browser_cell(row.get(key)) for key in columns) + " |"
        for row in records
    ]
    return "\n".join(lines)


def _catalog_tables_md(value) -> str:
    return _records_table_md(
        value, ("type", "global_id", "registry_id", "name"))


def _core_id_updates_tables_md(value) -> str:
    return _records_table_md(
        value, ("action", "core_GLOB_id", "registry_id", "name"))


def _core_blocks_tables_md(value) -> str:
    return _records_table_md(
        value, ("lit_num_id", "lit_id", "doi", "block_number",
                "BLKsubsys_id", "block_type", "system_type",
                "n_datapoints", "comp_num_ids", "prop_num_ids",
                "description"))


def _replace_json_section(result_md: str, heading: str, renderer) -> str:
    idx = result_md.find(heading)
    if idx < 0:
        return result_md
    section_start = idx + len(heading)
    next_heading = result_md.find("\n## ", section_start)
    section_end = next_heading if next_heading >= 0 else len(result_md)
    section = result_md[section_start:section_end]
    fence = re.search(r"```json\s*\n(.*?)\n```", section, re.DOTALL)
    if not fence:
        return result_md
    try:
        value = json.loads(fence.group(1))
    except json.JSONDecodeError:
        return result_md
    rendered = renderer(value)
    if not rendered:
        return result_md
    new_section = section[:fence.start()] + rendered + section[fence.end():]
    return result_md[:section_start] + new_section + result_md[section_end:]


def render_ledger_tables(result_md: str) -> str:
    """Replace the ledger section's raw JSON fence with readable tables.

    ``result.md`` files carry the deterministic evidence ledger as one
    ```json fence under the Data Inspections heading; the browser shows the
    same rows as markdown tables so identifiers become clickable links.
    Files without the section (or with unparseable JSON) pass through.
    """
    rendered = result_md
    for heading, renderer in (
        (_ID_CATALOG_HEADING, _catalog_tables_md),
        (_CORE_ID_UPDATES_HEADING, _core_id_updates_tables_md),
        (_CORE_BLOCKS_HEADING, _core_blocks_tables_md),
        (_LEDGER_HEADING, _ledger_tables_md),
    ):
        rendered = _replace_json_section(rendered, heading, renderer)
    return rendered


def _render_postanswer_envelope(result_md: str) -> str:
    """Deserialize an embedded post-answer JSON envelope into readable markdown.

    Agent runs store their final return as a JSON contract (answer +
    core_claims + confidence + sources) under ``## Answer``. The browser
    shows the ANSWER text verbatim and formats the post-answer analysis
    branches below it. Any file that does not carry a parseable envelope is
    returned unchanged.
    """
    marker = "## Answer"
    idx = result_md.find(marker)
    if idx < 0:
        return result_md
    tail = result_md[idx + len(marker):]
    body = tail.lstrip()
    if not body:
        return result_md
    try:
        if body.startswith("```json") or body.startswith("```\n"):
            fence = re.match(
                r"```(?:json)?\s*\n(.*?)\n```", body,
                re.DOTALL | re.IGNORECASE,
            )
            if not fence:
                return result_md
            envelope = json.loads(fence.group(1))
            consumed = fence.end()
        else:
            # A legacy envelope must be the first non-whitespace content
            # below ## Answer. Later JSON examples/ledger rows belong to the
            # already-rendered Markdown and must never replace its prose.
            if not body.startswith("{"):
                return result_md
            envelope, consumed = json.JSONDecoder().raw_decode(body)
    except (json.JSONDecodeError, ValueError):
        return result_md
    if not isinstance(envelope, dict) or not isinstance(envelope.get("answer"), str):
        return result_md

    # Legacy artifacts embedded the machine envelope directly in result.md.
    # Reuse the live writer's renderer so every ordinary field is Markdown;
    # structured ledgers remain JSON until render_ledger_tables handles them.
    display_payload = dict(envelope)
    if _LEDGER_HEADING in result_md:
        display_payload.pop("data_inspections", None)
    parts = [
        result_md[:idx + len(marker)],
        "",
        render_answer_envelope_markdown(display_payload),
    ]
    parts += structured_envelope_section_lines(display_payload)
    remainder = body[consumed:].strip()
    if remainder:
        parts += ["", remainder]
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Workflow (compaction Sankey) artifacts — generated post-run by the
# compaction_funnels API into <session>/workflow/ after completed root runs
# ---------------------------------------------------------------------------

_WF_POOLED_RE = re.compile(r"^(chars|ids|blocks|pts)_pooled_sankey_(.+)$")
_WF_METRIC_ORDER = {"chars": 0, "ids": 1, "blocks": 2, "pts": 3}


def list_workflow_figures(artifact_base: Path) -> Optional[dict]:
    """Parse <session>/workflow/ into the browser's Sankey panel data.
    Only the POOLED funnels are displayed (one context / answer / tool
    rail per delegation layer); the detailed and agentic figures stay
    on disk for the TSV/audit tooling."""
    wf = artifact_base / "workflow"
    if not wf.is_dir():
        return None
    figures: list[dict] = []
    for png in sorted(wf.glob("*.png")):
        match = _WF_POOLED_RE.match(png.stem)
        if not match:
            continue
        html = png.with_suffix(".html")
        figures.append({
            "label": f"{match.group(1)} · pooled",
            "png": png.name,
            "html": html.name if html.exists() else None,
            "_ord": _WF_METRIC_ORDER.get(match.group(1), 9),
        })
    figures.sort(key=lambda f: f.pop("_ord"))
    audit = {"FAIL": 0, "WARN": 0, "INFO": 0}
    for report in sorted(wf.glob("*audit*report.tsv")):
        try:
            with report.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    severity = row.get("severity", "")
                    if severity in audit:
                        audit[severity] += 1
        except OSError:
            pass
    tsv_files = sorted(p.name for p in wf.glob("*.tsv"))
    ordered = ([{"name": "Pooled workflow funnels", "figures": figures}]
               if figures else [])
    if not ordered and not tsv_files:
        return None
    return {"groups": ordered, "audit": audit, "tsv_files": tsv_files}


def get_run_files(agent_type: str, dir_name: str,
                  sub_run: Optional[str] = None,
                  scope: str = "all",
                  include_full_tool_history: bool = True) -> dict:
    """Get detailed file listing for a specific run.

    Full benchmark tool payloads are included by default. Callers requesting
    only summaries may omit their potentially large contents.
    """
    if agent_type not in AGENT_OUTPUT_DIRS:
        return {"error": "Unknown agent type"}

    run_root = resolve_run_dir(agent_type, dir_name, scope=scope)
    if run_root is None:
        return {"error": "Run not found"}

    sub_runs = {
        s.name for s in run_root.iterdir()
        if s.is_dir() and re.match(r"Q\d", s.name)
    }
    # Older Analysis/Main benchmark runners wrote flat Q1.1_result.md files
    # instead of Q1.1/result.md directories.
    sub_runs.update(
        match.group(1)
        for path in run_root.glob("Q*_result.md")
        if (match := re.match(r"^(Q[^_]+)_result\.md$", path.name))
    )
    sub_runs = sorted(sub_runs, key=_question_sort_key)
    base = run_root
    artifact_base = base
    flat_prefix = ""

    # If sub_run is specified, descend into a Query directory or address the
    # flat Q-prefixed files produced by older Analysis/Main test runners.
    if sub_run:
        safe_sub = os.path.basename(sub_run)
        nested = base / safe_sub
        if nested.is_dir():
            base = nested
            artifact_base = nested
        elif (base / f"{safe_sub}_result.md").is_file():
            flat_prefix = f"{safe_sub}_"
            resolved_artifacts = resolve_run_artifact_dir(
                agent_type, dir_name, sub_run=safe_sub, scope=scope)
            if resolved_artifacts is not None:
                artifact_base = resolved_artifacts
        else:
            return {"error": "Sub-run not found"}

    # Read markdown files
    result_md = ""
    memory_md = ""
    history_md = ""

    p = _latest_result_md_path(base, flat_prefix)
    if p.exists():
        result_md = render_ledger_tables(_render_postanswer_envelope(
            p.read_text(encoding="utf-8", errors="replace")))

    memory_names = (
        (f"{flat_prefix}final_context.md",) if flat_prefix else
        ("_working_memory.md", "working_memory.md")
    )
    for name in memory_names:
        p = base / name
        if p.exists():
            memory_md = p.read_text(encoding="utf-8", errors="replace")
            break

    history_names = (
        (f"{flat_prefix}history.md",) if flat_prefix else
        ("run_history.md", "history.md")
    )
    for name in history_names:
        p = base / name
        if p.exists():
            history_md = p.read_text(encoding="utf-8", errors="replace")
            break

    # Benchmark session histories were intentionally written in compact mode
    # and contain only a short result preview. The JSON test trace retains each
    # complete result_full payload, so use it for the benchmark History tab.
    if scope == "benchmark" and include_full_tool_history:
        trace_match = _find_benchmark_trace_record(
            agent_type, dir_name, sub_run)
        if trace_match:
            record, trace_path = trace_match
            history_md = _render_full_tool_history(
                agent_type, record, trace_path)

    stats_md = ""
    p = artifact_base / "reference_stats.md"
    if p.exists():
        stats_md = p.read_text(encoding="utf-8", errors="replace")

    images, csv_files = discover_run_artifacts(artifact_base)

    # Composition library (normalize_compositions artifact): self-contained
    # conversion metadata (molar masses + density-bridge RK coefficients) so
    # the UI can re-plot fit CSVs in any composition convention.
    composition_library = None
    lib_path = artifact_base / "data" / "composition_library.json"
    if lib_path.is_file():
        try:
            composition_library = json.loads(
                lib_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            composition_library = None

    return {
        "dir_name": dir_name,
        "sub_run": sub_run,
        "result_md": result_md,
        "memory_md": memory_md,
        "history_md": history_md,
        "stats_md": stats_md,
        "images": images,
        "csv_files": csv_files,
        "composition_library": composition_library,
        "sub_runs": sub_runs,
        "session_dir": str(base),
        "artifact_dir": str(artifact_base),
        "scope": scope,
        "workflow": list_workflow_figures(artifact_base),
    }
