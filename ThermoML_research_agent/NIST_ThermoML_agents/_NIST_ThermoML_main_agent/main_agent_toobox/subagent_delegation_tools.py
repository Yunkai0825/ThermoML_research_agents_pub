"""
Subagent delegation tools — call query and analysis agents as tools.
====================================================================
The main agent's primary tools.  Each wraps a subagent call and returns
the subagent's structured result.

Session nesting
---------------
When the main agent has an active session, each subagent run creates a
subfolder under ``<session_dir>/{query,analysis}_runs/run_N/`` so its
logs (stats, history, reasoning) are co-located with the main output.
After each run, tracking MDs are merged back into the main session.

Tools
-----
run_query_agent       — single query agent run
run_analysis_agent    — single analysis agent run
run_parallel_subagents — parallel dispatch of mixed query/analysis tasks;
                         oversized batches auto-compact through an internal
                         philosophy hook + per-task/consolidation agentic loop
"""
from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List

from ...general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog import ToolEntry
from ...general_db_query_engine.general_text_context_marker_catalog import (
    mark_subagent_answer_tool,
)
from ...general_db_query_engine.general_subagent_delegation_helpers import (
    SubagentSessionManager,
    build_full_question,
    extract_run_result,
    dispatch_parallel,
    merge_subagent_tracking,
    preserve_active_session,
)
from ...general_db_query_engine.general_tool_management_helpers.general_tool_results_compactor_agentic_hooks import (
    ToolResult,
)
from ..main_agent_context_hooks.compactor_hooks._tool_compactors import (
    compact_query_result,
    compact_analysis_result,
)

log = logging.getLogger("MAIN-SUBAGENT-TOOLS")


# ── session-nesting (injected from main hook_catalog) ───────

def _get_session():
    from ..main_agent_context_hooks.hook_catalog import session_manager
    return session_manager.get_session()

_session_mgr = SubagentSessionManager(_get_session)


def _merge_tracking(subagent_dir, agent_type: str, label: str = "") -> None:
    """Combine subagent tracking tables into the main session."""
    main_dir = _session_mgr.get_parent_session_dir()
    merge_subagent_tracking(main_dir, subagent_dir, agent_type, label=label)


#: Inside run_parallel_subagents, children are claimed by the batch step.
_PARALLEL_LAUNCHER: ContextVar[str] = ContextVar(
    "main_parallel_launcher", default="")


def _try_history_recorder():
    try:
        from ...general_db_query_engine.general_hooks_management_helpers.general_context_hooks.history_tracking_hooks import (
            get_active_history_recorder,
        )
        return get_active_history_recorder()
    except (RuntimeError, ImportError):
        return None


@contextmanager
def _nested_child_session(tool_default: str, label: str, short: str,
                          session_dir):
    """Reserve the child's nest identity (Q_n/A_n) before launch so its
    own history renders the full nest path, then link its run_history.md
    as a nested section claimed by the delegation tool's step."""
    from ...general_db_query_engine.general_hooks_management_helpers.general_context_hooks.history_tracking_hooks import (
        reset_pending_nest_prefix,
        set_pending_nest_prefix,
    )
    rec = _try_history_recorder()
    sid = token = None
    if rec is not None and session_dir:
        launcher = _PARALLEL_LAUNCHER.get() or tool_default
        sid = rec.reserve_child_session(launcher, label, short)
        token = set_pending_nest_prefix(rec.child_nest_prefix(sid))
    try:
        yield
    finally:
        if token is not None:
            reset_pending_nest_prefix(token)
        if rec is not None and sid is not None:
            rec.finalize_child_session(
                sid, Path(session_dir) / "run_history.md")


def _log_child_history(tool_name: str, label: str, session_dir,
                       short: str = "") -> None:
    """Register the child's own run_history.md as a nested section in
    the main history (no-op when the child already self-registered)."""
    if not session_dir:
        return
    rec = _try_history_recorder()
    if rec is None:
        return
    rec.log_child_session(tool_name, label,
                          Path(session_dir) / "run_history.md", short=short)


# ═══════════════════════════════════════════════════════════════
#  Query Agent Delegation
# ═══════════════════════════════════════════════════════════════

def _get_query_runner():
    """Lazy import to avoid circular imports at module load."""
    from ...NIST_ThermoML_query_agent.ThermoML_query_api import ThermoML_query_run
    return ThermoML_query_run


def _extract_query_entity_summary(tool_history: list[dict]) -> dict:
    """Derive lightweight entity counts from a query agent's tool history."""
    compounds: set[str] = set()
    properties: set[str] = set()
    dois: set[str] = set()
    native_blocks: set[tuple[str, str]] = set()
    subsystems: set[tuple[str, str, str]] = set()
    total_matching_datapoints = 0

    for entry in tool_history:
        tool_name = entry.get("tool", "")
        raw = entry.get("result_full", "")
        if not isinstance(raw, str):
            continue
        try:
            data = json.loads(raw) if raw.lstrip()[:1] in ("{", "[") else None
        except (json.JSONDecodeError, ValueError):
            data = None
        if not isinstance(data, dict):
            continue

        if tool_name == "resolve_compound_ids":
            if set(data) != {"entity_type", "queries", "n_results", "results"}:
                raise ValueError("resolve_compound_ids history has an invalid result schema")
            if data["entity_type"] != "compound" or not isinstance(data["results"], list):
                raise ValueError("resolve_compound_ids history has invalid entity/results values")
            for result in data["results"]:
                compounds.add(result["common_name"])
        elif tool_name == "resolve_property_ids":
            if set(data) != {"entity_type", "queries", "n_results", "results"}:
                raise ValueError("resolve_property_ids history has an invalid result schema")
            if data["entity_type"] != "property" or not isinstance(data["results"], list):
                raise ValueError("resolve_property_ids history has invalid entity/results values")
            for result in data["results"]:
                properties.add(result["prop_name"])
        elif tool_name == "L1_query":
            if set(data) == {"error", "error_code", "tool"}:
                # Failed L1 calls stay in history as error evidence; they
                # carry no entities and must not fail the whole delegation.
                continue
            expected = {
                "answer",
                "core_claims",
                "status",
                "summary",
                "core_id_updates",
                "core_blocks_found",
                "data_inspections",
            }
            if set(data) != expected:
                raise ValueError(
                    "L1_query history fields must be exactly "
                    f"{sorted(expected)}, received {sorted(data)}"
                )
            if not isinstance(data["core_blocks_found"], list):
                raise TypeError("L1_query history core_blocks_found must be an array")
            for block in data["core_blocks_found"]:
                if (
                    not isinstance(block, dict)
                    or "doi" not in block
                    or "lit_num_id" not in block
                    or "BLKsubsys_id" not in block
                    or "n_datapoints" not in block
                ):
                    raise ValueError("L1_query core_blocks_found entry has an invalid schema")
                dois.add(block["doi"])
                native_blocks.add((block["doi"], block["block_number"]))
                if block["BLKsubsys_id"] is not None:
                    subsystems.add((
                        block["doi"], block["block_number"], block["BLKsubsys_id"]
                    ))
                total_matching_datapoints += block["n_datapoints"]

    summary: dict[str, Any] = {}
    if compounds:
        summary["n_compounds"] = len(compounds)
    if properties:
        summary["n_properties"] = len(properties)
    if dois:
        summary["n_dois"] = len(dois)
    if native_blocks:
        summary["n_blocks"] = len(native_blocks)
    if subsystems:
        summary["n_subsystems"] = len(subsystems)
    if total_matching_datapoints:
        summary["n_matching_datapoints"] = total_matching_datapoints
    return summary


@mark_subagent_answer_tool
def run_query_agent(
    question: str,
    purpose: str,
    context: str,
) -> dict:
    """Run the ThermoML Query Agent on a database search question.

    The query agent resolves compound/property names, searches the
    ThermoML card databases, and returns structured results with
    DOIs, block numbers, and data summaries. For candidate comparisons,
    request measured values with their compositions and conditions.

    Parameters
    ----------
    question : str
        The database search question (e.g. "Find all viscosity data
        for ethanol + water binary mixtures at 298.15 K").
    purpose : str
        High-level intent for the search (helps the agent prioritize).
    context : str
        Prior context from earlier tool calls or working memory.

    Returns
    -------
    dict
        Keys: answer (str), verdict (str|None), iterations (int),
        elapsed_seconds (float), tool_history (list), timed_out (bool),
        data_inspections (list).
    """
    full_question = build_full_question(question, purpose=purpose, context=context)

    log.info("Dispatching to query agent: %s", question[:200])
    session_dir = _session_mgr.get_subagent_session_dir("query")
    with _nested_child_session("run_query_agent", "Q-agent", "Q",
                               session_dir), preserve_active_session():
        result = _get_query_runner()(
            full_question, session_dir=session_dir, run_verdict=True,
        )

    tool_hist = result.tool_history
    if not isinstance(tool_hist, list):
        raise TypeError("query subagent result.tool_history must be an array")
    entity_summary = _extract_query_entity_summary(tool_hist)

    subagent_session = session_dir
    _merge_tracking(subagent_session, "query", label=question[:60])

    return extract_run_result(
        result, "query",
        question=question,
        session_dir=subagent_session,
        extra=dict(entity_summary, verdict=result.verdict),
    )


# ═══════════════════════════════════════════════════════════════
#  Analysis Agent Delegation
# ═══════════════════════════════════════════════════════════════

def _get_analysis_runner():
    """Lazy import to avoid circular imports at module load."""
    from ...NIST_ThermoML_analysis_agent.ThermoML_analysis_api import ThermoML_analysis_run
    return ThermoML_analysis_run


@mark_subagent_answer_tool
def run_analysis_agent(
    question: str,
    purpose: str,
    context: str,
) -> dict:
    """Run the ThermoML Analysis Agent for fitting and modeling.

    The analysis agent searches for data via the query agent, extracts
    pure-component endpoints, fits Redlich-Kister polynomials, and
    returns quantitative fitting results with R² and RK coefficients.

    Parameters
    ----------
    question : str
        The analysis question (e.g. "Fit the excess molar volume of
        ethanol + water at 298.15 K using Redlich-Kister polynomials").
    purpose : str
        High-level intent (helps guide the analysis strategy).
    context : str
        Prior context from earlier tool calls or working memory.

    Returns
    -------
    dict
        Keys: answer (str), verdict (str|None), iterations (int),
        elapsed_seconds (float), tool_count (int), timed_out (bool),
        session_dir (str|None), output_files (list).
    """
    full_question = build_full_question(question, purpose=purpose, context=context)

    log.info("Dispatching to analysis agent: %s", question[:200])
    session_dir = _session_mgr.get_subagent_session_dir("analysis")
    with _nested_child_session("run_analysis_agent", "A-agent", "A",
                               session_dir), preserve_active_session():
        result = _get_analysis_runner()(full_question, run_verdict=True, session_dir=session_dir)

    subagent_session = session_dir
    _merge_tracking(subagent_session, "analysis", label=question[:60])

    return extract_run_result(
        result, "analysis",
        question=question,
        session_dir=subagent_session,
        extra={
            "verdict": result.verdict,
            "session_dir": str(subagent_session),
            "output_files": result.output_files,
        },
    )


# ═══════════════════════════════════════════════════════════════
#  Parallel Dispatch (unified — query + analysis in one call)
# ═══════════════════════════════════════════════════════════════

# Lazy singleton — the compactor pulls in the main Argo client stack.
_batch_compactor = None


def _get_batch_compactor():
    global _batch_compactor
    if _batch_compactor is None:
        from ..main_agent_context_hooks.compactor_hooks import MainToolResultCompactor
        _batch_compactor = MainToolResultCompactor()
    return _batch_compactor


def _batch_compaction_brief(tasks: list[dict]) -> tuple[str, str]:
    """Purpose/tasks brief for the batch compactor from the original task specs."""
    purpose_lines = [
        f"Parallel dispatch of {len(tasks)} subagent task(s). Compact the "
        "combined results for the orchestrator; per-task purposes:",
    ]
    task_lines = []
    for task in tasks:
        purpose_lines.append(f"- [{task['label']}] ({task['agent']}): {task['purpose']}")
        task_lines.append(f"- [{task['label']}] {task['question']}")
    task_lines.append(
        "Report each task under its [label] heading with verbatim key "
        "identifiers (GLOB*_N, PROPblock_N/RXNblock_N, DOIs), value ranges, "
        "and per-task error/timeout status. Partial failures must still be "
        "KEPT with the failing tasks flagged; DISCARD only if every task "
        "returned no usable data. The full JSON results are already recorded "
        "in working memory."
    )
    return "\n".join(purpose_lines), "\n".join(task_lines)


def _fallback_batch_text(batch: dict) -> str:
    """Whole-field snapshot when single-pass agentic compaction fails.

    Never slices strings: per task it reports run stats plus the child's
    own complete ``core_claims`` (its authored abstraction). The full
    results stay in working memory.
    """
    lines = [
        f"**Parallel dispatch** — {batch['n_tasks']} task(s); full JSON "
        "results are recorded in working memory.",
    ]
    for entry in batch["results"]:
        label = entry.get("label", "?")
        agent = entry.get("agent", "?")
        if "error" in entry and "answer" not in entry:
            lines += ["", f"### [{label}] ({agent}) — ERROR", entry["error"]]
            continue
        lines += ["", _task_status_line(entry).lstrip("- "), "",
                  _whole_field_digest(entry)]
    lines += [
        "",
        "Agentic compaction was unavailable for this batch; the sections "
        "above are the children's complete self-authored claims. Read "
        "working memory for the full results.",
    ]
    return "\n".join(lines)


# ── Automated two-stage compaction for oversized batches ───────


def _batch_char_limit() -> int:
    from ..ThermoML_main_argo_config import AGENT_CONFIG
    return AGENT_CONFIG.SUBAGENT_CHAR_LIMIT


def _task_status_line(entry: dict) -> str:
    label = entry.get("label", "?")
    agent = entry.get("agent", "?")
    if "error" in entry and "answer" not in entry:
        return f"- [{label}] ({agent}) ERROR — full message in working memory"
    flags = ", TIMED OUT" if entry.get("timed_out") else ""
    return (
        f"- [{label}] ({agent}) OK — {entry.get('iterations', 0)} iters, "
        f"{entry.get('elapsed_seconds', 0.0):.0f}s, "
        f"{entry.get('tool_count', 0)} tools, "
        f"answer {len(entry.get('answer', '')):,} chars{flags}"
    )


def _parse_answer_envelope(entry: dict) -> dict | None:
    """Child answers are usually serialized JSON envelopes; parse if so."""
    answer = entry.get("answer", "")
    if not isinstance(answer, str) or answer.lstrip()[:1] != "{":
        return None
    try:
        env = json.loads(answer)
    except (json.JSONDecodeError, ValueError):
        return None
    return env if isinstance(env, dict) else None


def _whole_field_digest(entry: dict, note: str = "") -> str:
    """Complete-field per-task digest — selection of whole fields only.

    Prefers the child's own ``core_claims`` (already an abstraction of
    its data). Fields are included complete or omitted entirely with a
    working-memory pointer; nothing is character-truncated.
    """
    lines = []
    if note:
        lines.append(f"**Note:** {note}")
    env = _parse_answer_envelope(entry)
    claims = env.get("core_claims") if env else None
    if isinstance(claims, list) and claims:
        lines.append("**Child-authored core claims (complete):**")
        lines += [f"- {c}" for c in claims]
        confidence = env.get("confidence")
        if isinstance(confidence, str) and confidence:
            lines.append(f"**Confidence:** {confidence}")
    if entry.get("verdict") is not None:
        lines.append(f"**Verdict:** {entry['verdict']}")
    if entry.get("agent") == "analysis":
        output_files = entry.get("output_files") or []
        if output_files:
            lines.append(
                "**Output files:** "
                + ", ".join(
                    f"{f.get('path', '?')} ({f.get('category', '?')})"
                    for f in output_files
                    if isinstance(f, dict)
                )
            )
    if not (isinstance(claims, list) and claims):
        lines.append(
            f"**Answer:** {len(entry.get('answer', '')):,} chars — no "
            "structured claims available; full text in working memory."
        )
    return "\n".join(lines)


def _reduce_entry_for_compaction(entry: dict, limit: int) -> dict:
    """Fit an oversized task under the subagent limit by dropping whole
    envelope fields (largest first, recorded in ``_omitted_fields``) —
    never by slicing strings."""
    reduced = {k: v for k, v in entry.items() if k != "answer"}
    env = _parse_answer_envelope(entry)
    if env is None:
        env = {"answer": entry.get("answer", "")}
    omitted: list[str] = []

    def _payload() -> str:
        return json.dumps(
            {**reduced, "answer_envelope": env, "_omitted_fields": omitted},
            indent=2, ensure_ascii=False, default=str,
        )

    while len(_payload()) > limit and env:
        biggest = max(
            env, key=lambda k: len(json.dumps(env[k], ensure_ascii=False, default=str))
        )
        size = len(json.dumps(env[biggest], ensure_ascii=False, default=str))
        omitted.append(
            f"{biggest} ({size:,} chars — full text in working memory)"
        )
        env.pop(biggest)
    return {**reduced, "answer_envelope": env, "_omitted_fields": omitted}


def _log_event(event: str, detail: str, output_chars: int = 0) -> None:
    """Best-effort history event; recorders are absent in bare test runs."""
    try:
        from ...general_db_query_engine.general_hooks_management_helpers.general_context_hooks.history_tracking_hooks import (
            get_active_history_recorder,
        )
        get_active_history_recorder().log_subagent_event(
            tool_name="run_parallel_subagents", event=event,
            detail=detail, output_chars=output_chars,
        )
    except Exception:
        log.debug("history recorder unavailable for %s event", event, exc_info=True)


def _default_philosophy(tasks: list[dict]) -> str:
    """Deterministic philosophy from the task purposes (elicitation fallback)."""
    goals = "; ".join(t["purpose"] for t in tasks)
    return (
        f"Serve these goals: {goals}. For each task keep the core findings, "
        "the most representative data points with values/units/conditions, "
        "differences in trends across systems, coverage or availability "
        "verdicts, and every identifier needed for citation (GLOB*_N, "
        "PROPblock_N/RXNblock_N, DOIs). Drop exhaustive tables, repeated "
        "metadata, and narrative framing."
    )


_MAX_HOOK_CORRECTIONS = 2  # extra hook rounds to fix malformed replies


def _numbered_results_context(tasks: list[dict], batch: dict) -> str:
    """Numbered per-result context for the hook: sizes + the arguments
    that produced each result (complete, never sliced)."""
    task_by_label = {t["label"]: t for t in tasks}
    lines = []
    for i, entry in enumerate(batch["results"], 1):
        label = entry.get("label", "?")
        spec = task_by_label.get(label, {})
        size = len(json.dumps(entry, ensure_ascii=False, default=str))
        if "error" in entry and "answer" not in entry:
            status = "ERROR"
        elif entry.get("timed_out"):
            status = "TIMED OUT"
        else:
            status = "OK"
        lines.append(
            f"#{i} [{label}] ({entry.get('agent', '?')}) {status} — "
            f"result {size:,} chars"
        )
        lines.append(
            f"    arguments: question={spec.get('question', entry.get('_question', ''))!r}; "
            f"purpose={spec.get('purpose', '')!r}; "
            f"context={spec.get('context', '')!r}"
        )
    return "\n".join(lines)


def _parse_hook_reply(reply: str, n_results: int) -> tuple[bool, set | None, str]:
    """Parse the hook reply; raises ValueError with a correction message.

    Returns ``(need, targets, philosophy)`` — targets is ``None`` for ALL,
    else a validated set of 1-based result numbers.
    """
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("empty reply")
    need_m = re.search(r"NEED_COMPACTION\s*:\s*(YES|NO)\b", reply, re.I)
    if need_m is None:
        raise ValueError("missing or invalid NEED_COMPACTION line (must be YES or NO)")
    need = need_m.group(1).upper() == "YES"
    phil_m = re.search(
        r"PHILOSOPHY\s*:\s*(.+?)(?:\n\s*(?:TARGET|NEED_COMPACTION)\s*:|\Z)",
        reply, re.I | re.S,
    )
    philosophy = phil_m.group(1).strip() if phil_m else ""
    if len(philosophy) < 20:
        raise ValueError(
            "missing or too-short PHILOSOPHY line (state goal, what to keep, "
            "what to drop; at least 20 characters)"
        )
    targets: set | None = None
    if need:
        tgt_m = re.search(r"^\s*TARGET\s*:\s*(.+)$", reply, re.I | re.M)
        if tgt_m is None:
            raise ValueError(
                "missing TARGET line (ALL, or the result numbers to compress, "
                "e.g. TARGET: 1,3)"
            )
        tgt = tgt_m.group(1).strip()
        if re.fullmatch(r"\[?\s*ALL\s*\]?", tgt, re.I):
            targets = None
        else:
            nums = [int(x) for x in re.findall(r"\d+", tgt)]
            if not nums:
                raise ValueError(
                    f"TARGET must be ALL or result numbers, got {tgt!r}"
                )
            targets = set(nums)
            bad = sorted(x for x in targets if not 1 <= x <= n_results)
            if bad:
                raise ValueError(
                    f"TARGET numbers {bad} out of range 1..{n_results}"
                )
    return need, targets, philosophy


def _elicit_philosophy(
    tasks: list[dict], batch: dict, payload_chars: int, limit: int,
) -> tuple[bool, set | None, str, str, int]:
    """Hook at the anchor point: ask the orchestrator's model tier whether
    compaction is needed, for which numbered results, and under what
    philosophy — with a bounded correction loop for malformed replies.

    Returns ``(need, targets, philosophy, source, correction_rounds)``;
    targets is ``None`` for ALL. Any failure falls back to the
    deterministic default philosophy over all results — the automated
    loop never dead-ends.
    """
    n_results = len(batch["results"])
    numbered = _numbered_results_context(tasks, batch)
    fmt = (
        "Reply in EXACTLY this format:\n"
        "NEED_COMPACTION: YES or NO\n"
        "TARGET: ALL, or the result numbers to compress (e.g. 1,3)\n"
        "PHILOSOPHY: <2-4 sentences>"
    )
    prompt = (
        f"A parallel dispatch you ordered has finished: {batch['n_tasks']} "
        f"task(s), combined results {payload_chars:,} chars — above the "
        f"{limit:,}-char single-pass compaction limit. Full results are "
        "already recorded in working memory.\n\n"
        f"Numbered results (sizes and the arguments that produced them):\n"
        f"{numbered}\n\n"
        "Targeted results will each be compacted by an agentic compactor "
        "guided by a compaction philosophy, then consolidated; untargeted "
        "results pass as their complete child-authored core claims. State "
        "which numbered results to compress (typically the large verbose "
        "ones) and the philosophy: the goal these results serve, which "
        "quantities, comparisons, or trends must be kept (most "
        "representative data, trend differences, citation identifiers), "
        "and what may be dropped. If the children's core claims are "
        "already sufficient for your synthesis, you may decline "
        "compaction.\n\n" + fmt
    )
    system = (
        "You are the orchestrator of a ThermoML multi-agent run deciding "
        "how oversized parallel results are compacted before entering "
        "your context. Be specific and brief."
    )
    try:
        factory = getattr(_get_batch_compactor(), "client_factory", None)
        if factory is None:
            raise RuntimeError("batch compactor exposes no client_factory")
        client = factory()
        reply = client.call(prompt, system, max_tokens=500)
        for round_no in range(_MAX_HOOK_CORRECTIONS + 1):
            try:
                need, targets, philosophy = _parse_hook_reply(reply, n_results)
                return need, targets, philosophy, "elicited", round_no
            except ValueError as err:
                if round_no >= _MAX_HOOK_CORRECTIONS:
                    raise
                _log_event("PHILOSOPHY_RETRY", f"round {round_no + 1}: {err}")
                correction = (
                    prompt
                    + "\n\n--- Your previous reply ---\n" + reply
                    + "\n\n--- Problem ---\n" + str(err)
                    + "\nCorrect it and reply again in EXACTLY the required format."
                )
                reply = client.call(correction, system, max_tokens=500)
        raise RuntimeError("unreachable")
    except Exception as exc:
        log.warning("Philosophy elicitation unavailable (%s); using default", exc)
        return True, None, _default_philosophy(tasks), "default", 0


def _auto_two_stage(
    tasks: list[dict], batch: dict, payload_chars: int, limit: int,
) -> ToolResult:
    """Automated oversized-batch loop: philosophy hook → targeted per-task
    agentic compaction → agentic consolidation. All fallbacks are
    whole-field digests — never character truncation."""
    _log_event("AUTO2STAGE", f"batch={payload_chars:,} chars > limit={limit:,}")
    need, targets, philosophy, source, rounds = _elicit_philosophy(
        tasks, batch, payload_chars, limit
    )
    target_desc = "ALL" if targets is None else ",".join(map(str, sorted(targets)))
    _log_event(
        "PHILOSOPHY",
        f"({source}, compaction={'YES' if need else 'NO'}, targets={target_desc}, "
        f"corrections={rounds}) {philosophy}",
    )
    task_by_label = {t["label"]: t for t in tasks}
    compactor = _get_batch_compactor()

    reports: Dict[str, str] = {}
    for i, entry in enumerate(batch["results"], 1):
        label = entry.get("label", "?")
        if "error" in entry and "answer" not in entry:
            reports[label] = f"**Status: ERROR** — {entry['error']}"
            continue
        targeted = need and (targets is None or i in targets)
        if not targeted:
            reports[label] = _whole_field_digest(
                entry,
                note="not targeted for compression — child-authored claims"
                if need else "",
            )
            continue
        spec = task_by_label.get(label, {})
        task_purpose = (
            f"{spec.get('purpose', '')}\n\n"
            f"Orchestrator compaction philosophy: {philosophy}"
        )
        task_question = spec.get("question", entry.get("_question", ""))
        payload_obj: dict = entry
        payload = json.dumps(entry, indent=2, ensure_ascii=False, default=str)
        if len(payload) > limit:
            payload_obj = _reduce_entry_for_compaction(entry, limit)
            payload = json.dumps(payload_obj, indent=2, ensure_ascii=False, default=str)
        try:
            per_task = compactor.call(
                f"run_parallel_subagents[{label}]",
                task_purpose, task_question, payload, payload_obj,
            )
            reports[label] = per_task.text
        except Exception as exc:
            log.warning("Per-task compaction failed for [%s]: %s", label, exc)
            reports[label] = _whole_field_digest(
                entry, note=f"agentic compaction failed ({exc})"
            )

    batch["_auto_compaction"] = {
        "philosophy": philosophy,
        "philosophy_source": source,
        "compaction": "agentic" if need else "declined",
        "targets": target_desc,
        "correction_rounds": rounds,
        "per_task_reports": reports,
    }
    scope = ("all tasks" if targets is None
             else f"tasks {target_desc} (others pass as complete claims)")
    action = (
        f"so {scope} were auto-compacted under this philosophy ({source}): "
        if need else
        f"but the compaction hook declined — complete child-authored claims "
        f"follow (philosophy guidance, {source}): "
    )
    header = (
        f"**Parallel dispatch** — {batch['n_tasks']} task(s); combined "
        f"results {payload_chars:,} chars exceeded the {limit:,}-char "
        f"single-pass limit, {action}{philosophy}\n"
        "Full JSON results are recorded in working memory.\n"
    )
    concat = "\n\n".join(f"### [{label}]\n{text}" for label, text in reports.items())
    if not need:
        return ToolResult(raw=batch, text=header + "\n" + concat)

    consolidation_purpose, consolidation_tasks = _batch_compaction_brief(tasks)
    consolidation_purpose += (
        f"\n\nOrchestrator compaction philosophy: {philosophy}\n"
        "The input is per-task reports already compacted under this "
        "philosophy; consolidate them, surface cross-task comparisons, "
        "and keep every [label] section."
    )
    if len(concat) <= limit:
        try:
            consolidated = compactor.call(
                "run_parallel_subagents[consolidate]",
                consolidation_purpose, consolidation_tasks, concat,
                {"per_task_reports": reports},
            )
            return ToolResult(raw=batch, text=header + "\n" + consolidated.text)
        except Exception as exc:
            log.warning(
                "Batch consolidation unavailable (%s); returning "
                "per-task reports directly", exc,
            )
    else:
        log.warning(
            "Per-task reports total %d chars > limit %d; returning them "
            "directly without a consolidation pass", len(concat), limit,
        )
    return ToolResult(raw=batch, text=header + "\n" + concat)


def run_parallel_subagents(tasks: list[dict]) -> ToolResult:
    """Run multiple query and/or analysis agent tasks in parallel.

    Each task specifies which agent to use via the ``agent`` field.
    This is the single parallel dispatch tool — use it whenever you
    need to run more than one subagent call concurrently, regardless
    of whether they are queries, analyses, or a mix of both.

    Parameters
    ----------
    tasks : list[dict]
        Task specs, each with keys:
        - agent: "query" or "analysis" (required)
        - label: identifier for this task (required)
        - question: the question to ask (required)
        - purpose: high-level intent (required)
        - context: prior context from working memory (required)

        Example::

            [
              {"agent": "query", "label": "density_search",
               "question": "Find density data for ethanol+water",
               "purpose": "Gather raw density measurements for comparison",
               "context": "User asked to compare ethanol-water and methanol-water"},
              {"agent": "analysis", "label": "ve_fit",
               "question": "Fit excess molar volume for ethanol+water at 298 K",
               "purpose": "Obtain RK coefficients for VE correlation",
               "context": "Need VE fit to answer the user's fitting request"}
            ]

    Returns
    -------
    str (compacted markdown)
        A compacted report covering every task under its ``[label]``
        heading — status (OK / ERROR / TIMED OUT), key identifiers, and
        headline data.

        When the combined results exceed the single-pass compaction
        limit they are auto-compacted by an internal loop (you are
        consulted mid-tool in a self-contained exchange); the report
        states what was applied.

        In every case the full per-task JSON results (same keys as
        ``run_query_agent`` / ``run_analysis_agent`` plus ``label`` and
        ``agent``) are recorded verbatim in working memory under
        "Query/Analysis Agent Results".
    """
    if not isinstance(tasks, list) or not tasks:
        raise TypeError("tasks must be a non-empty array of task objects")
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise TypeError(f"tasks[{index}] must be an object")
        required = {"agent", "label", "question", "purpose", "context"}
        missing = sorted(required - set(task))
        unknown = sorted(set(task) - required)
        if missing:
            raise ValueError(f"tasks[{index}] is missing fields {missing}")
        if unknown:
            raise ValueError(f"tasks[{index}] has unknown fields {unknown}")
        if task["agent"] not in {"query", "analysis"}:
            raise ValueError(f"tasks[{index}].agent must be 'query' or 'analysis'")
        for field in required - {"agent"}:
            if not isinstance(task[field], str) or not task[field].strip():
                raise TypeError(f"tasks[{index}].{field} must be a non-empty string")

    # Smart worker-count: queries are lighter (up to 3), analyses heavier (up to 2)
    n_queries = sum(1 for t in tasks if t.get("agent") == "query")
    n_analyses = len(tasks) - n_queries
    max_workers = min(len(tasks), max(min(n_queries, 3) + min(n_analyses, 2), 1))

    def _dispatch_one(question: str, purpose: str, context: str, agent: str) -> dict:
        runner = run_analysis_agent if agent == "analysis" else run_query_agent
        token = _PARALLEL_LAUNCHER.set("run_parallel_subagents")
        try:
            return runner(question=question, purpose=purpose, context=context)
        finally:
            _PARALLEL_LAUNCHER.reset(token)

    batch = dispatch_parallel(
        tasks,
        _dispatch_one,
        max_workers=max_workers,
        runner_kwargs_keys=("question", "purpose", "context", "agent"),
    )
    # Fallback for children whose pool context missed the recorder —
    # normally they self-register (Q_n/A_n) inside _dispatch_one.
    for entry in batch.get("results") or []:
        if isinstance(entry, dict) and entry.get("session_dir"):
            agent_kind = entry.get("agent", "query")
            _log_child_history(
                "run_parallel_subagents",
                "A-agent" if agent_kind == "analysis" else "Q-agent",
                entry["session_dir"],
                short="A" if agent_kind == "analysis" else "Q")

    # Agentic-only compaction: the orchestrator context receives compacted
    # markdown; the raw batch dict travels via ToolResult.raw into working
    # memory (memory instrumentation records it verbatim).
    purpose, tasks_brief = _batch_compaction_brief(tasks)
    payload = json.dumps(batch, indent=2, ensure_ascii=False)
    limit = _batch_char_limit()
    if len(payload) > limit:
        log.warning(
            "Parallel batch %d chars > limit %d; running automated "
            "philosophy-guided two-stage compaction", len(payload), limit,
        )
        return _auto_two_stage(tasks, batch, len(payload), limit)
    try:
        return _get_batch_compactor().call(
            "run_parallel_subagents", purpose, tasks_brief, payload, batch,
        )
    except Exception as exc:
        log.warning(
            "Parallel batch compaction unavailable (%s); "
            "returning whole-field snapshot", exc,
        )
        return ToolResult(raw=batch, text=_fallback_batch_text(batch))


# ═══════════════════════════════════════════════════════════════
#  Tool Entries (consumed by tool_catalog.py)
# ═══════════════════════════════════════════════════════════════

TOOL_ENTRIES = [
    ToolEntry(
        "run_query_agent", run_query_agent,
        group="subagent_delegation",
        compactor_fn=compact_query_result,
        skip_compactor=False, skip_subagent=False,
    ),
    ToolEntry(
        "run_analysis_agent", run_analysis_agent,
        group="subagent_delegation",
        compactor_fn=compact_analysis_result,
        skip_compactor=False, skip_subagent=False,
    ),
    ToolEntry(
        "run_parallel_subagents", run_parallel_subagents,
        group="parallel_delegation",
        # Catalog pass-through: the tool runs its own agentic batch
        # compaction (single-pass or automated two-stage) internally
        # and returns a ToolResult.
        skip_compactor=True, skip_subagent=True,
    ),
]


def _query_l0_tools():
    from ...NIST_ThermoML_query_agent.query_agent_workflows.L0_orchestrator.orchestrator import (
        available_query_l0_tools,
    )
    return available_query_l0_tools()


def _analysis_l0_tools():
    from ...NIST_ThermoML_analysis_agent.analysis_agent_toolbox.tool_catalog import (
        ANALYSIS_CATALOG,
    )
    return ANALYSIS_CATALOG.wrapped_tools()


from ...general_db_query_engine.general_tool_management_helpers.tool_capabilities import (
    bind_reachable_tools,
)

bind_reachable_tools(run_query_agent, _query_l0_tools)
bind_reachable_tools(run_analysis_agent, _analysis_l0_tools)
bind_reachable_tools(
    run_parallel_subagents,
    lambda: {
        "run_query_agent": run_query_agent,
        "run_analysis_agent": run_analysis_agent,
    },
)
