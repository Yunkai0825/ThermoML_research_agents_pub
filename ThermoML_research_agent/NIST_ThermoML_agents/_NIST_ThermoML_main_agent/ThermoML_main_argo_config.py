"""
Argo API configuration — ThermoML Main Agent.
==============================================
Lightweight orchestrator-only configuration.  The main agent delegates
all database work to the query agent and all fitting work to the
analysis agent.  It needs generous loop/time budgets because subagent
calls are inherently slow, but modest token budgets because its own
reasoning is short.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path as _Path

from NIST_ThermoML_agents.general_db_query_engine.general_argo_engine_helpers.engine_config import EngineConfig

_PROJECT_ROOT = _Path(__file__).absolute().parent
_REPO_ROOT = _PROJECT_ROOT.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))



def _default_api_user() -> str:
    """Read the configured Argo provider account."""
    return os.environ.get("ARGO_API_USER", "").strip()


def _main_output_dir() -> _Path:
    return _REPO_ROOT / "_output" / "Main"


def _main_transcript_dir() -> _Path:
    return _main_output_dir() / "transcripts"


@dataclass
class MainAgentConfig(EngineConfig):
    """Complete configuration for the ThermoML Main Agent.

    Inherits from ``EngineConfig`` (shared engine shape).
    Step 1 fields supply defaults for the 26 engine-required fields.
    Step 2 fields are main-agent-only parameters.

    Key differences from the sub-agents:
      - MAX_TOOL_ITERATIONS = 40 — multi-subagent workflows need many calls
      - MAX_TURN_SECONDS = 2400 (40 min) — subagent calls are slow
      - No PROTECT_TOOLS — all tool results can be compacted
      - No L1/L2 worker configs — delegates entirely to subagents
    """

    # ═══════════════════════════════════════════════════════════
    # Step 1 — Engine config fields (consumed by argo_engine_helpers)
    # ═══════════════════════════════════════════════════════════

    # ── API connection ──────────────────────────────────────
    API_URL: str = field(default_factory=lambda: os.environ.get(
        "ARGO_API_URL", "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"))
    API_USER: str = field(default_factory=_default_api_user)
    HEADERS: dict = field(default_factory=lambda: {"Content-Type": "application/json"})

    # ── Models ───────────────────────────────────────────────
    MODEL: str = "claudeopus46"           # L0 orchestrator model
    VERDICT_MODEL: str = "claudeopus46"   # post-job quality reviewer
    PLANNER_MODEL: str = "claudeopus46"   # strategy planner

    # ── Generation parameters ───────────────────────────────
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    MAX_TOKENS: int = 6000
    HTTP_TIMEOUT: int = 600

    # ── Agent loop limits (L0) ──────────────────────────────
    MAX_TOOL_ITERATIONS: int = 40     # generous — subagent calls are slow
    MAX_TURN_SECONDS: int = 2400      # 40 min total budget

    # ── L2 async limits (unused — no L2 workers) ────────────
    L2_MAX_ITERATIONS: int = 6
    L2_MAX_SECONDS: int = 90

    # ── Time warnings ───────────────────────────────────────
    WARN_THRESHOLDS: list[float] = field(default_factory=lambda: [0.65, 0.85, 0.95])
    MAX_WRAP_WARNINGS: int = 3
    MAX_EMPTY_WAITS: int = 3

    # ── Tool result truncation ──────────────────────────────
    TOOL_RESULT_CHAR_LIMIT: int = 30_000   # soft limit — oversized result delivered with a size-warning banner

    # ── Compaction (engine) ─────────────────────────────────
    COMPACTION_TRIGGER_CHARS: int = 100_000  # higher threshold — subagent results are verbose
    SUMMARY_MAX_CHARS: int = 1200
    GUIDANCE_MAX_TOKENS: int = 150

    # ── Verdict client (engine) ─────────────────────────────
    VERDICT_TEMPERATURE: float = 0.1
    VERDICT_MAX_TOKENS: int = 2000

    # ── Compactor client (engine) ───────────────────────────
    COMPACTOR_TEMPERATURE: float = 0.1
    COMPACTOR_MAX_TOKENS: int = 1200

    # ── Planner client (engine) ─────────────────────────────
    PLANNER_TEMPERATURE: float = 0.2
    PLANNER_MAX_TOKENS: int = 1000

    # ── Historical grounding compatibility ──────────────────
    # False: current grounding checks (default).
    # True: historical September grounding policy, with current tools.
    # Set this on the agent STARTING the run; its choice governs every child.
    # This does not restore the full old build or promise identical LLM text.
    GROUNDING_GATE_LEGACY_MODE: bool = False

    # ── Subagent-return ledger levels (engine) ──────────────
    # Ledger abstraction for EVERY transition pair reachable in a MAIN
    # session tree ("full" | "summary" | "none").  ROOT PRECEDENCE: main
    # is the root, so this map governs the whole tree — it overwrites the
    # query/analysis configs' own maps for all subagents nested under it,
    # hence it lists every subagent boundary down to L1/L2.
    # context = parent-loop tool result; memory = working-memory digest
    # of the same envelope (pairs with a WM digest surface only).
    SUBAGENT_LEDGER_LEVELS: dict = field(default_factory=lambda: {
        "context": {
            "main/Qi": "full",   # run_query_agent returns
            "main/Ai": "full",   # run_analysis_agent returns
            "main/L1": "full",   # menu-forwarded L1 worker relays
            "main/L2": "full",   # menu-forwarded L2 leaf evaluators
            "Ai/Qi": "full",     # nested query agents inside analysis children
            "Ai/L1": "full",     # query_thermoml / align_compositions relays
            "Qi/L1": "full",     # L1_query relays inside query children
            "L1/L2": "full",     # L2 leaf evaluators inside L1 workers
        },
        "memory": {
            "main/Qi": "summary",  # query child digests stored in main WM
            "main/Ai": "summary",  # analysis child digests stored in main WM
            "Ai/L1": "summary",    # L1 relay digests in analysis-child WM
            "Qi/L1": "summary",    # L1AutoSaver digests in query-child WM
        },
    })
    # ═══════════════════════════════════════════════════════
    # Step 2 — Main-agent-specific fields
    # ═══════════════════════════════════════════════════════
    # ── Generation extras ───────────────────────────────────
    REASONING_CAP: int = 10000
    SUBAGENT_MAX_TOKENS: int = 500

    # ── Menu-planner sub-agent (browse_subagent_tools) ──────
    MENU_PLANNER_MODEL: str = "claudesonnet46"  # lighter model for tool-menu planning
    MENU_PLANNER_MAX_TOKENS: int = 2000          # output budget for menu planner

    # ── Verdict extras ──────────────────────────────────────
    VERDICT_MAX_SECONDS: int = 90
    VERDICT_ANSWER_CHARS: int = 6_000
    VERDICT_MAX_WORDS: int = 200
    VERDICT_FORMAT_MAX_TOKENS: int = 1000

    # ── LLM-driven compaction ───────────────────────────────
    KEEP_RECENT_RESULTS: int = 2
    MIN_COMPRESS_CHARS: int = 600
    COMPRESS_SHORT_THRESHOLD: int = 2_000
    MAX_RETRY: int = 3
    MAX_IMMEDIATE_RETRY: int = 2
    COMPRESS_MAX_WORDS: int = 600
    COMPRESS_SUMMARY_CHARS: int = 1000
    COMPACTION_INTERVAL: int = 999    # > MAX_TOOL_ITERATIONS — turn-based compaction disabled; size trigger still active

    # ── Stage compaction (deterministic) ────────────────────
    STAGE_COMPACT_BUDGET: int = 5_000
    STAGE_NOTE_CHARS: int = 250

    # ── Tool-level subagent limits ──────────────────────────
    SUBAGENT_CHAR_LIMIT: int = 50_000

    # ── Protected tools — none for main agent ───────────────
    PROTECT_TOOLS: set[str] = field(default_factory=set)

    # ── Artifact directories ────────────────────────────────
    OUTPUT_DIR: _Path = field(default_factory=_main_output_dir)
    TRANSCRIPT_DIR: _Path = field(default_factory=_main_transcript_dir)


AGENT_CONFIG = MainAgentConfig()
