"""
Argo API configuration — single source of truth for the ThermoML query agent.
=============================================================================
All model names, API settings, and agent hyperparameters are defined here.
``QueryAgentConfig(EngineConfig)`` — Step 1 fields feed the shared engine
(``argo_engine_helpers``); Step 2 fields are agent-only (L1/L2 dispatchers,
compactors, verdict).  Consumed at runtime by ``orchestrator.py``,
``react_loop.py``, ``postjob_verdict.py``, and compactor modules.

Context window management philosophy:
  - No hardcoded context-char caps.  Instead, the agent itself decides
    when to compress via <compress> tags, and a sub-agent + validation
    cycle handles the actual summarization.
  - Tool results are truncated only when *physically* too large for a
    single LLM call (head+tail preserves both start context and end
    summaries).
  - Soft time budgets warn the agent to wrap up; the agent's own
    judgment decides when to stop searching and answer.
"""

import os
from dataclasses import dataclass, field

from NIST_ThermoML_agents.general_db_query_engine.general_argo_engine_helpers.engine_config import EngineConfig


def _default_api_user() -> str:
    """Read the configured Argo provider account."""
    return os.environ.get("ARGO_API_USER", "").strip()


def is_claude_model(model_name: str) -> bool:
    """Return True for any Claude/Opus model on the Argo proxy."""
    return "claude" in model_name.lower()


@dataclass
class QueryAgentConfig(EngineConfig):
    """Complete configuration for the ThermoML Query Agent.

    Inherits from ``EngineConfig`` (shared engine shape).
    Step 1 fields supply defaults for the 26 engine-required fields.
    Step 2 fields are query-agent-only (L1/L2 workers, compactors, verdict).
    """

    # ═══════════════════════════════════════════════════════════
    # Step 1 — Engine config fields (consumed by argo_engine_helpers)
    # ═══════════════════════════════════════════════════════════

    # ── API connection ──────────────────────────────────────
    API_URL: str = field(default_factory=lambda: os.environ.get(
        "ARGO_API_URL", "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"))  # Argo REST chat endpoint
    API_USER: str = field(default_factory=_default_api_user)  # Argo provider account
    HEADERS: dict = field(default_factory=lambda: {"Content-Type": "application/json"})  # HTTP headers for all Argo calls

    # ── Model names ─────────────────────────────────────────
    MODEL: str = "claudeopus46"           # L0 orchestrator + L1 worker model
    VERDICT_MODEL: str = "claudeopus46"   # post-job quality reviewer
    PLANNER_MODEL: str = "claudeopus46"   # strategy planner (react_loop guidance)

    # ── Generation parameters ───────────────────────────────
    TEMPERATURE: float = 0.3    # sampling temperature for primary MODEL
    TOP_P: float = 0.9          # nucleus sampling (omitted for Claude models)
    MAX_TOKENS: int = 6000      # max tokens per LLM response
    HTTP_TIMEOUT: int = 600     # seconds before HTTP timeout on Argo calls

    # ── Agent loop limits (L0) ──────────────────────────────
    MAX_TOOL_ITERATIONS: int = 25   # hard cap on L0 tool-call iterations → react_loop.agent_turn()
    MAX_TURN_SECONDS: int = 1000    # L0 wall-clock budget (≈17 min)

    # ── L2 async limits ─────────────────────────────────────
    L2_MAX_ITERATIONS: int = 6   # iteration cap per L2 leaf evaluator → react_loop_async
    L2_MAX_SECONDS: int = 90     # time budget per L2 evaluator call

    # ── Time warnings ───────────────────────────────────────
    WARN_THRESHOLDS: list[float] = field(default_factory=lambda: [0.65, 0.85, 0.95])  # fraction-of-budget breakpoints
    MAX_WRAP_WARNINGS: int = 3   # max "wrap up" reminders before forced stop
    MAX_EMPTY_WAITS: int = 3     # max consecutive empty LLM responses before abort

    # ── Tool result truncation ──────────────────────────────
    TOOL_RESULT_CHAR_LIMIT: int = 24_000  # soft limit — oversized result delivered with a size-warning banner

    # ── Compaction (engine) ─────────────────────────────────
    COMPACTION_TRIGGER_CHARS: int = 80_000  # context size triggering LLM-driven compaction
    SUMMARY_MAX_CHARS: int = 1000           # target length for each compacted block
    GUIDANCE_MAX_TOKENS: int = 150          # token budget for compaction-guidance call

    # ── Verdict client (engine) ─────────────────────────────
    VERDICT_TEMPERATURE: float = 0.1   # low temperature for deterministic review
    VERDICT_MAX_TOKENS: int = 500      # verdict response budget

    # ── Compactor client (engine) ───────────────────────────
    COMPACTOR_TEMPERATURE: float = 0.1   # temperature for compaction sub-agent
    COMPACTOR_MAX_TOKENS: int = 1000     # max tokens per compaction response

    # ── Planner client (engine) ─────────────────────────────
    PLANNER_TEMPERATURE: float = 0.2   # slightly creative for strategy planning
    PLANNER_MAX_TOKENS: int = 1000     # planner response budget

    # ── Historical grounding compatibility ──────────────────
    # False: current grounding checks (default).
    # True: historical September grounding policy, with current tools.
    # Set this on the agent STARTING the run; its choice governs every child.
    # This does not restore the full old build or promise identical LLM text.
    GROUNDING_GATE_LEGACY_MODE: bool = False

    # ── Subagent-return ledger levels (engine) ──────────────
    # Ledger abstraction for EVERY transition pair reachable in a QUERY
    # session tree ("full" | "summary" | "none").  ROOT PRECEDENCE:
    # governs standalone query runs down to L1/L2 (L1 workers run under
    # this config via dispatch_l1_query); under a main/analysis root the
    # parent's map wins instead.
    SUBAGENT_LEDGER_LEVELS: dict = field(default_factory=lambda: {
        "context": {
            "Qi/L1": "full",   # L1_query relays into the query L0 loop
            "L1/L2": "full",   # L2 leaf-evaluator returns inside L1 workers
        },
        "memory": {
            "Qi/L1": "summary",  # L1AutoSaver digests in query working memory
        },
    })
    # ═══════════════════════════════════════════════════════
    # Step 2 — Agent-specific fields (NOT in EngineConfig base)
    # ═══════════════════════════════════════════════════════
    # ── Additional model aliases ────────────────────────────
    L1_MODEL: str = "claudeopus46"   # model for L1 query dispatchers
    L2_MODEL: str = "claudeopus46"   # model for L2 leaf evaluators

    # ── Generation extras ───────────────────────────────────
    L2_MAX_TOKENS: int = 2000          # max tokens per L2 evaluator response → argo_client.for_l2()
    REASONING_CAP: int = 10000         # char cap on extended-thinking / reasoning blocks
    SUBAGENT_MAX_TOKENS: int = 700     # token budget for tool-result KEEP/DISCARD sub-agent

    # ── L1 / verdict loop limits ────────────────────────────
    L1_MAX_ITERATIONS: int = 12    # iteration cap for L1 query dispatcher → l1_query_dispatcher
    L1_MAX_SECONDS: int = 360      # wall-clock budget for one L1 dispatch (6 min)
    VERDICT_MAX_SECONDS: int = 60  # time limit for the post-job verdict call

    # ── LLM-driven compaction ───────────────────────────────
    KEEP_RECENT_RESULTS: int = 2             # number of recent tool results exempt from trimming
    MIN_COMPRESS_CHARS: int = 500            # results shorter than this skip compression
    COMPRESS_SHORT_THRESHOLD: int = 1_500    # short-result threshold for abbreviated compress
    MAX_RETRY: int = 5                       # max SELECT→COMPRESS→VALIDATE cycles per compaction
    MAX_IMMEDIATE_RETRY: int = 2             # max immediate retries on VALIDATE=RETRY
    COMPRESS_MAX_WORDS: int = 500            # word cap in compaction sub-agent prompt
    COMPRESS_SUMMARY_CHARS: int = 800        # summary preview chars sent to compactor
    L1_COMPACTION_INTERVAL: int = 999        # > L1 iteration cap — turn-based compaction disabled; size trigger still active

    # ── Stage compaction ────────────────────────────────────
    STAGE_COMPACT_BUDGET: int = 4_000    # char budget for deterministic stage-level compaction
    STAGE_NOTE_CHARS: int = 200          # max chars per stage-note annotation

    # ── Verdict extras ──────────────────────────────────────
    VERDICT_MAX_WORDS: int = 150   # hard word limit imposed on verdict agent response

    # ── Tool-level subagent limits ──────────────────────────
    SUBAGENT_CHAR_LIMIT: int = 40_000      # max raw chars passed to tool sub-agent
    BLOCK_CONDENSE_LIMIT: int = 10_000     # char limit triggering block-level condensation


AGENT_CONFIG = QueryAgentConfig()
