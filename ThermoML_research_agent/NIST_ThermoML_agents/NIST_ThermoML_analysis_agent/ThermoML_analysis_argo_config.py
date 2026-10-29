"""
Argo API configuration — ThermoML Analysis Agent.
==================================================
Analysis-specific budgets and parameters.  Shares the same Argo API
endpoint and credentials as the query agent.

Pipeline phases:
  Phase 0  Planning          — identify systems, properties, query strategy
  Phase 1  Data Retrieval    — call query agent search tools → get raw block data
  Phase 2  Fitting / Calc    — ideal baseline, Redlich-Kister, nonideality
  Phase 3  Verdict           — independent review of fit quality & results
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path as _Path
import datetime as _dt

from NIST_ThermoML_agents.general_db_query_engine.general_argo_engine_helpers.engine_config import EngineConfig

_PROJECT_ROOT = _Path(__file__).absolute().parent
_REPO_ROOT = _PROJECT_ROOT.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))



def _default_api_user() -> str:
    """Read the configured Argo provider account."""
    return os.environ.get("ARGO_API_USER", "").strip()


def _analysis_output_dir() -> _Path:
    return _REPO_ROOT / "_output" / "Analysis"


def _analysis_transcript_dir() -> _Path:
    return _analysis_output_dir() / "transcripts"


def is_claude_model(model_name: str) -> bool:
    return "claude" in model_name.lower()


@dataclass
class AnalysisAgentConfig(EngineConfig):
    """Complete configuration for the ThermoML Analysis Agent.

    Inherits from ``EngineConfig`` (shared engine shape).
    Step 1 fields supply defaults for the 26 engine-required fields.
    Step 2 fields are analysis-agent-only (fitting, verdict, sessions).

    Key differences from the Query agent:
      - MAX_TOOL_ITERATIONS = 30 (vs 25) — fitting pipelines need more calls
      - MAX_TURN_SECONDS = 1200 (vs 1000) — 20-min budget for multi-phase analysis
      - VERDICT_MAX_TOKENS = 2000 (vs 500) — richer review of fit quality
      - Separate L1_DATA / L1_FIT iteration caps
      - PROTECT_TOOLS set — fitting tools exempt from compaction
      - OUTPUT_DIR / TRANSCRIPT_DIR — filesystem artifact paths
    """

    # ═══════════════════════════════════════════════════════════
    # Step 1 — Engine config fields (consumed by argo_engine_helpers)
    # ═══════════════════════════════════════════════════════════

    # ── API connection ──────────────────────────────────────
    API_URL: str = field(default_factory=lambda: os.environ.get(
        "ARGO_API_URL", "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"))  # Argo REST chat endpoint
    API_USER: str = field(default_factory=_default_api_user)  # Argo provider account
    HEADERS: dict = field(default_factory=lambda: {"Content-Type": "application/json"})  # HTTP headers for all Argo calls

    # ── Models ───────────────────────────────────────────────
    MODEL: str = "claudeopus46"           # L0 orchestrator model
    VERDICT_MODEL: str = "claudeopus46"   # post-job quality reviewer
    PLANNER_MODEL: str = "claudeopus46"   # strategy planner (react_loop guidance)

    # ── Generation parameters ───────────────────────────────
    TEMPERATURE: float = 0.3    # sampling temperature for primary MODEL
    TOP_P: float = 0.9          # nucleus sampling (omitted for Claude models)
    MAX_TOKENS: int = 6000      # max tokens per LLM response
    HTTP_TIMEOUT: int = 600     # seconds before HTTP timeout on Argo calls

    # ── Agent loop limits (L0) ──────────────────────────────
    MAX_TOOL_ITERATIONS: int = 30    # hard cap on L0 tool-call iterations (higher than query agent)
    MAX_TURN_SECONDS: int = 1200     # L0 wall-clock budget (20 min; multi-phase pipeline)

    # ── L2 async limits ─────────────────────────────────────
    L2_MAX_ITERATIONS: int = 6   # iteration cap per L2 evaluator (if used)
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
    VERDICT_MAX_TOKENS: int = 2000     # generous budget for detailed fit-quality review

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
    # Ledger abstraction for EVERY transition pair reachable in an
    # ANALYSIS session tree ("full" | "summary" | "none").  ROOT
    # PRECEDENCE: governs standalone analysis runs down to L1/L2 and
    # nested query children; under a main root, main's map wins instead.
    # context = parent-loop tool result; memory = working-memory digest
    # of the same envelope (pairs with a WM digest surface only).
    SUBAGENT_LEDGER_LEVELS: dict = field(default_factory=lambda: {
        "context": {
            "Ai/Qi": "full",   # nested run_query_agent returns
            "Ai/L1": "full",   # query_thermoml / align_compositions relays
            "Qi/L1": "full",   # L1_query relays inside nested query children
            "L1/L2": "full",   # L2 leaf evaluators inside L1 workers
        },
        "memory": {
            "Ai/L1": "summary",  # L1 relay digests stored in analysis WM
            "Qi/L1": "summary",  # L1AutoSaver digests in nested query WM
        },
    })

    # ═══════════════════════════════════════════════════════════
    # Step 2 — Agent-specific fields (NOT in EngineConfig base)
    # ═══════════════════════════════════════════════════════════

    # ── Additional model alias ──────────────────────────────
    L1_MODEL: str = "claudeopus46"   # model for L1 data-retrieval and fitting workers

    # ── Generation extras ───────────────────────────────────
    REASONING_CAP: int = 10000         # char cap on extended-thinking / reasoning blocks
    SUBAGENT_MAX_TOKENS: int = 500     # token budget for tool-result KEEP/DISCARD sub-agent

    # ── L1 / verdict loop limits ────────────────────────────
    L1_DATA_MAX_ITERATIONS: int = 12   # iteration cap for L1 data-retrieval phase
    L1_DATA_MAX_SECONDS: int = 360     # wall-clock budget for data-retrieval L1 (6 min)
    L1_FIT_MAX_ITERATIONS: int = 15    # iteration cap for L1 fitting phase (needs more calls)
    L1_FIT_MAX_SECONDS: int = 300      # wall-clock budget for fitting L1 (5 min)
    L1_COMP_MAX_ITERATIONS: int = 16   # iteration cap for composition-alignment L1 (fits ≥2 candidates + cross-validation)
    L1_COMP_MAX_SECONDS: int = 480     # wall-clock budget for composition-alignment L1 (8 min)
    VERDICT_MAX_SECONDS: int = 60      # time limit for the post-job verdict call

    # ── LLM-driven compaction ───────────────────────────────
    KEEP_RECENT_RESULTS: int = 2             # number of recent tool results exempt from trimming
    MIN_COMPRESS_CHARS: int = 500            # results shorter than this skip compression
    COMPRESS_SHORT_THRESHOLD: int = 1_500    # short-result threshold for abbreviated compress
    MAX_RETRY: int = 3                       # max SELECT→COMPRESS→VALIDATE cycles (fewer than query agent)
    MAX_IMMEDIATE_RETRY: int = 2             # max immediate retries on VALIDATE=RETRY
    COMPRESS_MAX_WORDS: int = 500            # word cap in compaction sub-agent prompt
    COMPRESS_SUMMARY_CHARS: int = 800        # summary preview chars sent to compactor
    COMPACTION_INTERVAL: int = 9999          # > MAX_TOOL_ITERATIONS — turn-based compaction disabled; size trigger still active

    # ── Stage compaction (deterministic) ────────────────────
    STAGE_COMPACT_BUDGET: int = 4_000    # char budget for deterministic stage-level compaction
    STAGE_NOTE_CHARS: int = 200          # max chars per stage-note annotation

    # ── Verdict extras ──────────────────────────────────────
    VERDICT_SOLVER_CHARS: int = 3_000          # max chars of solver trace included in verdict prompt
    VERDICT_ANSWER_CHARS: int = 4_000          # max chars of agent answer included in verdict prompt
    VERDICT_MAX_WORDS: int = 150               # hard word limit imposed on verdict response
    VERDICT_FORMAT_MAX_TOKENS: int = 800       # token budget for verdict-formatting LLM call

    # ── Tool-level subagent limits ──────────────────────────
    SUBAGENT_CHAR_LIMIT: int = 40_000      # max raw chars passed to tool sub-agent

    # ── Protected tools ─────────────────────────────────────
    PROTECT_TOOLS: set[str] = field(       # tool names exempt from LLM compaction (deterministic output)
        default_factory=lambda: {
            "fit_block",
            "fit_multi_system",
            "predict_from_rk",
            "compute_ideal_baseline",
        }
    )

    # ── Artifact directories ────────────────────────────────
    OUTPUT_DIR: _Path = field(default_factory=_analysis_output_dir)       # base dir for session run artifacts
    TRANSCRIPT_DIR: _Path = field(default_factory=_analysis_transcript_dir)  # full conversation transcripts


AGENT_CONFIG = AnalysisAgentConfig()

# ── Fitting-specific parameters ──────────────────────────────
RK_MAX_ORDER         = 5             # maximum Redlich-Kister polynomial order
RK_MIN_DATAPOINTS    = 5             # minimum data points (excl. pure) for fitting
IDEAL_BASELINE_PROPS = [             # properties with ideal-solution baselines
    "molar_volume",
    "density",
    "viscosity",
    "refractive_index",
    "speed_of_sound",
]
ARRHENIUS_PROPS = [                  # properties fitted in log space (Arrhenius)
    "viscosity",
]


