"""L0 Orchestrator for the canonical ID alignment agent.

Loads the workflow markdown, builds the tool registry, injects
tool descriptions, and drives the ReAct loop via ``agent_turn()``.

Public API
----------
run(search_text, settings=None) -> AgentTurnResult
    Synchronous entry point — parse → resolve → validate → fill → finalize.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ── Ensure workspace root is on sys.path ────────────────────
from ThermoML_results_browser import _RESEARCH_ROOT
_WS_ROOT = str(_RESEARCH_ROOT)
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

# ── Engine helpers ──────────────────────────────────────────
from NIST_ThermoML_agents.general_db_query_engine.general_argo_engine_helpers import (
    agent_turn,
    AgentTurnResult,
    with_engine_config,
    AgentAnchorCollection,
    anchor,
    bind_hook,
    build_agent_lifecycle_bindings,
    compile_hooks,
)
from NIST_ThermoML_agents.general_db_query_engine.general_subagent_skill_schema_and_parser.subworkflow_md_tool_descriptions import (
    build_tool_instructions,
)

from NIST_ThermoML_agents.general_db_query_engine.general_hooks_management_helpers.general_context_hooks import (
    _context_hooks_anchors_catalog as ctx_anchor,
)
from NIST_ThermoML_agents.general_db_query_engine.general_hooks_management_helpers.general_context_hooks.history_tracking_hooks import HistoryRecorder
from NIST_ThermoML_agents.general_db_query_engine.general_hooks_management_helpers.general_context_hooks.stats_references_tracking_hooks import StatsRecorder
from NIST_ThermoML_agents.general_db_query_engine.general_hooks_management_helpers.general_context_hooks.reasoning_token_tracking_hooks import ReasoningTokenTracker
from NIST_ThermoML_agents.general_db_query_engine.general_tool_management_helpers import (
    _tool_hooks_anchors_catalog as tool_anchor,
)

# ── Alignment-agent specifics ──────────────────────────────
from ...alignment_agent_argo_config import AGENT_CONFIG
from ...alignment_agent_argo_engine.argo_client import AlignmentClient
from ...alignment_agent_toolbox.tool_catalog import L0_CATALOG
from ...alignment_agent_toolbox.mcp_tools import get_fields

log = logging.getLogger("Alignment-L0")

_HERE = Path(__file__).resolve().parent
_WORKFLOW_PATH = _HERE / "L0_orchestrator_workflow.md"
_ALIGNMENT_OUTPUT_ROOT = _RESEARCH_ROOT.parent / "_output" / "Alignment"


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _load_workflow_prompt() -> str:
    """Read the system prompt from the workflow markdown.

    Requires and extracts exactly one ``<system_prompt>`` block.
    """
    text = _WORKFLOW_PATH.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError("Alignment workflow requires YAML frontmatter")
    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError("Alignment workflow has malformed YAML frontmatter")
    body = parts[2]
    open_tag = "<system_prompt>"
    close_tag = "</system_prompt>"
    if body.count(open_tag) != 1 or body.count(close_tag) != 1:
        raise ValueError("Alignment workflow requires exactly one <system_prompt> block")
    prompt = body.split(open_tag, 1)[1].split(close_tag, 1)[0].strip()
    if not prompt:
        raise ValueError("Alignment workflow system prompt is empty")
    return prompt


# ═══════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════

@with_engine_config(AGENT_CONFIG)
def run(
    search_text: str,
    settings: Optional[Dict[str, Any]] = None,
) -> AgentTurnResult:
    """Run the alignment agent on a free-text search input.

    Parameters
    ----------
    search_text : str
        Raw user search text (e.g. "ethanol + viscosity ; water + density").
    settings : dict, optional
        Per-request entry limit, time budget, escalation and field-fill toggles.

    Returns
    -------
    AgentTurnResult
        Contains answer, iterations, elapsed time, tool history.
        The resolved fields are also available via ``get_fields().snapshot()``.
    """
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise TypeError("settings must be an object")

    # Per-run time budget.  Never mutate the module-level config singleton:
    # concurrent or later runs must retain the configured default.
    max_wait = settings["max_wait_seconds"] if "max_wait_seconds" in settings else None
    if max_wait is not None:
        if isinstance(max_wait, bool) or not isinstance(max_wait, int) or max_wait < 1:
            raise TypeError("settings.max_wait_seconds must be a positive integer")

    allow_qa = settings["allow_query_agent"] if "allow_query_agent" in settings else False
    agentic_all = settings["agentic_all_fields"] if "agentic_all_fields" in settings else False
    if not isinstance(allow_qa, bool) or not isinstance(agentic_all, bool):
        raise TypeError("allow_query_agent and agentic_all_fields must be booleans")

    # Reset shared field state for this run
    fields = get_fields()
    fields.reset()

    # Build LLM client
    client = AlignmentClient.for_l0()

    # Build tool registry
    tools: Dict[str, Any] = dict(L0_CATALOG.tools)
    if not allow_qa:
        tools.pop("escalate_to_query_agent", None)

    # Build system prompt
    system_prompt = _load_workflow_prompt()
    system_prompt += "\n\n" + build_tool_instructions(tools)

    # Compose the user question
    entry_limit = settings["entry_limit"] if "entry_limit" in settings else 20
    if isinstance(entry_limit, bool) or not isinstance(entry_limit, int) or entry_limit < 1:
        raise TypeError("settings.entry_limit must be a positive integer")
    limit_note = (
        f"\n\nIMPORTANT: The user wants up to {entry_limit} entries. "
        f"When calling resolve_entity or search_titles, use limit={entry_limit}."
    )
    user_message = (
        f"Please resolve the following search input and fill all search "
        f"form fields:\n\n{search_text}{limit_note}"
    )
    if max_wait is not None:
        user_message += (
            f"\n\nTIME BUDGET: You have {int(max_wait)} seconds. "
            "Be efficient — focus on the most impactful resolutions first."
        )

    # Escalation & block-filler settings
    if not allow_qa:
        user_message += (
            "\n\nESCALATION DISABLED: Do NOT call escalate_to_query_agent. "
            "If you cannot resolve an entity, mark it as unresolved and continue."
        )
    if agentic_all:
        user_message += (
            "\n\nAGENTIC ALL FIELDS: The user wants you to resolve and fill "
            "ALL filter blocks (bibliography, chemistry, properties). "
            "Use fill_bibliography_block, fill_chemistry_block, and "
            "fill_properties_block to fill each section systematically."
        )

    # Each run owns its telemetry. Shared lifecycle hooks restore any parent
    # recorders after completion or failure, including nested agent calls.
    _ALIGNMENT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    session_dir = Path(tempfile.mkdtemp(
        prefix=f"run_{datetime.now():%Y%m%d_%H%M%S}_",
        dir=_ALIGNMENT_OUTPUT_ROOT,
    ))
    history, stats, reasoning = HistoryRecorder(), StatsRecorder(), ReasoningTokenTracker()
    anchors = AgentAnchorCollection()
    bindings = list(build_agent_lifecycle_bindings(
        history=history, stats=stats, reasoning=reasoning, anchors=anchors,
    ))
    bindings.extend((
        bind_hook(reasoning.log_reasoning_from_response,
                  ctx_anchor.SYNC_LLM_RESPONSE_AFTER.anchor_type,
                  ctx_anchor.SYNC_COMPACTION_GUIDANCE_RESPONSE.anchor_type),
        bind_hook(stats.log_tool_result, tool_anchor.SYNC_TOOL_RESULT_RECORDED.anchor_type),
        bind_hook(history.log_tool_call, tool_anchor.SYNC_TOOL_CALL_RECORDED.anchor_type),
        bind_hook(history.log_error, tool_anchor.SYNC_TOOL_ERROR_RECORDED.anchor_type),
        bind_hook(history.log_purpose_tasks_error,
                  tool_anchor.SYNC_TOOL_PURPOSE_ERROR_RECORDED.anchor_type),
        bind_hook(history.log_stage_compaction,
                  tool_anchor.SYNC_STAGE_COMPACTION_RECORDED.anchor_type),
        bind_hook(history.log_compaction,
                  ctx_anchor.SYNC_COMPACTION_RECORDED.anchor_type,
                  ctx_anchor.SYNC_COMPACTION_SKIPPED.anchor_type),
        bind_hook(stats.log_compaction,
                  ctx_anchor.SYNC_COMPACTION_STATS_RECORDED.anchor_type,
                  ctx_anchor.SYNC_COMPACTION_SKIPPED.anchor_type),
    ))
    hooks = compile_hooks(bindings)
    status = "ERROR"
    try:
        anchor(anchors.start_tracking, hooks, agent="alignment-agent", prompt=search_text,
               history_out_path=session_dir / "run_history.md",
               stats_out_path=session_dir / "reference_stats.md",
               reasoning_out_path=session_dir / "reasoning_tokens_stripped.md")
        result = agent_turn(
            user_message,
            system_prompt=system_prompt,
            tools=tools,
            memory=[],
            client=client,
            timeout=max_wait,
            hooks=hooks,
        )
        result.session_dir = str(session_dir)
        status = "TIMEOUT" if result.timed_out else "OK"
    finally:
        anchor(anchors.finalize_tracking, hooks, status=status,
               stats_path=session_dir / "reference_stats.md")

    log.info(
        "Alignment L0 done: %d iterations, %.1fs, timed_out=%s",
        result.iterations, result.elapsed_seconds, result.timed_out,
    )

    return result
