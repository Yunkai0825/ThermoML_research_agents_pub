"""L0 Orchestrator — loads the workflow and wires tools to agents.

This module is the main entry point for the ThermoML analysis agent.
It parses the L0 workflow skill markdown, builds the tool registry,
and drives the ReAct agent loop via ``agent_turn()``.

Public API
----------
run(question, run_verdict=True) -> AnalysisRunResult
    Synchronous entry point — sets up everything and runs the full loop.

Architecture
------------
Uses the markdown workflow approach: agent behaviour is defined in
``L0_analysis_workflow.md`` and parsed by ``subworkflow_parser.py``.
The system prompt comes from the .md file, tool instructions are
auto-generated from the tool registry via ``build_tool_instructions()``.
"""
from __future__ import annotations

import functools
import json
import logging
import re
import time
import datetime as dt
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

log = logging.getLogger("ANALYSIS-L0")

from ....general_db_query_engine.general_text_context_marker_catalog import strip_reasoning as _strip_reasoning


# ── Configuration ────────────────────────────────────────────
from ...ThermoML_analysis_argo_config import AGENT_CONFIG as cfg

# ── Workflow parser ──────────────────────────────────────────
from .._subworkflow_md_parser.subworkflow_parser import parse_workflow, render_prompt

# ── Tool description injection (shared module) ──────────────
from ....general_db_query_engine.general_subagent_skill_schema_and_parser.subworkflow_md_tool_descriptions import (
    build_tool_instructions,
)

# ── Reuse the shared ReAct engine ─────────────────────────────
from ....general_db_query_engine.general_argo_engine_helpers import (
    agent_turn, AgentTurnResult, anchor, with_engine_config,
)

# ── Analysis client ──────────────────────────────────────────
from ...analysis_agent_argo_engine.argo_client import AnalysisClient

# ── Analysis-specific tools + ToolResult ─────────────────────
from ...analysis_agent_toolbox.tool_catalog import (
    ANALYSIS_CATALOG, COMPACTOR_CATALOG, ToolResult,
)
from ....general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog import ToolEntry
from ....general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog.health_check_helper import (
    run_health_check,
)
from ....general_db_query_engine.general_hooks_management_helpers.general_context_hooks import _context_hooks_anchors_catalog as ctx_anchor
from ....general_db_query_engine.general_hooks_management_helpers.general_memory_management_tools_hooks_helpers import _memory_hooks_anchors_catalog as mem_anchor
from ....general_db_query_engine.general_hooks_management_helpers.general_memory_management_tools_hooks_helpers.memory_tool_result_instrumentation_helper import make_instrumented_wrapper
from ....general_db_query_engine.general_hooks_management_helpers.general_postans_eval_hooks import (
    evaluate_and_assemble_return,
    prepare_answer_only_system_prompt,
)
from ....general_db_query_engine.general_data_grounding_gate import (
    core_claims_source_suffix,
    postanswer_checker_for_tools,
    export_inspections,
    harvest_inspections,
)
from ....NIST_ThermoML_query_agent.query_agent_workflows.L0_orchestrator.sources_block_validation import (
    validate_and_enrich_l0_sources,
)

# ── Analysis-specific compactor (protects fitting results) ───
from ...analysis_agent_context_hooks.hook_catalog import AnalysisInteractiveCompactor

# ── Session management ───────────────────────────────────────
from ...analysis_agent_context_hooks.hook_catalog import session_manager as _sess_mgr
from ...analysis_agent_context_hooks.hook_catalog import reopen_session as _reopen_session
from ....general_db_query_engine.general_hooks_management_helpers.general_memory_management_tools_hooks_helpers.session_manager_output_storage import (
    _filesystem_path,
    ensure_directory,
)
from ....general_db_query_engine.general_hooks_management_helpers.general_tracking_hooks_output_helpers.output_writers import (
    prepare_result_answer as _prepare_result_answer,
    split_answer_ledger as _split_answer_ledger,
    write_result_bundle as _write_result_bundle,
)

# ── Working memory ───────────────────────────────────────────
from ...analysis_agent_context_hooks.hook_catalog import AnalysisWorkingMemory

# ── Verdict agent ────────────────────────────────────────────
from ...analysis_agent_context_hooks.hook_catalog import (
    AnalysisAgentHooks,
    build_agent_hooks,
)

_HERE = Path(__file__).resolve().parent
_WORKFLOW_PATH = _HERE / "L0_analysis_workflow.md"
_COMPACTION_INTERVAL = cfg.COMPACTION_INTERVAL

# Module-level compactor instance used by the orchestrator loop
_analysis_compactor = AnalysisInteractiveCompactor()


# ═══════════════════════════════════════════════════════════════
#  Setup helpers
# ═══════════════════════════════════════════════════════════════

def load_workflow() -> dict[str, Any]:
    """Parse and return the L0 analysis workflow definition."""
    return parse_workflow(_WORKFLOW_PATH)


def build_system_prompt(workflow: dict[str, Any]) -> str:
    """Extract the system prompt from the parsed workflow."""
    return workflow["system_prompt"]


# ═══════════════════════════════════════════════════════════════
#  Session catalog tool (callable by the agent)
# ═══════════════════════════════════════════════════════════════

def list_session_files() -> dict:
    """Return a manifest of all output files generated in this session."""
    sess = _sess_mgr.get_session()
    if not sess:
        return {"files": [], "note": "No active session."}
    files = sess.list_files()
    return {
        "session_dir": str(sess.session_dir),
        "n_files": len(files),
        "files": files,
    }


# ═══════════════════════════════════════════════════════════════
#  AnalysisRunResult
# ═══════════════════════════════════════════════════════════════

_FIT_RESULT_CAP = 24


def _project_fit_results(fit_snapshots: list[dict]) -> list[dict]:
    """Deterministic envelope ``fit_results`` from wm fit snapshots.

    Snapshots are built by ``_fit_snapshot`` during the run (ID-validated
    at record time); this projection only renames/selects fields — no LLM
    ever authors fit numbers.
    """
    projected: list[dict] = []
    for snapshot in fit_snapshots[:_FIT_RESULT_CAP]:
        entry = {
            "doi": snapshot["doi"],
            "lit_num_id": snapshot["lit_num_id"],
            "block_number": snapshot["block_number"],
            "BLKsubsys_id": snapshot["BLKsubsys_id"],
            "property": snapshot.get("label")
            or (snapshot.get("property_response") or {}).get("prop_num_id"),
            "rk_order": snapshot["rk_order"],
            "rk_coeffs": snapshot["rk_coeffs"],
            "r_squared": snapshot["r_squared"],
            "rmse": snapshot["rmse"],
            "n_points": snapshot["n_mixture_points"],
            "temperature_K": snapshot.get("selected_temperature_K",
                                          snapshot.get("sweep_value")),
            "mixing_rule": snapshot.get("mixing_rule"),
        }
        projected.append(entry)
    return projected


@dataclass
class AnalysisRunResult:
    """Structured result from a single analysis run.

    The test runners convert this dataclass to plain dicts when needed.
    """
    answer: str
    verdict: str | None
    iterations: int
    elapsed_seconds: float
    tool_history: list[dict] = field(default_factory=list)
    timed_out: bool = False
    session_dir: str | None = None
    output_files: list[dict] = field(default_factory=list)
    final_context: str = ""
    # ── Continuation state (not serialised to result.md) ─────
    round_number: int = 1
    working_memory: Any = field(default=None, repr=False)


# ═══════════════════════════════════════════════════════════════
#  Tool registry
# ═══════════════════════════════════════════════════════════════

def _build_tool_registry() -> Dict[str, Callable]:
    """Build the L0 tool registry from ANALYSIS_CATALOG + session tool."""
    # Register the session tool in the catalog so the health check covers it
    if "list_session_files" not in ANALYSIS_CATALOG:
        ANALYSIS_CATALOG.register(ToolEntry(
            "list_session_files", list_session_files,
            group="session", skip_compactor=True, skip_subagent=True,
        ))
    tools: Dict[str, Callable] = ANALYSIS_CATALOG.wrapped_tools(
        expose_internal_results=True
    )
    # ── Pre-flight health check ──────────────────────────────
    run_health_check(
        ANALYSIS_CATALOG,
        compactor_catalog=COMPACTOR_CATALOG,
        actual_tools=tools,
        label="analysis-L0",
    )
    return tools


# ═══════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════

@with_engine_config(cfg)
def run(
    question: str,
    *,
    run_verdict: bool = True,
    # ── Continuation parameters ────────────────────────────
    round_number: int = 1,
    session_dir: Path | str | None = None,
    prev_context: str = "",
    prev_working_memory: "AnalysisWorkingMemory | None" = None,
    hooks: AnalysisAgentHooks | None = None,
) -> AnalysisRunResult:
    """Run the analysis agent on a single question.

    Parameters
    ----------
    round_number : int
        1-based round index inside a multi-round session.
    session_dir : Path | str | None
        If provided, reopen this existing session directory instead of
        creating a new one.  Enables multi-round conversations.
    prev_context : str
        Cleaned full context from the previous round (system prompt and
        working memory already stripped).
    prev_working_memory : AnalysisWorkingMemory | None
        Working memory object carrying entity catalog and history from
        previous rounds.

    Returns
    -------
    AnalysisRunResult
        Supports both attribute and dict-style access.
    """
    # ── Session setup ────────────────────────────────────────
    if session_dir is not None:
        sess = _reopen_session(Path(session_dir))
    else:
        sess = _sess_mgr.init_session(cfg.OUTPUT_DIR, question)
    log.info("Session dir: %s  (round %d)", sess.session_dir, round_number)

    # Round-aware file suffix
    _R = f"_R{round_number}" if round_number > 1 else ""

    _history_path = Path(sess.session_dir) / f"run_history{_R}.md"
    _stats_path = Path(sess.session_dir) / f"reference_stats{_R}.md"
    _reasoning_path = Path(sess.session_dir) / f"reasoning_tokens_stripped{_R}.md"

    # ── Load workflow from markdown ──────────────────────────
    workflow = load_workflow()
    system_prompt = build_system_prompt(workflow)

    client = AnalysisClient.for_l0()

    working_mem = prev_working_memory if prev_working_memory is not None else AnalysisWorkingMemory()

    agent_hooks = hooks or build_agent_hooks(
        working_memory_loader=working_mem.render,
        compactor=_analysis_compactor.compact,
        batch_validator=ANALYSIS_CATALOG.validate_batch,
    )

    memory: list[dict] = []
    if prev_context:
        _prior_context_block = (
            "## Prior Conversation Context\n"
            "Below is the full conversation history from the previous round. "
            "Continue from where it left off.\n\n"
            + prev_context
        )
        _prior_context_block = anchor(
            mem_anchor.SYNC_PRIOR_CONTEXT_APPEND,
            agent_hooks.engine_hooks,
            content=_prior_context_block,
            round_number=round_number,
        ) or _prior_context_block
        memory.append({"role": "user", "content": _prior_context_block})

    # Wrap tools to auto-record results into working memory.
    tools = _build_tool_registry()
    instrumented_tools = {
        name: make_instrumented_wrapper(
            name, fn,
            working_mem=working_mem,
            agent_hooks=agent_hooks,
            anchor_fn=anchor,
            mem_anchor_record=mem_anchor.SYNC_WORKING_MEMORY_RECORD,
        )
        for name, fn in tools.items()
    }

    # ── Inject tool-calling instructions into system prompt ──
    system_prompt = prepare_answer_only_system_prompt(
        system_prompt + "\n\n" + build_tool_instructions(instrumented_tools)
    )
    system_prompt = anchor(
        ctx_anchor.SYNC_SYSTEM_PROMPT_READY,
        agent_hooks.engine_hooks,
        system_prompt=system_prompt,
        tool_names=sorted(instrumented_tools.keys()),
        workflow=workflow,
    ) or system_prompt
    anchor(
        agent_hooks.anchors.start_tracking,
        agent_hooks.engine_hooks,
        agent="analysis-agent",
        prompt=question,
        history_out_path=_history_path,
        stats_out_path=_stats_path,
        reasoning_out_path=_reasoning_path,
    )

    log.info("=" * 60)
    log.info("ANALYSIS AGENT START")
    log.info("Question: %s", question[:200])
    log.info("=" * 60)

    result: AgentTurnResult = agent_turn(
        question,
        system_prompt=system_prompt,
        tools=instrumented_tools,
        memory=memory,
        client=client,
        max_iterations=cfg.MAX_TOOL_ITERATIONS,
        timeout=cfg.MAX_TURN_SECONDS,
        required_tools={"fit_block", "fit_block_derived", "fit_multi_system"},
        compaction_interval=_COMPACTION_INTERVAL,
        hooks=agent_hooks.engine_hooks,
    )

    log.info(
        "ANALYSIS AGENT DONE: %d iterations, %.1fs, %d tools called",
        result.iterations, result.elapsed_seconds, len(result.tool_history),
    )

    postanswer = evaluate_and_assemble_return(
        agent_answer=result.answer,
        postassembly_checker=postanswer_checker_for_tools(instrumented_tools),
        core_claims_system_suffix=core_claims_source_suffix(instrumented_tools),
        response_json_schema=workflow["response_json_schema"],
        client=client,
        label="analysis-L0",
        tool_history=result.tool_history,
        task_context=question,
        hooks=agent_hooks,
        id_alignment_processor=validate_and_enrich_l0_sources,
        deterministic_fields={
            "data_inspections": export_inspections(
                harvest_inspections(result.tool_history)
            ),
            "fit_results": _project_fit_results(working_mem.fit_results),
        },
    )
    result.answer = postanswer.to_json()

    # Verdict
    verdict = None
    if run_verdict and not result.timed_out:
        log.info("Running verdict agent...")
        # The ledger is machine-merged — reviewing it as "answer" text only
        # inflates/truncates the verdict input; rows stay in the trace.
        _verdict_answer, _ = _split_answer_ledger(result.answer)
        verdict = anchor(
            agent_hooks.anchors.run_verdict,
            agent_hooks.engine_hooks,
            question, _verdict_answer, result.tool_history, client=client,
        )
        log.info("Verdict: %s", verdict[:200] if verdict else "(none)")

    run_result = AnalysisRunResult(
        answer=_strip_reasoning(result.answer),
        verdict=verdict,
        iterations=result.iterations,
        elapsed_seconds=result.elapsed_seconds,
        tool_history=result.tool_history,
        timed_out=result.timed_out,
        session_dir=str(sess.session_dir),
        output_files=sess.list_files(),
        final_context=result.final_context,
        round_number=round_number,
        working_memory=working_mem,
    )

    _save_run_output(question, run_result)

    # Write raw final LLM context via shared postjob hook
    _final_ctx = result.final_context
    anchor(
        agent_hooks.anchors.save_final_context,
        agent_hooks.engine_hooks,
        session_dir=sess.session_dir, final_context=_final_ctx,
        filename=f"final_full_context{_R}.md",
    )

    anchor(
        agent_hooks.anchors.finalize_tracking,
        agent_hooks.engine_hooks,
        status="TIMEOUT" if run_result.timed_out else "OK",
        stats_path=_stats_path,
    )

    # Post-run workflow Sankey artifacts (site debug mode /
    # all completed root sessions; skips saved child runs of a
    # bigger session; never fails the run)
    from ....general_db_query_engine.general_posteval_helpers.compaction_funnels.api import (
        maybe_generate_workflow)
    maybe_generate_workflow(sess.session_dir, kind="analysis")

    return run_result


# ═══════════════════════════════════════════════════════════════
#  Output persistence
# ═══════════════════════════════════════════════════════════════

def _save_run_output(question: str, result: AnalysisRunResult) -> Optional[Path]:
    """Write result + history + manifest to the session directory."""
    if not result.session_dir:
        raise ValueError("AnalysisRunResult.session_dir is required for durable output")
    out_dir = Path(result.session_dir)
    if not _filesystem_path(out_dir).exists():
        raise FileNotFoundError(f"analysis session directory does not exist: {out_dir}")
    ensure_directory(out_dir)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _R = f"_R{result.round_number}" if result.round_number > 1 else ""

    # Result file
    _result_path = out_dir / f"result{_R}.md"
    _display_answer, _structured_sections = _prepare_result_answer(
        _result_path, result.answer)
    result_lines = [
        f"# Analysis Run — {ts} (Round {result.round_number})",
        f"",
        f"**Question:** {question}",
        f"",
        f"**Time:** {result.elapsed_seconds:.1f}s | "
        f"**Iterations:** {result.iterations} | "
        f"**Tools:** {len(result.tool_history)}",
        f"",
        f"---",
        f"",
        f"## Answer",
        f"",
        _display_answer,
    ]
    result_lines += _structured_sections
    if result.verdict:
        result_lines += ["", "---", "", "## Verdict", "", result.verdict]

    output_files = result.output_files or []
    if output_files:
        result_lines += ["", "---", "", "## Output Files", ""]
        for f in output_files:
            desc = f" — {f['description']}" if f.get("description") else ""
            result_lines.append(f"- **{f.get('category', 'file')}**: `{f['path']}`{desc}")

    _write_result_bundle(
        _result_path, "\n".join(result_lines), result.answer)

    # History file — written by the shared history_recorder (run_history.md)
    # (no longer generated here; the real-time recorder produces richer output)

    log.info("Output saved to %s", out_dir)
    return out_dir
