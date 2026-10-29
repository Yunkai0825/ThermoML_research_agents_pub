"""L0 Orchestrator — loads the workflow and wires tools to agents.

This module is the main entry point for the ThermoML query agent system.
It parses the L0 workflow skill markdown, initialises memory, builds the
tool registry, and drives the ReAct agent loop via ``agent_turn()``.

Public API
----------
run(user_question, memory_path=None, extra_tools=None, session_dir=None) -> AgentTurnResult
    Synchronous entry point — sets up everything and runs the full loop.

hardcoded_L0_wf_run(user_question, tool_executor, memory_path) -> dict
    Async entry point — executes the workflow phases in fixed order
    with optional parallel fan-out (no LLM-driven decisions).

Architecture
------------
Uses the ``QueryClient(ArgoClient)`` tier-based factory pattern.
LLM call configuration is centralised in ``ThermoML_query_argo_config``
and immutable per layer.

The tool registry maps tool names to Python callables:
  - memory_* tools  → memory_management_MCP_tools (direct Python calls)
  - L1_query        → launches a child agent_turn with L1 workflow+tools
"""

from __future__ import annotations

from functools import partial
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict

# Workflow parser + async runner
from .._subworkflow_md_parser.subworkflow_parser import parse_workflow, render_prompt
from ....general_db_query_engine.general_argo_engine_helpers.async_runner import (
    ToolExecutor,
    run_workflow,
)

# ReAct loop + result type from shared engine
from ....general_db_query_engine.general_argo_engine_helpers import (
    agent_turn,
    AgentTurnResult,
    anchor,
    with_engine_config,
)
from ....general_db_query_engine.general_argo_engine_helpers.engine_react_helpers.react_loop import (
    _apply_forced_final_grounding_gate,
)
# Query-agent ArgoClient with for_l0, for_l1, for_l2 factories
from ...query_agent_argo_engine.argo_client import QueryClient as ArgoClient
from ...ThermoML_query_argo_config import AGENT_CONFIG as _cfg

# Tool description injection
from ....general_db_query_engine.general_subagent_skill_schema_and_parser.subworkflow_md_tool_descriptions import build_tool_instructions

# L1 dispatchers
from ..L1_workers.l1_query_dispatcher import dispatch_l1_query, _get_catalog as _get_l1_catalog

# Memory tools
from ....general_db_query_engine.general_hooks_management_helpers.general_memory_management_tools_hooks_helpers import memory_management_MCP_tools as mem_tools
from ....general_db_query_engine.general_hooks_management_helpers.general_memory_management_tools_hooks_helpers.session_manager_output_storage import ensure_directory

# Query-agent tool catalog (single entry point)
from ...query_agent_toolbox.tool_catalog import L0_CATALOG
from ....general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog import ToolEntry
from ....general_db_query_engine.general_tool_management_helpers.tool_capabilities import bind_reachable_tools
from ....general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog.health_check_helper import (
    run_health_check,
)
from ....general_db_query_engine.general_hooks_management_helpers.general_context_hooks import _context_hooks_anchors_catalog as ctx_anchor
from ....general_db_query_engine.general_hooks_management_helpers.general_memory_management_tools_hooks_helpers import _memory_hooks_anchors_catalog as mem_anchor
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
from ....general_db_query_engine.general_hooks_management_helpers.general_tracking_hooks_output_helpers.output_writers import (
    split_answer_ledger as _split_answer_ledger,
)
from .sources_block_validation import validate_and_enrich_l0_sources

# Context hooks (query-agent specific)
from ...query_agent_context_hooks.hook_catalog import (
    QueryAgentHooks,
    L1AutoSaver,
    is_thin_answer,
    force_data_presentation,
    build_agent_hooks,
)

log = logging.getLogger("L0-Orchestrator")

_HERE = Path(__file__).resolve().parent
_WORKFLOW_PATH = _HERE / "L0_orchestrator_workflow.md"


# ═══════════════════════════════════════════════════════════════
#  Setup helpers
# ═══════════════════════════════════════════════════════════════

def load_workflow() -> dict[str, Any]:
    """Parse and return the L0 orchestrator workflow definition."""
    return parse_workflow(_WORKFLOW_PATH)


def init_memory(memory_path: str | Path) -> None:
    """Initialise the working-memory backend at an explicit session path."""
    mem_tools.init(Path(memory_path))


def build_system_prompt(workflow: dict[str, Any]) -> str:
    """Extract the system prompt from the parsed workflow."""
    return workflow["system_prompt"]


# ═══════════════════════════════════════════════════════════════
#  Tool registry
# ═══════════════════════════════════════════════════════════════

def _build_tool_registry() -> Dict[str, Callable]:
    """Build the L0 tool registry mapping names to callables.

    Memory tools are direct Python calls.
    L1 workers are stub functions that the outer harness should override
    or that launch child agent_turn loops when a real executor is wired.
    """
    return L0_CATALOG.wrapped_tools()


# ═══════════════════════════════════════════════════════════════
#  Main entry points
# ═══════════════════════════════════════════════════════════════

@with_engine_config(_cfg)
def run(
    user_question: str,
    memory_path: str | Path | None = None,
    extra_tools: Dict[str, Callable] | None = None,
    client: ArgoClient | None = None,
    *,
    run_verdict: bool = False,
    # ── Continuation parameters ────────────────────────────
    round_number: int = 1,
    session_dir: Path | str | None = None,
    prev_context: str = "",
    prev_working_memory: Any = None,             # unused (query uses file-based memory)
    hooks: QueryAgentHooks | None = None,
) -> AgentTurnResult:
    """Run the full L0 orchestration loop synchronously.

    Parameters
    ----------
    user_question : str
        The user's natural-language query.
    memory_path : str or Path, optional
        Explicit path to the working-memory file. If omitted, ``session_dir``
        is required and ``session_dir/working_memory.md`` is used.
    extra_tools : dict, optional
        Additional tool callables to merge into the registry.
    client : ArgoClient, optional
        LLM caller. Defaults to ArgoClient.for_l0().
    run_verdict : bool
        Whether to run the independent verdict agent after the loop
        (skipped on timeout). The verdict is stored on ``result.verdict``.
    round_number : int
        1-based round index inside a multi-round session.
    prev_context : str
        Cleaned full context from the previous round.

    Returns
    -------
    AgentTurnResult
        Contains answer, iterations, elapsed time, tool history, etc.
    """
    if memory_path is None:
        if session_dir is None:
            raise ValueError(
                "Query orchestration requires memory_path or session_dir; "
                "implicit shared working memory is not supported."
            )
        memory_path = Path(session_dir) / "working_memory.md"

    # Initialise
    init_memory(memory_path)
    workflow = load_workflow()
    system_prompt = build_system_prompt(workflow)

    if session_dir is not None:
        _out_dir = Path(session_dir)
        ensure_directory(_out_dir)
    else:
        _out_dir = Path(memory_path).parent
        ensure_directory(_out_dir)
    _stats_path = _out_dir / "reference_stats.md"
    _reasoning_path = _out_dir / "reasoning_tokens_stripped.md"
    _history_path = _out_dir / "run_history.md"

    if client is None:
        client = ArgoClient.for_l0()

    # Build tool registry
    tools = _build_tool_registry()

    # Working memory loader (injected at each iteration)
    _load_working_memory = mem_tools.memory_read
    _agent_hooks = hooks or build_agent_hooks(
        working_memory_loader=_load_working_memory,
        batch_validator=L0_CATALOG.validate_batch,
    )
    anchor(
        _agent_hooks.anchors.start_tracking,
        _agent_hooks.engine_hooks,
        agent="query-agent",
        prompt=user_question,
        history_out_path=_history_path,
        stats_out_path=_stats_path,
        reasoning_out_path=_reasoning_path,
    )

    # Wrap L1_query to auto-store results in working memory
    _l1_saver = L1AutoSaver(partial(dispatch_l1_query, hooks=_agent_hooks), mem_tools, hooks=_agent_hooks)
    bind_reachable_tools(_l1_saver, lambda: _get_l1_catalog().wrapped_tools())
    tools["L1_query"] = _l1_saver

    # Register L1_query in the catalog (if not yet) so the health check covers it
    if "L1_query" not in L0_CATALOG:
        L0_CATALOG.register(ToolEntry(
            "L1_query", _l1_saver,
            group="L1_dispatch", skip_compactor=True, skip_subagent=True,
        ))

    if extra_tools:
        tools.update(extra_tools)

    # ── Pre-flight health check ──────────────────────────────
    run_health_check(
        L0_CATALOG,
        actual_tools=tools,
        label="query-L0",
    )

    # Inject tool-calling instructions into system prompt
    system_prompt = prepare_answer_only_system_prompt(
        system_prompt + "\n\n" + build_tool_instructions(tools)
    )
    system_prompt = anchor(
        ctx_anchor.SYNC_SYSTEM_PROMPT_READY,
        _agent_hooks.engine_hooks,
        system_prompt=system_prompt,
        tool_names=sorted(tools.keys()),
        workflow=workflow,
    ) or system_prompt

    # Run the ReAct loop
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
            _agent_hooks.engine_hooks,
            content=_prior_context_block,
            round_number=round_number,
        ) or _prior_context_block
        memory.append({"role": "user", "content": _prior_context_block})
    log.info("L0 START (round %d)", round_number)
    result = agent_turn(
        user_question,
        system_prompt=system_prompt,
        tools=tools,
        memory=memory,
        client=client,
        hooks=_agent_hooks.engine_hooks,
    )

    # ── Post-loop answer quality check ──────────────────────
    result = _repair_thin_answer(
        result, _l1_saver.call_count, tools, mem_tools, memory, client,
        system_prompt,
    )

    postanswer = evaluate_and_assemble_return(
        agent_answer=result.answer,
        postassembly_checker=postanswer_checker_for_tools(tools),
        core_claims_system_suffix=core_claims_source_suffix(tools),
        response_json_schema=workflow["response_json_schema"],
        client=client,
        label="query-L0",
        tool_history=result.tool_history,
        task_context=user_question,
        hooks=_agent_hooks,
        id_alignment_processor=validate_and_enrich_l0_sources,
        deterministic_fields={
            "data_inspections": export_inspections(
                harvest_inspections(result.tool_history)
            ),
            "id_catalog_snapshot": json.loads(mem_tools.memory_catalog_list()),
        },
    )
    result.answer = postanswer.to_json()
    result.session_dir = str(_out_dir)
    result.round_number = round_number
    result.working_memory = mem_tools.memory_read()

    if run_verdict and not result.timed_out:
        log.info("Running verdict agent...")
        # The ledger is machine-merged — review the authored answer only.
        _verdict_answer, _ = _split_answer_ledger(result.answer)
        verdict = anchor(
            _agent_hooks.anchors.run_verdict,
            _agent_hooks.engine_hooks,
            user_question, _verdict_answer, result.tool_history,
        )
        result.verdict = verdict
        log.info("Verdict: %s", verdict[:200] if verdict else "(none)")

    log.info(
        "L0 completed: %d iterations, %.1fs, timed_out=%s",
        result.iterations, result.elapsed_seconds, result.timed_out,
    )

    # Write raw final LLM context via shared postjob hook
    _R = f"_R{round_number}" if round_number > 1 else ""
    anchor(
        _agent_hooks.anchors.save_final_context,
        _agent_hooks.engine_hooks,
        session_dir=_out_dir, final_context=result.final_context,
        filename=f"final_full_context{_R}.md",
    )

    # Tracking is part of the run contract, not best-effort side logging.
    anchor(
        _agent_hooks.anchors.finalize_tracking,
        _agent_hooks.engine_hooks,
        status="TIMEOUT" if result.timed_out else "OK",
        stats_path=_stats_path,
        flush_history=True,
    )

    # Post-run workflow Sankey artifacts (site debug mode /
    # all completed root sessions; skips saved child runs of a
    # bigger session; never fails the run)
    from ....general_db_query_engine.general_posteval_helpers.compaction_funnels.api import (
        maybe_generate_workflow)
    maybe_generate_workflow(_out_dir, kind="query")

    return result


async def hardcoded_L0_wf_run(
    user_question: str,
    tool_executor: ToolExecutor,
    memory_path: str | Path,
) -> dict[str, Any]:
    """Execute the workflow phases in fixed order (no LLM-driven decisions).

    This drives the L0 workflow through all phases.  Phases that declare
    ``parallel_dispatch`` will fan out tool calls via ``asyncio.gather()``.

    Parameters
    ----------
    user_question : str
        The user's natural-language query.
    tool_executor : async callable
        ``async (tool_name, **kwargs) -> str | dict``.  The harness must
        route tool names to actual implementations (LLM calls for L1
        subagents, Python calls for memory tools, etc.).
    memory_path : str or Path
        Explicit path to this run's working-memory file.

    Returns
    -------
    dict with:
        workflow       — parsed workflow definition
        prompts        — system_prompt + user_prompt
        phase_results  — list[PhaseResult] from the async runner
        memory_path    — path to the working memory file
    """
    init_memory(memory_path)
    workflow = load_workflow()

    id_catalog = mem_tools.memory_catalog_list()
    context_text = mem_tools.memory_read()
    user_prompt = render_prompt(
        workflow,
        purpose=user_question,
        instruction="",
        context=context_text,
        id_catalog=id_catalog,
    )

    context = {
        "user_question": user_question,
        "id_catalog": id_catalog,
        "memory": context_text,
    }

    phase_results = await run_workflow(workflow, tool_executor, context=context)

    return {
        "workflow": workflow,
        "prompts": {
            "system_prompt": workflow["system_prompt"],
            "user_prompt": user_prompt,
        },
        "phase_results": phase_results,
        "memory_path": str(memory_path),
    }


def _repair_thin_answer(
    result: AgentTurnResult,
    l1_call_count: int,
    tools: Dict[str, Callable],
    memory_tools,
    memory: list[dict],
    client,
    system_prompt: str,
) -> AgentTurnResult:
    """Check a post-loop data-presentation rewrite against the same gate."""
    if not is_thin_answer(result.answer, l1_call_count):
        return result
    rewritten = force_data_presentation(
        result, memory_tools, memory, client, system_prompt,
    )
    if rewritten is not result:
        rewritten.answer = _apply_forced_final_grounding_gate(
            rewritten.answer, tools, rewritten.tool_history, memory, client,
            system_prompt,
        )
    return rewritten


def available_query_l0_tools() -> Dict[str, Callable]:
    """Callable tools installed by a standard Query L0 run."""
    tools = _build_tool_registry()
    # run() always installs its L1 relay after constructing the memory tools.
    tools["L1_query"] = dispatch_l1_query
    return tools
