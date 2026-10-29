"""
Query-agent delegation tool — call the full L0 query agent as a sibling.
========================================================================
Unlike the existing ``query_delegation_tools`` (which bypass the L0
query agent and dispatch directly to L1 workers), this tool invokes the
complete L0 query agent with its own ReAct loop.  Use it for open-ended
or tricky database exploration questions that benefit from the query
agent's own reasoning.

Session nesting
---------------
When the analysis agent has an active session, each query-agent run
creates a subfolder under ``<session_dir>/query_runs/`` so its logs
(stats, history, reasoning) are co-located with the analysis output.
After each run, tracking MDs are merged back into the analysis session.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ....general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog import ToolEntry
from ....general_db_query_engine.general_text_context_marker_catalog import (
    mark_subagent_answer_tool,
)
from ....general_db_query_engine.general_subagent_delegation_helpers import (
    SubagentSessionManager,
    build_full_question,
    extract_run_result,
    dispatch_parallel,
    merge_subagent_tracking,
    preserve_active_session,
)

log = logging.getLogger("ANALYSIS-SIBLING-QUERY")


# ── session-nesting (injected from analysis hook_catalog) ───

def _get_session():
    from ...analysis_agent_context_hooks.hook_catalog import session_manager
    return session_manager.get_session()

_session_mgr = SubagentSessionManager(_get_session)


def _get_query_runner():
    """Lazy import to avoid circular imports at module load."""
    from ....NIST_ThermoML_query_agent.ThermoML_query_api import ThermoML_query_run
    return ThermoML_query_run


def _merge_tracking(subagent_dir, label: str = "") -> None:
    """Combine subagent tracking tables into the analysis session."""
    main_dir = _session_mgr.get_parent_session_dir()
    merge_subagent_tracking(main_dir, subagent_dir, "query", label=label)


def _run_one(
    question: str,
    purpose: str,
    tasks: str,
    context: str,
) -> Dict[str, Any]:
    """Run the full L0 query agent and return a flat result dict."""
    full_question = build_full_question(question, purpose=purpose, tasks=tasks, context=context)

    session_dir = _session_mgr.get_subagent_session_dir("query")
    with preserve_active_session():
        result = _get_query_runner()(full_question, session_dir=session_dir)

    subagent_session = session_dir
    _merge_tracking(subagent_session, label=question[:60])

    return extract_run_result(
        result, "query",
        question=question,
        session_dir=subagent_session,
    )


# ═══════════════════════════════════════════════════════════════
#  Public tools (called by the analysis agent)
# ═══════════════════════════════════════════════════════════════

@mark_subagent_answer_tool
def run_query_agent(
    question: str,
    purpose: str,
    tasks: str,
    context: str,
) -> dict:
    """Run the ThermoML Query Agent for in-depth database exploration.

    The query agent is a ThermoML database specialist with its own
    reasoning loop.  It resolves compound / property names, searches
    the ThermoML card databases, and returns structured results with
    DOIs, block numbers, and data summaries.

    Use this tool when you need:
    - Open-ended database exploration.
    - Complex compound/property resolution.
    - Queries that require multi-step reasoning over the database.

    For simpler, pre-structured queries, prefer ``query_thermoml``.

    Parameters
    ----------
    question : str
        The database search question (e.g. "Find all viscosity data
        for ethanol + water binary mixtures at 298.15 K").
    purpose : str
        High-level intent for the search — helps the query agent
        prioritize and focus its reasoning.
    tasks : str
        Specific tasks to accomplish (e.g. "1. Resolve compound IDs,
        2. Search for matching blocks, 3. Summarize data coverage").
    context : str
        Prior context from earlier tool calls or working memory
        (e.g. already-resolved compound IDs, prior search results).

    Returns
    -------
    dict
        Keys: answer (str), iterations (int), elapsed_seconds (float),
        tool_count (int), timed_out (bool), data_inspections (list —
        the child run's verbatim inspection ledger).
    """
    log.info("Dispatching to query agent: %s", question[:200])
    return _run_one(question, purpose=purpose, tasks=tasks, context=context)


@mark_subagent_answer_tool
def run_query_agents_parallel(queries: list[dict]) -> dict:
    """Run multiple query-agent tasks in parallel for broad exploration.

    Recommended when you need to explore multiple aspects of the
    database simultaneously — e.g. searching for different compounds,
    properties, or temperature ranges at the same time.

    Parameters
    ----------
    queries : list[dict]
        Query specs, each with keys:
        - label: identifier for this query (required)
        - question: the database question (required)
        - purpose: high-level intent (required)
        - tasks: specific tasks (required)
        - context: prior context or an explicit no-prior-context statement (required)

        Example::

            [
              {"label": "density_search",
               "question": "Find density data for ethanol + water at 298 K",
               "purpose": "Gather density measurements",
               "tasks": "Resolve IDs, search blocks, report coverage",
               "context": "No prior query results"},
              {"label": "viscosity_search",
               "question": "Find viscosity data for ethanol + water at 298 K",
               "purpose": "Gather viscosity measurements",
               "tasks": "Resolve IDs, search blocks, report coverage",
               "context": "Use the same target system and temperature"}
            ]

    Returns
    -------
    dict
        Keys: n_tasks (int), results (list of per-query result dicts).
    """
    return dispatch_parallel(
        queries,
        runner_fn=_run_one,
        max_workers=3,
        runner_kwargs_keys=("question", "purpose", "tasks", "context"),
        default_agent="query",
    )


# ═══════════════════════════════════════════════════════════════
#  Tool Entries (consumed by tool_catalog.py)
# ═══════════════════════════════════════════════════════════════

TOOL_ENTRIES = [
    ToolEntry(
        "run_query_agent", run_query_agent,
        group="sibling_delegation",
        skip_compactor=True, skip_subagent=True,
    ),
    ToolEntry(
        "run_query_agents_parallel", run_query_agents_parallel,
        group="sibling_delegation",
        skip_compactor=True, skip_subagent=True,
    ),
]


def _query_l0_tools():
    from ....NIST_ThermoML_query_agent.query_agent_workflows.L0_orchestrator.orchestrator import (
        available_query_l0_tools,
    )
    return available_query_l0_tools()


from ....general_db_query_engine.general_tool_management_helpers.tool_capabilities import (
    bind_reachable_tools,
)

bind_reachable_tools(run_query_agent, _query_l0_tools)
bind_reachable_tools(run_query_agents_parallel, _query_l0_tools)
