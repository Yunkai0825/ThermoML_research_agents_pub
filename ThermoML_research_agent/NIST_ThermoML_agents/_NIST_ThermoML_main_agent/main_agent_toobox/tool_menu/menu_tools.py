"""
Tool menu tools for the Main Agent.
====================================
Two tools that let the main agent *discover* and *execute*
subagent tools directly, without spinning up a full subagent run.

browse_subagent_tools  — miniature Argo subagent that takes purpose/tasks
                         and suggests a parallel execution plan with call
                         signatures from the hierarchical tool menu.
run_subagent_tool      — execute a subagent tool by name, applying the
                         complete Layer-1 + agentic Layer-2 compaction
                         pipeline from the owning subagent.

Tool descriptions **never** appear in the main agent's system prompt.
They only surface when the agent explicitly calls ``browse_subagent_tools``.
"""
from __future__ import annotations

import json
import logging
import time

from ....general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog import ToolEntry
from ....general_db_query_engine.general_tool_management_helpers.general_tool_results_compactor_agentic_hooks import (
    ToolResult,
    ToolResultCompactor,
)
from ._registry_builder import get_registry
from ...main_agent_argo_engine.argo_client import MainClient
from ...ThermoML_main_argo_config import AGENT_CONFIG as _menu_cfg

log = logging.getLogger("MAIN-TOOL-MENU")


# ═══════════════════════════════════════════════════════════════
#  Source-agent ToolResultCompactor instances (lazy)
# ═══════════════════════════════════════════════════════════════

_query_compactor: ToolResultCompactor | None = None
_analysis_compactor: ToolResultCompactor | None = None
_compactors_ready = False


def _ensure_compactors() -> None:
    """Lazy-build query and analysis ToolResultCompactor instances."""
    global _query_compactor, _analysis_compactor, _compactors_ready
    if _compactors_ready:
        return  # already initialised
    from ....NIST_ThermoML_query_agent.query_agent_context_hooks.compactor_hooks import (
        QueryToolResultCompactor,
    )
    from ....NIST_ThermoML_analysis_agent.analysis_agent_context_hooks.compactor_hooks import (
        AnalysisToolResultCompactor,
    )
    _query_compactor = QueryToolResultCompactor()
    _analysis_compactor = AnalysisToolResultCompactor()
    _compactors_ready = True
    log.info(
        "Compactors ready: query=%s, analysis=%s, merged=%d entries",
        _query_compactor is not None,
        _analysis_compactor is not None,
        len(_query_compactor.compactor_registry)
        + len(_analysis_compactor.compactor_registry),
    )


def _get_source_compactor(tool_name: str) -> ToolResultCompactor | None:
    """Return the source-agent ToolResultCompactor that owns *tool_name*.

    Checks query first, then analysis.  Returns None if the tool is not
    in either registry.
    """
    _ensure_compactors()
    if _query_compactor and tool_name in _query_compactor.compactor_registry:
        return _query_compactor
    if _analysis_compactor and tool_name in _analysis_compactor.compactor_registry:
        return _analysis_compactor
    return None
# ═══════════════════════════════════════════════════════════════
#  browse_subagent_tools  (miniature Argo planning subagent)
# ═══════════════════════════════════════════════════════════════

_BROWSE_SYSTEM = """\
You are a tool-selection advisor for the ThermoML Main Agent.

You have access to a hierarchical menu of subagent tools (database search
tools and analysis/fitting tools).  Given the user's PURPOSE and TASKS,
recommend which tools to call, in what order (or in parallel), and with
what arguments.

## Response Format

Return a structured plan in **exactly** this format:

### Recommended Plan

**Phase 1** (parallel):
- `tool_name_1(arg1="...", arg2=...)` — why this tool
- `tool_name_2(arg1="...", arg2=...)` — why this tool

**Phase 2** (after Phase 1 results):
- `tool_name_3(arg1="...", arg2=...)` — why this tool

### Notes
- Any caveats, dependencies, or alternative strategies.

Keep it concise.  Only recommend tools that appear in the MENU below.
Include actual argument values the caller should use (based on the tasks).
Numeric argument values (temperatures, ranges, limits) must come from
PURPOSE/TASKS or the MENU text — never invent example data values that
could be mistaken for database contents.
"""


def browse_subagent_tools(purpose: str, tasks: str) -> str:
    """Browse the tool menu and get a recommended execution plan.

    A miniature Argo subagent examines the full tool menu and returns
    a parallel execution plan with specific call signatures tailored
    to your purpose and tasks.

    Parameters
    ----------
    purpose : str
        High-level intent for using the tool menu (e.g.
        ``"Find VLE data for ethanol + water and fit RK polynomials"``).
    tasks : str
        Specific tasks or constraints (e.g.
        ``"Resolve compound IDs first, then search for blocks with
        excess enthalpy at 298.15 K"``).

    Returns
    -------
    str
        A structured execution plan with tool names, arguments,
        and phasing (parallel vs. sequential).
    """
    reg = get_registry()

    # Build full menu snapshot for the planner
    menu_md = reg.browse(path="")
    # Also include sub-menus for full visibility
    for top in ("query", "analysis"):
        sub = reg.browse(path=top)
        menu_md += f"\n\n{sub}"

    prompt = (
        f"## PURPOSE\n{purpose}\n\n"
        f"## TASKS\n{tasks}\n\n"
        f"## AVAILABLE TOOL MENU\n{menu_md}"
    )

    client = MainClient(
        model=_menu_cfg.MENU_PLANNER_MODEL,
        max_tokens=_menu_cfg.MENU_PLANNER_MAX_TOKENS,
        _tier="L0-menu-planner",
    )
    plan = client.call(prompt, _BROWSE_SYSTEM)
    log.info("Menu planner produced %d chars", len(plan))
    return plan


# ═══════════════════════════════════════════════════════════════
#  run_subagent_tool  (with hardcoded compaction)
# ═══════════════════════════════════════════════════════════════

def run_subagent_tool(
    tool_name: str,
    kwargs: dict,
    purpose: str,
    tasks: str,
) -> ToolResult | dict:
    """Execute a subagent tool directly with full compaction pipeline.

    Runs the tool, then applies the same two-layer compaction pipeline
    that the original subagent would use:

    - **Layer 1** — hardcoded dict→markdown (source agent's compactor)
    - **Layer 2** — agentic KEEP/DISCARD subagent (source agent's LLM client)

    Returns a ``ToolResult`` whose ``.text`` is the fully compacted
    output and ``.raw`` is the original dict.

    Parameters
    ----------
    tool_name : str
        Exact name of the tool (e.g. ``"search_blocks"``).
    kwargs : dict
        Keyword arguments for the tool.
    purpose : str
        High-level intent (passed to the compaction subagent).
    tasks : str
        Specific tasks (passed to the compaction subagent).

    Returns
    -------
    ToolResult
        ``.text`` = compacted markdown (Layer 1 + Layer 2),
        ``.raw``  = original tool result dict,
        ``.discarded`` = True if the subagent chose DISCARD.
        On error, returns a plain dict ``{"error": "..."}``.
    """
    reg = get_registry()

    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be an object")

    if not reg.has_tool(tool_name):
        raise KeyError(f"Unknown tool: {tool_name}")

    t0 = time.time()
    raw_result = reg.execute(tool_name, kwargs)

    elapsed = time.time() - t0

    if not isinstance(raw_result, dict):
        raise TypeError(
            f"Subagent tool {tool_name!r} returned {type(raw_result).__name__}; "
            "menu tools must return an object"
        )

    raw_result["_menu_tool"] = tool_name
    raw_result["_elapsed_s"] = round(elapsed, 2)

    # ── Full compaction pipeline (Layer 1 + Layer 2) ─────────
    #
    # Use the source agent's ToolResultCompactor which runs:
    #   Layer 1  hardcoded dict→markdown (same compactor the subagent uses)
    #   Layer 2  agentic KEEP/DISCARD subagent (with source agent's LLM client)
    #
    source_compactor = _get_source_compactor(tool_name)
    if source_compactor is None:
        raise RuntimeError(
            f"Tool {tool_name!r} has no source-agent compaction pipeline"
        )
    tool_result = source_compactor.compact_and_call(
        tool_name, purpose, tasks, raw_result,
    )
    log.info(
        "Full compaction %s: %d raw → %d compacted chars (discarded=%s)",
        tool_name, len(json.dumps(raw_result, default=str)),
        len(tool_result.text), tool_result.discarded,
    )
    return tool_result


# ═══════════════════════════════════════════════════════════════
#  TOOL_ENTRIES (consumed by tool_catalog.py)
# ═══════════════════════════════════════════════════════════════

TOOL_ENTRIES = [
    ToolEntry(
        "browse_subagent_tools", browse_subagent_tools,
        group="tool_menu",
        skip_compactor=True,
        skip_subagent=True,
    ),
    ToolEntry(
        "run_subagent_tool", run_subagent_tool,
        group="tool_menu",
        skip_compactor=True,
        skip_subagent=True,
    ),
]


from ....general_db_query_engine.general_tool_management_helpers.tool_capabilities import (
    bind_reachable_tools,
)

bind_reachable_tools(
    run_subagent_tool,
    lambda: get_registry().executable_tools,
)
