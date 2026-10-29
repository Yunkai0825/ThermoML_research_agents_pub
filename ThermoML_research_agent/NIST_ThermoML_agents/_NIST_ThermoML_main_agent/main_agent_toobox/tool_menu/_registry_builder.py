"""
Tool menu registry builder — lazy-loads subagent tools into a tree.
====================================================================
Populates a ``ToolMenuRegistry`` with tools from the query and analysis
agents on first access.  All imports are deferred to avoid circular
dependencies and heavy module loads at startup.

Tree structure built
--------------------
::

    query/
        id_resolution/     — resolve_ids, resolve_compound_ids, …
        block_search/      — search_blocks, search_system_registry, …
        compound_similarity/ — search_similar_compounds
    analysis/
        hardcoded_data/    — get_pure_values, find_similar_compounds, …
        fitting/           — fit_block, fit_multi_system, …
        discovery/         — discover_block_variables, discover_block_composition, …
"""
from __future__ import annotations

import logging

from ....general_db_query_engine.general_tool_management_helpers.general_tool_menu_tools import (
    ToolMenuRegistry,
)
from ....general_db_query_engine.general_data_grounding_gate import without_ledger_tools

log = logging.getLogger("MENU-BUILDER")

_registry = None


def get_registry():
    """Return the shared tool-menu registry (built lazily on first call)."""
    global _registry
    if _registry is None:
        _registry = _build()
    return _registry


def _build():
    reg = ToolMenuRegistry()
    _register_query_tools(reg)
    _register_analysis_tools(reg)

    # Category descriptions (shown when browsing that level)
    reg.set_category_description("query",
        "Database search tools from the Query Agent — ID resolution, "
        "block search, compound similarity.")
    reg.set_category_description("analysis",
        "Fitting and data-inspection tools from the Analysis Agent — "
        "RK fitting, pure-value extraction, block discovery.")

    # Pre-flight: every menu tool must have a valid callable
    reg.validate()

    log.info("Tool menu built: %d tools", reg.size)
    return reg


# ── Query agent: L1 search tools ─────────────────────────────

def _register_query_tools(reg) -> None:
    """Register the query agent's search tools under ``query/``."""
    from ....NIST_ThermoML_query_agent.query_agent_workflows.L1_workers.l1_query_dispatcher import (
        _SEARCH_ENTRIES,
    )
    # Each entry has a .group; register_from_tool_entries appends
    # group as a sub-level: query / <group> / <tool_name>
    # Gate off: the ledger's own tools are not offered at all.
    reg.register_from_tool_entries("query", without_ledger_tools(_SEARCH_ENTRIES))


# ── Analysis agent: fitting + hardcoded data + discovery ──────

def _register_analysis_tools(reg) -> None:
    """Register analysis agent tools under ``analysis/``."""
    from ....NIST_ThermoML_analysis_agent.analysis_agent_toolbox.hardcoded_data_tools import (
        TOOL_ENTRIES as hc_entries,
    )
    from ....NIST_ThermoML_analysis_agent.analysis_agent_toolbox.fitting_tools import (
        TOOL_ENTRIES as fit_entries,
    )
    from ....NIST_ThermoML_analysis_agent.analysis_agent_toolbox.discovery_tools import (
        TOOL_ENTRIES as disc_entries,
    )
    reg.register_from_tool_entries(
        "analysis", without_ledger_tools(hc_entries + fit_entries + disc_entries),
    )
