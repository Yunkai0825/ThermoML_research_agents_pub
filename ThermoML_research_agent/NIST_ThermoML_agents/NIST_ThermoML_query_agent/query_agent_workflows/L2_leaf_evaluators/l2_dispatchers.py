"""L2 leaf evaluator dispatchers — wire block-centric search tools to agent_turn.

Each ``dispatch_l2_*()`` function:
  1. Loads the corresponding L2 workflow markdown.
  2. Builds a tool registry via L2 catalog classes (class-based).
  3. Calls ``agent_turn()`` with the L2-specific ArgoClient + tool set.
  4. Launches parallel tool-free claim and ID/metadata agents.
  5. Returns the deterministically assembled downstream JSON string.

These dispatchers are registered as tools in the L1 query worker's
catalog, so the L1 agent can call ``L2_comp_eval(purpose, instruction,
context)`` and receive a structured downstream result.

Small L2 tools pass through directly. Potentially large block-level
measurement results are always deterministically compacted before entering
the L2 context; no agentic KEEP/DISCARD subagent is used at this layer.

Public API
----------
dispatch_l2_comp_eval(purpose, instruction, context, id_catalog="")  -> str
dispatch_l2_meas_eval(purpose, instruction, context, id_catalog="")  -> str
dispatch_l2_ref_eval(purpose, instruction, context, id_catalog="")   -> str
dispatch_l2_prop_eval(purpose, instruction, context, id_catalog="")  -> str

L2CompCatalog, L2MeasCatalog, L2RefCatalog, L2PropCatalog — catalog classes.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Callable, Dict

log = logging.getLogger("L2-Dispatch")

# ── Path setup for search tools ────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_AGENT_ROOT = _HERE.parent.parent             # NIST_ThermoML_query_agent
_SEARCH_ROOT = _AGENT_ROOT.parent.parent / "card_db_search_tools"
_BASIC_SEARCH = _SEARCH_ROOT / "basic_search_tools"
_BLOCK_SEARCH = _SEARCH_ROOT / "block_centric_search_tools"

# Inject search tool paths
for _p in (_BASIC_SEARCH, _BLOCK_SEARCH):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# ── Lazy imports (search tools use sys.path) ───────────────────────────────
_search_cache: Dict[str, Callable] = {}


def _get_search_fn(module_name: str, fn_name: str) -> Callable:
    """Lazy import a search function by module + function name."""
    key = f"{module_name}.{fn_name}"
    if key not in _search_cache:
        mod = importlib.import_module(module_name)
        _search_cache[key] = getattr(mod, fn_name)
    return _search_cache[key]


# ── Workflow loading ───────────────────────────────────────────────────────

def _load_workflow(filename: str) -> dict:

    return parse_workflow(_HERE / filename)


# ── Catalog infrastructure ─────────────────────────────────────────────────

from ....general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog import (
    AgentToolCatalog,
    ToolEntry,
)
from ....general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog.health_check_helper import (
    run_health_check,
)
from .._subworkflow_md_parser.subworkflow_parser import parse_workflow
from ....general_db_query_engine.general_argo_engine_helpers import (
    agent_turn,
    anchor,
    with_engine_config,
)
from ....general_db_query_engine.general_text_context_marker_catalog import (
    mark_subagent_answer_tool,
)
from ....general_db_query_engine.general_data_grounding_gate import (
    core_claims_source_suffix,
    postanswer_checker_for_tools,
)
from ....general_db_query_engine.general_hooks_management_helpers.general_postans_eval_hooks import (
    evaluate_and_assemble_return,
    prepare_answer_only_system_prompt,
)
from ...query_agent_argo_engine.argo_client import QueryClient
from ...query_agent_context_hooks.hook_catalog import QueryAgentHooks, build_agent_hooks
from ..strict_output_contracts import validate_l2_output
from .l2_field_construction import L2_ID_ALIGNMENT_PROCESSORS
from ....general_db_query_engine.general_subagent_skill_schema_and_parser.subworkflow_md_tool_descriptions import (
    build_tool_instructions,
)
from ....general_db_query_engine.general_hooks_management_helpers.general_context_hooks import _context_hooks_anchors_catalog as ctx_anchor
from ...ThermoML_query_argo_config import AGENT_CONFIG as _cfg

# ── Compactor catalog — L2 tools with skip_compactor=True don't need it,
# but block-centric tools that DO have compactors get auto-resolved.
from ...query_agent_context_hooks.compactor_hooks import (
    COMPACTOR_CATALOG as _COMPACTOR_CATALOG,
)


# ── Module-level L2 tool entries ───────────────────────────────────────────

_L2_COMP_ENTRIES = [
    ToolEntry("search_comp_from_block",
              _get_search_fn("search_comp_INDIV_and_DK_from_block", "search_comp_from_block"),
              group="block_centric", skip_compactor=True, skip_subagent=True),
    ToolEntry("search_compound_dk",
              _get_search_fn("3_compound_DK_search", "search_compound_dk"),
              group="basic_search", skip_compactor=True, skip_subagent=True),
    ToolEntry("search_compound_indiv",
              _get_search_fn("8_compound_INDIV_search", "search_compound_indiv"),
              group="basic_search", skip_compactor=True, skip_subagent=True),
]

_L2_MEAS_ENTRIES = [
    ToolEntry("search_meas_from_block",
              _get_search_fn("search_meas_INDIV_and_DK_from_block", "search_meas_from_block"),
              group="block_centric",
              compactor_fn=_COMPACTOR_CATALOG.registry["search_meas_from_block"],
              skip_subagent=True),
    ToolEntry("search_measurement_dk",
              _get_search_fn("5_meas_DK_search", "search_measurement_dk"),
              group="basic_search",
              compactor_fn=_COMPACTOR_CATALOG.registry["search_measurement_dk"],
              skip_subagent=True),
    ToolEntry("search_measurement_indiv",
              _get_search_fn("6_meas_INDIV_search", "search_measurement_indiv"),
              group="basic_search",
              compactor_fn=_COMPACTOR_CATALOG.registry["search_measurement_indiv"],
              skip_subagent=True),
]

_L2_REF_ENTRIES = [
    ToolEntry("search_reference_from_block",
              _get_search_fn("search_reference_from_block", "search_reference_from_block"),
              group="block_centric", skip_compactor=True, skip_subagent=True),
    ToolEntry("search_references",
              _get_search_fn("4_reference_search", "search_references"),
              group="basic_search", skip_compactor=True, skip_subagent=True),
]

_L2_PROP_ENTRIES = [
    ToolEntry("search_prop_dk_from_block",
              _get_search_fn("search_prop_DK_from_block", "search_prop_dk_from_block"),
              group="block_centric", skip_compactor=True, skip_subagent=True),
    ToolEntry("search_property_dk",
              _get_search_fn("7_prop_DK_search", "search_property_dk"),
              group="basic_search", skip_compactor=True, skip_subagent=True),
]


# ── L2 Catalog classes (lightweight — just register module-level entries) ──

class L2CompCatalog(AgentToolCatalog):
    """L2 compound evaluator — block-centric + DK/INDIV compound search."""
    pipeline_label = "query-l2-comp"

    def __init__(self) -> None:
        super().__init__(
            client_factory=QueryClient.for_l2,
            cfg=_cfg,
        )
        self.register_many(_L2_COMP_ENTRIES)


class L2MeasCatalog(AgentToolCatalog):
    """L2 measurement evaluator — block-centric + DK/INDIV measurement search."""
    pipeline_label = "query-l2-meas"

    def __init__(self) -> None:
        super().__init__(
            client_factory=QueryClient.for_l2,
            cfg=_cfg,
        )
        self.register_many(_L2_MEAS_ENTRIES)


class L2RefCatalog(AgentToolCatalog):
    """L2 reference evaluator — block-centric + basic reference search."""
    pipeline_label = "query-l2-ref"

    def __init__(self) -> None:
        super().__init__(
            client_factory=QueryClient.for_l2,
            cfg=_cfg,
        )
        self.register_many(_L2_REF_ENTRIES)


class L2PropCatalog(AgentToolCatalog):
    """L2 property evaluator — block-centric + DK property search."""
    pipeline_label = "query-l2-prop"

    def __init__(self) -> None:
        super().__init__(
            client_factory=QueryClient.for_l2,
            cfg=_cfg,
        )
        self.register_many(_L2_PROP_ENTRIES)


# ── L2 catalog singletons ─────────────────────────────────────────────────
_l2_comp: L2CompCatalog | None = None
_l2_meas: L2MeasCatalog | None = None
_l2_ref:  L2RefCatalog | None = None
_l2_prop: L2PropCatalog | None = None


def _get_l2_catalog(kind: str) -> AgentToolCatalog:
    global _l2_comp, _l2_meas, _l2_ref, _l2_prop
    if kind == "comp":
        if _l2_comp is None:
            _l2_comp = L2CompCatalog()
        return _l2_comp
    elif kind == "meas":
        if _l2_meas is None:
            _l2_meas = L2MeasCatalog()
        return _l2_meas
    elif kind == "ref":
        if _l2_ref is None:
            _l2_ref = L2RefCatalog()
        return _l2_ref
    elif kind == "prop":
        if _l2_prop is None:
            _l2_prop = L2PropCatalog()
        return _l2_prop
    raise ValueError(f"Unknown L2 kind: {kind}")


# ── Generic L2 dispatch ───────────────────────────────────────────────────

@with_engine_config(_cfg)
def _dispatch_l2(
    workflow_file: str,
    catalog_kind: str,
    *,
    purpose: str,
    instruction: str,
    context: str,
    id_catalog: str = "",
    hooks: QueryAgentHooks | None = None,
) -> str:
    """Generic L2 dispatch: load workflow, get catalog tools, run agent_turn."""
    workflow = _load_workflow(workflow_file)
    catalog = _get_l2_catalog(catalog_kind)
    tools = catalog.wrapped_tools()

    # ── Pre-flight health check ──────────────────────────────
    run_health_check(
        catalog,
        actual_tools=tools,
        label=f"query-L2-{catalog_kind}",
    )

    # Build system prompt with tool-calling instructions
    system_prompt = prepare_answer_only_system_prompt(
        workflow["system_prompt"] + "\n\n" + build_tool_instructions(tools)
    )

    # Build user message from purpose + instruction + context
    user_msg_parts = [f"## Purpose\n{purpose}"]
    if instruction:
        user_msg_parts.append(f"## Instruction\n{instruction}")
    if context:
        user_msg_parts.append(f"## Block Context\n{context}")
    if id_catalog:
        user_msg_parts.append(f"## ID Catalog\n{id_catalog}")
    user_message = "\n\n".join(user_msg_parts)

    client = QueryClient.for_l2()
    agent_hooks = hooks or build_agent_hooks()
    system_prompt = anchor(
        ctx_anchor.SYNC_SYSTEM_PROMPT_READY,
        agent_hooks.engine_hooks,
        system_prompt=system_prompt,
        tool_names=sorted(tools.keys()),
        workflow=workflow,
    ) or system_prompt
    user_message = anchor(
        ctx_anchor.SYNC_USER_MESSAGE_READY,
        agent_hooks.engine_hooks,
        user_message=user_message,
        purpose=purpose,
        instruction=instruction,
        context=context,
        id_catalog=id_catalog,
        workflow_file=workflow_file,
        catalog_kind=catalog_kind,
    ) or user_message
    # The evaluator's own card reads log into a nested section under the
    # calling L1 worker's section (layered tool-history policy)
    history_rec = getattr(agent_hooks, "history", None)
    section = None
    if history_rec is not None:
        section = history_rec.begin_nested_section(
            f"L2-{catalog_kind} eval", f"L2_{catalog_kind}_eval")
    try:
        result = agent_turn(
            user_message,
            system_prompt=system_prompt,
            tools=tools,
            memory=[],
            client=client,
            max_iterations=_cfg.L2_MAX_ITERATIONS,
            timeout=_cfg.L2_MAX_SECONDS,
            hooks=agent_hooks.engine_hooks,
            is_subagent=True,
        )

        log.info(
            "L2 %s: %d iters, %.1fs, timed_out=%s",
            workflow_file, result.iterations, result.elapsed_seconds,
            result.timed_out,
        )

        postanswer = evaluate_and_assemble_return(
            agent_answer=result.answer,
            postassembly_checker=postanswer_checker_for_tools(tools),
            core_claims_system_suffix=core_claims_source_suffix(tools),
            response_json_schema=workflow["response_json_schema"],
            client=client,
            label=f"query-L2-{catalog_kind}",
            tool_history=result.tool_history,
            task_context=user_message,
            hooks=agent_hooks,
            validator=lambda payload: validate_l2_output(catalog_kind, payload),
            id_alignment_processor=L2_ID_ALIGNMENT_PROCESSORS[catalog_kind],
        )
        return postanswer.to_json()
    finally:
        if section is not None:
            history_rec.end_nested_section(section)


# ── Public dispatchers ─────────────────────────────────────────────────────

@mark_subagent_answer_tool
def dispatch_l2_comp_eval(
    purpose: str, instruction: str = "", context: str = "", id_catalog: str = "", hooks: QueryAgentHooks | None = None,
) -> str:
    """Dispatch the L2 compound evaluator subagent."""
    return _dispatch_l2(
        "L2_comp_eval_workflow.md", "comp",
        purpose=purpose, instruction=instruction,
        context=context, id_catalog=id_catalog, hooks=hooks,
    )


@mark_subagent_answer_tool
def dispatch_l2_meas_eval(
    purpose: str, instruction: str = "", context: str = "", id_catalog: str = "", hooks: QueryAgentHooks | None = None,
) -> str:
    """Dispatch the L2 measurement evaluator subagent."""
    return _dispatch_l2(
        "L2_meas_eval_workflow.md", "meas",
        purpose=purpose, instruction=instruction,
        context=context, id_catalog=id_catalog, hooks=hooks,
    )


@mark_subagent_answer_tool
def dispatch_l2_ref_eval(
    purpose: str, instruction: str = "", context: str = "", id_catalog: str = "", hooks: QueryAgentHooks | None = None,
) -> str:
    """Dispatch the L2 reference evaluator subagent."""
    return _dispatch_l2(
        "L2_ref_eval_workflow.md", "ref",
        purpose=purpose, instruction=instruction,
        context=context, id_catalog=id_catalog, hooks=hooks,
    )


@mark_subagent_answer_tool
def dispatch_l2_prop_eval(
    purpose: str, instruction: str = "", context: str = "", id_catalog: str = "", hooks: QueryAgentHooks | None = None,
) -> str:
    """Dispatch the L2 property evaluator subagent."""
    return _dispatch_l2(
        "L2_prop_eval_workflow.md", "prop",
        purpose=purpose, instruction=instruction,
        context=context, id_catalog=id_catalog, hooks=hooks,
    )


from ....general_db_query_engine.general_tool_management_helpers.tool_capabilities import (
    bind_reachable_tools,
)

for _kind in ("comp", "meas", "ref", "prop"):
    bind_reachable_tools(
        globals()[f"dispatch_l2_{_kind}_eval"],
        lambda kind=_kind: _get_l2_catalog(kind).wrapped_tools(),
    )
