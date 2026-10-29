"""L1 query worker dispatcher — wires search tools + L2 dispatchers to agent_turn.

This module is the real implementation that replaces the L1_query stub in
the orchestrator.  It:
  1. Loads L1_query_workflow.md
  2. Builds a tool registry via ``QueryL1Catalog`` (class-based catalog).
  3. Calls agent_turn() with an L1-level ArgoClient + wrapped tools.
  4. Launches parallel tool-free claim and ID/metadata agents.
  5. Returns the deterministically assembled downstream JSON result.

The orchestrator calls this via::
    result_json = dispatch_l1_query(
        purpose="Find VLE data for hexane+ethanol",
        instruction="Search for binary systems with activity coefficients...",
        id_catalog="[{\"type\": \"comp\", \"global_id\": \"GLOBcomp_12\", ...}]",
        context="Previous search found 3 blocks..."
    )

Public API
----------
dispatch_l1_query(purpose, instruction="", id_catalog="", context="") -> str
QueryL1Catalog — class-based tool catalog with two-stage compaction.
"""

from __future__ import annotations

from functools import partial
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Dict

log = logging.getLogger("L1-Query")

# ── Path setup for search tools ────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_AGENT_ROOT = _HERE.parent.parent             # NIST_ThermoML_query_agent
_SEARCH_ROOT = _AGENT_ROOT.parent.parent / "card_db_search_tools"
_BASIC_SEARCH = _SEARCH_ROOT / "basic_search_tools"
_BLOCK_SEARCH = _SEARCH_ROOT / "block_centric_search_tools"

for _p in (_BASIC_SEARCH, _BLOCK_SEARCH):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# ── Lazy imports ───────────────────────────────────────────────────────────
_fn_cache: Dict[str, Callable] = {}


def _get_fn(module: str, name: str) -> Callable:
    key = f"{module}.{name}"
    if key not in _fn_cache:
        mod = importlib.import_module(module)
        _fn_cache[key] = getattr(mod, name)
    return _fn_cache[key]


def _load_workflow() -> dict:
    return parse_workflow(_HERE / "L1_query_workflow.md")


# ── Catalog infrastructure ─────────────────────────────────────────────────────────────

from ....general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog import (
    AgentToolCatalog,
    CompactorCatalog,
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
from ....general_db_query_engine.general_hooks_management_helpers.general_postans_eval_hooks import (
    evaluate_and_assemble_return,
    prepare_answer_only_system_prompt,
)
from ....general_db_query_engine.general_data_grounding_gate import (
    core_claims_source_suffix,
    postanswer_checker_for_tools,
    export_inspections,
    harvest_inspections,
    without_ledger_tools,
)
from ...query_agent_argo_engine.argo_client import QueryClient
from ....general_db_query_engine.general_subagent_skill_schema_and_parser.subworkflow_md_tool_descriptions import (
    build_tool_instructions,
)
from ....general_db_query_engine.general_hooks_management_helpers.general_context_hooks import _context_hooks_anchors_catalog as ctx_anchor
from ...ThermoML_query_argo_config import AGENT_CONFIG as _cfg
from ...query_agent_context_hooks.hook_catalog import (
    QueryAgentHooks,
    build_agent_hooks,
)
from ...query_agent_context_hooks.memory_hooks.working_memory_hooks import (
    QueryWorkingMemory,
)
from ....general_db_query_engine.general_hooks_management_helpers.general_memory_management_tools_hooks_helpers.memory_local_storage_io import (
    _TEMPLATE as _WM_TEMPLATE,
)
from ..strict_output_contracts import validate_l1_query_output
from .core_id_management import refine_and_enrich_l1_core_ids

# ── Compactor catalog (for auto-resolving compactor_fn per tool) ───────────

from ...query_agent_context_hooks.compactor_hooks import (
    COMPACTOR_CATALOG as _COMPACTOR_CATALOG,
)

# ── L2 dispatchers ────────────────────────────────────────────────────────
from ..L2_leaf_evaluators.l2_dispatchers import (
    dispatch_l2_comp_eval,
    dispatch_l2_meas_eval,
    dispatch_l2_ref_eval,
    dispatch_l2_prop_eval,
)


# ── Module-level search tool entries ───────────────────────────────────────
# No explicit compactor_fn= needed — AgentToolCatalog auto-resolves from
# the CompactorCatalog when compactor_catalog= is set in __init__.

# ── Lazy client factory for the L1 tool-result compactor subagent ─────────
_l1_client = None


def _l1_client_factory():
    """Create/reuse an ArgoClient for L1's Layer-2 KEEP/DISCARD subagent."""
    global _l1_client
    if _l1_client is None:
        _l1_client = QueryClient.for_l1()
    return _l1_client


_SEARCH_ENTRIES = [
    # ── id_resolution ───────────────────────────────
    ToolEntry(
        "resolve_ids",
        _get_fn("_id_alignment_search", "resolve_ids_tool"),
        group="id_resolution",
    ),
    ToolEntry(
        "resolve_compound_ids",
        _get_fn("_id_alignment_search", "resolve_compound_ids_tool"),
        group="id_resolution",
    ),
    ToolEntry(
        "resolve_property_ids",
        _get_fn("_id_alignment_search", "resolve_property_ids_tool"),
        group="id_resolution",
    ),
    ToolEntry(
        "resolve_measurement_ids",
        _get_fn("_id_alignment_search", "resolve_measurement_ids_tool"),
        group="id_resolution",
    ),
    ToolEntry(
        "resolve_reference_ids",
        _get_fn("_id_alignment_search", "resolve_reference_ids_tool"),
        group="id_resolution",
    ),
    ToolEntry(
        "search_id_alignment",
        _get_fn("_id_alignment_search", "search_id_alignment"),
        group="id_resolution",
    ),

    # ── block_search ────────────────────────────────
    ToolEntry(
        "search_blocks",
        _get_fn("1_block_search", "search_blocks"),
        group="block_search",
        adaptive_condense=True,   # 3-tier internally
    ),
    ToolEntry(
        "block_search_adv",
        _get_fn("12_block_search_adv", "block_search_adv"),
        group="block_search",
        adaptive_condense=True,   # bounded deterministic advanced-result Markdown
    ),
    ToolEntry(
        "inspect_block_table",
        _get_fn("13_block_rdp_inspection", "inspect_block_table"),
        group="block_search",
        # Verbatim grounding table — deterministic output, no LLM triage.
        skip_subagent=True,
    ),
    ToolEntry(
        "search_system_registry",
        _get_fn("2_system_registry_search", "search_system_registry"),
        group="block_search",
        adaptive_condense=True,   # 3-tier internally
    ),
    ToolEntry(
        "search_system_summary",
        _get_fn("9_system_summary_search", "search_system_summary"),
        group="block_search",
    ),

    # ── compound similarity ─────────────────────────
    ToolEntry(
        "search_similar_compounds",
        _get_fn("10_compound_similarity_search", "search_similar_compounds"),
        group="compound_similarity",
    ),

]


class QueryL1Catalog(AgentToolCatalog):
    """L1 query worker tool catalog — search tools + L2 dispatchers.

    Search tool entries are defined at module level in ``_SEARCH_ENTRIES``.
    L2 dispatcher entries are added lazily to avoid circular imports.
    Compactors are auto-resolved from ``COMPACTOR_CATALOG``.
    """

    pipeline_label = "query"

    def __init__(self) -> None:
        # Build an L1-filtered copy of the compactor catalog — the global one
        # contains compactors for L2 / internal tools that L1 never exposes.
        # Gate off: the ledger's own tools are not offered at all.
        search_entries = without_ledger_tools(_SEARCH_ENTRIES)
        l1_tool_names = {e.name for e in search_entries}
        l1_compactors = CompactorCatalog(
            e for e in _COMPACTOR_CATALOG.entries.values()
            if e.tool_name in l1_tool_names
        )

        super().__init__(
            compactor_catalog=l1_compactors,
            client_factory=_l1_client_factory,
            cfg=_cfg,
        )
        self.register_many(search_entries)

        self._hooks: QueryAgentHooks | None = None

        self.register_many([
            ToolEntry("L2_comp_eval", self._dispatch_l2_comp_eval,
                      group="L2_subagent", skip_compactor=True, skip_subagent=True),
            ToolEntry("L2_meas_eval", self._dispatch_l2_meas_eval,
                      group="L2_subagent", skip_compactor=True, skip_subagent=True),
            ToolEntry("L2_ref_eval", self._dispatch_l2_ref_eval,
                      group="L2_subagent", skip_compactor=True, skip_subagent=True),
            ToolEntry("L2_prop_eval", self._dispatch_l2_prop_eval,
                      group="L2_subagent", skip_compactor=True, skip_subagent=True),
        ])

    def _dispatch_l2_comp_eval(
        self, purpose: str, instruction: str = "", context: str = "", id_catalog: str = "",
    ) -> str:
        return dispatch_l2_comp_eval(
            purpose, instruction, context, id_catalog, hooks=self._hooks
        )

    def _dispatch_l2_meas_eval(
        self, purpose: str, instruction: str = "", context: str = "", id_catalog: str = "",
    ) -> str:
        return dispatch_l2_meas_eval(
            purpose, instruction, context, id_catalog, hooks=self._hooks
        )

    def _dispatch_l2_ref_eval(
        self, purpose: str, instruction: str = "", context: str = "", id_catalog: str = "",
    ) -> str:
        return dispatch_l2_ref_eval(
            purpose, instruction, context, id_catalog, hooks=self._hooks
        )

    def _dispatch_l2_prop_eval(
        self, purpose: str, instruction: str = "", context: str = "", id_catalog: str = "",
    ) -> str:
        return dispatch_l2_prop_eval(
            purpose, instruction, context, id_catalog, hooks=self._hooks
        )

    def bind_hooks(self, hooks: QueryAgentHooks) -> None:
        self._hooks = hooks

    def _on_raw_result(self, tool_name: str, raw: dict) -> None:
        """Side-log DOI/block + entity references for query stats."""
        if self._hooks is not None:
            anchor(
                self._hooks.anchors.record_references,
                self._hooks.engine_hooks,
                tool_name=tool_name,
                raw_result=raw,
            )


# ── Singleton catalog instance ─────────────────────────────────────────────
_l1_catalog: QueryL1Catalog | None = None


def _get_catalog() -> QueryL1Catalog:
    """Lazy singleton — avoids import-time side effects."""
    global _l1_catalog
    if _l1_catalog is None:
        _l1_catalog = QueryL1Catalog()
    return _l1_catalog


# ── Public dispatcher ──────────────────────────────────────────────────────

@with_engine_config(_cfg)
def _export_worker_artifacts(worker_dir: Path, history_rec, section, *,
                             user_message: str, result, shipped: str | None,
                             wm_loader, degraded: bool = False) -> None:
    """Delegated worker dirs get the full artifact set a saved child
    session has (own run history + detailed log, result, WM) — not just
    the final context.  Best-effort: never disturbs the dispatch."""
    try:
        worker_dir.mkdir(parents=True, exist_ok=True)
        wm_text = ""
        if callable(wm_loader):
            try:
                wm_text = wm_loader() or ""
            except Exception:
                wm_text = ""
        if history_rec is not None and section is not None:
            status = ("DEGRADED" if degraded
                      else "TIMEOUT" if result is not None and result.timed_out
                      else "OK" if shipped is not None else "ERROR")
            history_rec.export_section_history(
                section[0], worker_dir / "run_history.md",
                agent="query-L1 (delegated)", prompt=user_message,
                working_memory=wm_text, final_status=status)
        if wm_text:
            (worker_dir / "_working_memory.md").write_text(
                wm_text, encoding="utf-8")
        if shipped is not None:
            from ....general_db_query_engine.general_hooks_management_helpers.general_tracking_hooks_output_helpers.output_writers import (
                prepare_result_answer,
                write_result_bundle,
            )
            target = worker_dir / "result.md"
            display_answer, structured_sections = prepare_result_answer(
                target, shipped)
            elapsed = (float(result.elapsed_seconds)
                       if result is not None else 0.0)
            iterations = result.iterations if result is not None else 0
            tools = len(result.tool_history) if result is not None else 0
            timed_out = bool(result.timed_out) if result is not None else False
            lines = [
                "# Delegated Query L1 worker — result",
                "",
                f"**Question:** {user_message}",
                "",
                f"**Iterations:** {iterations}  |  "
                f"**Elapsed:** {elapsed:.1f}s  |  "
                f"**Tools:** {tools}  |  "
                f"**Timed out:** {timed_out}",
                "",
                "---",
                "",
                "## Answer",
                "",
                display_answer,
                *structured_sections,
                "",
            ]
            write_result_bundle(target, "\n".join(lines), shipped)
    except Exception:
        log.warning("worker artifact export failed (ignored)", exc_info=True)


class L1DegradedResultError(RuntimeError):
    """L1 search completed but the post-answer pipeline failed.

    Carries a salvage payload (raw answer + every core ID the run searched)
    so callers can feed the partial findings back to the parent context
    instead of losing the whole worker run.
    """

    def __init__(self, original_error: str, salvage: dict) -> None:
        self.original_error = original_error
        self.salvage = salvage
        ids = salvage.get("searched_ids", {})
        counts = ", ".join(f"{k}:{len(v)}" for k, v in ids.items() if v)
        super().__init__(
            f"{original_error} [degraded L1 — search completed; salvaged "
            f"IDs {counts or 'none'}, {len(salvage.get('doi_blocks', []))} "
            f"doi/block refs; raw answer retained]"
        )


def _harvest_salvage(result, error_text: str) -> dict:
    """Collect the searched core IDs + raw answer from a finished ReAct turn.

    Walks the native tool results the same way the stats recorder does;
    malformed IDs are skipped so salvage can never raise.
    """
    from ....general_db_query_engine.general_hooks_management_helpers.general_context_hooks.stats_references_tracking_hooks import (
        _ENTITY_BY_GLOBAL_FIELD,
        _walk_objects,
    )
    from ThermoML_raw_json_to_card_db_parsers.id_schema import (
        require_block_id,
        require_global_id,
    )

    searched: dict[str, list[str]] = {}
    doi_blocks: list[dict] = []
    seen_blocks: set[tuple] = set()
    for step in result.tool_history:
        try:
            native = json.loads(step.get("result_full", ""))
        except (json.JSONDecodeError, TypeError):
            continue  # text/subagent payloads carry no native IDs
        for obj in _walk_objects(native):
            if "error" in obj:
                continue
            for field_name, entity_type in _ENTITY_BY_GLOBAL_FIELD.items():
                if field_name not in obj or obj[field_name] is None:
                    continue
                values = (obj[field_name] if isinstance(obj[field_name], list)
                          else [obj[field_name]])
                bucket = searched.setdefault(entity_type, [])
                for value in values:
                    try:
                        gid = require_global_id(field_name, value)
                    except (TypeError, ValueError):
                        continue
                    if gid not in bucket and len(bucket) < 80:
                        bucket.append(gid)
            if "doi" in obj and "block_number" in obj and len(doi_blocks) < 60:
                doi = obj["doi"]
                if not isinstance(doi, str) or not doi:
                    continue
                try:
                    block = require_block_id(obj["block_number"])
                except (TypeError, ValueError):
                    continue
                key = (doi, str(block))
                if key not in seen_blocks:
                    seen_blocks.add(key)
                    doi_blocks.append({"doi": doi, "block_number": str(block)})
    answer = result.answer or ""
    return {
        "degraded": True,
        "error": error_text,
        "raw_answer": answer[:6000] + ("…[truncated]" if len(answer) > 6000 else ""),
        "searched_ids": searched,
        "doi_blocks": doi_blocks,
        "tools_run": [t["tool"] for t in result.tool_history],
    }


def dispatch_l1_query(
    purpose: str,
    instruction: str = "",
    id_catalog: str = "",
    context: str = "",
    hooks: QueryAgentHooks | None = None,
    launcher_tool: str = "L1_query",
    final_context_dir: Path | str | None = None,
) -> str:
    """Dispatch the L1 query worker agent.

    Parameters
    ----------
    purpose : str
        High-level goal from L0 (e.g. "Find VLE data for hexane+ethanol").
    instruction : str
        Detailed search instructions from L0.
    id_catalog : str
        JSON string of current ID Catalog entries.
    context : str
        Prior context from working memory.
    launcher_tool : str
        Parent-side relay tool name recorded on this worker's nested
        history section ("L1_query", "query_thermoml", ...).
    final_context_dir : Path or str, optional
        When set, the worker's raw final LLM context is saved there as
        ``final_full_context.md`` (used by delegated dispatches that
        keep per-worker folders).

    Returns
    -------
    str
        Downstream JSON assembled from the untouched L1 answer text and the
        parallel post-answer agents' core_claims and ID/metadata fields.
    """
    workflow = _load_workflow()
    catalog = _get_catalog()
    # A delegated L1 call shares the parent's recorder/memory objects but must
    # use the L1 catalog's own pre-execution batch validator. Reusing the L0
    # compiled hook map here would validate L1 calls against the wrong catalog.
    if hooks is None:
        # Route history/stats to the recorders that own the CURRENT session
        # (e.g. an analysis run delegating query_thermoml). Without this the
        # hooks default to the query-agent singletons, whose state in a
        # foreign session has no output path — every L1 tool step, stage
        # compaction and pipeline size record would be silently dropped.
        from ....general_db_query_engine.general_hooks_management_helpers.general_context_hooks.history_tracking_hooks import (
            get_active_history_recorder,
        )
        from ....general_db_query_engine.general_hooks_management_helpers.general_context_hooks.stats_references_tracking_hooks import (
            get_active_recorder,
        )
        try:
            active_history = get_active_history_recorder()
        except RuntimeError:
            active_history = None
        try:
            active_stats = get_active_recorder()
        except RuntimeError:
            active_stats = None
        # Delegated sessions get a fresh in-process working memory so the
        # L1 sees the same ID-catalog scaffold as query-native sessions
        # (empty catalog falls back to the standard scaffold template).
        delegated_wm = QueryWorkingMemory()

        def _delegated_wm_render() -> str:
            return delegated_wm.render() or _WM_TEMPLATE

        wm_loader = _delegated_wm_render
        agent_hooks = build_agent_hooks(
            history=active_history,
            stats=active_stats,
            working_memory_loader=_delegated_wm_render,
            batch_validator=catalog.validate_batch,
        )
    else:
        wm_loader = hooks.working_memory_loader
        agent_hooks = build_agent_hooks(
            history=hooks.history,
            stats=hooks.stats,
            reasoning=hooks.reasoning,
            working_memory_loader=hooks.working_memory_loader,
            compactor=hooks.compactor,
            batch_summary=hooks.batch_summary,
            batch_validator=catalog.validate_batch,
        )
    catalog.bind_hooks(agent_hooks)
    tools = catalog.wrapped_tools()
    tools["L2_comp_eval"] = mark_subagent_answer_tool(
        partial(dispatch_l2_comp_eval, hooks=agent_hooks)
    )
    tools["L2_meas_eval"] = mark_subagent_answer_tool(
        partial(dispatch_l2_meas_eval, hooks=agent_hooks)
    )
    tools["L2_ref_eval"] = mark_subagent_answer_tool(
        partial(dispatch_l2_ref_eval, hooks=agent_hooks)
    )
    tools["L2_prop_eval"] = mark_subagent_answer_tool(
        partial(dispatch_l2_prop_eval, hooks=agent_hooks)
    )

    # ── Pre-flight health check ──────────────────────────────
    run_health_check(
        catalog,
        actual_tools=tools,
        label="query-L1",
    )

    # Build system prompt with tool instructions
    system_prompt = prepare_answer_only_system_prompt(
        workflow["system_prompt"] + "\n\n" + build_tool_instructions(tools)
    )
    system_prompt = anchor(
        ctx_anchor.SYNC_SYSTEM_PROMPT_READY,
        agent_hooks.engine_hooks,
        system_prompt=system_prompt,
        tool_names=sorted(tools.keys()),
        workflow=workflow,
    ) or system_prompt

    # Build the user message
    parts = [f"## Purpose\n{purpose}"]
    if instruction:
        parts.append(f"## Instruction\n{instruction}")
    if id_catalog:
        parts.append(f"## ID Catalog (from working memory)\n{id_catalog}")
    if context:
        parts.append(f"## Prior Context\n{context}")
    user_message = "\n\n".join(parts)
    user_message = anchor(
        ctx_anchor.SYNC_USER_MESSAGE_READY,
        agent_hooks.engine_hooks,
        user_message=user_message,
        purpose=purpose,
        instruction=instruction,
        context=context,
        id_catalog=id_catalog,
    ) or user_message

    client = QueryClient.for_l1()
    # This worker's tool calls log into their own nested history section,
    # claimed by the parent's relay row (layered tool-history policy)
    history_rec = getattr(agent_hooks, "history", None)
    section = None
    if history_rec is not None:
        section = history_rec.begin_nested_section("L1-query", launcher_tool)
    result = None
    shipped: str | None = None
    degraded = False
    try:
        result = agent_turn(
            user_message,
            system_prompt=system_prompt,
            tools=tools,
            memory=[],
            client=client,
            max_iterations=_cfg.L1_MAX_ITERATIONS,
            timeout=_cfg.L1_MAX_SECONDS,
            compaction_interval=_cfg.L1_COMPACTION_INTERVAL,
            hooks=agent_hooks.engine_hooks,
            is_subagent=True,
        )
        log.info(
            "L1 query: %d iters, %.1fs, timed_out=%s, tools=%s",
            result.iterations, result.elapsed_seconds, result.timed_out,
            [t["tool"] for t in result.tool_history],
        )
        if final_context_dir is not None:
            from ....general_db_query_engine.general_hooks_management_helpers.general_context_hooks.postjob_verdict_hooks import (
                save_final_context,
            )
            save_final_context(final_context_dir, result.final_context)
        try:
            postanswer = evaluate_and_assemble_return(
                agent_answer=result.answer,
                postassembly_checker=postanswer_checker_for_tools(tools),
            core_claims_system_suffix=core_claims_source_suffix(tools),
                response_json_schema=workflow["response_json_schema"],
                client=client,
                label="query-L1",
                tool_history=result.tool_history,
                task_context=user_message,
                hooks=agent_hooks,
                validator=validate_l1_query_output,
                id_alignment_processor=refine_and_enrich_l1_core_ids,
                deterministic_fields={
                    "data_inspections": export_inspections(
                        harvest_inspections(result.tool_history)
                    ),
                },
            )
        except Exception as post_exc:
            # The search itself succeeded — salvage its findings instead of
            # discarding the whole worker run with the post-answer failure.
            salvage = _harvest_salvage(result, str(post_exc))
            shipped = json.dumps(salvage, ensure_ascii=False, indent=1)
            degraded = True
            log.warning(
                "L1 post-answer pipeline failed — shipping degraded salvage "
                "(%d ID types, %d doi/blocks): %s",
                len(salvage["searched_ids"]), len(salvage["doi_blocks"]),
                post_exc,
            )
            raise L1DegradedResultError(str(post_exc), salvage) from post_exc
        shipped = postanswer.to_json()
        return shipped
    finally:
        if section is not None:
            history_rec.end_nested_section(section)
        if final_context_dir is not None:
            _export_worker_artifacts(
                Path(final_context_dir), history_rec, section,
                user_message=user_message, result=result, shipped=shipped,
                wm_loader=wm_loader, degraded=degraded)


# Live child catalogs for capability checks in a parent agent turn.
from ....general_db_query_engine.general_tool_management_helpers.tool_capabilities import (
    bind_reachable_tools,
)
from ..L2_leaf_evaluators.l2_dispatchers import _get_l2_catalog

bind_reachable_tools(dispatch_l1_query, lambda: _get_catalog().wrapped_tools())
for _kind in ("comp", "meas", "ref", "prop"):
    bind_reachable_tools(
        getattr(QueryL1Catalog, f"_dispatch_l2_{_kind}_eval"),
        lambda kind=_kind: _get_l2_catalog(kind).wrapped_tools(),
    )
