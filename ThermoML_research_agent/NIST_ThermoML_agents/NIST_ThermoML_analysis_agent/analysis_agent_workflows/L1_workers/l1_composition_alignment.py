"""L1 composition-alignment worker — a chemist-reviewer react agent.

Builds the mapping between composition conventions for ONE binary
system strictly from ACTUAL literature data blocks:

  1. Loads ``L1_composition_alignment_workflow.md``.
  2. Builds a restricted tool registry via ``CompAlignmentCatalog``
     (search + inspect + fit + dual-basis validation).
  3. Runs ``agent_turn()`` with an L1-level AnalysisClient.
  4. Launches parallel tool-free claim and ID/metadata agents.
  5. Returns deterministic downstream JSON (basis inventory, bridge
     candidates with fit stats, chosen bridge + literature reason,
     cross-validation evidence, unsupported conversions) with the
     untouched answer text.

Division of labor: the AGENT chooses among literature sources and
documents why; deterministic code (fit_block, molar masses from block
formulas, dual-basis checks) computes every number.

Public API
----------
dispatch_composition_alignment(purpose, instruction="", id_catalog="",
                               context="") -> str
validate_dual_basis_block(doi, block_number) -> dict
register_estimated_bridge(...) -> dict   (L1 tool — estimation artifact)
consume_estimations() -> list[dict]      (assembler-side registry drain)
CompAlignmentCatalog — restricted class-based tool catalog.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np
from ThermoML_raw_json_to_card_db_parsers.id_schema import (
    require_block_id,
    require_block_local_id,
)

log = logging.getLogger("COMP-ALIGN-L1")

# ── Path setup for the query-side search tools ─────────────────────────────
_HERE = Path(__file__).resolve().parent
_SEARCH_ROOT = _HERE.parents[3] / "card_db_search_tools"
_BASIC_SEARCH = _SEARCH_ROOT / "basic_search_tools"
_BLOCK_SEARCH = _SEARCH_ROOT / "block_centric_search_tools"

for _p in (_BASIC_SEARCH, _BLOCK_SEARCH):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# ── Lazy imports for search tool functions ─────────────────────────────────
_fn_cache: Dict[str, Callable] = {}


def _get_fn(module: str, name: str) -> Callable:
    key = f"{module}.{name}"
    if key not in _fn_cache:
        mod = importlib.import_module(module)
        _fn_cache[key] = getattr(mod, name)
    return _fn_cache[key]


# ── Catalog / engine infrastructure ────────────────────────────────────────
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
from ....general_db_query_engine.general_subagent_skill_schema_and_parser.subworkflow_md_tool_descriptions import (
    build_tool_instructions,
)
from ....general_db_query_engine.general_hooks_management_helpers.general_context_hooks import (
    _context_hooks_anchors_catalog as ctx_anchor,
)
from ....general_db_query_engine.general_hooks_management_helpers.general_context_hooks.history_tracking_hooks import (
    get_active_history_recorder,
)
from ....general_db_query_engine.general_hooks_management_helpers.general_context_hooks.stats_references_tracking_hooks import (
    get_active_recorder,
)

from ...ThermoML_analysis_argo_config import AGENT_CONFIG as _cfg
from ...analysis_agent_argo_engine.argo_client import AnalysisClient
from ...analysis_agent_context_hooks.hook_catalog import (
    AnalysisAgentHooks,
    build_agent_hooks,
    get_session,
    COMPACTOR_CATALOG as _ANALYSIS_COMPACTORS,
)
from ....NIST_ThermoML_query_agent.query_agent_context_hooks.compactor_hooks import (
    COMPACTOR_CATALOG as _QUERY_COMPACTORS,
)

# ── Analysis-side deterministic tools ──────────────────────────────────────
from ...analysis_agent_toolbox.hardcoded_data_tools import inspect_block
from ...analysis_agent_toolbox.discovery_tools import get_pure_values
from ...analysis_agent_toolbox.fitting_tools import fit_block, _formula_weight
from ...ThermoML_core_calc_tools.csv_io_helpers import (
    companion_columns,
    extract_block_arrays,
    identify_columns,
)
from ...ThermoML_core_calc_tools.mixture_nonideality_calc.composition_transforms import (
    detect_composition_basis,
    mass_to_mole,
    molality_to_mole,
    mass_ratio_to_mole,
    amount_ratio_to_mole,
)

_WORKFLOW_PATH = _HERE / "L1_composition_alignment_workflow.md"

# Bases whose conversion to mole fraction is exact given molar masses
_EXACT_CHECKABLE = {
    "mole_fraction", "mass_fraction", "molality", "mass_ratio", "amount_ratio",
}


def _load_workflow() -> dict:
    return parse_workflow(_WORKFLOW_PATH)


# ═══════════════════════════════════════════════════════════════════════════
#  validate_dual_basis_block — literature-anchored conversion check
# ═══════════════════════════════════════════════════════════════════════════

def _to_mole_fraction_of(
    values: np.ndarray, basis: str, col_compound: str, ref_compound: str,
    molar_masses: dict[str, float], compounds: list[str],
) -> np.ndarray | None:
    """Convert a composition column to mole fraction of *ref_compound*.

    Returns None when the basis needs bridging data (not exactly
    checkable), the column's compound is unresolved (orientation
    ambiguous), or molar masses are missing.
    """
    if not col_compound or col_compound not in compounds:
        return None
    others = [c for c in compounds if c != col_compound]
    other = others[0] if others else None
    M1 = molar_masses.get(col_compound)
    M2 = molar_masses.get(other) if other else None

    if basis == "mole_fraction":
        x_col = values
    elif basis == "mass_fraction":
        if not (M1 and M2):
            return None
        x_col = mass_to_mole(values, M1, M2)
    elif basis == "molality":
        if not M2:
            return None
        x_col = molality_to_mole(values, M2)
    elif basis == "mass_ratio":
        if not (M1 and M2):
            return None
        x_col = mass_ratio_to_mole(values, M1, M2)
    elif basis == "amount_ratio":
        x_col = amount_ratio_to_mole(values)
    else:
        return None

    if col_compound == ref_compound:
        return np.asarray(x_col, dtype=float)
    return 1.0 - np.asarray(x_col, dtype=float)


def validate_dual_basis_block(
    doi: str,
    block_number: str,
    BLKsubsys_id: str | None = None,
) -> dict:
    """Cross-check the exact molar-mass conversion against a block's own
    dual-reported composition columns.

    For a block that reports the SAME composition on two bases (e.g.
    mole fraction AND mass fraction columns), converts one column to the
    other's basis using molar masses from the block's chemical formulas
    and reports per-row deviation statistics — direct literature
    confirmation (or refutation) of the stoichiometric mapping.
    """
    subsystem_id = (
        require_block_local_id("subsys", BLKsubsys_id)
        if BLKsubsys_id is not None else None
    )
    bd = extract_block_arrays(doi, block_number, BLKsubsys_id=subsystem_id)
    target = {
        "doi": doi,
        "block_number": require_block_id(block_number),
        "BLKsubsys_id": subsystem_id,
    }
    if not bd.ok:
        return {"error": bd.error, **target}

    match = identify_columns(bd.columns, "", metadata=bd.metadata)
    cc_map = dict(match.column_compound_map or {})

    # Compounds + molar masses from the block's own formulas
    compounds: list[str] = []
    molar_masses: dict[str, float] = {}
    for c in bd.metadata.get("compounds", []) or []:
        nm = c.get("name", "") if isinstance(c, dict) else str(c)
        if nm:
            compounds.append(nm)
        if isinstance(c, dict):
            mw = _formula_weight(c.get("formula") or "")
            if nm and mw:
                molar_masses[nm] = mw

    # All composition columns with a recognized basis.  A companion column
    # ("<col>_U", "<col>_limit") describes another column of the block and
    # is never a composition of its own: the "_U" column of a mole-fraction
    # property would otherwise count as a second composition column.
    companions = companion_columns(bd.columns)
    comp_cols: list[dict] = []
    for col in bd.columns:
        if col in companions:
            continue
        basis = detect_composition_basis(col)
        if basis in ("unknown",):
            continue
        comp_cols.append({
            "column": col,
            "basis": basis,
            "compound": cc_map.get(col) or "",
            "checkable": basis in _EXACT_CHECKABLE,
        })

    if len(comp_cols) < 2:
        return {
            **target,
            "n_composition_columns": len(comp_cols),
            "columns": comp_cols,
            "verdict": "not dual-basis — fewer than two composition columns",
        }

    ref = comp_cols[0]["compound"]
    if not ref:
        return {
            **target,
            "n_composition_columns": len(comp_cols),
            "columns": comp_cols,
            "verdict": (
                "not checkable — the primary composition column has no "
                "explicit DOIcomp_N component mapping"
            ),
        }
    pairs: list[dict] = []
    for i in range(len(comp_cols)):
        for j in range(i + 1, len(comp_cols)):
            a, b = comp_cols[i], comp_cols[j]
            pair: dict = {
                "col_a": a["column"], "basis_a": a["basis"],
                "col_b": b["column"], "basis_b": b["basis"],
            }
            if a["basis"] == b["basis"] and a["compound"] == b["compound"]:
                pair["note"] = "same basis and compound — nothing to cross-check"
                pairs.append(pair)
                continue
            if not (a["checkable"] and b["checkable"]):
                pair["note"] = (
                    "one basis is density-dependent — not exactly checkable "
                    "without a bridge (skipped)"
                )
                pairs.append(pair)
                continue
            va = np.asarray(bd.arrays.get(a["column"]), dtype=float)
            vb = np.asarray(bd.arrays.get(b["column"]), dtype=float)
            xa = _to_mole_fraction_of(va, a["basis"], a["compound"],
                                      ref, molar_masses, compounds)
            xb = _to_mole_fraction_of(vb, b["basis"], b["compound"],
                                      ref, molar_masses, compounds)
            if xa is None or xb is None:
                pair["note"] = ("missing molar masses or unresolved column "
                                "compound — conversion not checkable")
                pairs.append(pair)
                continue
            ok = ~np.isnan(xa) & ~np.isnan(xb)
            if not np.any(ok):
                pair["note"] = "no overlapping valid rows"
                pairs.append(pair)
                continue
            dev = np.abs(xa[ok] - xb[ok])
            denom = np.maximum(np.abs(xa[ok]), 1e-3)
            pair.update({
                "reference_compound": ref,
                "n_rows": int(np.count_nonzero(ok)),
                "mean_abs_dev_x": round(float(np.mean(dev)), 6),
                "max_abs_dev_x": round(float(np.max(dev)), 6),
                "max_rel_dev_pct": round(float(np.max(dev / denom) * 100), 3),
                "molar_masses_g_mol": {k: round(v, 4)
                                       for k, v in molar_masses.items()},
            })
            pairs.append(pair)

    checked = [p for p in pairs if "mean_abs_dev_x" in p]
    if checked:
        worst = max(p["max_rel_dev_pct"] for p in checked)
        verdict = (
            f"molar-mass conversion CONFIRMED by the paper's own dual "
            f"reporting (worst deviation {worst:g}%)" if worst < 1.0 else
            f"conversion deviates up to {worst:g}% from the paper's dual "
            f"reporting — inspect before trusting"
        )
    else:
        verdict = "no exactly-checkable basis pair in this block"

    return {
        **target,
        "n_composition_columns": len(comp_cols),
        "columns": comp_cols,
        "pairs": pairs,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  register_estimated_bridge — LAST-RESORT estimation with artifact
# ═══════════════════════════════════════════════════════════════════════════

# Estimations registered by the agent during ONE dispatch (drained by the
# align_compositions assembler after agent_turn returns — same process).
_ESTIMATION_REGISTRY: list[dict] = []


def consume_estimations() -> list[dict]:
    """Drain and return estimations registered during the last dispatch."""
    out = list(_ESTIMATION_REGISTRY)
    _ESTIMATION_REGISTRY.clear()
    return out


def register_estimated_bridge(
    x_compound: str,
    pure_density_x1: float,
    pure_density_x0: float,
    temperature_K: float,
    justification: str,
    references: list[dict],
    coeffs: list[float] | None = None,
) -> dict:
    """Register an ESTIMATED ρ(x) bridge — LAST RESORT when no literature
    density source exists at the requested state.

    The estimate MUST be grounded: *justification* explains the physical
    reasoning in detail and *references* lists the database blocks
    the numbers were derived from (e.g. pure densities at a nearby
    temperature via get_pure_values).  The estimate is persisted as a
    clearly-labeled session artifact and flows into the composition
    library flagged ``estimated: true`` — it never masquerades as
    measured data.

    Parameters
    ----------
    x_compound : str
        Compound whose mole fraction is the bridge's x axis.
    pure_density_x1, pure_density_x0 : float
        Estimated pure densities (kg/m³) at x=1 (x_compound) and x=0
        (the other component) at *temperature_K*.
    temperature_K : float
        State temperature the estimate targets.
    justification : str
        Detailed reasoning (≥ 80 chars): what was estimated, from which
        database evidence, with what assumption (e.g. ideal mixing,
        thermal-expansion extrapolation) and expected error.
    references : list[dict]
        Non-empty array of {"doi": ..., "block_number": ..., "provided":
        ...} database references backing the estimate. Omit ``BLKsubsys_id``
        for the declared parent; include an exact ``BLKsubsys_N`` only for an
        explicitly selected embedded subsystem.
    coeffs : list[float] | None
        Optional numeric RK coefficients for an excess-volume guess;
        omission means ideal mixing (linear ρ via molar volumes).
    """
    if not justification or len(justification.strip()) < 80:
        raise ValueError(
            "justification too short — provide detailed physical reasoning "
            "(what was estimated, from which database evidence, with what "
            "assumption, expected error; ≥ 80 chars)."
        )
    if not isinstance(references, list) or not references:
        raise TypeError("references must be a non-empty array")
    normalized_references = []
    for index, reference in enumerate(references):
        if not isinstance(reference, dict):
            raise TypeError(f"references[{index}] must be an object")
        reference_fields = {"doi", "block_number", "BLKsubsys_id", "provided"}
        required_fields = reference_fields - {"BLKsubsys_id"}
        missing = sorted(required_fields - set(reference))
        unknown = sorted(set(reference) - reference_fields)
        if missing or unknown:
            raise ValueError(
                f"references[{index}] has missing={missing} and unknown={unknown} fields"
            )
        normalized = dict(reference)
        normalized.setdefault("BLKsubsys_id", None)
        if any(
            not isinstance(normalized[field], str) or not normalized[field]
            for field in ("doi", "block_number", "provided")
        ):
            raise TypeError(f"references[{index}] text fields must be non-empty strings")
        require_block_id(normalized["block_number"])
        if normalized["BLKsubsys_id"] is not None:
            require_block_local_id("subsys", normalized["BLKsubsys_id"])
        normalized_references.append(normalized)
    if coeffs is None:
        coeffs = []
    if not isinstance(coeffs, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in coeffs
    ):
        raise TypeError("coeffs must be an array of JSON numbers")
    for field, value in {
        "pure_density_x1": pure_density_x1,
        "pure_density_x0": pure_density_x0,
        "temperature_K": temperature_K,
    }.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be a JSON number")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be finite and positive")
    rho1, rho0, T = pure_density_x1, pure_density_x0, temperature_K

    bridge = {
        "property": "mass_density",
        "doi": None,
        "block_number": None,
        "BLKsubsys_id": None,
        "x_compound": x_compound,
        "coeffs": coeffs,
        "pure_at_x1": rho1,
        "pure_at_x0": rho0,
        "temperature_K": T,
        "r_squared": None,
        "n_mixture_points": 0,
        "estimated": True,
        "route": "ESTIMATED — not a measured bridge",
        "justification": justification.strip(),
        "references": normalized_references,
    }

    artifact_path = None
    sess = get_session()
    if sess:
        idx = len(_ESTIMATION_REGISTRY) + 1
        path = sess.data_path(f"ESTIMATED_bridge_{idx}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(bridge, fh, indent=1, default=str)
        sess.register_file(
            "data", str(path),
            f"ESTIMATED density bridge (x={x_compound}, T={T} K) — "
            "agent estimate with justification + db references",
        )
        artifact_path = str(path)
    bridge["artifact_file"] = artifact_path

    _ESTIMATION_REGISTRY.append(bridge)
    log.info("ESTIMATED bridge registered: x=%s T=%.2fK refs=%d",
             x_compound, T, len(references))
    return {
        "registered": True,
        "estimated": True,
        "artifact_file": artifact_path,
        "bridge_preview": {
            "x_compound": x_compound, "temperature_K": T,
            "pure_at_x1": rho1, "pure_at_x0": rho0, "coeffs": coeffs,
        },
        "note": ("Estimate registered and persisted as artifact. It will "
                 "enter the library flagged estimated=true. Still list the "
                 "affected conversions in unsupported_conversions with a "
                 "pointer to this estimate."),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CompAlignmentCatalog
# ═══════════════════════════════════════════════════════════════════════════

_comp_client = None


def _comp_client_factory():
    """Create/reuse an AnalysisClient for the KEEP/DISCARD subagent layer."""
    global _comp_client
    if _comp_client is None:
        _comp_client = AnalysisClient.for_l1_data()
    return _comp_client


_TOOL_ENTRIES = [
    # ── db_search (query-side search functions) ─────────────
    ToolEntry(
        "search_blocks",
        _get_fn("1_block_search", "search_blocks"),
        group="db_search",
        adaptive_condense=True,
    ),
    ToolEntry(
        "inspect_block_table",
        _get_fn("13_block_rdp_inspection", "inspect_block_table"),
        group="db_search",
        # Verbatim grounding table — deterministic output, no LLM triage.
        skip_subagent=True,
    ),
    ToolEntry(
        "search_system_registry",
        _get_fn("2_system_registry_search", "search_system_registry"),
        group="db_search",
        adaptive_condense=True,
    ),
    ToolEntry(
        "resolve_compound_ids",
        _get_fn("_id_alignment_search", "resolve_compound_ids_tool"),
        group="db_search",
    ),

    # ── block_tools (analysis deterministic tools) ──────────
    ToolEntry(
        "inspect_block", inspect_block,
        group="block_tools",
        skip_subagent=True,
    ),
    ToolEntry(
        "get_pure_values", get_pure_values,
        group="block_tools",
        skip_subagent=True,
    ),

    # ── fitting ─────────────────────────────────────────────
    ToolEntry(
        "fit_block", fit_block,
        group="fitting",
        skip_subagent=True,
    ),

    # ── validation ──────────────────────────────────────────
    ToolEntry(
        "validate_dual_basis_block", validate_dual_basis_block,
        group="validation",
        skip_compactor=True,
        skip_subagent=True,
    ),
    # ── estimation (last resort, artifact-backed) ─────────────────
    ToolEntry(
        "register_estimated_bridge", register_estimated_bridge,
        group="estimation",
        skip_compactor=True,
        skip_subagent=True,
    ),]


class CompAlignmentCatalog(AgentToolCatalog):
    """Composition-alignment L1 catalog — search + inspect + fit +
    dual-basis validation.

    Compactors are auto-resolved from a UNION of the query agent's
    compactor catalog (search tools) and the analysis agent's compactor
    catalog (inspect/fit/pure-value tools).
    """

    pipeline_label = "analysis"

    def __init__(self) -> None:
        # Gate off: the ledger's own tools are not offered at all.
        entries = without_ledger_tools(_TOOL_ENTRIES)
        tool_names = {e.name for e in entries}
        union = {}
        for cat in (_QUERY_COMPACTORS, _ANALYSIS_COMPACTORS):
            for e in cat.entries.values():
                if e.tool_name in tool_names and e.tool_name not in union:
                    union[e.tool_name] = e
        super().__init__(
            compactor_catalog=CompactorCatalog(union.values()),
            client_factory=_comp_client_factory,
            cfg=_cfg,
        )
        self.register_many(entries)


# ── Singleton catalog instance ─────────────────────────────────────────────
_comp_catalog: CompAlignmentCatalog | None = None


def _get_catalog() -> CompAlignmentCatalog:
    global _comp_catalog
    if _comp_catalog is None:
        _comp_catalog = CompAlignmentCatalog()
    return _comp_catalog


# ═══════════════════════════════════════════════════════════════════════════
#  Public dispatcher
# ═══════════════════════════════════════════════════════════════════════════

@with_engine_config(_cfg)
def dispatch_composition_alignment(
    purpose: str,
    instruction: str = "",
    id_catalog: str = "",
    context: str = "",
    hooks: AnalysisAgentHooks | None = None,
) -> str:
    """Dispatch the composition-alignment L1 worker agent.

    Parameters
    ----------
    purpose : str
        High-level goal (e.g. "Build the composition mapping for
        ethanol + water at 298.15 K").
    instruction : str
        Detailed instructions from the caller.
    id_catalog : str
        System/state description (JSON or text).
    context : str
        Prior context from working memory.

    Returns
    -------
    str
        The agent's structured JSON answer per the workflow output schema.
    """
    workflow = _load_workflow()
    catalog = _get_catalog()
    agent_hooks = hooks or build_agent_hooks()
    tools = catalog.wrapped_tools()

    # Fresh estimation registry per dispatch
    _ESTIMATION_REGISTRY.clear()

    run_health_check(
        catalog,
        actual_tools=tools,
        label="comp-L1",
    )

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

    parts = [f"## Purpose\n{purpose}"]
    if instruction:
        parts.append(f"## Instruction\n{instruction}")
    if id_catalog:
        parts.append(f"## System / State\n{id_catalog}")
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

    active_history = active_stats = False
    try:
        get_active_history_recorder()
        active_history = True
    except RuntimeError:
        pass
    try:
        get_active_recorder()
        active_stats = True
    except RuntimeError:
        pass
    if active_history != active_stats:
        raise RuntimeError(
            "TRACKING_LIFECYCLE_ERROR: history and statistics recorders must "
            "be active together"
        )

    owns_tracking = not active_history
    if owns_tracking:
        anchor(
            agent_hooks.anchors.start_tracking,
            agent_hooks.engine_hooks,
            agent="analysis-composition-L1",
            prompt=user_message,
        )

    status = "ERROR"
    # Shared-recorder runs record this worker's tool calls in a nested
    # history section claimed by the align_compositions relay row
    _rec = None
    section = None
    if not owns_tracking:
        try:
            _rec = get_active_history_recorder()
        except RuntimeError:
            _rec = None
        if _rec is not None:
            section = _rec.begin_nested_section("L1-align",
                                                "align_compositions")
    try:
        client = AnalysisClient.for_l1_data()
        result = agent_turn(
            user_message,
            system_prompt=system_prompt,
            tools=tools,
            memory=[],
            client=client,
            max_iterations=_cfg.L1_COMP_MAX_ITERATIONS,
            timeout=_cfg.L1_COMP_MAX_SECONDS,
            compaction_interval=_cfg.COMPACTION_INTERVAL,
            hooks=agent_hooks.engine_hooks,
            is_subagent=True,
        )

        log.info(
            "comp-align L1: %d iters, %.1fs, timed_out=%s, tools=%s",
            result.iterations, result.elapsed_seconds, result.timed_out,
            [t["tool"] for t in result.tool_history],
        )

        postanswer = evaluate_and_assemble_return(
            agent_answer=result.answer,
            postassembly_checker=postanswer_checker_for_tools(tools),
            core_claims_system_suffix=core_claims_source_suffix(tools),
            response_json_schema=workflow["response_json_schema"],
            client=client,
            label="comp-L1",
            tool_history=result.tool_history,
            task_context=user_message,
            hooks=agent_hooks,
            deterministic_fields={
                "data_inspections": export_inspections(
                    harvest_inspections(result.tool_history)
                ),
            },
        )
        status = "TIMEOUT" if result.timed_out else "OK"
        return postanswer.to_json()
    finally:
        if section is not None:
            _rec.end_nested_section(section)
        if owns_tracking:
            anchor(
                agent_hooks.anchors.finalize_tracking,
                agent_hooks.engine_hooks,
                status=status,
            )
