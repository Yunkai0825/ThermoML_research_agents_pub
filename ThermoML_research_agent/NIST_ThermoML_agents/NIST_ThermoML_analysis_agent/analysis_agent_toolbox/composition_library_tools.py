"""Composition alignment tool (align_compositions) — L1-agent-driven.
======================================================================
Dispatches the COMPOSITION ALIGNMENT L1 agent (a chemist reviewer that
surveys, deduplicates, fits, and cross-validates ALL literature sources
for a binary system) and deterministically assembles its verdict into:

1. The COMPOSITION LIBRARY — molar masses from block formulas, the
   chosen literature-backed ρ(x) bridge (refit deterministically from
   the agent's chosen doi/block — the agent chooses, code computes),
   all bridge candidates with the agent's reasons, cross-validation
   evidence, and any artifact-backed estimates.
2. The STANDARD COMPOSITION DATA SHEET — a translation table on a
   mole-fraction grid with every convertible basis as a column, each
   column carrying its conversion relation, constants, and literature
   source (``composition_translation_table.csv`` + per-column metadata
   inside the library).

The library is registered for the fitting tools' basis gate
(composition_transforms.set_active_library), saved as
``composition_library.json`` in the session data dir (self-contained —
the answer display can re-plot CSVs in any convention from it), and
documented in working memory.

Non-blocking doctrine: alignment NEVER blocks the pipeline. Exact-basis
conversions inside fit_block need only block formulas; a missing bridge
merely leaves density-dependent blocks unusable (reported, not fatal).
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np

from ThermoML_raw_json_to_card_db_parsers.id_schema import (
    lit_num_id_for_doi,
    require_block_id,
    require_block_local_id,
    require_global_id,
)

from ...general_db_query_engine.general_text_context_marker_catalog import (
    mark_subagent_answer_tool,
)
from ...general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog import (
    ToolEntry,
    uses_compactors,
)
from ..analysis_agent_context_hooks.compactor_hooks._tool_compactors import (
    compact_align_compositions,
)
from ..analysis_agent_context_hooks.hook_catalog import get_session
from ..ThermoML_core_calc_tools.mixture_nonideality_calc.composition_transforms import (
    eval_bridge_y,
    mole_to_mass,
    mole_to_volume_fraction,
    set_active_library,
)
from ..ThermoML_core_calc_tools.property_response_preparation import (
    PropertyResponsePreparationError,
    prepare_property_response,
)

_log = logging.getLogger("COMP-ALIGN")

_GRID = np.round(np.arange(0.0, 1.0000001, 0.025), 6)


def _molar_masses_from_block(
    doi: str, block_number: str, BLKsubsys_id: str | None,
) -> dict[str, float]:
    """Molar masses from a block's own compound formulas."""
    from ..ThermoML_core_calc_tools.csv_io_helpers import extract_block_arrays
    from .fitting_tools import _formula_weight
    out: dict[str, float] = {}
    try:
        bd = extract_block_arrays(
            doi, block_number, BLKsubsys_id=BLKsubsys_id,
        )
        if not bd.ok:
            return out
        for c in bd.metadata.get("compounds", []) or []:
            if not isinstance(c, dict):
                continue
            nm = c.get("name", "")
            mw = _formula_weight(c.get("formula") or "")
            if nm and mw:
                out[nm] = mw
    except Exception as exc:  # noqa: BLE001
        _log.warning("Molar-mass extraction failed for %s %s: %s",
                     doi, block_number, exc)
    return out


def _resolve_pure_endpoints(
    pure_srcs: dict, state: dict, t_tol: float = 0.5,
    purity_min: float = 0.999,
) -> tuple[dict, dict]:
    """Resolve agent-chosen pure-endpoint sources (DoF-2) to values.

    ``pure_srcs`` maps compound name → {doi, block_number, ...}. Values
    are extracted deterministically from TRULY PURE rows (x ≥ 0.999 —
    a row at x=0.98 is a mixture, and averaging an edge window biases
    the endpoint), RESTRICTED to the requested temperature (±t_tol K)
    and pressure (state pressure ±5 kPa, or ambient ≤200 kPa when the
    question does not specify one — T×P grid blocks repeat each state
    at several pressures). A compound with no measured pure row at the
    state gets NO pin (the shape fit's own edges apply, and physical
    validation still guards).
    """
    from ..ThermoML_core_calc_tools.csv_io_helpers import (
        extract_block_arrays,
        identify_columns,
    )
    from ..ThermoML_core_calc_tools.mixture_nonideality_calc.composition_transforms import (
        detect_composition_basis,
    )
    T_target = state.get("temperature_k")
    values: dict[str, float] = {}
    prov: dict[str, dict] = {}
    for comp, src in pure_srcs.items():
        expected = {
            "doi", "lit_num_id", "block_number", "BLKsubsys_id",
            "endpoint_quality",
        }
        if not isinstance(src, dict) or set(src) != expected:
            raise ValueError(
                f"pure endpoint source for {comp!r} must contain exactly "
                f"{sorted(expected)}"
            )
        doi = src["doi"]
        lit_num_id = require_global_id("lit_num_id", src["lit_num_id"])
        expected_lit_num_id = lit_num_id_for_doi(doi)
        if lit_num_id != expected_lit_num_id:
            raise ValueError(
                f"pure endpoint DOI {doi!r} resolves to "
                f"{expected_lit_num_id!r}, not {lit_num_id!r}"
            )
        blk = require_block_id(src["block_number"])
        subsystem_id = (
            require_block_local_id("subsys", src["BLKsubsys_id"])
            if src["BLKsubsys_id"] is not None else None
        )
        try:
            bd = extract_block_arrays(
                doi, blk, BLKsubsys_id=subsystem_id,
                property_filter="mass_density",
            )
            if not bd.ok:
                prov[comp] = {"doi": doi,
                              "block_number": blk, "BLKsubsys_id": subsystem_id,
                              "rejected": f"extraction failed: {bd.error}"}
                continue
            # Density column directly — identify_columns requires a
            # composition axis, which dedicated pure blocks lack.
            y_col = next((c for c in bd.columns
                          if "mass_density" in c.lower()), None)
            if y_col is None:
                prov[comp] = {"doi": doi,
                              "block_number": blk, "BLKsubsys_id": subsystem_id,
                              "rejected": "no mass-density column"}
                continue
            y = np.asarray(bd.arrays[y_col], dtype=float)
            # Composition axis (when present) via identify_columns
            x_cols: list = []
            cc_map: dict = {}
            try:
                match = identify_columns(bd.columns, "mass_density",
                                         metadata=bd.metadata)
                if match.ok:
                    x_cols = match.x_columns or []
                    cc_map = dict(match.column_compound_map or {})
            except Exception:  # noqa: BLE001
                pass
            # Mole-fraction axis for THIS compound (direct or 1−x_other)
            x_arr = None
            for c in x_cols:
                if detect_composition_basis(c) != "mole_fraction":
                    continue
                mapped = str(cc_map.get(c, "")).lower()
                if mapped == str(comp).lower():
                    x_arr = np.asarray(bd.arrays[c], dtype=float)
                    break
            if x_arr is None:
                mf_cols = [c for c in x_cols
                           if detect_composition_basis(c) == "mole_fraction"]
                if len(mf_cols) == 1:
                    x_arr = 1.0 - np.asarray(bd.arrays[mf_cols[0]],
                                             dtype=float)
            if x_arr is None:
                # Dedicated PURE-COMPONENT block (no composition axis):
                # every row is the pure compound — accept when the block's
                # single compound matches, still T-filtered below.
                comp_names = [(c.get("name") if isinstance(c, dict)
                               else str(c))
                              for c in bd.metadata.get("compounds", []) or []]
                if (len(comp_names) == 1
                        and str(comp_names[0]).lower() == str(comp).lower()):
                    x_arr = np.ones_like(y)
                else:
                    prov[comp] = {
                        "doi": doi, "block_number": blk, "BLKsubsys_id": subsystem_id,
                        "rejected": ("no usable composition axis for this "
                                     "compound and not a single-compound "
                                     "pure block"),
                    }
                    continue
            mask = (x_arr >= purity_min) & np.isfinite(y)
            if not np.any(mask):
                mx = (float(np.nanmax(x_arr))
                      if np.any(np.isfinite(x_arr)) else float("nan"))
                prov[comp] = {
                    "doi": doi, "block_number": blk, "BLKsubsys_id": subsystem_id,
                    "rejected": (f"no truly pure row (x ≥ {purity_min}); "
                                 f"max x = {mx:.4g} — edge rows are "
                                 "mixtures, not endpoint measurements"),
                }
                continue
            t_note = "block has no temperature column"
            tcol = next((c for c in bd.columns
                         if "temperature" in c.lower()), None)
            if tcol is not None and T_target is not None:
                t_arr = np.asarray(bd.arrays[tcol], dtype=float)
                mask &= np.abs(t_arr - float(T_target)) <= t_tol
                t_note = f"rows filtered to T={T_target}±{t_tol} K"
            # Pressure guard: a pure-component block is often a PRESSURE
            # series — averaging across it corrupts the endpoint. Filter
            # to the requested pressure, or ambient when unspecified.
            p_note = None
            pcol = next((c for c in bd.columns
                         if "pressure" in c.lower()), None)
            if pcol is not None:
                p_arr = np.asarray(bd.arrays[pcol], dtype=float)
                P_target = state.get("pressure_kpa")
                if P_target is not None:
                    p_mask = np.abs(p_arr - float(P_target)) <= 5.0
                    p_note = f"P={P_target}±5 kPa"
                else:
                    p_mask = (p_arr <= 200.0) | ~np.isfinite(p_arr)
                    p_note = "ambient P (≤200 kPa; NaN rows kept)"
                if np.any(mask & p_mask):
                    mask &= p_mask
                elif np.any(mask):
                    p_note += " — NO ROWS at that pressure; source rejected"
                    prov[comp] = {
                        "doi": doi, "block_number": blk, "BLKsubsys_id": subsystem_id,
                        "rejected": ("near-pure rows exist only at other "
                                     f"pressures ({p_note})"),
                    }
                    continue
            if not np.any(mask):
                prov[comp] = {
                    "doi": doi, "block_number": blk, "BLKsubsys_id": subsystem_id,
                    "rejected": ("pure rows exist but none at the requested "
                                 "temperature — not pinned"),
                }
                continue

            # A density endpoint is not usable until the PCS presentation has
            # been reduced to absolute density.  Build the selected-row mole-
            # fraction map by global compound ID; direct presentations pass
            # through unchanged, while unresolved relative responses reject
            # this endpoint source instead of contaminating the bridge.
            compounds = [
                item for item in bd.metadata.get("compounds", [])
                if isinstance(item, dict)
            ]
            id_by_name = {
                str(item.get("name", "")).strip().casefold(): item.get("comp_num_id")
                for item in compounds
                if isinstance(item.get("comp_num_id"), str)
            }
            fractions: dict[str, np.ndarray] = {}
            for column in x_cols:
                if detect_composition_basis(column) != "mole_fraction":
                    continue
                name = str(cc_map.get(column, "")).strip().casefold()
                comp_num_id = id_by_name.get(name)
                if comp_num_id:
                    fractions[comp_num_id] = np.asarray(
                        bd.arrays[column], dtype=float
                    )[mask]
            if not fractions and len(compounds) == 1:
                only_id = compounds[0].get("comp_num_id")
                if isinstance(only_id, str):
                    fractions[only_id] = np.ones(int(np.count_nonzero(mask)))
            if len(compounds) == 2 and len(fractions) == 1:
                missing_ids = {
                    item.get("comp_num_id") for item in compounds
                    if isinstance(item.get("comp_num_id"), str)
                }.difference(fractions)
                if len(missing_ids) == 1:
                    fractions[missing_ids.pop()] = 1.0 - next(iter(fractions.values()))
            response_state = {
                "temperature_k": float(T_target) if T_target is not None else None,
                "pressure_kpa": (
                    float(state["pressure_kpa"])
                    if state.get("pressure_kpa") is not None else None
                ),
            }
            try:
                response = prepare_property_response(
                    metadata=bd.metadata,
                    y_column=y_col,
                    reported_values=y[mask],
                    normalized_composition=fractions,
                    state=response_state,
                    state_tolerances={
                        "temperature_k": t_tol,
                        "pressure_kpa": 5.0,
                    },
                )
            except PropertyResponsePreparationError as exc:
                prov[comp] = {
                    "doi": doi,
                    "lit_num_id": lit_num_id,
                    "block_number": blk,
                    "BLKsubsys_id": subsystem_id,
                    "rejected": f"{exc.code}: {exc}",
                }
                continue
            values[comp] = round(float(np.mean(response.values)), 6)
            entry = {
                "doi": doi, "lit_num_id": lit_num_id,
                "block_number": blk, "BLKsubsys_id": subsystem_id,
                "value": values[comp],
                "edge_max_x": round(float(np.max(x_arr[mask])), 4),
                "n_near_pure": int(np.count_nonzero(mask)),
                "t_note": t_note,
                "property_response": response.trace,
            }
            if p_note:
                entry["p_note"] = p_note
            if tcol is not None:
                t_arr = np.asarray(bd.arrays[tcol], dtype=float)
                entry["temperature_K"] = round(float(np.mean(t_arr[mask])), 2)
            prov[comp] = entry
        except Exception as exc:  # noqa: BLE001
            _log.warning("Endpoint source %s %s failed: %s", doi, blk, exc)
    return values, prov


def _validate_bridge(bridge: dict) -> dict:
    """Physical validation of a ρ(x) bridge over the FULL grid.

    R² on the fitted points says nothing about behaviour outside the
    data window — a high-order RK through a narrow/gappy subset can go
    negative mid-range. Checks: finite, strictly positive, endpoints
    positive, and excess magnitude bounded (mixture density stays
    within 25% of the linear baseline — real excess densities are a
    few percent).
    """
    try:
        p0 = float(bridge["pure_at_x0"])
        p1 = float(bridge["pure_at_x1"])
        rho = eval_bridge_y(bridge, _GRID)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "checks": {"evaluable": False},
                "detail": str(exc)}
    lin = _GRID * p1 + (1.0 - _GRID) * p0
    scale = 0.5 * (abs(p0) + abs(p1))
    checks = {
        "endpoints_positive": bool(p0 > 0 and p1 > 0),
        "all_finite": bool(np.all(np.isfinite(rho))),
        "all_positive": bool(np.all(rho > 0)),
    }
    max_exc = (float(np.max(np.abs(rho - lin))) / scale
               if scale > 0 and checks["all_finite"] else float("inf"))
    checks["excess_bounded_25pct"] = bool(max_exc <= 0.25)
    return {
        "ok": all(checks.values()),
        "max_excess_of_mean_pct": round(100.0 * max_exc, 2)
        if np.isfinite(max_exc) else None,
        "checks": checks,
    }


def _fit_density_bridge(
    doi: str, block_number: str, BLKsubsys_id: str | None, state: dict,
    endpoint_values: dict[str, float] | None = None,
) -> dict | None:
    """RK-fit a measured density block at *state* → bridge entry or None.

    Deterministic refit of the L1 agent's CHOSEN literature sources.
    Two-step: a discovery fit establishes the component orientation,
    then — when the agent chose pure-endpoint sources (DoF-2) — the fit
    is repeated with the endpoints PINNED so the RK part only carries
    the excess shape (DoF-3) and cannot corrupt the endpoints.

    STATE RIGOR: constrains temperature AND pressure. T×P grid blocks
    repeat each isotherm at several pressures — pooling them corrupts
    the shape. When the question gives no pressure, the dominant
    ambient pressure (≤200 kPa) in the block is constrained; a block
    with only elevated pressures at the state is rejected.
    """
    from .fitting_tools import fit_block
    from ..ThermoML_core_calc_tools.csv_io_helpers import extract_block_arrays

    constrained = {}
    if state.get("temperature_k") is not None:
        constrained["temperature_k"] = {"value": float(state["temperature_k"])}

    # Pressure handling — detect the block's pressure column (may be
    # component-scoped, e.g. pressure_kpa_<ethanol>).
    p_constraint_note = None
    try:
        bd = extract_block_arrays(
            doi, block_number, BLKsubsys_id=BLKsubsys_id,
            property_filter="mass_density",
        )
        pcol = next((c for c in (bd.columns if bd.ok else [])
                     if "pressure" in c.lower()), None)
        if pcol is not None:
            p_arr = np.asarray(bd.arrays[pcol], dtype=float)
            p_vals = np.unique(p_arr[np.isfinite(p_arr)])
            P_target = state.get("pressure_kpa")
            if P_target is not None:
                constrained[pcol] = {"value": float(P_target)}
                p_constraint_note = f"P constrained to {P_target} kPa (from state)"
            elif len(p_vals) > 1:
                ambient = p_vals[p_vals <= 200.0]
                if len(ambient) == 0:
                    _log.warning(
                        "Bridge source %s %s has only elevated pressures "
                        "%s — rejected for an ambient-state bridge",
                        doi, block_number, p_vals[:5])
                    return None
                constrained[pcol] = {"value": float(ambient[0])}
                p_constraint_note = (
                    f"block is a pressure series {p_vals[:5].tolist()} kPa; "
                    f"constrained to ambient {ambient[0]:g} kPa "
                    "(no pressure in the question)")
    except Exception as exc:  # noqa: BLE001
        _log.warning("Pressure detection failed for %s %s: %s",
                     doi, block_number, exc)
        return None

    kwargs = dict(
        doi=doi,
        block_number=block_number,
        BLKsubsys_id=BLKsubsys_id,
        property_hint="mass_density",
        property_type="density",
        x_vars_constrained=constrained or None,
    )
    res = fit_block(**kwargs)
    if not isinstance(res, dict) or "error" in res or "rk_coeffs" not in res:
        return None
    required_fit_fields = {
        "doi", "block_number", "BLKsubsys_id", "components", "pure_values", "rk_coeffs",
        "r_squared", "n_mixture_points", "pure_values_source",
    }
    missing_fit_fields = required_fit_fields - res.keys()
    if missing_fit_fields:
        raise ValueError(
            f"fit_block density bridge result missing {sorted(missing_fit_fields)}"
        )
    if (
        res["doi"] != doi
        or require_block_id(res["block_number"]) != block_number
        or res["BLKsubsys_id"] != BLKsubsys_id
    ):
        raise ValueError("fit_block density bridge result identity does not match request")
    comps = res["components"]
    pv = res["pure_values"]
    if len(comps) < 2 or comps[0] not in pv or comps[1] not in pv:
        return None

    pinned: list[str] = []
    pin_note = None
    ev = endpoint_values or {}
    pin1 = ev.get(comps[0])
    pin0 = ev.get(comps[1])
    # Pin only when both endpoints are literature-resolved; a partial
    # component-name mapping is invalid under the strict fitting contract.
    if pin1 is not None and pin0 is not None:
        res2 = fit_block(
            **kwargs,
            pure_values={comps[0]: pin1, comps[1]: pin0},
        )
        if (isinstance(res2, dict) and "error" not in res2
                and "rk_coeffs" in res2):
            res = res2
            pv = res["pure_values"]
            pinned = [comps[0], comps[1]]
    elif pin1 is not None or pin0 is not None:
        pin_note = (
            "only one pure endpoint was literature-resolved at the state — "
            "endpoints NOT pinned (fit_block needs both); block-edge "
            "endpoints used, physical validation applies"
        )

    entry = {
        "property": "mass_density",
        "doi": doi,
        "block_number": require_block_id(res["block_number"]),
        "BLKsubsys_id": BLKsubsys_id,
        "x_compound": comps[0],
        "coeffs": res["rk_coeffs"],
        "pure_at_x1": float(pv[comps[0]]),
        "pure_at_x0": float(pv[comps[1]]),
        "temperature_K": (
            res["selected_temperature_K"]
            if "selected_temperature_K" in res else None
        ),
        "r_squared": res["r_squared"],
        "n_mixture_points": res["n_mixture_points"],
        "pure_values_source": res["pure_values_source"],
        "endpoints_pinned": pinned,
        "estimated": False,
        "route": ("measured-bridge (endpoint-pinned RK refit of the "
                  "agent-chosen sources)" if pinned else
                  "measured-bridge (RK refit of the agent-chosen density "
                  "block; endpoints from block edges)"),
        "output_files": res["output_files"] if "output_files" in res else {},
    }
    if pin_note:
        entry["pin_note"] = pin_note
    if p_constraint_note:
        entry["pressure_note"] = p_constraint_note
    return entry


def _parse_agent_answer(answer: str) -> dict:
    """Parse the exact JSON document emitted by the guarded L1 agent."""
    from ...general_db_query_engine.general_argo_engine_helpers.json_answer_guard import (
        clean_json_answer,
    )
    obj = clean_json_answer(answer)
    if not isinstance(obj, dict):
        raise ValueError("composition-alignment agent did not return an exact JSON object")
    return obj


def _validate_doi_pair(record: dict, *, context: str) -> None:
    """Require one exact DOI↔GLOBlit pair in an alignment record."""
    doi = record["doi"]
    lit_num_id = require_global_id("lit_num_id", record["lit_num_id"])
    expected = lit_num_id_for_doi(doi)
    if lit_num_id != expected:
        raise ValueError(
            f"{context} DOI {doi!r} resolves to {expected!r}, not "
            f"{lit_num_id!r}"
        )


def _normalize_alignment_target(
    record: dict,
    *,
    fields: set[str],
    context: str,
) -> dict:
    """Copy one agent-emitted target and enrich an omitted parent selector."""
    if not isinstance(record, dict):
        raise TypeError(f"{context} must be an object")
    required = fields - {"BLKsubsys_id"}
    missing = sorted(required - set(record))
    unknown = sorted(set(record) - fields)
    if missing or unknown:
        raise ValueError(
            f"{context} has missing={missing} and unknown={unknown} fields"
        )
    normalized = dict(record)
    normalized.setdefault("BLKsubsys_id", None)
    if normalized["BLKsubsys_id"] is not None:
        require_block_local_id("subsys", normalized["BLKsubsys_id"])
    return normalized


def _validate_alignment_report(report: dict, *, system: list[str], state: dict) -> dict:
    """Validate the exact L1 composition-alignment answer contract."""
    required = {
        "answer", "core_claims", "status", "system", "state", "basis_inventory",
        "bridge_candidates", "chosen_sources", "estimated_bridge_used",
        "cross_validation", "unsupported_conversions", "caveats",
        "data_inspections",
    }
    missing = sorted(required - set(report))
    unknown = sorted(set(report) - required)
    if missing or unknown:
        raise ValueError(
            f"composition-alignment answer has missing={missing} and unknown={unknown} fields"
        )
    if report["status"] not in {"success", "partial", "no_data", "not_binary"}:
        raise ValueError(f"invalid composition-alignment status: {report['status']!r}")
    if not isinstance(report["answer"], str) or not report["answer"].strip():
        raise TypeError("composition-alignment answer must be non-empty text")
    if (
        not isinstance(report["core_claims"], list)
        or not report["core_claims"]
        or any(
            not isinstance(claim, str) or not claim.strip()
            for claim in report["core_claims"]
        )
    ):
        raise TypeError(
            "composition-alignment core_claims must be an array of non-empty strings"
        )
    if report["system"] != system or report["state"] != state:
        raise ValueError(
            "composition-alignment answer system/state does not match the dispatched request"
        )
    for field in (
        "basis_inventory", "bridge_candidates", "unsupported_conversions", "caveats",
        "data_inspections",
    ):
        if not isinstance(report[field], list):
            raise TypeError(f"composition-alignment answer {field} must be an array")
    if not isinstance(report["estimated_bridge_used"], bool):
        raise TypeError("estimated_bridge_used must be boolean")
    cross_validation = report["cross_validation"]
    if not isinstance(cross_validation, dict) or set(cross_validation) != {
        "pure_endpoint_agreement", "bridge_agreement", "dual_basis_checks"
    }:
        raise ValueError(
            "cross_validation must contain exactly pure_endpoint_agreement, "
            "bridge_agreement, and dual_basis_checks"
        )
    if not isinstance(cross_validation["dual_basis_checks"], list):
        raise TypeError("cross_validation.dual_basis_checks must be an array")
    normalized_inventory = []
    for index, raw_entry in enumerate(report["basis_inventory"]):
        entry = _normalize_alignment_target(
            raw_entry,
            fields={
            "doi", "lit_num_id", "block_number", "BLKsubsys_id",
            "bases", "properties",
            "n_datapoints", "solvents",
            },
            context=f"basis_inventory[{index}]",
        )
        _validate_doi_pair(entry, context=f"basis_inventory[{index}]")
        require_block_id(entry["block_number"])
        if not isinstance(entry["bases"], list) or not isinstance(entry["properties"], list):
            raise TypeError(f"basis_inventory[{index}] bases/properties must be arrays")
        if not isinstance(entry["solvents"], list):
            raise TypeError(f"basis_inventory[{index}].solvents must be an array")
        normalized_inventory.append(entry)
    report["basis_inventory"] = normalized_inventory
    candidate_fields = {
        "doi", "lit_num_id", "block_number", "BLKsubsys_id",
        "temperature_K", "r_squared",
        "n_mixture_points", "endpoint_quality", "considered",
    }
    normalized_candidates = []
    for index, raw_candidate in enumerate(report["bridge_candidates"]):
        candidate = _normalize_alignment_target(
            raw_candidate,
            fields=candidate_fields,
            context=f"bridge_candidates[{index}]",
        )
        _validate_doi_pair(candidate, context=f"bridge_candidates[{index}]")
        require_block_id(candidate["block_number"])
        if candidate["endpoint_quality"] not in {"direct_measured", "incomplete"}:
            raise ValueError(
                f"bridge_candidates[{index}].endpoint_quality must be direct_measured or incomplete"
            )
        normalized_candidates.append(candidate)
    report["bridge_candidates"] = normalized_candidates
    chosen = report["chosen_sources"]
    if chosen is not None:
        if not isinstance(chosen, dict) or set(chosen) != {
            "pure_density", "excess_shape", "reason"
        }:
            raise ValueError(
                "chosen_sources must be null or contain exactly pure_density, excess_shape, reason"
            )
        if not isinstance(chosen["reason"], str) or not chosen["reason"]:
            raise TypeError("chosen_sources.reason must be a non-empty string")
        chosen = dict(chosen)
        shape = _normalize_alignment_target(
            chosen["excess_shape"],
            fields={"doi", "lit_num_id", "block_number", "BLKsubsys_id"},
            context="chosen_sources.excess_shape",
        )
        _validate_doi_pair(shape, context="chosen_sources.excess_shape")
        require_block_id(shape["block_number"])
        pure = chosen["pure_density"]
        if not isinstance(pure, dict) or set(pure) != set(system):
            raise ValueError("chosen_sources.pure_density keys must exactly match system names")
        normalized_pure = {}
        for component, raw_source in pure.items():
            source = _normalize_alignment_target(
                raw_source,
                fields={
                    "doi", "lit_num_id", "block_number", "BLKsubsys_id",
                    "endpoint_quality",
                },
                context=f"chosen_sources.pure_density[{component!r}]",
            )
            _validate_doi_pair(
                source,
                context=f"chosen_sources.pure_density[{component!r}]",
            )
            require_block_id(source["block_number"])
            if source["endpoint_quality"] != "direct_measured":
                raise ValueError(
                    f"chosen_sources.pure_density[{component!r}] must be direct_measured"
                )
            normalized_pure[component] = source
        chosen["excess_shape"] = shape
        chosen["pure_density"] = normalized_pure
        report["chosen_sources"] = chosen
    return report


def _build_translation_table(
    system: list[str],
    molar_masses: dict[str, float],
    bridge: dict | None,
) -> tuple[str, list[dict]] | None:
    """Materialize the standard composition data sheet.

    Grid of mole fractions → every convertible basis as a column, each
    with relation, constants, and source metadata. Returns
    ``(csv_text, columns_meta)`` or None when molar masses are missing.
    """
    # Reference compound = bridge x_compound when a bridge exists
    # (so ρ(x) evaluates directly), else the first system compound.
    if bridge and bridge.get("x_compound") in molar_masses:
        A = bridge["x_compound"]
    elif system and system[0] in molar_masses:
        A = system[0]
    else:
        return None
    others = [c for c in molar_masses if c != A]
    if not others:
        return None
    B = others[0]
    MA, MB = molar_masses[A], molar_masses[B]

    x = _GRID
    w = mole_to_mass(x, MA, MB)
    with np.errstate(divide="ignore", invalid="ignore"):
        molality = 1000.0 * x / ((1.0 - x) * MB)          # mol/kg solvent B
        amount_ratio = x / (1.0 - x)
        mass_ratio = w / (1.0 - w)

    formula_src = {
        "kind": "block_formulas",
        "detail": f"M({A})={MA:g}, M({B})={MB:g} g/mol from the blocks' "
                  "own chemical formulas",
    }
    cols: list[tuple[str, np.ndarray]] = [
        (f"mole_fraction[{A}]", x),
        (f"mass_fraction[{A}]", w),
        (f"molality[{A}]_mol_per_kg[{B}]", molality),
        (f"amount_ratio[{A}]_per_[{B}]", amount_ratio),
        (f"mass_ratio[{A}]_per_[{B}]", mass_ratio),
    ]
    meta: list[dict] = [
        {"column": f"mole_fraction[{A}]", "basis": "mole_fraction",
         "relation": "grid axis (canonical fitting basis)",
         "source": {"kind": "definition"}},
        {"column": f"mass_fraction[{A}]", "basis": "mass_fraction",
         "relation": "w1 = x1·M1 / (x1·M1 + (1−x1)·M2) — exact",
         "source": formula_src},
        {"column": f"molality[{A}]_mol_per_kg[{B}]", "basis": "molality",
         "relation": "b = 1000·x1 / ((1−x1)·M2) — exact; solvent = "
                     f"{B}; diverges at x1→1",
         "source": formula_src},
        {"column": f"amount_ratio[{A}]_per_[{B}]", "basis": "amount_ratio",
         "relation": "nr = x1/(1−x1) — exact, no constants",
         "source": {"kind": "definition"}},
        {"column": f"mass_ratio[{A}]_per_[{B}]", "basis": "mass_ratio",
         "relation": "r = w1/(1−w1) — exact",
         "source": formula_src},
    ]

    if bridge:
        rho = eval_bridge_y(bridge, x)
        mbar = x * MA + (1.0 - x) * MB
        molarity = x * rho / mbar                # mol/L (ρ kg/m³, M g/mol)
        mass_conc = molarity * MA                # kg/m³ (= g/L)
        V1 = MA / float(bridge["pure_at_x1"])    # ratio-consistent units
        V2 = MB / float(bridge["pure_at_x0"])
        phi = mole_to_volume_fraction(x, V1, V2)
        bridge_src = {
            "kind": "estimated_bridge" if bridge.get("estimated")
                    else "measured_bridge",
            "doi": bridge["doi"],
            "block_number": bridge["block_number"],
            "BLKsubsys_id": bridge["BLKsubsys_id"],
            "temperature_K": bridge.get("temperature_K"),
            "r_squared": bridge.get("r_squared"),
            "route": bridge.get("route", ""),
        }
        if bridge.get("estimated"):
            bridge_src["justification"] = bridge.get("justification", "")
            bridge_src["references"] = bridge.get("references", [])
        cols += [
            ("mass_density_kg_m3", rho),
            (f"molarity[{A}]_mol_per_L", molarity),
            (f"mass_concentration[{A}]_kg_m3", mass_conc),
            (f"volume_fraction[{A}]", phi),
        ]
        est_tag = (" — ESTIMATED bridge, see source"
                   if bridge.get("estimated") else "")
        meta += [
            {"column": "mass_density_kg_m3", "basis": "(bridge)",
             "relation": "ρ(x) = x·ρ1 + (1−x)·ρ0 + x(1−x)·ΣAk(2x−1)^k"
                         + est_tag,
             "source": bridge_src},
            {"column": f"molarity[{A}]_mol_per_L", "basis": "molarity",
             "relation": "c = x·ρ(x) / (x·M1 + (1−x)·M2)" + est_tag,
             "source": bridge_src},
            {"column": f"mass_concentration[{A}]_kg_m3",
             "basis": "mass_concentration",
             "relation": "c_mass = c·M1" + est_tag, "source": bridge_src},
            {"column": f"volume_fraction[{A}]", "basis": "volume_fraction",
             "relation": "φ1 = x·V1/(x·V1+(1−x)·V2), Vi = Mi/ρi,pure from "
                         "the bridge endpoints; pre-mixing convention"
                         + est_tag,
             "source": bridge_src},
        ]

    header = ",".join(name for name, _ in cols)
    lines = [header]
    for i in range(len(x)):
        row = []
        for _, arr in cols:
            v = arr[i]
            row.append(f"{v:.6g}" if np.isfinite(v) else "")
        lines.append(",".join(row))
    return "\n".join(lines) + "\n", meta


@mark_subagent_answer_tool
@uses_compactors(compact_align_compositions)
def align_compositions(
    system: list[str],
    state: dict | None = None,
    instruction: str = "",
    purpose: str = "",
) -> dict:
    """Run the COMPOSITION ALIGNMENT L1 agent and assemble its verdict
    into the composition library + standard composition data sheet.

    The L1 agent (a chemist reviewer) surveys ALL literature blocks for
    the binary system, deduplicates sources, fits every viable density
    bridge candidate at the state, cross-validates them (including
    against papers' own dual-basis tables), and chooses the bridge with
    a documented literature reason — or registers an artifact-backed
    estimate when the database has no source. This assembler then refits
    the chosen source deterministically and materializes the translation
    table (mole-fraction grid × every convertible basis, each column
    with relation + constants + source).

    Alignment never blocks the pipeline: exact-basis conversions in
    fit_block need only block formulas.

    Parameters
    ----------
    system : list[str]
        Exactly two compound names, e.g. ``["ethanol", "water"]``.
    state : dict | None
        State object, e.g. ``{"temperature_k": 298.15}`` or
        ``{"temperature_k": 298.15, "pressure_kpa": 100}``.
        Include pressure_kpa whenever the question specifies one —
        otherwise ambient (≤200 kPa) is assumed for endpoint rows and
        pressure-series blocks.
    instruction : str
        Extra instructions forwarded to the L1 agent (optional), e.g.
        known DOIs to include or bases the question cares about.
    """
    if not isinstance(system, list) or len(system) != 2:
        raise TypeError("system must be an array of exactly two compound names")
    if any(not isinstance(name, str) or not name.strip() for name in system):
        raise TypeError("both system entries must be non-empty strings")
    if state is None:
        state = {}
    if not isinstance(state, dict):
        raise TypeError("state must be an object")
    unknown_state = sorted(set(state) - {"temperature_k", "pressure_kpa"})
    if unknown_state:
        raise ValueError(f"state has unknown fields {unknown_state}")
    for field, value in state.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"state.{field} must be numeric")

    # ── Dispatch the L1 chemist-reviewer agent ──────────────
    from ..analysis_agent_workflows.L1_workers.l1_composition_alignment import (
        consume_estimations,
        dispatch_composition_alignment,
    )

    state_txt = ", ".join(f"{k}={v}" for k, v in state.items()) or "any state"
    l1_purpose = (purpose or
                  f"Build the literature-backed composition mapping for "
                  f"{system[0]} + {system[1]} at {state_txt}.")
    l1_instruction = (
        "Survey ALL blocks for the system, inventory composition bases, "
        "deduplicate sources, fit every viable density-bridge candidate "
        "at the state, cross-validate independent sources (and dual-basis "
        "blocks), then choose the bridge with a documented reason per the "
        "output schema. Estimate via register_estimated_bridge ONLY if no "
        "literature source exists."
        + (f"\nAdditional instructions from the caller: {instruction}"
           if instruction else "")
    )
    id_catalog = json.dumps({"system": system, "state": state})

    answer = dispatch_composition_alignment(
        purpose=l1_purpose,
        instruction=l1_instruction,
        id_catalog=id_catalog,
    )

    report = _validate_alignment_report(
        _parse_agent_answer(answer), system=system, state=state
    )
    estimates = consume_estimations()
    if report["estimated_bridge_used"]:
        if report["chosen_sources"] is not None or len(estimates) != 1:
            raise ValueError(
                "estimated_bridge_used=true requires chosen_sources=null and exactly one "
                "registered estimated bridge"
            )
    elif estimates:
        raise ValueError(
            "an estimated bridge was registered but estimated_bridge_used is false"
        )

    # ── Deterministic assembly of the agent's per-DoF choices ──
    # DoF-2: resolve pure-endpoint sources → pinned values
    chosen_sources = report["chosen_sources"]
    shape = chosen_sources["excess_shape"] if chosen_sources is not None else None
    pure_srcs = chosen_sources["pure_density"] if chosen_sources is not None else {}
    chosen_reason = chosen_sources["reason"] if chosen_sources is not None else None
    endpoint_values, endpoint_prov = _resolve_pure_endpoints(pure_srcs, state)

    # DoF-3: refit shape candidates (agent's chosen first, then its
    # ranked alternates) with pinned endpoints; PHYSICAL VALIDATION on
    # the full grid gates entry into the library.
    shape_candidates: list[dict] = []
    if shape is not None:
        shape_candidates.append(shape)
    for c in report["bridge_candidates"]:
        candidate_block_id = require_block_id(c["block_number"])
        if any(
            c["doi"] == source["doi"]
            and candidate_block_id == require_block_id(source["block_number"])
            and c["BLKsubsys_id"] == source["BLKsubsys_id"]
            for source in shape_candidates
        ):
            continue
        shape_candidates.append(c)

    bridge = None
    bridge_note = None
    bridge_validation: list[dict] = []
    for i, cand in enumerate(shape_candidates):
        b = _fit_density_bridge(
            cand["doi"], cand["block_number"], cand["BLKsubsys_id"], state,
            endpoint_values=endpoint_values,
        )
        if b is None:
            bridge_validation.append({
                "doi": cand["doi"],
                "block_number": cand["block_number"],
                "BLKsubsys_id": cand["BLKsubsys_id"],
                "ok": False, "detail": "refit failed",
            })
            continue
        v = _validate_bridge(b)
        bridge_validation.append({
            "doi": b["doi"], "block_number": b["block_number"],
            "BLKsubsys_id": b["BLKsubsys_id"], **v,
        })
        if v["ok"]:
            b["validation"] = v
            if endpoint_prov:
                b["pure_endpoint_sources"] = endpoint_prov
            bridge = b
            if i > 0:
                bridge_note = (
                    f"Agent's first-choice shape source failed physical "
                    f"validation (negative/unbounded ρ outside its data "
                    f"window); used its ranked alternate {b['doi']} "
                    f"{b['block_number']} instead."
                )
            break
    if bridge is None and shape_candidates:
        bridge_note = (
            "No shape candidate passed physical validation (positivity + "
            "bounded excess on the full grid) — no measured bridge."
        )
    if bridge is None and report["estimated_bridge_used"]:
        bridge = estimates[0]
        bridge_note = ("Using the agent's ESTIMATED bridge (artifact-backed, "
                       "flagged estimated=true) — no measured literature "
                       "source at this state.")

    # ── Molar masses (deterministic, from block formulas) ───
    molar_masses: dict[str, float] = {}
    mm_candidates: list[tuple[str, str, str | None]] = []
    if bridge and bridge["doi"] is not None:
        mm_candidates.append(
            (bridge["doi"], bridge["block_number"], bridge["BLKsubsys_id"])
        )
    for entry in report["basis_inventory"]:
        mm_candidates.append(
            (entry["doi"], entry["block_number"], entry["BLKsubsys_id"])
        )
    for candidate in report["bridge_candidates"]:
        mm_candidates.append(
            (candidate["doi"], candidate["block_number"], candidate["BLKsubsys_id"])
        )
    for doi_, blk_, subsystem_id_ in mm_candidates:
        molar_masses = _molar_masses_from_block(doi_, blk_, subsystem_id_)
        if len(molar_masses) >= 2:
            break

    inventory = report["basis_inventory"]
    basis_counts: dict[str, int] = {}
    for e in inventory:
        for bs in e["bases"]:
            basis_counts[bs] = basis_counts.get(bs, 0) + 1

    caveats = list(report["caveats"])
    for std in (
        "Conversions assume molecular (non-dissociating) species; "
        "electrolyte compositions are formal.",
        "Volume-fraction definitions vary between papers "
        "(pre-mixing pure volumes assumed).",
    ):
        if std not in caveats:
            caveats.append(std)

    # Reconcile the agent's unsupported list with the assembler outcome:
    # a successfully refit MEASURED bridge supersedes any "needs a bridge /
    # not yet fitted" entries written from the agent's mid-run perspective.
    unsupported = list(report["unsupported_conversions"])
    if bridge is not None and not bridge.get("estimated"):
        _bridge_terms = ("molarity", "volume_fraction", "volume fraction",
                         "mass_concentration", "mass concentration",
                         "density bridge")
        dropped = [u for u in unsupported
                   if any(t in str(u).lower() for t in _bridge_terms)]
        if dropped:
            unsupported = [u for u in unsupported if u not in dropped]
            caveats.append(
                "Assembler note: the chosen bridge was refit successfully "
                "after the agent's report, superseding its provisional "
                "'bridge not fitted' entries: "
                + " | ".join(str(d)[:80] for d in dropped)
            )

    library = {
        "system": system,
        "state": state,
        "alignment_status": report["status"],
        "molar_masses_g_mol": molar_masses,
        "basis_counts": basis_counts,
        "n_blocks_surveyed": len(inventory),
        "bridges": {"mass_density": bridge} if bridge else {},
        "inventory": inventory,
        "bridge_candidates": report["bridge_candidates"],
        "bridge_validation": bridge_validation,
        "chosen_sources": chosen_sources,
        "chosen_bridge_reason": chosen_reason,
        "cross_validation": report["cross_validation"],
        "unsupported_conversions": unsupported,
        "estimates": estimates,
        "caveats": caveats,
        "note": ("Alignment never blocks: exact-basis fits proceed via "
                 "molar masses even without a bridge."),
    }
    if bridge_note:
        library["bridge_note"] = bridge_note

    # ── Standard composition data sheet (translation table) ─
    table = None
    if len(molar_masses) >= 2:
        table = _build_translation_table(system, molar_masses, bridge)
    if table is None:
        library["translation_table_note"] = (
            "Translation table not generated — molar masses unavailable "
            "(no block with usable formulas for both compounds)."
        )

    # ── Register + persist ──────────────────────────────────
    set_active_library(library)
    sess = get_session()
    if sess:
        if table is not None:
            csv_text, columns_meta = table
            csv_path = sess.data_path("composition_translation_table.csv")
            with open(csv_path, "w", encoding="utf-8") as fh:
                fh.write(csv_text)
            sess.register_file(
                "data", str(csv_path),
                f"Standard composition data sheet — {system[0]} + "
                f"{system[1]} ({state_txt}); every convertible basis "
                "with per-column source metadata",
            )
            library["translation_table"] = {
                "file": str(csv_path),
                "grid": "mole fraction 0..1, step 0.025",
                "columns": columns_meta,
            }
        lib_path = sess.data_path("composition_library.json")
        with open(lib_path, "w", encoding="utf-8") as fh:
            json.dump(library, fh, indent=1, default=str)
        sess.register_file(
            "data", str(lib_path),
            f"Composition library — {system[0]} + {system[1]}",
        )
        library["library_file"] = str(lib_path)
    elif table is not None:
        # No session — keep the sheet metadata in-memory anyway
        csv_text, columns_meta = table
        library["translation_table"] = {
            "file": None,
            "grid": "mole fraction 0..1, step 0.025",
            "columns": columns_meta,
            "csv_text": csv_text,
        }

    _log.info(
        "Composition alignment: status=%s, %d blocks, bases %s, bridge=%s%s",
        library["alignment_status"], len(inventory), basis_counts,
        bool(bridge), " (ESTIMATED)" if bridge and bridge.get("estimated")
        else "",
    )
    return library


TOOL_ENTRIES = [
    ToolEntry(
        "align_compositions", align_compositions,
        group="query_delegation",
        skip_subagent=True,
    ),
]


def _alignment_tools():
    from ..analysis_agent_workflows.L1_workers.l1_composition_alignment import _get_catalog
    return _get_catalog().wrapped_tools()


from ...general_db_query_engine.general_tool_management_helpers.tool_capabilities import (
    bind_reachable_tools,
)

bind_reachable_tools(align_compositions, _alignment_tools)
