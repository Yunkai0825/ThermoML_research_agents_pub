"""Phase 1 — Data discovery, inspection, and pure-value extraction.

Each function returns a raw dict.  The catalog's ``wrap_tool`` handles
compaction (Layer 1) and agentic KEEP/DISCARD (Layer 2).
"""

from __future__ import annotations

import numpy as np

from ...general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog import (
    ToolEntry,
    uses_compactors,
)
from ..analysis_agent_context_hooks.compactor_hooks._tool_compactors import (
    compact_query_system_summary,
    compact_query_blocks,
    compact_inspect_block,
    compact_find_similar_compounds,
    compact_get_pure_values,
)
import importlib as _il
import sys
from pathlib import Path

from ..ThermoML_core_calc_tools.csv_io_helpers import (
    extract_block_arrays,
    identify_columns,
)
from ..ThermoML_core_calc_tools.mixture_nonideality_calc.ideal_baseline import (
    extract_pure_from_edges,
)
from ..ThermoML_core_calc_tools.mixture_nonideality_calc.composition_transforms import (
    detect_composition_basis,
)
from ..ThermoML_core_calc_tools.property_response_preparation import (
    PropertyResponsePreparationError,
    inspect_property_response_contract,
    prepare_property_response,
    property_response_error_result,
)

_WORKSPACE = Path(__file__).resolve().parents[3]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

search_blocks = getattr(
    _il.import_module("card_db_search_tools.basic_search_tools.1_block_search"),
    "search_blocks",
)
search_system_summary = getattr(
    _il.import_module("card_db_search_tools.basic_search_tools.9_system_summary_search"),
    "search_system_summary",
)
search_similar_compounds = getattr(
    _il.import_module("card_db_search_tools.basic_search_tools.10_compound_similarity_search"),
    "search_similar_compounds",
)
_inspect_block_table_impl = getattr(
    _il.import_module("card_db_search_tools.basic_search_tools.13_block_rdp_inspection"),
    "inspect_block_table",
)


@uses_compactors(compact_query_system_summary)
def query_system_summary(
    compounds: list[str],
    properties: list[str] | None = None,
    system_scope: str = "declared",
    limit: int = 20,
) -> dict:
    """Search ThermoML for data availability summary.

    Parameters
    ----------
    compounds : list[str]
        Exact compound names or scoped global compound IDs.
    properties : list[str] | None
        Exact property names or scoped global property IDs.
    limit : int
        Maximum results to return.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise TypeError("limit must be a positive integer and is never coerced")
    if not isinstance(compounds, list) or not compounds:
        raise TypeError("compounds must be a non-empty array")
    if properties is not None and not isinstance(properties, list):
        raise TypeError("properties must be an array when supplied")
    if system_scope not in {"declared", "subsystem", "either"}:
        raise ValueError("system_scope must be declared, subsystem, or either")

    return search_system_summary(
        compound=compounds,
        property=properties,
        system_scope=system_scope,
        limit=limit,
    )


@uses_compactors(compact_query_blocks)
def query_blocks(
    compounds: list[str],
    properties: list[str] | None = None,
    system_scope: str = "declared",
    limit: int = 50,
    temperature_range: list[float] | None = None,
    pressure_range: list[float] | None = None,
) -> dict:
    """Retrieve block-level metadata from ThermoML.

    Returns DOIs, block_numbers, compounds, variable/property lists,
    and data ranges -- NOT actual data values.

    Parameters
    ----------
    compounds : list[str]
        Exact compound names or scoped global compound IDs.
    properties : list[str] | None
        Exact property names or scoped global property IDs.
    limit : int
        Max blocks to return.
    temperature_range : list[float] | None
        Exact ``[minimum, maximum]`` range in kelvin.
    pressure_range : list[float] | None
        Exact ``[minimum, maximum]`` range in kPa.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise TypeError("limit must be a positive integer and is never coerced")
    if not isinstance(compounds, list) or not compounds:
        raise TypeError("compounds must be a non-empty array")
    if properties is not None and not isinstance(properties, list):
        raise TypeError("properties must be an array when supplied")
    if system_scope not in {"declared", "subsystem", "either"}:
        raise ValueError("system_scope must be declared, subsystem, or either")

    kwargs: dict = dict(
        compound=compounds, system_scope=system_scope, limit=limit
    )
    if properties is not None:
        kwargs["property"] = properties
    if temperature_range is not None:
        kwargs["temperature_range"] = temperature_range
    if pressure_range is not None:
        kwargs["pressure_range"] = pressure_range
    return search_blocks(**kwargs)


@uses_compactors(compact_inspect_block)
def inspect_block(
    doi: str,
    block_number: str,
    BLKsubsys_id: str | None = None,
    property_filter: str = "",
) -> dict:
    """Inspect a block's structure: column names, row count, value ranges.

    Does NOT return raw data.  Use this before fit_block to verify
    the block has the data you expect.

    Parameters
    ----------
    doi : str
        Paper DOI.
    block_number : str
        Strict typed block identifier (e.g. ``"PROPblock_2"`` or
        ``"RXNblock_2"``). Legacy ``block_N`` and bare numbers are rejected.
    property_filter : str
        Only include property columns matching this substring.
    """
    bd = extract_block_arrays(
        doi,
        block_number,
        BLKsubsys_id=BLKsubsys_id,
        property_filter=property_filter or None,
    )
    if not bd.ok:
        return {"error": bd.error, "doi": doi, "block_number": block_number, "BLKsubsys_id": BLKsubsys_id}

    match = identify_columns(bd.columns, property_filter, metadata=bd.metadata)

    raw = {
        "doi": bd.doi,
        "lit_num_id": bd.metadata["lit_num_id"],
        "block_number": bd.block_number,
        "BLKsubsys_id": bd.BLKsubsys_id,
        "block_type": bd.metadata["block_type"],
        "compounds": bd.metadata["compounds"],
        "compound_map": bd.metadata["compound_map"],
        "variables": bd.metadata["variables"],
        "properties": bd.metadata["properties"],
        "constraints": bd.metadata["constraints"],
        "solvents": bd.metadata["solvents"],
        "columns": bd.columns,
        "n_rows": bd.n_rows,
        "identified_x": match.x_columns,
        "identified_y": match.y_columns,
        "column_compound_map": match.column_compound_map,
        "column_ranges": {
            col: {
                "min": round(float(np.nanmin(arr)), 6) if np.any(~np.isnan(arr)) else None,
                "max": round(float(np.nanmax(arr)), 6) if np.any(~np.isnan(arr)) else None,
                "n_valid": int(np.count_nonzero(~np.isnan(arr))),
            }
            for col, arr in bd.arrays.items()
            if col not in {"BLKpoint_id", "_source"}
        },
    }
    return raw


def inspect_block_table(
    literature: str,
    block_number: str,
    where: str | None = None,
    nearest: dict | None = None,
    BLKsubsys_id: str | None = None,
    property_filter: str = "",
) -> dict:
    """Return the verbatim data sub-table of ONE block for answer grounding.

    MANDATORY before quoting data points: every numeric data value cited
    in an answer must appear verbatim in a table returned by this tool
    (or another tool's returned output).  Unlike ``inspect_block`` (which
    reports structure only), this returns the actual rows.  ``_U`` columns
    are expanded combined uncertainties from the source; report value ± U
    when present; say that no uncertainty is recorded only when the column
    is empty.

    Parameters
    ----------
    literature : str
        Literature identifier — the tool-returned ``GLOBlit_N`` from your
        ID catalog (preferred; never reconstruct a DOI from memory) or the
        verbatim paper DOI.
    block_number : str
        Strict typed block identifier (e.g. ``"PROPblock_2"``).
    where : str, optional
        Row filter over column aliases; AND-joined conditions only.
        Grammar: ``col <op> number`` (= != < <= > >=),
        ``col BETWEEN lo AND hi``, ``col IN (v1, v2)``.  Aliases are
        lowercase with underscores (``temperature_k``,
        ``mole_fraction_ethanol``).  Call without ``where`` once to
        discover aliases and ranges.
    nearest : dict, optional
        ``{"column": alias, "value": number, "k": 1..8}`` — the k rows
        nearest to *value* plus a bracketing/gap statement.  Use instead
        of interpolating or guessing a row near a target condition.
    BLKsubsys_id : str, optional
        Restrict to one composition subsystem.
    property_filter : str
        Keep only property columns matching this substring.

    Returns
    -------
    dict
        Verbatim ``rows_shown``, ``table_mode`` (complete | rdp | nearest
        | head), per-column ``stats``, ``bracket``, ``inspection_id``.
        In ``rdp`` mode quote only shown rows or stats; narrow ``where``.
    """
    return _inspect_block_table_impl(
        block_number,
        literature=literature,
        where=where,
        nearest=nearest,
        BLKsubsys_id=BLKsubsys_id,
        property_filter=property_filter or None,
    )


@uses_compactors(compact_find_similar_compounds)
def find_similar_compounds(
    compound: str,
    top_k: int = 10,
    min_similarity: float = 0.5,
) -> dict:
    """Find structurally similar compounds from ThermoML database.

    Parameters
    ----------
    compound : str
        Compound name, SMILES, or InChI.
    top_k : int
        Number of results.
    min_similarity : float
        Minimum Tanimoto similarity threshold.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise TypeError("top_k must be a positive integer and is never coerced")
    if isinstance(min_similarity, bool) or not isinstance(min_similarity, (int, float)):
        raise TypeError("min_similarity must be numeric and is never coerced")
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between 0 and 1")

    return search_similar_compounds(
        name=compound,
        top_k=top_k,
        min_similarity=min_similarity,
    )


@uses_compactors(compact_get_pure_values)
def get_pure_values(
    doi: str,
    block_number: str,
    property_hint: str,
    BLKsubsys_id: str | None = None,
    composition_hint: str = "mole_fraction",
    edge_threshold: float = 0.02,
    x_vars_constrained: dict | None = None,
) -> dict:
    """Extract pure-component property values from a ThermoML block.

    Uses *edge extraction*: for each component, finds data points where
    that component's mole fraction is near 1.0 and reads the measured
    property value there.  Works for binary, ternary, and higher systems.

    Components without a directly observed point inside *edge_threshold*
    are omitted. No endpoint substitution or extrapolation is performed.

    Parameters
    ----------
    doi : str
        Paper DOI.
    block_number : str
        Block identifier.
    property_hint : str
        Substring to match the property column (e.g. "density").
    composition_hint : str
        Substring to match composition columns (default "mole_fraction").
    edge_threshold : float
        Points with x_i > 1 - threshold are "nearly pure".
        Default 0.02 (2 %).
    x_vars_constrained : dict | None
        Optional constraint object, for example
        ``{"temperature_k": {"value": 298.15, "tol": 0.5}}``.
        Restricts edge extraction to matching rows so multi-temperature
        blocks yield state-specific pure values instead of averages
        across temperatures.

    Returns
    -------
    dict
        pure_values: {compound_name: value} (pass directly as
        ``pure_values`` to ``fit_block``), fit_block_mapping,
        edge extraction provenance, temperature, pressure, or error.
    """
    bd = extract_block_arrays(
        doi,
        block_number,
        BLKsubsys_id=BLKsubsys_id,
        property_filter=property_hint or None,
    )
    if not bd.ok:
        return {"error": bd.error, "doi": doi, "block_number": block_number, "BLKsubsys_id": BLKsubsys_id}

    if isinstance(edge_threshold, bool) or not isinstance(edge_threshold, (int, float)):
        raise TypeError("edge_threshold must be numeric and is never coerced")
    if not 0.0 <= edge_threshold < 1.0:
        raise ValueError("edge_threshold must be in [0, 1)")

    match = identify_columns(bd.columns, property_hint, composition_hint, metadata=bd.metadata)
    if not match.ok:
        err: dict = {
            "error": match.error, "doi": doi,
            "block_number": block_number, "BLKsubsys_id": BLKsubsys_id,
        }
        comps_meta_hint = bd.metadata.get("compounds")
        if (
            "No composition column" in (match.error or "")
            and isinstance(comps_meta_hint, list) and len(comps_meta_hint) == 1
        ):
            only = comps_meta_hint[0]
            name = only.get("name") if isinstance(only, dict) else None
            err["hint"] = (
                f"{block_number} is a single-compound block"
                + (f" (pure {name})" if name else "")
                + ": every row already IS a pure-component value, so edge "
                "extraction does not apply. Read the values directly with "
                "inspect_block_table and pass them to fit_block as pure_values."
            )
        return err

    y = bd.arrays[match.y_column]

    # ---- Optional state constraints (same JSON format as fit_block) ----
    applied_constraints: dict = {}
    row_mask = np.ones(len(y), dtype=bool)
    if x_vars_constrained is not None:
        if not isinstance(x_vars_constrained, dict):
            raise TypeError("x_vars_constrained must be an object")
        for c_col, c_spec in x_vars_constrained.items():
            if c_col not in bd.arrays:
                return {"error": f"Constraint column '{c_col}' not found in block",
                        "columns_available": bd.columns,
                        "doi": doi, "block_number": block_number, "BLKsubsys_id": BLKsubsys_id}
            if not isinstance(c_spec, dict) or "value" not in c_spec:
                return {"error": f"Constraint for '{c_col}' must be a dict with a 'value' key",
                        "doi": doi, "block_number": block_number, "BLKsubsys_id": BLKsubsys_id}
            if set(c_spec) - {"value", "tol"}:
                return {
                    "error": f"Constraint for '{c_col}' accepts only value and tol",
                    "doi": doi,
                    "block_number": block_number,
                    "BLKsubsys_id": BLKsubsys_id,
                }
            c_val = c_spec["value"]
            c_tol = c_spec.get("tol", 0.5)
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in (c_val, c_tol)
            ):
                raise TypeError(f"Constraint '{c_col}' value and tol must be numeric")
            row_mask &= np.abs(bd.arrays[c_col] - c_val) <= c_tol
            applied_constraints[c_col] = {"value": c_val, "tol": c_tol}
        if not np.any(row_mask):
            return {"error": "No rows match x_vars_constrained",
                    "applied_constraints": applied_constraints,
                    "doi": doi, "block_number": block_number, "BLKsubsys_id": BLKsubsys_id}
    y = y[row_mask]

    # Build composition dict from ALL identified x columns
    x_dict: dict = {}
    for xc in match.x_columns:
        if xc not in match.column_compound_map:
            raise ValueError(f"composition column {xc!r} is missing its compound mapping")
        comp_name = match.column_compound_map[xc]
        x_dict[comp_name] = bd.arrays[xc][row_mask]

    # Get canonical compound names from metadata
    if "compounds" not in bd.metadata:
        raise ValueError("block metadata is missing compounds")
    comps_meta = bd.metadata["compounds"]
    if not isinstance(comps_meta, list):
        raise TypeError("block metadata compounds must be an array")
    comp_names: list[str] = []
    for index, compound in enumerate(comps_meta):
        if not isinstance(compound, dict) or not isinstance(compound.get("name"), str):
            raise ValueError(f"block metadata compounds[{index}] is missing name")
        comp_names.append(compound["name"])

    if not x_dict:
        if len(comp_names) != 1:
            return {
                "error": "Block has no composition columns and is not unary",
                "doi": doi,
                "block_number": block_number,
                "BLKsubsys_id": BLKsubsys_id,
            }
        n_components = 1
    elif len(x_dict) == 1 and len(comp_names) == 2:
        n_components = 2
        # Binary blocks store one explicit axis; the second is exactly 1-x.
        existing_comp = next(iter(x_dict))
        x_arr = next(iter(x_dict.values()))
        other_names = [n for n in comp_names if n != existing_comp]
        if len(other_names) != 1:
            return {
                "error": "Binary block metadata does not identify the implicit component",
                "doi": doi,
                "block_number": block_number,
                "BLKsubsys_id": BLKsubsys_id,
            }
        other_name = other_names[0]
        x_dict[other_name] = 1.0 - x_arr
    elif len(x_dict) == len(comp_names):
        n_components = len(comp_names)
    else:
        return {
            "error": (
                f"Composition-column count {len(x_dict)} does not match "
                f"compound count {len(comp_names)}"
            ),
            "doi": doi,
            "block_number": block_number,
            "BLKsubsys_id": BLKsubsys_id,
        }

    # PCS presentation is part of the numerical property definition.  Build
    # an exact global-ID composition map and materialize the absolute response
    # before treating any row as a pure endpoint.
    compounds_by_name = {
        item["name"].strip().casefold(): item["comp_num_id"]
        for item in comps_meta
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("comp_num_id"), str)
    }
    normalized_composition: dict[str, np.ndarray] = {}
    if x_dict:
        try:
            normalized_composition = {
                compounds_by_name[name.strip().casefold()]: np.asarray(values, dtype=float)
                for name, values in x_dict.items()
            }
        except KeyError as exc:
            return {
                "error": "Composition component could not be aligned to a global compound ID",
                "doi": doi,
                "block_number": block_number,
                "BLKsubsys_id": BLKsubsys_id,
                "component": str(exc),
            }
    elif len(comp_names) == 1:
        comp_id = compounds_by_name.get(comp_names[0].strip().casefold())
        if comp_id is None:
            return {
                "error": "Unary component could not be aligned to a global compound ID",
                "doi": doi,
                "block_number": block_number,
                "BLKsubsys_id": BLKsubsys_id,
            }
        normalized_composition = {comp_id: np.ones(len(y), dtype=float)}

    try:
        response_contract = inspect_property_response_contract(
            bd.metadata, match.y_column
        )
        if (
            response_contract["requires_reference_materialization"]
            and response_contract["requires_normalized_composition"]
        ):
            non_mole = [
                column for column in match.x_columns
                if detect_composition_basis(column) != "mole_fraction"
            ]
            if non_mole:
                raise PropertyResponsePreparationError(
                    "REFERENCE_COMPOSITION_BASIS_UNRESOLVED",
                    "Same-proportion materialization requires mole-fraction composition columns.",
                    details={"columns": non_mole},
                )
        canonical_state = {"temperature_k": None, "pressure_kpa": None}
        state_tolerances = {"temperature_k": 0.5, "pressure_kpa": 1.0}
        for column, spec in applied_constraints.items():
            key = column.casefold()
            canonical = (
                "temperature_k" if "temperature" in key
                else "pressure_kpa" if "pressure" in key
                else None
            )
            if canonical is not None:
                canonical_state[canonical] = float(spec["value"])
                state_tolerances[canonical] = float(spec.get("tol", state_tolerances[canonical]))
        if response_contract["requires_reference_materialization"]:
            for quantity, token, ref_field in (
                ("temperature_k", "temperature", "ref_temperature_K"),
                ("pressure_kpa", "pressure", "ref_pressure_kPa"),
            ):
                if (
                    canonical_state[quantity] is not None
                    or response_contract.get(ref_field) is not None
                ):
                    continue
                state_arrays = []
                for column in bd.columns:
                    if token not in column.casefold() or column not in bd.arrays:
                        continue
                    array = np.asarray(bd.arrays[column], dtype=float)[row_mask]
                    finite = array[np.isfinite(array)]
                    if finite.size:
                        state_arrays.append(finite)
                if state_arrays:
                    distinct = np.unique(
                        np.round(np.concatenate(state_arrays), decimals=10)
                    )
                    if len(distinct) > 1:
                        return {
                            "error": (
                                "STATE_REFINEMENT_REQUIRED: reference-relative "
                                f"property has {len(distinct)} selected {quantity} "
                                "values; constrain the state before extracting endpoints"
                            ),
                            "error_code": "STATE_REFINEMENT_REQUIRED",
                            "doi": doi,
                            "lit_num_id": bd.metadata["lit_num_id"],
                            "block_number": block_number,
                            "BLKsubsys_id": BLKsubsys_id,
                            "state_quantity": quantity,
                            "state_values": distinct.astype(float).tolist(),
                        }
                    canonical_state[quantity] = float(distinct[0])
        response = prepare_property_response(
            metadata=bd.metadata,
            y_column=match.y_column,
            reported_values=y,
            normalized_composition=normalized_composition,
            state=canonical_state,
            state_tolerances=state_tolerances,
        )
    except PropertyResponsePreparationError as exc:
        return property_response_error_result(
            exc,
            doi=doi,
            lit_num_id=bd.metadata["lit_num_id"],
            block_number=block_number,
            BLKsubsys_id=BLKsubsys_id,
        )
    y = response.values

    # Unary blocks have no composition axis; every valid row is pure.
    if not x_dict and len(comp_names) == 1 and bd.n_rows > 0:
        valid = ~np.isnan(y)
        if not np.any(valid):
            return {"error": "Unary block contains no valid property values",
                    "doi": doi, "block_number": block_number, "BLKsubsys_id": BLKsubsys_id}
        name = comp_names[0]
        pure_values = {name: round(float(np.nanmean(y[valid])), 6)}
        edge_info = {
            name: {
                "value": pure_values[name],
                "max_x": 1.0,
                "n_near_pure": int(np.sum(valid)),
                "method": "direct",
                "quality": "good",
            }
        }
    else:
        try:
            edge_info = extract_pure_from_edges(x_dict, y, threshold=edge_threshold)
        except Exception as exc:
            return {
                "error": f"Could not extract pure values: {exc}",
                "doi": doi,
                "block_number": block_number,
                "BLKsubsys_id": BLKsubsys_id,
            }
        pure_values = {comp: details["value"] for comp, details in edge_info.items()}
        if not pure_values:
            return {
                "error": "No pure-component endpoints found within edge_threshold",
                "doi": doi,
                "block_number": block_number,
                "BLKsubsys_id": BLKsubsys_id,
            }

    # Extract temperature / pressure if available (over the SAME row subset)
    temp_col = next((c for c in bd.columns if "temperature" in c.lower()), None)
    pres_col = next((c for c in bd.columns if "pressure" in c.lower()), None)

    result: dict = {
        "doi": doi,
        "lit_num_id": bd.metadata["lit_num_id"],
        "block_number": block_number,
        "BLKsubsys_id": BLKsubsys_id,
        "pure_values": pure_values,
        "edge_coverage": {
            comp: {"max_x": d["max_x"], "n_near_pure": d["n_near_pure"],
                   "method": d["method"], "quality": d["quality"]}
            for comp, d in edge_info.items()
        },
        "x_columns": match.x_columns,
        "y_column": match.y_column,
        "components": comp_names,
        "n_components": n_components,
        "property_response": response.trace,
    }

    # Explicit composition-axis mapping so downstream fits are unambiguous:
    # pure_values can be passed to fit_block as-is.
    if match.x_column:
        if match.x_column not in match.column_compound_map:
            raise ValueError(
                f"composition column {match.x_column!r} is missing its compound mapping"
            )
        x_comp = match.column_compound_map[match.x_column]
        others = [c for c in x_dict if c != x_comp]
        result["fit_block_mapping"] = {
            "x_column": match.x_column,
            "pure_at_x1": x_comp,
            "pure_at_x0": others[0] if len(others) == 1 else others,
            "hint": ("Pass this pure_values dict directly to fit_block as "
                     "pure_values — names are mapped to the axis "
                     "automatically."),
        }
    if applied_constraints:
        result["applied_constraints"] = applied_constraints
    if temp_col and temp_col in bd.arrays:
        arr = bd.arrays[temp_col][row_mask]
        valid_t = arr[~np.isnan(arr)]
        if len(valid_t) > 0:
            result["temperature_K"] = round(float(np.nanmean(valid_t)), 2)
            t_spread = float(np.max(valid_t) - np.min(valid_t))
            if t_spread > 1.0 and not applied_constraints:
                result["temperature_note"] = (
                    f"Block spans {np.min(valid_t):.2f}–{np.max(valid_t):.2f} K; "
                    f"pure values above average edge rows ACROSS temperatures. "
                    f"Pass x_vars_constrained (e.g. "
                    f'{{"{temp_col}": {{"value": 298.15, "tol": 0.5}}}}) '
                    f"for state-specific values."
                )
    if pres_col and pres_col in bd.arrays:
        arr = bd.arrays[pres_col][row_mask]
        valid_p = arr[~np.isnan(arr)]
        if len(valid_p) > 0:
            result["pressure_kPa"] = round(float(np.nanmean(valid_p)), 2)

    return result


# ── Catalog entries ─────────────────────────────────────────────────────

TOOL_ENTRIES = [
    ToolEntry(
        "query_system_summary", query_system_summary,
        group="discovery",
    ),
    ToolEntry(
        "query_blocks", query_blocks,
        group="discovery",
    ),
    ToolEntry(
        "inspect_block", inspect_block,
        group="discovery",
    ),
    ToolEntry(
        "inspect_block_table", inspect_block_table,
        group="discovery",
        # Verbatim grounding table — deterministic output, no LLM triage.
        skip_subagent=True,
    ),
    ToolEntry(
        "find_similar_compounds", find_similar_compounds,
        group="discovery",
    ),
    ToolEntry(
        "get_pure_values", get_pure_values,
        group="discovery",
        skip_subagent=True,
    ),
]
