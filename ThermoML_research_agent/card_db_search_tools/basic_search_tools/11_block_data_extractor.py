"""
11_block_data_extractor.py — Extract structured CSV data from ThermoML blocks.
=============================================================================
Given a DOI and block_number (from search tools), reads definitions from the
PCS card and every numerical row from the authoritative raw ThermoML database.

The query agent only resolves IDs and metadata.  This tool goes to the raw
DB, parses the data_points array, and produces a clean CSV string with
human-readable column headers suitable for direct input to fitting tools.

Public API
----------
    extract_block_csv(doi, block_number, *, property_filter=None)
        -> dict with "doi", "block_number", "csv_text", "columns",
           "n_rows", and "metadata"
"""

import csv
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalization_helpers.paths import card_db_path
from normalization_helpers.compound_id_helpers import (
    resolve_comp_in_id,
    build_column_map,
    build_compound_map,
    extract_comp_ref,
)
from normalization_helpers.db_helpers import open_db, project_pcs_block_target
from normalization_helpers.strict_id_inputs import (
    refinement_errors_as_results,
    require_typed_block_input,
)
from ThermoML_raw_json_to_card_db_parsers.id_schema import (
    block_local_id,
    block_local_ordinal,
    require_block_id,
    require_block_local_id,
    require_doi_comp_id,
    require_global_id,
)

_PCS_DB = card_db_path("PCS_INDIV")
_RAW_DB = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "ThermoML.v2020-09-30.db", "thermoml_raw.db",
))

_resolve_comp_in_id = resolve_comp_in_id
_build_column_map = build_column_map
_build_compound_map = build_compound_map
_extract_comp_ref = extract_comp_ref


UNCERTAINTY_SUFFIX = "_U"
"""Column suffix of a source-reported expanded (combined) uncertainty."""

LIMIT_SUFFIX = "_limit"
"""Column suffix of the reported bound of a censored (``PropLimit``) value."""

_UNCERTAINTY_SPECS = {
    "var": ("VarUncertainty", "nExpandUncertValue"),
    "prop": ("CombinedUncertainty", "nCombExpandUncertValue"),
}


def _first_entry(entries, field: str):
    """First object in *entries* (an object or a list of them) carrying *field*."""
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get(field) is not None:
            return entry[field]
    return None


def _expanded_uncertainty(item: dict, kind: str):
    """First expanded uncertainty recorded on one raw value entry, or None.

    ThermoML keeps ``CombinedUncertainty`` (properties) and
    ``VarUncertainty`` (variables) on each ``NumValues`` value entry as one
    object or, for several assessments, a list of them; the first entry
    with a numeric expanded value is taken (its ``nCombUncertAssessNum``
    keys the block-level assessment reported in the metadata header).
    """
    collection, value_field = _UNCERTAINTY_SPECS[kind]
    return _first_entry(item.get(collection), value_field)


def _extract_raw_point(raw_point: dict, col_map: dict) -> dict:
    """Map one uncapped raw ThermoML ``NumValues`` row to PCS columns.

    Besides the value columns the row carries ``<column>_U`` for every value
    entry whose source records an expanded uncertainty and
    ``<property>_limit`` for a censored property value (the reported bound;
    the value column itself stays empty).
    """
    if not isinstance(raw_point, dict):
        raise ValueError("raw ThermoML NumValues entries must be objects")
    specs = {
        "var": ("VariableValue", "nVarNumber", "nVarValue"),
        "prop": ("PropertyValue", "nPropNumber", "nPropValue"),
    }
    row = {}
    for raw_key, col_name in col_map.items():
        if raw_key.startswith("BLKconstr_"):
            row[col_name] = None
            continue
        if raw_key.startswith("BLKvar_"):
            kind = "var"
        elif raw_key.startswith("BLKprop_"):
            kind = "prop"
        else:
            raise ValueError(f"Unscoped data-point key {raw_key!r}")
        collection, number_field, value_field = specs[kind]
        ordinal = block_local_ordinal(kind, raw_key)
        items = raw_point.get(collection, [])
        if not isinstance(items, list):
            raise ValueError(f"raw NumValues.{collection} must be an array")
        matches = [
            item for item in items
            if isinstance(item, dict) and item.get(number_field) == ordinal
        ]
        if len(matches) > 1:
            raise ValueError(
                f"raw NumValues.{collection} contains duplicate number {ordinal}"
            )
        if not matches:
            row[col_name] = None
            continue
        item = matches[0]
        if value_field in item:
            row[col_name] = item[value_field]
        elif kind == "prop" and isinstance(
            item.get("PropLimit"), (dict, list)
        ):
            # Censored value: the source reports a bound, not a value.
            row[col_name] = None
            bound = _first_entry(item["PropLimit"], "nPropLimitValue")
            if bound is not None:
                row[f"{col_name}{LIMIT_SUFFIX}"] = bound
        else:
            raise ValueError(
                f"raw NumValues.{collection}[{ordinal}] lacks {value_field}"
            )
        expanded = _expanded_uncertainty(item, kind)
        if expanded is not None:
            row[f"{col_name}{UNCERTAINTY_SUFFIX}"] = expanded
    return row


def _find_raw_block(paper: dict, block_id: str) -> dict:
    """Resolve one strict typed block in the source ThermoML paper."""
    prefix, ordinal_text = block_id.rsplit("_", 1)
    if prefix == "PROPblock":
        family = "PureOrMixtureData"
        number_field = "nPureOrMixtureDataNumber"
    elif prefix == "RXNblock":
        family = "ReactionData"
        number_field = "nReactionDataNumber"
    else:  # guarded by require_typed_block_input
        raise ValueError(f"Unsupported typed block ID {block_id!r}")
    blocks = paper.get(family, [])
    if not isinstance(blocks, list):
        raise ValueError(f"raw ThermoML paper field {family} must be an array")
    ordinal = int(ordinal_text)
    matches = [
        block for block in blocks
        if isinstance(block, dict) and block.get(number_field) == ordinal
    ]
    if len(matches) != 1:
        raise ValueError(
            f"raw ThermoML source resolved {len(matches)} matches for {block_id}"
        )
    return matches[0]


def _block_uncertainty(properties: list) -> dict | None:
    """Block-level combined-uncertainty assessment of the first property carrying one.

    The PCS card keeps the source's ``CombinedUncertainty`` declaration per
    property (evaluation method, confidence level, evaluator, assessment
    number); the per-point expanded values travel in the ``<property>_U``
    columns of the extracted rows.
    """
    for prop in properties:
        entry = prop.get("uncertainty") if isinstance(prop, dict) else None
        if not isinstance(entry, dict):
            continue
        picked = {
            "evaluation_method": entry.get("evaluation_method"),
            "confidence_level": entry.get("confidence_level"),
            "evaluator": entry.get("evaluator"),
            "assessment_num": entry.get("assessment_num"),
        }
        if any(value is not None for value in picked.values()):
            return picked
    return None


def _block_metadata(card: dict, block: dict, subsystem: dict | None = None) -> dict:
    """Extract chemistry metadata for one exact declared/subsystem target."""
    if "key" not in card or not isinstance(card["key"], dict):
        raise ValueError("PCS card requires a key object")
    key = card["key"]
    for field in ("doi", "lit_num_id"):
        if field not in key:
            raise ValueError(f"PCS card key requires {field}")
    for field in (
        "block_number", "block_type", "compounds", "variables",
        "properties", "constraints", "solvents", "data_summary",
    ):
        if field not in block:
            raise ValueError(f"PCS block requires {field}")

    full_cmap = _build_compound_map(block, card=card)
    projected = project_pcs_block_target(block, subsystem)

    def compound_rows(compounds):
        return [
            {
                "comp_num_id": require_global_id(
                    "comp_num_id", item["comp_num_id"]
                ),
                "org_num": require_doi_comp_id(item["org_num"]),
                "name": item["name"],
                "formula": item["formula"],
            }
            for item in compounds
        ]

    active_orgs = {item["org_num"] for item in projected["compounds"]}
    active_cmap = {
        org_num: name for org_num, name in full_cmap.items()
        if org_num in active_orgs
    }
    return {
        "doi": key["doi"],
        "lit_num_id": require_global_id("lit_num_id", key["lit_num_id"]),
        "block_number": require_block_id(block["block_number"]),
        "block_type": block["block_type"],
        "system_type": (
            subsystem["effective_system_type"]
            if subsystem is not None else block["system_type"]
        ),
        "declared_system_type": block["system_type"],
        "compounds": compound_rows(projected["compounds"]),
        "declared_compounds": compound_rows(block["compounds"]),
        "compound_map": active_cmap,
        "variables": [
            {
                "var_num_id": require_global_id("var_num_id", v["var_num_id"]),
                "BLKvar_id": v["BLKvar_id"],
                "var_id": v["var_id"],
                "column_name": _resolve_comp_in_id(v["var_id"], full_cmap),
                "compound": _extract_comp_ref(v["var_id"], full_cmap),
            }
            for v in projected["variables"]
        ],
        "properties": [
            {
                "prop_num_id": require_global_id("prop_num_id", prop["prop_num_id"]),
                "BLKprop_id": prop["BLKprop_id"],
                "prop_ID": prop["prop_ID"],
                "column_name": _resolve_comp_in_id(prop["prop_ID"], full_cmap),
                "prop_group": prop["group"],
                "compound": _extract_comp_ref(prop["prop_ID"], full_cmap),
                "component_org_num": (
                    require_doi_comp_id(prop["component_org_num"])
                    if prop["component_org_num"] is not None else None
                ),
                "meas_num_id": (
                    require_global_id("meas_num_id", prop["meas_num_id"])
                    if prop["meas_num_id"] is not None else None
                ),
                "meas_ID": prop["meas_ID"],
                "method_standard": prop["method_standard"],
                "method_custom": prop["method_custom"],
                "presentation": prop["presentation"],
                "standard_state": prop["standard_state"],
                "ref_state_type": prop["ref_state_type"],
                "ref_temperature_K": prop["ref_temperature_K"],
                "ref_temperature_digits": prop["ref_temperature_digits"],
                "ref_pressure_kPa": prop["ref_pressure_kPa"],
                "ref_pressure_digits": prop["ref_pressure_digits"],
                "temperature_K": prop["temperature_K"],
                "temperature_digits": prop["temperature_digits"],
                "pressure_kPa": prop["pressure_kPa"],
                "pressure_digits": prop["pressure_digits"],
                "property_phase": prop["property_phase"],
                "ref_phase": prop["ref_phase"],
            }
            for prop in projected["properties"]
        ],
        "constraints": [
            {
                "constr_num_id": require_global_id(
                    "constr_num_id", constraint["constr_num_id"]
                ),
                "BLKconstr_id": constraint["BLKconstr_id"],
                "constr_id": constraint["constr_id"],
                "column_name": _resolve_comp_in_id(
                    constraint["constr_id"], full_cmap
                ),
                "compound": _extract_comp_ref(
                    constraint["constr_id"], full_cmap
                ),
                "value": constraint["value"],
            }
            for constraint in projected["constraints"]
        ],
        "solvents": projected["solvents"],
        "uncertainty": _block_uncertainty(projected["properties"]),
        "BLKsubsys_id": subsystem["BLKsubsys_id"] if subsystem else None,
        "matched_subsystem": subsystem,
        "n_datapoints": (
            subsystem["n_points"]
            if subsystem is not None else block["data_summary"]["n_points"]
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════════════

@refinement_errors_as_results
def extract_block_csv(
    doi: str,
    block_number: str,
    *,
    BLKsubsys_id: str | None = None,
    property_filter: str | None = None,
) -> dict:
    """Extract data points from a ThermoML block as CSV text.

    Parameters
    ----------
    doi : str
        Paper DOI (e.g. "10.1016/j.fluid.2004.11.019").
    block_number : str
        Typed block identifier (e.g. ``PROPblock_2`` or ``RXNblock_1``).
    property_filter : str, optional
        If given, only include columns matching this substring
        (e.g. "density" to keep only density property columns).

    Returns
    -------
    dict
        csv_text : str — full CSV with header row
        columns : list[str] — column names in order: BLKpoint_id, the
            variables, constraints and properties, then ``<prop>_U`` (the
            source's expanded combined uncertainty per property column,
            empty where none is recorded), ``<var>_U`` where recorded, and
            ``<prop>_limit`` (the reported bound of a censored value)
        n_rows : int — number of data rows
        metadata : dict — DOI, block info, compounds; ``uncertainty`` =
            the block-level assessment {evaluation_method,
            confidence_level, evaluator, assessment_num} or None
        error : str | None — error message if failed
    """
    block_id = require_typed_block_input(block_number)
    subsystem_id = (
        require_block_local_id("subsys", BLKsubsys_id)
        if BLKsubsys_id is not None else None
    )

    if not os.path.exists(_PCS_DB):
        return {"doi": doi, "block_number": block_id, "error": f"PCS_INDIV.db not found at {_PCS_DB}", "csv_text": "", "columns": [], "n_rows": 0, "metadata": {}}
    if not os.path.exists(_RAW_DB):
        return {"doi": doi, "block_number": block_id, "error": f"thermoml_raw.db not found at {_RAW_DB}", "csv_text": "", "columns": [], "n_rows": 0, "metadata": {}}

    conn = open_db(_PCS_DB)
    try:
        row = conn.execute(
            "SELECT json_data FROM cards WHERE doi = ?", (doi,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return {"doi": doi, "block_number": block_id, "error": f"DOI '{doi}' not found in PCS_INDIV.db", "csv_text": "", "columns": [], "n_rows": 0, "metadata": {}}

    card = json.loads(row["json_data"])

    raw_conn = open_db(_RAW_DB)
    try:
        raw_row = raw_conn.execute(
            "SELECT json_data FROM papers WHERE doi = ?", (doi,)
        ).fetchone()
    finally:
        raw_conn.close()
    if not raw_row:
        return {"doi": doi, "block_number": block_id, "error": f"DOI '{doi}' not found in thermoml_raw.db", "csv_text": "", "columns": [], "n_rows": 0, "metadata": {}}
    raw_paper = json.loads(raw_row["json_data"])

    # Find the target block
    block = None
    if "blocks" not in card or not isinstance(card["blocks"], list):
        raise ValueError("PCS card requires a blocks array")
    for b in card["blocks"]:
        if not isinstance(b, dict) or "block_number" not in b:
            raise ValueError("PCS blocks entries require block_number")
        if require_typed_block_input(b["block_number"]) == block_id:
            block = b
            break

    if block is None:
        available = [b["block_number"] for b in card["blocks"]]
        return {
            "doi": doi, "block_number": block_id,
            "error": f"Block '{block_id}' not found in DOI '{doi}'. Available: {available}",
            "csv_text": "", "columns": [], "n_rows": 0, "metadata": {},
        }

    manifests = card["blocks_summary"]["derived_indexes"]["composition_subsystems"]
    if block_id not in manifests:
        raise ValueError(f"PCS composition_subsystems is missing {doi!r} {block_id!r}")
    subsystem = None
    if subsystem_id is not None:
        matches = [
            item for item in manifests[block_id]
            if item["BLKsubsys_id"] == subsystem_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{subsystem_id!r} does not resolve uniquely in {doi!r} {block_id!r}"
            )
        subsystem = matches[0]

    # Build the raw column map, then project active declarations to the
    # exact target. Excluded-component zero columns remain represented by the
    # subsystem manifest evidence, not as active fitting axes.
    cmap = _build_compound_map(block, card=card)
    col_map = _build_column_map(block, cmap=cmap)
    projected = project_pcs_block_target(block, subsystem)
    active_local_ids = {
        item[field]
        for section, field in (
            ("variables", "BLKvar_id"),
            ("properties", "BLKprop_id"),
            ("constraints", "BLKconstr_id"),
        )
        for item in projected[section]
    }
    col_map = {
        local_id: column_name
        for local_id, column_name in col_map.items()
        if local_id in active_local_ids
    }

    # Apply property filter if requested (space and underscore forms both match)
    if property_filter:
        pf = property_filter.lower().replace(" ", "_")
        col_map = {
            k: v for k, v in col_map.items()
            if not k.startswith("BLKprop_") or pf in v.lower().replace(" ", "_")
        }

    # Extract every authoritative source row. PCS data_points are a display
    # sample and must never be used for numerical export or fitting.
    raw_block = _find_raw_block(raw_paper, block_id)
    data_points = raw_block.get("NumValues")
    if not isinstance(data_points, list):
        raise ValueError("raw ThermoML block requires a NumValues array")
    declared_n = block["data_summary"]["n_points"]
    if len(data_points) != declared_n:
        raise ValueError(
            f"raw row count {len(data_points)} disagrees with declared "
            f"n_datapoints {declared_n} for {doi}#{block_id}"
        )
    if not data_points:
        return {
            "doi": doi, "block_number": block_id,
            "error": "No data points in this block",
            "csv_text": "", "columns": list(col_map.values()), "n_rows": 0,
            "metadata": _block_metadata(card, block),
        }

    selected_ordinals = None
    if subsystem is not None:
        selected_ordinals = set()
        for start, end in subsystem["point_membership"]["runs"]:
            if start < 1 or end < start or end > len(data_points):
                raise ValueError(
                    f"invalid subsystem point run {[start, end]!r} for "
                    f"{doi}#{block_id} with {len(data_points)} rows"
                )
            selected_ordinals.update(range(start, end + 1))
        if len(selected_ordinals) != subsystem["n_points"]:
            raise ValueError(
                f"subsystem point runs contain {len(selected_ordinals)} rows, "
                f"not declared n_points={subsystem['n_points']}"
            )

    # Build rows with stable raw-row identities.
    rows = []
    for point_ordinal, dp in enumerate(data_points, start=1):
        if selected_ordinals is not None and point_ordinal not in selected_ordinals:
            continue
        row_data = _extract_raw_point(dp, col_map)
        row_data["BLKpoint_id"] = block_local_id("point", point_ordinal)
        rows.append(row_data)

    # Backfill BLOCK-LEVEL constraint values: isothermal/isobaric blocks
    # store the constraint value ONCE in the constraint definition, not
    # per data point — without this the constraint column is all-empty
    # and downstream temperature filtering discards the whole block.
    for c in block["constraints"]:
        col = col_map.get(c["BLKconstr_id"])
        cval = c["value"]
        if not col or cval is None:
            continue
        for r in rows:
            if r.get(col) in (None, ""):
                r[col] = cval

    # Determine column order: variables first, then constraints, then
    # properties; then the source uncertainties — ``<prop>_U`` for EVERY
    # property column (empty where the source records none) and ``<var>_U``
    # where a variable carries one — and last the ``<prop>_limit`` bounds of
    # censored values where the source records any.
    columns = ["BLKpoint_id"]
    ordered = {"var": [], "constr": [], "prop": []}
    for kind in ("var", "constr", "prop"):
        local_keys = [
            key for key in col_map if key.startswith(f"BLK{kind}_")
        ]
        for raw_key in sorted(
            local_keys, key=lambda key: block_local_ordinal(kind, key)
        ):
            columns.append(col_map[raw_key])
            ordered[kind].append(col_map[raw_key])
    present = set()
    for r in rows:
        present.update(r)
    columns += [f"{c}{UNCERTAINTY_SUFFIX}" for c in ordered["prop"]]
    columns += [
        f"{c}{UNCERTAINTY_SUFFIX}" for c in ordered["var"]
        if f"{c}{UNCERTAINTY_SUFFIX}" in present
    ]
    columns += [
        f"{c}{LIMIT_SUFFIX}" for c in ordered["prop"]
        if f"{c}{LIMIT_SUFFIX}" in present
    ]

    # Write CSV
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

    csv_text = buf.getvalue()

    return {
        "doi": doi,
        "block_number": block_id,
        "BLKsubsys_id": subsystem_id,
        "csv_text": csv_text,
        "columns": columns,
        "n_rows": len(rows),
        "metadata": _block_metadata(card, block, subsystem),
        "error": None,
    }


def extract_multi_block_csv(
    blocks: list[dict],
    *,
    property_filter: str | None = None,
) -> dict:
    """Extract and merge data from multiple blocks into a single CSV.

    Parameters
    ----------
    blocks : list[dict]
        Each dict must have "doi" and "block_number" keys.
        Optionally "label" for a source column.
    property_filter : str, optional
        Only include property columns matching this substring.

    Returns
    -------
    dict
        csv_text : str — merged CSV with a "source" column prepended
        columns : list[str]
        n_rows : int
        per_block : list[dict] — summary per block (n_rows, doi, block)
        errors : list[str]
    """
    all_rows = []
    all_columns = set()
    per_block = []
    errors = []

    for spec in blocks:
        if not isinstance(spec, dict):
            raise TypeError("blocks entries must be objects")
        allowed = {"doi", "block_number", "BLKsubsys_id", "label"}
        unknown = set(spec) - allowed
        missing = {"doi", "block_number"} - spec.keys()
        if unknown or missing:
            raise ValueError(
                f"block specification mismatch: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        doi = spec["doi"]
        bn = require_block_id(spec["block_number"])
        subsystem_id = spec.get("BLKsubsys_id")
        if subsystem_id is not None:
            subsystem_id = require_block_local_id("subsys", subsystem_id)
        label = spec.get(
            "label", f"{doi}#{bn}" + (f"#{subsystem_id}" if subsystem_id else "")
        )

        result = extract_block_csv(
            doi, bn, BLKsubsys_id=subsystem_id, property_filter=property_filter
        )
        if result.get("error"):
            errors.append(f"{label}: {result['error']}")
            per_block.append({"label": label, "n_rows": 0, "error": result["error"]})
            continue

        # Parse rows back from CSV
        reader = csv.DictReader(io.StringIO(result["csv_text"]))
        block_rows = []
        for r in reader:
            r["_source"] = label
            block_rows.append(r)
            all_columns.update(r.keys())

        all_rows.extend(block_rows)
        per_block.append({"label": label, "n_rows": len(block_rows), "error": None})

    if not all_rows:
        return {
            "csv_text": "", "columns": [], "n_rows": 0,
            "per_block": per_block, "errors": errors,
        }

    # Build final CSV with _source as first column
    columns = ["_source"] + sorted(c for c in all_columns if c != "_source")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in all_rows:
        writer.writerow(r)

    return {
        "csv_text": buf.getvalue(),
        "columns": columns,
        "n_rows": len(all_rows),
        "per_block": per_block,
        "errors": errors,
    }


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    res = extract_block_csv("10.1016/j.fluid.2004.11.019", "PROPblock_2")
    print(f"Columns: {res['columns']}")
    print(f"Rows: {res['n_rows']}")
    print(f"Error: {res['error']}")
    if res["csv_text"]:
        lines = res["csv_text"].strip().split("\n")
        for line in lines[:6]:
            print(line)
        if len(lines) > 6:
            print(f"... ({len(lines)-1} total rows)")
