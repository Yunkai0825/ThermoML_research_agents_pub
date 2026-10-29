"""Exact structured-output contracts for query-agent worker boundaries.

These validators deliberately reject missing fields, extra fields, wrong
types, unscoped identifiers, and retired identifier prefixes.  Worker JSON is
never repaired or translated to an older schema after generation.
"""

from __future__ import annotations

import re

from typing import Any

from ThermoML_raw_json_to_card_db_parsers.id_schema import require_global_id
from .L1_workers.core_id_management import (
    CORE_BLOCK_ENRICHED_FIELDS,
    CORE_BLOCK_INPUT_FIELDS,
    CORE_ID_UPDATE_ENRICHED_FIELDS,
    construct_core_blocks_found,
    construct_core_id_updates,
)
from .L2_leaf_evaluators.l2_field_construction import (
    L2_COMP_CORE_FIELDS,
    L2_COMP_ENRICHED_FIELDS,
    L2_MEAS_CORE_FIELDS,
    L2_MEAS_ENRICHED_FIELDS,
    L2_PROP_CORE_FIELDS,
    L2_PROP_ENRICHED_FIELDS,
    L2_REF_CORE_FIELDS,
    L2_REF_ENRICHED_FIELDS,
    construct_l2_compounds,
    construct_l2_measurements,
    construct_l2_properties,
    construct_l2_reference,
)

_STATUSES = {"success", "partial", "no_results"}

def _object(value: Any, context: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _array(value: Any, context: str) -> list:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    return value


def _exact(row: dict, fields: set[str], context: str) -> None:
    actual = set(row)
    if actual != fields:
        raise ValueError(
            f"{context} fields must be exactly {sorted(fields)}; "
            f"missing={sorted(fields - actual)}, extra={sorted(actual - fields)}"
        )


def _string(value: Any, context: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        qualifier = "non-empty " if nonempty else ""
        raise TypeError(f"{context} must be a {qualifier}string")
    return value



def _status_summary(payload: dict, context: str) -> None:
    if payload["status"] not in _STATUSES:
        raise ValueError(f"{context}.status must be one of {sorted(_STATUSES)}")
    _string(payload["summary"], f"{context}.summary")


def _answer_core_claims(payload: dict, context: str) -> None:
    _string(payload["answer"], f"{context}.answer", nonempty=True)
    core_claims = _array(payload["core_claims"], f"{context}.core_claims")
    if not core_claims:
        raise ValueError(f"{context}.core_claims must contain at least one core claim")
    for index, claim in enumerate(core_claims):
        _string(claim, f"{context}.core_claims[{index}]", nonempty=True)


def _data_inspections(payload: dict, context: str) -> None:
    """Deterministically harvested verbatim inspection tables (may be empty)."""
    items = _array(payload["data_inspections"], f"{context}.data_inspections")
    required = {
        "doi", "lit_num_id", "block_number", "table_mode",
        "columns", "rows_shown", "inspection_id",
    }
    allowed = required | {"BLKsubsys_id", "grounding_lines"}
    for index, item in enumerate(items):
        where = f"{context}.data_inspections[{index}]"
        item = _object(item, where)
        actual = set(item)
        if not required <= actual or not actual <= allowed:
            raise ValueError(
                f"{where} fields must be {sorted(required)} "
                f"(+ optional BLKsubsys_id, grounding_lines); "
                f"got {sorted(actual)}"
            )
        _string(item["doi"], f"{where}.doi", nonempty=True)
        _string(item["lit_num_id"], f"{where}.lit_num_id", nonempty=True)
        if not re.fullmatch(r"(?:PROP|RXN)block_\d+", item["block_number"]):
            raise ValueError(f"{where}.block_number must be a typed block ID")
        if "BLKsubsys_id" in item and not re.fullmatch(
            r"BLKsubsys_\d+", str(item["BLKsubsys_id"])
        ):
            raise ValueError(f"{where}.BLKsubsys_id must be BLKsubsys_<N>")
        if item["table_mode"] not in {"complete", "rdp", "nearest", "head"}:
            raise ValueError(f"{where}.table_mode is invalid")
        columns = _array(item["columns"], f"{where}.columns")
        if not columns or any(
            not isinstance(c, str) or not c.strip() for c in columns
        ):
            raise ValueError(f"{where}.columns must be non-empty strings")
        rows = _array(item["rows_shown"], f"{where}.rows_shown")
        if not rows:
            raise ValueError(f"{where}.rows_shown must not be empty")
        for row_index, row in enumerate(rows):
            row = _object(row, f"{where}.rows_shown[{row_index}]")
            for key, cell in row.items():
                _string(cell, f"{where}.rows_shown[{row_index}].{key}")
        if "grounding_lines" in item:
            grounding_lines = _array(
                item["grounding_lines"], f"{where}.grounding_lines",
            )
            for line_index, line in enumerate(grounding_lines):
                _string(
                    line, f"{where}.grounding_lines[{line_index}]",
                    nonempty=True,
                )
        if not re.fullmatch(r"INSP_[0-9a-f]{12}", item["inspection_id"]):
            raise ValueError(f"{where}.inspection_id must be INSP_<12 hex>")


def validate_l1_query_output(value: Any) -> dict:
    payload = _object(value, "L1 query output")
    _exact(
        payload,
        {
            "answer", "core_claims", "status", "summary",
            "core_id_updates", "core_blocks_found", "data_inspections",
        },
        "L1 query output",
    )
    _answer_core_claims(payload, "L1 query output")
    _status_summary(payload, "L1 query output")
    _data_inspections(payload, "L1 query output")

    enriched_updates = _array(payload["core_id_updates"], "core_id_updates")
    minimal_updates: list[dict] = []
    for index, update in enumerate(enriched_updates):
        update = _object(update, f"core_id_updates[{index}]")
        _exact(
            update,
            set(CORE_ID_UPDATE_ENRICHED_FIELDS),
            f"core_id_updates[{index}]",
        )
        if update["action"] != "add":
            raise ValueError(f"core_id_updates[{index}].action must be 'add'")
        _string(
            update["core_GLOB_id"],
            f"core_id_updates[{index}].core_GLOB_id",
            nonempty=True,
        )
        _string(
            update["registry_id"],
            f"core_id_updates[{index}].registry_id",
            nonempty=True,
        )
        _string(update["name"], f"core_id_updates[{index}].name", nonempty=True)
        minimal_updates.append(
            {
                "action": update["action"],
                "core_GLOB_id": update["core_GLOB_id"],
            }
        )
    expected_updates = construct_core_id_updates(minimal_updates)["core_id_updates"]
    if enriched_updates != expected_updates:
        raise ValueError(
            "core_id_updates must equal the canonical output of "
            "construct_core_id_updates"
        )

    enriched_blocks = _array(payload["core_blocks_found"], "core_blocks_found")
    minimal_blocks: list[dict] = []
    for index, block in enumerate(enriched_blocks):
        block = _object(block, f"core_blocks_found[{index}]")
        _exact(
            block,
            set(CORE_BLOCK_ENRICHED_FIELDS),
            f"core_blocks_found[{index}]",
        )
        minimal_blocks.append(
            {field: block[field] for field in CORE_BLOCK_INPUT_FIELDS}
        )
    expected_blocks = construct_core_blocks_found(minimal_blocks)[
        "core_blocks_found"
    ]
    if enriched_blocks != expected_blocks:
        raise ValueError(
            "core_blocks_found must equal the canonical output of "
            "construct_core_blocks_found"
        )
    return payload


def validate_l2_output(kind: str, value: Any) -> dict:
    validators = {
        "comp": _validate_l2_comp,
        "meas": _validate_l2_meas,
        "prop": _validate_l2_prop,
        "ref": _validate_l2_ref,
    }
    if kind not in validators:
        raise ValueError(f"unknown L2 output kind {kind!r}")
    return validators[kind](value)


def _validate_constructed_list(
    value: Any,
    *,
    context: str,
    field: str,
    core_fields: frozenset[str],
    enriched_fields: frozenset[str],
    constructor: Any,
) -> dict:
    payload = _object(value, context)
    _exact(
        payload,
        {"answer", "core_claims", "status", field, "summary"},
        context,
    )
    _answer_core_claims(payload, context)
    _status_summary(payload, context)
    enriched = _array(payload[field], field)
    minimal: list[dict] = []
    for index, row in enumerate(enriched):
        row = _object(row, f"{field}[{index}]")
        _exact(row, set(enriched_fields), f"{field}[{index}]")
        minimal.append({key: row[key] for key in core_fields})
    expected = constructor(minimal)[field]
    if enriched != expected:
        raise ValueError(
            f"{field} must equal the authoritative output of "
            f"{constructor.__name__}"
        )
    return payload


def _validate_l2_comp(value: Any) -> dict:
    return _validate_constructed_list(
        value,
        context="L2 compound output",
        field="compounds",
        core_fields=L2_COMP_CORE_FIELDS,
        enriched_fields=L2_COMP_ENRICHED_FIELDS,
        constructor=construct_l2_compounds,
    )


def _validate_l2_meas(value: Any) -> dict:
    return _validate_constructed_list(
        value,
        context="L2 measurement output",
        field="measurements",
        core_fields=L2_MEAS_CORE_FIELDS,
        enriched_fields=L2_MEAS_ENRICHED_FIELDS,
        constructor=construct_l2_measurements,
    )


def _validate_l2_prop(value: Any) -> dict:
    return _validate_constructed_list(
        value,
        context="L2 property output",
        field="properties",
        core_fields=L2_PROP_CORE_FIELDS,
        enriched_fields=L2_PROP_ENRICHED_FIELDS,
        constructor=construct_l2_properties,
    )


def _validate_l2_ref(value: Any) -> dict:
    payload = _object(value, "L2 reference output")
    _exact(
        payload,
        {"answer", "core_claims", "status", "reference", "summary"},
        "L2 reference output",
    )
    _answer_core_claims(payload, "L2 reference output")
    _status_summary(payload, "L2 reference output")
    reference = _object(payload["reference"], "reference")
    _exact(reference, set(L2_REF_ENRICHED_FIELDS), "reference")
    minimal = {key: reference[key] for key in L2_REF_CORE_FIELDS}
    expected = construct_l2_reference(minimal)["reference"]
    if reference != expected:
        raise ValueError(
            "reference must equal the authoritative output of "
            "construct_l2_reference"
        )
    return payload