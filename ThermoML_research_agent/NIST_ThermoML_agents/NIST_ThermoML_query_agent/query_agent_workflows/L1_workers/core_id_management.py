"""Authoritative post-answer ID validation and chemistry-context enrichment.

The Query L1 chemistry agent never owns this structure.  Its post-answer
ID-alignment worker emits only small ``core_*`` records:

* ``core_blocks_found`` identifies a block by ``lit_num_id`` and
  ``block_number`` and repeats a few registry values as validation checks;
* ``core_id_updates`` requests one catalog addition by canonical ``GLOB*`` ID.

This module checks those records against the generated ThermoML registries,
replaces all registry-owned values with database values, and adds the
block-level chemistry context needed downstream.  A bounded ReAct-style
review/correction loop is used when validation fails and once more before
the enriched JSON is accepted for final assembly.
"""

from __future__ import annotations

import csv
import importlib
import json
import logging
import math
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from ThermoML_raw_json_to_card_db_parsers.id_schema import (
    GLOBAL_PREFIX_BY_FIELD,
    require_block_id,
    require_block_local_id,
    require_global_id,
)

from ....general_db_query_engine.general_argo_engine_helpers import anchor
from ....general_db_query_engine.general_argo_engine_helpers.json_answer_guard import (
    guard_json_answer,
)
from ....general_db_query_engine.general_hooks_management_helpers.general_postans_eval_hooks import (
    _postans_eval_anchors_catalog as postans_anchor,
)
from ....general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog import (
    AgentToolCatalog,
    ToolEntry,
)


_REPO_ROOT = Path(__file__).resolve().parents[4]
_BASIC_SEARCH = _REPO_ROOT / "card_db_search_tools" / "basic_search_tools"
if str(_BASIC_SEARCH) not in sys.path:
    sys.path.insert(0, str(_BASIC_SEARCH))

from normalization_helpers.paths import csv_path, registry_db_path  # noqa: E402
from normalization_helpers.registry_parsers import (  # noqa: E402
    format_pm_row,
    format_rxn_row,
)

_extract_block_csv = getattr(
    importlib.import_module("card_db_search_tools.basic_search_tools.11_block_data_extractor"),
    "extract_block_csv",
)

log = logging.getLogger("L1-core-ids")


CORE_BLOCK_INPUT_FIELDS = frozenset(
    {
        "lit_num_id",
        "block_number",
        "BLKsubsys_id",
        "system_type",
        "comp_num_ids",
        "prop_num_ids",
        "description",
    }
)
CORE_BLOCK_REQUIRED_INPUT_FIELDS = frozenset(
    set(CORE_BLOCK_INPUT_FIELDS) - {"BLKsubsys_id"}
)
CORE_ID_UPDATE_INPUT_FIELDS = frozenset({"action", "core_GLOB_id"})

CORE_BLOCK_ENRICHED_FIELDS = frozenset(
    set(CORE_BLOCK_INPUT_FIELDS)
    | {
        "doi",
        "lit_id",
        "block_type",
        "blocktype_num_id",
        "n_datapoints",
        "n_components",
        "compounds",
        "solvents",
        "constraints",
        "variables",
        "properties",
        "phases",
        "reaction_type",
        "participants",
        "notes",
        "declared_system_type",
        "declared_n_components",
        "declared_compounds",
        "parent_n_datapoints",
        "subsystem_scope",
        "subsystem_point_runs",
        "subsystem_path_class",
        "subsystem_evidence_quality",
        "subsystem_condition_ranges",
    }
)
CORE_ID_UPDATE_ENRICHED_FIELDS = frozenset(
    {"action", "core_GLOB_id", "registry_id", "name"}
)

_ID_ALIGNMENT_FIELDS = frozenset(
    {"status", "summary", "core_id_updates", "core_blocks_found"}
)
_STATUSES = frozenset({"success", "partial", "no_results"})

_REGISTRY_SPECS = {
    "lit_num_id": ("reference_ids", "lit_id", "doi"),
    "comp_num_id": ("compound_ids", "comp_id", "common_name"),
    "prop_num_id": ("property_ids", "prop_id", "prop_name"),
    "var_num_id": ("variable_ids", "var_id", "var_name"),
    "constr_num_id": ("constraint_ids", "constr_id", "constr_name"),
    "meas_num_id": ("measurement_ids", "meas_id", "method_name"),
    "phase_num_id": ("phase_ids", "phase_id", "phase_name"),
    "solvent_num_id": ("solvent_components", "inchi_key", "common_name"),
    "blocktype_num_id": ("block_types", None, None),
    "rxn_type_num_id": ("reaction_type_ids", "rxn_type_id", "rxn_type_name"),
}

_CORRECTION_SYSTEM = """
You are correcting the minimal core-ID payload for a ThermoML post-answer
stage. Use the validation observation and the working answer; authoritative
values carried by the observation override everything else. Return
exactly the requested JSON object.

For each core_blocks_found item emit only: lit_num_id, block_number,
system_type, comp_num_ids, prop_num_ids, and description. Omit BLKsubsys_id
for a declared parent-block anchor. Include BLKsubsys_id only when the
evidence explicitly selects an embedded composition subsystem, and then copy
the exact recorded BLKsubsys_N value. Never emit a null BLKsubsys_id in the
minimal agent payload. The compound,
property, and system values are checksums for the identified literature
block, so copy the authoritative values in the observation exactly.

For each core_id_updates item emit only: action and core_GLOB_id. Never emit
registry_id, name, type, DOI, datapoint counts, ranges, BLK IDs, or other
enrichment; deterministic code owns those fields. Remove a record when the
completed run does not support a valid replacement. Do not invent IDs.

Do not emit or revise summary; the hidden summary tool owns it.

Output one JSON object only, without prose, tags, or code fences.
""".strip()

_SUBMISSION_REVIEW_SYSTEM = """
You are the final submission reviewer in a bounded ThermoML post-answer
ReAct loop. You have no tools. Review the candidate JSON against the working
answer; the database-enriched fields in the candidate are authoritative.

Return {"decision":"submit","reason":"..."} when the core block selections
and core global-ID updates are supported and sufficient for the answer.
Return {"decision":"refine","reason":"..."} only when a supported core block
or core global ID is missing, extra, or inconsistent. Database-enriched
names, DOI, BLK IDs, ranges, variables, properties, constraints, phases,
measurements, and datapoint counts are authoritative and must not be edited.
Do not request stylistic changes to the chemistry answer or core claims.

Output exactly one JSON object and no prose.
""".strip()


class CoreIDValidationError(ValueError):
    """A minimal core-ID record conflicts with an authoritative registry."""


def _require_exact(row: object, fields: frozenset[str], context: str) -> dict:
    if not isinstance(row, dict):
        raise TypeError(f"{context} must be an object")
    actual = set(row)
    if actual != set(fields):
        raise CoreIDValidationError(
            f"{context} fields must be exactly {sorted(fields)}; "
            f"missing={sorted(set(fields) - actual)}, "
            f"extra={sorted(actual - set(fields))}"
        )
    return row


def _require_core_block_input(row: object, context: str) -> dict:
    """Accept the required anchor plus an optional explicit subsystem selector."""
    if not isinstance(row, dict):
        raise TypeError(f"{context} must be an object")
    actual = set(row)
    missing = set(CORE_BLOCK_REQUIRED_INPUT_FIELDS) - actual
    extra = actual - set(CORE_BLOCK_INPUT_FIELDS)
    if missing or extra:
        raise CoreIDValidationError(
            f"{context} fields must contain {sorted(CORE_BLOCK_REQUIRED_INPUT_FIELDS)} "
            "and may additionally contain BLKsubsys_id; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    normalized = dict(row)
    normalized.setdefault("BLKsubsys_id", None)
    return normalized


def _require_nonempty_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be non-empty text")
    return value.strip()


def _require_unique_global_ids(
    values: object,
    *,
    field: str,
    context: str,
) -> list[str]:
    if not isinstance(values, list):
        raise TypeError(f"{context} must be an array")
    result = [require_global_id(field, value) for value in values]
    if len(result) != len(set(result)):
        raise CoreIDValidationError(f"{context} must not contain duplicate IDs")
    return result


def _finite_column_values(rows: list[dict], column_name: str) -> list[float]:
    values = []
    for row in rows:
        raw = row[column_name]
        if raw in (None, ""):
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    return values


def _project_authoritative_raw_ranges(parsed: dict) -> None:
    """Replace parent declarations/ranges with exact raw subsystem data."""
    extracted = _extract_block_csv(
        parsed["doi"],
        parsed["block_number"],
        BLKsubsys_id=parsed["BLKsubsys_id"],
    )
    if extracted["error"]:
        raise CoreIDValidationError(
            "raw target extraction failed: {}".format(extracted["error"])
        )
    if extracted["n_rows"] != parsed["n_datapoints"]:
        raise CoreIDValidationError(
            "raw target row count disagrees with the subsystem registry"
        )
    rows = list(csv.DictReader(extracted["csv_text"].splitlines()))
    metadata = extracted["metadata"]

    active_variables = {
        item["BLKvar_id"]: item for item in metadata["variables"]
    }
    projected_variables = []
    for source in parsed["variables"]:
        target = active_variables.get(source["BLKvar_id"])
        if target is None:
            continue
        values = _finite_column_values(rows, target["column_name"])
        variable = dict(source)
        variable["range"] = {
            "BLKvar_id": source["BLKvar_id"],
            "name": source["name"],
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "n_unique": len(set(values)),
        }
        variable["range_min"] = variable["range"]["min"]
        variable["range_max"] = variable["range"]["max"]
        projected_variables.append(variable)

    active_properties = {
        item["BLKprop_id"]: item for item in metadata["properties"]
    }
    projected_properties = []
    for source in parsed["properties"]:
        target = active_properties.get(source["BLKprop_id"])
        if target is None:
            continue
        values = _finite_column_values(rows, target["column_name"])
        mean = sum(values) / len(values) if values else None
        variance = (
            sum((value - mean) ** 2 for value in values) / len(values)
            if values else None
        )
        prop = dict(source)
        prop["range"] = {
            "BLKprop_id": source["BLKprop_id"],
            "name": source["name"],
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": mean,
            "std": math.sqrt(variance) if variance is not None else None,
            "n": len(values),
        }
        prop["range_min"] = prop["range"]["min"]
        prop["range_max"] = prop["range"]["max"]
        projected_properties.append(prop)

    active_constraints = {
        item["BLKconstr_id"] for item in metadata["constraints"]
    }
    active_solvents = {
        item["component_org_num"] for item in metadata["solvents"]
    }
    active_owner_ids = set(active_variables) | set(active_properties) | active_constraints
    parsed["variables"] = projected_variables
    parsed["properties"] = projected_properties
    parsed["constraints"] = [
        item for item in parsed["constraints"]
        if item["BLKconstr_id"] in active_constraints
    ]
    parsed["solvents"] = [
        item for item in parsed["solvents"]
        if item["component_org_num"] in active_solvents
    ]
    parsed["phases"] = [
        item for item in parsed["phases"]
        if item["owner_id"] in active_owner_ids
    ]


def _fetch_authoritative_block(
    lit_num_id: str,
    block_number: str,
    BLKsubsys_id: str | None,
) -> dict:
    """Load one parent registry row and optionally its normalized subsystem."""
    require_global_id("lit_num_id", lit_num_id)
    require_block_id(block_number)
    subsystem_id = (
        require_block_local_id("subsys", BLKsubsys_id)
        if BLKsubsys_id is not None else None
    )
    found: list[tuple[str, sqlite3.Row]] = []
    for registry_key, kind in (
        ("PM_REGISTRY", "property"),
        ("RXN_REGISTRY", "reaction"),
    ):
        path = registry_db_path(registry_key)
        if not Path(path).is_file():
            raise FileNotFoundError(f"required block registry is missing: {path}")
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT * FROM block_registry "
                "WHERE lit_num_id = ? AND block_number = ?",
                (lit_num_id, block_number),
            ).fetchall()
        finally:
            connection.close()
        found.extend((kind, row) for row in rows)

    if not found:
        raise CoreIDValidationError(
            f"no block exists for ({lit_num_id}, {block_number})"
        )
    if len(found) != 1:
        raise CoreIDValidationError(
            f"({lit_num_id}, {block_number}) resolves to {len(found)} blocks"
        )
    kind, row = found[0]
    parsed = format_rxn_row(row) if kind == "reaction" else format_pm_row(row)
    parsed["reaction_type"] = parsed.get("reaction_type")
    parsed["participants"] = parsed.get("participants", [])
    parsed.update({
        "BLKsubsys_id": None,
        "declared_system_type": parsed["system_type"],
        "declared_n_components": parsed["n_components"],
        "declared_compounds": deepcopy(parsed["compounds"]),
        "parent_n_datapoints": parsed["n_datapoints"],
        "subsystem_scope": None,
        "subsystem_point_runs": [],
        "subsystem_path_class": None,
        "subsystem_evidence_quality": None,
        "subsystem_condition_ranges": [],
    })
    if subsystem_id is None:
        return parsed
    if kind != "property":
        raise CoreIDValidationError(
            f"ReactionData block {block_number!r} cannot select {subsystem_id!r}"
        )

    path = registry_db_path("PM_REGISTRY")
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        matches = connection.execute(
            "SELECT s.* FROM block_subsystems s JOIN block_registry b "
            "ON b.doi=s.doi AND b.block_number=s.block_number "
            "WHERE b.lit_num_id=? AND s.block_number=? AND s.BLKsubsys_id=?",
            (lit_num_id, block_number, subsystem_id),
        ).fetchall()
        if len(matches) != 1:
            raise CoreIDValidationError(
                f"({lit_num_id}, {block_number}, {subsystem_id}) resolves to "
                f"{len(matches)} subsystems"
            )
        subsystem = matches[0]
        retained_ids = {
            item["comp_num_id"]
            for item in connection.execute(
                "SELECT comp_num_id FROM block_subsystem_compounds "
                "WHERE doi=? AND block_number=? AND BLKsubsys_id=? "
                "AND component_role='retained'",
                (parsed["doi"], block_number, subsystem_id),
            ).fetchall()
        }
        supported_properties = {
            item["BLKprop_id"]
            for item in connection.execute(
                "SELECT BLKprop_id FROM block_subsystem_properties "
                "WHERE doi=? AND block_number=? AND BLKsubsys_id=? "
                "AND support_role='bulk_property' AND phase_compatible=1",
                (parsed["doi"], block_number, subsystem_id),
            ).fetchall()
        }
    finally:
        connection.close()

    retained_compounds = [
        item for item in parsed["compounds"]
        if item["comp_num_id"] in retained_ids
    ]
    properties = [
        item for item in parsed["properties"]
        if item["BLKprop_id"] in supported_properties
    ]
    if len(retained_compounds) != subsystem["n_retained_components"]:
        raise CoreIDValidationError(
            f"{subsystem_id!r} retained compound projection is inconsistent"
        )
    parsed.update({
        "BLKsubsys_id": subsystem_id,
        "system_type": subsystem["effective_system_type"],
        "n_components": subsystem["n_retained_components"],
        "compounds": retained_compounds,
        "properties": properties,
        "n_datapoints": subsystem["n_points"],
        "subsystem_scope": json.loads(subsystem["scope_json"]),
        "subsystem_point_runs": json.loads(subsystem["point_runs_json"]),
        "subsystem_path_class": subsystem["path_class"],
        "subsystem_evidence_quality": subsystem["evidence_quality"],
        "subsystem_condition_ranges": json.loads(
            subsystem["condition_ranges_json"]
        ),
    })
    _project_authoritative_raw_ranges(parsed)
    return parsed

def _validate_core_block(block: object, index: int) -> tuple[dict, dict]:
    context = f"core_blocks_found[{index}]"
    core = _require_core_block_input(block, context)
    lit_num_id = require_global_id("lit_num_id", core["lit_num_id"])
    block_number = require_block_id(core["block_number"])
    subsystem_id = (
        require_block_local_id("subsys", core["BLKsubsys_id"])
        if core["BLKsubsys_id"] is not None else None
    )
    system_type = _require_nonempty_text(core["system_type"], f"{context}.system_type")
    description = _require_nonempty_text(
        core["description"], f"{context}.description"
    )
    comp_num_ids = _require_unique_global_ids(
        core["comp_num_ids"],
        field="comp_num_id",
        context=f"{context}.comp_num_ids",
    )
    prop_num_ids = _require_unique_global_ids(
        core["prop_num_ids"],
        field="prop_num_id",
        context=f"{context}.prop_num_ids",
    )
    authoritative = _fetch_authoritative_block(
        lit_num_id, block_number, subsystem_id
    )
    expected_comp_ids = [
        compound["comp_num_id"] for compound in authoritative["compounds"]
    ]
    expected_prop_ids = [
        prop["prop_num_id"] for prop in authoritative["properties"]
    ]

    # An empty checksum list asserts nothing: the ID-alignment agent copies
    # IDs only from the answer text and leaves unsupported fields empty, and
    # answers rarely print GLOBcomp/GLOBprop IDs. Such a list is filled from
    # the authoritative block instead of bouncing as a mismatch that no
    # correction from the answer can repair. A non-empty list stays strict.
    unasserted = []
    if not comp_num_ids:
        comp_num_ids = list(expected_comp_ids)
        unasserted.append("comp_num_ids")
    if not prop_num_ids:
        prop_num_ids = list(expected_prop_ids)
        unasserted.append("prop_num_ids")
    if unasserted:
        log.info(
            "%s (%s, %s): empty %s filled from the authoritative block",
            context, lit_num_id, block_number, " and ".join(unasserted),
        )

    mismatches: list[str] = []
    if system_type != authoritative["system_type"]:
        mismatches.append(
            f"system_type expected {authoritative['system_type']!r}, "
            f"received {system_type!r}"
        )
    if set(comp_num_ids) != set(expected_comp_ids):
        mismatches.append(
            f"comp_num_ids expected {expected_comp_ids!r}, "
            f"received {comp_num_ids!r}"
        )
    if set(prop_num_ids) != set(expected_prop_ids):
        mismatches.append(
            f"prop_num_ids expected {expected_prop_ids!r}, "
            f"received {prop_num_ids!r}"
        )
    if mismatches:
        raise CoreIDValidationError(
            f"{context} checksum mismatch for ({lit_num_id}, {block_number}): "
            + "; ".join(mismatches)
        )

    normalized_core = {
        "lit_num_id": lit_num_id,
        "block_number": block_number,
        "BLKsubsys_id": subsystem_id,
        "system_type": authoritative["system_type"],
        "comp_num_ids": expected_comp_ids,
        "prop_num_ids": expected_prop_ids,
        "description": description,
    }
    return normalized_core, authoritative


def construct_core_blocks_found(core_blocks_found: object) -> dict[str, list[dict]]:
    """Construct enriched block records from a batch of minimal core anchors."""
    if not isinstance(core_blocks_found, list):
        raise TypeError("core_blocks_found must be an array")
    enriched: list[dict] = []
    seen: set[tuple[str, str, str | None]] = set()
    for index, block in enumerate(core_blocks_found):
        core, authoritative = _validate_core_block(block, index)
        identity = (
            core["lit_num_id"], core["block_number"], core["BLKsubsys_id"]
        )
        if identity in seen:
            raise CoreIDValidationError(
                f"core_blocks_found contains duplicate block {identity!r}"
            )
        seen.add(identity)
        enriched.append(
            {
                **core,
                "doi": authoritative["doi"],
                "lit_id": authoritative["lit_id"],
                "block_type": authoritative["block_type"],
                "blocktype_num_id": authoritative["blocktype_num_id"],
                "n_datapoints": authoritative["n_datapoints"],
                "n_components": authoritative["n_components"],
                "compounds": authoritative["compounds"],
                "solvents": authoritative["solvents"],
                "constraints": authoritative["constraints"],
                "variables": authoritative["variables"],
                "properties": authoritative["properties"],
                "phases": authoritative["phases"],
                "reaction_type": authoritative["reaction_type"],
                "participants": authoritative["participants"],
                "notes": authoritative["notes"],
                "declared_system_type": authoritative["declared_system_type"],
                "declared_n_components": authoritative["declared_n_components"],
                "declared_compounds": authoritative["declared_compounds"],
                "parent_n_datapoints": authoritative["parent_n_datapoints"],
                "subsystem_scope": authoritative["subsystem_scope"],
                "subsystem_point_runs": authoritative["subsystem_point_runs"],
                "subsystem_path_class": authoritative["subsystem_path_class"],
                "subsystem_evidence_quality": authoritative["subsystem_evidence_quality"],
                "subsystem_condition_ranges": authoritative["subsystem_condition_ranges"],
            }
        )
    return {"core_blocks_found": enriched}


def _global_field_for_id(core_global_id: object) -> str:
    if not isinstance(core_global_id, str) or not core_global_id.strip():
        raise TypeError("core_GLOB_id must be non-empty text")
    matches = [
        field
        for field, prefix in GLOBAL_PREFIX_BY_FIELD.items()
        if core_global_id.startswith(prefix)
    ]
    if len(matches) != 1:
        raise CoreIDValidationError(
            f"core_GLOB_id {core_global_id!r} has no unique canonical GLOB prefix"
        )
    return matches[0]


@lru_cache(maxsize=None)
def _registry_rows(csv_key: str, global_field: str) -> dict[str, dict[str, str]]:
    path = Path(csv_path(csv_key))
    if not path.is_file():
        raise FileNotFoundError(f"required global ID registry is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if global_field not in set(reader.fieldnames or ()):
            raise ValueError(f"{path.name} does not contain {global_field}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            global_id = require_global_id(global_field, row[global_field])
            if global_id in rows:
                raise ValueError(f"{path.name} contains duplicate {global_id}")
            rows[global_id] = dict(row)
    return rows


def _resolved_registry_values(core_global_id: object) -> tuple[str, str]:
    global_field = _global_field_for_id(core_global_id)
    global_id = require_global_id(global_field, core_global_id)
    csv_key, registry_field, name_field = _REGISTRY_SPECS[global_field]
    try:
        row = _registry_rows(csv_key, global_field)[global_id]
    except KeyError as exc:
        raise CoreIDValidationError(
            f"{global_id} is absent from canonical registry {csv_key}"
        ) from exc

    if global_field == "blocktype_num_id":
        registry_id = f"{row['block_type']}|{row['system_type']}"
        name = f"{row['block_type']} ({row['system_type']})"
    else:
        assert registry_field is not None and name_field is not None
        registry_id = row[registry_field]
        name = row[name_field]
    return (
        _require_nonempty_text(registry_id, f"{global_id}.registry_id"),
        _require_nonempty_text(name, f"{global_id}.name"),
    )


def construct_core_id_updates(core_id_updates: object) -> dict[str, list[dict]]:
    """Construct canonical catalog updates from a batch of minimal GLOB IDs."""
    if not isinstance(core_id_updates, list):
        raise TypeError("core_id_updates must be an array")
    enriched: list[dict] = []
    seen: set[str] = set()
    for index, update in enumerate(core_id_updates):
        context = f"core_id_updates[{index}]"
        core = _require_exact(update, CORE_ID_UPDATE_INPUT_FIELDS, context)
        if core["action"] != "add":
            raise CoreIDValidationError(f"{context}.action must be 'add'")
        global_field = _global_field_for_id(core["core_GLOB_id"])
        global_id = require_global_id(global_field, core["core_GLOB_id"])
        if global_id in seen:
            raise CoreIDValidationError(
                f"core_id_updates contains duplicate {global_id}"
            )
        seen.add(global_id)
        registry_id, name = _resolved_registry_values(global_id)
        enriched.append(
            {
                "action": "add",
                "core_GLOB_id": global_id,
                "registry_id": registry_id,
                "name": name,
            }
        )
    return {"core_id_updates": enriched}


class CoreIDManagementToolCatalog(AgentToolCatalog):
    """The two deterministic tools owned by the post-answer ID stage."""

    pipeline_label = "postanswer-core-id"

    def __init__(self) -> None:
        super().__init__(
            entries=(
                ToolEntry(
                    "construct_core_blocks_found",
                    construct_core_blocks_found,
                    group="core_id_construction",
                    skip_compactor=True,
                    skip_subagent=True,
                ),
                ToolEntry(
                    "construct_core_id_updates",
                    construct_core_id_updates,
                    group="core_id_construction",
                    skip_compactor=True,
                    skip_subagent=True,
                ),
            )
        )


CORE_ID_MANAGEMENT_TOOL_CATALOG = CoreIDManagementToolCatalog()


def validate_minimal_core_id_alignment(payload: object) -> dict:
    """Validate the agent-owned shape without adding database context."""
    result = dict(_require_exact(payload, _ID_ALIGNMENT_FIELDS, "ID alignment"))
    if result["status"] not in _STATUSES:
        raise CoreIDValidationError(
            f"status must be one of {sorted(_STATUSES)}"
        )
    if not isinstance(result["summary"], str):
        raise TypeError("summary must be text")
    if not isinstance(result["core_blocks_found"], list):
        raise TypeError("core_blocks_found must be an array")
    if not isinstance(result["core_id_updates"], list):
        raise TypeError("core_id_updates must be an array")
    normalized_blocks = []
    for index, block in enumerate(result["core_blocks_found"]):
        normalized_blocks.append(
            _require_core_block_input(block, f"core_blocks_found[{index}]")
        )
    result["core_blocks_found"] = normalized_blocks
    for index, update in enumerate(result["core_id_updates"]):
        _require_exact(
            update, CORE_ID_UPDATE_INPUT_FIELDS, f"core_id_updates[{index}]"
        )
    return result


def construct_l1_core_id_alignment(
    payload: object,
    *,
    engine_hooks: Any = None,
) -> dict:
    """Run both registered batch tools and combine their constructed fields."""
    core = validate_minimal_core_id_alignment(payload)
    tools = CORE_ID_MANAGEMENT_TOOL_CATALOG.tools
    tool_names = (
        "construct_core_blocks_found",
        "construct_core_id_updates",
    )
    anchor(
        postans_anchor.POSTANS_CORE_ID_TOOLS_BEFORE,
        engine_hooks,
        tool_names=tool_names,
        core_blocks_found=core["core_blocks_found"],
        core_id_updates=core["core_id_updates"],
    )
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="postanswer-core-id-tool",
    ) as pool:
        blocks_future = pool.submit(
            tools["construct_core_blocks_found"],
            core["core_blocks_found"],
        )
        updates_future = pool.submit(
            tools["construct_core_id_updates"],
            core["core_id_updates"],
        )
        errors: list[str] = []
        blocks_result: dict[str, list[dict]] | None = None
        updates_result: dict[str, list[dict]] | None = None
        try:
            blocks_result = blocks_future.result()
        except Exception as exc:
            errors.append(f"construct_core_blocks_found: {exc}")
        try:
            updates_result = updates_future.result()
        except Exception as exc:
            errors.append(f"construct_core_id_updates: {exc}")
    anchor(
        postans_anchor.POSTANS_CORE_ID_TOOLS_AFTER,
        engine_hooks,
        tool_names=tool_names,
        errors=tuple(errors),
        blocks_result=blocks_result,
        updates_result=updates_result,
    )
    if errors:
        raise CoreIDValidationError("; ".join(errors))
    if (
        not isinstance(blocks_result, dict)
        or set(blocks_result) != {"core_blocks_found"}
        or not isinstance(updates_result, dict)
        or set(updates_result) != {"core_id_updates"}
    ):
        raise TypeError("core ID construction tools returned invalid envelopes")
    return {
        "status": core["status"],
        "summary": core["summary"],
        "core_id_updates": updates_result["core_id_updates"],
        "core_blocks_found": blocks_result["core_blocks_found"],
    }


def _correct_core_payload(
    *,
    client: Any,
    base_prompt: str,
    schema: dict[str, Any],
    current: dict[str, Any],
    observation: str,
    label: str,
    cycle: int,
    max_tokens: int,
    engine_hooks: Any = None,
) -> dict[str, Any]:
    anchor(
        postans_anchor.POSTANS_CORE_ID_REFINEMENT,
        engine_hooks,
        label=label,
        cycle=cycle,
        observation=observation,
        current=current,
    )
    summary = current["summary"]
    correction_schema = {
        key: value for key, value in schema.items() if key != "summary"
    }
    editable = {
        key: value for key, value in current.items() if key != "summary"
    }
    prompt = (
        base_prompt
        + "\n\n## ReAct validation observation\n"
        + observation
        + "\n\n## Current minimal core-ID action\n"
        + json.dumps(editable, indent=2, ensure_ascii=False)
        + "\n\nReturn a corrected minimal core-ID JSON action."
    )
    raw = client.call(prompt, _CORRECTION_SYSTEM, max_tokens=max_tokens)
    parsed, _ = guard_json_answer(
        raw,
        client=client,
        label=f"{label}-core-id-refinement-{cycle}",
        schema=correction_schema,
        max_retries=1,
        max_tokens=max_tokens,
    )
    if not isinstance(parsed, dict):
        raise CoreIDValidationError("core-ID refinement returned invalid JSON")
    # Only the top-level shape is checked here; the next cycle validates the
    # records, so a still-broken record becomes the next observation (or the
    # final "exhausted" error) instead of escaping the loop.
    return dict(_require_exact(
        {**parsed, "summary": summary}, _ID_ALIGNMENT_FIELDS, "ID alignment",
    ))


def _review_submission(
    *,
    client: Any,
    answer: str,
    core_claims_evaluation: dict[str, Any],
    enriched_id_alignment: dict[str, Any],
    completed_run_record: str,
    label: str,
    cycle: int,
    max_tokens: int,
    engine_hooks: Any = None,
) -> dict[str, str]:
    candidate = {
        "answer": answer,
        "core_claims": core_claims_evaluation["core_claims"],
        **enriched_id_alignment,
    }
    prompt = (
        "## Working answer\n"
        + answer
        + "\n\n## Candidate final JSON\n"
        + json.dumps(candidate, indent=2, ensure_ascii=False)
    )
    schema = {"decision": "submit | refine", "reason": "brief reason"}
    raw = client.call(prompt, _SUBMISSION_REVIEW_SYSTEM, max_tokens=max_tokens)
    parsed, _ = guard_json_answer(
        raw,
        client=client,
        label=f"{label}-submission-review-{cycle}",
        schema=schema,
        max_retries=1,
        max_tokens=max_tokens,
    )
    if not isinstance(parsed, dict) or set(parsed) != set(schema):
        raise CoreIDValidationError(
            "submission reviewer must return exactly decision and reason"
        )
    decision = parsed["decision"]
    reason = parsed["reason"]
    if decision not in {"submit", "refine"}:
        raise CoreIDValidationError(
            "submission reviewer decision must be 'submit' or 'refine'"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise CoreIDValidationError("submission reviewer reason must be non-empty")
    result = {"decision": decision, "reason": reason.strip()}
    anchor(
        postans_anchor.POSTANS_SUBMISSION_REVIEW,
        engine_hooks,
        label=label,
        cycle=cycle,
        decision=result["decision"],
        reason=result["reason"],
    )
    return result


def refine_and_enrich_l1_core_ids(
    *,
    id_alignment: dict[str, Any],
    client: Any,
    id_alignment_schema: dict[str, Any],
    id_alignment_prompt: str,
    answer: str,
    core_claims_evaluation: dict[str, Any],
    completed_run_record: str,
    label: str,
    max_tokens: int,
    max_cycles: int = 2,
    engine_hooks: Any = None,
) -> dict[str, Any]:
    """Run validation observations and a final adequacy decision as a loop."""
    if not isinstance(max_cycles, int) or max_cycles < 1:
        raise ValueError("max_cycles must be a positive integer")
    # The top-level shape stays fatal; record-level errors (a missing
    # description, a stray key inside a block) are validated inside the loop
    # so they reach the correction call like any other observation.
    current = dict(_require_exact(
        deepcopy(id_alignment), _ID_ALIGNMENT_FIELDS, "ID alignment",
    ))
    last_error: Exception | None = None

    for cycle in range(1, max_cycles + 1):
        try:
            enriched = construct_l1_core_id_alignment(
                current,
                engine_hooks=engine_hooks,
            )
        except (CoreIDValidationError, TypeError, ValueError) as exc:
            last_error = exc
            if cycle == max_cycles:
                break
            current = _correct_core_payload(
                client=client,
                base_prompt=id_alignment_prompt,
                schema=id_alignment_schema,
                current=current,
                observation=str(exc),
                label=label,
                cycle=cycle,
                max_tokens=max_tokens,
                engine_hooks=engine_hooks,
            )
            continue

        review = _review_submission(
            client=client,
            answer=answer,
            core_claims_evaluation=core_claims_evaluation,
            enriched_id_alignment=enriched,
            completed_run_record=completed_run_record,
            label=label,
            cycle=cycle,
            max_tokens=max_tokens,
            engine_hooks=engine_hooks,
        )
        if review["decision"] == "submit":
            return enriched
        last_error = CoreIDValidationError(review["reason"])
        if cycle == max_cycles:
            break
        current = _correct_core_payload(
            client=client,
            base_prompt=id_alignment_prompt,
            schema=id_alignment_schema,
            current=current,
            observation="Final submission review requested refinement: "
            + review["reason"],
            label=label,
            cycle=cycle,
            max_tokens=max_tokens,
            engine_hooks=engine_hooks,
        )

    raise CoreIDValidationError(
        f"{label} core-ID refinement exhausted {max_cycles} cycles: {last_error}"
    )


__all__ = [
    "CORE_BLOCK_INPUT_FIELDS",
    "CORE_BLOCK_REQUIRED_INPUT_FIELDS",
    "CORE_ID_UPDATE_INPUT_FIELDS",
    "CORE_BLOCK_ENRICHED_FIELDS",
    "CORE_ID_UPDATE_ENRICHED_FIELDS",
    "CoreIDValidationError",
    "validate_minimal_core_id_alignment",
    "construct_core_blocks_found",
    "construct_core_id_updates",
    "CoreIDManagementToolCatalog",
    "CORE_ID_MANAGEMENT_TOOL_CATALOG",
    "construct_l1_core_id_alignment",
    "refine_and_enrich_l1_core_ids",
]
