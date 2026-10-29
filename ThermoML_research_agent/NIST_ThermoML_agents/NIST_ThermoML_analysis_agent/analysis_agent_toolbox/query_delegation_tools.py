"""Query delegation tools — thin L0 wrappers around L1 query dispatch.

These tools are called by the analysis agent's L0 orchestrator.
They delegate all database searching to the ThermoML query agent's
L1 workers.  The analysis agent never touches the database directly.
"""

from __future__ import annotations

import json
import logging

from ThermoML_raw_json_to_card_db_parsers.id_schema import (
    require_global_id,
)

from ..analysis_agent_workflows.L1_workers.l1_query_delegation import (
    dispatch_query,
    dispatch_queries_parallel,
)
from ...general_db_query_engine.general_text_context_marker_catalog import (
    mark_subagent_answer_tool,
)
from ...NIST_ThermoML_query_agent.query_agent_workflows.strict_output_contracts import (
    validate_l1_query_output,
)

log = logging.getLogger("ANALYSIS-QUERY-TOOLS")

_GLOBAL_FIELD_BY_TYPE = {
    "lit": "lit_num_id",
    "comp": "comp_num_id",
    "prop": "prop_num_id",
    "var": "var_num_id",
    "constr": "constr_num_id",
    "meas": "meas_num_id",
    "phase": "phase_num_id",
    "blocktype": "blocktype_num_id",
    "rxntype": "rxn_type_num_id",
    "solvent": "solvent_num_id",
}


def _validate_id_catalog(catalog: object, *, context: str) -> list[dict]:
    if not isinstance(catalog, list):
        raise TypeError(f"{context} must be an array")
    validated: list[dict] = []
    required = {"type", "global_id", "registry_id", "name"}
    for index, entry in enumerate(catalog):
        if not isinstance(entry, dict) or set(entry) != required:
            received = (sorted(entry) if isinstance(entry, dict)
                        else type(entry).__name__)
            raise ValueError(
                f"{context}[{index}] must contain exactly {sorted(required)}; "
                f"received {received}"
            )
        entity_type = entry["type"]
        if entity_type not in _GLOBAL_FIELD_BY_TYPE:
            raise ValueError(
                f"{context}[{index}].type is invalid: {entity_type!r}; "
                f"valid types: {sorted(_GLOBAL_FIELD_BY_TYPE)}"
            )
        require_global_id(_GLOBAL_FIELD_BY_TYPE[entity_type], entry["global_id"])
        if not isinstance(entry["registry_id"], str) or not isinstance(entry["name"], str):
            raise TypeError(f"{context}[{index}] registry_id and name must be strings")
        validated.append(dict(entry))
    return validated


def _validate_query_result(parsed: object, *, context: str) -> dict:
    try:
        return validate_l1_query_output(parsed)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{context}: {exc}") from exc


@mark_subagent_answer_tool
def query_thermoml(
    purpose: str,
    instruction: str = "",
    id_catalog: list[dict] | None = None,
    context: str = "",
) -> dict:
    """Send a database query to the ThermoML query agent.

    The query agent resolves compound/property IDs, searches for
    matching data blocks, and returns structured results.

    Parameters
    ----------
    purpose : str
        High-level goal (e.g. "Find viscosity data for water + ethanol").
    instruction : str
        Detailed search instructions (e.g. "Search for binary systems
        with viscosity at 298.15 K, return DOIs and block numbers").
    id_catalog : list of dict, optional
        Already-resolved strict ID catalog entries. Each entry contains
        exactly the keys ``type``, ``global_id``, ``registry_id``, and
        ``name``. Valid ``type`` values: ``lit``, ``comp``, ``prop``,
        ``var``, ``constr``, ``meas``, ``phase``, ``blocktype``,
        ``rxntype``, ``solvent`` (e.g. ``comp`` — never ``compound``).
    context : str
        Prior context or constraints (optional).

    Returns
    -------
    dict
        The query worker's strict structured result.
    """
    validated_catalog = _validate_id_catalog(
        [] if id_catalog is None else id_catalog,
        context="id_catalog",
    )
    result = dispatch_query(
        purpose=purpose,
        instruction=instruction,
        id_catalog=json.dumps(validated_catalog, ensure_ascii=False),
        context=context,
    )
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError as exc:
        raise ValueError("Query L1 returned non-JSON content") from exc
    return _validate_query_result(parsed, context="Query L1 result")


@mark_subagent_answer_tool
def query_thermoml_parallel(queries: list[dict]) -> dict:
    """Send multiple database queries to the ThermoML query agent in parallel.

    Parameters
    ----------
    queries : list of dict
        Array of query specs, each with keys:
        - label: identifier for this query
        - purpose: high-level goal
        - instruction: detailed search instructions (optional)
        - id_catalog: resolved IDs (optional)
        - context: prior context (optional)

    Returns
    -------
    dict
        Ordered structured results from all parallel queries.
    """
    if not queries:
        raise ValueError("queries must contain at least one query specification")
    required = {"label", "purpose", "instruction", "id_catalog", "context"}
    validated: list[dict] = []
    for index, spec in enumerate(queries):
        if not isinstance(spec, dict):
            raise TypeError(f"queries[{index}] must be an object")
        missing = sorted(required - set(spec))
        unknown = sorted(set(spec) - required)
        if missing or unknown:
            raise ValueError(
                f"queries[{index}] has missing={missing} and unknown={unknown} fields"
            )
        purpose = spec["purpose"]
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError(f"queries[{index}].purpose must be a non-empty string")
        if not isinstance(spec["label"], str) or not spec["label"]:
            raise ValueError(f"queries[{index}].label must be a non-empty string")
        id_catalog = _validate_id_catalog(
            spec["id_catalog"], context=f"queries[{index}].id_catalog"
        )
        if not isinstance(spec["instruction"], str) or not isinstance(spec["context"], str):
            raise TypeError(f"queries[{index}] instruction and context must be strings")
        validated.append({
            "label": spec["label"],
            "purpose": purpose,
            "instruction": spec["instruction"],
            "id_catalog": json.dumps(id_catalog, ensure_ascii=False),
            "context": spec["context"],
        })

    raw_results = dispatch_queries_parallel(validated)
    results: list[dict] = []
    for index, item in enumerate(raw_results):
        label = item["label"]
        if "error" in item:
            entry = {"label": label, "error": item["error"]}
            if "salvage" in item:
                entry["salvage"] = item["salvage"]
            results.append(entry)
            continue
        try:
            parsed = json.loads(item["result"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Parallel query {label!r} returned non-JSON content") from exc
        results.append({
            "label": label,
            "result": _validate_query_result(parsed, context=f"Parallel query {label!r}"),
        })
    return {"n_queries": len(results), "results": results}


# ── Catalog entries ─────────────────────────────────────────────────────
from ...general_db_query_engine.general_tool_management_helpers.general_agent_tool_catalog import ToolEntry

TOOL_ENTRIES = [
    ToolEntry(
        "query_thermoml", query_thermoml,
        group="query_delegation",
        skip_compactor=True, skip_subagent=True,
    ),
    ToolEntry(
        "query_thermoml_parallel", query_thermoml_parallel,
        group="query_delegation",
        skip_compactor=True, skip_subagent=True,
    ),
]


def _query_l1_tools():
    from ...NIST_ThermoML_query_agent.query_agent_workflows.L1_workers.l1_query_dispatcher import (
        _get_catalog,
    )
    return _get_catalog().wrapped_tools()


from ...general_db_query_engine.general_tool_management_helpers.tool_capabilities import (
    bind_reachable_tools,
)

bind_reachable_tools(query_thermoml, _query_l1_tools)
bind_reachable_tools(query_thermoml_parallel, _query_l1_tools)
