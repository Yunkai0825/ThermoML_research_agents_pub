"""
Canonical ID Alignment Agent — Public API
==========================================
Single entry-point for search field alignment, supporting both the
LLM-driven Argo agent and the deterministic hardcoded pipeline.

Usage::

    from ThermoML_results_browser.cannonical_id_alignment_search_agent.alignment_agent_api import (
        alignment_agent_run,
    )

    # Agentic (LLM) path — default
    result = alignment_agent_run("ethanol + viscosity")

    # Deterministic (fast) path
    result = alignment_agent_run(
        "ethanol + viscosity", settings={'use_agent': False}
    )

    fields = result.fields  # dict of resolved search form fields
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ThermoML_raw_json_to_card_db_parsers.id_schema import attach_lit_num_ids

if TYPE_CHECKING:
    from NIST_ThermoML_agents.general_db_query_engine.general_argo_engine_helpers import AgentTurnResult


class AlignmentResult:
    """Wraps agent/hardcoded results + resolved fields snapshot."""

    def __init__(
        self,
        fields: dict,
        answer: str = "",
        iterations: int = 0,
        elapsed_seconds: float = 0.0,
        timed_out: bool = False,
        resolution_log: list | None = None,
        _agent_result: object | None = None,
        query_groups: list[dict] | None = None,
    ):
        self.answer = answer
        self.iterations = iterations
        self.elapsed_seconds = elapsed_seconds
        self.timed_out = timed_out
        self.fields = fields
        self.resolution_log = resolution_log or []
        self._agent_result = _agent_result
        self.query_groups = query_groups or []

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "iterations": self.iterations,
            "elapsed_seconds": self.elapsed_seconds,
            "timed_out": self.timed_out,
            "fields": self.fields,
            "resolution_log": self.resolution_log,
            "query_groups": self.query_groups,
        }


def alignment_agent_run(
    search_text: str,
    settings: Optional[Dict[str, Any]] = None,
) -> AlignmentResult:
    """Run field alignment on free-text search input.

    Parameters
    ----------
    search_text : str
        Raw search text; semicolons separate independent OR clauses.
    settings : dict, optional
        - ``use_agent`` (bool): ``True`` → Argo LLM agent, ``False`` → deterministic.
          Default: ``True``.
        - ``entry_limit`` (int): Positive maximum entries for the agent.
        - ``max_wait_seconds`` (int): Time budget hint for the agent.

    Returns
    -------
    AlignmentResult
        Contains ``.answer``, ``.fields`` (resolved search form dict),
        ``.iterations``, ``.elapsed_seconds``, and ``.query_groups``. Multiple
        clauses keep separate fields in query_groups and leave .fields empty.
    """
    if not isinstance(search_text, str):
        raise TypeError("search_text must be a string")
    queries = [group.strip() for group in search_text.split(";") if group.strip()]
    if not queries:
        raise ValueError("Enter at least one search query.")

    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise TypeError("settings must be an object")
    allowed_settings = {
        "use_agent", "entry_limit", "max_wait_seconds",
        "allow_query_agent", "agentic_all_fields",
    }
    unknown = sorted(set(settings) - allowed_settings)
    if unknown:
        raise ValueError(f"Unknown alignment settings: {unknown}")
    use_agent = settings["use_agent"] if "use_agent" in settings else True
    if not isinstance(use_agent, bool):
        raise TypeError("settings.use_agent must be a boolean")

    max_wait = settings.get("max_wait_seconds")
    if max_wait is not None and (
        isinstance(max_wait, bool) or not isinstance(max_wait, int) or max_wait < 1
    ):
        raise TypeError("settings.max_wait_seconds must be a positive integer")
    started = time.monotonic()
    deadline = started + max_wait if max_wait is not None else None
    groups = []
    results = []
    for index, query in enumerate(queries, 1):
        group_settings = dict(settings)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Search time budget exhausted before query group {index}: {query}"
                )
            group_settings["max_wait_seconds"] = max(1, math.ceil(remaining))
        result = (
            _run_argo_agent(query, group_settings) if use_agent
            else _run_hardcoded(query, group_settings)
        )
        if result.timed_out:
            raise TimeoutError(f"Query group {index} ({query}) exceeded its time budget.")
        _validate_query_fields(query, index, result.fields)
        groups.append({
            "query": query,
            "fields": result.fields,
            "answer": result.answer,
            "iterations": result.iterations,
            "elapsed_seconds": result.elapsed_seconds,
            "timed_out": result.timed_out,
            "resolution_log": result.resolution_log,
        })
        results.append(result)

    if len(results) == 1:
        results[0].query_groups = groups
        return results[0]
    return AlignmentResult(
        fields={},
        answer="\n\n".join(
            f"{index}. {group['query']}\n{group['answer']}"
            for index, group in enumerate(groups, 1)
        ),
        iterations=sum(result.iterations for result in results),
        elapsed_seconds=time.monotonic() - started,
        timed_out=any(result.timed_out for result in results),
        resolution_log=[
            f"[{group['query']}] {entry}"
            for group in groups for entry in group["resolution_log"]
        ],
        query_groups=groups,
    )


_FILTER_FIELDS = (
    "compounds", "properties", "phases", "measurements", "variables",
    "constraints", "title_keywords", "doi", "authors", "journal",
    "year_min", "year_max", "system_type", "block_type", "n_components",
    "min_datapoints", "formula",
)


def _validate_query_fields(query: str, index: int, fields: dict) -> None:
    """Never turn an unresolved OR clause into an unrestricted search."""
    if fields.get("unresolved"):
        raise ValueError(
            f"Query group {index} ({query}) has unresolved terms: {fields['unresolved']}"
        )
    if not any(fields.get(name) for name in _FILTER_FIELDS):
        raise ValueError(f"Query group {index} ({query}) did not resolve any search filters.")


# ── Argo agent path ──────────────────────────────────────────────────────

def _run_argo_agent(
    search_text: str,
    settings: Dict[str, Any],
) -> AlignmentResult:
    """LLM-driven Argo ReAct alignment."""
    from .alignment_agent_workflows.L0_orchestrator.orchestrator import run
    from .alignment_agent_toolbox.mcp_tools import alignment_fields_context

    # The form state must remain active through both the ReAct run and the
    # result snapshot.  The context is then restored so concurrent and later
    # browser requests cannot observe this request's mutable fields.
    with alignment_fields_context() as fields:
        agent_result = run(search_text, settings=settings)
        fields_snapshot = attach_lit_num_ids(
            fields.snapshot(),
            path="alignment_result.fields",
        )
        if not isinstance(fields_snapshot, dict):
            raise AssertionError("Alignment field enrichment changed result type")

    return AlignmentResult(
        fields=fields_snapshot,
        answer=agent_result.answer,
        iterations=agent_result.iterations,
        elapsed_seconds=agent_result.elapsed_seconds,
        timed_out=agent_result.timed_out,
        resolution_log=fields_snapshot["resolution_log"],
        _agent_result=agent_result,
    )


# ── Deterministic hardcoded path ─────────────────────────────────────────

def _run_hardcoded(
    search_text: str,
    settings: Dict[str, Any],
) -> AlignmentResult:
    """Fast deterministic pipeline: parse → resolve → validate → review."""
    t0 = time.time()

    from .hardcoded_search_helpers.dispatcher import dispatch_search

    rqs = dispatch_search(search_text)
    if not rqs:
        return AlignmentResult(
            fields={},
            answer="No search blocks parsed.",
            elapsed_seconds=time.time() - t0,
            resolution_log=["No parseable search blocks found."],
        )

    rq = rqs[0]  # primary AND-block

    # Convert ResolvedQuery → fields dict (same shape as the Argo path)
    fields: Dict[str, Any] = {"unresolved": list(rq.unresolved)}
    if rq.compounds:
        fields['compounds'] = [
            {'name': c.display_name, 'global_id': c.global_id, 'score': c.score}
            for c in rq.compounds
        ]
    if rq.properties:
        fields['properties'] = [
            {'name': p.display_name, 'global_id': p.global_id, 'score': p.score}
            for p in rq.properties
        ]
    if rq.measurements:
        fields['measurements'] = [
            {'name': m.display_name, 'global_id': m.global_id, 'score': m.score}
            for m in rq.measurements
        ]
    if rq.variables:
        fields['variables'] = [
            {'name': v.display_name, 'global_id': v.global_id, 'score': v.score}
            for v in rq.variables
        ]
    if rq.constraints:
        fields['constraints'] = [
            {'name': c.display_name, 'global_id': c.global_id, 'score': c.score}
            for c in rq.constraints
        ]
    if rq.title_words:
        fields['title_keywords'] = ' '.join(rq.title_words)
    if rq.dois:
        fields['doi'] = rq.dois[0]
    if rq.authors:
        fields['authors'] = rq.authors[0]
    if rq.journals:
        fields['journal'] = rq.journals[0]
    if rq.year_min:
        fields['year_min'] = rq.year_min
    if rq.year_max:
        fields['year_max'] = rq.year_max
    if rq.formulas:
        fields['formula'] = rq.formulas[0]
    if rq.system_type:
        fields['system_type'] = rq.system_type
    if rq.block_type:
        fields['block_type'] = rq.block_type
    if rq.n_components:
        fields['n_components'] = rq.n_components
    if rq.min_datapoints:
        fields['min_datapoints'] = rq.min_datapoints

    fields = attach_lit_num_ids(fields, path="alignment_result.fields")
    if not isinstance(fields, dict):
        raise AssertionError("Alignment field enrichment changed result type")
    elapsed = time.time() - t0
    return AlignmentResult(
        fields=fields,
        answer=f"Deterministic alignment: resolved {len(rq.compounds)} compounds, "
               f"{len(rq.properties)} properties in {elapsed:.2f}s.",
        elapsed_seconds=elapsed,
        resolution_log=rq.resolution_log,
    )
