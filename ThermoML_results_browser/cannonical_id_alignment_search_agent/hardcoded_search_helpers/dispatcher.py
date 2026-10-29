"""
Dispatcher — miniature orchestrator for the canonical ID alignment agent.

Takes parsed SearchBlocks from the search_parser and resolves each
token into canonical global IDs using the ID workers.  Returns a
``ResolvedQuery`` that the browser search engine can convert to SQL.

Workflow per SearchBlock:
  1. Hard-parse tokens (already done by search_parser)
  2. For each entity-type bucket (compounds, properties, etc.),
     run the corresponding ID worker to get canonical global IDs
  3. Preserve unresolved tokens for explicit refinement; never reclassify
     them through a different entity registry
  4. Assemble ResolvedQuery with all resolved IDs
  5. **Validation step**: validate all fields via validation_tools
  6. **Review workers**: dispatch review workers to examine top
     search results (especially literature) and suggest field edits
  7. **Escalation**: if validation fails or workers cannot resolve,
     escalate to the ThermoML query agent

The dispatcher is deterministic by default.  Agentic escalation
(ThermoML query agent) is optional and controlled by settings.
"""

from __future__ import annotations

import sqlite3
import os
from dataclasses import dataclass, field

from .search_parser import SearchBlock, parse_search_input
from .id_workers import (
    ResolvedID,
    resolve_compound,
    resolve_property,
    resolve_measurement,
    resolve_variable,
    resolve_constraint,
    resolve_reference,
)


@dataclass
class ResolvedQuery:
    """Fully resolved search query with canonical IDs, ready for SQL."""
    # Each list entry is a resolved ID (may have multiple per entity type)
    compounds: list[ResolvedID] = field(default_factory=list)
    properties: list[ResolvedID] = field(default_factory=list)
    measurements: list[ResolvedID] = field(default_factory=list)
    variables: list[ResolvedID] = field(default_factory=list)
    constraints: list[ResolvedID] = field(default_factory=list)

    # Direct filters (no resolution needed)
    dois: list[str] = field(default_factory=list)
    title_words: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    journals: list[str] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    system_type: str = ""
    block_type: str = ""
    min_datapoints: int | None = None
    n_components: int | None = None

    # Diagnostics
    unresolved: list[str] = field(default_factory=list)
    resolution_log: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([
            self.compounds, self.properties, self.measurements,
            self.variables, self.constraints, self.dois,
            self.title_words, self.authors, self.journals,
            self.formulas, self.year_min, self.year_max,
            self.system_type, self.block_type, self.min_datapoints,
            self.n_components,
        ])

    def to_dict(self) -> dict:
        d = {}
        for k in ('compounds', 'properties', 'measurements',
                   'variables', 'constraints'):
            items = getattr(self, k)
            if items:
                d[k] = [r.to_dict() for r in items]
        for k in ('dois', 'title_words', 'authors', 'journals',
                   'formulas', 'unresolved', 'resolution_log'):
            v = getattr(self, k)
            if v:
                d[k] = v
        for k in ('year_min', 'year_max', 'min_datapoints', 'n_components'):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        for k in ('system_type', 'block_type'):
            v = getattr(self, k)
            if v:
                d[k] = v
        return d


# ── Dispatch: SearchBlock → ResolvedQuery ─────────────────────────────────

def resolve_block(block: SearchBlock) -> ResolvedQuery:
    """Resolve a single SearchBlock into canonical IDs."""
    rq = ResolvedQuery()

    # Pass through direct filters
    rq.dois = list(block.dois)
    rq.title_words = list(block.title_words)
    rq.authors = list(block.authors)
    rq.journals = list(block.journals)
    rq.formulas = list(block.formulas)
    rq.year_min = block.year_min
    rq.year_max = block.year_max
    rq.system_type = block.system_type
    rq.block_type = block.block_type
    rq.min_datapoints = block.min_datapoints
    rq.n_components = block.n_components

    # Resolve compounds
    for name in block.compounds:
        hits = resolve_compound(name, limit=3)
        if hits:
            best = hits[0]
            rq.compounds.append(best)
            rq.resolution_log.append(
                f"compound '{name}' → {best.display_name} "
                f"(global_id={best.global_id}, score={best.score}, via={best.match_type})"
            )
        else:
            rq.unresolved.append(f"compound:{name}")
            rq.resolution_log.append(f"compound '{name}' → no registry match")

    # Resolve properties (by keyword match from parser)
    for kw in block.properties:
        hits = resolve_property(kw, limit=3)
        if hits:
            best = hits[0]
            rq.properties.append(best)
            rq.resolution_log.append(
                f"property '{kw}' → {best.display_name} "
                f"(global_id={best.global_id}, score={best.score})"
            )
        else:
            rq.unresolved.append(f"prop:{kw}")
            rq.resolution_log.append(f"property '{kw}' → no match")

    # Resolve measurements
    for kw in block.measurements:
        hits = resolve_measurement(kw, limit=3)
        if hits:
            best = hits[0]
            rq.measurements.append(best)
            rq.resolution_log.append(
                f"measurement '{kw}' → {best.display_name} "
                f"(global_id={best.global_id}, score={best.score})"
            )
        else:
            rq.unresolved.append(f"meas:{kw}")
            rq.resolution_log.append(f"measurement '{kw}' → no match")

    # Resolve variables
    for kw in block.variables:
        hits = resolve_variable(kw, limit=3)
        if hits:
            rq.variables.append(hits[0])

    # Resolve constraints
    for kw in block.constraints:
        hits = resolve_constraint(kw, limit=3)
        if hits:
            rq.constraints.append(hits[0])

    # Unclassified text is never guessed into a different entity namespace.
    for token in block.unresolved:
        rq.unresolved.append(token)
        rq.resolution_log.append(f"unresolved '{token}' requires explicit refinement")

    return rq


# ── Main entry point: raw text → list[ResolvedQuery] ─────────────────────

def dispatch_search(raw_text: str) -> list[ResolvedQuery]:
    """Parse and resolve a free-text search into canonical IDs.

    Returns one ResolvedQuery per OR-group (separated by ``;``).
    Most queries will return a single-element list.
    """
    blocks = parse_search_input(raw_text)
    if not blocks:
        return []
    return [resolve_block(b) for b in blocks]


# ── Enhanced entry point with validation + review ─────────────────────────

def dispatch_search_validated(
    raw_text: str,
    db_path: str = "",
    *,
    enable_validation: bool = True,
    enable_review_workers: bool = True,
    enable_query_agent: bool = False,
    auto_apply_edits: bool = True,
    min_edit_confidence: int = 60,
) -> dict:
    """Parse, resolve, validate, and review a free-text search.

    Full pipeline:
      1. Parse + resolve → ResolvedQuery
      2. Validate all fields → ValidationReport
      3. Run review workers on top results → FieldEdits
      4. Apply accepted edits
      5. If still failing, optionally escalate to query agent

    Parameters
    ----------
    raw_text : str
        Free-text search input.
    db_path : str
        Path to ThermoML_index.db.
    enable_validation : bool
        Run the field validator after initial resolution.
    enable_review_workers : bool
        Dispatch review workers to refine fields.
    enable_query_agent : bool
        Allow escalation to the ThermoML query agent.
    auto_apply_edits : bool
        Automatically apply worker-suggested edits.
    min_edit_confidence : int
        Minimum confidence for auto-applied edits.

    Returns
    -------
    dict with keys:
        resolved_queries : list[ResolvedQuery]
        validation_report : ValidationReport | None
        review_results : list[ReviewResult]
        applied_edits : list[str]
        escalated : bool
        escalation_result : str | None
    """
    from .validation_tools import validate_all_fields
    from .review_workers import run_all_workers, apply_edits
    from .bibliography_scorer import score_titles

    # Step 1: Parse + resolve
    rqs = dispatch_search(raw_text)
    if not rqs:
        return {
            "resolved_queries": [],
            "validation_report": None,
            "review_results": [],
            "applied_edits": [],
            "escalated": False,
            "escalation_result": None,
        }

    rq = rqs[0]  # primary AND-block
    validation_report = None
    review_results = []
    applied_edits = []
    escalated = False
    escalation_result = None

    # Step 2: Validate
    if enable_validation:
        validation_report = validate_all_fields(rq, db_path=db_path)
        rq.resolution_log.append(
            f"Validation: {validation_report.overall_status} "
            f"(escalation={'needed' if validation_report.needs_escalation else 'not needed'})"
        )

    # Step 3: Review workers
    if enable_review_workers:
        title_hits = None
        if rq.title_words and db_path:
            kw = " ".join(rq.title_words)
            title_hits = score_titles(kw, db_path, limit=15)
        review_results = run_all_workers(rq, title_hits=title_hits, db_path=db_path)

        # Step 4: Apply accepted edits
        if auto_apply_edits:
            all_edits = []
            for rr in review_results:
                all_edits.extend(rr.edits)
            if all_edits:
                applied_edits = apply_edits(rq, all_edits, min_confidence=min_edit_confidence)
                for log_line in applied_edits:
                    rq.resolution_log.append(f"Review edit: {log_line}")

    # Step 5: Escalation check
    needs_escalation = False
    if validation_report and validation_report.needs_escalation:
        needs_escalation = True
    for rr in review_results:
        if rr.needs_escalation:
            needs_escalation = True

    if needs_escalation and enable_query_agent:
        escalated = True
        escalation_result = _escalate_to_query_agent(raw_text, rq)
        rq.resolution_log.append(f"Escalated to query agent: {escalation_result or 'no result'}")

    return {
        "resolved_queries": [item.to_dict() for item in rqs],
        "validation_report": (
            validation_report.to_dict() if validation_report is not None else None
        ),
        "review_results": [item.to_dict() for item in review_results],
        "applied_edits": applied_edits,
        "escalated": escalated,
        "escalation_result": escalation_result,
    }


def _escalate_to_query_agent(raw_text: str, rq: ResolvedQuery) -> str:
    """Escalate an ambiguous query to the ThermoML query agent.

    The query agent uses LLM reasoning to interpret the user's intent
    and returns structured search parameters.
    """
    from ..alignment_agent_toolbox.mcp_tools import escalate_to_query_agent

    context_parts = []
    if rq.unresolved:
        context_parts.append(f"Unresolved terms: {', '.join(rq.unresolved)}")
    if rq.compounds:
        context_parts.append(
            f"Partially resolved compounds: "
            f"{', '.join(c.display_name for c in rq.compounds)}"
        )
    context = "; ".join(context_parts)

    question = (
        f"I need to identify the correct ThermoML canonical IDs for this search: "
        f"{raw_text}. {context}. "
        f"Please return the canonical compound IDs, property IDs, and any "
        f"other relevant entity IDs."
    )

    return escalate_to_query_agent(question)["answer"]


# ── SQL builder: ResolvedQuery → WHERE clauses for index DB ──────────────

def build_index_sql(rq: ResolvedQuery) -> tuple[str, str, list]:
    """Convert a ResolvedQuery into SQL for the ThermoML index DB.

    Returns (select_sql, count_sql, params) ready for execution.
    """
    joins: list[str] = []
    wheres: list[str] = []
    params: list = []

    # -- Bibliography --
    for doi in rq.dois:
        wheres.append("r.doi LIKE ?")
        params.append(f"%{doi}%")
    for word in rq.title_words:
        wheres.append("r.title LIKE ?")
        params.append(f"%{word}%")
    for author in rq.authors:
        wheres.append("r.first_author LIKE ?")
        params.append(f"%{author}%")
    for journal in rq.journals:
        wheres.append("r.journal = ?")
        params.append(journal)
    if rq.year_min:
        wheres.append("r.year >= ?")
        params.append(rq.year_min)
    if rq.year_max:
        wheres.append("r.year <= ?")
        params.append(rq.year_max)

    # -- Compounds (by strict canonical global ID) --
    for i, comp in enumerate(rq.compounds):
        alias = f"bc_{i}"
        joins.append(
            f"JOIN block_compounds {alias} ON {alias}.doi = bi.doi "
            f"AND {alias}.block_number = bi.block_number "
            f"AND {alias}.block_type = bi.block_type"
        )
        wheres.append(f"{alias}.comp_num_id = ?")
        params.append(comp.global_id)

    # Formula is an explicit independent user filter.
    for formula in rq.formulas:
        alias_f = "bcf"
        if alias_f not in " ".join(joins):
            joins.append(
                f"JOIN block_compounds {alias_f} ON {alias_f}.doi = bi.doi "
                f"AND {alias_f}.block_number = bi.block_number "
                f"AND {alias_f}.block_type = bi.block_type"
            )
        wheres.append(f"{alias_f}.comp_formula LIKE ?")
        params.append(f"%{formula}%")

    # -- Properties (by strict canonical global ID) --
    for i, prop in enumerate(rq.properties):
        alias = f"bp_{i}"
        joins.append(
            f"JOIN block_properties {alias} ON {alias}.doi = bi.doi "
            f"AND {alias}.block_number = bi.block_number "
            f"AND {alias}.block_type = bi.block_type"
        )
        wheres.append(f"{alias}.prop_num_id = ?")
        params.append(prop.global_id)

    # -- Measurements (by strict canonical global ID) --
    for i, meas in enumerate(rq.measurements):
        alias = f"bm_{i}"
        joins.append(
            f"JOIN block_measurements {alias} ON {alias}.doi = bi.doi "
            f"AND {alias}.block_number = bi.block_number "
            f"AND {alias}.block_type = bi.block_type"
        )
        wheres.append(f"{alias}.meas_num_id = ?")
        params.append(meas.global_id)

    # -- Variables (by strict canonical global ID) --
    for i, var in enumerate(rq.variables):
        alias = f"bv_{i}"
        joins.append(
            f"JOIN block_variables {alias} ON {alias}.doi = bi.doi "
            f"AND {alias}.block_number = bi.block_number "
            f"AND {alias}.block_type = bi.block_type"
        )
        wheres.append(f"{alias}.var_num_id = ?")
        params.append(var.global_id)

    # -- Constraints (by strict canonical global ID) --
    for i, con in enumerate(rq.constraints):
        alias = f"bcon_{i}"
        joins.append(
            f"JOIN block_constraints {alias} ON {alias}.doi = bi.doi "
            f"AND {alias}.block_number = bi.block_number "
            f"AND {alias}.block_type = bi.block_type"
        )
        wheres.append(f"{alias}.constr_num_id = ?")
        params.append(con.global_id)

    # -- Direct filters --
    if rq.system_type:
        wheres.append("bi.system_type = ?")
        params.append(rq.system_type)
    if rq.block_type:
        wheres.append("bi.block_type = ?")
        params.append(rq.block_type)
    if rq.n_components:
        wheres.append("bi.n_components = ?")
        params.append(rq.n_components)
    if rq.min_datapoints:
        wheres.append("bi.n_datapoints >= ?")
        params.append(rq.min_datapoints)

    # -- Assemble SQL --
    base_cols = (
        "bi.doi, bi.block_number, bi.block_type, bi.compound_system, "
        "bi.n_datapoints, bi.system_type, bi.n_components"
    )
    select_sql = f"SELECT {base_cols} FROM block_index bi JOIN ref_index r ON r.doi = bi.doi"
    for j in joins:
        select_sql += f" {j}"
    if wheres:
        select_sql += " WHERE " + " AND ".join(wheres)

    count_sql = (
        "SELECT COUNT(*), COUNT(DISTINCT bi.doi), COALESCE(SUM(bi.n_datapoints), 0) "
        "FROM block_index bi JOIN ref_index r ON r.doi = bi.doi"
    )
    for j in joins:
        count_sql += f" {j}"
    if wheres:
        count_sql += " WHERE " + " AND ".join(wheres)

    return select_sql, count_sql, params
