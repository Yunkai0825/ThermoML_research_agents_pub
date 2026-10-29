"""
MCP tools for the canonical ID alignment search agent.
=====================================================
These are the callable tools the Argo ReAct agent can invoke during
its loop.  Each tool is a simple Python function that the agent_turn
engine can dispatch.

Tool categories
---------------
1. **ID Resolution** — resolve free-text → canonical IDs
2. **Field Validation** — check resolved IDs against registries
3. **Bibliography Search** — score paper titles by keyword relevance
4. **Review & Edit** — run review workers, apply edits
5. **Field Setters** — set individual search form fields
6. **Query Agent Escalation** — call the ThermoML query agent

All tools return native JSON-serialisable objects.  The ReAct engine performs
the one serialization step at the tool boundary; tools never return a second,
string-encoded JSON layer.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from datetime import datetime
from pathlib import Path
import os
import tempfile

from ThermoML_results_browser import _RESEARCH_ROOT

from card_db_search_tools.basic_search_tools.normalization_helpers.strict_id_inputs import (
    IdentifierRefinementError,
    require_global_tool_id,
)

# ── Paths ─────────────────────────────────────────────────────

_CARD_DB_DIR = str(_RESEARCH_ROOT / 'card_databases_storage')
_DB_PATH = os.path.join(_CARD_DB_DIR, 'ThermoML_index.db')
_QUERY_OUTPUT_ROOT = _RESEARCH_ROOT.parent / '_output' / 'Query'


# ── Shared state: the search form fields being built ──────────────────────

class _SearchFields:
    """Mutable state holding the current search form field values.

    Initialised at the start of each alignment run.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.compounds = []       # list of {"name": str, "global_id": str, "score": int}
        self.properties = []
        self.measurements = []
        self.variables = []
        self.constraints = []
        self.phases = []
        self.doi = ""
        self.authors = ""
        self.title_keywords = ""
        self.journal = ""
        self.year_min = None
        self.year_max = None
        self.formula = ""
        self.system_type = ""
        self.block_type = ""
        self.n_components = None
        self.min_datapoints = None
        self.unresolved = []
        self.resolution_log = []

    def snapshot(self) -> dict:
        """Return current state as a dict."""
        return {
            "compounds": list(self.compounds),
            "properties": list(self.properties),
            "measurements": list(self.measurements),
            "variables": list(self.variables),
            "constraints": list(self.constraints),
            "phases": list(self.phases),
            "doi": self.doi or None,
            "authors": self.authors,
            "title_keywords": self.title_keywords,
            "journal": self.journal,
            "year_min": self.year_min,
            "year_max": self.year_max,
            "formula": self.formula,
            "system_type": self.system_type,
            "block_type": self.block_type,
            "n_components": self.n_components,
            "min_datapoints": self.min_datapoints,
            "unresolved": list(self.unresolved),
            "resolution_log": list(self.resolution_log),
        }


# Context-local form state.  Concurrent Flask requests and nested agent runs
# must never share mutable search fields.
_fields_context: ContextVar[_SearchFields | None] = ContextVar(
    "thermoml_alignment_search_fields",
    default=None,
)


def get_fields() -> _SearchFields:
    fields = _fields_context.get()
    if fields is None:
        fields = _SearchFields()
        _fields_context.set(fields)
    return fields


@contextmanager
def alignment_fields_context():
    """Own one alignment form from public API entry through result snapshot."""
    fields = _SearchFields()
    token = _fields_context.set(fields)
    try:
        yield fields
    finally:
        _fields_context.reset(token)


# ═══════════════════════════════════════════════════════════════
#  Tool 1: resolve_entity — resolve a free-text query to canonical IDs
# ═══════════════════════════════════════════════════════════════

def resolve_entity(entity_type: str, query: str, limit: int = 5) -> dict:
    """Resolve a free-text name to canonical ThermoML IDs.

    Parameters
    ----------
    entity_type : str
        One of: compound, property, measurement, variable, constraint, phase
    query : str
        Free-text name or keyword to resolve (e.g. "ethanol", "viscosity").
    limit : int
        Maximum number of candidates to return (default 5).

    Returns a native result object with ``entity_type``, ``query``,
    ``n_results``, and a ``results`` array. Each candidate has:
      entity_type, global_id, canonical_id, display_name, score, match_type
    """
    from ..hardcoded_search_helpers.id_workers import _resolve
    hits = _resolve(entity_type, query, limit=limit, min_score=30)
    rows = [h.to_dict() for h in hits]
    return {"entity_type": entity_type, "query": query, "n_results": len(rows), "results": rows}


# ═══════════════════════════════════════════════════════════════
#  Tool 2: search_titles — score paper titles by keyword relevance
# ═══════════════════════════════════════════════════════════════

def search_titles(keywords: str, limit: int = 15) -> dict:
    """Search paper titles by keyword relevance scoring.

    Parameters
    ----------
    keywords : str
        Space-separated keywords (e.g. "viscosity ethanol binary").
    limit : int
        Maximum results to return (default 15).

    Returns a native result object containing a ``results`` array of title
    hits with: doi, title, first_author,
    year, journal, score, match_detail, matched_words, n_blocks,
    total_datapoints.
    """
    from ..hardcoded_search_helpers.bibliography_scorer import score_titles
    hits = score_titles(keywords, _DB_PATH, limit=limit, min_score=30)
    rows = [h.to_dict() for h in hits]
    return {"keywords": keywords, "n_results": len(rows), "results": rows}


# ═══════════════════════════════════════════════════════════════
#  Tool 3: validate_field — check a resolved value against registry
# ═══════════════════════════════════════════════════════════════

def validate_field(entity_type: str, value: str) -> dict:
    """Validate a single field value against its canonical registry.

    Parameters
    ----------
    entity_type : str
        One of: compound, property, measurement, variable, constraint, phase
    value : str
        The value to validate (e.g. a compound name or property keyword).

    Returns a native verdict object with: field_name, original_value, status
    (pass/warn/fail), confidence, best_match, alternatives, message.
    """
    from ..hardcoded_search_helpers.validation_tools import validate_field as _validate
    verdict = _validate(entity_type, value)
    return verdict.to_dict()


# ═══════════════════════════════════════════════════════════════
#  Tool 4: validate_bibliography — check title/author/DOI
# ═══════════════════════════════════════════════════════════════

def validate_bibliography(title_keywords: str = "",
                          author: str = "",
                          doi: str = "") -> dict:
    """Validate bibliography fields against the reference index.

    Parameters
    ----------
    title_keywords : str
        Keywords to search in paper titles.
    author : str
        Author surname to check.
    doi : str
        DOI to verify exists in the database.

    Returns a native verdict object with title/author/DOI status and matching titles.
    """
    from ..hardcoded_search_helpers.validation_tools import validate_bibliography as _validate_bib
    verdict = _validate_bib(
        title_kw=title_keywords, author=author, doi=doi, db_path=_DB_PATH
    )
    return verdict.to_dict()


# ═══════════════════════════════════════════════════════════════
#  Tool 5: set_field — set a search form field value
# ═══════════════════════════════════════════════════════════════

def set_field(field_name: str, value: str, global_id: str = "",
              score: int = 0) -> dict:
    """Set a search form field value.

    Parameters
    ----------
    field_name : str
        Field to set. One of: compound, property, measurement, variable,
        constraint, phase, doi, authors, title_keywords, journal,
        formula, system_type, block_type. Numeric fields use their typed block
        fillers rather than this string setter.
    value : str
        The value to set (display name or text).
    global_id : str
        Strict canonical ID such as GLOBcomp_51. Required for entity fields.
    score : int
        Resolution confidence score (0-100).

    Returns confirmation string.
    """
    f = get_fields()
    if not isinstance(field_name, str) or not field_name:
        raise TypeError("field_name must be a non-empty string")
    if not isinstance(value, str) or not value:
        raise TypeError("value must be a non-empty string")
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
        raise TypeError("score must be an integer from 0 through 100")
    entity_fields = {
        "compound", "property", "measurement", "variable", "constraint", "phase",
    }
    entry = None
    if field_name in entity_fields:
        try:
            checked_id = require_global_tool_id(
                field_name, global_id, field="global_id"
            )
        except IdentifierRefinementError as exc:
            return exc.as_result()
        entry = {"name": value, "global_id": checked_id, "score": score}

    # Entity fields (append to list)
    if field_name == "compound":
        f.compounds.append(entry)
        f.resolution_log.append(f"Set compound: {value} (global_id={global_id}, score={score})")
    elif field_name == "property":
        f.properties.append(entry)
        f.resolution_log.append(f"Set property: {value} (global_id={global_id}, score={score})")
    elif field_name == "measurement":
        f.measurements.append(entry)
        f.resolution_log.append(f"Set measurement: {value} (global_id={global_id}, score={score})")
    elif field_name == "variable":
        f.variables.append(entry)
        f.resolution_log.append(f"Set variable: {value} (global_id={global_id}, score={score})")
    elif field_name == "constraint":
        f.constraints.append(entry)
        f.resolution_log.append(f"Set constraint: {value} (global_id={global_id}, score={score})")
    elif field_name == "phase":
        f.phases.append(entry)
        f.resolution_log.append(f"Set phase: {value} (global_id={global_id}, score={score})")
    # Scalar fields
    elif field_name == "doi":
        f.doi = value
    elif field_name == "authors":
        f.authors = value
    elif field_name == "title_keywords":
        f.title_keywords = value
    elif field_name == "journal":
        f.journal = value
    elif field_name == "formula":
        f.formula = value
    elif field_name == "system_type":
        allowed_system_types = {
            "unary", "binary", "ternary", "quaternary",
            "5-component", "6-component", "7-component", "8-component", "9-component",
        }
        if value not in allowed_system_types:
            raise ValueError(f"system_type must be one of {sorted(allowed_system_types)}")
        f.system_type = value
    elif field_name == "block_type":
        if value not in {"PureOrMixtureData", "ReactionData"}:
            raise ValueError("block_type must be PureOrMixtureData or ReactionData")
        f.block_type = value
    else:
        return {"status": "TOOL_ARGUMENT_REFINEMENT_REQUIRED", "field": "field_name", "error": f"Unknown field: {field_name}"}

    return {"status": "ok", "field": field_name, "value": value}


# ═══════════════════════════════════════════════════════════════
#  Tool 6: get_current_fields — inspect what's been set so far
# ═══════════════════════════════════════════════════════════════

def get_current_fields() -> dict:
    """Return the current state of all search form fields.

    Returns a native object of all field values currently set by the agent.
    Use this to review what has been filled and what still needs attention.
    """
    return get_fields().snapshot()


# ═══════════════════════════════════════════════════════════════
#  Tool 7: parse_smart_search — run the deterministic parser
# ═══════════════════════════════════════════════════════════════

def parse_smart_search(query: str) -> dict:
    """Parse a free-text search query using the deterministic parser.

    Parameters
    ----------
    query : str
        Free-text search input (supports + for AND, ; for OR).

    Returns a native object whose ``blocks`` array contains parsed search
    blocks with classified tokens:
    compounds, properties, measurements, variables, constraints, phases,
    formulas, dois, authors, title_words, year ranges, system_type, etc.

    Use this as a first pass before calling resolve_entity for each token.
    """
    from ..hardcoded_search_helpers.search_parser import parse_search_input
    blocks = parse_search_input(query)
    rows = [b.to_dict() for b in blocks]
    return {"query": query, "n_blocks": len(rows), "blocks": rows}


# ═══════════════════════════════════════════════════════════════
#  Tool 8: review_top_results — run review workers on resolved query
# ═══════════════════════════════════════════════════════════════

def review_top_results(title_keywords: str = "") -> dict:
    """Run review workers to check resolved fields against top search results.

    Parameters
    ----------
    title_keywords : str
        Keywords for literature review (finds related titles and extracts
        compound/property mentions). Leave empty to skip literature review.

    Returns a native report object with suggested field edits from:
    - LiteratureReviewWorker (extracts entities from top title hits)
    - CompoundReviewWorker (re-checks compound resolutions)
    - PropertyReviewWorker (re-checks property resolutions)
    """
    from ..hardcoded_search_helpers.review_workers import (
        review_literature,
        review_compounds,
        review_properties,
    )
    from ..hardcoded_search_helpers.bibliography_scorer import score_titles
    from ..hardcoded_search_helpers.id_workers import ResolvedID

    f = get_fields()
    results = []

    # Literature review
    if title_keywords:
        hits = score_titles(title_keywords, _DB_PATH, limit=15)
        comp_names = [c["name"] for c in f.compounds]
        prop_names = [p["name"] for p in f.properties]
        rr = review_literature(hits, comp_names, prop_names)
        results.append(rr.to_dict())

    # Compound review — build ResolvedID objects from current fields
    comp_rids = [
        ResolvedID(
            entity_type="compound", global_id=c["global_id"],
            canonical_id="", display_name=c["name"],
            score=c["score"], match_type="set_by_agent"
        ) for c in f.compounds
    ]
    results.append(review_compounds(comp_rids, f.unresolved).to_dict())

    # Property review
    prop_rids = [
        ResolvedID(
            entity_type="property", global_id=p["global_id"],
            canonical_id="", display_name=p["name"],
            score=p["score"], match_type="set_by_agent"
        ) for p in f.properties
    ]
    results.append(review_properties(prop_rids, f.unresolved).to_dict())

    return {"n_reviews": len(results), "reviews": results}


# ═══════════════════════════════════════════════════════════════
#  Tool 9: fill_bibliography_block — resolve & fill bibliography fields
# ═══════════════════════════════════════════════════════════════

def fill_bibliography_block(
    doi: str = "",
    authors: str = "",
    title_keywords: str = "",
    journal: str = "",
    year_min: int | None = None,
    year_max: int | None = None,
) -> dict:
    """Resolve and fill bibliography fields in one step.

    Validates each provided field against the database, then sets all
    valid values at once. Returns a report of what was set and any
    validation warnings.

    Parameters
    ----------
    doi : str
        DOI to set (validated for existence in DB).
    authors : str
        First author surname to set.
    title_keywords : str
        Title keywords for literature matching.
    journal : str
        Journal name to set.
    year_min, year_max : int, optional
        Year range boundaries.
    """
    f = get_fields()
    report = {"set": [], "warnings": []}

    # Validate & set DOI
    if doi:
        from ..hardcoded_search_helpers.validation_tools import validate_bibliography as _vb
        bib = _vb(doi=doi, db_path=_DB_PATH)
        if bib.doi_status == "pass":
            f.doi = doi
            report["set"].append(f"doi={doi}")
            f.resolution_log.append(f"Bibliography: set doi={doi}")
        else:
            report["warnings"].append(f"DOI '{doi}' not found in database")
            f.unresolved.append(f"doi:{doi}")

    # Validate & set author
    if authors:
        f.authors = authors
        report["set"].append(f"authors={authors}")
        f.resolution_log.append(f"Bibliography: set authors={authors}")

    # Title keywords
    if title_keywords:
        f.title_keywords = title_keywords
        report["set"].append(f"title_keywords={title_keywords}")
        f.resolution_log.append(f"Bibliography: set title_keywords={title_keywords}")

    if journal:
        f.journal = journal
        report["set"].append(f"journal={journal}")
    if year_min is not None:
        if isinstance(year_min, bool) or not isinstance(year_min, int):
            raise TypeError("year_min must be an integer")
        f.year_min = year_min
        report["set"].append(f"year_min={year_min}")
    if year_max is not None:
        if isinstance(year_max, bool) or not isinstance(year_max, int):
            raise TypeError("year_max must be an integer")
        f.year_max = year_max
        report["set"].append(f"year_max={year_max}")
    if year_min is not None and year_max is not None and year_min > year_max:
        raise ValueError("year_min cannot exceed year_max")

    return report


# ═══════════════════════════════════════════════════════════════
#  Tool 10: fill_chemistry_block — resolve & fill chemistry fields
# ═══════════════════════════════════════════════════════════════

def fill_chemistry_block(
    compound_names: list[str] | None = None,
    formula: str = "",
    system_type: str = "",
    n_components: int | None = None,
) -> dict:
    """Resolve compound names and fill chemistry fields in one step.

    Each compound name is resolved to its canonical ID via the registry.
    Unresolved compounds are flagged for manual review or escalation.

    Parameters
    ----------
    compound_names : list[str], optional
        Exact JSON array of compound names (e.g. ``["ethanol", "water"]``).
    formula : str
        Molecular formula to set.
    system_type : str
        Canonical system type: unary, binary, ternary, quaternary,
        or ``5-component`` through ``9-component``.
    n_components : int, optional
        Number of components.
    """
    f = get_fields()
    report = {"set": [], "warnings": [], "resolved_compounds": []}

    if compound_names is not None:
        if not isinstance(compound_names, list) or not compound_names or any(
            not isinstance(name, str) or not name for name in compound_names
        ):
            raise TypeError("compound_names must be a non-empty array of non-empty strings")
        from ..hardcoded_search_helpers.id_workers import _resolve
        for name in compound_names:
            hits = _resolve("compound", name, limit=3, min_score=50)
            if hits:
                best = hits[0]
                entry = {"name": best.display_name, "global_id": best.global_id, "score": best.score}
                f.compounds.append(entry)
                f.resolution_log.append(
                    f"Chemistry: resolved compound '{name}' → "
                    f"{best.display_name} (global_id={best.global_id}, score={best.score})"
                )
                report["resolved_compounds"].append(entry)
                report["set"].append(f"compound={best.display_name} (global_id={best.global_id})")
            else:
                report["warnings"].append(f"Could not resolve compound '{name}'")
                f.unresolved.append(f"compound:{name}")

    if formula:
        f.formula = formula
        report["set"].append(f"formula={formula}")
    if system_type:
        allowed_system_types = {
            "unary", "binary", "ternary", "quaternary",
            "5-component", "6-component", "7-component", "8-component", "9-component",
        }
        if system_type not in allowed_system_types:
            raise ValueError(
                f"system_type must be one of {sorted(allowed_system_types)}"
            )
        f.system_type = system_type
        report["set"].append(f"system_type={system_type}")
    if n_components is not None:
        if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
            raise TypeError("n_components must be a positive integer")
        f.n_components = n_components
        report["set"].append(f"n_components={n_components}")

    return report


# ═══════════════════════════════════════════════════════════════
#  Tool 11: fill_properties_block — resolve & fill property fields
# ═══════════════════════════════════════════════════════════════

def fill_properties_block(
    property_names: list[str] | None = None,
    phase: str = "",
    block_type: str = "",
    min_datapoints: int | None = None,
) -> dict:
    """Resolve property names and fill property/measurement fields.

    Each property name is resolved to its canonical ID via the registry.

    Parameters
    ----------
    property_names : list[str], optional
        Exact JSON array of property names (e.g. ``["viscosity", "density"]``).
    phase : str
        Phase filter (e.g. "Liquid").
    block_type : str
        Block type: PureOrMixtureData or ReactionData.
    min_datapoints : int, optional
        Minimum number of data points.
    """
    f = get_fields()
    report = {"set": [], "warnings": [], "resolved_properties": []}

    if property_names is not None:
        if not isinstance(property_names, list) or not property_names or any(
            not isinstance(name, str) or not name for name in property_names
        ):
            raise TypeError("property_names must be a non-empty array of non-empty strings")
        from ..hardcoded_search_helpers.id_workers import _resolve
        for name in property_names:
            hits = _resolve("property", name, limit=3, min_score=50)
            if hits:
                best = hits[0]
                entry = {"name": best.display_name, "global_id": best.global_id, "score": best.score}
                f.properties.append(entry)
                f.resolution_log.append(
                    f"Properties: resolved '{name}' → "
                    f"{best.display_name} (global_id={best.global_id}, score={best.score})"
                )
                report["resolved_properties"].append(entry)
                report["set"].append(f"property={best.display_name} (global_id={best.global_id})")
            else:
                report["warnings"].append(f"Could not resolve property '{name}'")
                f.unresolved.append(f"property:{name}")

    if phase:
        from ..hardcoded_search_helpers.id_workers import _resolve
        phase_hits = _resolve("phase", phase, limit=3, min_score=50)
        if not phase_hits:
            raise ValueError(f"Phase {phase!r} did not resolve to a strict GLOBphase_N ID")
        best_phase = phase_hits[0]
        phase_entry = {
            "name": best_phase.display_name,
            "global_id": best_phase.global_id,
            "score": best_phase.score,
        }
        f.phases.append(phase_entry)
        report["set"].append(
            f"phase={best_phase.display_name} (global_id={best_phase.global_id})"
        )
    if block_type:
        if block_type not in {"PureOrMixtureData", "ReactionData"}:
            raise ValueError("block_type must be PureOrMixtureData or ReactionData")
        f.block_type = block_type
        report["set"].append(f"block_type={block_type}")
    if min_datapoints is not None:
        if isinstance(min_datapoints, bool) or not isinstance(min_datapoints, int) or min_datapoints < 1:
            raise TypeError("min_datapoints must be a positive integer")
        f.min_datapoints = min_datapoints
        report["set"].append(f"min_datapoints={min_datapoints}")

    return report


# ═══════════════════════════════════════════════════════════════
#  Tool 12: fill_variables_block — resolve & fill variable fields
# ═══════════════════════════════════════════════════════════════

def fill_variables_block(
    variable_names: list[str],
) -> dict:
    """Resolve variable names and fill variable fields in one step.

    Each variable name is resolved to its canonical ID via the registry.

    Parameters
    ----------
    variable_names : list[str]
        Exact JSON array of variable names
        (e.g. ``["Temperature, K", "Pressure, kPa", "Mole fraction"]``).
    """
    f = get_fields()
    report = {"set": [], "warnings": [], "resolved_variables": []}

    if not isinstance(variable_names, list) or not variable_names or any(
        not isinstance(name, str) or not name for name in variable_names
    ):
        raise TypeError("variable_names must be a non-empty array of non-empty strings")
    from ..hardcoded_search_helpers.id_workers import _resolve
    for name in variable_names:
        hits = _resolve("variable", name, limit=3, min_score=50)
        if hits:
            best = hits[0]
            entry = {"name": best.display_name, "global_id": best.global_id, "score": best.score}
            f.variables.append(entry)
            f.resolution_log.append(
                f"Variables: resolved '{name}' → "
                f"{best.display_name} (global_id={best.global_id}, score={best.score})"
            )
            report["resolved_variables"].append(entry)
            report["set"].append(f"variable={best.display_name} (global_id={best.global_id})")
        else:
            report["warnings"].append(f"Could not resolve variable '{name}'")
            f.unresolved.append(f"variable:{name}")

    return report


# ═══════════════════════════════════════════════════════════════
#  Tool 13: fill_measurements_block — resolve & fill measurement fields
# ═══════════════════════════════════════════════════════════════

def fill_measurements_block(
    measurement_names: list[str],
) -> dict:
    """Resolve measurement method names and fill measurement fields.

    Each measurement name is resolved to its canonical ID via the
    registry.

    Parameters
    ----------
    measurement_names : list[str]
        Exact JSON array of measurement method names
        (e.g. ``["Calorimetry", "Density measurement"]``).
    """
    f = get_fields()
    report = {"set": [], "warnings": [], "resolved_measurements": []}

    if not isinstance(measurement_names, list) or not measurement_names or any(
        not isinstance(name, str) or not name for name in measurement_names
    ):
        raise TypeError("measurement_names must be a non-empty array of non-empty strings")
    from ..hardcoded_search_helpers.id_workers import _resolve
    for name in measurement_names:
        hits = _resolve("measurement", name, limit=3, min_score=50)
        if hits:
            best = hits[0]
            entry = {"name": best.display_name, "global_id": best.global_id, "score": best.score}
            f.measurements.append(entry)
            f.resolution_log.append(
                f"Measurements: resolved '{name}' → "
                f"{best.display_name} (global_id={best.global_id}, score={best.score})"
            )
            report["resolved_measurements"].append(entry)
            report["set"].append(f"measurement={best.display_name} (global_id={best.global_id})")
        else:
            report["warnings"].append(f"Could not resolve measurement '{name}'")
            f.unresolved.append(f"measurement:{name}")

    return report


# ═══════════════════════════════════════════════════════════════
#  Tool 14: fill_constraints_block — resolve & fill constraint fields
# ═══════════════════════════════════════════════════════════════

def fill_constraints_block(
    constraint_names: list[str],
) -> dict:
    """Resolve constraint names and fill constraint fields.

    Each constraint name is resolved to its canonical ID via the
    registry.

    Parameters
    ----------
    constraint_names : list[str]
        Exact JSON array of constraint names
        (e.g. ``["Temperature", "Pressure"]``).
    """
    f = get_fields()
    report = {"set": [], "warnings": [], "resolved_constraints": []}

    if not isinstance(constraint_names, list) or not constraint_names or any(
        not isinstance(name, str) or not name for name in constraint_names
    ):
        raise TypeError("constraint_names must be a non-empty array of non-empty strings")
    from ..hardcoded_search_helpers.id_workers import _resolve
    for name in constraint_names:
        hits = _resolve("constraint", name, limit=3, min_score=50)
        if hits:
            best = hits[0]
            entry = {"name": best.display_name, "global_id": best.global_id, "score": best.score}
            f.constraints.append(entry)
            f.resolution_log.append(
                f"Constraints: resolved '{name}' → "
                f"{best.display_name} (global_id={best.global_id}, score={best.score})"
            )
            report["resolved_constraints"].append(entry)
            report["set"].append(f"constraint={best.display_name} (global_id={best.global_id})")
        else:
            report["warnings"].append(f"Could not resolve constraint '{name}'")
            f.unresolved.append(f"constraint:{name}")

    return report


# ═══════════════════════════════════════════════════════════════
#  Tool 15: escalate_to_query_agent — call ThermoML query agent
# ═══════════════════════════════════════════════════════════════

def escalate_to_query_agent(question: str, context: str = "") -> dict:
    """Escalate an ambiguous query to the ThermoML query agent.

    Parameters
    ----------
    question : str
        The question or instruction for the query agent.
    context : str
        Additional context about what has been resolved so far
        and what remains unresolved.

    Returns a native object containing the query agent's answer.
    Only use this when resolution has failed and the agent toggle
    is enabled. The query agent uses LLM reasoning + full DB search
    tools to identify canonical IDs.
    """
    from NIST_ThermoML_agents.NIST_ThermoML_query_agent.ThermoML_query_api import (
        ThermoML_query_run,
    )
    if not isinstance(question, str) or not question:
        raise TypeError("question must be a non-empty string")
    if not isinstance(context, str):
        raise TypeError("context must be a string")
    full_q = question
    if context:
        full_q += f"\n\nContext from alignment agent:\n{context}"
    # Query orchestration requires an explicit, isolated session. Keep search
    # escalations with the browser's shared Query runs, with an atomic unique
    # directory for concurrent requests. Tests can patch this one output root.
    _QUERY_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    session_dir = Path(tempfile.mkdtemp(
        prefix=f"run_{datetime.now():%Y%m%d_%H%M%S}_alignment_",
        dir=_QUERY_OUTPUT_ROOT,
    ))
    # Query owns its tracking context; isolate it even when a nested run fails.
    result = copy_context().run(
        ThermoML_query_run, full_q, session_dir=session_dir, run_verdict=False,
    )
    if not hasattr(result, "answer") or not isinstance(result.answer, str):
        raise TypeError("ThermoML_query_run must return an object with a string answer")
    if getattr(result, "timed_out", False):
        return {"status": "timeout", "answer": result.answer, "timed_out": True}
    return {"status": "ok", "answer": result.answer}


# ═══════════════════════════════════════════════════════════════
#  Tool 16: finalize_fields — mark fields as complete
# ═══════════════════════════════════════════════════════════════

def finalize_fields() -> dict:
    """Mark field filling as complete and return the final field snapshot.

    Call this when you are satisfied with all resolved fields.
    Returns the final state of all search form fields.
    """
    snapshot = get_fields().snapshot()
    snapshot["_finalized"] = True
    return snapshot
