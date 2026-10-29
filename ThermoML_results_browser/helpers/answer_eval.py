"""Saved evaluations of one answer, for the Result panel's Eval switch.

An answer folder may hold ``eval/evaluation.json`` (the numerical validation:
each compared value in result.md with its outcome and stored source) and
``eval/claim_review.json`` (the claim review: each reviewed claim with its
category, reason and evidence). Both place their items on the folder's own
``result.md`` by 1-based line and 1-based, end-exclusive code-point columns.

Only what the display needs is returned, with the result.md lines it refers
to so the page can tell a line it shows rewritten. The evaluated copy of the
answer is never sent: ``result.md`` stays the one answer source.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from zipfile import BadZipFile

from ThermoML_results_browser.helpers.benchmark_zip import ZipPath, as_path, is_within

log = logging.getLogger(__name__)

NUMBERS_FILE = "evaluation.json"
CLAIMS_FILE = "claim_review.json"

_DOI_BLOCK = re.compile(r"(10\.\d{4,9}/[^\s:;,|\]\"'<>]+?):PROPblock_(\d+)")
_VALUE_FIELDS = ("token_id", "text", "column", "end_column", "group", "outcome", "misassigned",
                 "property", "claimed_unit", "final_reference", "reason", "band_check", "display",
                 "category", "classification", "source_kind", "numeric_outcome", "condition_outcome",
                 "precision_policy", "precision_role", "precision_previous_reason", "source_outcome", "unit_outcome", "basis_outcome", "raw_numeric_outcome", "artifact_inspected", "artifact_reference",
                 "artifact_search", "curve_verification", "calculated_reference_value", "population_interpretation", "magnitude_interpretation", "reported_value", "stored_value", "stored_unit", "converted_stored_value",
                 "initial_stored_value", "initial_stored_unit", "reason_code", "evidence")
_OUTSIDE_FIELDS = ("token_id", "text", "column", "end_column", "group")
_CLAIM_FIELDS = ("claim_id", "status", "category", "line", "column", "end_column", "text",
                 "reason", "evidence")


def _read(answer_dir: Path | ZipPath, name: str):
    """One eval file as JSON, or None when absent, outside the folder or unreadable."""
    path = answer_dir / "eval" / name
    try:
        if not is_within(path, answer_dir) or not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, BadZipFile) as exc:
        log.warning("unreadable answer evaluation %s: %s", path, exc)
        return None


def _ref_ids(value, found: set) -> set:
    """Every DOI:PROPblock_N named anywhere in *value*: objects with doi+block, and DOI:PROPblock_N text."""
    if isinstance(value, dict):
        doi, block = value.get("doi"), value.get("block")
        number = re.fullmatch(r"(?:PROPblock_)?(\d+)", str(block).strip()) if block is not None else None
        if doi and number:  # a block may also be "NA: ..." text
            found.add((str(doi), int(number.group(1))))
        for item in value.values():
            _ref_ids(item, found)
    elif isinstance(value, list):
        for item in value:
            _ref_ids(item, found)
    elif isinstance(value, str):
        for match in _DOI_BLOCK.finditer(value):
            found.add((match.group(1).rstrip("."), int(match.group(2))))
    return found


def _numbers(doc) -> dict | None:
    """Lines holding a compared value, and the DOI:block ids the validation relates to the answer."""
    if not isinstance(doc, dict) or not isinstance(doc.get("excerpts"), list):
        return None
    excerpts, cites = [], {}
    for excerpt in doc["excerpts"]:
        if not isinstance(excerpt, dict) or not isinstance(excerpt.get("values"), list):
            continue
        values = [v for v in excerpt["values"] if isinstance(v, dict)]
        if not any(v.get("group") != "outside" for v in values):
            continue  # only lines with a compared value are highlighted
        kept = []
        for value in values:
            if value.get("group") == "outside":
                kept.append({k: value[k] for k in _OUTSIDE_FIELDS if k in value})
                continue
            kept.append({k: value[k] for k in _VALUE_FIELDS if k in value})
            for key in _ref_ids(value.get("final_reference"), set()):
                cites[key] = cites.get(key, 0) + 1
        excerpts.append({"line": excerpt.get("line"), "values": kept})
    coverage = doc.get("coverage") if isinstance(doc.get("coverage"), dict) else {}
    omitted = _ref_ids(coverage.get("omitted_items"), set())
    related = sorted(set(cites) | _ref_ids(doc.get("factual_claims"), set()) | _ref_ids(coverage, set()))
    counts = doc.get("counts") if isinstance(doc.get("counts"), dict) else {}
    numeric_counts = counts.get("numeric") if isinstance(counts.get("numeric"), dict) else {}
    return {"excerpts": excerpts, "schema": doc.get("schema"),
            "numeric_counts": numeric_counts,
            "related": [{"doi": doi, "block": block, "values": cites.get((doi, block), 0),
                         "omitted": (doi, block) in omitted} for doi, block in related]}


def _claims(doc) -> list | None:
    if not isinstance(doc, dict) or not isinstance(doc.get("claims"), list):
        return None
    return [{k: claim[k] for k in _CLAIM_FIELDS if k in claim}
            for claim in doc["claims"] if isinstance(claim, dict)]


def _source_lines(answer_dir: Path | ZipPath, found: dict) -> dict:
    """The result.md lines the evaluation places items on, by line number. The
    page shows result.md rewritten in places (ledger tables, an old answer
    envelope), so it places an item only where its line is shown unchanged."""
    candidates = [excerpt.get("line") for excerpt in (found["numbers"] or {}).get("excerpts", [])]
    candidates.extend(claim.get("line") for claim in found["claims"] or [])
    wanted = {n for n in candidates if isinstance(n, int) and not isinstance(n, bool) and n > 0}
    if not wanted:
        return {}
    try:
        result_path = answer_dir / "result.md"
        if not is_within(result_path, answer_dir):
            return {}
        lines = result_path.read_text(encoding="utf-8", errors="replace").split("\n")
    except (OSError, ValueError, BadZipFile):
        return {}
    return {str(n): lines[n - 1] for n in sorted(wanted) if n <= len(lines)}


def load_answer_eval(answer_dir: Path | ZipPath) -> dict:
    """``{"numbers": ... or None, "claims": [...] or None, "lines": {line: text}}``
    for one answer folder."""
    answer_dir = as_path(answer_dir)
    found = {"numbers": _numbers(_read(answer_dir, NUMBERS_FILE)),
             "claims": _claims(_read(answer_dir, CLAIMS_FILE))}
    found["lines"] = _source_lines(answer_dir, found)
    return found
