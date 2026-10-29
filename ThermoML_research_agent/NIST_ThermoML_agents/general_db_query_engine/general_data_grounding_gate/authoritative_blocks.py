"""Authoritative full-table fetch for the grounding gate.

Resolves a cited ``GLOBlit_N :: block`` to its complete data table via the
same extractor that powers ``inspect_block_table`` (``extract_block_csv``).
Used ONLY to sharpen gate verdicts and feed the passive evidence log — the
inspection mandate is never waived by an authoritative match.

All imports are lazy and every failure degrades to ``None`` (no authoritative
evidence), so unit tests and DB-less environments are unaffected.
"""

from __future__ import annotations

import csv
import importlib
import io
import logging
from functools import lru_cache

log = logging.getLogger("grounding-gate-authoritative")


def _float_or_none(cell: str) -> float | None:
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=256)
def default_authoritative_fetch(
    lit_num_id: str | None, block_number: str,
) -> dict | None:
    """Full-table values of one registered block, or ``None`` when unavailable.

    Returns ``{"doi", "lit_num_id", "block_number", "n_rows", "columns",
    "values": frozenset[float], "column_ranges": {col: (min, max)},
    "column_values": {col: frozenset[float]}}`` (a numeric ``BLKpoint_id``
    lands in ``values`` only).
    """
    if not lit_num_id:
        return None
    try:
        from ThermoML_raw_json_to_card_db_parsers.id_schema import (
            doi_for_lit_num_id,
        )
        extract_block_csv = getattr(
            importlib.import_module("11_block_data_extractor"),
            "extract_block_csv",
        )
    except Exception:
        log.debug("authoritative fetch unavailable (imports)", exc_info=True)
        return None
    try:
        doi = doi_for_lit_num_id(lit_num_id)
        result = extract_block_csv(doi, block_number)
    except Exception:
        log.debug(
            "authoritative fetch failed for %s::%s",
            lit_num_id, block_number, exc_info=True,
        )
        return None
    if (
        not isinstance(result, dict)
        or result.get("error")
        or not result.get("csv_text")
    ):
        return None

    values: set[float] = set()
    by_column: dict[str, set[float]] = {}
    ranges: dict[str, tuple[float, float]] = {}
    reader = csv.reader(io.StringIO(result["csv_text"]))
    rows = list(reader)
    if len(rows) < 2:
        return None
    columns = rows[0]
    for row in rows[1:]:
        for column, cell in zip(columns, row):
            value = _float_or_none(cell)
            if value is None:
                continue
            values.add(value)
            if column != "BLKpoint_id":
                by_column.setdefault(column, set()).add(value)
                low, high = ranges.get(column, (value, value))
                ranges[column] = (min(low, value), max(high, value))
    return {
        "doi": doi,
        "lit_num_id": lit_num_id,
        "block_number": block_number,
        "n_rows": result.get("n_rows", len(rows) - 1),
        "columns": columns,
        "values": frozenset(values),
        "column_ranges": ranges,
        "column_values": {
            column: frozenset(cells) for column, cells in by_column.items()
        },
    }
