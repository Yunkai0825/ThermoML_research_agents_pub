"""
13_block_rdp_inspection.py — Verbatim sub-block data-table inspection.
======================================================================
Mandatory grounding tool: before quoting ANY data point in an answer, an
agent inspects the exact sub-table it needs.  The tool returns ONLY the
RDP-or-complete data table plus deterministic per-column stats — never
cards or prose — so every returned number is a verbatim database value.

Rows are read from the authoritative raw ThermoML database via
``extract_block_csv`` (11_block_data_extractor); filtering, nearest-row
lookup, and RDP simplification are fully deterministic.

Public API
----------
    inspect_block_table(block_number, literature=None, *, where=None,
                        nearest=None, BLKsubsys_id=None,
                        property_filter=None) -> dict
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _id_alignment_search import resolve_reference_ids
from normalization_helpers.strict_id_inputs import (
    IdentifierRefinementError,
    refinement_errors_as_results,
    require_typed_block_input,
)
from ThermoML_raw_json_to_card_db_parsers.id_schema import (
    require_block_local_id,
)
from ThermoML_card_json_to_md_compactors._data_points_compaction_topology.rdp_topology import (
    classify_curve,
    compact_block_data,
)

_extract_block_csv = getattr(
    importlib.import_module("11_block_data_extractor"), "extract_block_csv",
)

COMPLETE_MAX_ROWS = 12
"""Filtered row count at or below which the table is returned complete."""

_NEAREST_KEYS = {"column", "value", "k"}
_NUM_RE = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
# Idents are sanitized aliases (\w only) — raw names with <>/, are not lexable.
_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>" + _NUM_RE + r")"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op><=|>=|!=|==|=|<|>)"
    r"|(?P<punct>[(),]))"
)


def _alias(column: str) -> str:
    """Sanitized filter alias: non-alphanumerics collapse to underscores."""
    return re.sub(r"[^0-9A-Za-z]+", "_", column).strip("_").lower()


def _fmt(value: float) -> str:
    return f"{value:.6g}"


# ═════════════════════════════════════════════════════════════════════════════
# Minimal WHERE parser — conditions joined by AND
# ═════════════════════════════════════════════════════════════════════════════

class _WhereError(ValueError):
    pass


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens, pos = [], 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(text, pos)
        if not m or m.start() != pos:
            raise _WhereError(f"unreadable character at position {pos}: {text[pos:pos+12]!r}")
        pos = m.end()
        for kind in ("num", "ident", "op", "punct"):
            val = m.group(kind)
            if val is not None:
                if kind == "ident" and val.upper() in ("AND", "BETWEEN", "IN"):
                    tokens.append(("kw", val.upper()))
                else:
                    tokens.append((kind, val))
                break
    return tokens


def _parse_where(text: str, resolve_col) -> list[dict]:
    """Parse ``col OP num | col BETWEEN a AND b | col IN (v, ...)`` chains.

    Returns a list of condition dicts: {"column", "alias", "op", ...}.
    """
    tokens = _tokenize(text)
    conds, i = [], 0

    def _expect_num(idx):
        if idx >= len(tokens) or tokens[idx][0] != "num":
            raise _WhereError("expected a plain number (no units/quotes) in `where`")
        return float(tokens[idx][1]), idx + 1

    while i < len(tokens):
        if tokens[i][0] != "ident":
            raise _WhereError(f"expected a column alias, got {tokens[i][1]!r}")
        column = resolve_col(tokens[i][1])
        i += 1
        if i >= len(tokens):
            raise _WhereError(f"dangling column {tokens[i-1][1]!r} — missing operator")
        kind, val = tokens[i]
        if kind == "op":
            op = "=" if val in ("=", "==") else val
            num, i = _expect_num(i + 1)
            conds.append({"column": column, "op": op, "value": num})
        elif (kind, val) == ("kw", "BETWEEN"):
            lo, i = _expect_num(i + 1)
            if i >= len(tokens) or tokens[i] != ("kw", "AND"):
                raise _WhereError("BETWEEN requires `BETWEEN lo AND hi`")
            hi, i = _expect_num(i + 1)
            if hi < lo:
                lo, hi = hi, lo
            conds.append({"column": column, "op": "between", "lo": lo, "hi": hi})
        elif (kind, val) == ("kw", "IN"):
            i += 1
            if i >= len(tokens) or tokens[i] != ("punct", "("):
                raise _WhereError("IN requires a parenthesized number list")
            i += 1
            values = []
            while True:
                num, i = _expect_num(i)
                values.append(num)
                if i < len(tokens) and tokens[i] == ("punct", ","):
                    i += 1
                    continue
                break
            if i >= len(tokens) or tokens[i] != ("punct", ")"):
                raise _WhereError("IN list is missing the closing parenthesis")
            i += 1
            conds.append({"column": column, "op": "in", "values": values})
        else:
            raise _WhereError(
                f"expected an operator, BETWEEN, or IN after column, got {val!r}"
            )
        if i < len(tokens):
            if tokens[i] != ("kw", "AND"):
                raise _WhereError(
                    f"conditions join with AND only (no OR), got {tokens[i][1]!r}"
                )
            i += 1
            if i >= len(tokens):
                raise _WhereError("dangling AND at end of `where`")
    if not conds:
        raise _WhereError("`where` contains no conditions")
    return conds


def _cond_matches(cond: dict, value: float | None) -> bool:
    if value is None:
        return False
    op = cond["op"]
    if op == "between":
        return cond["lo"] <= value <= cond["hi"]
    if op == "in":
        return any(value == v for v in cond["values"])
    ref = cond["value"]
    return {
        "=": value == ref,
        "!=": value != ref,
        "<": value < ref,
        "<=": value <= ref,
        ">": value > ref,
        ">=": value >= ref,
    }[op]


def _cond_text(cond: dict, alias_of) -> str:
    a = alias_of(cond["column"])
    if cond["op"] == "between":
        return f"{a} BETWEEN {_fmt(cond['lo'])} AND {_fmt(cond['hi'])}"
    if cond["op"] == "in":
        return f"{a} IN ({', '.join(_fmt(v) for v in cond['values'])})"
    return f"{a} {cond['op']} {_fmt(cond['value'])}"


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

@refinement_errors_as_results
def inspect_block_table(
    block_number: str,
    literature=None,
    *,
    where: str | None = None,
    nearest: dict | None = None,
    BLKsubsys_id: str | None = None,
    property_filter: str | None = None,
) -> dict:
    """Return the verbatim data sub-table of ONE block for answer grounding.

    MANDATORY before quoting data points: every numeric data value cited in
    an answer must appear in a table returned by this tool (or in another
    tool's returned output).  Quote returned rows verbatim at source
    precision; rows not shown must not be quoted — narrow ``where`` instead.
    ``_U`` columns are expanded combined uncertainties from the source;
    report value ± U when present; say that no uncertainty is recorded
    only when the column is empty.

    Parameters
    ----------
    block_number : str
        ``GLOBlit_<N>::PROPblock_<M>`` (or ``::RXNblock_<M>``), or a bare
        typed block ID together with ``literature=``.
    literature : str, optional
        DOI, lit_id, or lit_num_id scope for a bare block ID.
    where : str, optional
        Row filter over column aliases; conditions join with AND only.
        Grammar: ``col <op> number`` (op: = != < <= > >=),
        ``col BETWEEN lo AND hi``, ``col IN (v1, v2, ...)``.
        Aliases are lowercase with non-alphanumerics as underscores, e.g.
        ``temperature_k``, ``mole_fraction_ethanol``, ``pressure_kpa``.
        Plain numbers only — units are fixed by the column (K, kPa, ...).
        Call without ``where`` once to discover aliases and ranges.
    nearest : dict, optional
        ``{"column": alias, "value": number, "k": 1..8}`` — return the k
        rows nearest to *value* on that column (applied after ``where``),
        plus a bracketing statement: the nearest row below/above and
        whether any row exists in the open interval.  Use this instead of
        interpolating or guessing a row near a target condition.
    BLKsubsys_id : str, optional
        Restrict to one composition subsystem of the block.
    property_filter : str, optional
        Keep only property columns whose name contains this substring.

    Returns
    -------
    dict
        ``rows_shown`` (verbatim cells), ``table_mode`` (complete | rdp |
        nearest | head), ``stats`` (per-column min/max/n over matched rows
        and the full block), ``bracket`` (nearest-mode gap statement),
        ``topology``, ``constraints``, ``uncertainty`` (the block-level
        assessment behind the ``_U`` columns, or None), ``markdown``,
        ``inspection_id``.
        With ``table_mode="rdp"`` only a shape-preserving subset is shown:
        quote only shown rows or the stats line, or narrow ``where``.
    """
    # ── Addressing: mirror search_blocks' literature-scoped block target ──
    if not isinstance(block_number, str) or not block_number.strip():
        raise IdentifierRefinementError(
            "block_number", block_number,
            "GLOBlit_<N>::PROPblock_<M> or PROPblock_<M> with literature=",
            "block inspection takes one typed block string",
        )
    candidate = block_number.strip()
    if "::" in candidate:
        lit_part, _, block_part = candidate.partition("::")
        lit_part, block_part = lit_part.strip(), block_part.strip()
        if not re.fullmatch(r"GLOBlit_[1-9]\d*", lit_part):
            raise IdentifierRefinementError(
                "block_number", block_number,
                "GLOBlit_<N>::PROPblock_<M> (or ::RXNblock_<M>)",
                "the scope before '::' must be one GLOBlit id",
            )
        if literature is not None:
            raise IdentifierRefinementError(
                "block_number", block_number,
                "either GLOBlit_<N>::block form OR literature= + bare block",
                "the literature scope was given twice",
            )
        block_id = require_typed_block_input(block_part, field="block_number")
        literature = lit_part
    else:
        if literature is None:
            raise IdentifierRefinementError(
                "block_number", block_number,
                "GLOBlit_<N>::PROPblock_<M>, or literature= plus this bare id",
                "block numbers repeat across papers and need a literature scope",
            )
        block_id = require_typed_block_input(candidate, field="block_number")

    subsystem_id = (
        require_block_local_id("subsys", BLKsubsys_id)
        if BLKsubsys_id is not None else None
    )

    ref_hits = resolve_reference_ids(literature)
    if not ref_hits:
        return _error_result(
            block_id, subsystem_id,
            f"Could not resolve literature {literature!r} to any known DOI.",
        )
    doi = ref_hits[0]["doi"]

    extracted = _extract_block_csv(
        doi, block_id,
        BLKsubsys_id=subsystem_id, property_filter=property_filter,
    )
    if not isinstance(extracted, dict):
        raise TypeError("extract_block_csv returned an invalid result object")
    if extracted.get("error"):
        return _error_result(block_id, subsystem_id, extracted["error"], doi=doi)

    columns = extracted["columns"]
    metadata = extracted["metadata"]
    all_rows = list(csv.DictReader(io.StringIO(extracted["csv_text"])))
    if not all_rows:
        return _error_result(
            block_id, subsystem_id, "Block has no data rows.", doi=doi,
        )

    data_columns = [c for c in columns if c != "BLKpoint_id"]

    # Numeric float view per column (None for blank/non-numeric cells).
    floats: dict[str, list[float | None]] = {}
    for col in data_columns:
        vals = []
        for row in all_rows:
            cell = row.get(col)
            try:
                vals.append(float(cell) if cell not in (None, "") else None)
            except ValueError:
                vals.append(None)
        floats[col] = vals

    # Alias resolution table (sanitized, case-insensitive).
    alias_to_cols: dict[str, set[str]] = {}
    for col in data_columns:
        alias_to_cols.setdefault(_alias(col), set()).add(col)
    column_aliases = {
        a: next(iter(cols)) for a, cols in alias_to_cols.items()
        if len(cols) == 1
    }

    def _resolve_col(token: str) -> str:
        key = _alias(token)
        hits = alias_to_cols.get(key, set())
        if len(hits) == 1:
            return next(iter(hits))
        if not hits:
            raise _WhereError(f"unknown column alias {token!r}")
        raise _WhereError(
            f"alias {token!r} is ambiguous between {sorted(hits)}"
        )

    def _alias_of(col: str) -> str:
        return _alias(col)

    def _columns_help() -> list[dict]:
        rows = []
        for col in data_columns:
            finite = [v for v in floats[col] if v is not None]
            rows.append({
                "alias": _alias(col),
                "column": col,
                "n_finite": len(finite),
                "min": _fmt(min(finite)) if finite else None,
                "max": _fmt(max(finite)) if finite else None,
            })
        return rows

    # ── WHERE filter ──────────────────────────────────────────────────────
    conds: list[dict] = []
    if where is not None:
        if not isinstance(where, str) or not where.strip():
            return _error_result(
                block_id, subsystem_id,
                "`where` must be a non-empty filter string when given",
                doi=doi, columns_help=_columns_help(),
            )
        try:
            conds = _parse_where(where, _resolve_col)
        except _WhereError as exc:
            return _error_result(
                block_id, subsystem_id, f"where filter error: {exc}",
                doi=doi, columns_help=_columns_help(),
            )
    where_echo = " AND ".join(_cond_text(c, _alias_of) for c in conds) or None

    matched_idx = [
        i for i in range(len(all_rows))
        if all(_cond_matches(c, floats[c["column"]][i]) for c in conds)
    ]
    if not matched_idx:
        return _error_result(
            block_id, subsystem_id,
            f"No rows match where: {where_echo}. Widen the window.",
            doi=doi, columns_help=_columns_help(),
        )

    # ── nearest post-filter + bracketing gap ──────────────────────────────
    bracket = None
    nearest_echo = None
    if nearest is not None:
        if (not isinstance(nearest, dict)
                or not {"column", "value"} <= set(nearest)
                or not set(nearest) <= _NEAREST_KEYS):
            return _error_result(
                block_id, subsystem_id,
                'nearest must be {"column": alias, "value": number, "k": 1..8}',
                doi=doi, columns_help=_columns_help(),
            )
        try:
            n_col = _resolve_col(str(nearest["column"]))
        except _WhereError as exc:
            return _error_result(
                block_id, subsystem_id, f"nearest column error: {exc}",
                doi=doi, columns_help=_columns_help(),
            )
        n_val = nearest["value"]
        if isinstance(n_val, bool) or not isinstance(n_val, (int, float)):
            return _error_result(
                block_id, subsystem_id, "nearest value must be a number",
                doi=doi, columns_help=_columns_help(),
            )
        k = nearest.get("k", 1)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 8:
            return _error_result(
                block_id, subsystem_id, "nearest k must be an integer in 1..8",
                doi=doi, columns_help=_columns_help(),
            )
        candidates = [
            i for i in matched_idx if floats[n_col][i] is not None
        ]
        if not candidates:
            return _error_result(
                block_id, subsystem_id,
                f"no matched row has a numeric {_alias(n_col)} value",
                doi=doi, columns_help=_columns_help(),
            )
        ranked = sorted(
            candidates, key=lambda i: (abs(floats[n_col][i] - n_val), i)
        )
        shown_idx = sorted(ranked[:k], key=lambda i: floats[n_col][i])
        table_mode = "nearest"
        nearest_echo = {"column": _alias(n_col), "value": n_val, "k": k}
        bracket = _bracket_info(
            n_col, n_val, candidates, floats, all_rows, _alias(n_col)
        )
    elif len(matched_idx) <= COMPLETE_MAX_ROWS:
        shown_idx = matched_idx
        table_mode = "complete"
    else:
        shown_idx, table_mode = _rdp_select(
            matched_idx, all_rows, floats, data_columns, metadata,
        )

    topology = _topology_lines(matched_idx, floats, data_columns, metadata)

    # ── Assemble payload ─────────────────────────────────────────────────
    rows_shown = [
        {c: (all_rows[i].get(c) or "") for c in columns} for i in shown_idx
    ]
    stats = {
        "matched": _column_stats(matched_idx, floats, all_rows, data_columns),
        "full": _column_stats(
            list(range(len(all_rows))), floats, all_rows, data_columns,
        ),
    }
    constraints = [
        {"column": c["column_name"], "value": c["value"]}
        for c in metadata.get("constraints", [])
        if c.get("value") is not None
    ]

    table_csv = _rows_csv(columns, rows_shown)
    inspection_id = "INSP_" + hashlib.sha1(
        "|".join([
            doi, block_id, subsystem_id or "", table_mode, table_csv,
        ]).encode("utf-8")
    ).hexdigest()[:12]

    result = {
        "tool": "inspect_block_table",
        "doi": doi,
        "lit_num_id": metadata.get("lit_num_id"),
        "block_number": block_id,
        "BLKsubsys_id": subsystem_id,
        "system_type": metadata.get("system_type"),
        "compounds": [c["name"] for c in metadata.get("compounds", [])],
        "columns": columns,
        "column_aliases": column_aliases,
        "where": where_echo,
        "nearest": nearest_echo,
        "n_rows_total": len(all_rows),
        "n_rows_matched": len(matched_idx),
        "n_rows_shown": len(rows_shown),
        "table_mode": table_mode,
        "rows_shown": rows_shown,
        "bracket": bracket,
        "stats": stats,
        "topology": topology,
        "constraints": constraints,
        "uncertainty": metadata.get("uncertainty"),
        "inspection_id": inspection_id,
        "error": None,
    }
    result["markdown"] = _render_markdown(result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Internals
# ═════════════════════════════════════════════════════════════════════════════

def _bracket_info(col, target, candidate_idx, floats, all_rows, alias):
    """Nearest-mode gap evidence: rows bracketing *target* on *col*."""
    below = [(floats[col][i], i) for i in candidate_idx if floats[col][i] <= target]
    above = [(floats[col][i], i) for i in candidate_idx if floats[col][i] >= target]
    exact = [i for i in candidate_idx if floats[col][i] == target]

    def _cell(pair):
        value, i = pair
        return {
            "BLKpoint_id": all_rows[i].get("BLKpoint_id", ""),
            "value": all_rows[i].get(col) or _fmt(value),
        }

    info = {
        "column": alias,
        "target": target,
        "exact_match": bool(exact),
        "below": _cell(max(below)) if below else None,
        "above": _cell(min(above)) if above else None,
    }
    if exact:
        info["statement"] = (
            f"exact row(s) at {alias} = {_fmt(target)} exist in the matched set"
        )
    elif below and above:
        lo, hi = max(below), min(above)
        n_inside = sum(
            1 for i in candidate_idx if lo[0] < floats[col][i] < hi[0]
        )
        info["statement"] = (
            f"no matched row at {alias} = {_fmt(target)}; nearest below "
            f"{info['below']['BLKpoint_id']} ({info['below']['value']}), "
            f"nearest above {info['above']['BLKpoint_id']} "
            f"({info['above']['value']}); {n_inside} matched rows strictly "
            f"inside that interval"
        )
    else:
        side = "below" if not below else "above"
        finite = [floats[col][i] for i in candidate_idx]
        info["statement"] = (
            f"{alias} = {_fmt(target)} lies outside the matched range "
            f"{_fmt(min(finite))}–{_fmt(max(finite))} (no row {side} it)"
        )
    return info


def _rdp_select(matched_idx, all_rows, floats, data_columns, metadata):
    """Shape-preserving row subset via the house RDP policy."""
    var_by_col = {
        v["column_name"]: v["BLKvar_id"] for v in metadata.get("variables", [])
    }
    prop_by_col = {
        p["column_name"]: p["BLKprop_id"] for p in metadata.get("properties", [])
    }
    variables = [
        {"BLKvar_id": var_by_col[c], "name": c, "component_org_num": None}
        for c in data_columns if c in var_by_col
    ]
    properties = [
        {"BLKprop_id": prop_by_col[c]} for c in data_columns if c in prop_by_col
    ]
    if not variables or not properties:
        # No numeric axes to preserve — deterministic head sample.
        return matched_idx[:COMPLETE_MAX_ROWS], "head"

    data_points = []
    for i in matched_idx:
        data_points.append({
            "variable_values": {
                v["BLKvar_id"]: {"value": floats[v["name"]][i]}
                for v in variables
            },
            "property_values": {
                prop_by_col[c]: {"value": floats[c][i]}
                for c in data_columns if c in prop_by_col
            },
        })
    compacted = compact_block_data(variables, properties, data_points)
    kept = sorted(compacted["kept_indices"])
    if not kept:
        return matched_idx[:COMPLETE_MAX_ROWS], "head"
    return [matched_idx[j] for j in kept], "rdp"


def _topology_lines(matched_idx, floats, data_columns, metadata):
    """Per-property curve classification over the matched rows."""
    var_cols = [
        v["column_name"] for v in metadata.get("variables", [])
        if v["column_name"] in data_columns
    ]
    prop_cols = [
        p["column_name"] for p in metadata.get("properties", [])
        if p["column_name"] in data_columns
    ]
    if len(var_cols) != 1 or not prop_cols:
        return []
    lines = []
    x_col = var_cols[0]
    for p_col in prop_cols:
        pts = sorted(
            (floats[x_col][i], floats[p_col][i])
            for i in matched_idx
            if floats[x_col][i] is not None and floats[p_col][i] is not None
        )
        if len(pts) >= 3:
            lines.append({
                "property": _alias(p_col),
                "vs": _alias(x_col),
                "shape": classify_curve(pts),
            })
    return lines


def _column_stats(idx, floats, all_rows, data_columns):
    """Verbatim min/max cells + counts per column over *idx* rows."""
    stats = {}
    for col in data_columns:
        pairs = [
            (floats[col][i], i) for i in idx if floats[col][i] is not None
        ]
        entry = {"n": len(idx), "n_finite": len(pairs)}
        if pairs:
            lo, hi = min(pairs), max(pairs)
            entry["min"] = all_rows[lo[1]].get(col) or _fmt(lo[0])
            entry["max"] = all_rows[hi[1]].get(col) or _fmt(hi[0])
        stats[_alias(col)] = entry
    return stats


def _rows_csv(columns, rows_shown):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows_shown:
        writer.writerow(row)
    return buf.getvalue()


def _error_result(block_id, subsystem_id, message, *, doi=None, columns_help=None):
    lines = [f"**inspect_block_table error** — {message}"]
    if columns_help:
        lines.append("")
        lines.append("| alias | n_finite | min | max |")
        lines.append("|---|---|---|---|")
        for c in columns_help:
            lines.append(
                f"| {c['alias']} | {c['n_finite']} | {c['min'] or ''} "
                f"| {c['max'] or ''} |"
            )
        lines.append("")
        lines.append(
            "Use these aliases in `where` / `nearest` with plain numbers."
        )
    return {
        "tool": "inspect_block_table",
        "doi": doi,
        "block_number": block_id,
        "BLKsubsys_id": subsystem_id,
        "error": message,
        "columns_help": columns_help or [],
        "rows_shown": [],
        "n_rows_shown": 0,
        "markdown": "\n".join(lines),
    }


def _uncertainty_line(uncertainty) -> str | None:
    """One header line naming the source's uncertainty assessment, or None.

    Renders ``uncertainty: expanded combined (U), level 95 %, evaluated by
    Data file compiler`` from the block-level ``CombinedUncertainty``
    declaration; the per-point values sit in the ``_U`` table columns.
    """
    if not isinstance(uncertainty, dict):
        return None
    bits = ["expanded combined (U)"]
    level = uncertainty.get("confidence_level")
    if level is not None:
        bits.append(f"level {level} %")
    evaluator = uncertainty.get("evaluator")
    if evaluator:
        bits.append(f"evaluated by {evaluator}")
    if len(bits) == 1:
        return None
    return "uncertainty: " + ", ".join(bits)


def _render_markdown(result: dict) -> str:
    system = " + ".join(result["compounds"])
    head = (
        f"**Data inspection** — {result['doi']} :: {result['block_number']}"
        + (f" [{result['BLKsubsys_id']}]" if result["BLKsubsys_id"] else "")
        + (f" ({result['system_type']}: {system})" if system else "")
    )
    lines = [head]

    filter_bits = []
    if result["where"]:
        filter_bits.append(f"where {result['where']}")
    if result["nearest"]:
        n = result["nearest"]
        filter_bits.append(f"nearest {n['column']} ≈ {_fmt(n['value'])} (k={n['k']})")
    lines.append("filter: " + (" | ".join(filter_bits) if filter_bits else "none (whole block)"))

    if result["constraints"]:
        lines.append("block constraints: " + "; ".join(
            f"{_alias(c['column'])} = {c['value']}" for c in result["constraints"]
        ))

    uncertainty_line = _uncertainty_line(result.get("uncertainty"))
    if uncertainty_line:
        lines.append(uncertainty_line)

    columns = result["columns"]
    lines.append("")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * len(columns))
    for row in result["rows_shown"]:
        lines.append("| " + " | ".join(row.get(c, "") for c in columns) + " |")
    lines.append("")

    mode = result["table_mode"]
    n_shown, n_matched, n_total = (
        result["n_rows_shown"], result["n_rows_matched"], result["n_rows_total"],
    )
    if mode == "complete":
        lines.append(
            f"mode: complete — all {n_matched} matched rows shown "
            f"(block has {n_total})"
        )
    elif mode == "rdp":
        lines.append(
            f"mode: RDP — {n_shown} of {n_matched} matched rows shown "
            f"(shape-preserving; block has {n_total}). Quote ONLY shown rows "
            f"or the stats line; narrow `where` for other rows."
        )
    elif mode == "nearest":
        lines.append(
            f"mode: nearest — {n_shown} row(s) of {n_matched} matched "
            f"(block has {n_total})"
        )
    else:
        lines.append(
            f"mode: head — first {n_shown} of {n_matched} matched rows "
            f"(no numeric axes for RDP). Narrow `where` for other rows."
        )

    if result["bracket"]:
        lines.append(f"bracket: {result['bracket']['statement']}")

    for t in result["topology"]:
        lines.append(f"topology: {t['property']} vs {t['vs']}: {t['shape']}")

    def _stats_text(stats):
        bits = []
        for a, s in stats.items():
            if s["n_finite"]:
                bits.append(f"{a} {s['min']}–{s['max']} (n={s['n_finite']})")
            else:
                bits.append(f"{a} (no numeric values)")
        return "; ".join(bits)

    lines.append(f"stats (matched, n={n_matched}): " + _stats_text(result["stats"]["matched"]))
    if n_matched != n_total:
        lines.append(f"stats (full block, n={n_total}): " + _stats_text(result["stats"]["full"]))
    lines.append(f"inspection_id: {result['inspection_id']}")
    lines.append(
        "Data points may be quoted ONLY verbatim from the table above or "
        "the stats lines."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import json as _json

    demo = inspect_block_table(
        "PROPblock_1",
        literature=sys.argv[1] if len(sys.argv) > 1 else "10.1016/j.jct.2005.03.012",
    )
    print(demo.get("markdown") or _json.dumps(demo, indent=2)[:2000])
