"""Deterministic harvest of verbatim data inspections from a run's tool history.

Two sources feed the session inspection registry:

1. ``inspect_block_table`` results — the deterministic markdown rendered by
   ``13_block_rdp_inspection`` (this module parses its own house format).
2. ``data_inspections`` arrays embedded in child-agent envelopes that appear
   in tool results (downstream reuse without re-fetching).

Everything here is pure text processing — no DB access, no LLM.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

INSPECT_TOOL_NAME = "inspect_block_table"
_MD_HEAD = "**Data inspection**"

_QUALIFIED_PAIR_RE = re.compile(
    r"(GLOBlit_\d+)\s*::\s*((?:PROP|RXN)block_\d+)"
)
_BARE_BLOCK_RE = re.compile(r"^(?:PROP|RXN)block_\d+$")

# Deterministic renders of the fitting-tool family (fit_block, fit_block_
# derived, fit_multi_system, sweeps, predict_from_rk, compute_ideal_baseline,
# get_pure_values).  Agent-built blocks (register_custom_block) are NOT a
# grounded rail and are deliberately excluded.
_FIT_HEAD_RE = re.compile(
    r"\*\*(?:[\w-]+ )*RK (?:Fit|Prediction)\*\*"
    r"|\*\*Pure values\*\*|\*\*Ideal baseline\*\*"
)
# DOIs/id tokens would otherwise shed junk floats ("2018.02" out of
# 10.1016/j.jct.2018.02.022) that could falsely ground answer values.
_FIT_MASK_RE = re.compile(
    r"10\.\d{4,5}/\S+|GLOB\w+_\d+|(?:PROP|RXN)block_\d+|BLK\w+_\d+"
    r"|INSP_[0-9a-f]+"
)
_FIT_BLOCK_RE = re.compile(r"(?:PROP|RXN)block_\d+")

_HEAD_RE = re.compile(
    r"\*\*Data inspection\*\*\s*[—-]+\s*(?P<doi>\S+)"
    # validate_native_tool_result annotates DOIs with their GLOBlit sibling
    r"(?:\s*\[(?P<lit>GLOBlit_\d+)\])?\s*::\s*"
    r"(?P<block>(?:PROP|RXN)block_\d+)(?:\s*\[(?P<subsys>BLKsubsys_\d+)\])?"
)
_MODE_RE = re.compile(r"^mode:\s*(?P<mode>complete|RDP|nearest|head)", re.M)
_INSP_ID_RE = re.compile(r"^inspection_id:\s*(INSP_[0-9a-f]{12})\s*$", re.M)
_NUM_TOKEN_RE = re.compile(
    r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
)


def _table_rows(lines: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Extract the first pipe table as (columns, row dicts of verbatim cells)."""
    columns: list[str] = []
    rows: list[dict[str, str]] = []
    for line in lines:
        text = line.strip()
        if not text.startswith("|"):
            if columns and rows:
                break
            continue
        cells = [c.strip() for c in text.strip("|").split("|")]
        if not columns:
            columns = cells
            continue
        if all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        if len(cells) == len(columns):
            rows.append(dict(zip(columns, cells)))
    return columns, rows


def _float_or_none(cell: str) -> float | None:
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def parse_inspection_markdown(text: str) -> dict[str, Any] | None:
    """Parse one ``inspect_block_table`` markdown result into a registry entry.

    Returns ``None`` when *text* is not a successful inspection rendering
    (error results and unrelated tool output are silently skipped).
    """
    if not isinstance(text, str) or _MD_HEAD not in text:
        return None
    head = _HEAD_RE.search(text)
    insp_id = _INSP_ID_RE.search(text)
    if not head or not insp_id:
        return None
    lines = text.splitlines()
    columns, rows = _table_rows(lines)
    if not columns or not rows:
        return None

    grounded: set[float] = set()
    for row in rows:
        for cell in row.values():
            value = _float_or_none(cell)
            if value is not None:
                grounded.add(value)
    # Stats / bracket / constraints lines carry groundable aggregates.
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("stats (", "bracket:", "block constraints:")):
            for token in _NUM_TOKEN_RE.findall(stripped):
                value = _float_or_none(token)
                if value is not None:
                    grounded.add(value)

    mode = _MODE_RE.search(text)
    return {
        "doi": head.group("doi"),
        "lit_num_id": head.group("lit"),
        "block_number": head.group("block"),
        "BLKsubsys_id": head.group("subsys"),
        "table_mode": (mode.group("mode").lower() if mode else "complete"),
        "columns": columns,
        "rows_shown": rows,
        "grounded_values": grounded,
        "inspection_id": insp_id.group(1),
        "markdown": text,
    }


def entry_from_structured(item: dict[str, Any]) -> dict[str, Any] | None:
    """Build a registry entry from an envelope ``data_inspections`` item."""
    if not isinstance(item, dict):
        return None
    required = ("doi", "block_number", "columns", "rows_shown", "inspection_id")
    if any(field not in item for field in required):
        return None
    rows = item["rows_shown"]
    if not isinstance(rows, list) or not rows:
        return None
    grounded: set[float] = set()
    for row in rows:
        if not isinstance(row, dict):
            return None
        for cell in row.values():
            value = _float_or_none(cell)
            if value is not None:
                grounded.add(value)
    stats = item.get("stats")
    if isinstance(stats, dict):
        for scope in stats.values():
            if not isinstance(scope, dict):
                continue
            for entry in scope.values():
                if not isinstance(entry, dict):
                    continue
                for key in ("min", "max", "n", "n_finite"):
                    value = _float_or_none(entry.get(key))
                    if value is not None:
                        grounded.add(value)
    bracket = item.get("bracket")
    if isinstance(bracket, dict):
        for key in ("target",):
            value = _float_or_none(bracket.get(key))
            if value is not None:
                grounded.add(value)
        for side in ("below", "above"):
            cell = bracket.get(side)
            if isinstance(cell, dict):
                value = _float_or_none(cell.get("value"))
                if value is not None:
                    grounded.add(value)
    return {
        "doi": item["doi"],
        "lit_num_id": item.get("lit_num_id"),
        "block_number": item["block_number"],
        "BLKsubsys_id": item.get("BLKsubsys_id"),
        "table_mode": item.get("table_mode", "complete"),
        "columns": list(item["columns"]),
        "rows_shown": rows,
        "grounded_values": grounded,
        "inspection_id": item["inspection_id"],
        "markdown": item.get("markdown", ""),
    }


def _iter_embedded_envelope_inspections(text: str):
    """Yield ``data_inspections`` items from child-envelope JSON in a result."""
    if not isinstance(text, str) or '"data_inspections"' not in text:
        return
    start = text.find("{")
    while start != -1:
        try:
            decoded, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        stack = [decoded]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                items = node.get("data_inspections")
                if isinstance(items, list):
                    yield from items
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        return


def harvest_inspections(tool_history: list[dict] | None) -> list[dict[str, Any]]:
    """Collect all verbatim inspections visible in this run, deduplicated.

    Detection is content-based: the deterministic inspection markdown is
    recognized wherever it appears (direct calls, menu-executed runs, child
    envelopes) — tool names differ across layers.
    """
    if not tool_history:
        return []
    entries: dict[str, dict[str, Any]] = {}
    for item in tool_history:
        if not isinstance(item, dict):
            continue
        result = item.get("result_full")
        if not isinstance(result, str):
            result = json.dumps(result, default=str) if result is not None else ""
        if _MD_HEAD in result:
            parsed = parse_inspection_markdown(result)
            if parsed:
                entries.setdefault(parsed["inspection_id"], parsed)
                continue
        for embedded in _iter_embedded_envelope_inspections(result):
            entry = entry_from_structured(embedded)
            if entry:
                entries.setdefault(entry["inspection_id"], entry)
    return list(entries.values())


def harvest_fit_artifacts(
    tool_history: list[dict] | None,
) -> list[dict[str, Any]]:
    """Pool numeric artifacts from deterministic fit-tool renderings.

    Fit tools are their own grounded rail: coefficients, fit statistics,
    pure-component values, baselines and predictions exist in no database
    table, so the gate needs the rendered fit-family results as a
    comparison pool.  Detection is content-based (house render headings,
    whichever layer ran the tool); each render is pooled under a
    deterministic ``FIT_<sha1[:12]>`` id so matches can cite the artifact.
    Pure text processing — no DB access, no LLM.
    """
    entries: dict[str, dict[str, Any]] = {}
    for item in tool_history or []:
        if not isinstance(item, dict):
            continue
        result = item.get("result_full")
        if not isinstance(result, str) or not _FIT_HEAD_RE.search(result):
            continue
        fit_id = "FIT_" + hashlib.sha1(
            result.encode("utf-8")).hexdigest()[:12]
        if fit_id in entries:
            continue
        values: set[float] = set()
        for token in _NUM_TOKEN_RE.findall(_FIT_MASK_RE.sub(" ", result)):
            value = _float_or_none(token)
            if value is not None:
                values.add(value)
        heading = next(
            (line.strip() for line in result.splitlines() if line.strip()),
            "",
        )
        entries[fit_id] = {
            "fit_id": fit_id,
            "heading": heading[:120],
            "values": values,
            "blocks": sorted(set(_FIT_BLOCK_RE.findall(result))),
        }
    return list(entries.values())


def harvest_inspection_targets(
    tool_history: list[dict] | None,
) -> dict[str, list[str]]:
    """Pool lit↔block chains from every recorded tool ARGUMENT set.

    Every inspection / targeted-search call had to name its literature and
    block to execute at all, so the deduplicated argument pool recovers the
    identity chains even for calls that errored and left no parseable
    result — exactly the blocks the answer's numbers came from.  Returns
    ``{block_number: [lit hints, first-seen order]}``; hints are GLOBlit
    ids verbatim or DOIs passed through (resolution is the caller's job).
    Pure text processing — no DB access, no LLM.
    """
    targets: dict[str, list[str]] = {}

    def _note(block: str, hint: str) -> None:
        hints = targets.setdefault(block, [])
        if hint and hint not in hints:
            hints.append(hint)

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            block = node.get("block_number")
            lit = node.get("literature") or node.get("doi")
            if (
                isinstance(block, str) and _BARE_BLOCK_RE.match(block.strip())
                and isinstance(lit, str) and lit.strip()
            ):
                _note(block.strip(), lit.strip())
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)
        elif isinstance(node, str):
            for match in _QUALIFIED_PAIR_RE.finditer(node):
                _note(match.group(2), match.group(1))

    for item in tool_history or []:
        if isinstance(item, dict):
            _walk(item.get("arguments"))
    return targets


def export_inspections(entries: list[dict[str, Any]], *, cap: int = 24) -> list[dict]:
    """Envelope-ready ``data_inspections`` payload (JSON-safe, bounded).

    ``lit_num_id`` is enriched deterministically from the reference
    registry when the source entry lacks it — downstream agents reference
    literature by ``GLOBlit_N``, never by reconstructing DOIs.
    """
    from ThermoML_raw_json_to_card_db_parsers.id_schema import lit_num_id_for_doi

    exported = []
    for entry in entries[:cap]:
        item = {
            "doi": entry["doi"],
            "block_number": entry["block_number"],
            "table_mode": entry["table_mode"],
            "columns": entry["columns"],
            "rows_shown": entry["rows_shown"],
            "inspection_id": entry["inspection_id"],
        }
        lit = entry.get("lit_num_id") or lit_num_id_for_doi(
            entry["doi"], allow_unregistered=True,
        )
        if lit:
            item["lit_num_id"] = lit
        # Subsystem anchoring is opt-in — never emit a null filler.
        if entry.get("BLKsubsys_id"):
            item["BLKsubsys_id"] = entry["BLKsubsys_id"]
        exported.append(item)
    return exported
