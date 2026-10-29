"""Mandatory data-grounding gate — deterministic final-answer check.

Enforces the inspection mandate at every layer that can quote data points:
any numeric data literal in a block-anchored answer segment must be grounded
in a verbatim ``inspect_block_table`` result from this run (or a child
envelope's ``data_inspections``).  Fit tools are their own grounded rail:
their deterministic renders are pooled (``harvest_fit_artifacts``) and a
literal matching a pooled artifact is verified with its ``FIT_`` id; fit-
metric prose context stays exempt as before.

Beyond the in-session evidence, the gate hardcode-fetches the AUTHORITATIVE
full tables of every cited block (``authoritative_fetch``) to sharpen its
verdicts — never to waive the inspection mandate:

- a literal found in the cited block's true table but never inspected →
  ``UNINSPECTED_VALUE`` (bounce; the suggested call is guaranteed to land);
- a literal found in exactly ONE other source → ``MISATTRIBUTED_VALUE``
  (the true block is cited in the flag);
- a literal found in several sources → ``AMBIGUOUS_VALUE`` (flag only —
  listing every match would be noise);
- markdown tables in the answer merge the evidence of ALL blocks anchored
  inside them (one table may legitimately mix several sources).

Authoritative matching uses tightened tolerances (displayed precision,
scales 1/10³/10⁻³ only) — full tables are dense, so the generous in-session
sweep would over-match.

The literal-extraction and matching rules are ported from the 103-session
handover audit (scale-tolerant, displayed-precision-tolerant matching).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .inspection_harvest import (
    harvest_fit_artifacts,
    harvest_inspection_targets,
    harvest_inspections,
)

GATE_MARKER = "[DATA GROUNDING]"
FLAGS_HEADER = "[UNGROUNDED DATA FLAGS]"
MAX_GATE_BOUNCES = 2
_MAX_LISTED = 8

# Menu-driven layers reach the inspection tool through run_subagent_tool;
# the analysis L0 inspects directly (inspect_block) and receives row-level
# inspections through child envelopes.
_GATE_ENABLING_TOOLS = {
    "inspect_block_table", "inspect_block", "run_subagent_tool",
}


def gate_enabled(tools) -> bool:
    """The mandate applies wherever the inspection tool is reachable."""
    try:
        names = set(tools)
    except TypeError:
        return False
    return bool(names & _GATE_ENABLING_TOOLS)

# ── Ported literal-extraction rules (audit-proven) ─────────────────────────
# U+2212 (true minus) is how LLM tables write negative values — without the
# translation every negative literal parses positive and can never match.
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺−", "0123456789-+-")
_ID_RE = re.compile(
    r"10\.\d{4,5}/[^\s|,;()\[\]\"']+|GLOB\w+_\d+|PROPblock_\d+|RXNblock_\d+"
    r"|BLK\w+_\d+|DOIcomp\w*_\d+|C\d*H\d+\w*|\d{4}-\w+-\w+-\d+|INSP_[0-9a-f]+"
)
_NUM_RE = re.compile(
    # "#" blocks table/row citation ordinals (A#1_Table#10_Row#37)
    r"(?<![\w/.\-#])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\s*%)"
)
_FIT_RE = re.compile(
    r"R²|R\^?2|RMSE|BIC|AIC|A[₀₁₂₃₄₅]|coefficient|order|Redlich|residual"
    r"|fit_block|polynomial", re.I,
)
_SCALES = (1.0, 1e3, 1e-3, 1e2, 1e-2, 1e6, 1e-6, 1e9, 1e-9)
_TIGHT_SCALES = (1.0, 1e3, 1e-3)

_BLOCK_ANCHOR_RE = re.compile(
    r"(?:(GLOBlit_\d+)\s*::\s*)?((?:PROP|RXN)block_\d+)"
)


def _norm(text: str) -> str:
    text = text.translate(_SUP)
    return re.sub(
        r"(\d[\d.]*)\s*[×x]\s*10(-?\+?\d+)",
        lambda m: f"{m.group(1)}e{int(m.group(2))}",
        text,
    )


def _decimals(token: str) -> int:
    if "e" in token.lower():
        mantissa, exponent = token.lower().split("e")
        places = len(mantissa.split(".")[1]) if "." in mantissa else 0
        return places - int(exponent)
    return len(token.split(".")[1]) if "." in token else 0


def _near(literal: float, token: str, value: float) -> bool:
    """Displayed-precision + power-of-ten unit-scale tolerant match."""
    tol = 0.51 * 10 ** (-_decimals(token))
    for scale in _SCALES:
        scaled = value * scale
        if abs(literal - scaled) <= tol:
            return True
        if scaled and abs(literal - scaled) / abs(scaled) <= 5e-4:
            return True
    return False


def _near_tight(literal: float, token: str, value: float) -> bool:
    """Authoritative-table match: displayed precision, common scales only."""
    tol = 0.51 * 10 ** (-_decimals(token))
    return any(abs(literal - value * s) <= tol for s in _TIGHT_SCALES)


def _literals(text: str) -> list[tuple[float, str, str]]:
    """Checkable data literals with context (IDs/years/counts/fit lines out)."""
    masked = _ID_RE.sub(" ", _norm(text))
    found = []
    for m in _NUM_RE.finditer(masked):
        token = m.group(0).replace(",", "")
        try:
            value = float(token)
        except ValueError:
            continue
        if "." not in token and "e" not in token.lower() and (
            abs(value) <= 12 or 1900 <= value <= 2099
        ):
            continue
        context = masked[max(0, m.start() - 60):m.end() + 40].replace("\n", " ")
        if _FIT_RE.search(context):
            continue
        found.append((value, token, context.strip()))
    return found


def _segments(text: str) -> list[tuple[str | None, str, str]]:
    """Split *text* into (lit_id, block_number, segment) runs at block anchors."""
    marks = [
        (m.start(), m.group(1), m.group(2))
        for m in _BLOCK_ANCHOR_RE.finditer(text)
    ]
    segments = []
    for index, (pos, lit, block) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else len(text)
        segments.append((lit, block, text[pos:end]))
    return segments


def _table_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of contiguous markdown-table line runs."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    pos = 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("|"):
            if start is None:
                start = pos
        elif start is not None:
            spans.append((start, pos))
            start = None
        pos += len(line)
    if start is not None:
        spans.append((start, len(text)))
    return spans


def _add_anchor(anchors: list[tuple[str | None, str]], lit: str | None,
                block: str) -> None:
    """Append (lit, block) deduplicated by block; a lit upgrades a bare entry."""
    for index, (known_lit, known_block) in enumerate(anchors):
        if known_block == block:
            if lit and not known_lit:
                anchors[index] = (lit, block)
            return
    anchors.append((lit, block))


def _regions(text: str) -> list[tuple[list[tuple[str | None, str]], str, bool]]:
    """Split *text* into (anchors, region_text, is_table) evidence regions.

    Prose follows the last-anchor-before rule (one anchor per region);
    a markdown table merges ALL anchors that appear inside it — one answer
    table may legitimately mix several source blocks.  Anchor-less regions
    inherit the carry-over anchor; text before the first anchor is never
    checked.
    """
    regions: list[tuple[list[tuple[str | None, str]], str, bool]] = []
    carry: tuple[str | None, str] | None = None

    def _prose(chunk: str) -> None:
        nonlocal carry
        marks = list(_BLOCK_ANCHOR_RE.finditer(chunk))
        lead_end = marks[0].start() if marks else len(chunk)
        if carry is not None and chunk[:lead_end].strip():
            regions.append(([carry], chunk[:lead_end], False))
        for index, mark in enumerate(marks):
            end = marks[index + 1].start() if index + 1 < len(marks) else len(chunk)
            carry = (mark.group(1), mark.group(2))
            regions.append(([carry], chunk[mark.start():end], False))

    cursor = 0
    for tstart, tend in _table_spans(text):
        _prose(text[cursor:tstart])
        table = text[tstart:tend]
        anchors: list[tuple[str | None, str]] = []
        for mark in _BLOCK_ANCHOR_RE.finditer(table):
            _add_anchor(anchors, mark.group(1), mark.group(2))
        if not anchors and carry is not None:
            anchors = [carry]
        if anchors:
            regions.append((anchors, table, True))
            carry = anchors[-1]
        cursor = tend
    _prose(text[cursor:])
    return regions


# ── Gate result object ─────────────────────────────────────────────────────

@dataclass
class GateReport:
    ok: bool
    violations: list[dict] = field(default_factory=list)
    inspected_blocks: list[str] = field(default_factory=list)
    # Anchors whose quoted literals ALL matched inspections; each entry
    # carries the verified literal tokens: {"anchor": str, "values": [str]}.
    verified: list[dict] = field(default_factory=list)
    # Flagged anchors that also have grounded literals — same entry shape.
    partially_grounded: list[dict] = field(default_factory=list)
    # JSON-safe evidence trail (cited blocks, per-literal verdicts) for the
    # passive grounding-evidence log — never rendered into prompts.
    evidence: dict = field(default_factory=dict)

    @property
    def report_text(self) -> str:
        return build_violation_report(self)


def run_data_grounding_gate(
    answer: str,
    tool_history: list[dict] | None,
    *,
    inspections: list[dict] | None = None,
    authoritative_fetch=None,
) -> GateReport:
    """Check every block-anchored data literal in *answer* against inspections.

    ``authoritative_fetch(lit_num_id, block_number) -> dict | None`` supplies
    the true full-table values of cited blocks (default: registry-backed
    adapter; inject a stub in tests).  It sharpens verdicts and powers the
    evidence log but NEVER waives the inspection mandate.

    Violation kinds:

    - ``UNINSPECTED_BLOCK`` — a region quotes data literals but none of its
      cited blocks was inspected in this run (mandatory inspection).
    - ``UNINSPECTED_VALUE`` — the literal exists in the cited block's
      authoritative table but was never shown in an inspection.
    - ``MISATTRIBUTED_VALUE`` — the literal belongs to exactly one OTHER
      source; that block is cited in the flag.
    - ``AMBIGUOUS_VALUE`` — the literal matches several sources.
    - ``UNGROUNDED_LITERAL`` — the literal matches nothing anywhere.
    """
    if not isinstance(answer, str) or not answer.strip():
        return GateReport(ok=True)
    if inspections is None:
        inspections = harvest_inspections(tool_history)
    if authoritative_fetch is None:
        from .authoritative_blocks import default_authoritative_fetch
        authoritative_fetch = default_authoritative_fetch
    fit_artifacts = harvest_fit_artifacts(tool_history)

    def _fit_source(value: float, token: str) -> str | None:
        for fit in fit_artifacts:
            if any(_near_tight(value, token, v) for v in fit["values"]):
                return fit["fit_id"]
        return None

    by_block: dict[str, list[dict]] = {}
    for entry in inspections:
        by_block.setdefault(entry["block_number"], []).append(entry)

    regions = _regions(answer)

    # Authoritative tables, fetched once per cited block (all regions).
    cited: dict[str, str | None] = {}
    for anchors, _, _ in regions:
        for lit, block in anchors:
            if block not in cited or (lit and not cited[block]):
                cited[block] = lit
    # Bare anchors: resolve the literature from this run's inspections so
    # the authoritative fetch and repair calls still engage.
    for block, lit in list(cited.items()):
        if not lit:
            for entry in by_block.get(block, []):
                if entry.get("lit_num_id"):
                    cited[block] = entry["lit_num_id"]
                    break
    # Third tier: the pooled tool-call ARGUMENTS — every inspect/targeted
    # call named its lit+block to execute, covering even errored calls.
    arg_targets = harvest_inspection_targets(tool_history)

    def _hint_to_lit(hint: str) -> str | None:
        if hint.startswith("GLOBlit_"):
            return hint
        try:  # DOI hint — lazy registry lookup, best-effort
            from ThermoML_raw_json_to_card_db_parsers.id_schema import (
                lit_num_id_for_doi,
            )
            return lit_num_id_for_doi(hint, allow_unregistered=True)
        except Exception:
            return None

    for block, lit in list(cited.items()):
        if not lit:
            for hint in arg_targets.get(block, []):
                resolved = _hint_to_lit(hint)
                if resolved:
                    cited[block] = resolved
                    break
    authoritative: dict[str, dict | None] = {}
    for block, lit in cited.items():
        try:
            authoritative[block] = authoritative_fetch(lit, block) if lit else None
        except Exception:  # fetch is best-effort — never breaks the gate
            authoritative[block] = None

    def _auth_match(block: str, value: float, token: str) -> bool:
        table = authoritative.get(block)
        return bool(table) and any(
            _near_tight(value, token, v) for v in table["values"]
        )

    violations: list[dict] = []
    verified_tokens: dict[str, list[str]] = {}
    flagged_anchors: set[str] = set()
    verdicts: list[dict] = []

    def _note_verified(anchor: str, token: str) -> None:
        tokens = verified_tokens.setdefault(anchor, [])
        if token not in tokens:
            tokens.append(token)

    for anchors, segment, _is_table in regions:
        found = _literals(segment)
        if not found:
            continue
        blocks = [block for _, block in anchors]
        anchor = " + ".join(
            f"{lit}::{block}" if lit else block for lit, block in anchors
        )
        insp_entries = [e for block in blocks for e in by_block.get(block, [])]

        grounded: set[float] = set()
        for entry in insp_entries:
            grounded |= entry["grounded_values"]

        # Fit tools are their own grounded rail: a literal emitted by a fit
        # rendering this run is grounded wherever it is quoted (inspection
        # row matches still take precedence as plain `verified` below).
        if fit_artifacts:
            remaining = []
            for value, token, context in found:
                fit_id = (
                    None
                    if any(_near(value, token, g) for g in grounded)
                    else _fit_source(value, token)
                )
                if fit_id:
                    _note_verified(anchor, token)
                    verdicts.append({
                        "token": token, "region": anchor,
                        "verdict": "fit_artifact", "source": fit_id,
                    })
                else:
                    remaining.append((value, token, context))
            found = remaining
            if not found:
                continue

        if not insp_entries:
            # Mandatory inspection: never waived, even when the database
            # confirms the values — but say so, the repair is then certain.
            db_confirmed = [
                token for value, token, _ in found
                if any(_auth_match(block, value, token) for block in blocks)
            ]
            note = (
                f" (values {', '.join(db_confirmed[:4])} verified against the "
                "database — inspection is still mandatory)"
                if db_confirmed else ""
            )
            flagged_anchors.add(anchor)
            sample = ", ".join(token for _, token, _ in found[:4])
            violations.append({
                "kind": "UNINSPECTED_BLOCK",
                "anchor": anchor,
                "block_number": blocks[0],
                "lit_id": anchors[0][0],
                "literals": [token for _, token, _ in found],
                "detail": (
                    f"{anchor} is quoted with data values ({sample}, …) but "
                    f"was never inspected in this run{note}"
                ),
                "suggested_call": "; ".join(
                    _suggest_call(cited.get(block) or lit, block)
                    for lit, block in anchors
                ),
            })
            verdicts.extend(
                {"token": token, "region": anchor, "verdict": "uninspected_block"}
                for _, token, _ in found
            )
            continue

        for value, token, context in found:
            # 1. in-session inspections of this region's blocks (merged)
            if any(_near(value, token, g) for g in grounded):
                _note_verified(anchor, token)
                verdicts.append(
                    {"token": token, "region": anchor, "verdict": "verified"})
                continue
            # 2. authoritative table of the cited blocks (tight tolerance)
            auth_block = next(
                (block for block in blocks if _auth_match(block, value, token)),
                None,
            )
            if auth_block is not None:
                flagged_anchors.add(anchor)
                table = authoritative[auth_block] or {}
                violations.append({
                    "kind": "UNINSPECTED_VALUE",
                    "anchor": anchor,
                    "block_number": auth_block,
                    "lit_id": cited.get(auth_block),
                    "literals": [token],
                    "detail": (
                        f"{token} exists in {auth_block}'s database table but "
                        "was never shown in an inspection of this run"
                    ),
                    "suggested_call": _suggest_call(
                        cited.get(auth_block), auth_block, focus_value=value,
                        entries=by_block.get(auth_block),
                        column_ranges=table.get("column_ranges"),
                    ),
                })
                verdicts.append({
                    "token": token, "region": anchor,
                    "verdict": "uninspected_value", "source": auth_block,
                })
                continue
            # 3. attribution scan across all OTHER evidence (tight tolerance)
            fits: set[str] = set()
            for other_block, entries in by_block.items():
                if other_block in blocks:
                    continue
                other_grounded: set[float] = set()
                for entry in entries:
                    other_grounded |= entry["grounded_values"]
                if any(_near_tight(value, token, g) for g in other_grounded):
                    fits.add(other_block)
            for other_block in cited:
                if other_block not in blocks and other_block not in fits:
                    if _auth_match(other_block, value, token):
                        fits.add(other_block)
            flagged_anchors.add(anchor)
            if len(fits) == 1:
                true_block = next(iter(fits))
                true_lit = cited.get(true_block) or (
                    by_block[true_block][0].get("lit_num_id")
                    if true_block in by_block else None
                )
                true_anchor = (
                    f"{true_lit}::{true_block}" if true_lit else true_block
                )
                inspected = true_block in by_block
                violations.append({
                    "kind": "MISATTRIBUTED_VALUE",
                    "anchor": anchor,
                    "block_number": true_block,
                    "lit_id": true_lit,
                    "literals": [token],
                    "detail": (
                        f"{token} does not belong to {anchor}; it matches only "
                        f"{true_anchor}"
                        + (" (inspected this run)" if inspected
                           else " (cited but uninspected)")
                    ),
                    "suggested_call": (
                        f"re-anchor {token} to {true_anchor} — no new tool call needed"
                        if inspected else _suggest_call(true_lit, true_block)
                    ),
                })
                verdicts.append({
                    "token": token, "region": anchor,
                    "verdict": "misattributed", "source": true_block,
                })
            elif fits:
                violations.append({
                    "kind": "AMBIGUOUS_VALUE",
                    "anchor": anchor,
                    "block_number": blocks[0],
                    "lit_id": anchors[0][0],
                    "literals": [token],
                    "detail": (
                        f"{token} matches {len(fits)} different sources — "
                        "ambiguous; re-quote it from an inspection of the "
                        "block you mean"
                    ),
                    "suggested_call": _suggest_call(
                        anchors[0][0], blocks[0], focus_value=value,
                        entries=by_block.get(blocks[0]),
                    ),
                })
                verdicts.append({
                    "token": token, "region": anchor, "verdict": "ambiguous",
                    "source_count": len(fits),
                })
            else:
                violations.append({
                    "kind": "UNGROUNDED_LITERAL",
                    "anchor": anchor,
                    "block_number": blocks[0],
                    "lit_id": anchors[0][0],
                    "literals": [token],
                    "detail": (
                        f"{token} (near '…{context[:60]}…') matches no "
                        f"inspected row/stat, database table, or other source "
                        f"of {anchor}"
                    ),
                    "suggested_call": _suggest_call(
                        anchors[0][0], blocks[0], focus_value=value,
                        entries=by_block.get(blocks[0]),
                    ),
                })
                verdicts.append({
                    "token": token, "region": anchor, "verdict": "ungrounded"})

    evidence = {
        "fit_artifacts": [
            {
                "fit_id": fit["fit_id"],
                "heading": fit["heading"],
                "blocks": fit["blocks"],
                "n_values": len(fit["values"]),
            }
            for fit in fit_artifacts
        ],
        "cited_blocks": {
            block: {
                "lit_num_id": lit,
                "inspected": block in by_block,
                "inspection_ids": [
                    entry["inspection_id"] for entry in by_block.get(block, [])
                ],
                "authoritative": (
                    {
                        "doi": authoritative[block]["doi"],
                        "n_rows": authoritative[block]["n_rows"],
                        "columns": authoritative[block]["columns"],
                        "column_ranges": authoritative[block]["column_ranges"],
                    }
                    if authoritative.get(block) else None
                ),
            }
            for block, lit in cited.items()
        },
        "literal_verdicts": verdicts,
    }

    return GateReport(
        ok=not violations,
        violations=violations,
        inspected_blocks=sorted(by_block),
        verified=[
            {"anchor": anchor, "values": tokens}
            for anchor, tokens in verified_tokens.items()
            if anchor not in flagged_anchors
        ],
        partially_grounded=[
            {"anchor": anchor, "values": tokens}
            for anchor, tokens in verified_tokens.items()
            if anchor in flagged_anchors
        ],
        evidence=evidence,
    )


def _suggest_call(
    lit: str | None,
    block: str,
    *,
    focus_value: float | None = None,
    entries: list[dict] | None = None,
    column_ranges: dict | None = None,
) -> str:
    if lit:
        target = f'block_number="{lit}::{block}"'
    else:
        target = f'block_number="{block}", literature="<GLOBlit_N or DOI>"'
    if focus_value is not None and (entries or column_ranges):
        axis = None
        if column_ranges:
            # Prefer the column whose true range contains the value.
            axis = next(
                (
                    column for column, bounds in column_ranges.items()
                    if column != "BLKpoint_id" and bounds
                    and bounds[0] is not None and bounds[1] is not None
                    and bounds[0] <= focus_value <= bounds[1]
                ),
                None,
            )
        if axis is None and entries:
            columns = entries[0].get("columns") or []
            axis = next((c for c in columns if c != "BLKpoint_id"), None)
        if axis is None and column_ranges:
            axis = next(
                (c for c in column_ranges if c != "BLKpoint_id"), "…",
            )
        return (
            f"inspect_block_table({target}, nearest={{\"column\": "
            f"\"{axis}\", \"value\": {focus_value}}})"
        )
    return f"inspect_block_table({target})"


def build_violation_report(report: GateReport) -> str:
    """Agent-facing violation summary with concrete repair calls.

    The fix line carries the epistemic weight the gate actually has:
    kinds where the database PROVES the value is (or must be) DB data
    get a required fix; kinds without DB proof get a conditional fix —
    inspect only if the number is meant to be a database datum.
    """
    _FIX_PREFIX = {
        "UNINSPECTED_BLOCK": (
            "fix (required — this region quotes data from cited database"
            " blocks)"),
        "UNINSPECTED_VALUE": (
            "fix (required — this value IS in the cited block's database"
            " table)"),
        "MISATTRIBUTED_VALUE": (
            "fix (required — this IS a database value, under a different"
            " block)"),
    }
    lines = []
    for violation in report.violations[:_MAX_LISTED]:
        prefix = _FIX_PREFIX.get(violation["kind"])
        if prefix:
            fix = f"  {prefix}: `{violation['suggested_call']}`"
        else:  # AMBIGUOUS_VALUE / UNGROUNDED_LITERAL — no DB proof
            fix = (
                "  fix (ONLY if this is meant to be a database datum): "
                f"`{violation['suggested_call']}` — if it is a value you"
                " computed or an external reference, keep it and label its"
                " provenance instead"
            )
        lines.append(
            f"- **{violation['kind']}** {violation['anchor']}: "
            f"{violation['detail']}\n{fix}"
        )
    overflow = report.violations[_MAX_LISTED:]
    if overflow:
        # Same kinds, without repeated fix lines — but still itemize every
        # offending value with its anchor.
        by_anchor: dict[str, list[str]] = {}
        for violation in overflow:
            tokens = by_anchor.setdefault(violation["anchor"], [])
            for token in violation["literals"]:
                if token not in tokens:
                    tokens.append(token)
        itemized = "; ".join(
            f"{anchor}: {', '.join(tokens)}"
            for anchor, tokens in by_anchor.items()
        )
        lines.append(
            f"- … and {len(overflow)} more of the same kinds — {itemized}"
        )
    if report.inspected_blocks:
        lines.append(
            "Inspected so far: " + ", ".join(report.inspected_blocks)
        )
    return "\n".join(lines)


def build_grounded_context(report: GateReport) -> str:
    """Verified/partially-grounded values, itemized, so repairs stay surgical."""
    lines: list[str] = []
    if report.verified:
        lines.append(
            "Verified as grounded — correct as quoted, keep unchanged:"
        )
        for entry in report.verified:
            lines.append(f"- {entry['anchor']}: {', '.join(entry['values'])}")
    if report.partially_grounded:
        lines.append(
            "Partially grounded — the values listed below ARE verified and "
            "must stay as quoted; change ONLY the flagged one(s) above:"
        )
        for entry in report.partially_grounded:
            lines.append(f"- {entry['anchor']}: {', '.join(entry['values'])}")
    return "\n".join(lines)


def build_gate_nudge(report: GateReport) -> str:
    """Bounce message for the normal final-answer path (tools available)."""
    body = report.report_text
    grounded_note = build_grounded_context(report)
    if grounded_note:
        body += "\n\n" + grounded_note
    return (
        f"{GATE_MARKER} Your answer quotes data points that are not grounded "
        "in a verbatim `inspect_block_table` result from this run:\n\n"
        + body
        + "\n\nRepair surgically — this is NOT a rewrite. Keep every verified "
        "value and all unflagged prose of your previous answer unchanged, and "
        "do not add new data values. Triage each flagged value by what it IS:\n"
        "- Fixes marked `required` concern values that are (or must be) "
        "database data: run the suggested `inspect_block_table` call(s) now "
        "(if you have no direct inspection tool, dispatch a targeted data "
        "query for exactly those blocks — its worker returns the inspection) "
        "and re-quote each value verbatim at source precision. MISATTRIBUTED "
        "values need only their anchor fixed to the cited true block.\n"
        "- Fixes marked `ONLY if` — decide the value's provenance first. "
        "Meant as a database datum → inspect it the same way. A quantity you "
        "computed (difference, interpolation, unit conversion) → keep it but "
        "label it explicitly as derived from the inspected rows. An external "
        "reference value (other literature, general knowledge) → keep it but "
        "state its origin explicitly — never present it as data from this "
        "database. None of these → remove it or state it was not inspected "
        "in this run; never estimate a replacement.\n"
        "Re-derive any comparative or interpretive statement that depended "
        "on a changed value from the inspected rows — do not carry over "
        "conclusions formed before inspection."
    )


def build_timeout_gate_instruction(report: GateReport) -> str:
    """Rewrite instruction for forced-final paths (no tools left)."""
    body = report.report_text
    grounded_note = build_grounded_context(report)
    if grounded_note:
        body += "\n\n" + grounded_note
    return (
        f"{GATE_MARKER} TIME IS UP and your answer contains data points that "
        "were never inspected in this run:\n\n"
        + body
        + "\n\nYou cannot run tools anymore. Rewrite your final answer NOW: "
        "DROP every uninspected value whose fix is marked `required` (it is, "
        "or must be, database data) — replace it with the block-level "
        "aggregates you actually retrieved, or state 'not inspected in this "
        "run'. Values with an `ONLY if` fix may stay ONLY when you label "
        "their provenance explicitly (derived from inspected rows / external "
        "reference with its origin) — never presented as data from this "
        "database. Keep every verified value exactly as quoted. Do NOT "
        "invent, round-trip, or estimate replacement numbers. Keep "
        "everything else unchanged."
    )


def build_flags_note(report: GateReport) -> str:
    """Deterministic disclaimer appended when bounces are exhausted."""
    return (
        f"\n\n---\n{FLAGS_HEADER} The following quoted values could not be "
        "verified against any verbatim data inspection in this run and must "
        "be treated as UNVERIFIED:\n" + report.report_text
    )


def count_gate_bounces(memory: list[dict]) -> int:
    """How many grounding bounces already happened in this conversation."""
    return sum(
        1 for message in memory
        if message.get("role") == "user"
        and GATE_MARKER in str(message.get("content", ""))
    )
