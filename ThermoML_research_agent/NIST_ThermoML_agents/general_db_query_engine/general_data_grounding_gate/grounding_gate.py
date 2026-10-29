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

from . import profile

import logging
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable

from .inspection_harvest import (
    INSPECT_TOOL_NAME,
    harvest_fit_artifacts,
    harvest_inspection_targets,
    harvest_inspections,
)

GATE_MARKER = "[DATA GROUNDING]"
FLAGS_HEADER = "[UNGROUNDED DATA FLAGS]"
MAX_GATE_BOUNCES = 2
_MAX_LISTED = 8

# One-time bounce when the answer after a grounding bounce collapsed to a
# stub (e.g. "No changes."): below this share of the bounced answer's length,
# and only when the bounced answer was long enough to be worth keeping.
CONTINUITY_MARKER = "[ANSWER CONTINUITY]"
_STUB_SHARE = 0.25
_STUB_MIN_BOUNCED_CHARS = 400

# Menu-driven layers reach the inspection tool through run_subagent_tool;
# the analysis L0 inspects directly (inspect_block) and receives row-level
# inspections through child envelopes.
_GATE_ENABLING_TOOLS = {
    "inspect_block_table", "inspect_block", "run_subagent_tool",
}

# The one switch of the active gate (R1 revision, E2): ``THERMOML_GROUNDING_GATE=off``
# in the process environment turns ``gate_enabled`` off everywhere it is
# consulted — no final-answer check, no bounce, no ``[DATA GROUNDING]`` nudge
# and no forced-final rewrite — while the passive grounding-evidence writer
# (``evidence_log.write_grounding_evidence``, called by the engine hooks
# without consulting this switch) keeps recording the same verdicts per run.
# Absent or any other value = on (the published behaviour).
# Gate off = no gate AND no ledger: the workflow prompts drop their marked
# grounding-ledger spans (subworkflow_md_parser) and the agents are offered
# none of the ledger's tools (``without_ledger_tools``).
GATE_ENV_VAR = "THERMOML_GROUNDING_GATE"
_GATE_OFF_WORDS = {"off", "0", "false", "no"}
_log = logging.getLogger(__name__)
_off_logged = False


def gate_switched_off() -> bool:
    """True when ``THERMOML_GROUNDING_GATE`` is set to off / 0 / false / no."""
    return os.environ.get(GATE_ENV_VAR, "").strip().lower() in _GATE_OFF_WORDS


# The grounding ledger's own tools.
LEDGER_TOOLS = frozenset({INSPECT_TOOL_NAME})


def without_ledger_tools(entries) -> list:
    """``entries`` (tool entries or tool names) minus the ledger's tools when the gate is switched off."""
    entries = list(entries)
    if not gate_switched_off():
        return entries
    return [e for e in entries if getattr(e, "name", e) not in LEDGER_TOOLS]


def gate_enabled(tools) -> bool:
    """Apply the gate wherever inspection is reachable in the available tools.

    Inspection capability includes descendants exposed by delegation and
    menu tools. The same eligibility rule applies to every agent, without
    agent-specific exemption. The legacy profile uses historical immediate
    tool eligibility. GATE_ENV_VAR remains the explicit
    master override for gate-off runs.
    """
    global _off_logged
    if gate_switched_off():
        if not _off_logged:
            _off_logged = True
            _log.info(
                "Data-grounding gate switched OFF by %s=%s (no check, no "
                "bounce; the passive grounding-evidence log is still written)",
                GATE_ENV_VAR, os.environ.get(GATE_ENV_VAR, ""),
            )
        return False
    if profile.is_legacy_mode():
        from .historical_september_2026.grounding_gate import gate_enabled as historical_enabled
        return historical_enabled(tools)
    try:
        names = set(tools)
    except TypeError:
        return False
    if names & _GATE_ENABLING_TOOLS:
        return True
    # Name-only callers cannot carry child providers; preserve that API.
    if not isinstance(tools, Mapping):
        return False
    from ..general_tool_management_helpers.tool_capabilities import (
        reachable_tool_names,
    )
    # The roots are this agent's actual tool map. Descendants come from
    # declared providers on those callables, never a union of global tools
    # or an exception keyed to an agent's name.
    return bool(reachable_tool_names(tools) & _GATE_ENABLING_TOOLS)

# ── Ported literal-extraction rules (audit-proven) ─────────────────────────
# U+2212 (true minus) is how LLM tables write negative values — without the
# translation every negative literal parses positive and can never match.
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺−", "0123456789-+-")
_ID_RE = re.compile(
    r"10\.\d{4,9}/[^\s|,;()\[\]\"']+|GLOB\w+_\d+|PROPblock_\d+|RXNblock_\d+"
    r"|BLK\w+_\d+|DOIcomp\w*_\d+|C\d*H\d+\w*|\d{4}-\w+-\w+-\d+|INSP_[0-9a-f]+"
)
_NUM_RE = re.compile(
    # "#" blocks table/row citation ordinals (A#1_Table#10_Row#37)
    r"(?<![\w/.\-#])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\s*%)"
)
# The default-on gate reads typographic thousands grouping as one number.
# Only horizontal spaces used for digit grouping are allowed: newlines,
# tabs and table-cell separators must remain separate literals.
_GROUPED_NUM_RE = re.compile(
    r"(?<![\w/.\-#])[-+]?(?:\d{1,3}(?:[, \u00a0\u202f]\d{3})+|\d+)"
    r"(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\s*%)"
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
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s<>\[\]|,;()\"'`]+", re.IGNORECASE)


def _normalize_doi(value: str) -> str:
    """Normalize citation wrappers, not DOI contents or source identity."""
    value = re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)+",
        "", value.strip(), flags=re.IGNORECASE,
    )
    return value.rstrip(".?!:;)]}*\"'").casefold()


def _cited_dois(answer: str) -> list[str]:
    return list(dict.fromkeys(
        _normalize_doi(match.group(0)) for match in _DOI_RE.finditer(answer)
    ))


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


def _table_cell(
    table: dict, literal: float, token: str,
) -> tuple[str | None, float]:
    """The table's own (column, cell) behind a quoted literal.

    `nearest` searches the raw cells of ONE column, so 2.7 quoted in mPa·s
    must ask for 0.0027 on the Pa·s column that holds it. The literal as
    printed (scale 1) wins. The column is None when the table carries no
    per-column values or only ``BLKpoint_id`` holds the cell.
    """
    tol = 0.51 * 10 ** (-_decimals(token))
    by_column = table.get("column_values") or {}
    columns = table.get("columns") or list(by_column)
    for scale in _TIGHT_SCALES:
        for column in columns:
            for cell in sorted(by_column.get(column) or ()):
                if abs(literal - cell * scale) <= tol:
                    return column, cell
        for cell in sorted(table.get("values") or ()):
            if abs(literal - cell * scale) <= tol:
                return None, cell
    return None, literal


_CELSIUS_UNIT_RE = re.compile(
    r"\s*(?:°\s*C\b|℃|degrees?\s+Celsius\b)", re.IGNORECASE,
)


def _literal_hits(text: str) -> list[tuple[float, str, str, str | None]]:
    """Checkable literals plus an explicit Celsius unit at each token offset."""
    masked = _ID_RE.sub(" ", _norm(text))
    number_re = _NUM_RE if gate_switched_off() else _GROUPED_NUM_RE
    found = []
    for m in number_re.finditer(masked):
        token = m.group(0).replace(",", "").replace(" ", "")
        token = token.replace("\u00a0", "").replace("\u202f", "")
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
        unit = _CELSIUS_UNIT_RE.match(masked, m.end())
        quoted_celsius = token + unit.group(0) if unit else None
        found.append((value, token, context.strip(), quoted_celsius))
    return found


def _literals(text: str) -> list[tuple[float, str, str]]:
    """Historical three-field literal API for existing callers."""
    return [(value, token, context) for value, token, context, _
            in _literal_hits(text)]


def _temperature_rows(
    entries: list[dict], value: float, token: str,
) -> list[dict]:
    """Inspected temperature_k cells matching an explicit Celsius quote."""
    kelvin = value + 273.15
    tol = 0.51 * 10 ** (-_decimals(token))
    matched = []
    for entry in entries:
        for row in entry.get("rows_shown") or []:
            if not isinstance(row, dict):
                continue
            try:
                stored = float(row["temperature_k"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(stored) and abs(kelvin - stored) <= tol:
                matched.append({
                    "stored_k": stored,
                    "inspection_id": entry["inspection_id"],
                })
    return matched


def _temperature_table(table: dict | None, value: float, token: str) -> bool:
    """Match only authoritative temperature_k cells, never another column."""
    if not table:
        return False
    kelvin = value + 273.15
    tol = 0.51 * 10 ** (-_decimals(token))
    return any(
        abs(kelvin - stored) <= tol
        for stored in (table.get("column_values") or {}).get("temperature_k", ())
    )


def _temperature_call(
    lit: str | None, block: str, entries: list[dict],
    table: dict | None, kelvin: float,
) -> str:
    """Suggest Kelvin nearest only when temperature_k is a known column."""
    has_temperature = (
        "temperature_k" in (table or {}).get("columns", ())
        or any("temperature_k" in row
               for entry in entries for row in entry.get("rows_shown") or []
               if isinstance(row, dict))
    )
    return _suggest_call(
        lit, block, focus_value=kelvin if has_temperature else None,
        axis="temperature_k" if has_temperature else None,
    )


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
                block: str, *, source_scope: bool = False) -> None:
    """Append an anchor; ON keeps distinct literature sources in a table."""
    for index, (known_lit, known_block) in enumerate(anchors):
        if known_block == block and (not source_scope or known_lit == lit):
            if lit and not known_lit:
                anchors[index] = (lit, block)
            return
    anchors.append((lit, block))


def _regions(
    text: str, *, source_scope: bool = False,
) -> list[tuple[list[tuple[str | None, str]], str, bool]]:
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
            _add_anchor(
                anchors, mark.group(1), mark.group(2),
                source_scope=source_scope,
            )
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


@profile.historical_dispatch("grounding_gate")
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

    # A DOI can close an answer without a usable PROPblock anchor. A trailing
    # block ID cannot make preceding DOI-cited claims vacuously pass either.
    # In mixed answers, check only the prefix that explicitly carries a DOI;
    # later block-anchored regions retain their historical parser.
    grounding_active = not gate_switched_off()
    regions = _regions(answer, source_scope=grounding_active)
    first_block = _BLOCK_ANCHOR_RE.search(answer)
    prefix = answer[:first_block.start()] if first_block else answer
    doi_report: GateReport | None = None
    if grounding_active and _cited_dois(answer):
        if first_block and _cited_dois(prefix) and _literals(prefix):
            doi_report = _run_doi_grounding_gate(
                prefix, inspections, authoritative_fetch, fit_artifacts,
            )
        if not any(_literals(segment) for _, segment, _ in regions):
            return doi_report or _run_doi_grounding_gate(
                answer, inspections, authoritative_fetch, fit_artifacts,
            )
        if not first_block:
            return _run_doi_grounding_gate(
                answer, inspections, authoritative_fetch, fit_artifacts,
            )

    by_block: dict[str, list[dict]] = {}
    for entry in inspections:
        by_block.setdefault(entry["block_number"], []).append(entry)

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
        found = _literal_hits(segment)
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
            for value, token, context, celsius in found:
                fit_id = (
                    None
                    if (grounding_active and celsius)
                    or any(_near(value, token, g) for g in grounded)
                    else _fit_source(value, token)
                )
                if fit_id:
                    _note_verified(anchor, token)
                    verdicts.append({
                        "token": token, "region": anchor,
                        "verdict": "fit_artifact", "source": fit_id,
                    })
                else:
                    remaining.append((value, token, context, celsius))
            found = remaining
            if not found:
                continue

        if not insp_entries:
            # Mandatory inspection: never waived, even when the database
            # confirms the values — but say so, the repair is then certain.
            db_confirmed = [
                token for value, token, _, celsius in found
                if any(
                    _temperature_table(authoritative.get(block), value, token)
                    if grounding_active and celsius else
                    _auth_match(block, value, token)
                    for block in blocks
                )
            ]
            note = (
                f" (values {', '.join(db_confirmed[:4])} verified against the "
                "database — inspection is still mandatory)"
                if db_confirmed else ""
            )
            flagged_anchors.add(anchor)
            sample = ", ".join(token for _, token, _, _ in found[:4])
            violations.append({
                "kind": "UNINSPECTED_BLOCK",
                "anchor": anchor,
                "block_number": blocks[0],
                "lit_id": anchors[0][0],
                "literals": [token for _, token, _, _ in found],
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
                for _, token, _, _ in found
            )
            continue

        for value, token, context, celsius in found:
            if grounding_active and celsius:
                kelvin = value + 273.15
                matched = _temperature_rows(insp_entries, value, token)
                if matched:
                    _note_verified(anchor, token)
                    verdicts.append({
                        "token": token, "quoted_token": celsius,
                        "canonical_k": kelvin,
                        "stored_k": sorted({row["stored_k"] for row in matched}),
                        "inspection_ids": sorted({
                            row["inspection_id"] for row in matched
                        }),
                        "region": anchor, "verdict": "verified",
                        "dimension": "temperature_k",
                    })
                    continue
                temp_block = next((
                    block for block in blocks
                    if _temperature_table(authoritative.get(block), value, token)
                ), None)
                if temp_block is not None:
                    flagged_anchors.add(anchor)
                    table = authoritative[temp_block]
                    violations.append({
                        "kind": "UNINSPECTED_VALUE", "anchor": anchor,
                        "block_number": temp_block,
                        "lit_id": cited.get(temp_block),
                        "literals": [token],
                        "detail": (
                            f"{celsius} corresponds to {kelvin:g} K in "
                            f"{temp_block}'s temperature_k column, but "
                            "that temperature was not shown in an "
                            "inspection of this run"
                        ),
                        "suggested_call": _temperature_call(
                            cited.get(temp_block), temp_block,
                            by_block.get(temp_block, []), table, kelvin,
                        ),
                    })
                    verdicts.append({
                        "token": token, "quoted_token": celsius,
                        "canonical_k": kelvin, "region": anchor,
                        "verdict": "uninspected_value",
                        "dimension": "temperature_k", "source": temp_block,
                    })
                    continue
                flagged_anchors.add(anchor)
                violations.append({
                    "kind": "UNGROUNDED_LITERAL", "anchor": anchor,
                    "block_number": blocks[0], "lit_id": anchors[0][0],
                    "literals": [token],
                    "detail": (
                        f"{celsius} corresponds to {kelvin:g} K, but no "
                        f"temperature_k cell for {anchor} was inspected"
                    ),
                    "suggested_call": "; ".join(
                        _temperature_call(
                            cited.get(block) or lit, block,
                            by_block.get(block, []),
                            authoritative.get(block), kelvin,
                        )
                        for lit, block in anchors
                    ),
                })
                verdicts.append({
                    "token": token, "quoted_token": celsius,
                    "canonical_k": kelvin, "region": anchor,
                    "verdict": "ungrounded", "dimension": "temperature_k",
                })
                continue
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
                axis, cell = _table_cell(table, value, token)
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
                        cited.get(auth_block), auth_block,
                        focus_value=cell, axis=axis,
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
                        column_ranges=(
                            authoritative.get(blocks[0]) or {}
                        ).get("column_ranges"),
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
                        column_ranges=(
                            authoritative.get(blocks[0]) or {}
                        ).get("column_ranges"),
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

    report = GateReport(
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
    if grounding_active:
        _enforce_source_scope(
            report, answer, regions, inspections, authoritative_fetch,
            fit_artifacts,
        )
    if doi_report is not None:
        report.ok = report.ok and doi_report.ok
        report.violations = doi_report.violations + report.violations
        report.verified = doi_report.verified + report.verified
        report.partially_grounded = (
            doi_report.partially_grounded + report.partially_grounded
        )
        report.evidence["doi_scope"] = doi_report.evidence["doi_scope"]
        report.evidence["cited_blocks"].update(
            doi_report.evidence["cited_blocks"]
        )
        report.evidence["literal_verdicts"] = (
            doi_report.evidence["literal_verdicts"]
            + report.evidence["literal_verdicts"]
        )
    return report



def _enforce_source_scope(
    report: GateReport,
    answer: str,
    regions: list[tuple[list[tuple[str | None, str]], str, bool]],
    inspections: list[dict],
    authoritative_fetch,
    fit_artifacts: list[dict],
) -> None:
    """Reject block-number matches from a different literature/DOI source.

    The historical block rail still supplies detailed repair hints. This
    additional default-ON check makes its verdict fail closed where local
    PROPblock numbers collide across sources; the legacy OFF rail is untouched.
    """
    tables: dict[tuple[str, str], dict | None] = {}
    doi_lits: dict[str, str | None] = {}
    bad: dict[str, set[str]] = {}
    scoped_citations: dict[str, dict] = {}

    def _entry_lit(entry: dict) -> str | None:
        if entry.get("lit_num_id"):
            return str(entry["lit_num_id"])
        doi = _normalize_doi(str(entry.get("doi") or ""))
        if not doi:
            return None
        if doi not in doi_lits:
            try:
                from ThermoML_raw_json_to_card_db_parsers.id_schema import (
                    lit_num_id_for_doi,
                )
                doi_lits[doi] = lit_num_id_for_doi(
                    doi, allow_unregistered=True,
                )
            except Exception:  # noqa: BLE001 - unknown DOI cannot prove identity
                doi_lits[doi] = None
        return doi_lits[doi]

    def _table(lit: str | None, block: str) -> dict | None:
        if not lit:
            return None
        key = (lit, block)
        if key not in tables:
            try:
                tables[key] = authoritative_fetch(lit, block)
            except Exception:  # noqa: BLE001 - same best-effort fetch policy as block rail
                tables[key] = None
        return tables[key]

    def _source_match(
        entry: dict, lit: str | None, block: str, doi: str,
    ) -> bool:
        if entry.get("block_number") != block:
            return False
        entry_lit = _entry_lit(entry)
        entry_doi = _normalize_doi(str(entry.get("doi") or ""))
        if lit and entry_lit and entry_lit != lit:
            return False
        if doi and entry_doi and entry_doi != doi:
            return False
        return bool(
            (lit and entry_lit == lit)
            or (doi and entry_doi == doi)
            or (not lit and not doi and not entry_lit and not entry_doi)
        )

    first_block = _BLOCK_ANCHOR_RE.search(answer)
    prefix_dois = _cited_dois(answer[:first_block.start()]) if first_block else []
    for region_index, (anchors, segment, _) in enumerate(regions):
        found = _literal_hits(segment)
        if not found:
            continue
        anchor = " + ".join(
            f"{lit}::{block}" if lit else block for lit, block in anchors
        )
        block_entries = [
            entry for entry in inspections
            if any(entry.get("block_number") == block for _, block in anchors)
        ]
        selected: list[dict] = []
        selected_tables: list[dict] = []
        rejected: list[tuple[float, str, str, str | None]] = []
        segment_dois = _cited_dois(segment)
        # A DOI can scope a single anchor. In a multi-source table it cannot
        # safely be assigned to every cited block.
        local_doi = ""
        if len(anchors) == 1:
            if len(segment_dois) == 1:
                local_doi = segment_dois[0]
            elif region_index == 0 and len(prefix_dois) == 1:
                local_doi = prefix_dois[0]

        for lit, block in anchors:
            candidates = [
                entry for entry in block_entries
                if entry.get("block_number") == block
            ]
            if not lit and local_doi:
                candidates = [
                    entry for entry in candidates
                    if _normalize_doi(str(entry.get("doi") or "")) == local_doi
                ]
            if not lit:
                known_lits = {
                    resolved for entry in candidates
                    if (resolved := _entry_lit(entry))
                }
                known_dois = {
                    _normalize_doi(str(entry["doi"])) for entry in candidates
                    if entry.get("doi")
                }
                if len(known_lits) > 1 or len(known_dois) > 1:
                    report.violations.append({
                        "kind": "AMBIGUOUS_BLOCK_SCOPE", "anchor": anchor,
                        "block_number": block, "lit_id": None,
                        "literals": [token for _, token, _, _ in found],
                        "detail": (
                            f"{block} identifies multiple inspected literature "
                            "sources; qualify it with GLOBlit_N::"
                            f"{block} before quoting values"
                        ),
                        "suggested_call": (
                            "qualify the block citation with its literature ID"
                        ),
                    })
                    bad.setdefault(anchor, set()).update(
                        token for _, token, _, _ in found
                    )
                    rejected = found
                    break
                lit = next(iter(known_lits), None)
                if not lit and not known_dois:
                    # The original rail already handles a truly uninspected
                    # bare block. No source can be assigned here.
                    continue
                if not lit:
                    local_doi = next(iter(known_dois))
            table = _table(lit, block)
            table_doi = _normalize_doi(str((table or {}).get("doi") or ""))
            if local_doi and table_doi and local_doi != table_doi:
                report.violations.append({
                    "kind": "SOURCE_DOI_CONFLICT", "anchor": anchor,
                    "block_number": block, "lit_id": lit,
                    "literals": [token for _, token, _, _ in found],
                    "detail": (
                        f"{anchor} cites DOI {local_doi}, but {lit}::{block} "
                        f"belongs to DOI {table_doi}"
                    ),
                    "suggested_call": "correct the DOI or block citation",
                })
                bad.setdefault(anchor, set()).update(
                    token for _, token, _, _ in found
                )
                rejected = found
                break
            doi = local_doi or table_doi
            matches = [
                entry for entry in candidates
                if _source_match(entry, lit, block, doi)
            ]
            selected.extend(matches)
            if table:
                selected_tables.append(table)
            source_anchor = f"{lit}::{block}" if lit else block
            scoped_citations[source_anchor] = {
                "lit_num_id": lit,
                "inspected": bool(matches),
                "inspection_ids": [
                    entry.get("inspection_id") for entry in matches
                ],
                "authoritative": (
                    {
                        "doi": table.get("doi"),
                        "n_rows": table.get("n_rows"),
                        "columns": table.get("columns"),
                        "column_ranges": table.get("column_ranges"),
                    } if table else None
                ),
            }
        if rejected:
            continue

        for value, token, _, celsius in found:
            if not celsius and any(
                any(_near_tight(value, token, v) for v in fit["values"])
                for fit in fit_artifacts
            ):
                continue
            matched = (
                bool(_temperature_rows(selected, value, token))
                if celsius else
                any(
                    _near(value, token, cell)
                    for entry in selected
                    for cell in entry.get("grounded_values", ())
                )
            )
            if matched:
                continue
            # The legacy rail may already flag this literal. Only add a
            # source-scope violation if it incorrectly accepted the value.
            old_verdict = next((
                item.get("verdict")
                for item in report.evidence.get("literal_verdicts", ())
                if item.get("region") == anchor and item.get("token") == token
            ), None)
            if old_verdict not in {"verified", "fit_artifact"}:
                continue
            in_table = any(
                _temperature_table(table, value, token) if celsius else
                any(_near_tight(value, token, v)
                    for v in table.get("values", ()))
                for table in selected_tables
            )
            other_matches = {
                (entry.get("lit_num_id"), _normalize_doi(
                    str(entry.get("doi") or "")
                ), entry.get("block_number"))
                for entry in inspections
                if entry not in selected and (
                    bool(_temperature_rows([entry], value, token))
                    if celsius else any(
                        _near_tight(value, token, cell)
                        for cell in entry.get("grounded_values", ())
                    )
                )
            }
            kind = (
                "UNINSPECTED_BLOCK" if not selected else
                "UNINSPECTED_VALUE" if in_table else
                "MISATTRIBUTED_VALUE" if len(other_matches) == 1 else
                "AMBIGUOUS_VALUE" if other_matches else
                "UNGROUNDED_LITERAL"
            )
            report.violations.append({
                "kind": kind, "anchor": anchor,
                "block_number": anchors[0][1],
                "lit_id": anchors[0][0], "literals": [token],
                "detail": (
                    f"{token} is not grounded by an inspection of {anchor}; "
                    "a matching value from another literature or DOI source "
                    "cannot verify this citation"
                ),
                "suggested_call": "; ".join(
                    _suggest_call(lit, block) for lit, block in anchors
                ),
            })
            bad.setdefault(anchor, set()).add(token)

    report.evidence["cited_blocks"].update(scoped_citations)
    if not bad:
        return
    report.ok = False
    for entry in list(report.verified):
        if entry["anchor"] not in bad:
            continue
        report.verified.remove(entry)
        remaining = [v for v in entry["values"]
                     if v not in bad[entry["anchor"]]]
        if remaining:
            report.partially_grounded.append({
                "anchor": entry["anchor"], "values": remaining,
            })
    for entry in report.partially_grounded:
        if entry["anchor"] in bad:
            entry["values"] = [
                v for v in entry["values"] if v not in bad[entry["anchor"]]
            ]
    report.partially_grounded[:] = [
        entry for entry in report.partially_grounded if entry["values"]
    ]
    for item in report.evidence.get("literal_verdicts", ()):
        if item.get("token") in bad.get(item.get("region"), set()):
            item["verdict"] = "source_scope_rejected"


def _run_doi_grounding_gate(
    answer: str,
    inspections: list[dict],
    authoritative_fetch,
    fit_artifacts: list[dict],
) -> GateReport:
    """Check a DOI-only answer against this run's DOI-matched inspections.

    DOI identity names a *candidate set*, not a block. Even when one value
    matches one candidate, the evidence records the entire scope and never
    silently assigns the prose to that block.
    """
    dois = _cited_dois(answer)
    found = _literal_hits(answer)
    inspected_blocks = sorted({e["block_number"] for e in inspections})
    scoped: dict[tuple[str | None, str], list[dict]] = {}
    if len(dois) == 1:
        for entry in inspections:
            if _normalize_doi(str(entry.get("doi") or "")) == dois[0]:
                key = (entry.get("lit_num_id"), entry["block_number"])
                scoped.setdefault(key, []).append(entry)

    def _label(key: tuple[str | None, str]) -> str:
        lit, block = key
        return f"{lit}::{block}" if lit else block

    keys = sorted(scoped, key=_label)
    labels = [_label(key) for key in keys]
    anchor = (
        f"DOI {dois[0]} [candidate set: {', '.join(labels)}]"
        if len(dois) == 1 and labels else
        "DOI citation scope"
    )
    evidence = {
        "doi_scope": {
            "cited_dois": dois,
            "mode": "inherited_inspection_candidate_set",
            "candidate_blocks": labels,
        },
        "cited_blocks": {
            _label(key): {
                "lit_num_id": key[0],
                "inspected": True,
                "inspection_ids": [e["inspection_id"] for e in scoped[key]],
            }
            for key in keys
        },
        "literal_verdicts": [],
    }
    if not found:
        return GateReport(ok=True, inspected_blocks=inspected_blocks,
                          evidence=evidence)
    if len(dois) != 1 or not keys:
        many = len(dois) != 1
        kind = "AMBIGUOUS_DOI_SCOPE" if many else "UNINSPECTED_DOI_SCOPE"
        detail = (
            "Several DOIs cite numeric claims without exact block anchors; "
            "the gate cannot assign those claims to one inspected source"
            if many else
            f"DOI {dois[0]} has no matching block inspection in this run"
        )
        violation = {
            "kind": kind, "anchor": anchor, "block_number": None,
            "lit_id": None, "literals": [token for _, token, _, _ in found],
            "detail": detail,
            "suggested_call": (
                "Cite the exact inspected GLOBlit_N::PROPblock_N (or "
                "RXNblock_N) beside each claim, and inspect any block "
                "whose rows have not yet been shown in this run"
            ),
        }
        evidence["literal_verdicts"] = [
            {"token": token, "region": anchor, "verdict": kind.lower()}
            for _, token, _, _ in found
        ]
        return GateReport(ok=False, violations=[violation],
                          inspected_blocks=inspected_blocks, evidence=evidence)

    authoritative: dict[tuple[str | None, str], dict | None] = {}
    for key in keys:
        try:
            authoritative[key] = (
                authoritative_fetch(*key) if key[0] else None
            )
        except Exception:  # noqa: BLE001 — mirror the best-effort fetch rail
            authoritative[key] = None
        table = authoritative[key]
        evidence["cited_blocks"][_label(key)]["authoritative"] = (
            {
                "doi": table.get("doi"), "n_rows": table.get("n_rows"),
                "columns": table.get("columns"),
                "column_ranges": table.get("column_ranges"),
            } if table else None
        )

    violations: list[dict] = []
    verified: list[str] = []
    verdicts = evidence["literal_verdicts"]
    for value, token, context, celsius in found:
        temperature_matches = (
            {key: _temperature_rows(scoped[key], value, token) for key in keys}
            if celsius else {}
        )
        matched = [
            key for key in keys
            if (
                bool(temperature_matches[key]) if celsius else
                any(
                    _near(value, token, cell)
                    for entry in scoped[key]
                    for cell in entry["grounded_values"]
                )
            )
        ]
        if matched:
            if token not in verified:
                verified.append(token)
            verdict = {
                "token": token, "region": anchor, "verdict": "verified",
                "matching_candidates": [_label(key) for key in matched],
            }
            if celsius:
                rows = [row for key in matched
                        for row in temperature_matches[key]]
                verdict.update({
                    "quoted_token": celsius, "canonical_k": value + 273.15,
                    "stored_k": sorted({row["stored_k"] for row in rows}),
                    "inspection_ids": sorted({
                        row["inspection_id"] for row in rows
                    }),
                    "dimension": "temperature_k",
                })
            verdicts.append(verdict)
            continue
        fit = next((
            item["fit_id"] for item in fit_artifacts
            if not celsius and any(
                _near_tight(value, token, cell) for cell in item["values"]
            )
        ), None)
        if fit:
            if token not in verified:
                verified.append(token)
            verdicts.append({
                "token": token, "region": anchor,
                "verdict": "fit_artifact", "source": fit,
            })
            continue
        uninspected = [
            key for key in keys
            if (
                _temperature_table(authoritative[key], value, token)
                if celsius else
                authoritative[key] and any(
                    _near_tight(value, token, cell)
                    for cell in authoritative[key]["values"]
                )
            )
        ]
        if uninspected:
            calls = []
            for key in uninspected:
                table = authoritative[key] or {}
                if celsius:
                    calls.append(_temperature_call(
                        *key, scoped[key], table, value + 273.15,
                    ))
                else:
                    axis, cell = _table_cell(table, value, token)
                    calls.append(_suggest_call(
                        *key, focus_value=cell, axis=axis,
                        entries=scoped[key],
                        column_ranges=table.get("column_ranges"),
                    ))
            kind = "UNINSPECTED_VALUE"
            detail = (
                f"{celsius or token} exists in the DOI candidate "
                f"{'temperature_k column(s) of' if celsius else 'table(s)'} "
                f"{', '.join(_label(key) for key in uninspected)} but was "
                "not shown in an inspection of this run"
            )
            suggested_call = "; ".join(calls)
            matches = [_label(key) for key in uninspected]
        else:
            foreign = sorted({
                (entry.get("lit_num_id"), entry["block_number"])
                for entry in inspections
                if (entry.get("lit_num_id"), entry["block_number"]) not in scoped
                and (
                    bool(_temperature_rows([entry], value, token))
                    if celsius else
                    any(_near_tight(value, token, cell)
                        for cell in entry["grounded_values"])
                )
            }, key=_label)
            if len(foreign) == 1:
                kind = "MISATTRIBUTED_VALUE"
                detail = (
                    f"{token} does not match the DOI candidate set; it "
                    f"matches inspected {_label(foreign[0])}"
                )
                suggested_call = (
                    f"re-anchor {token} to {_label(foreign[0])} — "
                    "no new tool call needed"
                )
            elif foreign:
                kind = "AMBIGUOUS_VALUE"
                detail = (
                    f"{token} matches {len(foreign)} other inspected "
                    "sources; cite the intended inspected block explicitly"
                )
                suggested_call = (
                    "Cite the exact inspected block beside this claim"
                )
            else:
                kind = "UNGROUNDED_LITERAL"
                detail = (
                    f"{celsius} corresponds to {value + 273.15:g} K but "
                    "matches no inspected temperature_k cell or DOI "
                    "candidate temperature_k column"
                    if celsius else
                    f"{token} (near '…{context[:60]}…') matches no "
                    "inspected row/stat or DOI candidate table"
                )
                suggested_call = "; ".join(
                    _temperature_call(
                        *key, scoped[key], authoritative[key],
                        value + 273.15,
                    ) if celsius else
                    _suggest_call(*key, focus_value=value,
                                  entries=scoped[key],
                                  column_ranges=(
                                      authoritative[key] or {}
                                  ).get("column_ranges"))
                    for key in keys
                )
            matches = [_label(key) for key in foreign]
        violations.append({
            "kind": kind, "anchor": anchor,
            "block_number": uninspected[0][1] if len(uninspected) == 1 else None,
            "lit_id": uninspected[0][0] if len(uninspected) == 1 else None,
            "literals": [token], "detail": detail,
            "suggested_call": suggested_call,
            "candidate_blocks": labels,
        })
        verdict = {
            "token": token, "region": anchor,
            "verdict": kind.lower(), "matching_candidates": matches,
        }
        if celsius:
            verdict.update({
                "quoted_token": celsius, "canonical_k": value + 273.15,
                "dimension": "temperature_k",
            })
        verdicts.append(verdict)

    grounded = {"anchor": anchor, "values": verified}
    return GateReport(
        ok=not violations, violations=violations,
        inspected_blocks=inspected_blocks,
        verified=[grounded] if verified and not violations else [],
        partially_grounded=[grounded] if verified and violations else [],
        evidence=evidence,
    )


def _containing_column(ranges: dict, value: float) -> str | None:
    """The ONE data column whose (min, max) range holds *value*.

    None when no column or several columns hold it: ranges overlap (a 0-1
    mole fraction holds most small numbers), so several holders name no
    safe axis.
    """
    holders = [
        column for column, bounds in ranges.items()
        if column != "BLKpoint_id" and bounds
        and bounds[0] is not None and bounds[1] is not None
        and bounds[0] <= value <= bounds[1]
    ]
    return holders[0] if len(holders) == 1 else None


def _shown_column_ranges(entries: list[dict]) -> dict[str, tuple[float, float]]:
    """Per-column (min, max) over the rows this run's inspections showed."""
    ranges: dict[str, tuple[float, float]] = {}
    for entry in entries:
        for row in entry.get("rows_shown") or []:
            if not isinstance(row, dict):
                continue
            for column, cell in row.items():
                try:
                    value = float(cell)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                low, high = ranges.get(column, (value, value))
                ranges[column] = (min(low, value), max(high, value))
    return ranges


def _suggest_call(
    lit: str | None,
    block: str,
    *,
    focus_value: float | None = None,
    axis: str | None = None,
    entries: list[dict] | None = None,
    column_ranges: dict | None = None,
) -> str:
    if lit:
        target = f'block_number="{lit}::{block}"'
    else:
        target = f'block_number="{block}", literature="<GLOBlit_N or DOI>"'
    if focus_value is not None and axis is None and (entries or column_ranges):
        # Without a known column, `nearest` only on the ONE column whose
        # range holds the value (the authoritative ranges, else the rows
        # shown this run). Otherwise the plain call names the block alone:
        # a guessed axis would send the model to, e.g., a mole-fraction
        # column for "25" meaning 25 °C.
        ranges = column_ranges or _shown_column_ranges(entries or [])
        axis = _containing_column(ranges, focus_value)
    if focus_value is not None and axis is not None:
        return (
            f"inspect_block_table({target}, nearest={{\"column\": "
            f"\"{axis}\", \"value\": {focus_value}}})"
        )
    return f"inspect_block_table({target})"


@profile.historical_dispatch("grounding_gate")
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
        if violation["kind"] in {
            "AMBIGUOUS_DOI_SCOPE", "UNINSPECTED_DOI_SCOPE",
        }:
            fix = f"  fix (required — source scope is unresolved): {violation['suggested_call']}"
        elif prefix:
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


@profile.historical_dispatch("grounding_gate")
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


@profile.historical_dispatch("grounding_gate")
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
        "conclusions formed before inspection.\n"
        "When the repairs are done, emit your COMPLETE corrected final "
        "answer. It replaces the previous answer entirely: a reply that only "
        "lists the changes, or says 'no changes' or 'same as above', ships "
        "as the whole answer. Do not narrate the repair inside the answer — "
        "a value repeated in such a note is checked again."
    )


@profile.historical_dispatch("grounding_gate")
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


@profile.historical_dispatch("grounding_gate")
def build_flags_note(report: GateReport) -> str:
    """Deterministic disclaimer appended when bounces are exhausted."""
    return (
        f"\n\n---\n{FLAGS_HEADER} The following quoted values could not be "
        "verified against any verbatim data inspection in this run and must "
        "be treated as UNVERIFIED:\n" + report.report_text
    )


@profile.historical_dispatch("grounding_gate")
def count_gate_bounces(memory: list[dict]) -> int:
    """How many grounding bounces already happened in this conversation."""
    return sum(
        1 for message in memory
        if message.get("role") == "user"
        and GATE_MARKER in str(message.get("content", ""))
    )


def bounced_answer(memory: list[dict]) -> str | None:
    """The assistant answer the newest grounding bounce rejected, if any."""
    for index in range(len(memory) - 1, 0, -1):
        message = memory[index]
        if (
            message.get("role") == "user"
            and GATE_MARKER in str(message.get("content", ""))
        ):
            previous = memory[index - 1]
            if previous.get("role") == "assistant":
                return str(previous.get("content", ""))
            return None
    return None


def build_continuity_nudge(
    memory: list[dict],
    answer: str,
    *,
    clean: Callable[[str], str] | None = None,
) -> str | None:
    """One-time bounce when the answer after a grounding bounce is a stub.

    The gate asks for a surgical repair, but the next final answer still
    REPLACES the bounced one, so a stub such as "No changes." would ship in
    place of the repaired answer. *answer* is the cleaned final answer;
    *clean* cleans the raw bounced message the same way. Returns the nudge,
    or None when no bounce happened, the bounced answer was short, the new
    answer kept its substance, or the nudge was already sent once.
    """
    if profile.is_legacy_mode():
        return None
    if any(
        message.get("role") == "user"
        and CONTINUITY_MARKER in str(message.get("content", ""))
        for message in memory
    ):
        return None
    previous = bounced_answer(memory)
    if previous is None:
        return None
    bounced_chars = len((clean or str.strip)(previous))
    answer_chars = len(answer.strip())
    if (
        bounced_chars < _STUB_MIN_BOUNCED_CHARS
        or answer_chars >= _STUB_SHARE * bounced_chars
    ):
        return None
    return (
        f"{CONTINUITY_MARKER} Your final answer has {answer_chars} "
        f"characters; the answer the data-grounding gate bounced had "
        f"{bounced_chars}. The final answer REPLACES the bounced one — it is "
        "not a patch on top of it, and the bounced text is discarded. "
        "Re-emit your COMPLETE corrected final answer now: keep every "
        "verified value and all unflagged prose, apply the repairs from the "
        "inspections you ran, and do not narrate the repair. If the short "
        "answer really is your complete answer, submit it again unchanged."
    )
