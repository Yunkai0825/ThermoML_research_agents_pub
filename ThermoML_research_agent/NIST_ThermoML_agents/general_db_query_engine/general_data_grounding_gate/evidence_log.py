"""Passive grounding-evidence log — JSON + Markdown per run.

Rebuilds the final gate report from ``(answer, tool_history)`` (the gate is
pure/deterministic) and writes a user-reviewable evidence bundle next to the
run's other diagnostics:

- ``grounding_evidence.json`` — cited blocks, per-literal verdicts,
  violations, and every verbatim inspection (markdown included);
- ``grounding_evidence.md``   — the human-readable rendering.

Never enters any prompt; purely an audit artifact.
"""

from __future__ import annotations

from . import profile

import datetime as dt
import json
import logging
from pathlib import Path

from .grounding_gate import (
    FLAGS_HEADER,
    GateReport,
    gate_switched_off,
    run_data_grounding_gate,
)
from .inspection_harvest import harvest_inspections

log = logging.getLogger("grounding-evidence")


def build_evidence_log(
    report: GateReport, inspections: list[dict],
) -> tuple[dict, str]:
    """(json_payload, markdown_text) for one final gate state."""
    payload = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "gate_ok": report.ok,
        "grounding_profile": profile.profile_metadata(),
        "violations": report.violations,
        "verified": report.verified,
        "partially_grounded": report.partially_grounded,
        "evidence": report.evidence,
        "inspections": [
            {
                key: value for key, value in entry.items()
                if key != "grounded_values"          # set — not JSON-safe
            }
            for entry in inspections
        ],
    }

    lines = [
        "# Data grounding evidence",
        "",
        f"- generated: {payload['generated']}",
        f"- grounding profile: {payload['grounding_profile']['name']}",
        f"- gate verdict: {'OK' if report.ok else f'{len(report.violations)} violation(s)'}",
        "",
        "## Cited blocks",
        "",
        "| block | lit_num_id | inspected | inspections | db rows |",
        "|---|---|---|---|---|",
    ]
    doi_scope = report.evidence.get("doi_scope")
    if doi_scope:
        lines[5:5] = [
            "## DOI citation scope",
            "",
            "- cited DOIs: " + ", ".join(doi_scope["cited_dois"]),
            "- inherited inspection candidates: " + (
                ", ".join(doi_scope["candidate_blocks"]) or "none"
            ),
            (
                "- each verified literal was checked against this candidate set; "
                "the DOI alone does not identify one block"
            ),
            "",
        ]
    cited = report.evidence.get("cited_blocks", {})
    for block, info in sorted(cited.items()):
        auth = info.get("authoritative") or {}
        lines.append(
            f"| {block} | {info.get('lit_num_id') or '—'} "
            f"| {'yes' if info.get('inspected') else 'NO'} "
            f"| {', '.join(info.get('inspection_ids', [])) or '—'} "
            f"| {auth.get('n_rows', '—')} |"
        )
    if not cited:
        lines.append("| — | — | — | — | — |")

    fits = report.evidence.get("fit_artifacts", [])
    if fits:
        lines += [
            "",
            "## Fit artifacts (grounded rail)",
            "",
            "| fit_id | heading | blocks | values pooled |",
            "|---|---|---|---|",
        ]
        for fit in fits:
            lines.append(
                f"| {fit['fit_id']} | {fit['heading']} "
                f"| {', '.join(fit['blocks']) or '—'} | {fit['n_values']} |"
            )

    lines += [
        "",
        "## Literal verdicts",
        "",
        "| token | region | verdict | source |",
        "|---|---|---|---|",
    ]
    for verdict in report.evidence.get("literal_verdicts", []):
        source = verdict.get("source") or (
            f"{verdict['source_count']} sources"
            if verdict.get("source_count") else "—"
        )
        lines.append(
            f"| {verdict['token']} | {verdict['region']} "
            f"| {verdict['verdict']} | {source} |"
        )

    if report.violations:
        lines += ["", "## Violations", ""]
        for violation in report.violations:
            lines.append(
                f"- **{violation['kind']}** {violation['anchor']}: "
                f"{violation['detail']}"
            )

    lines += ["", "## Inspections (verbatim)", ""]
    if not inspections:
        lines.append("_none in this run_")
    for entry in inspections:
        anchor = (
            f"{entry.get('lit_num_id') or entry['doi']}::{entry['block_number']}"
        )
        lines += [
            f"### {entry['inspection_id']} — {anchor} "
            f"({entry['table_mode']}, {len(entry['rows_shown'])} rows shown)",
            "",
        ]
        markdown = entry.get("markdown", "")
        lines.append(markdown if markdown else "_structured envelope entry (no markdown)_")
        lines.append("")

    return payload, "\n".join(lines)


def write_grounding_evidence(
    out_dir: str | Path,
    answer: str,
    tool_history: list[dict] | None,
    *,
    authoritative_fetch=None,
) -> Path | None:
    """Write the evidence bundle; returns the .md path or None when skipped."""
    inspections = harvest_inspections(tool_history)
    # The active gate checked the authored answer before machine diagnostics
    # were appended. Re-scanning their quoted values/citations would count the
    # same claims again. Only exact append boundaries are excluded; OFF retains
    # the historical full-answer rescan.
    audited_answer = answer
    if not profile.is_legacy_mode() and not gate_switched_off():
        for machine_tail in (
            f"\n\n---\n{FLAGS_HEADER}",
            "\n\n[POSTANSWER DATA FLAGS]",
        ):
            audited_answer = audited_answer.partition(machine_tail)[0]
    report = run_data_grounding_gate(
        audited_answer, tool_history,
        inspections=inspections,
        authoritative_fetch=authoritative_fetch,
    )
    if (not inspections and not report.evidence.get("cited_blocks")
            and not report.evidence.get("doi_scope")):
        return None                       # nothing data-grounded in this run
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload, markdown = build_evidence_log(report, inspections)
    (out / "grounding_evidence.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    md_path = out / "grounding_evidence.md"
    md_path.write_text(markdown, encoding="utf-8")
    log.info("grounding evidence written: %s", md_path)
    return md_path
