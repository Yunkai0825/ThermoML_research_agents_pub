"""Mandatory data-grounding gate + verbatim inspection registry (v4 workflow).

Public surface for the react loop, post-answer assembly, and dispatchers.
"""

from .inspection_harvest import (
    INSPECT_TOOL_NAME,
    entry_from_structured,
    export_inspections,
    harvest_fit_artifacts,
    harvest_inspection_targets,
    harvest_inspections,
    parse_inspection_markdown,
)
from .authoritative_blocks import default_authoritative_fetch
from .postanswer_claims import (
    check_postanswer_claims,
    core_claims_source_suffix,
    postanswer_checker_for_tools,
)
from .evidence_log import build_evidence_log, write_grounding_evidence
from .grounding_gate import (
    CONTINUITY_MARKER,
    FLAGS_HEADER,
    GATE_ENV_VAR,
    GATE_MARKER,
    LEDGER_TOOLS,
    MAX_GATE_BOUNCES,
    GateReport,
    bounced_answer,
    build_continuity_nudge,
    build_flags_note,
    build_gate_nudge,
    build_grounded_context,
    build_timeout_gate_instruction,
    build_violation_report,
    count_gate_bounces,
    gate_enabled,
    gate_switched_off,
    run_data_grounding_gate,
    without_ledger_tools,
)

__all__ = [
    "INSPECT_TOOL_NAME",
    "entry_from_structured",
    "export_inspections",
    "harvest_fit_artifacts",
    "harvest_inspection_targets",
    "harvest_inspections",
    "parse_inspection_markdown",
    "default_authoritative_fetch",
    "check_postanswer_claims",
    "core_claims_source_suffix",
    "postanswer_checker_for_tools",
    "build_evidence_log",
    "write_grounding_evidence",
    "CONTINUITY_MARKER",
    "FLAGS_HEADER",
    "GATE_ENV_VAR",
    "GATE_MARKER",
    "LEDGER_TOOLS",
    "MAX_GATE_BOUNCES",
    "GateReport",
    "bounced_answer",
    "build_continuity_nudge",
    "build_flags_note",
    "build_gate_nudge",
    "build_grounded_context",
    "build_timeout_gate_instruction",
    "build_violation_report",
    "count_gate_bounces",
    "gate_enabled",
    "gate_switched_off",
    "run_data_grounding_gate",
    "without_ledger_tools",
]
