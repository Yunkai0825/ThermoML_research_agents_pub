"""Workflow-skill markdown parser.

Reads a workflow ``.md`` file that follows the schema defined in
``_workflow_parsing_schema.md`` and returns a structured dict used by the
orchestrator to assemble agent prompts.

Public API
----------
parse_workflow(path) -> dict
    Parse a workflow markdown file and return structured sections.

render_prompt(parsed, *, purpose, instruction, context="", id_catalog="") -> str
    Substitute template variables and return the final prompt string.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)

_TOOL_LINE_RE = re.compile(
    r"^-\s+(?P<sig>[^—\n]+?)(?:\s+—\s+(?P<desc>.+))?$",
    re.MULTILINE,
)

_PHASE_HEADER_RE = re.compile(
    r"^##\s+phase_(?P<num>\d+):\s*(?P<name>.+)$",
    re.MULTILINE,
)

_TOOLS_LINE_RE = re.compile(
    r"^tools:\s*\[(?P<groups>[^\]]*)\]\s*$",
    re.MULTILINE,
)

_PARALLEL_RE = re.compile(
    r"^parallel_dispatch:\s*\{(?P<body>[^}]*)\}\s*$",
    re.MULTILINE,
)

_JSON_FENCE_RE = re.compile(
    r"```json\s*\n(?P<body>.*?)```",
    re.DOTALL,
)

_SYSTEM_PROMPT_TAG_RE = re.compile(
    r"<system_prompt>\s*\n?(.*?)\n?\s*</system_prompt>",
    re.DOTALL,
)

# A workflow marks its grounding-ledger instructions with two comment lines
# (<!-- grounding-ledger:begin --> ... <!-- grounding-ledger:end -->). The
# marker lines never reach a prompt; the lines between them stay unless the
# grounding gate is switched off (THERMOML_GROUNDING_GATE=off = no gate AND no
# ledger). An optional <!-- grounding-ledger:else --> line inside a span
# gives the lines that stand in the span's place when the gate is off.
_LEDGER_SPAN_RE = re.compile(
    r"^[ \t]*<!--[ \t]*grounding-ledger:begin[ \t]*-->[ \t]*\n(?P<body>.*?)"
    r"(?:^[ \t]*<!--[ \t]*grounding-ledger:else[ \t]*-->[ \t]*\n(?P<alt>.*?))?"
    r"^[ \t]*<!--[ \t]*grounding-ledger:end[ \t]*-->[ \t]*(?:\n|\Z)",
    re.DOTALL | re.MULTILINE,
)

_RESPONSE_JSON_SCHEMA_HEADING = "Response JSON Schema"
_RESPONSE_JSON_SCHEMA_SECTION_RE = re.compile(
    r"^#\s+Response JSON Schema[ \t]*\n"
    r".*?(?=^#\s+[^\n]+\n|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _apply_ledger_spans(text: str) -> str:
    """Drop the marker lines; when the gate is off, drop the marked spans too
    (a span's else part, if any, takes its place)."""
    from ..general_data_grounding_gate.grounding_gate import gate_switched_off

    keep = not gate_switched_off()
    return _LEDGER_SPAN_RE.sub(
        lambda m: m.group("body") if keep else (m.group("alt") or ""), text
    )


def _extract_front_matter(text: str) -> tuple[dict, str]:
    """Return (front_matter_dict, remaining_text)."""
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Missing YAML front-matter (need two --- delimiters)")
    fm = yaml.safe_load(parts[1])
    if not isinstance(fm, dict):
        raise ValueError("Front-matter must be a YAML mapping")
    for key in ("agent_id", "layer", "parent"):
        if key not in fm:
            raise ValueError(f"Front-matter missing required key: {key}")
    body = parts[2]
    return fm, body


def _split_h1_sections(body: str) -> dict[str, str]:
    """Split body into {heading: content} by H1 boundaries."""
    positions = [(m.start(), m.group(1).strip()) for m in _H1_RE.finditer(body)]
    sections: dict[str, str] = {}
    for i, (start, heading) in enumerate(positions):
        # content runs from end of this heading line to start of next section
        line_end = body.index("\n", start) + 1 if "\n" in body[start:] else len(body)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(body)
        sections[heading] = body[line_end:end].strip()
    return sections


def _parse_tools(section_text: str) -> dict[str, list[dict[str, str]]]:
    """Parse ## group headers and tool bullet lines."""
    groups: dict[str, list[dict[str, str]]] = {}
    current_group: str | None = None
    for line in section_text.splitlines():
        h2 = re.match(r"^##\s+(.+)$", line)
        if h2:
            current_group = h2.group(1).strip()
            groups[current_group] = []
            continue
        if current_group is None:
            continue
        m = _TOOL_LINE_RE.match(line)
        if m:
            sig = m.group("sig").strip()
            desc = (m.group("desc") or "").strip()
            # extract bare name from signature
            name = sig.split("(")[0].strip() if "(" in sig else sig
            groups[current_group].append(
                {"name": name, "signature": sig, "description": desc}
            )
    return groups


def _parse_phases(section_text: str) -> list[dict[str, Any]]:
    """Parse phase sub-sections."""
    phases: list[dict[str, Any]] = []
    headers = list(_PHASE_HEADER_RE.finditer(section_text))
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(section_text)
        block = section_text[start:end].strip()

        # tools: [group1, group2]
        tools_m = _TOOLS_LINE_RE.search(block)
        tool_groups = (
            [g.strip() for g in tools_m.group("groups").split(",") if g.strip()]
            if tools_m
            else []
        )

        # parallel_dispatch: {tool_group: ..., fan_out_over: ..., max_concurrent: N}
        parallel = None
        par_m = _PARALLEL_RE.search(block)
        if par_m:
            parallel = {}
            for kv in par_m.group("body").split(","):
                kv = kv.strip()
                if ":" not in kv:
                    raise ValueError(
                        f"Malformed parallel_dispatch entry in phase_{m.group('num')}: {kv!r}"
                    )
                k, v = kv.split(":", 1)
                k, v = k.strip(), v.strip()
                if k == "max_concurrent":
                    if not re.fullmatch(r"[1-9][0-9]*", v):
                        raise ValueError(
                            "parallel_dispatch.max_concurrent must be a positive integer literal"
                        )
                    v = int(v)
                parallel[k] = v
            required_parallel = {"tool_group", "fan_out_over", "max_concurrent"}
            if set(parallel) != required_parallel:
                raise ValueError(
                    "parallel_dispatch must contain exactly tool_group, fan_out_over, "
                    f"and max_concurrent; received {sorted(parallel)}"
                )

        # guidance: everything after "guidance:" line
        guidance = ""
        gm = re.search(r"^guidance:\s*\|?\s*$", block, re.MULTILINE)
        if gm:
            guidance = block[gm.end():].strip()
        elif re.search(r"^guidance:\s*(.+)$", block, re.MULTILINE):
            guidance = re.search(r"^guidance:\s*(.+)$", block, re.MULTILINE).group(1).strip()

        phases.append(
            {
                "number": int(m.group("num")),
                "name": m.group("name").strip(),
                "tools": tool_groups,
                "parallel_dispatch": parallel,
                "guidance": guidance,
            }
        )
    return phases


def _parse_response_json_schema(section_text: str) -> dict[str, Any]:
    """Extract the one required fenced JSON object for the response."""
    matches = list(_JSON_FENCE_RE.finditer(section_text))
    if len(matches) != 1:
        raise ValueError(
            "Response JSON Schema must contain exactly one fenced JSON object"
        )
    parsed = json.loads(matches[0].group("body"))
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("Response JSON Schema JSON must be a non-empty object")
    missing = {"answer", "core_claims"} - set(parsed)
    if missing:
        raise ValueError(
            "Response JSON Schema is missing required fields: "
            + ", ".join(sorted(missing))
        )
    if not isinstance(parsed["answer"], str):
        raise ValueError(
            "Response JSON Schema field 'answer' must be a string example"
        )
    if (
        not isinstance(parsed["core_claims"], list)
        or len(parsed["core_claims"]) != 1
        or not isinstance(parsed["core_claims"][0], str)
    ):
        raise ValueError(
            "Response JSON Schema field 'core_claims' must be a one-element "
            "string array example"
        )
    return parsed


def _extract_response_schema_and_redact_prompt(
    *,
    body: str,
    system_prompt: str,
    workflow_name: str,
) -> tuple[dict[str, Any], str]:
    """Return the machine schema and a system prompt from which it is absent.

    The response schema is deliberately stored in the workflow/skill Markdown
    but travels on a parser-only channel.  The chemistry answer agent receives
    the redacted prompt, while the parallel post-answer workers receive the
    parsed schema object.
    """
    heading_matches = [
        match
        for match in _H1_RE.finditer(body)
        if match.group(1).strip() == _RESPONSE_JSON_SCHEMA_HEADING
    ]
    if len(heading_matches) != 1:
        raise ValueError(
            f"Workflow {workflow_name} must contain exactly one "
            f"'# {_RESPONSE_JSON_SCHEMA_HEADING}' section; "
            f"found {len(heading_matches)}"
        )

    sections = _split_h1_sections(body)
    response_schema = _parse_response_json_schema(
        sections[_RESPONSE_JSON_SCHEMA_HEADING]
    )
    redacted_prompt, substitutions = _RESPONSE_JSON_SCHEMA_SECTION_RE.subn(
        "",
        system_prompt,
    )
    if substitutions != 1:
        raise ValueError(
            f"Workflow {workflow_name} must place '# "
            f"{_RESPONSE_JSON_SCHEMA_HEADING}' inside <system_prompt> so the "
            "parser can create the separate schema and answer-prompt channels"
        )
    redacted_prompt = redacted_prompt.strip()
    if not redacted_prompt:
        raise ValueError(
            f"Workflow {workflow_name} has an empty answer-agent system prompt "
            "after response-schema redaction"
        )
    if _RESPONSE_JSON_SCHEMA_HEADING in redacted_prompt:
        raise ValueError(
            f"Workflow {workflow_name} leaked response-schema content into "
            "the answer-agent system prompt"
        )
    return response_schema, redacted_prompt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_workflow(path: str | Path) -> dict[str, Any]:
    """Parse a workflow markdown file into a structured dict.

    Parameters
    ----------
    path : str or Path
        Path to the ``.md`` workflow skill file.

    Returns
    -------
    dict with keys: front_matter, system_prompt, tools, prompt_template,
    response_json_schema, phases.

    ``system_prompt`` is the schema-redacted prompt for the plain-text
    chemistry answer agent. ``response_json_schema`` is the separate
    machine-facing schema consumed only after that agent terminates.
    """
    text = _apply_ledger_spans(Path(path).read_text(encoding="utf-8"))
    front_matter, body = _extract_front_matter(text)

    # <system_prompt>…</system_prompt> tags for system prompt extraction
    sp_match = _SYSTEM_PROMPT_TAG_RE.search(body)
    if not sp_match:
        raise ValueError(
            f"Workflow {Path(path).name} is missing <system_prompt>…</system_prompt> tags. "
            "All workflow files must wrap their system prompt content in these tags."
        )
    raw_system_prompt = sp_match.group(1).strip()

    sections = _split_h1_sections(body)
    response_json_schema, system_prompt = _extract_response_schema_and_redact_prompt(
        body=body,
        system_prompt=raw_system_prompt,
        workflow_name=Path(path).name,
    )

    return {
        "front_matter": front_matter,
        "system_prompt": system_prompt,
        "tools": _parse_tools(sections.get("Tools", "")),
        "prompt_template": sections.get("Prompt Template", ""),
        "response_json_schema": response_json_schema,
        "phases": _parse_phases(sections.get("Phases", "")),
    }


def render_prompt(
    parsed: dict[str, Any],
    *,
    purpose: str = "",
    instruction: str = "",
    context: str = "",
    id_catalog: str = "",
) -> str:
    """Substitute template variables into the prompt template.

    Parameters
    ----------
    parsed : dict
        Output of ``parse_workflow()``.
    purpose, instruction, context, id_catalog : str
        Values to insert into ``{{variable}}`` placeholders.

    Returns
    -------
    str  — the rendered prompt ready for the agent.
    """
    if "prompt_template" not in parsed or not isinstance(parsed["prompt_template"], str):
        raise ValueError("parsed workflow is missing the prompt_template string")
    tmpl = parsed["prompt_template"]
    replacements = {
        "purpose": purpose,
        "instruction": instruction,
        "context": context,
        "id_catalog": id_catalog,
    }
    for var, val in replacements.items():
        if not isinstance(val, str):
            raise TypeError(f"render_prompt {var} must be a string")
        tmpl = tmpl.replace("{{" + var + "}}", val)
    return tmpl
