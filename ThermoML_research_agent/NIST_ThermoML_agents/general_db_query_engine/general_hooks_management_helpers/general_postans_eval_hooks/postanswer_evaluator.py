"""Shared answer-only → parallel evaluation → downstream-JSON lifecycle.

The working ReAct agent is responsible only for the complete chemistry
answer. Once the ReAct loop terminates, one shared post-answer anchor launches
two independent, tool-free LLM agents concurrently:

* core-claims distillation from the answer text;
* ID/metadata alignment from the answer alone — the raw tool record never
  reaches an LLM branch; deterministic validators fetch the authoritative
  registry records for every emitted identifier and bounce observations
  back on mismatch.

When the declared return schema contains ``summary``, a hidden answer-summary
tool runs beside those agents and contributes its result to the ID-alignment
branch. A caller may then insert hidden deterministic ID-construction tools.
Their validation failures can drive bounded correction and submission review
before the assembler restores the untouched answer, validates identifier
pairs, and emits the downstream JSON document.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any, Callable

from ThermoML_raw_json_to_card_db_parsers.id_schema import (
    attach_lit_num_ids,
    reconcile_literature_identities,
    validate_nested_identifiers,
)

from ...general_argo_engine_helpers import anchor
from ...general_argo_engine_helpers.json_answer_guard import guard_json_answer
from . import _postans_eval_anchors_catalog as postans_anchor
from .summary_construction import SUMMARY_CONSTRUCTION_TOOL_CATALOG

log = logging.getLogger("postanswer-evaluator")


_ANSWER_ONLY_CONTRACT = """
# Working-agent final-answer contract

When the chemistry exploration is complete, emit exactly one
`<answer>...</answer>` block containing the full user-facing answer text.
The answer should include concise chemical insight, explanations, and useful
examples when supported by the completed work.

Do not emit JSON, core_claims arrays, source objects, identifier catalogs, schema
fields, or any other separate machine-facing structure in the final answer.
Identifiers or citations may appear naturally inside the answer text when
they are needed to explain a supported result, but they must not be emitted
as an additional structure. Two parallel, tool-free post-answer agents will
distill core_claims and organize ID/metadata fields after this agent terminates.
Earlier instructions that ask you to assemble final JSON are superseded by
this contract.
""".strip()

_CORE_CLAIMS_AGENT_SYSTEM = """
You are the core-claims distillation agent for a ThermoML post-answer hook.
You have no tools and must not request or simulate tool calls. This is one
isolated model call, not a ReAct chain.

Read only the completed chemistry answer. Distill it into one or several
well-supported central summary statements in `core_claims`. Each core claim
must capture a main conclusion, essential condition, or central limitation
that the answer states directly. Prefer the fewest claims that preserve the
answer's main scientific meaning. Preserve qualifiers, conditions, and
uncertainty. Exclude incidental details and examples unless they are central.
Do not add evidence, identifiers, source objects, metadata, or conclusions
that are merely implied by the task. Do not rewrite or include the full
answer. Copy any numeric values and units VERBATIM from the answer; never
introduce, recompute, round, or unit-convert a number.

If the answer asserts a comparative or derived relationship (e.g. "higher
than a weighted average of the endpoints") that fails simple arithmetic on
the numbers the answer itself quotes, do not carry that assertion into
`core_claims` — keep only the qualitative conclusion the quoted data
support.

When the declared schema includes a `confidence` field, also rate how
strongly the answer's central claims are supported by the answer's own
evidence trail, using one of the allowed values shown in the schema
example (e.g. high | medium | low): high — central claims quote verified
data and carry no unresolved flags; medium — central claims rest on
estimates, interpolations, or partial coverage the answer itself
acknowledges; low — the answer reports major gaps, unresolved UNVERIFIED
flags, or no results. Judge only from the answer text.

Example input answer:
At 298 K, the reported liquid density decreases as the ethanol mole fraction
increases. The available blocks cover only ambient pressure, so this trend
should not be extrapolated to high pressure.

Example output:
{"core_claims":["At 298 K and ambient pressure, the reported liquid density decreases with increasing ethanol mole fraction.","The available evidence does not support extrapolation to high pressure."]}

Output exactly one JSON object with the declared `core_claims` field. No
prose, tags, code fences, or undeclared fields.
""".strip()

_ID_ALIGNMENT_AGENT_SYSTEM = """
You are the ID-and-metadata alignment agent for a ThermoML post-answer hook.
You have no tools and must not request or simulate tool calls. This is one
isolated model call, not a ReAct chain.

The working agent has already produced the complete chemistry answer. The
raw tool record is withheld: read the answer and the original task context
only, and populate exactly the supplied agent-specific fields. Deterministic
validators fetch the authoritative registry records for every identifier
you emit, enrich them with metadata and ranges, and bounce a precise
observation back to a correction call on any mismatch.

1. Copy every identifier (GLOB*_N, PROPblock_N / RXNblock_N, BLKsubsys_N,
   DOI) character-for-character from the answer text. Never infer, repair,
   translate, or invent an identifier or value.
2. For evidence arrays (`sources`, `core_blocks_found`, and similar), emit
   one entry per database record the answer actually relies on. An entry
   the answer does not rely on must not appear; empty arrays are valid.
3. Only when the schema declares a `sources` array: emit its entries as
   MINIMAL references: `lit_num_id` and/or `doi`, plus `block`, plus
   `BLKsubsys_id` ONLY when the answer explicitly cites that subsystem. Do
   NOT write `description` in a `sources` entry — the block validator
   authors it after fetching the authoritative metadata and ranges. Never
   add a `sources` field the schema does not declare.
4. Every other evidence array (`core_blocks_found` and similar) carries the
   per-entry fields its schema example shows (system types,
   component/property IDs, descriptions); fill them from the answer alone.
   A `description` shown there is required in every entry: one sentence on
   what the answer uses that record for. Component/property ID lists the
   answer does not state may stay empty arrays; the validator fills them
   from the authoritative record. When a validation observation returns
   authoritative values, copy those exactly in your correction.
5. Treat BLKsubsys_id as opt-in: omit it for a declared parent-block
   anchor. Never add null merely to fill a minimal agent-authored anchor.
6. Do not emit `answer`, `core_claims`, or `summary`; the other post-answer
   branch, the hidden summary tool, and the deterministic assembler own
   those fields.
7. If the answer does not support a field, use only an empty/null value
   permitted by the schema.

Output exactly one JSON object with the requested top-level fields. No
prose, tags, code fences, or undeclared fields.
""".strip()


@dataclass(frozen=True)
class PostAnswerEvaluationResult:
    """Products of the two parallel branches and deterministic join."""

    answer: str
    core_claims_evaluation: dict[str, Any]
    id_alignment: dict[str, Any]
    assembled: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.assembled, indent=2, ensure_ascii=False)


def prepare_answer_only_system_prompt(system_prompt: str) -> str:
    """Add the answer contract to an already schema-redacted system prompt."""
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise TypeError("system_prompt must be non-empty text")
    if "# Response JSON Schema" in system_prompt:
        raise ValueError(
            "ANSWER_PROMPT_SCHEMA_LEAK: parse the agent skill Markdown before "
            "building the plain-text answer-agent prompt"
        )
    cleaned = system_prompt.rstrip() + "\n\n" + _ANSWER_ONLY_CONTRACT
    return (
        cleaned.rstrip()
        + "\n\n# Final-answer precedence\n\n"
        + "The working-agent final-answer contract above has precedence over "
        + "all earlier phase guidance. Final output is answer text only; the "
        + "parallel post-answer agents create every JSON field after "
        + "termination."
    )


def validate_response_json_schema(
    response_json_schema: dict[str, Any],
) -> dict[str, Any]:
    """Validate and copy the exact schema declared by the agent skill."""
    if not isinstance(response_json_schema, dict) or not response_json_schema:
        raise TypeError("response_json_schema must be a non-empty object")
    missing = {"answer", "core_claims"} - set(response_json_schema)
    if missing:
        raise ValueError(
            "response_json_schema is missing required fields: "
            + ", ".join(sorted(missing))
        )
    if not isinstance(response_json_schema["answer"], str):
        raise TypeError("response_json_schema.answer must be a string example")
    core_claims = response_json_schema["core_claims"]
    if not isinstance(core_claims, list) or len(core_claims) != 1:
        raise TypeError(
            "response_json_schema.core_claims must be a one-element array example"
        )
    return dict(response_json_schema)


def build_core_claims_schema(
    response_json_schema: dict[str, Any],
) -> dict[str, Any]:
    """Return the schema owned by the core-claims distillation agent.

    ``confidence`` is a support-strength judgment about the claims, so the
    claims agent authors it whenever the final schema declares it.
    """
    final_schema = validate_response_json_schema(response_json_schema)
    schema = {"core_claims": final_schema["core_claims"]}
    if "confidence" in final_schema:
        schema["confidence"] = final_schema["confidence"]
    return schema


def build_id_alignment_schema(
    response_json_schema: dict[str, Any],
) -> dict[str, Any]:
    """Return fields owned only by the ID/metadata-alignment agent."""
    final_schema = validate_response_json_schema(response_json_schema)
    return {
        key: value
        for key, value in final_schema.items()
        if key not in {"answer", "core_claims", "confidence"}
    }


def build_id_alignment_agent_schema(
    response_json_schema: dict[str, Any],
    deterministic_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return only fields authored by the tool-free ID-alignment agent.

    ``summary`` is produced by the hidden summary tool;
    ``data_inspections`` and every key in *deterministic_fields* are
    assembled deterministically — none is ever authored by an LLM branch.
    """
    hidden = {"summary", "data_inspections", *(deterministic_fields or {})}
    return {
        key: value
        for key, value in build_id_alignment_schema(response_json_schema).items()
        if key not in hidden
    }


def extract_answer_text(agent_answer: str) -> str:
    """Accept only the plain-text chemistry answer."""
    if not isinstance(agent_answer, str) or not agent_answer.strip():
        raise ValueError("working agent returned an empty answer")
    text = agent_answer.strip()
    try:
        json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text
    raise ValueError(
        "WORKING_AGENT_ANSWER_CONTRACT_ERROR: final answer must be plain text, "
        "not JSON"
    )


def _require_exact_fields(
    value: dict[str, Any],
    schema: dict[str, Any],
    *,
    branch: str,
) -> None:
    expected = set(schema)
    received = set(value)
    if received != expected:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        raise ValueError(
            f"POSTANSWER_{branch.upper()}_SCHEMA_ERROR: top-level fields "
            f"must be exactly {sorted(expected)}; missing={missing}, extra={extra}"
        )


def _require_core_claims(core_claims_evaluation: dict[str, Any]) -> None:
    core_claims = core_claims_evaluation.get("core_claims")
    if not isinstance(core_claims, list) or not core_claims or any(
        not isinstance(claim, str) or not claim.strip()
        for claim in core_claims
    ):
        raise TypeError(
            "POSTANSWER_CORE_CLAIMS_SCHEMA_ERROR: core_claims must be an array "
            "of non-empty strings"
        )
    if "confidence" in core_claims_evaluation:
        confidence = core_claims_evaluation["confidence"]
        if not isinstance(confidence, str) or not confidence.strip():
            raise TypeError(
                "POSTANSWER_CORE_CLAIMS_SCHEMA_ERROR: confidence must be a "
                "non-empty string"
            )


def assemble_return_json(
    *,
    answer: str,
    core_claims_evaluation: dict[str, Any],
    id_alignment: dict[str, Any],
    response_json_schema: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically join answer, core-claims, and ID/metadata branches."""
    final_schema = validate_response_json_schema(response_json_schema)
    core_claims_schema = build_core_claims_schema(response_json_schema)
    id_schema = build_id_alignment_schema(response_json_schema)
    _require_exact_fields(
        core_claims_evaluation,
        core_claims_schema,
        branch="core_claims",
    )
    _require_core_claims(core_claims_evaluation)
    _require_exact_fields(
        id_alignment,
        id_schema,
        branch="id_alignment",
    )

    assembled: dict[str, Any] = {"answer": answer}
    assembled.update(core_claims_evaluation)
    assembled.update(id_alignment)
    if set(assembled) != set(final_schema):
        raise AssertionError("post-answer assembly did not match the final schema")

    # GLOBlit_N is the authoritative literature identity: mismatched or
    # missing DOIs are repaired from the registry, entries with no
    # verifiable identity are scrubbed, and the completed run is never
    # destroyed at final assembly.
    scrubbed, identity_notes = reconcile_literature_identities(
        assembled,
        path="postanswer_return",
    )
    if identity_notes:
        log.error(
            "post-answer assembly reconciled %d literature identit(ies): %s",
            len(identity_notes),
            "; ".join(identity_notes[:8]),
        )
    if not isinstance(scrubbed, dict):
        raise AssertionError("literature-identity reconciliation changed result type")
    enriched = attach_lit_num_ids(
        scrubbed,
        path="postanswer_return",
    )
    if not isinstance(enriched, dict):
        raise AssertionError("post-answer identifier enrichment changed result type")
    validate_nested_identifiers(enriched, path="postanswer_return")
    return enriched


def _run_json_branch(
    *,
    client: Any,
    prompt: str,
    system: str,
    schema: dict[str, Any],
    label: str,
    branch: str,
    max_retries: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Run and exactly validate one tool-free post-answer branch."""
    raw = client.call(
        prompt,
        system,
        max_tokens=max_tokens,
    )
    parsed, _ = guard_json_answer(
        raw,
        client=client,
        label=f"{label}-postanswer-{branch}",
        schema=schema,
        max_retries=max_retries,
        max_tokens=max_tokens,
    )
    if not isinstance(parsed, dict):
        raise ValueError(
            f"{label} post-answer {branch} worker returned invalid JSON"
        )
    _require_exact_fields(parsed, schema, branch=branch)
    if branch == "core_claims":
        _require_core_claims(parsed)
    return parsed


def evaluate_and_assemble_return(
    *,
    agent_answer: str,
    response_json_schema: dict[str, Any],
    client: Any,
    label: str,
    tool_history: list[dict] | None = None,
    task_context: str = "",
    hooks: Any = None,
    validator: Callable[[dict[str, Any]], Any] | None = None,
    id_alignment_processor: Callable[..., dict[str, Any]] | None = None,
    deterministic_fields: dict[str, Any] | None = None,
    postassembly_checker: Callable[..., dict[str, Any]] | None = None,
    core_claims_system_suffix: str = "",
    max_retries: int = 2,
    max_tokens: int = 6000,
) -> PostAnswerEvaluationResult:
    """Launch parallel evaluators, optional hidden tools, and JSON assembly."""
    if not isinstance(label, str) or not label.strip():
        raise TypeError("label must be non-empty text")
    if not isinstance(task_context, str):
        raise TypeError("task_context must be text")
    if not isinstance(core_claims_system_suffix, str):
        raise TypeError("core_claims_system_suffix must be text")
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    if deterministic_fields is not None and not isinstance(deterministic_fields, dict):
        raise TypeError("deterministic_fields must be an object when given")
    deterministic_fields = dict(deterministic_fields or {})
    if {"answer", "core_claims", "summary"} & set(deterministic_fields):
        raise ValueError(
            "deterministic_fields cannot override agent-authored branches"
        )

    answer = extract_answer_text(agent_answer)
    final_schema = validate_response_json_schema(response_json_schema)
    core_claims_schema = build_core_claims_schema(response_json_schema)
    id_schema = build_id_alignment_schema(response_json_schema)
    missing_declared = set(deterministic_fields) - set(id_schema)
    if missing_declared:
        raise ValueError(
            "deterministic_fields must be declared in response_json_schema: "
            + ", ".join(sorted(missing_declared))
        )
    # LLM branches never see or author deterministic fields.
    branch_id_schema = {
        key: value for key, value in id_schema.items()
        if key not in deterministic_fields
    }
    id_agent_schema = build_id_alignment_agent_schema(
        response_json_schema, deterministic_fields,
    )
    needs_summary = "summary" in id_schema
    engine_hooks = getattr(hooks, "engine_hooks", hooks)

    anchor(
        postans_anchor.POSTANS_ANSWER_RECEIVED,
        engine_hooks,
        label=label,
        answer=answer,
        final_schema=final_schema,
    )

    core_claims_prompt = (
        "## Working agent answer\n"
        + answer
        + "\n\n## Required core-claims JSON schema\n"
        + json.dumps(core_claims_schema, indent=2, ensure_ascii=False)
    )
    # The raw tool record never reaches the LLM branch: identifiers come
    # from the answer; deterministic validators (id_alignment_processor)
    # fetch the authoritative records for every emitted ID.
    if id_alignment_processor is None and "sources" in id_agent_schema:
        raise ValueError(
            "a 'sources'-bearing schema requires an id_alignment_processor "
            "to validate and enrich the minimal source references"
        )
    completed_run_record = "[]"
    record_section = (
        "(withheld — emit minimal identifier references from the answer; "
        "deterministic validators fetch and verify the authoritative "
        "records after this call)"
    )
    id_prompt = (
        "## Original task context\n"
        + (task_context.strip() or "(not provided)")
        + "\n\n## Working agent answer\n"
        + answer
        + "\n\n## Completed run record\n"
        + record_section
        + "\n\n## Required ID/metadata JSON schema\n"
        + json.dumps(id_agent_schema, indent=2, ensure_ascii=False)
    )
    anchored_prompts = anchor(
        postans_anchor.POSTANS_PARALLEL_AGENTS_DISPATCH,
        engine_hooks,
        label=label,
        core_claims_prompt=core_claims_prompt,
        id_alignment_prompt=id_prompt,
        core_claims_schema=core_claims_schema,
        id_alignment_schema=id_agent_schema,
        branches=("core_claims", "id_alignment"),
    )
    if anchored_prompts is not None:
        if (
            not isinstance(anchored_prompts, dict)
            or set(anchored_prompts) != {"core_claims_prompt", "id_alignment_prompt"}
            or not all(isinstance(value, str) for value in anchored_prompts.values())
        ):
            raise TypeError(
                "POSTANS_EVALUATION_BEFORE hook must return exactly "
                "{'core_claims_prompt': str, 'id_alignment_prompt': str}"
            )
        core_claims_prompt = anchored_prompts["core_claims_prompt"]
        id_prompt = anchored_prompts["id_alignment_prompt"]

    try:
        # A separate context copy is required for each concurrently entered
        # branch. This preserves the active StatsRecorder/run bindings used
        # by ArgoClient without attempting to enter one Context twice.
        core_claims_context = copy_context()
        id_context = copy_context()
        summary_context = copy_context()
        if needs_summary:
            anchor(
                postans_anchor.POSTANS_SUMMARY_TOOL_BEFORE,
                engine_hooks,
                label=label,
                answer=answer,
                tool_name="construct_answer_summary",
            )
        with ThreadPoolExecutor(
            max_workers=3 if needs_summary else 2,
            thread_name_prefix=f"{label}-postanswer",
        ) as pool:
            core_claims_future = pool.submit(
                core_claims_context.run,
                _run_json_branch,
                client=client,
                prompt=core_claims_prompt,
                system=_CORE_CLAIMS_AGENT_SYSTEM + core_claims_system_suffix,
                schema=core_claims_schema,
                label=label,
                branch="core_claims",
                max_retries=max_retries,
                max_tokens=max_tokens,
            )
            id_future = pool.submit(
                id_context.run,
                _run_json_branch,
                client=client,
                prompt=id_prompt,
                system=_ID_ALIGNMENT_AGENT_SYSTEM,
                schema=id_agent_schema,
                label=label,
                branch="id_alignment",
                max_retries=max_retries,
                max_tokens=max_tokens,
            )
            summary_future = None
            if needs_summary:
                summary_future = pool.submit(
                    summary_context.run,
                    SUMMARY_CONSTRUCTION_TOOL_CATALOG.tools[
                        "construct_answer_summary"
                    ],
                    answer,
                    client=client,
                    label=label,
                    max_retries=max_retries,
                    max_tokens=min(max_tokens, 1000),
                )
            core_claims_evaluation = core_claims_future.result()
            id_alignment = id_future.result()
            # Self-heal: strip deterministic/hidden keys a drifting branch
            # may still emit — authoritative values are merged below.
            stray = set(id_alignment) & (
                set(deterministic_fields) | {"data_inspections"}
            )
            if stray:
                log.warning(
                    "[%s] id-alignment branch emitted deterministic field(s) "
                    "%s — dropped (deterministic assembly owns them)",
                    label, sorted(stray),
                )
                id_alignment = {
                    key: value for key, value in id_alignment.items()
                    if key not in stray
                }
            if summary_future is not None:
                summary_result = summary_future.result()
                id_alignment = {**id_alignment, **summary_result}
                anchor(
                    postans_anchor.POSTANS_SUMMARY_TOOL_AFTER,
                    engine_hooks,
                    label=label,
                    tool_name="construct_answer_summary",
                    summary=summary_result["summary"],
                )
        _require_exact_fields(
            id_alignment,
            branch_id_schema,
            branch="id_alignment",
        )
        if id_alignment_processor is not None:
            id_alignment = id_alignment_processor(
                id_alignment=id_alignment,
                client=client,
                id_alignment_schema=branch_id_schema,
                id_alignment_prompt=id_prompt,
                answer=answer,
                core_claims_evaluation=core_claims_evaluation,
                completed_run_record=completed_run_record,
                label=label,
                max_tokens=max_tokens,
                engine_hooks=engine_hooks,
            )
            if not isinstance(id_alignment, dict):
                raise TypeError("id_alignment_processor must return an object")
            _require_exact_fields(
                id_alignment,
                branch_id_schema,
                branch="id_alignment",
            )
        if deterministic_fields:
            id_alignment = {**id_alignment, **deterministic_fields}
    except Exception as exc:
        anchor(
            postans_anchor.POSTANS_VALIDATION_FAILED,
            engine_hooks,
            label=label,
            stage="parallel_evaluation",
            answer=answer,
            error=str(exc),
        )
        raise

    anchor(
        postans_anchor.POSTANS_EVALUATION_AFTER,
        engine_hooks,
        label=label,
        core_claims_evaluation=core_claims_evaluation,
        id_alignment=id_alignment,
    )

    anchor(
        postans_anchor.POSTANS_ASSEMBLY_BEFORE,
        engine_hooks,
        label=label,
        answer=answer,
        core_claims_evaluation=core_claims_evaluation,
        id_alignment=id_alignment,
    )
    try:
        assembled = assemble_return_json(
            answer=answer,
            core_claims_evaluation=core_claims_evaluation,
            id_alignment=id_alignment,
            response_json_schema=response_json_schema,
        )
        if postassembly_checker is not None:
            assembled = postassembly_checker(
                assembled=assembled,
                agent_answer=answer,
                tool_history=tool_history,
            )
            if not isinstance(assembled, dict) or set(assembled) != set(final_schema):
                raise ValueError(
                    "postassembly_checker must retain the assembled response fields"
                )
            if not isinstance(assembled["answer"], str):
                raise TypeError("postassembly_checker must retain a text answer")
            _require_core_claims(assembled)
            validate_nested_identifiers(assembled, path="postanswer_return")
        if validator is not None:
            validator(assembled)
    except Exception as exc:
        anchor(
            postans_anchor.POSTANS_VALIDATION_FAILED,
            engine_hooks,
            label=label,
            stage="assembly",
            error=str(exc),
            core_claims_evaluation=core_claims_evaluation,
            id_alignment=id_alignment,
        )
        raise

    anchor(
        postans_anchor.POSTANS_ASSEMBLY_AFTER,
        engine_hooks,
        label=label,
        assembled=assembled,
    )
    return PostAnswerEvaluationResult(
        answer=assembled["answer"],
        core_claims_evaluation={
            key: assembled[key] for key in core_claims_evaluation
        },
        id_alignment={key: assembled[key] for key in id_alignment},
        assembled=assembled,
    )


__all__ = [
    "PostAnswerEvaluationResult",
    "prepare_answer_only_system_prompt",
    "validate_response_json_schema",
    "build_core_claims_schema",
    "build_id_alignment_schema",
    "build_id_alignment_agent_schema",
    "extract_answer_text",
    "assemble_return_json",
    "evaluate_and_assemble_return",
]
