"""JSON-stage guard — validate model-emitted JSON and correct formatting.
========================================================================
ThermoML working agents now finish with answer text only. The shared
post-answer hook launches parallel core-claims and ID/metadata agents, and this
guard validates each agent's machine-facing JSON. It can also be reused by
any other isolated JSON-producing stage.

1. ``clean_json_answer`` parses one exact JSON document.  ONE markdown
   code fence wrapping the whole document (```` ``` ```` or ```` ```json ````)
   is removed by the guard itself (``unwrap_json_fence``) — no corrective
   model call for that, and the unwrapped text is what the guard propagates.
   Tags, prose outside the fence, trailing commas, and concatenated
   documents remain invalid.
2. ``validate_against_schema`` checks structural conformance against
   the workflow's parsed ``response_json_schema`` (the fenced JSON example
   from the agent's own workflow md): top-level keys present, no undeclared
   extra keys, value types match the example (nulls tolerated), list
   elements typed.
3. When BOTH pass, the guard passes SILENTLY (no extra LLM call, debug log).
   Otherwise ``guard_json_answer`` asks the SAME client to correct the
   answer — the prompt carries the actual schema JSON and the specific
   violations — up to ``max_retries`` times.
4. When correction also fails, ``parsed`` is ``None``.  Callers must stop
   the dispatch rather than propagating a malformed agent result.

Usage::

    from ...general_argo_engine_helpers.json_answer_guard import guard_json_answer
    parsed, answer = guard_json_answer(
        evaluator_answer, client=client, label="comp-L1-postanswer",
        schema=evaluation_schema,
    )
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

log = logging.getLogger("json-answer-guard")

# ONE markdown code fence around the whole answer: ``` or ```json (any case), the
# document, the closing ```.  Anything outside the fence makes it NOT match.
_FENCE_RE = re.compile(r"\A```[ \t]*(?:json)?[ \t]*\r?\n?(.*?)\r?\n?[ \t]*```\Z",
                       re.DOTALL | re.IGNORECASE)

_CORRECTION_SYSTEM = (
    "You are a strict formatting corrector. Convert the given agent answer "
    "into ONE valid JSON document conforming to the required schema. "
    "Preserve every declared key's content faithfully — NEVER invent or "
    "alter values. REMOVE top-level keys the schema does not declare "
    "(their content is discarded, not relocated). "
    "Output ONLY the JSON: no prose, no tags, no code fences."
)


def unwrap_json_fence(answer: str) -> str:
    """Return the text inside ONE markdown code fence that wraps the whole
    answer (```` ``` ```` or ```` ```json ````), stripped; otherwise the
    stripped answer unchanged.  Prose before or after the fence is left alone
    (the answer then fails parsing and goes to correction as before)."""
    text = (answer or "").strip()
    m = _FENCE_RE.match(text)
    if m is None:
        return text
    return m.group(1).strip()


def clean_json_answer(answer: str) -> Any | None:
    """Parse one exact JSON document (a single wrapping code fence is removed
    first), returning ``None`` when invalid."""
    if not answer or not answer.strip():
        return None
    try:
        return json.loads(unwrap_json_fence(answer))
    except (json.JSONDecodeError, ValueError):
        return None


def _type_name(v: Any) -> str:
    return {dict: "object", list: "array", str: "string", bool: "boolean",
            int: "number", float: "number",
            type(None): "null"}.get(type(v), type(v).__name__)


def _type_ok(example: Any, value: Any) -> bool:
    """Example-based type compatibility with explicit nullable values."""
    if value is None:
        return True
    if example is None:
        return False
    if isinstance(example, bool):
        return isinstance(value, bool)
    if isinstance(example, (int, float)):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(example, str):
        return isinstance(value, str)
    if isinstance(example, list):
        return isinstance(value, list)
    if isinstance(example, dict):
        return isinstance(value, dict)
    return True


def validate_against_schema(parsed: Any, schema: Any) -> list[str]:
    """Structural conformance against the workflow's example schema.

    The schema is the parsed fenced-JSON EXAMPLE from the agent's
    workflow md (``parse_workflow()['response_json_schema']``), not a formal
    JSON Schema — so validation is deliberately lenient about VALUES:
    top-level keys must exist with example-compatible types (null allowed);
    elements of arrays whose example element is an object must be objects.
    Undeclared extra top-level keys are violations: the domain-specific
    exact-field validators that run after this guard raise WITHOUT a
    correction loop, so extras must be repaired here. Returns a list
    of violations (empty = OK).
    """
    if not isinstance(schema, (dict, list)):
        raise TypeError("response schema must be a JSON object or array example")
    problems: list[str] = []
    if isinstance(schema, dict):
        if not isinstance(parsed, dict):
            return [f"top-level must be an object, got {_type_name(parsed)}"]
        for k in parsed:
            if k not in schema:
                problems.append(f"undeclared extra key '{k}'")
        for k, ex in schema.items():
            if k not in parsed:
                problems.append(f"missing key '{k}'")
            elif not _type_ok(ex, parsed[k]):
                problems.append(
                    f"key '{k}' should be {_type_name(ex)}, "
                    f"got {_type_name(parsed[k])}"
                )
            elif (isinstance(ex, list) and ex and isinstance(ex[0], dict)
                  and isinstance(parsed[k], list)):
                for i, item in enumerate(parsed[k]):
                    if not isinstance(item, dict):
                        problems.append(f"'{k}[{i}]' should be an object, "
                                        f"got {_type_name(item)}")
    elif isinstance(schema, list):
        if not isinstance(parsed, list):
            return [f"top-level must be an array, got {_type_name(parsed)}"]
    return problems


def guard_json_answer(
    answer: str,
    *,
    client: Any,
    label: str,
    schema: Any,
    require: type = dict,
    max_retries: int = 1,
    max_tokens: int = 4000,
    call_fn: Callable | None = None,
) -> tuple[Any | None, str]:
    """Guard one JSON-stage result with an optional corrective model call.

    Parameters
    ----------
    answer : str
        The JSON-producing stage's result.
    client
        The agent's ArgoClient (``.call(prompt, system, max_tokens=)``);
        used ONLY when cleaning or schema validation fails.
    label : str
        Agent label for logging (e.g. "comp-L1").
    schema
        The parsed response-schema EXAMPLE from the agent's workflow md
        (``parse_workflow()['response_json_schema']``). When given, the answer
        must also CONFORM structurally — violations trigger correction
        with the schema JSON embedded in the prompt.
    require : type
        Required top-level type (default dict).
    max_retries : int
        Corrective LLM calls to attempt (default 1).
    call_fn : callable, optional
        Override for the correction call (testing) — signature
        ``call_fn(prompt, system, max_tokens) -> str``.

    Returns
    -------
    (parsed, answer_text)
        ``parsed`` is the JSON object or None; ``answer_text`` is the
        (possibly corrected) answer string to propagate downstream.  When
        the only flaw was a wrapping markdown code fence, ``answer_text``
        is the unwrapped JSON text (the fence is gone).
    """
    parsed = clean_json_answer(answer)
    violations = (validate_against_schema(parsed, schema)
                  if isinstance(parsed, require) else [])
    if isinstance(parsed, require) and not violations:
        unwrapped = unwrap_json_fence(answer)
        if unwrapped != (answer or "").strip():
            log.info("[%s] markdown code fence removed from the JSON answer "
                     "(schema conform, no correction call).", label)
            return parsed, unwrapped
        log.debug("[%s] exact JSON answer is schema-conform "
                  "(schema conform).", label)
        return parsed, answer

    if not (answer or "").strip():
        log.warning("[%s] empty answer — nothing to correct.", label)
        return None, answer

    schema_txt = json.dumps(schema, indent=1, default=str)
    do_call = call_fn or (
        lambda prompt, system, mt: client.call(prompt, system, max_tokens=mt)
    )
    best_answer = answer
    for attempt in range(1, max_retries + 1):
        if not isinstance(parsed, require):
            problem_txt = "it could not be parsed as JSON at all"
        else:
            problem_txt = ("it does not conform to the required schema: "
                           + "; ".join(violations[:8]))
        log.warning("[%s] answer rejected — %s — requesting correction "
                    "(attempt %d/%d).", label, problem_txt, attempt,
                    max_retries)
        prompt = (
            "Your previous answer was rejected: " + problem_txt + ".\n\n"
            "Previous answer:\n<<<\n" + best_answer + "\n>>>\n\n"
            + "Required output schema (example — match its keys and types):\n"
            + schema_txt + "\n\n"
            + "Re-emit the content as ONE valid JSON "
            + ("object" if require is dict else "document")
            + " with EXACTLY the schema's top-level keys — drop undeclared "
              "keys, never invent values. Output ONLY the JSON — no prose, "
              "no tags, no code fences."
        )
        try:
            corrected = do_call(prompt, _CORRECTION_SYSTEM, max_tokens)
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] correction call failed: %s", label, exc)
            break
        parsed = clean_json_answer(corrected)
        violations = (validate_against_schema(parsed, schema)
                      if isinstance(parsed, require) else [])
        if isinstance(parsed, require):
            best_answer = unwrap_json_fence(corrected)
            if not violations:
                log.info("[%s] JSON corrected and schema-conform on "
                         "attempt %d.", label, attempt)
                return parsed, best_answer

    final_problem = ("unparseable" if not isinstance(parsed, require)
                     else "; ".join(violations[:8]))
    log.error(
        "[%s] JSON answer failed the exact schema after %d correction "
        "attempt(s): %s",
        label,
        max_retries,
        final_problem,
    )
    return None, best_answer

