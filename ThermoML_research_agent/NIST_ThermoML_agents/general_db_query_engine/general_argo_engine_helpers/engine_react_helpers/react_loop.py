"""
ReAct loop — synchronous ``agent_turn()`` implementation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Callable, Dict, List

from ..engine_config import get_config
from ..argo_client_caller import ArgoClient
from ..engine_tool_interface_helpers.tool_call_parser import extract_all_tool_calls
from .react_helpers import (
    _validate_tool_arguments,
    validate_native_tool_result,
    require_result_within_limit,
    AgentTurnResult,
    _get_compaction_trigger_chars,
)
from ..engine_hooks_anchors import anchor as _anchor, EngineHooks, AGENT_RECORD_REFERENCES as _AGENT_RECORD_REFERENCES

from ...general_hooks_management_helpers.general_context_hooks.context_cleanup_compactor_hooks import (
    SUMMARY_BLOCK_RE as _SUMMARY_BLOCK_RE,
    compact_tool_calls_for_memory as _compact_tool_calls_for_memory,
    stage_compact_batch as _stage_compact_batch,
)
from ...general_hooks_management_helpers.general_context_hooks.self_compactor_interactive_hooks import (
    build_compaction_reminder as _build_compaction_reminder,
    compact_memory as _compact_memory,
    parse_compaction_guidance as _parse_compaction_guidance,
)
from ...general_hooks_management_helpers.general_context_hooks import _context_hooks_anchors_catalog as ctx_anchor
from ...general_hooks_management_helpers.general_memory_management_tools_hooks_helpers import _memory_hooks_anchors_catalog as mem_anchor
from ...general_tool_management_helpers import _tool_hooks_anchors_catalog as tool_anchor
from ...general_text_context_marker_catalog import (
    ALL_CONTEXT_MARKERS_RE as _ALL_MARKERS_RE,
    MARKERS,
    wrap_subagent_answer,
)
from ...general_data_grounding_gate.profile import profile_metadata as _grounding_profile_metadata
from ...general_data_grounding_gate import (
    MAX_GATE_BOUNCES as _MAX_GATE_BOUNCES,
    build_continuity_nudge as _build_continuity_nudge,
    build_flags_note as _build_flags_note,
    build_gate_nudge as _build_gate_nudge,
    build_timeout_gate_instruction as _build_timeout_gate_instruction,
    count_gate_bounces as _count_gate_bounces,
    gate_enabled as _gate_enabled,
    run_data_grounding_gate as _run_data_grounding_gate,
)
from .subagent_context_render import build_subagent_context_display

TAG_TOOL_CALL_OPEN = MARKERS.tool_call.open
TAG_TOOL_RESULT_OPEN = MARKERS.tool_result.open
TAG_TOOL_RESULT_CLOSE = MARKERS.tool_result.close
TAG_SUMMARY_OPEN = MARKERS.summary.open
TAG_SUMMARY_CLOSE = MARKERS.summary.close
TAG_COMPACT_NOTE_OPEN = MARKERS.compact_note.open
TAG_COMPACT_NOTE_CLOSE = MARKERS.compact_note.close
TAG_COMPACTION_REMINDER_OPEN = MARKERS.compaction_reminder.open
TAG_WAIT = MARKERS.wait_tag
TAG_SYSTEM_PROMPT_OPEN = MARKERS.system_prompt.open
TAG_SYSTEM_PROMPT_CLOSE = MARKERS.system_prompt.close
TAG_ANSWER_OPEN = MARKERS.answer.open
TAG_ANSWER_CLOSE = MARKERS.answer.close
TAG_MEMORY_OPEN = MARKERS.memory.open
TAG_MEMORY_CLOSE = MARKERS.memory.close
_COMPACT_NOTE_RE = MARKERS.compact_note_re
_COMPRESS_TAG_RE = MARKERS.compress_re
TAG_VALIDATION_OPEN = MARKERS.validation_guidance.open
TAG_VALIDATION_CLOSE = MARKERS.validation_guidance.close
_VALIDATION_GUIDANCE_RE = MARKERS.validation_guidance_re

# Regex that strips full <reasoning>…</reasoning> blocks (with content)
_REASONING_BLOCK_RE = MARKERS.reasoning_strip_re
# Regex that strips full <summary>…</summary> blocks (with content)
_SUMMARY_STRIP_RE = re.compile(r"<summary>.*?</summary>\s*", re.DOTALL)


def _clean_answer_text(raw: str) -> str:
    """Strip reasoning/summary blocks and leftover marker tags from a final answer."""
    text = _REASONING_BLOCK_RE.sub("", raw)
    text = _SUMMARY_STRIP_RE.sub("", text)
    text = _ALL_MARKERS_RE.sub("", text)
    return text.strip()


# ── Session event log plumbing (labels + verbatim outputs) ───────
# Best-effort only: every helper degrades to a no-op outside a
# tracked run and never interferes with the ReAct loop itself.

def _active_history_recorder():
    """Active run-history recorder, or None outside a tracked run."""
    try:
        from ...general_hooks_management_helpers.general_context_hooks.history_tracking_hooks import (
            get_active_history_recorder,
        )
        return get_active_history_recorder()
    except Exception:
        return None


def _push_history_activity(segment: str):
    """Push a label segment (e.g. ``Turn#3``) on the active recorder."""
    recorder = _active_history_recorder()
    if recorder is None:
        return None
    try:
        return (recorder, recorder.push_activity(segment))
    except Exception:
        return None


def _pop_history_activity(handle) -> None:
    if not handle:
        return
    recorder, token = handle
    try:
        recorder.pop_activity(token)
    except Exception:
        pass


def _begin_tool_event(tool_name: str):
    """Open a labelled tool-dispatch event in the session event log."""
    recorder = _active_history_recorder()
    if recorder is None:
        return None
    try:
        return (recorder, recorder.begin_tool_activity(tool_name))
    except Exception:
        return None


def _end_tool_event(handle, result_chars: int = 0, error: str = "") -> None:
    if not handle:
        return
    recorder, inner = handle
    try:
        recorder.end_tool_activity(
            inner, result_chars=result_chars, error=error)
    except Exception:
        pass


def _log_answer_event(answer_text: str, kind: str = "answer") -> None:
    """Record the shipped answer verbatim in the session event log."""
    recorder = _active_history_recorder()
    if recorder is None:
        return
    try:
        recorder.log_answer_event(answer_text, kind=kind)
    except Exception:
        pass


_REJECTED_BLOCK_CAP = 1200


def _strip_rejected_turn(response: str) -> str:
    """Archive form of a syntax-rejected turn: tool-call blocks only.

    Narration around the blocks is dropped so runaway fabricated content
    never re-enters the context verbatim; the blocks themselves are kept
    (capped) so the agent can see and correct its own syntax.
    """
    blocks: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in MARKERS.any_tool_call_re.finditer(response):
        spans.append(m.span())
        block = m.group(0)
        if len(block) > _REJECTED_BLOCK_CAP:
            block = block[:_REJECTED_BLOCK_CAP] + " …[truncated]</tool_call>"
        blocks.append(block)
    open_tag = MARKERS.tool_call.open
    pos = response.find(open_tag)
    while pos != -1:
        if not any(start <= pos < end for start, end in spans):
            orphan = response[pos:pos + _REJECTED_BLOCK_CAP]
            blocks.append(orphan + " …[unclosed <tool_call> truncated]")
        pos = response.find(open_tag, pos + 1)
    dropped = max(len(response) - sum(len(b) for b in blocks), 0)
    if not blocks:
        return (
            "[SYNTAX-REJECTED TURN — no parseable <tool_call> blocks; "
            f"narration stripped from context ({len(response)} chars). "
            "Re-emit the corrected batch.]"
        )
    note = (
        "[SYNTAX-REJECTED TURN — narration stripped from context "
        f"({dropped} chars dropped). The tool calls above were NOT executed; "
        "re-emit the corrected batch.]"
    )
    return "\n".join([*blocks, note])


def _clean_syntax_rejection_traces(memory: list[dict]) -> None:
    """Collapse archived syntax-rejected turns once a later batch executes.

    The malformed blocks served their corrective purpose the moment a valid
    batch ran; from then on they only contaminate the working context. The
    full detail remains in the run log (WARNING at rejection time) and the
    validation-blocked history tables.
    """
    for msg in memory:
        if msg["role"] != "assistant":
            continue
        if "[SYNTAX-REJECTED TURN" not in msg["content"]:
            continue
        msg["content"] = (
            "(syntax-rejected turn — malformed tool calls never executed; "
            "corrected on a later turn. Full detail in the run log.)"
        )


def _clean_validation_traces(memory: list[dict]) -> None:
    """Replace old ``<validation_guidance>`` blocks with a compact note.

    Scans user-role messages for ``<validation_guidance>…</validation_guidance>``
    tags.  Extracts the blocked tool names, then replaces the verbose guidance
    body with a one-liner noting which tools were blocked.
    This keeps context lean when the agent eventually fixes params.
    """
    for msg in memory:
        if msg["role"] != "user":
            continue
        content = msg["content"]
        if TAG_VALIDATION_OPEN not in content:
            continue

        def _compact_match(m: re.Match) -> str:
            body = m.group(0)
            # Strip the wrapper tags to get the guidance body
            body = body.replace(TAG_VALIDATION_OPEN, "").replace(TAG_VALIDATION_CLOSE, "")
            blocked = [
                line.split("**")[1].split("\u2717 ")[-1].strip()
                for line in body.split("\n")
                if "\u2717 " in line
            ]
            if blocked:
                names = ", ".join(blocked)
                return f"(validation blocked {names} earlier — corrected)"
            return "(validation fired earlier — corrected)"

        msg["content"] = _VALIDATION_GUIDANCE_RE.sub(_compact_match, content)


def _prevalidate_batch_arguments(
    tool_calls: list[dict], tools: Dict[str, Callable]
) -> str | None:
    """Atomically validate names and argument contracts before execution.

    This engine-level gate applies even when an agent does not install a
    catalog-specific batch-validation hook. Domain guidance hooks still run
    afterward, but only once every call is syntactically executable.
    """
    blocked: list[tuple[str, str]] = []
    ready: list[str] = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        fn = tools.get(tool_name)
        if fn is None:
            blocked.append((
                tool_name,
                f"UNKNOWN_TOOL: {tool_name!r} is not registered; available tools are "
                f"{sorted(tools)}",
            ))
            continue
        try:
            _validate_tool_arguments(tool_name, tool_call["arguments"], fn)
        except Exception as exc:  # exact correction is returned to ReAct
            blocked.append((tool_name, str(exc)))
        else:
            ready.append(tool_name)
    if not blocked:
        return None
    parts = [
        f"**\u2717 {name}** — tool syntax or arguments need correction:\n{reason}"
        for name, reason in blocked
    ]
    if ready:
        parts.append(
            f"**\u2713 {', '.join(ready)}** — syntax-valid but held back because "
            "the batch is atomic"
        )
    parts.append(
        "_No tool in this batch executed. Re-emit the complete batch with only "
        "registered tool names and exact declared JSON arguments._"
    )
    return "\n\n".join(parts)

log = logging.getLogger("ThermoML-UI")


def _build_syntax_rejection_note(syntax_rejections: list[dict]) -> str:
    """Answer-rejection reminder: syntax-rejected calls produced NO results."""
    if not syntax_rejections:
        return ""
    n_blocks = sum(len(r["errors"]) for r in syntax_rejections)
    turns = ", ".join(str(r["iteration"]) for r in syntax_rejections)
    return (
        f"\n\nNOTE: {n_blocks} tool-call block(s) on turn(s) {turns} were "
        "rejected for malformed syntax and NEVER EXECUTED — no result exists "
        "for them. Repair the answer from executed tool results only; do not "
        "quote anything you 'remember' from those rejected turns."
    )


def _apply_forced_final_grounding_gate(
    clean_answer: str,
    tools: Dict[str, Callable],
    tool_history: list[dict],
    memory: list[dict],
    client,
    system_prompt: str,
    syntax_rejections: list[dict] | None = None,
) -> str:
    """Mandatory gate on forced-final answers: one tool-free rewrite, then flags.

    The time/iteration budget is spent, so violations get exactly one
    rewrite pass (drop uninspected values, keep aggregates); anything
    still ungrounded is deterministically flagged — never silently shipped.
    """
    if not _gate_enabled(tools):
        return clean_answer
    report = _run_data_grounding_gate(clean_answer, tool_history)
    if report.ok:
        return clean_answer
    log.warning(
        "Data-grounding gate (forced final): %d violation(s) — rewrite pass\n%s",
        len(report.violations),
        report.report_text,
    )
    try:
        memory.append({
            "role": "user",
            "content": (
                _build_timeout_gate_instruction(report)
                + _build_syntax_rejection_note(syntax_rejections or [])
            ),
        })
        rewrite = client.call(
            "\n\n".join(
                f"## {'User' if t['role'] == 'user' else 'Assistant'}\n{t['content']}"
                for t in memory
            ),
            system_prompt,
            event_kind="gate-rewrite",
        )
        memory.append({"role": "assistant", "content": rewrite})
        candidate = _clean_answer_text(rewrite)
        second = _run_data_grounding_gate(candidate, tool_history)
        if second.ok:
            return candidate
        if len(second.violations) < len(report.violations):
            return candidate + _build_flags_note(second)
    except Exception:
        log.error(
            "Forced-final grounding rewrite failed — flagging instead",
            exc_info=True,
        )
    return clean_answer + _build_flags_note(report)


# ═══════════════════════════════════════════════════════════════
#  Main ReAct loop
# ═══════════════════════════════════════════════════════════════

def agent_turn(
    user_message: str,
    *,
    system_prompt: str,
    tools: Dict[str, Callable],
    memory: List[dict],
    client: ArgoClient | None = None,
    max_iterations: int | None = None,
    timeout: int | None = None,
    required_tools: set[str] | None = None,
    compaction_trigger_chars: int | None = None,
    compaction_interval: int | None = None,
    hooks: EngineHooks | None = None,
    is_subagent: bool = False,
) -> AgentTurnResult:
    """Execute one user turn in the ReAct agentic loop.

    Parameters
    ----------
    user_message : str
        The user's question or instruction.
    system_prompt : str
        System message for the LLM.
    tools : dict[str, Callable]
        Tool registry mapping tool names to callables.
    memory : list[dict]
        Mutable conversation history (role/content dicts). Modified in place.
    client : ArgoClient, optional
        LLM caller. Defaults to ArgoClient.for_l0().
    max_iterations : int, optional
        Override MAX_TOOL_ITERATIONS.
    timeout : int, optional
        Override MAX_TURN_SECONDS.
    required_tools : set[str], optional
        If given, at least ONE of these tools must be called before the
        agent can give a final answer.  If the agent tries to answer
        without calling any, a correction is injected once.
    compaction_trigger_chars : int, optional
        Override the default context-size threshold that triggers
        compaction.  Defaults to ``_COMPACTION_TRIGGER_CHARS``.
    compaction_interval : int, optional
        If set, run the LLM compactor every N tool calls regardless of
        context size.  The compactor's selection step can still choose
        to skip ("NONE") if nothing needs compacting.
    hooks : EngineHooks, optional
        Compiled hook bindings keyed by catalog-defined anchor types.
    is_subagent : bool, optional
        When True, the first user message in the flat prompt uses
        ``## Subagent`` instead of ``## User``.  Set by subagent
        delegation tools to mark delegated queries.

    Returns
    -------
    AgentTurnResult
        The final answer, iteration count, timing, and tool history.
    """
    if not isinstance(user_message, str) or not user_message.strip():
        raise TypeError("user_message must be a non-empty string")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise TypeError("system_prompt must be a non-empty string")
    if not isinstance(tools, dict) or not tools:
        raise TypeError("tools must be a non-empty name-to-callable object")
    for name, fn in tools.items():
        if not isinstance(name, str) or not name or not callable(fn):
            raise TypeError("every tool entry must have a non-empty string name and callable value")
    if not isinstance(memory, list):
        raise TypeError("memory must be a list")
    for index, turn in enumerate(memory):
        if not isinstance(turn, dict) or set(turn) != {"role", "content"}:
            raise TypeError(f"memory[{index}] must contain exactly role and content")
        if turn["role"] not in {"user", "assistant"} or not isinstance(turn["content"], str):
            raise TypeError(f"memory[{index}] has an invalid role/content")
    if not isinstance(is_subagent, bool):
        raise TypeError("is_subagent must be a boolean")
    _is_subagent = is_subagent

    if client is None:
        client = ArgoClient.with_tier("L0")

    # Build an argo_fn wrapper for the compactor (injects call semantics)
    def _argo_fn(prompt, system, stop=None):
        return client.call(prompt, system, stop=stop)

    _cfg = get_config()
    log.info(
        "Data-grounding policy: profile=%s active_for_this_tool_tree=%s",
        _grounding_profile_metadata()["name"], _gate_enabled(tools),
    )
    _max_iter = _cfg.MAX_TOOL_ITERATIONS if max_iterations is None else max_iterations
    _timeout = _cfg.MAX_TURN_SECONDS if timeout is None else timeout
    if isinstance(_max_iter, bool) or not isinstance(_max_iter, int) or _max_iter <= 0:
        raise TypeError("max_iterations must be a positive integer")
    if isinstance(_timeout, bool) or not isinstance(_timeout, int) or _timeout <= 0:
        raise TypeError("timeout must be a positive integer")
    if compaction_trigger_chars is not None and (
        isinstance(compaction_trigger_chars, bool)
        or not isinstance(compaction_trigger_chars, int)
        or compaction_trigger_chars <= 0
    ):
        raise TypeError("compaction_trigger_chars must be a positive integer")
    if compaction_interval is not None and (
        isinstance(compaction_interval, bool)
        or not isinstance(compaction_interval, int)
        or compaction_interval <= 0
    ):
        raise TypeError("compaction_interval must be a positive integer")
    if required_tools is not None:
        if not isinstance(required_tools, set) or any(
            not isinstance(name, str) or not name for name in required_tools
        ):
            raise TypeError("required_tools must be a set of non-empty tool names")
        unknown_required = required_tools.difference(tools)
        if unknown_required:
            raise ValueError(f"required_tools are not registered: {sorted(unknown_required)}")

    memory.append({"role": "user", "content": user_message})
    _anchor(mem_anchor.SYNC_USER_MESSAGE_APPEND, hooks,
            iteration=0, content=user_message)
    t0 = time.time()
    _time_tracker = _anchor(
        ctx_anchor.SYNC_TIME_BUDGET_TRACKER_CREATE,
        hooks,
        timeout=_timeout,
        thresholds=_cfg.WARN_THRESHOLDS,
        max_warnings=_cfg.MAX_WRAP_WARNINGS,
    )
    tool_history: list[dict] = []
    consecutive_empty_waits = 0               # stuck-loop detector
    _MAX_EMPTY_WAITS = _cfg.MAX_EMPTY_WAITS     # break after N consecutive empty waits

    # Stage-compaction cache for ordered batches. Cache the already-rendered
    # parent display exactly as shown (without the ordered-batch label). The
    # trusted inspect closure replays it directly: no JSON round-trip and no
    # second renderer pass.
    _batch_cache: dict[int, str] = {}
    _batch_inspect_fn: Callable | None = None
    # Syntax-rejected batches: ledger only — never enters tool_history, so
    # malformed placeholders cannot leak into logs, envelopes, tool counts,
    # or evaluator summaries. Resurfaces only in answer-rejection feedback.
    _syntax_rejections: list[dict] = []
    _last_context: str = ""     # raw LLM input+output for the final turn

    for iteration in range(1, _max_iter + 1):
        elapsed = time.time() - t0

        remaining = _timeout - elapsed

        # ── Time budget warnings & hard stop ────────────────
        _anchor(
            ctx_anchor.SYNC_TIME_BUDGET_WARNINGS_APPLY,
            hooks,
            tracker=_time_tracker,
            elapsed=elapsed,
            memory=memory,
        )

        if _anchor(
            ctx_anchor.SYNC_TIME_BUDGET_HARD_STOP_CHECK,
            hooks,
            tracker=_time_tracker,
            elapsed=elapsed,
        ):
            log.error("Hard time limit reached (%.0fs > %.0fs)", elapsed, _timeout)
            break

        # ── Build flat prompt ───────────────────────────────
        flat_parts: list[str] = []

        # ── 🔗 Anchor: WORKING_MEMORY_RENDER ───────────────
        # Inject persistent working memory (outside memory list)
        wm_chars = 0
        wm = _anchor(mem_anchor.SYNC_WORKING_MEMORY_RENDER, hooks, iteration=iteration)
        if wm:
            wm_part = (
                f"## Working Memory\n"
                f"{TAG_MEMORY_OPEN}\n{wm}\n{TAG_MEMORY_CLOSE}"
            )
            flat_parts.append(wm_part)
            wm_chars = len(wm_part)

        # ── 🔗 Anchor: FLAT_PROMPT_BUILD ───────────────────
        # Append conversation memory — first user turn uses "Subagent"
        # header when this is a delegated subagent call.
        _first_user_seen = False
        user_chars = 0
        asst_chars = 0
        turn_details: list[str] = []
        for turn in memory:
            if turn["role"] == "user" and _is_subagent and not _first_user_seen:
                tag = "Subagent"
                _first_user_seen = True
            elif turn["role"] == "user":
                tag = "User"
            else:
                tag = "Assistant"
            part = f"## {tag}\n{turn['content']}"
            flat_parts.append(part)
            plen = len(part)
            if turn["role"] == "user":
                user_chars += plen
            else:
                asst_chars += plen
            turn_details.append(f"{tag[:1]}:{plen}")

        flat_prompt = "\n\n".join(flat_parts)
        _anchor(ctx_anchor.SYNC_FLAT_PROMPT_BUILD, hooks,
                iteration=iteration, prompt_chars=len(flat_prompt))

        # ── Context breakdown table ─────────────────────────
        log.info(
            "CTX turn=%d | system=%d wm=%d user=%d asst=%d "
            "total_prompt=%d | turns=[%s]",
            iteration, len(system_prompt), wm_chars,
            user_chars, asst_chars, len(flat_prompt),
            ", ".join(turn_details),
        )

        # ── 🔗 Anchor: LLM_CALL_BEFORE ─────────────────────
        _anchor(ctx_anchor.SYNC_LLM_CALL_BEFORE, hooks, iteration=iteration)

        # ── Call LLM ────────────────────────────────────────
        _tagged_system = (
            f"{TAG_SYSTEM_PROMPT_OPEN}\n{system_prompt}\n{TAG_SYSTEM_PROMPT_CLOSE}"
        )
        _turn_activity = _push_history_activity(f"Turn#{iteration}")
        try:
            response = client.call(flat_prompt, _tagged_system,
                                   event_kind="ReAct")
        finally:
            _pop_history_activity(_turn_activity)

        # ── 🔗 Anchor: LLM_RESPONSE_AFTER + ON_REASONING ───
        _anchor(ctx_anchor.SYNC_LLM_RESPONSE_AFTER, hooks, iteration, response)

        # Capture raw context as the LLM saw it
        _last_context = (
            f"## System Prompt\n{system_prompt}\n\n"
            f"{flat_prompt}\n\n"
            f"## LLM Response\n{response}"
        )
        _last_context = _anchor(
            ctx_anchor.SYNC_FINAL_CONTEXT_CAPTURED,
            hooks,
            iteration=iteration,
            final_context=_last_context,
            response=response,
        ) or _last_context

        # ── Extract tool call(s) or final answer ────────────
        batch = extract_all_tool_calls(response)
        all_tool_calls = batch.calls
        _requests_batch_inspection = (
            _batch_inspect_fn is not None
            and tools.get("inspect_batch_result") is _batch_inspect_fn
            and any(
                tc.get("name") == "inspect_batch_result"
                for tc in all_tool_calls
            )
        )

        # ── Atomic wire-syntax gate ─────────────────────────
        # One malformed pre-wait block rejects the whole batch. Previously
        # malformed blocks were silently dropped, allowing valid siblings to
        # execute before the agent corrected the bad call.
        if not batch.syntax_valid:
            correction = batch.correction_message()
            log.warning(
                "Tool batch rejected before execution: %d malformed block(s), "
                "%d otherwise parseable call(s)",
                len(batch.syntax_errors), len(all_tool_calls),
            )
            for error in batch.syntax_errors:
                log.warning("Malformed tool-call detail: %s", error)
            # Archive stripped: narration from rejected turns must not
            # poison later turns with fabricated content.
            _stripped_turn = _strip_rejected_turn(response)
            # Durable inspection record — survives the post-recovery scrub.
            log.warning(
                "Rejected turn %d archive (for inspection):\n%s",
                iteration, _stripped_turn,
            )
            memory.append({
                "role": "assistant",
                "content": _stripped_turn,
            })
            memory.append({
                "role": "user",
                "content": (
                    f"{TAG_VALIDATION_OPEN}\n{correction}\n{TAG_VALIDATION_CLOSE}"
                ),
            })
            _syntax_rejections.append({
                "iteration": iteration,
                "errors": list(batch.syntax_errors),
            })
            for index, error in enumerate(batch.syntax_errors, start=1):
                _anchor(
                    tool_anchor.SYNC_TOOL_VALIDATION_BLOCKED,
                    hooks,
                    iteration=iteration,
                    tool_name=f"<malformed_tool_call_{index}>",
                    status="syntax-blocked",
                    detail=error,
                    args={},
                )
            consecutive_empty_waits = 0
            continue

        # Keep stage-compacted details through the response that explicitly
        # requests them. A malformed response also retains the cache so the
        # agent can correct its inspect call. Any other canonical response
        # closes the temporary inspection window.
        # A canonical response has already consumed any prior one-turn
        # inspection payload. Evict it by explicit state before executing the
        # next request. Consecutive inspections keep the rendered-display
        # cache/tool but
        # replace (rather than accumulate) the temporary parent-visible copy.
        if any(
            item.get("_temporary_stage_inspection", False)
            for item in memory
        ):
            _inspection_cleaned_memory: list[dict] = []
            for item in memory:
                if not item.get("_temporary_stage_inspection", False):
                    _inspection_cleaned_memory.append(item)
                    continue
                # A turn may inspect cached output and run ordinary tools in
                # the same ordered batch. Remove only the one-turn inspection
                # display while retaining those ordinary tools' rendered
                # results. Private cleanup state never reaches later prompts.
                _durable_content = item.get(
                    "_stage_inspection_durable_content", ""
                )
                if _durable_content:
                    _durable_item = dict(item)
                    _durable_item["content"] = _durable_content
                    _durable_item.pop("_temporary_stage_inspection", None)
                    _durable_item.pop(
                        "_stage_inspection_durable_content", None
                    )
                    _inspection_cleaned_memory.append(_durable_item)
            memory[:] = _inspection_cleaned_memory
        if _batch_cache and not _requests_batch_inspection:
            _batch_cache.clear()
            if tools.get("inspect_batch_result") is _batch_inspect_fn:
                tools.pop("inspect_batch_result", None)
            _batch_inspect_fn = None
            log.info("Stage inspect cleanup: cache cleared, tool removed")

        # ── Empty-wait guard: LLM emitted <wait/> with no tools ──
        if batch.wait_detected and not all_tool_calls:
            consecutive_empty_waits += 1
            log.warning("Empty <wait/> #%d (no tool calls before barrier)",
                        consecutive_empty_waits)
            memory.append({"role": "assistant", "content": response})
            if consecutive_empty_waits >= _MAX_EMPTY_WAITS:
                memory.append({
                    "role": "user",
                    "content": (
                        f"[WAIT LOOP BREAK] You have emitted {TAG_WAIT} without "
                        "any tool calls %d times consecutively.  You MUST "
                        f"either emit concrete {TAG_TOOL_CALL_OPEN} blocks or provide "
                        "your final answer NOW."
                    ) % consecutive_empty_waits,
                })
                log.error("Empty-wait stuck loop — forcing answer after %d empty waits",
                          consecutive_empty_waits)
            else:
                memory.append({
                    "role": "user",
                    "content": (
                        f"[EMPTY WAIT] You emitted {TAG_WAIT} but no tool calls "
                        f"preceded it.  Place your {TAG_TOOL_CALL_OPEN} blocks BEFORE "
                        f"{TAG_WAIT}, not after.  Re-emit the tool calls you need."
                    ),
                })
            continue

        if not all_tool_calls:
            # ── Answer-marker guard ─────────────────────────
            # A response with NO tool calls and NO <answer> block is
            # almost always a token-cap truncation mid-"thinking out
            # loud" (planning monologue cut mid-sentence).  Accepting
            # it silently ships a broken answer.  Nudge the agent to
            # emit the real final answer (bounded retries).
            if (TAG_ANSWER_OPEN not in response
                    and not batch.wait_detected):
                _no_marker_nudges = sum(
                    1 for m in memory[-6:]
                    if m.get("role") == "user"
                    and "[NO ANSWER MARKER]" in str(m.get("content", ""))
                )
                if _no_marker_nudges < 2:
                    log.warning(
                        "Response has no tool calls and no %s block "
                        "(likely token-cap truncation) — requesting the "
                        "final answer explicitly (nudge %d/2).",
                        TAG_ANSWER_OPEN, _no_marker_nudges + 1,
                    )
                    memory.append({"role": "assistant", "content": response})
                    memory.append({
                        "role": "user",
                        "content": (
                            "[NO ANSWER MARKER] Your previous message "
                            "contained neither tool calls nor a final "
                            f"answer wrapped in {TAG_ANSWER_OPEN}"
                            f"{TAG_ANSWER_CLOSE} tags — it appears to be "
                            "an unfinished reasoning monologue (possibly "
                            "cut off). Do NOT continue planning. Emit "
                            "your COMPLETE final answer NOW, wrapped in "
                            f"{TAG_ANSWER_OPEN}...{TAG_ANSWER_CLOSE}, "
                            "using the results you already have. Keep "
                            "reasoning outside the answer to a minimum."
                        ),
                    })
                    continue
                raise RuntimeError(
                    f"AGENT_PROTOCOL_ERROR: response still omitted required "
                    f"{TAG_ANSWER_OPEN}...{TAG_ANSWER_CLOSE} after two retries"
                )

            # ── Soft fitting reminder ───────────────────────
            # If required_tools were specified but none were called,
            # append a note so the reader knows no actual fitting
            # was performed.  Does NOT block the answer — some
            # queries (data coverage, discovery) don't need fitting.
            tools_called = {h.get("tool") for h in tool_history}
            if required_tools and not (tools_called & required_tools):
                _missing = ", ".join(sorted(required_tools - tools_called))
                response += (
                    "\n\n---\n*Note: No fitting tools were called during "
                    "this run (%s). All numbers above come from database "
                    "queries only — no Redlich-Kister or other curve fits "
                    "were performed.*" % _missing
                )
                log.info("Fitting reminder appended (none of %s called)", required_tools)
                _anchor(
                    ctx_anchor.SYNC_REQUIRED_TOOLS_NOTE_APPEND,
                    hooks,
                    iteration=iteration,
                    required_tools=sorted(required_tools),
                    missing_tools=sorted(required_tools - tools_called),
                )

            # ── 🔗 Anchor: FINAL_ANSWER_BEFORE ─────────────
            _anchor(ctx_anchor.SYNC_FINAL_ANSWER_BEFORE, hooks,
                    iteration=iteration, answer=response)

            # Strip reasoning/summary blocks and leftover marker tags
            _clean_answer = _clean_answer_text(response)

            # ── Answer continuity after a grounding bounce ──
            # The bounce asks for a surgical repair, but the next final
            # answer REPLACES the bounced one; a stub ("No changes.") would
            # ship in its place. Ask once for the complete answer.
            if _gate_enabled(tools):
                _continuity_nudge = _build_continuity_nudge(
                    memory, _clean_answer, clean=_clean_answer_text,
                )
                if _continuity_nudge:
                    log.warning(
                        "Final answer after a grounding bounce is a stub "
                        "(%d chars) — asking once for the complete answer",
                        len(_clean_answer.strip()),
                    )
                    memory.append({"role": "assistant", "content": response})
                    memory.append(
                        {"role": "user", "content": _continuity_nudge})
                    continue

            # ── Mandatory data-grounding gate ──────────────
            # Every block-anchored data literal must be grounded in a
            # verbatim inspect_block_table result from this run. Bounded
            # bounces re-engage the agent with concrete repair calls;
            # exhausted bounces ship deterministic UNVERIFIED flags.
            if _gate_enabled(tools):
                _gate_report = _run_data_grounding_gate(
                    _clean_answer, tool_history,
                )
                if not _gate_report.ok:
                    _gate_bounces = _count_gate_bounces(memory)
                    if _gate_bounces < _MAX_GATE_BOUNCES:
                        log.warning(
                            "Data-grounding gate: %d violation(s) — "
                            "bounce %d/%d\n%s",
                            len(_gate_report.violations),
                            _gate_bounces + 1, _MAX_GATE_BOUNCES,
                            _gate_report.report_text,
                        )
                        memory.append(
                            {"role": "assistant", "content": response})
                        memory.append({
                            "role": "user",
                            "content": (
                                _build_gate_nudge(_gate_report)
                                + _build_syntax_rejection_note(_syntax_rejections)
                            ),
                        })
                        continue
                    log.error(
                        "Data-grounding gate: %d violation(s) persist after "
                        "%d bounces — shipping with UNVERIFIED flags\n%s",
                        len(_gate_report.violations), _MAX_GATE_BOUNCES,
                        _gate_report.report_text,
                    )
                    _clean_answer += _build_flags_note(_gate_report)

            # Final answer
            memory.append({"role": "assistant", "content": response})
            _anchor(ctx_anchor.SYNC_FINAL_ANSWER_AFTER, hooks,
                    iteration=iteration, answer=_clean_answer,
                    tool_history=tool_history)
            _log_answer_event(_clean_answer)
            return AgentTurnResult(
                answer=_clean_answer,
                iterations=iteration,
                elapsed_seconds=time.time() - t0,
                tool_history=tool_history,
                final_context=_last_context,
            )

        # ── Force final answer after last warning ───────────
        if _anchor(ctx_anchor.SYNC_TIME_BUDGET_ALL_WARNINGS_CHECK, hooks, tracker=_time_tracker):
            memory.append({"role": "assistant", "content": response})
            memory.append({
                "role": "user",
                "content": _anchor(
                    ctx_anchor.SYNC_TIME_BUDGET_FINAL_MESSAGE_BUILD,
                    hooks,
                    tracker=_time_tracker,
                ),
            })
            # One more LLM call without tools
            _turn_activity = _push_history_activity(f"Turn#{iteration}")
            try:
                final_response = client.call(
                    "\n\n".join(
                        f"## {'User' if t['role'] == 'user' else 'Assistant'}\n{t['content']}"
                        for t in memory
                    ),
                    system_prompt,
                    event_kind="ReAct-forced-final",
                )
            finally:
                _pop_history_activity(_turn_activity)
            _last_context = (
                f"## System Prompt\n{system_prompt}\n\n"
                + "\n\n".join(
                    f"## {'User' if t['role'] == 'user' else 'Assistant'}\n{t['content']}"
                    for t in memory
                )
                + f"\n\n## LLM Response\n{final_response}"
            )
            _last_context = _anchor(
                ctx_anchor.SYNC_FINAL_CONTEXT_CAPTURED,
                hooks,
                iteration=iteration,
                final_context=_last_context,
                response=final_response,
                timed_out=True,
            ) or _last_context
            memory.append({"role": "assistant", "content": final_response})
            _clean_final = _clean_answer_text(final_response)
            _clean_final = _apply_forced_final_grounding_gate(
                _clean_final, tools, tool_history, memory, client,
                system_prompt, syntax_rejections=_syntax_rejections,
            )
            _anchor(ctx_anchor.SYNC_WARNING_FINAL_ANSWER_AFTER, hooks,
                    iteration=iteration, answer=_clean_final, timed_out=True,
                    tool_history=tool_history)
            _log_answer_event(_clean_final, kind="answer-forced-final")
            return AgentTurnResult(
                answer=_clean_final,
                iterations=iteration,
                elapsed_seconds=time.time() - t0,
                tool_history=tool_history,
                timed_out=True,
                final_context=_last_context,
            )

        # ── Execute tool(s) — ordered batch support ─────────
        # If the LLM emitted multiple <tool_call> blocks, execute
        # all of them in emitted order and bundle results into one
        # <tool_result>.
        # This counts as ONE turn for compaction purposes.

        # Extract reasoning text before the first <tool_call>
        tc_idx_first = response.find(TAG_TOOL_CALL_OPEN)
        _reasoning_text = response[:tc_idx_first].strip() if tc_idx_first > 0 else ""

        result_parts: list[str] = []
        result_displays: list[str] = []
        durable_result_parts: list[str] = []

        # ── Pre-execution validation gate ───────────────────
        # First run the engine's universal exact-name/type preflight. Then run
        # the catalog hook for domain guidance and safe auto-filled fields.
        # Either gate rejects the entire batch before any tool executes.
        _pre_validate = _prevalidate_batch_arguments(all_tool_calls, tools)
        if _pre_validate is None:
            _pre_validate = _anchor(
                tool_anchor.SYNC_BATCH_PRE_VALIDATE,
                hooks,
                tool_calls=all_tool_calls,
                tools=tools,
                engine_hooks=hooks,
            )
        if _pre_validate is None:
            log.debug(
                "Batch pre-validation: %d tool(s) passed",
                len(all_tool_calls),
            )
        if _pre_validate is not None:
            # Tag the guidance so it can be cleaned up later
            combined_result = (
                f"{TAG_VALIDATION_OPEN}\n{_pre_validate}\n{TAG_VALIDATION_CLOSE}"
            )
            log.info(
                "Batch pre-validation: guidance emitted for %d tool(s), "
                "skipping execution",
                len(all_tool_calls),
            )

            # Record in tool_history so the browser terminal sees it.
            # Tag each entry with its individual validation status so
            # the agent (and the human) can see which tool was blocked
            # vs. which was just held back by the batch.
            _blocked_names = {
                line.split("**")[1].split("\u2717 ")[-1].strip()
                for line in _pre_validate.split("\n")
                if "\u2717 " in line
            }
            for tc in all_tool_calls:
                _tn = tc.get("name", "?")
                _status = (
                    "blocked" if _tn in _blocked_names
                    else "held-back-by-batch"
                )
                # Extract per-tool detail from the report (section between this
                # tool's header and the next header / end of report).
                _per_tool_detail = ""
                _marker = f"**\u2717 {_tn}**"
                _idx_start = _pre_validate.find(_marker)
                if _idx_start >= 0:
                    _idx_end = _pre_validate.find("\n\n**", _idx_start + len(_marker))
                    _per_tool_detail = (
                        _pre_validate[_idx_start:_idx_end].strip()
                        if _idx_end > 0
                        else _pre_validate[_idx_start:].strip()
                    )

                tool_history.append({
                    "iteration": iteration,
                    "tool": _tn,
                    "args_keys": sorted(tc.get("arguments", {}).keys()),
                    "arguments": tc.get("arguments", {}),
                    "result_chars": len(_pre_validate),
                    "result_full": f"[validation {_status}] {_per_tool_detail or _pre_validate[:300]}",
                    "reasoning": _reasoning_text if tc is all_tool_calls[0] else "",
                    "elapsed_s": 0.0,
                })
                _anchor(tool_anchor.SYNC_TOOL_VALIDATION_BLOCKED, hooks,
                        iteration=iteration, tool_name=_tn,
                        status=_status,
                        detail=_per_tool_detail or _pre_validate[:500],
                        args=tc.get("arguments", {}))

            # Skip the entire execution loop — jump to memory injection.
            # We still need to record the assistant message + tool_result.
            compact_tc = _anchor(
                tool_anchor.SYNC_TOOL_CALLS_MEMORY_COMPACT,
                hooks,
                tool_calls=all_tool_calls,
                deferred_count=batch.deferred_count,
            )
            if compact_tc is None:
                compact_tc = _compact_tool_calls_for_memory(
                    all_tool_calls, batch.deferred_count,
                )

            kept_prefix = ""
            if tc_idx_first > 0:
                pre_tc = response[:tc_idx_first]
                sm = _SUMMARY_BLOCK_RE.search(pre_tc)
                summary_text = sm.group(1).strip() if sm else ""
                if len(summary_text) > _cfg.SUMMARY_MAX_CHARS:
                    summary_text = summary_text[:_cfg.SUMMARY_MAX_CHARS] + "…"
                _preserved_tags = []
                for _tag_pattern in (_COMPACT_NOTE_RE, _COMPRESS_TAG_RE):
                    for _tm in _tag_pattern.finditer(pre_tc):
                        _preserved_tags.append(_tm.group())
                kept_parts = []
                if summary_text:
                    kept_parts.append(f"{TAG_SUMMARY_OPEN}{summary_text}{TAG_SUMMARY_CLOSE}")
                for _tag in _preserved_tags:
                    kept_parts.append(_tag)
                kept_prefix = "\n".join(kept_parts) if kept_parts else ""

            response = (kept_prefix + "\n" + compact_tc) if kept_prefix else compact_tc
            memory.append({"role": "assistant", "content": response})
            _anchor(mem_anchor.SYNC_ASSISTANT_MESSAGE_APPEND, hooks,
                    iteration=iteration, content=response)
            memory.append({
                "role": "user",
                "content": f"{TAG_TOOL_RESULT_OPEN}\n{combined_result}\n{TAG_TOOL_RESULT_CLOSE}",
            })
            _anchor(tool_anchor.SYNC_TOOL_RESULT_INJECT, hooks,
                    iteration=iteration, result_chars=len(combined_result))
            consecutive_empty_waits = 0
            continue

        # ── Validation passed — clean old guidance traces ───────
        # Replace verbose <validation_guidance> blocks in earlier
        # memory entries with a compact note so context stays lean.
        _clean_validation_traces(memory)
        # A valid batch is about to execute — the correction succeeded, so
        # collapse archived syntax-rejected turns to a one-line stub.
        _clean_syntax_rejection_traces(memory)

        for tc in all_tool_calls:
            tool_name = tc.get("name", "")
            raw_args = tc.get("arguments", {})
            fn = tools.get(tool_name)
            _is_batch_inspection_call = (
                tool_name == "inspect_batch_result"
                and fn is not None
                and fn is _batch_inspect_fn
            )
            _is_trusted_cached_display = _is_batch_inspection_call

            tool_t0 = time.time()
            _tool_event = _begin_tool_event(tool_name)

            if fn is None:
                native_result: dict | str = {
                    "error": f"Unknown tool: {tool_name}",
                    "error_code": "UNKNOWN_TOOL",
                    "available_tools": sorted(tools.keys()),
                }
            else:
                try:
                    args = _validate_tool_arguments(tool_name, raw_args, fn)
                    if _is_trusted_cached_display:
                        native_result = fn(**args)
                        if (
                            not isinstance(native_result, str)
                            or not native_result.strip()
                        ):
                            raise TypeError(
                                "trusted batch inspection must return "
                                "non-empty cached Markdown"
                            )
                        one_result = native_result
                    else:
                        native_result = validate_native_tool_result(
                            tool_name, fn(**args)
                        )
                        if isinstance(native_result, dict):
                            one_result = json.dumps(native_result, indent=2)
                            if len(one_result) > 2000:
                                log.debug(
                                    "Tool %s returned a raw dict (%d chars) — "
                                    "skip-subagent tool (normal for L2)",
                                    tool_name, len(one_result),
                                )
                        else:
                            one_result = native_result
                except Exception as e:
                    # A failed trusted inspection is now an ordinary
                    # structured tool error, not a cached display replay.
                    _is_trusted_cached_display = False
                    _err_text = str(e)
                    if "TOOL_ARGUMENT_REFINEMENT_REQUIRED" in _err_text:
                        # agent-input error, not a system failure: relay the
                        # refinement message without traceback noise
                        log.warning("Tool %s argument refinement required: %s",
                                    tool_name, _err_text)
                        _err_code = "TOOL_ARGUMENT_REFINEMENT_REQUIRED"
                    else:
                        log.error("Tool %s execution failed: %s",
                                  tool_name, e, exc_info=True)
                        _err_code = "TOOL_EXECUTION_ERROR"
                    native_result = {
                        "error": _err_text,
                        "error_code": _err_code,
                        "tool": tool_name,
                    }
                    one_result = json.dumps(native_result, indent=2)

            if fn is None:
                one_result = json.dumps(native_result, indent=2)

            if (
                not _is_trusted_cached_display
                and
                fn is not None
                and getattr(fn, "_returns_subagent_answer", False)
                and not (
                    isinstance(native_result, dict)
                    and "error_code" in native_result
                )
            ):
                one_result = wrap_subagent_answer(one_result)

            tool_elapsed = time.time() - tool_t0
            if not _is_trusted_cached_display:
                one_result = require_result_within_limit(
                    tool_name,
                    one_result,
                )

            _history_entry = {
                "iteration": iteration,
                "tool": tool_name,
                "args_keys": sorted(raw_args.keys()) if raw_args else [],
                "arguments": raw_args,
                "result_chars": len(one_result),
                "result_full": one_result,
                "reasoning": _reasoning_text if tc is all_tool_calls[0] else "",
                "elapsed_s": round(tool_elapsed, 1),
            }
            if _is_trusted_cached_display:
                _history_entry.update({
                    "result_format": "cached_markdown_display",
                    "result_source": "ordered_batch_stage_cache",
                    "source_tool_number": raw_args.get("tool_number"),
                })
            tool_history.append(_history_entry)

            # ── 🔗 Anchor: ON_TOOL_RESULT (stats) ──────────────
            _anchor(tool_anchor.SYNC_TOOL_RESULT_RECORDED, hooks,
                  iteration=iteration, tool_name=tool_name,
                  args=raw_args or {}, raw_result_chars=len(one_result),
                  elapsed_s=round(tool_elapsed, 1))

            # ── 🔗 Anchor: AGENT_RECORD_REFERENCES (entity/DOI) ──
            # Native dict results are hard-recorded before any textual compaction.
            if isinstance(native_result, dict):
                _anchor(_AGENT_RECORD_REFERENCES, hooks,
                        tool_name=tool_name, raw_result=native_result)

            _is_json_err = isinstance(native_result, dict) and "error_code" in native_result
            _is_purpose_err = one_result.startswith("ERROR: Tool") and "purpose" in one_result[:200]
            if _is_purpose_err:
                # ── 🔗 Anchor: ON_PURPOSE_ERROR ────────────
                    _anchor(tool_anchor.SYNC_TOOL_PURPOSE_ERROR_RECORDED, hooks,
                      iteration=iteration, tool_name=tool_name,
                      error_text=one_result)
            elif _is_json_err:
                # ── 🔗 Anchor: ON_TOOL_ERROR ───────────────
                    _anchor(tool_anchor.SYNC_TOOL_ERROR_RECORDED, hooks,
                      iteration=iteration, tool_name=tool_name,
                      error_text=one_result, args=raw_args)
            else:
                # ── 🔗 Anchor: ON_TOOL_CALL ────────────────
                    _anchor(tool_anchor.SYNC_TOOL_CALL_RECORDED, hooks,
                      iteration=iteration, tool_name=tool_name,
                      args=raw_args or {}, result_chars=len(one_result),
                      elapsed_s=round(tool_elapsed, 1),
                      result_preview=one_result[:500],
                      result_full=one_result)

            _end_tool_event(
                _tool_event, result_chars=len(one_result),
                error=(one_result[:300]
                       if (_is_json_err or _is_purpose_err) else ""))

            log.info(
                "🔧 %s(%s) → %d chars in %.1fs",
                tool_name,
                ", ".join(f"{k}=..." for k in sorted(raw_args.keys())[:3]),
                len(one_result),
                tool_elapsed,
            )

            # The actual registered inspect closure is the sole trusted
            # cached-display path. It bypasses JSON validation, subagent
            # evidence instrumentation, and rendering; arbitrary tools named
            # inspect_batch_result do not receive this treatment.
            if _is_trusted_cached_display:
                display_result = one_result
            else:
                display_result = build_subagent_context_display(
                    tool_name, fn, native_result, one_result, tool_history)
            result_displays.append(display_result)
            if len(all_tool_calls) > 1:
                _display_part = f"### {tool_name}\n{display_result}"
            else:
                _display_part = display_result
            result_parts.append(_display_part)
            if not _is_batch_inspection_call:
                durable_result_parts.append(_display_part)

        combined_result = "\n\n".join(result_parts)

        # ── Batch summary hook (menu tool batches) ──────────
        # When a batch_summary_hook is provided, give it first crack
        # at summarising the batch.  If it returns a string, use that
        # and skip the generic stage compaction.
        _hook_handled = False
        if len(all_tool_calls) > 1 and not _requests_batch_inspection:
            try:
                _hook_result = _anchor(
                    tool_anchor.SYNC_BATCH_SUMMARY_BUILD,
                    hooks,
                    tool_calls=all_tool_calls,
                    result_parts=result_parts,
                )
                if _hook_result is not None:
                    orig_chars = len(combined_result)
                    combined_result = _hook_result
                    _hook_handled = True
                    log.info(
                        "Batch summary hook: %d tools, %d → %d chars",
                        len(all_tool_calls), orig_chars, len(combined_result),
                    )
            except Exception as exc:
                log.warning("Batch summary hook failed: %s — falling through", exc, exc_info=True)

        # ── Stage compaction for ordered batches ────────────
        # When a multi-tool batch produces a large combined result,
        # replace it with a compact summary table.  The full results
        # are cached so the agent can inspect specific ones on demand.
        if (not _hook_handled
                and not _requests_batch_inspection
                and len(all_tool_calls) > 1
                and len(combined_result) > _cfg.STAGE_COMPACT_BUDGET):
            if len(result_displays) != len(result_parts):
                raise RuntimeError(
                    "ordered-batch display result count mismatch"
                )
            _batch_cache.clear()
            for idx, display in enumerate(result_displays, 1):
                _batch_cache[idx] = display

            orig_chars = len(combined_result)
            _stage_result = _anchor(
                tool_anchor.SYNC_STAGE_COMPACTION_BUILD,
                hooks,
                tool_calls=all_tool_calls,
                result_parts=result_parts,
                note_chars=_cfg.STAGE_NOTE_CHARS,
            )
            combined_result = (
                _stage_result
                if _stage_result is not None
                else _stage_compact_batch(
                    all_tool_calls,
                    result_parts,
                    note_chars=_cfg.STAGE_NOTE_CHARS,
                )
            )
            if not isinstance(combined_result, str):
                raise TypeError(
                    "stage-compaction hook must return a string or None"
                )
            combined_result += (
                "\n*Use `inspect_batch_result(tool_number)` to read "
                "the full output of any tool above before proceeding.*"
            )

            # Register the temporary inspect tool
            def inspect_batch_result(tool_number: int) -> str:
                """Read the full result of a specific tool from the last
                ordered batch. Use the tool_number from the summary table.

                Parameters
                ----------
                tool_number : int
                    1-based index from the batch summary table.
                """
                cached = _batch_cache.get(int(tool_number))
                if cached:
                    return cached
                valid = sorted(_batch_cache.keys())
                return f"No result #{tool_number}. Valid numbers: {valid}"
            _batch_inspect_fn = inspect_batch_result
            tools["inspect_batch_result"] = inspect_batch_result

            log.info(
                "Stage compaction: %d tools, %d → %d chars. "
                "inspect_batch_result registered.",
                len(all_tool_calls), orig_chars, len(combined_result),
            )

            # ── 🔗 Anchor: ON_STAGE_COMPACT ────────────────
            _anchor(tool_anchor.SYNC_STAGE_COMPACTION_RECORDED, hooks,
                  iteration=iteration, n_tools=len(all_tool_calls),
                  before_chars=orig_chars, after_chars=len(combined_result))

        # Reset empty-wait counter — we successfully executed tools
        consecutive_empty_waits = 0

        batch_elapsed = time.time() - tool_t0 if len(all_tool_calls) == 1 else (
            time.time() - (t0 + elapsed)  # approximate batch wall time
        )

        if len(all_tool_calls) > 1:
            log.info("Ordered batch: %d tools executed in turn %d",
                     len(all_tool_calls), iteration)

        # ── Wait barrier feedback ───────────────────────────
        # When <wait/> was detected, append a status note so the LLM
        # knows the pre-wait tools are done and it should re-plan.
        if batch.wait_detected:
            tool_names = [tc.get("name", "?") for tc in all_tool_calls]
            wait_note = (
                f"\n\n[WAIT BARRIER COMPLETE] {len(all_tool_calls)} tool(s) "
                f"finished ({', '.join(tool_names)}). "
                f"{batch.deferred_count} deferred call(s) were discarded. "
                f"Use the results above to plan your next tool calls with "
                f"correct IDs/DOIs — do NOT guess."
            )
            combined_result += wait_note
            log.info("Wait barrier complete: %d executed, %d deferred",
                     len(all_tool_calls), batch.deferred_count)

        # ── Append to conversation memory ───────────────────
        # Hardcode-strip <reasoning> blocks entirely and replace
        # verbose <tool_call> XML with compact name(key_args) form.
        compact_tc = _anchor(
            tool_anchor.SYNC_TOOL_CALLS_MEMORY_COMPACT,
            hooks,
            tool_calls=all_tool_calls,
            deferred_count=batch.deferred_count,
        )
        if compact_tc is None:
            compact_tc = _compact_tool_calls_for_memory(
                all_tool_calls, batch.deferred_count,
            )

        kept_prefix = ""
        if tc_idx_first > 0:
            pre_tc = response[:tc_idx_first]
            # Extract <summary> block if present
            sm = _SUMMARY_BLOCK_RE.search(pre_tc)
            summary_text = sm.group(1).strip() if sm else ""
            if len(summary_text) > _cfg.SUMMARY_MAX_CHARS:
                summary_text = summary_text[:_cfg.SUMMARY_MAX_CHARS] + "…"
            # Extract compaction tags
            _preserved_tags = []
            for _tag_pattern in (
                _COMPACT_NOTE_RE,
                _COMPRESS_TAG_RE,
            ):
                for _tm in _tag_pattern.finditer(pre_tc):
                    _preserved_tags.append(_tm.group())
            # Rebuild: summary + tags only (reasoning discarded)
            kept_parts = []
            if summary_text:
                kept_parts.append(f"{TAG_SUMMARY_OPEN}{summary_text}{TAG_SUMMARY_CLOSE}")
            for _tag in _preserved_tags:
                kept_parts.append(_tag)
            kept_prefix = "\n".join(kept_parts) if kept_parts else ""

        response = (kept_prefix + "\n" + compact_tc) if kept_prefix else compact_tc
        # ── 🔗 Anchor: MEMORY_APPEND_ASSISTANT ─────────────
        memory.append({"role": "assistant", "content": response})
        _anchor(mem_anchor.SYNC_ASSISTANT_MESSAGE_APPEND, hooks,
                iteration=iteration, content=response)
        # ── 🔗 Anchor: TOOL_RESULT_INJECT ──────────────────
        _result_memory_entry = {
            "role": "user",
            "content": f"{TAG_TOOL_RESULT_OPEN}\n{combined_result}\n{TAG_TOOL_RESULT_CLOSE}",
        }
        if _requests_batch_inspection:
            # Explicit state replaces the former rendered-heading search. The
            # next canonical non-inspection response evicts this potentially
            # large one-turn payload before continuing. If the same ordered
            # batch also ran normal tools, retain only their rendered output.
            _result_memory_entry["_temporary_stage_inspection"] = True
            if durable_result_parts:
                _durable_result = "\n\n".join(durable_result_parts)
                _result_memory_entry[
                    "_stage_inspection_durable_content"
                ] = (
                    f"{TAG_TOOL_RESULT_OPEN}\n{_durable_result}\n"
                    f"{TAG_TOOL_RESULT_CLOSE}"
                )
        memory.append(_result_memory_entry)
        _anchor(tool_anchor.SYNC_TOOL_RESULT_INJECT, hooks,
                iteration=iteration, result_chars=len(combined_result))

        # ── LLM-driven context compaction ────────────────────────
        # Two triggers (either can fire):
        #   1. Interval-based: every N turns (compaction_interval)
        #   2. Size-based: when total context exceeds char threshold
        # The LLM selection step can still choose "NONE" to skip.
        # Note: iteration counts LLM turns, not individual tool calls.
        total_chars = sum(len(m["content"]) for m in memory)
        _trigger = (
            _get_compaction_trigger_chars()
            if compaction_trigger_chars is None
            else compaction_trigger_chars
        )
        _interval_hit = bool(
            compaction_interval is not None
            and iteration >= compaction_interval
            and iteration % compaction_interval == 0
        )
        _size_hit = total_chars > _trigger
        if _requests_batch_inspection:
            # The full inspected result is intentionally a one-turn view.
            # Do not let interval/size compaction rewrite it into durable
            # memory before the explicit eviction on the next response.
            _interval_hit = False
            _size_hit = False

        if _interval_hit or _size_hit:
            reason = (f"interval={compaction_interval}" if _interval_hit
                      else f"chars={total_chars}>{_trigger}")
            log.info("Compaction triggered (%s, turn %d) — requesting guidance",
                     reason, iteration)

            # ── Guidance hook: ask the main agent for purpose/tasks ──
            purpose = ""
            tasks = ""
            reminder_msg = _anchor(
                ctx_anchor.SYNC_COMPACTION_REMINDER_BUILD,
                hooks,
                total_chars=total_chars,
                iteration=iteration,
            )
            if reminder_msg is None:
                reminder_msg = _build_compaction_reminder(total_chars, iteration)
            if not isinstance(reminder_msg, str) or not reminder_msg:
                raise RuntimeError("Compaction reminder must be a non-empty string")
            memory.append({"role": "user", "content": reminder_msg})
            try:
                _guidance_parts = []
                _wm = _anchor(mem_anchor.SYNC_WORKING_MEMORY_RENDER, hooks, iteration=iteration)
                if _wm:
                    _guidance_parts.append(f"## Working Memory\n{_wm}")
                for _t in memory:
                    _tag = "User" if _t["role"] == "user" else "Assistant"
                    _guidance_parts.append(f"## {_tag}\n{_t['content']}")
                _guidance_prompt = "\n\n".join(_guidance_parts)
                guidance = None
                _parse_error: Exception | None = None
                # One malformed guidance reply must not kill the run: retry
                # once with a format reminder, then degrade to SKIP (the
                # trigger re-fires on the next interval/size check).
                for _attempt in (1, 2):
                    _guidance_resp = client.call(
                        _guidance_prompt if _attempt == 1 else (
                            _guidance_prompt
                            + "\n\n### FORMAT CORRECTION REQUIRED\nYour previous "
                            f"reply was rejected: {_parse_error}. Reply again with "
                            "the required <compaction_guidance> block exactly."
                        ),
                        system_prompt,
                        max_tokens=_cfg.GUIDANCE_MAX_TOKENS,
                        event_kind="compaction-guidance",
                    )
                    _anchor(
                        ctx_anchor.SYNC_COMPACTION_GUIDANCE_RESPONSE,
                        hooks,
                        iteration,
                        _guidance_resp,
                        source="compaction-guidance",
                    )
                    try:
                        guidance = _anchor(
                            ctx_anchor.SYNC_COMPACTION_GUIDANCE_PARSE,
                            hooks,
                            guidance_response=_guidance_resp,
                        )
                        if guidance is None:
                            guidance = _parse_compaction_guidance(_guidance_resp)
                        break
                    except ValueError as parse_exc:
                        _parse_error = parse_exc
                        log.warning(
                            "Compaction-guidance reply malformed (attempt %d/2): %s",
                            _attempt, parse_exc,
                        )
                if guidance is None:
                    log.warning(
                        "Compaction guidance unusable after retry — skipping "
                        "this compaction round"
                    )
                    guidance = ("guidance reply malformed twice", "SKIP")
                if (
                    not isinstance(guidance, tuple)
                    or len(guidance) != 2
                    or not all(isinstance(item, str) and item for item in guidance)
                ):
                    raise RuntimeError("Compaction-guidance hook returned an invalid contract")
                purpose, tasks = guidance
                if tasks.upper().strip() == "SKIP":
                    log.info("Agent requested SKIP compaction this round")
                    memory.pop()  # remove reminder before continue
                    _anchor(ctx_anchor.SYNC_COMPACTION_SKIPPED, hooks,
                          iteration=iteration, before_chars=total_chars,
                          after_chars=total_chars, trigger=reason,
                          purpose=purpose, tasks="SKIP",
                          outcome="skipped_by_agent")
                    continue  # skip compaction entirely
                log.info("Compaction guidance — PURPOSE: %s | TASKS: %s",
                         purpose[:80], tasks[:80])
            except Exception as e:
                raise RuntimeError("Mandatory compaction-guidance cycle failed") from e
            finally:
                # Always strip the reminder from memory
                if memory and memory[-1]["role"] == "user" \
                        and TAG_COMPACTION_REMINDER_OPEN in memory[-1]["content"]:
                    memory.pop()

            try:
                _receipt = _anchor(
                    ctx_anchor.SYNC_COMPACTION_EXECUTE,
                    hooks,
                    memory=memory,
                    argo_fn=_argo_fn,
                    purpose=purpose,
                    tasks=tasks,
                )
                if _receipt is None:
                    _receipt = _compact_memory(
                        memory,
                        _argo_fn,
                        cfg=_cfg,
                        purpose=purpose,
                        tasks=tasks,
                    )
                if not isinstance(_receipt, str):
                    raise RuntimeError("Context-compaction hook returned an invalid contract")
                new_total = sum(len(m["content"]) for m in memory)
                _outcome = "compacted" if _receipt else "skipped_by_selector"
                log.info("Compaction done: %d → %d chars (%s)",
                         total_chars, new_total, _outcome)
                if _receipt:
                    memory.append({
                        "role": "user",
                        "content": f"{TAG_COMPACT_NOTE_OPEN}{_receipt}{TAG_COMPACT_NOTE_CLOSE}",
                    })
                    log.info("Compaction receipt injected: %s", _receipt[:120])
                # ── 🔗 Anchor: ON_COMPACTION_STATS + ON_COMPACTION ─
                _anchor(ctx_anchor.SYNC_COMPACTION_STATS_RECORDED, hooks,
                      before_chars=total_chars, after_chars=new_total,
                      trigger=reason, outcome=_outcome)
                _anchor(ctx_anchor.SYNC_COMPACTION_RECORDED, hooks,
                      iteration=iteration, before_chars=total_chars,
                      after_chars=new_total, trigger=reason,
                      purpose=purpose, tasks=tasks,
                      receipt=_receipt, outcome=_outcome)
            except Exception as e:
                raise RuntimeError(
                    "Mandatory context compaction failed; context was not trimmed "
                    "or rewritten by a compatibility path"
                ) from e

    # Loop exhausted — force one final LLM call for a useful summary
    # instead of the static placeholder.
    memory.append({
        "role": "user",
        "content": (
            "[HARD STOP] Iteration/time limit reached. "
            "Summarise ALL findings from tool results so far into a "
            "complete answer. Do NOT call any more tools."
        ),
    })
    try:
        final_response = client.call(
            "\n\n".join(
                f"## {'User' if t['role'] == 'user' else 'Assistant'}\n{t['content']}"
                for t in memory
            ),
            system_prompt,
            event_kind="ReAct-hard-stop",
        )
    except Exception as exc:
        raise RuntimeError("Final max-iteration summary call failed") from exc
    _last_context = (
        f"## System Prompt\n{system_prompt}\n\n"
        + "\n\n".join(
            f"## {'User' if t['role'] == 'user' else 'Assistant'}\n{t['content']}"
            for t in memory
        )
        + f"\n\n## LLM Response\n{final_response}"
    )
    _last_context = _anchor(
        ctx_anchor.SYNC_FINAL_CONTEXT_CAPTURED,
        hooks,
        iteration=_max_iter,
        final_context=_last_context,
        response=final_response,
        timed_out=True,
    ) or _last_context
    _clean_final = _clean_answer_text(final_response)
    _clean_final = _apply_forced_final_grounding_gate(
        _clean_final, tools, tool_history, memory, client, system_prompt,
        syntax_rejections=_syntax_rejections,
    )
    _anchor(ctx_anchor.SYNC_HARD_STOP_FINAL_ANSWER_AFTER, hooks,
            iteration=_max_iter, answer=_clean_final, timed_out=True,
            tool_history=tool_history)
    _log_answer_event(_clean_final, kind="answer-hard-stop")
    return AgentTurnResult(
        answer=_clean_final,
        iterations=_max_iter,
        elapsed_seconds=time.time() - t0,
        tool_history=tool_history,
        timed_out=True,
        final_context=_last_context,
    )
