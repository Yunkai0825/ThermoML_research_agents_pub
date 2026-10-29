"""
AgentToolCatalog — base class for agent-specific tool registries.
=================================================================

Central authority for what tools an agent layer owns, how each tool's
raw dict output is compacted, and whether that compacted markdown is
further triaged by the LLM KEEP/DISCARD subagent.

Two-stage compaction pipeline
-----------------------------
For every tool call that returns a ``dict``:

1. **Layer 1 — hardcoded compactor** (``compact_tool_result``)
   Looks up ``ToolEntry.compactor_fn`` and converts the dict to
   concise markdown. Missing or failing compactors are contract errors.
   Skipped when ``ToolEntry.skip_compactor`` is True.

   **Condensation levels** — some tools support tiered compaction:

   - ``"full"`` — default, uses ``compactor_fn``.
   - ``"condense"`` — uses ``condense_fn`` (lighter, strips data rows).
   - ``"ultra_condense"`` — uses ``ultra_condense_fn`` (one-liner per block).
   - ``adaptive_condense`` — compactor_fn handles levels internally
     based on output size (e.g. via ``_apply_block_condensation``).

2. **Layer 2 — agentic subagent** (``call_subagent``)
   Sends the markdown to a lightweight ReAct LLM subagent that
   decides KEEP (extract tables / key identifiers) or DISCARD
   (explain + suggest refinement).
   Skipped when ``ToolEntry.skip_subagent`` is True.

Subclasses should set:
  * ``pipeline_label`` — ``"query"`` or ``"analysis"``

and call ``register`` / ``register_many`` in ``__init__``.

Usage example (agent subclass)::

    class QueryL1Catalog(AgentToolCatalog):
        pipeline_label = "query"

        def __init__(self, *, client_factory, cfg):
            super().__init__(client_factory=client_factory, cfg=cfg)
            self.register_many([
                ToolEntry("resolve_ids", resolve_ids,
                          group="id_resolution",
                          compactor_fn=compact_resolve_ids),
                ToolEntry("L2_comp_eval", dispatch_l2_comp_eval,
                          group="L2_subagent",
                          skip_compactor=True, skip_subagent=True),
            ])

Then the dispatch code can call::

    catalog = QueryL1Catalog(client_factory=..., cfg=...)
    # Strict wrapped tool text for agent_turn():
    agent_turn(..., tools=catalog.wrapped_tools(), ...)
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, Dict, Iterable, Optional

from .tool_entry import ToolEntry
from .agent_tool_compactor_hooks_catalog import CompactorCatalog
from ..general_tool_results_compactor_agentic_hooks import (
    ToolResult, ToolResultCompactor,
)
from ...general_hooks_management_helpers.general_context_hooks.history_tracking_hooks import (
    get_active_history_recorder,
)
from .. import _tool_hooks_anchors_catalog as _tool_anchor
from ...general_argo_engine_helpers.engine_hooks_anchors import anchor as _anchor
from ...general_argo_engine_helpers.engine_react_helpers.react_helpers import (
    validate_native_tool_result,
)
from ...general_text_context_marker_catalog import wrap_subagent_answer

log = logging.getLogger("agent-tool-catalog")


class AgentToolCatalog:
    """Base class for every agent-level tool catalog."""

    pipeline_label: str = "agent"

    # ── Construction ────────────────────────────────────────────

    def __init__(
        self,
        *,
        client_factory: Optional[Callable] = None,
        cfg: Any = None,
        entries: Optional[Iterable[ToolEntry]] = None,
        extra_compactors: Optional[Dict[str, Callable]] = None,
        compactor_catalog: Optional[CompactorCatalog] = None,
    ) -> None:
        self._entries: Dict[str, ToolEntry] = {}
        self._extra_compactors: Dict[str, Callable] = dict(extra_compactors or {})
        self._client_factory = client_factory
        self._cfg = cfg
        self._compactor_catalog = compactor_catalog
        # Lazily built — invalidated when entries change
        self._compactor_obj: Optional[Any] = None
        if entries:
            self.register_many(entries)

    # ── Entry management ────────────────────────────────────────

    def register(self, entry: ToolEntry) -> None:
        """Add or replace a single tool entry.

        When a ``compactor_catalog`` was provided at construction,
        entries without an explicit ``compactor_fn`` are auto-resolved
        from the catalog by matching ``entry.name``.
        """
        if self._compactor_catalog is not None:
            entry = self._resolve_entry_compactor(entry)
        if entry.skip_compactor and not entry.skip_subagent:
            raise ValueError(
                f"Tool {entry.name!r} cannot run an agentic subagent without "
                "a deterministic compactor"
            )
        if not entry.skip_compactor and entry.compactor_fn is None:
            raise KeyError(
                f"Tool {entry.name!r} has no deterministic compactor; "
                "register its exact compactor or explicitly bypass both stages"
            )
        self._entries[entry.name] = entry
        self._compactor_obj = None  # invalidate

    def register_many(self, entries: Iterable[ToolEntry]) -> None:
        for e in entries:
            self.register(e)
        self._compactor_obj = None

    def add_extra_compactor(self, tool_name: str, fn: Callable) -> None:
        """Register a compactor for a name that has no ToolEntry.

        Useful when an agent needs compactor coverage for intermediate
        result keys that aren't standalone tools (e.g. analysis agent's
        ``resolve_compounds`` which is a sub-step of ``query_thermoml``).
        """
        self._extra_compactors[tool_name] = fn
        self._compactor_obj = None

    def _resolve_entry_compactor(self, entry: ToolEntry) -> ToolEntry:
        """Auto-resolve compactor_fn from ``self._compactor_catalog``.

        If the entry already has a ``compactor_fn`` or explicitly bypasses
        both compaction stages, it is unchanged. Otherwise its exact name
        must exist in the configured compactor catalog.
        """
        if entry.compactor_fn is not None or entry.skip_compactor:
            return entry
        cat = self._compactor_catalog
        if cat is not None and entry.name in cat:
            fn = cat.registry[entry.name]
            # Frozen dataclass — rebuild with the resolved compactor
            return ToolEntry(
                name=entry.name,
                fn=entry.fn,
                group=entry.group,
                compactor_fn=fn,
                condense_fn=entry.condense_fn,
                ultra_condense_fn=entry.ultra_condense_fn,
                adaptive_condense=entry.adaptive_condense,
                skip_compactor=entry.skip_compactor,
                skip_subagent=entry.skip_subagent,
                description=entry.description,
            )
        raise KeyError(f"No deterministic compactor registered for tool {entry.name!r}")

    # ── Read accessors ──────────────────────────────────────────

    @property
    def entries(self) -> Dict[str, ToolEntry]:
        """Shallow copy of the entry dict."""
        return dict(self._entries)

    def __getitem__(self, name: str) -> ToolEntry:
        return self._entries[name]

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # ── Tool dict (for agent_turn) ──────────────────────────────

    @property
    def tools(self) -> Dict[str, Callable]:
        """``{name: callable}`` — raw (unwrapped) tool dict."""
        return {n: e.fn for n, e in self._entries.items()}

    def tools_by_group(self, group: str) -> Dict[str, Callable]:
        """Subset of ``tools`` belonging to *group*."""
        return {n: e.fn for n, e in self._entries.items() if e.group == group}

    # ── Compactor registry (derived from entries) ───────────────

    @property
    def compactor_registry(self) -> Dict[str, Callable]:
        """Merged compactor map: tool entries + extra compactors."""
        reg: Dict[str, Callable] = {}
        for n, e in self._entries.items():
            if e.compactor_fn and not e.skip_compactor:
                reg[n] = e.compactor_fn
        reg.update(self._extra_compactors)
        return reg

    # ── Internal ToolResultCompactor (lazy) ─────────────────────

    @property
    def _compactor(self):
        """Lazily builds and caches a ``ToolResultCompactor``."""
        if self._compactor_obj is None:
            self._compactor_obj = ToolResultCompactor(
                client_factory=self._client_factory,
                cfg=self._cfg,
                pipeline_label=self.pipeline_label,
                compactor_registry=self.compactor_registry,
            )
        return self._compactor_obj

    # ── Layer 1: hardcoded dict→markdown ────────────────────────

    def compact_tool_result(
        self,
        tool_name: str,
        data: dict,
        *,
        level: str = "full",
    ) -> str:
        """Apply the hardcoded compactor for *tool_name* (Layer 1).

        Parameters
        ----------
        tool_name : str
            Registered tool name.
        data : dict
            Raw tool output dict.
        level : str
            Condensation level: ``"full"`` (default), ``"condense"``,
            or ``"ultra_condense"``. The requested level must be registered;
            compactor errors are surfaced as contract failures.
        """
        if tool_name not in self._entries:
            raise KeyError(f"Unknown tool: {tool_name!r}")
        entry = self._entries[tool_name]
        validated = validate_native_tool_result(tool_name, data)
        if not isinstance(validated, dict):
            raise AssertionError("validated compactor input changed type")
        data = validated

        if level == "full":
            return self._compactor.compact_tool_result(tool_name, data)
        if level not in {"condense", "ultra_condense"}:
            raise ValueError(f"Unknown compaction level: {level!r}")
        fn = entry.condense_fn if level == "condense" else entry.ultra_condense_fn
        if fn is None:
            raise KeyError(
                f"Tool {tool_name!r} has no {level!r} compactor registered"
            )
        try:
            return fn(data)
        except Exception as exc:
            raise RuntimeError(
                f"{level} compactor for {tool_name!r} failed"
            ) from exc

    # ── Layer 2: agentic KEEP/DISCARD ───────────────────────────

    def call_subagent(
        self,
        tool_name: str,
        purpose: str,
        tasks: str,
        compact_md: str,
        raw: dict,
    ):
        """Send *compact_md* to the LLM subagent for KEEP/DISCARD (Layer 2)."""
        return self._compactor.call(tool_name, purpose, tasks, compact_md, raw)

    # ── Two-stage: compact → subagent (with skip logic) ─────────

    def compact_and_call(
        self,
        tool_name: str,
        purpose: str,
        tasks: str,
        raw: dict,
        *,
        level: str = "full",
    ):
        """Run the full two-stage pipeline respecting skip flags.

        Parameters
        ----------
        tool_name, purpose, tasks, raw
            Standard two-stage arguments.
        level : str
            Condensation level for Layer 1: ``"full"``, ``"condense"``,
            or ``"ultra_condense"``.  Tools with ``adaptive_condense``
            handle this internally through ``compactor_fn``.

        Returns
        -------
        ToolResult
            ``.raw`` = original dict, ``.text`` = final markdown,
            ``.discarded`` = True if subagent chose DISCARD.
        """
        if tool_name not in self._entries:
            raise KeyError(f"Unknown tool: {tool_name!r}")
        entry = self._entries[tool_name]
        validated = validate_native_tool_result(tool_name, raw)
        if not isinstance(validated, dict):
            raise AssertionError("validated compact-and-call input changed type")
        raw = validated
        native_chars = len(json.dumps(raw, ensure_ascii=False, default=str))

        # Layer 1
        compact_md = self.compact_tool_result(tool_name, raw, level=level)

        # Layer 2 (skip when flagged)
        if entry.skip_subagent:
            result = ToolResult(raw=raw, text=compact_md)
            verdict = "SKIP"
        else:
            result = self._compactor.call(tool_name, purpose, tasks, compact_md, raw)
            verdict = "DISCARD" if result.discarded else "KEEP"
        self._log_pipeline_sizes(
            tool_name, native_chars, len(compact_md), len(result.text), verdict,
        )
        return result

    def _log_pipeline_sizes(
        self,
        tool_name: str,
        native_chars: int,
        compact_chars: int,
        final_chars: int,
        verdict: str,
    ) -> None:
        """Record the per-call compaction chain on the session-active recorder."""
        log.info(
            "Compaction pipeline %s [%s]: native %d → hardcoded %d → agentic %d (%s)",
            tool_name, self.pipeline_label,
            native_chars, compact_chars, final_chars, verdict,
        )
        try:
            recorder = get_active_history_recorder()
        except RuntimeError:
            return  # standalone/test usage without an active session
        recorder.log_compaction_pipeline(
            tool_name=tool_name,
            pipeline=self.pipeline_label,
            native_chars=native_chars,
            compact_chars=compact_chars,
            final_chars=final_chars,
            verdict=verdict,
        )

    # ── Extensibility hooks ───────────────────────────────────

    def _on_raw_result(self, tool_name: str, raw: dict) -> None:
        """Called after a wrapped tool returns an object, before compaction.

        Override in subclasses to add side-effects like stats logging.
        The default implementation is a no-op.
        """

    # ── Pre-execution batch validation ────────────────────────

    def validate_batch(
        self,
        tool_calls: list[dict],
        tools: dict[str, Callable],
        *,
        hooks=None,
    ) -> str | None:
        """Check all tool calls for guidance before execution.

        Fires the ``SYNC_TOOL_GUIDANCE_CHECK`` anchor for each tool
        call.  Guidance hooks bound to that anchor filter by
        ``tool_name`` and return:

        - ``None`` — hook does not handle this tool (no guidance).
        - ``""``   — handled OK (may have auto-filled kwargs).
        - non-empty ``str`` — blocked with guidance text.

        Builds a per-tool status report:

        - **blocked** tools with error detail
        - **ready** tools that passed (with auto-filled param notes)
        - **no-validation** tools held back by the batch

        If ANY tool emits guidance, returns a combined report string
        (replaces batch execution).  The agent can fix blocked tools
        and re-submit, or drop them and retry with the rest.
        Returns ``None`` when all tools pass.

        Parameters
        ----------
        tool_calls : list[dict]
            Parsed tool calls, each with ``name`` and ``arguments`` keys.
        tools : dict[str, Callable]
            The wrapped tools dict (for signature-based arg normalisation).
        hooks : EngineHooks or None
            Compiled engine hooks dict for anchor dispatch.

        Returns
        -------
        str or None
            Per-tool validation report, or None if all pass.
        """
        # Per-tool tracking
        blocked: list[tuple[str, str]] = []       # (name, guidance_text)
        ready_filled: list[tuple[str, dict]] = [] # (name, {param: new_val})
        ready_clean: list[str] = []               # names that passed as-is
        no_guidance: list[str] = []               # tools without guidance hook

        for tc in tool_calls:
            tool_name = tc.get("name", "")

            raw_args = tc.get("arguments", {})
            fn = tools.get(tool_name)
            if fn is not None:
                try:
                    from ...general_argo_engine_helpers import _validate_tool_arguments
                    call_kwargs = _validate_tool_arguments(tool_name, raw_args, fn)
                except Exception as exc:
                    blocked.append((tool_name, str(exc)))
                    continue
            else:
                call_kwargs = dict(raw_args)

            # Snapshot before anchor fires (for change detection)
            kwargs_before = dict(call_kwargs)

            # Fire per-tool guidance anchor — zero hardcoded hook calls.
            result = _anchor(
                _tool_anchor.SYNC_TOOL_GUIDANCE_CHECK,
                hooks,
                tool_name=tool_name,
                call_kwargs=call_kwargs,
            )

            if result is None:
                # No guidance hook handled this tool
                no_guidance.append(tool_name)
            elif result:
                # Non-empty string → blocked
                blocked.append((tool_name, result))
            else:
                # Empty string → handled OK, detect auto-filled params
                changes = {
                    k: call_kwargs[k]
                    for k in call_kwargs
                    if call_kwargs[k] != kwargs_before.get(k)
                }
                tc["arguments"] = call_kwargs
                if changes:
                    ready_filled.append((tool_name, changes))
                else:
                    ready_clean.append(tool_name)

        if not blocked:
            return None

        # ── Build per-tool status report ──────────────────────
        parts: list[str] = []

        for name, reason in blocked:
            parts.append(f"**\u2717 {name}** — needs correction:\n{reason}")

        if ready_filled:
            for name, changes in ready_filled:
                fills = ", ".join(
                    f"`{k}`=`{v}`" for k, v in changes.items()
                )
                parts.append(f"**\u2713 {name}** — ready (auto-filled: {fills})")

        if ready_clean:
            parts.append(f"**\u2713 {', '.join(ready_clean)}** — ready")

        if no_guidance:
            parts.append(
                f"**\u00B7 {', '.join(no_guidance)}** — "
                f"no validation needed (held back by batch)"
            )

        parts.append(
            "_Fix the blocked tool(s) and re-submit, "
            "or drop them and re-submit only the ready ones "
            "(you are able to add new tools to the batch if you really need to)._"
        )
        return "\n\n".join(parts)

    # ── Tool wrapping (adds purpose/tasks to signature) ─────────

    def wrap_tool(
        self, tool_name: str, *, expose_internal_result: bool = False
    ) -> Callable:
        """Return a wrapped callable that runs the two-stage pipeline.

        The wrapper:
        1. Reads the required ``purpose`` and ``tasks`` wrapper fields.
        2. Forwards them only when the underlying function explicitly
           declares them, and calls the underlying function.
        3. If the result is a dict, runs ``compact_and_call``.
        4. Returns compacted text at the agent boundary. Orchestrators that
           immediately add the memory instrumentation wrapper may request the
           internal ``ToolResult`` long enough to hard-record its raw dict.

        Tools with both ``skip_compactor`` **and** ``skip_subagent``
        True are returned unwrapped.
        """
        entry = self._entries[tool_name]

        # Fully pass-through tools need no wrapper
        if entry.skip_compactor and entry.skip_subagent:
            return entry.fn

        fn = entry.fn
        sig = inspect.signature(fn)
        catalog_ref = self  # closure

        def wrapped(**kwargs):
            purpose = kwargs.pop("purpose")
            tasks = kwargs.pop("tasks")

            if not isinstance(purpose, str) or not purpose.strip():
                raise TypeError(
                    f"TOOL_ARGUMENT_REFINEMENT_REQUIRED: {tool_name}.purpose "
                    "must be a non-empty string"
                )
            if not isinstance(tasks, str) or not tasks.strip():
                raise TypeError(
                    f"TOOL_ARGUMENT_REFINEMENT_REQUIRED: {tool_name}.tasks "
                    "must be a non-empty string"
                )

            # Purpose/tasks belong to the shared wrapper contract. A tool may
            # also explicitly consume either field (for example, to brief a
            # nested worker); forward only declared parameters.
            if "purpose" in sig.parameters:
                kwargs["purpose"] = purpose
            if "tasks" in sig.parameters:
                kwargs["tasks"] = tasks

            from ...general_argo_engine_helpers import _validate_tool_arguments
            call_kwargs = _validate_tool_arguments(tool_name, kwargs, fn)
            result = fn(**call_kwargs)
            returns_subagent_answer = bool(
                getattr(fn, "_returns_subagent_answer", False)
            )

            # Agent-generated JSON bypasses content compaction: the parent
            # receives the complete validated result in its dedicated marker.
            if isinstance(result, str):
                return (
                    wrap_subagent_answer(result)
                    if returns_subagent_answer
                    else result
                )

            if not isinstance(result, dict):
                raise TypeError(
                    f"Tool {tool_name!r} returned {type(result).__name__}; "
                    "a compacted tool must return an object or a pre-compacted string"
                )
            raw = validate_native_tool_result(tool_name, result)
            if not isinstance(raw, dict):
                raise AssertionError("validated object tool result changed type")

            # Optional subclass hook (e.g. stats logging)
            catalog_ref._on_raw_result(tool_name, raw)

            if returns_subagent_answer:
                return wrap_subagent_answer(raw)

            compacted = catalog_ref.compact_and_call(
                tool_name, purpose, tasks, raw,
            )
            if not isinstance(compacted, ToolResult):
                raise TypeError(
                    f"compaction pipeline for {tool_name!r} must return ToolResult"
                )
            if not isinstance(compacted.text, str) or not compacted.text.strip():
                raise ValueError(
                    f"compaction pipeline for {tool_name!r} returned empty text"
                )
            return compacted if expose_internal_result else compacted.text

        wrapped.__wrapped__ = fn  # preserve live child-tool capability metadata
        wrapped.__name__ = getattr(fn, "__name__", tool_name)
        wrapped.__doc__ = getattr(fn, "__doc__", "")
        if getattr(fn, "_returns_subagent_answer", False):
            wrapped._returns_subagent_answer = True

        # Preserve original signature + add purpose/tasks (dedup if already present)
        _EXTRA_NAMES = {"purpose", "tasks"}
        orig_params = [
            p for p in sig.parameters.values()
            if p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ) and p.name not in _EXTRA_NAMES
        ]
        extra = [
            inspect.Parameter(
                "purpose", inspect.Parameter.KEYWORD_ONLY, annotation=str,
            ),
            inspect.Parameter(
                "tasks", inspect.Parameter.KEYWORD_ONLY, annotation=str,
            ),
        ]
        wrapped.__signature__ = sig.replace(parameters=orig_params + extra)
        return wrapped

    def wrapped_tools(
        self, *, expose_internal_results: bool = False
    ) -> Dict[str, Callable]:
        """Tool dict with compaction wrapping where applicable.

        Tools that skip both layers are returned unwrapped.
        """
        return {
            name: self.wrap_tool(
                name, expose_internal_result=expose_internal_results
            )
            for name in self._entries
        }

    # ── Helpers ─────────────────────────────────────────────────

    def groups(self) -> list[str]:
        """Sorted list of distinct group labels."""
        return sorted({e.group for e in self._entries.values() if e.group})

    def summary(self) -> str:
        """Human-readable summary of all entries."""
        lines = [f"AgentToolCatalog ({self.pipeline_label}) — {len(self)} tools"]
        for g in self.groups():
            members = [n for n, e in self._entries.items() if e.group == g]
            lines.append(f"  [{g}] {', '.join(members)}")
        ungrouped = [n for n, e in self._entries.items() if not e.group]
        if ungrouped:
            lines.append(f"  [ungrouped] {', '.join(ungrouped)}")
        return "\n".join(lines)
