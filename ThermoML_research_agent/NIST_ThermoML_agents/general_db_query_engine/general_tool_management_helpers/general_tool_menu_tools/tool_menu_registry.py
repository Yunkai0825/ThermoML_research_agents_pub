"""
Tool Menu Registry — hierarchical browsable tool menu framework.
=================================================================
Organizes tools from multiple agents into an **arbitrarily-deep tree**.
Each tool is placed at a *path* — a tuple of category segments — and the
agent can walk the tree one level at a time:

    browse("")                           → top-level categories
    browse("query")                      → sub-categories under query
    browse("query/id_resolution")        → tools in that leaf group
    browse("query/id_resolution/resolve_compound_ids")  → full docstring

Two core operations:

1. **Browse** — walk the menu tree level-by-level, returning only the
   immediate children (sub-menus or tool one-liners).  The agent never
   sees tool descriptions until it drills down.
2. **Execute** — call any registered tool by its canonical name.

The registry **never injects tool descriptions** into the host agent's
system prompt.  Tool details only surface when the agent explicitly
calls the browse operation.

Path convention
---------------
Paths are ``/``-separated strings: ``"query/id_resolution"``.
The final segment of a tool's path is its canonical name.
Intermediate segments are category nodes (sub-menus).
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

log = logging.getLogger("TOOL-MENU")

# Type alias for a path: tuple of category segments
PathTuple = Tuple[str, ...]


def _normalise_path(raw: Union[str, Sequence[str]]) -> PathTuple:
    """Convert ``"a/b/c"`` or ``("a","b","c")`` to a tuple, stripping blanks."""
    if isinstance(raw, str):
        return tuple(s for s in raw.split("/") if s)
    return tuple(s for s in raw if s)


# ═══════════════════════════════════════════════════════════════
#  MenuEntry — leaf node (a callable tool)
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MenuEntry:
    """Single browsable tool in the menu tree."""

    name: str
    path: PathTuple                                     # full path segments
    description: str = ""
    fn: Optional[Callable] = field(repr=False, default=None)

    def one_liner(self) -> str:
        """Short description (auto-derived from docstring if missing)."""
        if self.description:
            return self.description
        if self.fn and self.fn.__doc__:
            first = self.fn.__doc__.strip().split("\n")[0]
            return first[:120]
        return ""

    def detail(self) -> str:
        """Full docstring + signature for inspection."""
        parts = [f"**{self.name}**  (path: `{'/'.join(self.path)}`)"]
        if self.fn:
            try:
                sig = inspect.signature(self.fn)
                parts.append(f"Signature: `{self.name}{sig}`")
            except (ValueError, TypeError) as exc:
                log.debug("Failed to inspect signature for %s: %s", self.name, exc)
            if self.fn.__doc__:
                parts.append(self.fn.__doc__.strip())
        return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  _MenuNode — internal tree node
# ═══════════════════════════════════════════════════════════════

class _MenuNode:
    """Internal tree node.  Children can be sub-nodes or ``MenuEntry`` leaves."""

    __slots__ = ("label", "children", "description")

    def __init__(self, label: str, description: str = "") -> None:
        self.label = label
        self.description = description
        # key → _MenuNode | MenuEntry
        self.children: Dict[str, Union[_MenuNode, MenuEntry]] = {}

    def ensure_child(self, key: str, description: str = "") -> "_MenuNode":
        """Return or create a category sub-node."""
        child = self.children.get(key)
        if isinstance(child, _MenuNode):
            if description and not child.description:
                child.description = description
            return child
        node = _MenuNode(key, description)
        self.children[key] = node
        return node


# ═══════════════════════════════════════════════════════════════
#  ToolMenuRegistry — the public API
# ═══════════════════════════════════════════════════════════════

class ToolMenuRegistry:
    """Hierarchical registry of browsable/executable tools.

    Designed to be instantiated once per host agent and populated with
    tools from any number of source agents at startup time.  Supports
    arbitrarily-deep menu trees.
    """

    def __init__(self) -> None:
        self._root = _MenuNode("root")
        self._tools: Dict[str, MenuEntry] = {}     # flat lookup by name

    # ── Registration ──────────────────────────────────────────

    def register(
        self,
        path: Union[str, Sequence[str]],
        name: str,
        fn: Callable,
        description: str = "",
    ) -> None:
        """Register a single tool at the given tree path.

        Parameters
        ----------
        path : str | Sequence[str]
            Category path, e.g. ``"query/id_resolution"`` or
            ``("query", "id_resolution")``.  The tool *name* is
            appended automatically as the final tree segment.
        name : str
            Canonical tool name (used for lookup and execution).
        fn : Callable
            The callable.
        description : str
            One-line description (auto-derived from docstring if empty).
        """
        segments = _normalise_path(path)
        full_path = segments + (name,)

        entry = MenuEntry(name=name, path=full_path,
                          description=description, fn=fn)
        self._tools[name] = entry

        # Walk / create intermediate nodes, then attach the leaf
        node = self._root
        for seg in segments:
            node = node.ensure_child(seg)
        node.children[name] = entry

    def register_from_tool_entries(
        self,
        path: Union[str, Sequence[str]],
        entries,
    ) -> None:
        """Bulk-register from a list of ``ToolEntry`` objects.

        Each entry's ``group`` is appended as a sub-level under *path*,
        so the final tree location is ``path / group / tool_name``.
        Only entries with a non-None ``fn`` are registered.
        """
        segments = _normalise_path(path)
        for e in entries:
            if getattr(e, "fn", None) is None:
                continue
            group = getattr(e, "group", "") or ""
            tool_path = segments + ((group,) if group else ())
            self.register(
                tool_path, e.name, e.fn,
                description=getattr(e, "description", ""),
            )

    def set_category_description(
        self,
        path: Union[str, Sequence[str]],
        description: str,
    ) -> None:
        """Attach a short description to an existing category node."""
        node = self._resolve_node(_normalise_path(path))
        if isinstance(node, _MenuNode):
            node.description = description

    # ── Browsing ──────────────────────────────────────────────

    @property
    def executable_tools(self) -> Dict[str, Callable]:
        """The exact callable targets accepted by execute()."""
        return {name: entry.fn for name, entry in self._tools.items()
                if entry.fn is not None}

    @property
    def size(self) -> int:
        """Total registered tools."""
        return len(self._tools)

    def browse(self, path: str = "") -> str:
        """Navigate the menu tree.

        * ``""`` — list top-level categories
        * ``"query"`` — list children of the *query* category
        * ``"query/id_resolution"`` — list tools in that sub-category
        * ``"query/id_resolution/resolve_ids"`` — full tool detail

        Returns a compact markdown string.
        """
        segments = _normalise_path(path)

        # Direct tool-name shortcut (works from any depth)
        if segments and segments[-1] in self._tools:
            return self._tools[segments[-1]].detail()

        target = self._resolve_node(segments)

        if target is None:
            return self._not_found(path)

        if isinstance(target, MenuEntry):
            return target.detail()

        # It's a _MenuNode — list its children
        return self._render_node(target, segments)

    def tool_detail(self, tool_name: str) -> str:
        """Full docstring + signature for a specific tool (flat lookup)."""
        entry = self._tools.get(tool_name)
        if entry is None:
            return f"Unknown tool: `{tool_name}`"
        return entry.detail()

    # ── Execution ─────────────────────────────────────────────

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def execute(self, tool_name: str, kwargs: Dict[str, Any]) -> Any:
        """Look up and call a registered tool by canonical name.

        Raises ``KeyError`` if not found, ``RuntimeError`` if no callable.
        """
        entry = self._tools.get(tool_name)
        if entry is None:
            raise KeyError(f"Tool '{tool_name}' not found in menu")
        if entry.fn is None:
            raise RuntimeError(f"Tool '{tool_name}' has no callable")
        log.info("Menu execute: %s", tool_name)
        return entry.fn(**kwargs)

    # ── Health check ──────────────────────────────────────────

    def validate(self) -> None:
        """Verify every registered menu tool has a valid callable.

        Raises ``RuntimeError`` with a detailed listing of broken tools
        if any entry is missing its ``fn`` or ``fn`` is not callable.
        """
        errors: list[str] = []
        for name, entry in self._tools.items():
            if entry.fn is None:
                errors.append(
                    f"[NULL_FN] Menu tool {name!r} "
                    f"(path={'/'.join(entry.path)}) has fn=None"
                )
            elif not callable(entry.fn):
                errors.append(
                    f"[NOT_CALLABLE] Menu tool {name!r} "
                    f"(path={'/'.join(entry.path)}) fn is not callable: "
                    f"{type(entry.fn).__name__}"
                )
        if errors:
            detail = "\n  ".join([""] + errors)
            msg = (
                f"ToolMenuRegistry validation FAILED — "
                f"{len(errors)} error(s):{detail}"
            )
            log.error(msg)
            raise RuntimeError(msg)
        log.info(
            "ToolMenuRegistry validation PASSED: %d tools, all callable",
            len(self._tools),
        )

    # ── Private helpers ───────────────────────────────────────

    def _resolve_node(
        self, segments: PathTuple,
    ) -> Union[_MenuNode, MenuEntry, None]:
        """Walk from root along *segments*, returning the target node."""
        node: Union[_MenuNode, MenuEntry] = self._root
        for seg in segments:
            if isinstance(node, _MenuNode):
                child = node.children.get(seg)
                if child is None:
                    return None
                node = child
            else:
                return None
        return node

    def _render_node(
        self, node: _MenuNode, segments: PathTuple,
    ) -> str:
        """Render the immediate children of a tree node."""
        prefix = "/".join(segments) if segments else ""
        header = f"## {prefix}" if prefix else "## Tool Menu"

        lines: list[str] = [header]
        if node.description:
            lines.append(node.description)
            lines.append("")

        # Separate sub-menus from leaf tools
        sub_menus: list[_MenuNode] = []
        tools: list[MenuEntry] = []
        for child in node.children.values():
            if isinstance(child, _MenuNode):
                sub_menus.append(child)
            else:
                tools.append(child)

        if sub_menus:
            lines.append("**Sub-menus:**")
            for sm in sorted(sub_menus, key=lambda n: n.label):
                n_tools = self._count_tools(sm)
                p = f"{prefix}/{sm.label}" if prefix else sm.label
                desc = f" — {sm.description}" if sm.description else ""
                lines.append(f"- 📂 **{sm.label}** ({n_tools} tools){desc}  →  submenu `{p}`")

        if tools:
            lines.append("**Tools:**")
            for t in sorted(tools, key=lambda e: e.name):
                lines.append(f"- 🔧 **{t.name}** — {t.one_liner()}")

        if not sub_menus and not tools:
            lines.append("(empty)")

        return "\n".join(lines)

    @staticmethod
    def _count_tools(node: _MenuNode) -> int:
        """Recursively count leaf tools under a node."""
        count = 0
        for child in node.children.values():
            if isinstance(child, MenuEntry):
                count += 1
            elif isinstance(child, _MenuNode):
                count += ToolMenuRegistry._count_tools(child)
        return count

    def _not_found(self, path: str) -> str:
        """Fallback when a path doesn't match anything."""
        top = sorted(self._root.children.keys())
        return (
            f"Path `{path}` not found.\n"
            f"Top-level categories: {', '.join(top) or '(none)'}"
        )
