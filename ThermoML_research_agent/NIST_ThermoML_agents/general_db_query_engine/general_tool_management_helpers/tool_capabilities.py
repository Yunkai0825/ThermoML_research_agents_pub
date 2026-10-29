"""Resolve tools an agent can call through its live dispatchers.

Dispatcher metadata is attached to callables, so a tool tree is rooted in the
tools passed to an agent turn rather than in a global catalog. Providers are
evaluated only for reachable dispatchers. A broken provider is a configuration
error: silently omitting its children could disable a gate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

_CHILD_PROVIDER = "_reachable_tool_provider"


def bind_reachable_tools(fn: Callable, provider: Callable[[], Mapping[str, Callable]]) -> Callable:
    """Declare the callable tool map reached when *fn* dispatches a child.

    The provider runs lazily, after modules and catalogs have initialized.
    """
    if not callable(fn) or not callable(provider):
        raise TypeError("tool and child-tool provider must be callable")
    target = getattr(fn, "__func__", fn)
    setattr(target, _CHILD_PROVIDER, provider)
    return fn


def _provider_for(fn: Callable):
    """Follow ordinary wrappers and bound methods to capability metadata."""
    seen: set[int] = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        provider = getattr(fn, _CHILD_PROVIDER, None)
        if provider is not None:
            return provider
        underlying = getattr(fn, "__func__", None)
        if underlying is not None and underlying is not fn:
            fn = underlying
            continue
        fn = getattr(fn, "__wrapped__", None) or getattr(fn, "func", None)
    return None


def reachable_tool_names(tools: Mapping[str, Callable]) -> set[str]:
    """Return names in the live roots and all registered child tool maps.

    A disabled tool is absent from its agent's actual map and thus absent from
    the result. Cycles between dispatchers are visited once.
    """
    if not isinstance(tools, Mapping):
        raise TypeError("tools must be a mapping of names to callables")
    names: set[str] = set()
    # Hold the objects, not just their ids: fresh catalog wrappers can be
    # freed during traversal and Python can recycle an id for a later tool.
    visited_callables: dict[int, Callable] = {}
    visited_providers: dict[int, Callable] = {}
    pending = [tools]
    while pending:
        current = pending.pop()
        if not isinstance(current, Mapping):
            raise TypeError("child-tool provider must return a mapping")
        for name, fn in current.items():
            if not isinstance(name, str) or not callable(fn):
                raise TypeError("tool maps must contain string names and callable values")
            names.add(name)
            target = getattr(fn, "__func__", fn)
            identity = id(target)
            if identity in visited_callables:
                continue
            visited_callables[identity] = target
            provider = _provider_for(fn)
            if provider is not None:
                provider_id = id(provider)
                if provider_id in visited_providers:
                    continue
                visited_providers[provider_id] = provider
                children = provider()
                if not isinstance(children, Mapping):
                    raise TypeError(f"child-tool provider for {name!r} must return a mapping")
                pending.append(children)
    return names

