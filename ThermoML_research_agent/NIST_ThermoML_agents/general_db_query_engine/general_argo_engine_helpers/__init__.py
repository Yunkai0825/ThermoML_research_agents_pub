"""argo_engine_helpers — sub-modules for the ReAct agentic loop engine.

EngineConfig / load_config / get_config are imported eagerly (lightweight).
Everything else is lazy-loaded on first access to avoid circular imports
when agent config modules subclass EngineConfig.
"""

from .engine_config import (  # noqa: F401
    EngineConfig,
    load_config,
    get_config,
    get_grounding_gate_legacy_mode,
    with_engine_config,
)
from .engine_hooks_anchors import (  # noqa: F401
    AnchorPoint,
    AnchorCollection,
    AgentAnchorCollection,
    HookBinding,
    EngineHooks,
    AgentHookCollection,
    EMPTY_HOOKS,
    AGENT_START_TRACKING,
    AGENT_FINALIZE_TRACKING,
    AGENT_RECORD_REFERENCES,
    AGENT_SAVE_FINAL_CONTEXT,
    AGENT_RUN_VERDICT,
    define_anchor,
    define_anchor_collection,
    bind_hook,
    build_agent_lifecycle_bindings,
    compile_hooks,
    anchor,
)

_LAZY_NAMES = {
    "ArgoClient",
    "AgentTurnResult",
    "agent_turn",
    "async_agent_turn",
    "require_result_within_limit",
    "_validate_tool_arguments",
}


def __getattr__(name: str):
    if name in _LAZY_NAMES:
        from . import _argo_engine_entry_point as _ep
        val = getattr(_ep, name)
        globals()[name] = val  # cache for subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EngineConfig",
    "load_config",
    "get_config",
    "get_grounding_gate_legacy_mode",
    "with_engine_config",
    "AnchorPoint",
    "AnchorCollection",
    "AgentAnchorCollection",
    "HookBinding",
    "EngineHooks",
    "AgentHookCollection",
    "EMPTY_HOOKS",
    "AGENT_START_TRACKING",
    "AGENT_FINALIZE_TRACKING",
    "AGENT_RECORD_REFERENCES",
    "AGENT_SAVE_FINAL_CONTEXT",
    "AGENT_RUN_VERDICT",
    "define_anchor",
    "define_anchor_collection",
    "bind_hook",
    "build_agent_lifecycle_bindings",
    "compile_hooks",
    "anchor",
    *_LAZY_NAMES,
]
