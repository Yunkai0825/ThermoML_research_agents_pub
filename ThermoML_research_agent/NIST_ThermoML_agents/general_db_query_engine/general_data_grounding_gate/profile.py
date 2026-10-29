"""Select current or frozen historical grounding for one complete run tree."""
from __future__ import annotations

from functools import wraps
from importlib import import_module

HISTORICAL_COMMIT = "bdf50547fb4cb8f1b5037dd55b4878ccb1cb21bf"


def is_legacy_mode() -> bool:
    # Lazy import keeps the gate usable before an agent config is loaded.
    from ..general_argo_engine_helpers.engine_config import get_grounding_gate_legacy_mode
    return get_grounding_gate_legacy_mode()


def profile_metadata() -> dict:
    legacy = is_legacy_mode()
    return {
        "name": "historical_september_2026" if legacy else "current",
        "legacy_mode": legacy,
        "historical_source_commit": HISTORICAL_COMMIT if legacy else None,
        "scope": "grounding policy; current tools, prompts, models and compaction",
    }


def historical_dispatch(module_name: str):
    """Keep the frozen policy self-contained; dispatch at public boundaries."""
    def decorate(current):
        @wraps(current)
        def wrapped(*args, **kwargs):
            if is_legacy_mode():
                legacy = import_module(
                    f".historical_september_2026.{module_name}", __package__)
                return getattr(legacy, current.__name__)(*args, **kwargs)
            return current(*args, **kwargs)
        return wrapped
    return decorate
