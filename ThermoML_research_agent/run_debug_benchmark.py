#!/usr/bin/env python
"""Thin launcher for the ThermoML debug-prompt benchmark rerun.

Runs the EXISTING per-agent ``run_tests`` runners in priority order
(main -> query -> analysis), parallel within each agent, writing into the
the shared _benchmark directory. All heavy lifting stays in the existing runners; this file
only reads provider settings, computes benchmark output dirs, and sequences the
three subprocesses.

    python run_debug_benchmark.py          # preflight: health-check + show plan (NO runs)
    python run_debug_benchmark.py --go     # launch the runs (Argo backend)
    python run_debug_benchmark.py --go --anthropic   # launch via the Anthropic adapter instead
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import datetime as dt
from pathlib import Path
from urllib.parse import urlsplit

# Near-total ERROR/TIMEOUT rate in one phase = likely systemic algorithmic
# failure (e.g. context lost on subagent handover), not normal content misses.
FATAL_MIN_PROMPTS = 5
FATAL_RATE = 0.9


WORKSPACE_ROOT = Path(__file__).resolve().parent  # ThermoML_research_agent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# --anthropic routes every subprocess through the Anthropic adapter (see
# general_argo_engine_helpers/anthropic_argo_adapter.py) instead of Argo.
# Same JOBS, same prompts, same configs — only the HTTP backend changes.
USE_ANTHROPIC = False
BOOTSTRAP_SCRIPT = WORKSPACE_ROOT / "_anthropic_adapter_bootstrap.py"


RESUME_TS = None


BENCHMARK_ROOT = WORKSPACE_ROOT.parent / "_benchmark"
TS = RESUME_TS or dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# Main: retain the selected 17-case cohort, excluding 1.1.1, 1.1.3, 1.2.3, 1.2.4
# and the appended examples 1.1.5, 3.7, 4.5 under the current prompt numbering.
MAIN_IDS = ["1.1.2", "1.1.4", "1.2.1", "1.2.2",
            "2.1", "2.2", "2.3",
            "3.1", "3.2", "3.3", "3.4", "3.5", "3.6",
            "4.1", "4.2", "4.3", "4.4"]

# (agent, run_tests module, explicit IDs [] = all, workers, gap seconds)
JOBS = [
    ("main",     "NIST_ThermoML_agents._NIST_ThermoML_main_agent._DEBUG_script.run_tests",    MAIN_IDS, 3, 12.0),
    ("query",    "NIST_ThermoML_agents.NIST_ThermoML_query_agent._DEBUG_script.run_tests",    [],       5, 6.0),
    ("analysis", "NIST_ThermoML_agents.NIST_ThermoML_analysis_agent._DEBUG_script.run_tests", [],       5, 6.0),
]

# Entry-point + config imports used only by the preflight health check.
_ENTRYPOINTS = [
    ("main",     "NIST_ThermoML_agents._NIST_ThermoML_main_agent.ThermoML_main_api",       "ThermoML_main_run"),
    ("query",    "NIST_ThermoML_agents.NIST_ThermoML_query_agent.ThermoML_query_api",      "ThermoML_query_run"),
    ("analysis", "NIST_ThermoML_agents.NIST_ThermoML_analysis_agent.ThermoML_analysis_api", "ThermoML_analysis_run"),
]


def out_dir(agent: str) -> Path:
    return BENCHMARK_ROOT / agent.capitalize() / f"test_run_{TS}"


def prompt_selection(agent: str, module: str, ids: list[str]) -> tuple[list[str], list[str]]:
    """Report the exact prompt cohort the per-agent parser can run."""
    from NIST_ThermoML_agents.general_db_query_engine.general_DEBUG_test_runner_helpers import (
        parse_prompts_simple, parse_prompts_with_sections,
    )
    prompt_file = WORKSPACE_ROOT.joinpath(*module.split(".")).parent / "test_prompts.md"
    if agent == "query":
        available = [row["id"] for row in parse_prompts_with_sections(prompt_file)]
    else:
        available = [pid for pid, _ in parse_prompts_simple(prompt_file)]
    matched = [pid for pid in available if not ids or pid in ids]
    missing = [pid for pid in ids if pid not in available]
    return matched, missing


def build_cmd(agent, module, ids, workers, gap):
    out = out_dir(agent)
    if USE_ANTHROPIC:
        cmd = [sys.executable, str(BOOTSTRAP_SCRIPT), module, *ids,
               "--workers", str(workers), "--gap", str(gap), "--output", str(out)]
    else:
        cmd = [sys.executable, "-m", module, *ids,
               "--workers", str(workers), "--gap", str(gap), "--output", str(out)]
    if RESUME_TS:
        cmd.append("--skip-existing")
    return cmd, out


def _check_fatal_pattern(agent: str, out: Path) -> str | None:
    """Return a fatal-error report string, or None if the phase looks healthy.

    A crashed subprocess is already caught by its non-zero exit code; this
    catches the other failure class the runners swallow per-prompt (e.g. a
    handover/context-loss bug) — every prompt still reports status=ERROR
    inside the JSON trace even though the harness itself exits 0.
    """
    from NIST_ThermoML_agents.general_db_query_engine.general_hooks_management_helpers.general_tracking_hooks_output_helpers import get_status

    traces = sorted(out.glob("TEST_TRACE_*.json"))
    if not traces:
        return f"[{agent}] no TEST_TRACE_*.json found in {out}"
    results = json.loads(traces[-1].read_text(encoding="utf-8"))
    total = len(results)
    if total == 0:
        return None
    bad = [r for r in results if get_status(r) in ("ERROR", "TIMEOUT")]
    rate = len(bad) / total
    if rate == 1.0 or (total >= FATAL_MIN_PROMPTS and rate >= FATAL_RATE):
        lines = [f"[{agent}] FATAL PATTERN: {len(bad)}/{total} prompts ERROR/TIMEOUT ({rate:.0%})"]
        for r in bad[:3]:
            lines.append(f"  Q{r.get('prompt_id')}: {str(r.get('answer', r.get('error', '')))[:200]}")
        return "\n".join(lines)
    print(f"[{agent}] status OK ({total - len(bad)}/{total} OK) — no fatal pattern, continuing.")
    return None


def _probe(url: str, timeout: float = 5.0) -> str:
    parts = urlsplit(url)
    host, port = parts.hostname or "", parts.port or 443
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return f"reachable ({host}:{port})"
    except Exception as exc:  # network/VPN not available — reported, not fatal
        return f"UNREACHABLE ({host}:{port}) - {type(exc).__name__}: {exc}"


def preflight(*, check_network: bool = False, require_credentials: bool = False) -> bool:
    """Check local imports and configuration; network probing is opt-in."""
    ok = True
    print("=" * 72)
    print("PREFLIGHT / health check")
    print("=" * 72)
    print(f"interpreter : {sys.executable}")
    print(f"python      : {sys.version.split()[0]}")
    print(f"ARGO_API_USER: {os.environ.get('ARGO_API_USER', '(not configured)')}")
    print(f"workspace   : {WORKSPACE_ROOT}")

    from NIST_ThermoML_agents._NIST_ThermoML_main_agent.ThermoML_main_argo_config import AGENT_CONFIG as M
    from NIST_ThermoML_agents.NIST_ThermoML_analysis_agent.ThermoML_analysis_argo_config import AGENT_CONFIG as A
    from NIST_ThermoML_agents.NIST_ThermoML_query_agent.ThermoML_query_argo_config import AGENT_CONFIG as Q

    print("-" * 72)
    for name, cfg, comp in (("main", M, M.COMPACTION_INTERVAL),
                            ("query", Q, Q.L1_COMPACTION_INTERVAL),
                            ("analysis", A, A.COMPACTION_INTERVAL)):
        print(f"[{name:8}] MODEL={cfg.MODEL} API_USER={cfg.API_USER!r} "
              f"compaction={comp} MAX_TURN_SECONDS={cfg.MAX_TURN_SECONDS} "
              f"HTTP_TIMEOUT={cfg.HTTP_TIMEOUT}")
        if require_credentials and not USE_ANTHROPIC and not cfg.API_USER:
            print("           !! Set ARGO_API_USER before launching a live benchmark")
            ok = False
    print(f"API_URL     : {M.API_URL}")

    if USE_ANTHROPIC:
        print("-" * 72)
        from NIST_ThermoML_agents.general_db_query_engine.general_argo_engine_helpers import anthropic_argo_adapter as _adapter
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        print(f"ANTHROPIC_API_KEY: {'set (' + str(len(key)) + ' chars)' if key else 'MISSING'}")
        if require_credentials and not key:
            ok = False
        for name, cfg in (("main", M), ("query", Q), ("analysis", A)):
            print(f"[{name:8}] {cfg.MODEL} -> {_adapter.map_model(cfg.MODEL)}")

    print("-" * 72)
    for name, module, attr in _ENTRYPOINTS:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            print(f"[{name:8}] import OK  ({module}.{attr})")
        except Exception as exc:
            print(f"[{name:8}] import FAIL - {type(exc).__name__}: {exc}")
            ok = False

    if check_network:
        print(f"Argo host   : {_probe(M.API_URL)}")
    else:
        print("Network     : not checked (use --check-network to probe the Argo host)")

    print("-" * 72)
    print("Planned commands (priority order main -> query -> analysis):")
    for agent, module, ids, workers, gap in JOBS:
        cmd, out = build_cmd(agent, module, ids, workers, gap)
        matched, missing = prompt_selection(agent, module, ids)
        print(f"\n[{agent}]  prompts={len(matched)}  workers={workers}  gap={gap}s")
        if missing:
            print(f"  requested IDs absent from current prompt file: {', '.join(missing)}")
        print(f"  out: {out}")
        print("  cmd: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))

    print("\n" + "=" * 72)
    print("PREFLIGHT OK" if ok else "PREFLIGHT FAILED")
    print("=" * 72)
    return ok


def launch(*, check_network: bool = False) -> int:
    if not preflight(check_network=check_network, require_credentials=True):
        print("\nAborting: preflight failed.")
        return 1
    print("\n>>> LAUNCHING (--go) <<<\n")
    for agent, module, ids, workers, gap in JOBS:
        cmd, out = build_cmd(agent, module, ids, workers, gap)
        env = os.environ.copy()
        if USE_ANTHROPIC:
            env["ANTHROPIC_ADAPTER_CAPTURE"] = str(out / "anthropic_raw_capture.jsonl")
        print(f"\n===== {agent.upper()} -> {out} =====")
        result = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), env=env)
        if result.returncode != 0:
            print(f"\n[{agent}] runner exited {result.returncode}; stopping.")
            return result.returncode
        fatal = _check_fatal_pattern(agent, out)
        if fatal:
            print(f"\n{fatal}\n\nStopping before the next agent \u2014 this looks like a systemic "
                  f"algorithmic failure, not isolated content misses.")
            return 2
    print("\nAll agents complete.")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", action="store_true", help="Launch live provider-backed benchmarks")
    parser.add_argument("--anthropic", action="store_true", help="Use ANTHROPIC_API_KEY instead of Argo")
    parser.add_argument("--resume", metavar="TIMESTAMP", help="Resume an existing test_run timestamp")
    network = parser.add_mutually_exclusive_group()
    network.add_argument("--offline", action="store_true", help="Check local imports without network access (default)")
    network.add_argument("--check-network", action="store_true", help="Probe the configured Argo host")
    args = parser.parse_args(argv)
    if args.offline and args.go:
        parser.error("--offline cannot be combined with --go")
    global USE_ANTHROPIC, RESUME_TS, TS
    USE_ANTHROPIC = args.anthropic
    RESUME_TS = args.resume
    TS = RESUME_TS or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.go:
        return launch(check_network=args.check_network)
    ok = preflight(check_network=args.check_network)
    print("\n(no runs performed — re-run with --go to launch)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
