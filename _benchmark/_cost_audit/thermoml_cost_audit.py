"""ThermoML benchmark cost audit for saved full-capability answers.

Reads directories or benchmark ZIPs without extraction, including each answer's
nested worker captures. ARCHIVES.json restores logical collection paths; ZIP
filenames provide a fallback. Exact captured token counts calibrate character
estimates for uncaptured answers. Prices below are explicit historical assumptions,
not a current-price lookup. CSV estimated flags retain each row's provenance.

The default audit + layers steps write only three CSVs and four PNGs next to this
script. No cache simulations, per-call intermediates, or OriginPro exports are made.
Usage: python thermoml_cost_audit.py [audit | layers | all] [--root ROOT] [--out OUT]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from zipfile import BadZipFile

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from ThermoML_results_browser.helpers.benchmark_zip import ZipPath

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
BENCH = OUT.parent
PHASES = ("Main", "Analysis", "Query")


# ════ audit ═══════════════════════════════════════════════════════════════════════

# Assumed list prices, $/MTok (input, output) by Anthropic model prefix.
PRICES = {
    "claude-opus":   (5.00, 25.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-haiku":  (1.00, 5.00),
}
ARGO_TO_ANTHROPIC_PREFIX = {
    "claudeopus": "claude-opus", "claudesonnet": "claude-sonnet",
    "claudehaiku": "claude-haiku",
}


def price_for(model: str) -> tuple[float, float]:
    m = (model or "").lower()
    for prefix, p in PRICES.items():
        if m.startswith(prefix):
            return p
    for argo, prefix in ARGO_TO_ANTHROPIC_PREFIX.items():
        if m.startswith(argo):
            return PRICES[prefix]
    return PRICES["claude-opus"]  # conservative default


def cost_usd(tin: int, tout: int, model: str) -> float:
    ri, ro = price_for(model)
    return tin / 1e6 * ri + tout / 1e6 * ro


# ── per-call tier labels from the session's Argo transcript ─────────────

EVENT_ROW = re.compile(
    r"\|\s*E\d+\s*\|[^|]*\|[^|]*\|\s*argo\s*\|[^|]*\|\s*"
    r"([\d,]+)\u00b7([\d,]+)\u2192([\d,]+)\s*\|[^|]*\|\s*`?([^|`]*)`?\s*\|\s*"
    r"([^|\u00b7]+)\u00b7\s*([^|\u00b7]+)\u00b7[^|]*\|")


def load_tier_queues(sdir: Path) -> dict[tuple[int, int], list[tuple[int, str, str]]]:
    """(system_chars, prompt_chars) -> FIFO of (response_chars, tier, event_tag)
    from run_history_detailed.md, for joining capture records to worker tiers."""
    md = sdir / "run_history_detailed.md"
    queues: dict[tuple[int, int], list[tuple[int, str, str]]] = {}
    if not md.exists():
        return queues
    for line in md.read_text(encoding="utf-8", errors="replace").splitlines():
        m = EVENT_ROW.match(line.strip())
        if not m:
            continue
        s, p, r = (int(x.replace(",", "")) for x in m.groups()[:3])
        queues.setdefault((s, p), []).append((r, m.group(6).strip(), m.group(4).strip()))
    return queues


def assign_tier(queues, rec) -> tuple[str, str]:
    """Greedy join: exact (sys,prompt,resp) pops its event; a same-(sys,prompt)
    retry/error attempt inherits the labels without consuming the event."""
    q = queues.get((rec["system_chars"], rec["prompt_chars"]))
    if not q:
        return "?", ""
    for i, (r, tier, tag) in enumerate(q):
        if r == rec["response_chars"]:
            q.pop(i)
            return tier, tag
    return q[0][1], q[0][2]


# ── captured-call parsing ────────────────────────────────────────────────

def parse_capture(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        u = (rec.get("response") or {}).get("usage") or {}
        tin, tout = u.get("input_tokens") or 0, u.get("output_tokens") or 0
        model = rec.get("anthropic_model") or rec.get("argo_model") or ""
        rows.append({
            "seq": rec.get("seq"), "ts": rec.get("ts"),
            "elapsed_s": rec.get("elapsed_s"),
            "argo_model": rec.get("argo_model"),
            "anthropic_model": rec.get("anthropic_model"),
            "status": rec.get("status_code"),
            "system_chars": rec.get("system_chars") or 0,
            "prompt_chars": rec.get("prompt_chars") or 0,
            "response_chars": rec.get("response_chars") or 0,
            "input_tokens": tin, "output_tokens": tout,
            "cache_creation_tokens": u.get("cache_creation_input_tokens") or 0,
            "cache_read_tokens": u.get("cache_read_input_tokens") or 0,
            "cost_usd": cost_usd(tin, tout, model),
        })
    return rows


def sub_role(rel: Path) -> str:
    parts = rel.parts
    if "query_runs" in parts:
        return "query_sub"
    if "analysis_runs" in parts:
        return "analysis_sub"
    return "top"


# ── session discovery ────────────────────────────────────────────────────

_SESSION_NAME = re.compile(
    r"^(?:Q(?P<prompt>\d+(?:\.\d+)*)_)?run_"
    r"(?P<day>\d{8})_(?P<clock>\d{4}|\d{6})"
    r"(?:_\d{6}_[0-9a-fA-F]{8})?$"
)


def _session_timestamp_tag(name: str) -> str:
    """Compare chronology independently of prompt prefixes and removed hashes."""
    match = _SESSION_NAME.fullmatch(name)
    if not match:
        return name  # Query campaign children already use Q<id>.
    return f"{match.group('day')}_{match.group('clock').ljust(6, '0')}"


def _manifest_collections(root: Path) -> dict[str, PurePosixPath]:
    """Archive paths map to original collection paths, not multipart names."""
    manifest = root / "ARCHIVES.json"
    if not manifest.is_file():
        return {}
    document = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict) or not isinstance(document.get("archives"), list):
        raise ValueError(f"{manifest}: expected an archives list")
    mappings = {}
    for record in document["archives"]:
        if not isinstance(record, dict):
            raise ValueError(f"{manifest}: archive records must be objects")
        values = []
        for field in ("path", "source_collection"):
            value = record.get(field)
            if (not isinstance(value, str) or not value or "\\" in value or "\x00" in value
                    or PureWindowsPath(value).drive or value.startswith("/")
                    or any(part in {"", ".", ".."} for part in value.split("/"))):
                raise ValueError(f"{manifest}: invalid {field}: {value!r}")
            values.append(PurePosixPath(value))
        archive, collection = values
        if archive.suffix.lower() != ".zip":
            raise ValueError(f"{manifest}: archive path must end in .zip: {archive}")
        key = archive.as_posix()
        if key in mappings and mappings[key] != collection:
            raise ValueError(f"{manifest}: conflicting source collections for {archive}")
        mappings[key] = collection
    return mappings


def _logical_path(root: Path, folder: Path | ZipPath, mappings) -> PurePosixPath:
    if not isinstance(folder, ZipPath):
        return PurePosixPath(folder.relative_to(root).as_posix())
    archive = PurePosixPath(folder.archive.relative_to(root).as_posix())
    collection = mappings.get(archive.as_posix())
    if collection is None:
        stem = re.sub(r"__part\d+(?:__.*)?$", "", archive.stem, flags=re.IGNORECASE)
        collection = archive.parent if archive.parent.name == stem else archive.parent / stem
    return collection / folder.member


def active_sessions():
    """Full-capability answers once; workers are read only inside their parent.

    An unpacked answer takes precedence over an archived copy with the same
    original path. Separate dated campaigns remain separate answers.
    """
    root = BENCH.resolve()
    mappings = _manifest_collections(root)
    found = {}

    def walk(folder):
        if (folder / "result.md").is_file():
            logical = _logical_path(root, folder, mappings)
            if len(logical.parts) < 3 or logical.parts[0] not in PHASES or logical.parts[1] != "test_run_full_capability":
                return
            match = re.match(r"^Q(\d+(?:\.\d+)*)(?:_run_|$)", folder.name)
            if not match:
                raise ValueError(f"Unmapped active answer: {folder}")
            key = logical.as_posix()
            previous = found.get(key)
            if previous is None or (isinstance(previous[2], ZipPath) and not isinstance(folder, ZipPath)):
                found[key] = (logical.parts[0], match.group(1), folder)
            return
        for child in sorted(folder.iterdir(), key=lambda item: item.name):
            if child.name.startswith(("_", ".")):
                continue
            if not isinstance(child, ZipPath) and child.is_symlink():
                continue
            if child.is_dir():
                walk(child)
            elif not isinstance(child, ZipPath) and child.suffix.lower() == ".zip":
                try:
                    walk(ZipPath(child))
                except (OSError, ValueError, BadZipFile) as exc:
                    raise ValueError(f"Cannot read benchmark archive {child}: {exc}") from exc

    for phase in PHASES:
        folder = root / phase / "test_run_full_capability"
        if folder.is_dir():
            walk(folder)
    return [found[key] for key in sorted(found, key=lambda value: (PHASES.index(found[value][0]), value))]


def collect_sessions():
    """Captured active runs, including top-level and nested workers."""
    return [(phase, pid, folder) for phase, pid, folder in active_sessions()
            if any(folder.rglob("anthropic_raw_capture.jsonl"))]


# ── Uncaptured estimation from reference_stats tier tables ─────────────

TIER_ROW = re.compile(
    r"^\|\s*([^|*]+?)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|"
    r"\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*[\d,]+\s*\|\s*[\d.]+\s*\|\s*([^|]*)\|")


def parse_tier_table(md: Path) -> list[dict]:
    rows = []
    text = md.read_text(encoding="utf-8", errors="replace")
    sec = text.split("## 1. Argo API Call Summary", 1)[-1].split("\n## ", 1)[0]
    for line in sec.splitlines():
        m = TIER_ROW.match(line.strip())
        if not m or m.group(1).strip().upper() == "TIER":
            continue
        tier, calls, sysc, prc, rsc, sent = (m.group(1).strip(),
                                             *(int(x.replace(",", "")) for x in m.groups()[1:6]))
        model = (m.group(7) or "").strip().split(",")[0].strip()
        rows.append({"tier": tier, "calls": calls, "system_chars": sysc,
                     "prompt_chars": prc, "response_chars": rsc,
                     "sent_chars": sent, "model": model})
    return rows


def uncaptured_sessions():
    """Runs without raw captures use the recorded tier character counts."""
    return [(phase, pid, folder) for phase, pid, folder in active_sessions()
            if not any(folder.rglob("anthropic_raw_capture.jsonl"))]


# ── run it ───────────────────────────────────────────────────────────────

def run_audit() -> None:
    active = active_sessions()
    missing = set(PHASES) - {phase for phase, _, _ in active}
    if missing:
        raise ValueError(f"No full-capability answers for: {', '.join(sorted(missing))}")
    sessions = [(phase, pid, folder) for phase, pid, folder in active
                if any(folder.rglob("anthropic_raw_capture.jsonl"))]
    uncaptured = [(phase, pid, folder) for phase, pid, folder in active
                  if not any(folder.rglob("anthropic_raw_capture.jsonl"))]

    # 1. parse all captures, keyed per session
    call_rows, per_session = [], {}
    for phase, pid, sdir in sessions:
        tag = _session_timestamp_tag(sdir.name)
        agg = {}
        for cap in sorted(sdir.rglob("anthropic_raw_capture.jsonl")):
            role = sub_role(cap.parent.relative_to(sdir))
            queues = load_tier_queues(cap.parent)
            for r in parse_capture(cap):
                ev_tier, ev_tag = assign_tier(queues, r)
                r.update(phase=phase, prompt_id=pid, session=sdir.name, role=role,
                         tier=ev_tier, event_tag=ev_tag)
                call_rows.append(r)
                a = agg.setdefault(role, [0, 0, 0, 0.0])
                a[0] += 1; a[1] += r["input_tokens"]; a[2] += r["output_tokens"]
                a[3] += r["cost_usd"]
        per_session[(phase, pid, sdir.name)] = (tag, agg)

    # 2. canonical vs superseded (latest session per prompt wins)
    latest = {}
    for (phase, pid, sname), (tag, _) in per_session.items():
        k = (phase, pid)
        if k not in latest or tag > latest[k][0]:
            latest[k] = (tag, sname)
    superseded = {(p, i, s): latest[(p, i)][1] != s for (p, i, s) in per_session}

    # 3. calibration ratios from OK captured calls
    cin = ctin = cout = ctout = 0
    for r in call_rows:
        if r["status"] == 200 and r["input_tokens"]:
            cin += r["system_chars"] + r["prompt_chars"]; ctin += r["input_tokens"]
            cout += r["response_chars"]; ctout += r["output_tokens"]
    if not ctin or not ctout:
        raise ValueError("Calibration requires successful captured calls with nonzero input and output tokens")
    r_in, r_out = cin / ctin, cout / ctout
    print(f"calibration: {r_in:.3f} input chars/token, {r_out:.3f} output chars/token "
          f"(from {ctin:,} in-tok / {ctout:,} out-tok)")

    # 4. Uncaptured-run estimation
    est_rows, est_session = [], {}
    for phase, pid, sdir in uncaptured:
        agg = {}
        for md in sorted(sdir.rglob("reference_stats.md")):
            role = sub_role(md.parent.relative_to(sdir))
            for t in parse_tier_table(md):
                tin = round((t["system_chars"] + t["prompt_chars"]) / r_in)
                tout = round(t["response_chars"] / r_out)
                c = cost_usd(tin, tout, t["model"])
                est_rows.append({
                    "phase": phase, "prompt_id": pid, "session": sdir.name,
                    "role": role, "tier": t["tier"], "calls": t["calls"],
                    "model": t["model"],
                    "input_chars": t["system_chars"] + t["prompt_chars"],
                    "response_chars": t["response_chars"],
                    "est_input_tokens": tin, "est_output_tokens": tout,
                    "est_cost_usd": c,
                })
                a = agg.setdefault(role, [0, 0, 0, 0.0])
                a[0] += t["calls"]; a[1] += tin; a[2] += tout; a[3] += c
        est_session[(phase, pid, sdir.name)] = agg

    # 5. CSV exports
    # per worker-tier rollup (captured exact + log estimated)
    tier_agg: dict[tuple, list] = {}
    for r in call_rows:
        k = (r["phase"], r["prompt_id"], r["session"], r["role"], r["tier"], False)
        a = tier_agg.setdefault(k, [0, 0, 0, 0.0])
        a[0] += 1; a[1] += r["input_tokens"]; a[2] += r["output_tokens"]
        a[3] += r["cost_usd"]
    for e in est_rows:
        k = (e["phase"], e["prompt_id"], e["session"], e["role"], e["tier"], True)
        a = tier_agg.setdefault(k, [0, 0, 0, 0.0])
        a[0] += e["calls"]; a[1] += e["est_input_tokens"]
        a[2] += e["est_output_tokens"]; a[3] += e["est_cost_usd"]
    tier_rows = [{"phase": p, "prompt_id": i, "session": s, "role": ro,
                  "tier": t, "estimated": est, "calls": a[0],
                  "input_tokens": a[1], "output_tokens": a[2],
                  "cost_usd": round(a[3], 4)}
                 for (p, i, s, ro, t, est), a in sorted(tier_agg.items())]
    with open(OUT / "per_tier_costs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tier_rows[0]))
        w.writeheader(); w.writerows(tier_rows)

    prompt_rows = []
    for (phase, pid, sname), (tag, agg) in sorted(per_session.items()):
        tot = [sum(a[i] for a in agg.values()) for i in range(4)]
        prompt_rows.append({
            "phase": phase, "prompt_id": pid, "session": sname,
            "estimated": False, "superseded": superseded[(phase, pid, sname)],
            "calls": tot[0], "input_tokens": tot[1], "output_tokens": tot[2],
            "cost_total_usd": round(tot[3], 4),
            "cost_top_usd": round(agg.get("top", [0]*4)[3], 4),
            "cost_query_subs_usd": round(agg.get("query_sub", [0]*4)[3], 4),
            "cost_analysis_subs_usd": round(agg.get("analysis_sub", [0]*4)[3], 4),
        })
    for (phase, pid, sname), agg in sorted(est_session.items()):
        tot = [sum(a[i] for a in agg.values()) for i in range(4)]
        prompt_rows.append({
            "phase": phase, "prompt_id": pid, "session": sname,
            "estimated": True, "superseded": False,
            "calls": tot[0], "input_tokens": tot[1], "output_tokens": tot[2],
            "cost_total_usd": round(tot[3], 4),
            "cost_top_usd": round(agg.get("top", [0]*4)[3], 4),
            "cost_query_subs_usd": round(agg.get("query_sub", [0]*4)[3], 4),
            "cost_analysis_subs_usd": round(agg.get("analysis_sub", [0]*4)[3], 4),
        })
    # Put captured and estimated rows in the same question order.
    prompt_rows.sort(key=lambda row: (
        row["phase"], tuple(int(part) for part in row["prompt_id"].split(".")),
        _session_timestamp_tag(row["session"]),
    ))
    with open(OUT / "per_prompt_costs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prompt_rows[0]))
        w.writeheader(); w.writerows(prompt_rows)

    # 6. per-phase stats (canonical, non-superseded; uncaptured estimates included)
    canon = [r for r in prompt_rows if not r["superseded"]]
    stat_rows = []
    for phase in ("Main", "Analysis", "Query"):
        costs = [r["cost_total_usd"] for r in canon if r["phase"] == phase]
        subs = [r["cost_query_subs_usd"] + r["cost_analysis_subs_usd"]
                for r in canon if r["phase"] == phase]
        stat_rows.append({
            "phase": phase, "prompts": len(costs), "total_usd": round(sum(costs), 2),
            "mean_usd": round(stats.mean(costs), 3),
            "median_usd": round(stats.median(costs), 3),
            "min_usd": round(min(costs), 3), "max_usd": round(max(costs), 3),
            "stdev_usd": round(stats.stdev(costs), 3) if len(costs) > 1 else 0.0,
            "subagent_share": round(sum(subs) / sum(costs), 3) if sum(costs) else 0.0,
        })
    with open(OUT / "phase_stats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stat_rows[0]))
        w.writeheader(); w.writerows(stat_rows)

    # 7. plots
    def pkey(pid):
        try:
            return [int(x) for x in pid.split(".")]
        except ValueError:
            return [99]

    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    colors = {"top": "#4c72b0", "query_sub": "#dd8452", "analysis_sub": "#55a868"}
    for ax, phase in zip(axes, ("Main", "Analysis", "Query")):
        rows = sorted((r for r in canon if r["phase"] == phase),
                      key=lambda r: pkey(r["prompt_id"]))
        xs = range(len(rows))
        bot = [0.0] * len(rows)
        # Only Main separates launched subagents; other panels show session totals.
        segments = (("cost_top_usd", "agent itself", "top"),
                    ("cost_query_subs_usd", "query subagents", "query_sub"),
                    ("cost_analysis_subs_usd", "analysis subagents", "analysis_sub")) if phase == "Main" else (
                    ("cost_total_usd", "agent total", "top"),)
        for part, lbl, role in segments:
            vals = [r[part] for r in rows]
            ax.bar(xs, vals, bottom=bot, color=colors[role], label=lbl,
                   hatch=None, edgecolor="white", linewidth=0.4)
            bot = [b + v for b, v in zip(bot, vals)]
        ax.set_xticks(list(xs))
        ax.set_xticklabels([r["prompt_id"] + ("*" if r["estimated"] else "")
                            for r in rows], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("cost (USD)")
        tot = sum(r["cost_total_usd"] for r in rows)
        ax.set_title(f"{phase} agent — per-prompt Anthropic cost "
                     f"({len(rows)} prompts, total ${tot:.2f})", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].text(0.995, 0.95, "* = estimated from Argo char transcript\n"
                 "(no raw capture; char-calibrated estimates)",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=8,
                 bbox=dict(fc="lightyellow", ec="gray"))
    fig.suptitle("ThermoML benchmark — Anthropic API cost per prompt "
                 "(opus $5/$25, sonnet $3/$15 per MTok assumed)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "cost_per_prompt.png", dpi=200)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    data = [[r["cost_total_usd"] for r in canon if r["phase"] == p]
            for p in ("Main", "Analysis", "Query")]
    ax2.boxplot(data, tick_labels=["Main", "Analysis", "Query"], showmeans=True)
    for i, d in enumerate(data, 1):
        ax2.scatter([i + 0.08] * len(d), d, s=14, alpha=0.6, color="#4c72b0")
    ax2.set_ylabel("per-prompt cost (USD)")
    ax2.set_title("Per-prompt cost distribution by agent phase")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT / "cost_distributions.png", dpi=200)
    plt.close(fig2)

    # worker-layer decomposition per phase (canonical sessions only)
    canon_keys = {(r["phase"], r["prompt_id"], r["session"]) for r in canon}
    phase_tier: dict[str, dict[str, float]] = {}
    for t in tier_rows:
        if (t["phase"], t["prompt_id"], t["session"]) not in canon_keys:
            continue
        label = t["tier"] if t["role"] == "top" else f"{t['role'].split('_')[0]}-sub {t['tier']}"
        phase_tier.setdefault(t["phase"], {}).setdefault(label, 0.0)
        phase_tier[t["phase"]][label] += t["cost_usd"]
    fig3, ax3 = plt.subplots(figsize=(11, 6))
    phases = ("Main", "Analysis", "Query")
    labels = sorted({l for d in phase_tier.values() for l in d},
                    key=lambda l: (-max(d.get(l, 0) for d in phase_tier.values())))
    cmap = plt.get_cmap("tab20")
    bot = [0.0] * len(phases)
    for j, lbl in enumerate(labels):
        vals = [phase_tier.get(p, {}).get(lbl, 0.0) for p in phases]
        ax3.bar(phases, vals, bottom=bot, label=lbl, color=cmap(j % 20),
                edgecolor="white", linewidth=0.4)
        for i, (b, v) in enumerate(zip(bot, vals)):
            if v > 3.5:
                ax3.text(i, b + v / 2, f"{lbl}\n${v:.0f}", ha="center",
                         va="center", fontsize=7)
        bot = [b + v for b, v in zip(bot, vals)]
    ax3.set_ylabel("phase total cost (USD)")
    ax3.set_title("Worker-layer cost decomposition per phase (canonical runs; "
                  "includes nested sessions and uncaptured-run estimates)")
    ax3.legend(fontsize=7, ncols=2, loc="upper right")
    ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(OUT / "cost_worker_layers.png", dpi=200)
    plt.close(fig3)

    # 8. console summary
    print(f"\ncaptured calls: {len(call_rows):,}   sessions: {len(per_session)}"
          f"   estimated sessions: {len(est_session)}")
    for s in stat_rows:
        print(f"  {s['phase']:<9} n={s['prompts']:>2}  total=${s['total_usd']:>7.2f}  "
              f"mean=${s['mean_usd']:>6.3f}  median=${s['median_usd']:>6.3f}  "
              f"range=${s['min_usd']:.2f}–${s['max_usd']:.2f}  "
              f"subagent share={s['subagent_share']:.0%}")
    sup = [(p, i, s) for (p, i, s), v in superseded.items() if v]
    if sup:
        print("superseded sessions (excluded from stats, kept in CSVs):")
        for p, i, s in sup:
            print(f"  {p} {i}: {s}")
    unk = sorted({(r["phase"], r["prompt_id"]) for r in prompt_rows
                  if str(r["prompt_id"]).startswith("?")})
    if unk:
        print("UNMAPPED sessions:", unk)
    n_unjoined = sum(1 for r in call_rows if r["tier"] == "?")
    print(f"tier join: {len(call_rows) - n_unjoined}/{len(call_rows)} capture "
          f"records matched to transcript tiers")
    print("\nworker-layer cost by phase (canonical):")
    for p in phases:
        for lbl, v in sorted(phase_tier.get(p, {}).items(), key=lambda kv: -kv[1]):
            print(f"  {p:<9} {lbl:<28} ${v:8.2f}")
    print(f"\noutputs -> {OUT}")


# ════ layers (horizontal figure) ═══════════════════════════════════════

def run_layers() -> None:
    """Horizontal layer figure from per_prompt_costs.csv and per_tier_costs.csv."""
    MAIN_FAMS = ["1.1", "1.2", "2", "3", "4"]
    Q_FAMS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]


    def family(phase: str, pid: str) -> str:
        if phase == "Main":
            return pid if pid.startswith(("1.1", "1.2")) and pid.count(".") >= 2 \
                else pid.split(".")[0]
        return "Q" + pid.split(".")[0]


    def fam_of(phase, pid):
        f = family(phase, pid)
        return ".".join(f.split(".")[:2]) if phase == "Main" and f.count(".") >= 2 else f


    # ── load canonical rows ──────────────────────────────────────────────────

    with open(OUT / "per_prompt_costs.csv", encoding="utf-8") as fh:
        pp = [r for r in csv.DictReader(fh) if r["superseded"] == "False"]
    canon_sessions = {(r["phase"], r["prompt_id"], r["session"]) for r in pp}

    with open(OUT / "per_tier_costs.csv", encoding="utf-8") as fh:
        pt = [r for r in csv.DictReader(fh)
              if (r["phase"], r["prompt_id"], r["session"]) in canon_sessions]

    # per-prompt totals + role components
    prompts: dict[tuple, dict] = {}
    for r in pp:
        k = (r["phase"], r["prompt_id"])
        prompts[k] = {
            "family": fam_of(r["phase"], r["prompt_id"]),
            "estimated": r["estimated"] == "True",
            "total": float(r["cost_total_usd"]),
            "top": float(r["cost_top_usd"] if r["phase"] == "Main" else r["cost_total_usd"]),
            "query_sub": float(r["cost_query_subs_usd"]),
            "analysis_sub": float(r["cost_analysis_subs_usd"]),
        }


    def tier_bucket(phase: str, role: str, tier: str) -> str:
        if phase != "Main":
            return tier if tier in ("L0-main", "L1-worker") else "other"
        pre = {"top": "own", "query_sub": "q-sub", "analysis_sub": "a-sub"}[role]
        if tier == "L0-main":
            return f"{pre} L0-main"
        if tier == "L1-worker":
            return f"{pre} L1-worker"
        return f"{pre} misc"


    tiers_per_prompt: dict[tuple, dict] = defaultdict(lambda: defaultdict(float))
    for r in pt:
        k = (r["phase"], r["prompt_id"])
        tiers_per_prompt[k][tier_bucket(r["phase"], r["role"], r["tier"])] += float(r["cost_usd"])

    SEGS_ROLE = {"Main": [("top", "agent itself"), ("query_sub", "query subagents"),
                          ("analysis_sub", "analysis subagents")],
                 "Analysis": [("top", "agent total")],
                 "Query": [("top", "agent total")]}
    SEGS_TIER = {
        "Main": ["own L0-main", "own L1-worker", "own misc",
                 "q-sub L0-main", "q-sub L1-worker", "q-sub misc",
                 "a-sub L0-main", "a-sub L1-worker", "a-sub misc"],
        "Analysis": ["L0-main", "L1-worker", "other"],
        "Query": ["L0-main", "L1-worker", "other"],
    }
    ROLE_COLORS = {"top": "#4c72b0", "query_sub": "#dd8452", "analysis_sub": "#55a868"}
    TIER_COLORS = {
        "own L0-main": "#2f4b7c", "own L1-worker": "#6e8fc9", "own misc": "#b9c9e8",
        "q-sub L0-main": "#b35a1f", "q-sub L1-worker": "#e8945a", "q-sub misc": "#f5c9a8",
        "a-sub L0-main": "#2e6b45", "a-sub L1-worker": "#6fb98f", "a-sub misc": "#bfe3cf",
        "L0-main": "#2f4b7c", "L1-worker": "#6e8fc9", "other": "#b9c9e8",
    }


    def fam_stats(vals):
        v = np.array(vals, float)
        return dict(n=len(v), mean=v.mean(), median=np.median(v),
                    p10=np.percentile(v, 10), p90=np.percentile(v, 90),
                    min=v.min(), max=v.max())


    # ── assemble per (agent, mode, family): segment means + totals ───────────

    panel_data = {}
    for phase, fams in (("Main", MAIN_FAMS), ("Analysis", Q_FAMS), ("Query", Q_FAMS)):
        for mode in ("sessions", "workers"):
            per_fam = {}
            for fam in fams:
                ks = [k for k, p in prompts.items() if k[0] == phase and p["family"] == fam]
                if not ks:
                    continue
                totals = [prompts[k]["total"] for k in ks]
                if mode == "sessions":
                    segs = {lbl: float(np.mean([prompts[k][key] for k in ks]))
                            for key, lbl in SEGS_ROLE[phase]}
                    colors = {lbl: ROLE_COLORS[key] for key, lbl in SEGS_ROLE[phase]}
                else:
                    segs = {b: float(np.mean([tiers_per_prompt[k].get(b, 0.0) for k in ks]))
                            for b in SEGS_TIER[phase]}
                    segs = {b: v for b, v in segs.items() if v > 1e-9}
                    colors = {b: TIER_COLORS[b] for b in segs}
                st = fam_stats(totals)
                per_fam[fam] = dict(segs=segs, colors=colors, stats=st,
                                    dots=[(prompts[k]["total"], prompts[k]["estimated"],
                                           k[1]) for k in ks])
            panel_data[(phase, mode)] = per_fam

    # ── figure ───────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    MODES = (("sessions", "by subagent sessions"), ("workers", "by worker layers"))
    for row, phase in enumerate(("Main", "Analysis", "Query")):
        xmax = 0.0
        for col, (mode, mode_lbl) in enumerate(MODES):
            ax = axes[row][col]
            per_fam = panel_data[(phase, mode)]
            fams = [f for f in (MAIN_FAMS if phase == "Main" else Q_FAMS) if f in per_fam]
            ys = np.arange(len(fams))[::-1]  # first family on top
            seg_names = list(dict.fromkeys(s for f in fams for s in per_fam[f]["segs"]))
            left = np.zeros(len(fams))
            for s in seg_names:
                vals = np.array([per_fam[f]["segs"].get(s, 0.0) for f in fams])
                seg_color = next((per_fam[f]["colors"][s] for f in fams
                                  if s in per_fam[f]["colors"]), "#999999")
                ax.barh(ys, vals, left=left, height=0.52, color=seg_color,
                        label=s, edgecolor="white", linewidth=0.5)
                left += vals
            for y, f in zip(ys, fams):
                st = per_fam[f]["stats"]
                # stats symbols on the bar centerline
                ax.plot(st["median"], y, marker="|", ms=22, mew=2.4, color="black")
                ax.plot(st["mean"], y, marker="D", ms=7, color="black",
                        mfc="white", mew=1.6)
                ax.plot(st["p10"], y, marker=">", ms=6, color="black", mfc="none", mew=1.2)
                ax.plot(st["p90"], y, marker="<", ms=6, color="black", mfc="none", mew=1.2)
                for edge in (st["min"], st["max"]):
                    ax.plot(edge, y, marker="x", ms=6, color="#b22222", mew=1.6)
                # per-prompt total dots in a band above the bar
                dots = per_fam[f]["dots"]
                off = np.linspace(-0.07, 0.07, len(dots)) if len(dots) > 1 else [0.0]
                for (tot, est, _pid), dy in zip(sorted(dots), off):
                    ax.plot(tot, y + 0.38 + dy, marker="o", ms=4.5,
                            color="#333333", mfc="none" if est else "#333333",
                            mew=1.1, alpha=0.85, linestyle="none")
                xmax = max(xmax, st["max"])
            ax.set_yticks(ys)
            ax.set_yticklabels([f"{f}  (n={per_fam[f]['stats']['n']})" for f in fams],
                               fontsize=9)
            ax.set_title(f"{phase} — {mode_lbl}", fontsize=11)
            ax.grid(axis="x", alpha=0.3)
            ax.legend(fontsize=7, loc="lower right", ncols=1 if col == 0 else 2)
            if row == 2:
                ax.set_xlabel("cost per prompt (USD)")
        for col in (0, 1):
            axes[row][col].set_xlim(0, xmax * 1.08)

    sym = [Line2D([], [], marker="o", ls="none", color="#333", mfc="#333", ms=5,
                  label="prompt total"),
           Line2D([], [], marker="o", ls="none", color="#333", mfc="none", ms=5,
                  label="prompt total (estimated)"),
           Line2D([], [], marker="D", ls="none", color="black", mfc="white", ms=7,
                  label="mean (= bar length)"),
           Line2D([], [], marker="|", ls="none", color="black", ms=12, mew=2.4,
                  label="median"),
           Line2D([], [], marker=">", ls="none", color="black", mfc="none", ms=6,
                  label="p10"),
           Line2D([], [], marker="<", ls="none", color="black", mfc="none", ms=6,
                  label="p90"),
           Line2D([], [], marker="x", ls="none", color="#b22222", ms=6,
                  label="min / max edge cases")]
    fig.legend(handles=sym, loc="lower center", ncols=7, fontsize=8, frameon=False)
    fig.suptitle("ThermoML benchmark — layer-aggregated Anthropic cost per prompt\n"
                 "stacked segments = family means of component costs "
                 "(opus $5/$25, sonnet $3/$15 per MTok assumed)", fontsize=12)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    fig.savefig(OUT / "cost_layers_horizontal.png", dpi=200)
    plt.close(fig)
    print("figure -> cost_layers_horizontal.png")


STEPS = {"audit": run_audit, "layers": run_layers}
DEFAULT_STEPS = ("audit", "layers")


def main(argv: list[str] | None = None) -> None:
    """Generate only the published cost figures and aggregate CSVs."""
    global BENCH, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", nargs="*", help="audit, layers, or all (default: audit layers)")
    parser.add_argument("--root", type=Path, default=BENCH, help="benchmark root containing Main, Analysis and Query")
    parser.add_argument("--out", type=Path, default=OUT, help="output directory for the seven cost-audit artifacts")
    args = parser.parse_args(argv)
    steps = [step.lower() for step in args.steps] or list(DEFAULT_STEPS)
    if steps == ["all"]:
        steps = list(DEFAULT_STEPS)
    unknown = [step for step in steps if step not in STEPS]
    if unknown:
        parser.error(f"unknown step(s) {unknown}; choose audit, layers, or all")
    BENCH, OUT = args.root.resolve(), args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    for step in steps:
        print(f"\n=== {step} ===", flush=True)
        STEPS[step]()


if __name__ == "__main__":
    main()
