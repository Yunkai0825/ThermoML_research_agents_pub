"""Rendering + TSV export for projected graphs.

TSV contract (kept): Source/Target/Value first, then source_full/
target_full, flow_type, ids, provenance, edge_order.  The final
appearance_order/appearance_key columns enumerate the unique node keys
in first-seen source_full-then-target_full order.  A sibling nodes TSV
carries per-node full id/block lists, dispatch args and artifact anchors.
"""

from __future__ import annotations

import csv
import logging
import os
import threading
from pathlib import Path

import plotly.graph_objects as go

from funnel_assembly.projections import ProjectedGraph

log = logging.getLogger("workflow-png")

NODE_COLORS = {
    "raw": "#4C78A8",
    "hardcoded": "#2F855A",
    "agentic": "#59A14F",
    "validation": "#22A3C4",
    "delegated": "#9C755F",
    "merge": "#8064A2",
    "answer": "#F28E2B",
    "main": "#EDC948",
    "removed": "#B8B8B8",
    "introduced": "#E15759",
    "synth": "#C77CA6",
    "envelope": "#B08C2E",
    "hardmeta": "#8C6D52",
    "payload": "#7D69B9",
}
LINK_COLORS = {
    "retained": "rgba(65, 130, 90, 0.55)",
    "context": "rgba(76, 120, 168, 0.48)",
    "deduplicated": "rgba(128, 100, 162, 0.45)",
    "removed": "rgba(150, 150, 150, 0.28)",
    "introduced": "rgba(225, 87, 89, 0.38)",
    "synthesis": "rgba(242, 142, 43, 0.50)",
    "relay": "rgba(90, 140, 180, 0.45)",
    "metadata": "rgba(140, 109, 82, 0.45)",
    "memory": "rgba(125, 105, 185, 0.45)",
}

_UNITS = {"chars": "chars", "ids": "IDs", "blocks": "blocks", "pts": "pts"}

LEGEND = (
    "<br><sup>Registry-driven funnel: every tool chain and child relay"
    " merges into ITS CALLER's agent context (the id dedup layer;"
    " duplicate copies exit via the dedup sink) \u2192 synthesis \u2192 answer"
    " \u2192 envelope parts (verbatim answer + LLM-written fields +"
    " WM-assembled fields + DB-hydrated fields + deterministic"
    " inspection ledger + JSON scaffolding) \u2192 deliverable to the"
    " dispatching parent. Stage sizes are the LOGGED pipeline-table"
    " values; pre-delivery id content is re-executed from the verbatim"
    " logged arguments (pure DB tools only \u2014 stateful fit tools"
    " inherit det ids). Menu calls (run_subagent_tool) are the"
    " CALLER's tools, compacted by the owning catalog's pipeline."
    " DISCARD stubs enter context as chars only. STRICT CONSERVATION:"
    " zero-inflow nodes are only tool calls and supportive contexts;"
    " every lane draws from its context budget (each id exits once,"
    " shortfalls come from declared sources); WM recording is a"
    " chars-only copy drawn out of the context (unshipped payload"
    " exits via the WM-retained sink). blocks/pts views"
    " canonicalize bare block ids onto their unique qualified twin"
    " and weight by DB datapoint counts.</sup>")

MODE_NOTES = {
    "detailed": "<br><sup>DETAILED: one chain per logged tool call,"
                " nodes keyed by instance path (delegation-tree"
                " scoped; no cross-lane merging).</sup>",
    "agentic": "<br><sup>AGENTIC: detailed nodes merged per"
               " (instance \u00d7 tool family \u00d7 stage); support per class;"
               " sinks per family. Parallel contributions stack —"
               " dedup only at the agent contexts.</sup>",
    "pooled": "<br><sup>POOLED: same-type nodes of each delegation"
              " LAYER merged (main / Q / A / L1 / L2): one context,"
              " answer, merged envelope and tool rail per layer, plus"
              " global pools — supporting content (briefs · seeds ·"
              " scaffolding · wrappers · utility/error returns ·"
              " rendering expansion · LLM additions), memory carry"
              " (WM render · transient), and ONE deterministic"
              " DB-evidence pool"
              " across all agents (inspection tools + ledger + DB"
              " hydration / enrichment, each inspected id exits once)."
              " In-flows ADD per-instance copies; ids/blocks/pts"
              " deduplicate ONLY at each layer's agent context —"
              " answers show the deduplicated union. Cross-layer"
              " sinks: ONE hardcoded compaction cut, ONE agentic"
              " triage cut, ONE condensed-in-synthesis, ONE"
              " not-cited-downstream, ONE dedup (re-confirmed known"
              " IDs) and ONE memory-only sink (WM retained + uncited"
              " memory carry + uncited ledger).</sup>",
}

# PNG export goes through kaleido/Chromium, whose global scope is not
# thread-safe: when an agent thread renders plots concurrently it can
# corrupt the scope and make an export block forever, wedging the
# process-global generation lock and thus the whole run.  Every render
# is bounded by a timeout.  The first hang trips the breaker of the
# workflow being built, so that workflow's later figures skip PNG at
# once; the next workflow build starts with a fresh breaker (kaleido
# runs each export in its own browser, so a hung export does not block
# the next one).  The interactive HTML is always written.  Every figure
# left without a PNG is logged and listed in PNG_SKIPPED_TSV in the
# workflow folder.  THERMOML_WORKFLOW_PNG=0 disables PNG up front and
# THERMOML_WORKFLOW_PNG_TIMEOUT sets the timeout in seconds (default
# 45); both are read when a workflow build starts.
PNG_SKIPPED_TSV = "png_skipped.tsv"
PNG_SKIPPED_COLUMNS = ["figure", "png", "reason", "detail", "older_png"]


class PngExport:
    """PNG export state of ONE workflow build (one workflow folder)."""

    def __init__(self) -> None:
        self.timeout_s = float(
            os.environ.get("THERMOML_WORKFLOW_PNG_TIMEOUT", "45"))
        self.switched_off = os.environ.get(
            "THERMOML_WORKFLOW_PNG", "").strip().lower() in (
                "0", "false", "no", "off")
        self.tripped_by = ""          # the PNG whose export hung
        self.figures = 0
        self.skipped: list[dict] = []

    def export(self, figure, png: Path) -> Path | None:
        """Best-effort PNG via kaleido; returns the path or None.  The
        image is rendered in a daemon thread bounded by ``timeout_s`` and
        written here, so an export that hangs never writes a file later."""
        self.figures += 1
        if self.switched_off:
            return self._skip(png, "switched off",
                              "THERMOML_WORKFLOW_PNG turns PNG export off")
        if self.tripped_by:
            return self._skip(png, "breaker",
                              f"an earlier PNG of this workflow timed out"
                              f" ({self.tripped_by})")
        box: dict = {}

        def _run() -> None:
            try:
                box["png"] = figure.to_image(format="png", scale=2)
            except Exception as exc:  # noqa: BLE001 — reported below
                box["error"] = exc

        worker = threading.Thread(target=_run, name="kaleido-png",
                                  daemon=True)
        worker.start()
        worker.join(self.timeout_s)
        if worker.is_alive():
            self.tripped_by = png.name
            return self._skip(png, "timeout",
                              f"no image after {self.timeout_s:g}s; the"
                              f" rest of this workflow skips PNG")
        if "error" in box:
            return self._skip(png, "error", f"export failed: {box['error']}")
        try:
            png.write_bytes(box["png"])
        except OSError as exc:
            return self._skip(png, "error", f"write failed: {exc}")
        return png

    def _skip(self, png: Path, reason: str, detail: str) -> None:
        older = png.exists()
        self.skipped.append({
            "figure": png.with_suffix(".html").name, "png": png.name,
            "reason": reason, "detail": detail,
            "older_png": "yes" if older else "no",
        })
        level = (logging.INFO if reason in ("switched off", "breaker")
                 else logging.WARNING)
        log.log(level, "PNG skipped (%s): %s: %s; HTML kept%s", reason,
                png.name, detail,
                "; an older PNG from an earlier build is still there"
                if older else "")
        return None

    def finish(self, out_dir: Path) -> Path | None:
        """List this build's figures without a PNG in the workflow folder
        (a list left by an earlier build is removed when every figure got
        its PNG); returns the list's path, or None."""
        listing = Path(out_dir) / PNG_SKIPPED_TSV
        if not self.skipped:
            if listing.exists():
                listing.unlink()
            return None
        _write(listing, self.skipped, PNG_SKIPPED_COLUMNS)
        deliberate = all(r["reason"] == "switched off" for r in self.skipped)
        log.log(logging.INFO if deliberate else logging.WARNING,
                "%d of %d workflow figures have no PNG; listed in %s",
                len(self.skipped), self.figures, listing)
        return listing


def render_figure(graph: ProjectedGraph, title: str, stem: str,
                  out_dir: Path, mode: str = "",
                  width: int = 1900, height: int = 1080,
                  png_export: PngExport | None = None) -> list[Path]:
    """Write ``<stem>.html`` and, best effort, ``<stem>.png``.  Pass the
    build's ``png_export`` so one breaker and one skip list cover the
    whole workflow; without it the figure gets its own (logged, not
    listed)."""
    unit = _UNITS[graph.metric]
    nodes = sorted(graph.nodes.values(), key=lambda n: (n.order, n.key))
    index = {n.key: i for i, n in enumerate(nodes)}
    edges = graph.edges
    figure = go.Figure(go.Sankey(
        domain=dict(x=[0, 1], y=[0.02, 0.96]),
        arrangement="snap",
        orientation="h",
        valueformat=",d",
        valuesuffix=f" {unit}",
        node=dict(
            pad=12, thickness=16,
            line=dict(color="rgba(55,55,55,0.65)", width=0.6),
            label=[f"{n.label}<br><b>{n.value:,}</b> {unit}" for n in nodes],
            color=[NODE_COLORS.get(n.kind, "#999999") for n in nodes],
            customdata=[_hover(n) for n in nodes],
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=[index[e.source] for e in edges],
            target=[index[e.target] for e in edges],
            value=[e.value for e in edges],
            color=[LINK_COLORS.get(e.flow, "rgba(120,120,120,0.35)")
                   for e in edges],
            customdata=[e.flow for e in edges],
            hovertemplate=("%{source.label} \u2192 %{target.label}"
                           "<br>%{value:,} " + unit +
                           "<br>%{customdata}<extra></extra>"),
        ),
    ))
    figure.update_layout(
        title=title + LEGEND + MODE_NOTES.get(mode, ""),
        font=dict(family="Segoe UI, Arial", size=11),
        width=width, height=height,
        margin=dict(l=16, r=16, t=150, b=18),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / f"{stem}.html"
    png = out_dir / f"{stem}.png"
    figure.write_html(str(html), include_plotlyjs="cdn")
    outputs = [html]
    saved = (png_export or PngExport()).export(figure, png)
    if saved is not None:
        outputs.append(saved)
    return outputs


def _hover(node) -> str:
    ids = sorted(node.ids)
    shown = ", ".join(ids[:12]) + (f" \u2026 +{len(ids) - 12}"
                                   if len(ids) > 12 else "")
    prov = "<br>".join(node.provenance[:4])
    args = str(node.meta.get("args", ""))[:220]
    parts = [node.key, f"{node.value:,}"]
    if shown:
        parts.append(f"ids: {shown}")
    if args:
        parts.append(f"args: {args}\u2026" if len(args) == 220 else
                     f"args: {args}")
    if prov:
        parts.append(prov)
    return "<br>".join(parts)


EDGE_COLUMNS = ["Source", "Target", "Value", "source_full", "target_full",
                "flow_type", "ids", "provenance", "edge_order",
                "appearance_order", "appearance_key"]
NODE_COLUMNS = ["key", "label", "kind", "role", "is_source", "is_sink",
                "value", "n_ids", "ids", "blocks", "order", "provenance",
                "args", "meta"]


def write_edge_tsv(graph: ProjectedGraph, path: Path) -> Path:
    rows = []
    for order, e in enumerate(graph.edges, 1):
        src, tgt = graph.nodes[e.source], graph.nodes[e.target]
        rows.append({
            "Source": src.label, "Target": tgt.label,
            "Value": e.value,
            # node KEYS: labels can repeat (e.g. ok- and error-branch
            # calls of one tool) — keys keep flow accounting unambiguous
            "source_full": e.source, "target_full": e.target,
            "flow_type": e.flow,
            "ids": ";".join(sorted(e.ids)),
            "provenance": " | ".join(e.provenance),
            "edge_order": order,
        })
    _append_appearance(rows)
    _write(path, rows, EDGE_COLUMNS)
    _write(path.with_name(f"{path.stem}_Origin_ready.tsv"),
           _origin_ready_rows(graph), EDGE_COLUMNS)
    return path


def _origin_ready_rows(graph: ProjectedGraph) -> list[dict]:
    """OriginPro-ready view: parallel edges with the same source and
    target merged into ONE row (values sum, ids union, flows and
    provenance joined distinct, in first-appearance order)."""
    merged: dict = {}
    for e in graph.edges:
        entry = merged.setdefault((e.source, e.target), {
            "value": 0, "ids": set(), "flows": [], "prov": []})
        entry["value"] += e.value
        entry["ids"] |= e.ids
        if e.flow not in entry["flows"]:
            entry["flows"].append(e.flow)
        for p in e.provenance:
            if p not in entry["prov"]:
                entry["prov"].append(p)
    rows = []
    for order, ((src_key, tgt_key), entry) in enumerate(merged.items(), 1):
        src, tgt = graph.nodes[src_key], graph.nodes[tgt_key]
        rows.append({
            "Source": src.label, "Target": tgt.label,
            "Value": entry["value"],
            "source_full": src_key, "target_full": tgt_key,
            "flow_type": ";".join(entry["flows"]),
            "ids": ";".join(sorted(entry["ids"])),
            "provenance": " | ".join(entry["prov"]),
            "edge_order": order,
        })
    _append_appearance(rows)
    return rows


def _append_appearance(rows: list[dict]) -> None:
    """Ranking of the node keys by first appearance, as two parallel
    columns alongside the edge list."""
    appearance = []
    seen = set()
    for row in rows:
        for key in (row["source_full"], row["target_full"]):
            if key not in seen:
                seen.add(key)
                appearance.append(key)
    # A sparse graph can have more unique endpoint labels than edges
    # (for example, one edge has two labels).  Preserve the complete
    # appearance list in metadata-only rows rather than truncating it.
    rows.extend({} for _ in range(len(appearance) - len(rows)))
    for order, key in enumerate(appearance, 1):
        rows[order - 1]["appearance_order"] = order
        rows[order - 1]["appearance_key"] = key


def write_node_tsv(graph: ProjectedGraph, path: Path) -> Path:
    rows = []
    for node in sorted(graph.nodes.values(), key=lambda n: n.order):
        meta = {k: v for k, v in node.meta.items() if k != "args"}
        rows.append({
            "key": node.key, "label": node.label, "kind": node.kind,
            "role": node.role,
            "is_source": int(node.source), "is_sink": int(node.sink),
            "value": node.value,
            "n_ids": len(node.ids), "ids": ";".join(sorted(node.ids)),
            "blocks": ";".join(sorted(i for i in node.ids
                                      if "block" in i.lower()
                                      and "blocktype" not in i.lower())),
            "order": node.order,
            "provenance": " | ".join(node.provenance),
            "args": node.meta.get("args", ""),
            "meta": ";".join(f"{k}={v}" for k, v in sorted(meta.items())),
        })
    _write(path, rows, NODE_COLUMNS)
    return path


def _write(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
