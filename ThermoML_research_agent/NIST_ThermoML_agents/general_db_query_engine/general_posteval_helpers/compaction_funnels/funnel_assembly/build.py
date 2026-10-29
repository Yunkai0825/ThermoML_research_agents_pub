"""Build orchestrator — canonical skeleton → all views for one session.

Derivation order (user doctrine): detailed chars+ids are canonical;
blocks/pts are id projections of the SAME skeleton; agentic = node
merge of detailed by agentic_group; pooled = node merge of agentic by
pooled_group.  Per-agent sub_ figures are detailed-skeleton subgraphs;
session figures are the agentic level under a session title.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from funnel_assembly.graph_model import METRICS, Skeleton
from funnel_assembly.projections import ProjectedGraph, project
from funnel_assembly.render_export import (PngExport, render_figure,
                                           write_edge_tsv, write_node_tsv)
from funnel_assembly.skeleton_build import build_skeleton
from funnel_evidence.delegation_tree import build_tree

_LEVELS = ("detailed", "agentic", "pooled")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def build_all(session: Path, kind: str, label: str, out_dir: Path,
              render: bool = True) -> dict:
    """Generate the full output inventory; returns build artifacts for
    the audits: {'skeletons': {level: Skeleton}, 'graphs': {(metric,
    level): ProjectedGraph}, 'outputs': [Path], 'tree': tree}."""
    session = Path(session)
    out_dir = Path(out_dir)
    tree = build_tree(session, kind)
    detailed = build_skeleton(tree)
    skeletons = {
        "detailed": detailed,
        "agentic": detailed.merged("agentic_group"),
    }
    skeletons["pooled"] = skeletons["agentic"].merged("pooled_group")

    graphs: dict[tuple[str, str], ProjectedGraph] = {}
    outputs: list[Path] = []
    case = label or session.name
    # one PNG breaker and one skip list per workflow build
    png_export = PngExport() if render else None

    for level in _LEVELS:
        for metric in METRICS:
            graph = project(skeletons[level], metric)
            graphs[(metric, level)] = graph
            stem = f"{metric}_{level}_sankey"
            outputs.append(write_edge_tsv(graph,
                                          out_dir / f"{stem}_edges.tsv"))
            outputs.append(write_node_tsv(graph,
                                          out_dir / f"{stem}_nodes.tsv"))
            if render:
                title = (f"{case} \u2014 {metric} \u00b7 {level} \u2014 "
                         f"registry funnel")
                outputs.extend(render_figure(
                    graph, title, f"{stem}_{_slug(case)}", out_dir,
                    mode=level, png_export=png_export))

    # per-agent detailed subgraphs (session-owning children)
    for inst in tree.instances():
        if inst.parent is None or not inst.owns_session:
            continue
        sub = _subgraph(detailed, inst.sid)
        # browser pattern requires the (query|analysis) kind token
        kind_word = "query" if inst.spec.key.startswith("Q") else "analysis"
        for metric in ("chars", "ids"):
            graph = project(sub, metric)
            if not graph.nodes:
                continue
            stem = (f"sub_{_slug(inst.sid.split('/')[-1])}_{metric}"
                    f"_detailed_{kind_word}_sankey")
            outputs.append(write_edge_tsv(graph,
                                          out_dir / f"{stem}_edges.tsv"))
            outputs.append(write_node_tsv(graph,
                                          out_dir / f"{stem}_nodes.tsv"))
            if render:
                title = (f"{case} \u2014 {inst.sid} \u2014 {metric} \u00b7"
                         f" detailed subgraph")
                outputs.extend(render_figure(
                    graph, title, f"{stem}_{_slug(case)}", out_dir,
                    mode="detailed", png_export=png_export))

    # session views (agentic level under the session banner)
    if render:
        for metric in ("chars", "ids"):
            graph = graphs[(metric, "agentic")]
            title = f"{case} \u2014 whole session \u2014 {metric}"
            outputs.extend(render_figure(
                graph, title, f"session_{metric}_sankey_{_slug(case)}",
                out_dir, mode="agentic", png_export=png_export))
        listing = png_export.finish(out_dir)
        if listing is not None:
            outputs.append(listing)

    return {"skeletons": skeletons, "graphs": graphs, "outputs": outputs,
            "tree": tree}


def _subgraph(skel: Skeleton, sid: str) -> Skeleton:
    """Nodes whose key path starts with the instance sid (its whole
    delegation subtree), plus edges among them.  A node whose consumer
    lies outside the slice IS the slice's terminal hand-off: it
    becomes a sink and absorbs its residual accounting edges."""
    out = Skeleton()
    prefix = f"{sid}"
    for key, node in skel.nodes.items():
        base = key.split(":", 1)[0] if not key.startswith("grp:") else ""
        if base == prefix or base.startswith(prefix + "/"):
            out.nodes[key] = node
    boundary = {e.source for e in skel.edges
                if e.source in out.nodes and e.target not in out.nodes}
    out.edges = [e for e in skel.edges
                 if e.source in out.nodes and e.target in out.nodes
                 and e.source not in boundary]
    for key in boundary:
        out.nodes[key] = replace(out.nodes[key], sink=True)
    return out
