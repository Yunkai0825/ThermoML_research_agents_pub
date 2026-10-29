"""Identity projections of a skeleton (chars / ids / blocks / pts)."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass

from framework_registry.stage_templates import NodeRole
from funnel_assembly.graph_model import Skeleton
from funnel_evidence.measures import (build_block_canon, qualified_block_pts)


@dataclass
class ProjEdge:
    source: str
    target: str
    value: int
    flow: str
    ids: frozenset
    provenance: tuple


@dataclass
class ProjNode:
    key: str
    label: str
    kind: str
    role: str
    value: int
    ids: frozenset
    order: int
    provenance: tuple
    meta: dict
    source: bool = False
    sink: bool = False


@dataclass
class ProjectedGraph:
    metric: str
    nodes: dict
    edges: list


def _edge_ids(measure, metric: str, canon: dict) -> frozenset:
    ids = measure.id_view(metric)
    if metric in ("blocks", "pts") and canon:
        ids = frozenset(canon.get(i, i) for i in ids)
    if metric == "pts":
        ids = frozenset(i for i in ids
                        if "::" in i and qualified_block_pts(i) > 0)
    return ids


def _edge_value(ids: frozenset, measure, metric: str) -> int:
    if metric == "chars":
        return measure.chars
    if metric == "pts":
        return sum(qualified_block_pts(i) for i in ids if "::" in i)
    return len(ids)


_SINK_FLOWS = ("removed", "deduplicated")


def _single_exit(edges: list, ctx_keys: frozenset = frozenset()) -> list:
    """Conservation in set-metric space (ids / canonical blocks): group
    merges union parallel in-edges (dropping duplicate arrivals) while
    out-edges to distinct targets keep theirs, and bare/qualified
    aliases of one block can sit on complementary lanes — either way a
    node would emit an id more often than it receives it.  Walk the
    DAG in topological order and cap each node's exits per id at its
    POST-cap arrival count (1 for roots); evidence lanes draw before
    sink lanes so surplus — not evidence — is what gets dropped.
    CONTEXT nodes forward each id at most ONCE in total (the dedup
    layer: downstream sees the union; duplicate copies exit via the
    dedup lanes)."""
    indeg: Counter = Counter()
    outs: dict[str, list] = defaultdict(list)
    nodes: set = set()
    for idx, e in enumerate(edges):
        nodes.add(e.source)
        nodes.add(e.target)
        indeg[e.target] += 1
        outs[e.source].append(idx)
    roots = {n for n in nodes if indeg[n] == 0}
    arrived: dict[str, Counter] = defaultdict(Counter)
    result = list(edges)
    queue = deque(sorted(roots))
    while queue:
        u = queue.popleft()
        cap = None if u in roots else arrived[u]
        is_ctx = u in ctx_keys
        used: Counter = Counter()
        used_fwd: Counter = Counter()
        lanes = sorted(outs[u],
                       key=lambda i: (result[i].flow in _SINK_FLOWS, i))
        for idx in lanes:
            e = result[idx]
            sinkish = e.flow in _SINK_FLOWS
            keep = set()
            for i in e.ids:
                allowed = 1 if cap is None else cap[i]
                if used[i] >= allowed:
                    continue
                if is_ctx and not sinkish and used_fwd[i] >= 1:
                    continue
                used[i] += 1
                if not sinkish:
                    used_fwd[i] += 1
                keep.add(i)
            if keep != set(e.ids):
                result[idx] = ProjEdge(
                    source=e.source, target=e.target, value=e.value,
                    flow=e.flow, ids=frozenset(keep),
                    provenance=e.provenance)
        for idx in outs[u]:
            e = result[idx]
            for i in e.ids:
                arrived[e.target][i] += 1
            indeg[e.target] -= 1
            if indeg[e.target] == 0:
                queue.append(e.target)
    return result


# aggregation points where multiple in-lanes can legally deliver the
# same id / alias pair — their surplus copies flush to an explicit sink
_FLUSH_ROLES = {"context", "answer", "deliverable", "batch",
                "envelope_part"}
_COLLAPSED_KEY = "proj:sink:collapsed"


def _flush_collapsed(edges: list, skel: Skeleton, metric: str) -> list:
    """Set-metric copy accounting: an edge carries an id ONCE however
    many copies converge on it (3rd+ duplicates, bare/qualified alias
    pairs riding one lane).  At aggregation nodes, route the surplus
    copies (arrivals minus exits, only for ids that DO exit at least
    once — an id with zero exits stays unbalanced and fails the
    audit) into an explicit collapsed-copies sink so in == out holds
    exactly at every internal node."""
    ins: dict[str, list] = defaultdict(list)
    outs: dict[str, list] = defaultdict(list)
    for e in edges:
        ins[e.target].append(e)
        outs[e.source].append(e)
    extra: list[ProjEdge] = []
    for key, node in skel.nodes.items():
        if node.role not in _FLUSH_ROLES or node.sink:
            continue
        arr: Counter = Counter()
        ex: Counter = Counter()
        for e in ins.get(key, ()):
            for i in e.ids:
                arr[i] += 1
        for e in outs.get(key, ()):
            for i in e.ids:
                ex[i] += 1
        surplus_ids: set = set()
        value = 0
        for i, a in arr.items():
            x = ex.get(i, 0)
            if x < 1 or a <= x:
                continue
            w = ((qualified_block_pts(i) if "::" in i else 0)
                 if metric == "pts" else 1)
            gain = w * (a - x)
            if gain > 0:
                value += gain
                surplus_ids.add(i)
        if value > 0:
            extra.append(ProjEdge(source=key, target=_COLLAPSED_KEY,
                                  value=value, flow="deduplicated",
                                  ids=frozenset(surplus_ids),
                                  provenance=()))
    return extra


def project(skel: Skeleton, metric: str) -> ProjectedGraph:
    canon: dict = {}
    if metric in ("blocks", "pts"):
        universe = set()
        for e in skel.edges:
            universe |= e.measure.ids
        canon = build_block_canon(
            universe, groups=[e.measure.ids for e in skel.edges])

    edges: list[ProjEdge] = []
    for e in skel.edges:
        ids = _edge_ids(e.measure, metric, canon)
        edges.append(ProjEdge(source=e.source, target=e.target, value=0,
                              flow=e.flow, ids=ids,
                              provenance=e.provenance))
    if metric in ("ids", "blocks", "pts"):
        ctx_keys = frozenset(k for k, n in skel.nodes.items()
                             if n.role == NodeRole.CONTEXT)
        edges = _single_exit(edges, ctx_keys)

    kept: list[ProjEdge] = []
    for pe, se in zip(edges, skel.edges):
        pe.value = _edge_value(pe.ids, se.measure, metric)
        if pe.value > 0:
            kept.append(pe)
    edges = kept
    if metric != "chars":
        edges.extend(_flush_collapsed(edges, skel, metric))

    inflow: dict[str, int] = {}
    outflow: dict[str, int] = {}
    node_ids: dict[str, set] = {}
    for e in edges:
        inflow[e.target] = inflow.get(e.target, 0) + e.value
        outflow[e.source] = outflow.get(e.source, 0) + e.value
        node_ids.setdefault(e.target, set()).update(e.ids)
        node_ids.setdefault(e.source, set()).update(e.ids)

    nodes: dict[str, ProjNode] = {}
    for key, node in skel.nodes.items():
        value = max(inflow.get(key, 0), outflow.get(key, 0))
        if value <= 0:
            continue
        nodes[key] = ProjNode(
            key=key, label=node.label, kind=node.kind, role=node.role,
            value=value, ids=frozenset(node_ids.get(key, ())),
            order=node.order, provenance=node.provenance, meta=node.meta,
            source=node.source, sink=node.sink)
    if _COLLAPSED_KEY in inflow:
        nodes[_COLLAPSED_KEY] = ProjNode(
            key=_COLLAPSED_KEY,
            label="duplicate copies collapsed (set-view dedup)",
            kind="removed", role="sink", value=inflow[_COLLAPSED_KEY],
            ids=frozenset(node_ids.get(_COLLAPSED_KEY, ())),
            order=10**9, provenance=(), meta={}, source=False, sink=True)
    edges = [e for e in edges if e.source in nodes and e.target in nodes]
    return ProjectedGraph(metric=metric, nodes=nodes, edges=edges)
