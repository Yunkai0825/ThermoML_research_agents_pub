"""Measures — the four flow identities carried by every funnel node.

A ``Measure`` bundles chars + ThermoML id set; blocks and pts are
derived projections of the id set (block tokens / DB-hydrated datapoint
weights).  DB metadata lookups (datapoint counts, DOI resolution) are
ALLOWED here — they hydrate weights, they never re-run session tools.
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[6]
_AGENT_REPO = _REPO / "ThermoML_research_agent"
for _p in (_AGENT_REPO, _AGENT_REPO / "card_db_search_tools",
           _AGENT_REPO / "card_db_search_tools" / "basic_search_tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_GLOBAL_ID = r"GLOB(?:lit|comp|prop|var|constr|meas|phase|blocktype|rxntype|solvent)_[1-9]\d*"
_BLOCK_ID = r"(?:PROP|RXN)block_[1-9]\d*"
_ID_RE = re.compile(
    rf"(?<!\w)(?:GLOBlit_[1-9]\d*::{_BLOCK_ID}|{_GLOBAL_ID}|"
    rf"DOIcompSample_[1-9]\d*_[1-9]\d*|DOIcomp_[1-9]\d*|"
    rf"BLKpropAssessment_[1-9]\d*_[1-9]\d*|"
    rf"BLK(?:prop|var|constr|point|subsys)_[1-9]\d*|"
    rf"{_BLOCK_ID})(?!\w)"
)
_DOI_RE = re.compile(r"(?<!\w)10\.\d{4,9}/[^\s|,;\]\[(){}<>\"']+", re.I)


def thermoml_ids(text: str) -> set[str]:
    """Extract canonical ThermoML identifiers and DOI references;
    session-structural tokens (WM record labels, section sids) are not
    knowledge ids and are dropped."""
    text = text or ""
    return set(_ID_RE.findall(text)) | {
        doi.rstrip(".\\") for doi in _DOI_RE.findall(text)
    }


def block_tokens(ids) -> frozenset:
    """Block-identity tokens: qualified GLOBlit::block plus bare blocks."""
    out = set()
    for i in ids:
        low = i.lower()
        if "blocktype" in low:
            continue
        if "::" in i or "propblock" in low or "rxnblock" in low:
            out.add(i)
    return frozenset(out)


@dataclass(frozen=True)
class Measure:
    chars: int = 0
    ids: frozenset = frozenset()

    @property
    def blocks(self) -> frozenset:
        return block_tokens(self.ids)

    @property
    def pts(self) -> int:
        return sum(qualified_block_pts(b) for b in self.blocks if "::" in b)

    def value(self, metric: str) -> int:
        if metric == "chars":
            return self.chars
        if metric == "ids":
            return len(self.ids)
        if metric == "blocks":
            return len(self.blocks)
        if metric == "pts":
            return self.pts
        raise ValueError(metric)

    def id_view(self, metric: str) -> frozenset:
        return self.blocks if metric in ("blocks", "pts") else self.ids


def measure(text: str = "", chars: int | None = None,
            ids=None) -> Measure:
    return Measure(chars=len(text) if chars is None else chars,
                   ids=frozenset(ids) if ids is not None
                   else frozenset(thermoml_ids(text)))


# ── block datapoint weights (DB metadata hydration, cached) ──

_PTS_CACHE: dict[tuple[str, str], int] = {}
_DOI_CACHE: dict[str, str] = {}


def _extract_rows(doi: str, block: str) -> int:
    mod = importlib.import_module("basic_search_tools.11_block_data_extractor")
    r = mod.extract_block_csv(doi=doi, block_number=block)
    return int(r.get("n_rows") or 0)


def qualified_block_pts(qual: str) -> int:
    """Datapoints for one 'GLOBlit_X::PROPblock_Y' qualified block id."""
    if "::" not in qual:
        return 0
    lit, block = qual.split("::", 1)
    key = (lit, block)
    if key in _PTS_CACHE:
        return _PTS_CACHE[key]
    if lit not in _DOI_CACHE:
        doi = ""
        try:
            mod = importlib.import_module("search_reference_from_block")
            ref = mod.search_reference_from_block(block_number=block,
                                                  lit_num_id=lit)
            doi = ref.get("doi") or ref.get("reference", {}).get("doi") or ""
        except Exception:
            pass
        if not doi:
            try:
                mod = importlib.import_module(
                    "basic_search_tools.4_reference_search")
                res = mod.search_references(literature=[lit], limit=1)
                rows = res.get("results") or []
                doi = rows[0].get("doi", "") if rows else ""
            except Exception:
                pass
        _DOI_CACHE[lit] = doi
    doi = _DOI_CACHE[lit]
    try:
        _PTS_CACHE[key] = _extract_rows(doi, block) if doi else 0
    except Exception as exc:
        print(f"warning: datapoint count {lit} {block} failed: {exc}")
        _PTS_CACHE[key] = 0
    return _PTS_CACHE[key]


# ── canonical spelling map (bare block ↔ qualified twin) ──────

def build_block_canon(evidence_ids, groups=()) -> dict[str, str]:
    """bare block token → its unique qualified twin seen in evidence.

    Globally ambiguous bare tokens resolve by CO-OCCURRENCE: the twin
    consistent with EVERY evidence lane that carries the bare token
    alongside qualified twins (intersection).  A unique common twin is
    the block the citing agent meant; anything else stays bare."""
    twins: dict[str, set[str]] = {}
    for i in evidence_ids:
        if "::" in i:
            bare = i.split("::", 1)[1]
            twins.setdefault(bare, set()).add(i)
    canon = {bare: next(iter(quals))
             for bare, quals in twins.items() if len(quals) == 1}
    bare_present = set(evidence_ids)
    for bare, quals in twins.items():
        if bare in canon or bare not in bare_present:
            continue
        riders: set[str] | None = None
        for group in groups:
            if bare in group:
                local = quals & group
                if local:
                    riders = local if riders is None else (riders & local)
        if riders is not None and len(riders) == 1:
            canon[bare] = next(iter(riders))
    return canon
