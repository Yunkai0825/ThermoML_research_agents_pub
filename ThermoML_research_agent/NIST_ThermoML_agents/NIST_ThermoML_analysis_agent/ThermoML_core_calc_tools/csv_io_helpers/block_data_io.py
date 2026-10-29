"""
block_data_io — Extract & parse ThermoML block data into typed arrays.
======================================================================
Self-contained: ALL CSV-related parsing lives here.
No LLM, no agentic logic.  Pure deterministic Python.

Reads from:
  - card_db_search_tools.basic_search_tools.11_block_data_extractor
    (returns {"csv_text": ..., "columns": ..., "metadata": ...})

Outputs:
  - BlockData dataclass  — numpy arrays, column names, metadata
  - ColumnMatch dataclass — identified x/y columns for fitting

Downstream consumers (ideal_baseline, rk_fitter, analysis_tools)
never import csv or io.StringIO — they receive numpy arrays directly.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ThermoML_raw_json_to_card_db_parsers.id_schema import (
    require_block_id,
    require_block_local_id,
)

log = logging.getLogger("CSV-IO")

# ── Ensure search tools importable ─────────────────────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.abspath(os.path.join(_THIS, "..", "..", "..", ".."))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

import importlib as _il
_extractor = _il.import_module(
    "card_db_search_tools.basic_search_tools.11_block_data_extractor"
)
_raw_extract_block_csv = _extractor.extract_block_csv
_raw_extract_multi_block_csv = _extractor.extract_multi_block_csv

from . import virtual_block_registry as _vbr  # noqa: E402  (agent-built fallback blocks)


# ═══════════════════════════════════════════════════════════════
#  Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class BlockData:
    """Parsed block data — numpy arrays ready for fitting."""
    doi: str
    block_number: str
    columns: List[str]               # ordered column names
    arrays: Dict[str, np.ndarray]    # column_name → 1-D float array
    n_rows: int
    metadata: Dict[str, Any]         # compounds, block_type, solvent, ...
    error: Optional[str] = None
    BLKsubsys_id: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.n_rows > 0


@dataclass
class ColumnMatch:
    """Identified composition (x) and property (y) columns.

    For binary systems, x_column is the single composition variable.
    For ternary+, x_columns lists ALL composition columns found.
    """
    x_column: str                    # primary composition column
    y_column: str                    # primary property column
    x_columns: List[str] = field(default_factory=list)   # ALL composition cols
    y_columns: List[str] = field(default_factory=list)   # ALL property cols
    other_columns: List[str] = field(default_factory=list)
    column_compound_map: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.x_column) and bool(self.y_column)

    @property
    def n_composition_vars(self) -> int:
        return len(self.x_columns)


# ═══════════════════════════════════════════════════════════════
#  CSV text → arrays (internal)
# ═══════════════════════════════════════════════════════════════

def parse_csv_text(csv_text: str) -> Tuple[List[str], List[Dict[str, Optional[str]]]]:
    """Parse a CSV string into (column_names, rows_as_dicts).

    Returns
    -------
    columns : list[str]
    rows : list[dict]     — each dict maps column_name → raw string value
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    columns = reader.fieldnames or []
    rows = list(reader)
    return list(columns), rows


def _rows_to_arrays(
    columns: List[str],
    rows: List[Dict[str, Optional[str]]],
) -> Dict[str, np.ndarray]:
    """Convert string-valued row dicts to float64 arrays.

    Non-numeric cells become NaN.
    """
    arrays: Dict[str, np.ndarray] = {}
    for col in columns:
        vals = []
        for r in rows:
            raw = r.get(col)
            if raw is None or raw == "":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(raw))
                except (ValueError, TypeError):
                    vals.append(np.nan)
        arrays[col] = np.array(vals, dtype=np.float64)
    return arrays


def arrays_to_csv(columns: List[str], data: Dict[str, np.ndarray]) -> str:
    """Convert column arrays back to a CSV string.

    Parameters
    ----------
    columns : list[str]
        Column names in order.
    data : dict
        column_name → 1-D array.

    Returns
    -------
    str
        CSV text with header row.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    n = len(next(iter(data.values()))) if data else 0
    for i in range(n):
        row = []
        for col in columns:
            v = data[col][i]
            if np.isnan(v):
                row.append("")
            else:
                row.append(str(round(v, 8)))
        writer.writerow(row)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════
#  Column identification
# ═══════════════════════════════════════════════════════════════

# Composition-related column patterns (mole_fraction, mass_fraction, etc.)
_COMPOSITION_HINTS = [
    "mole_fraction", "mass_fraction", "volume_fraction",
    "molality", "concentration",
]

# Common property patterns
_PROPERTY_HINTS = {
    "viscosity": ["viscosity", "eta"],
    "density": ["density", "mass_density", "molar_volume"],
    "speed_of_sound": ["speed_of_sound"],
    "refractive_index": ["refractive_index", "refraction"],
    "heat_capacity": ["heat_capacity", "cp_", "cv_"],
    "activity_coefficient": ["activity_coefficient", "gamma"],
    "surface_tension": ["surface_tension"],
    "thermal_conductivity": ["thermal_conductivity"],
}


def _normalize_hint_text(text: str) -> str:
    """Hints and columns compare in lowercase underscore form, so
    "surface tension" matches "surface_tension_liquidgas_n_m"."""
    return text.lower().replace(" ", "_")


# Companion columns: the extractor emits "<col>_U" (the expanded combined
# uncertainty of <col>) and "<col>_limit" beside the column they describe.
# The suffixes are read from the extractor so the two files cannot drift.
_COMPANION_SUFFIXES = tuple(
    getattr(_extractor, name, default)
    for name, default in (("UNCERTAINTY_SUFFIX", "_U"), ("LIMIT_SUFFIX", "_limit"))
)


def companion_columns(columns: List[str]) -> set:
    """The columns of *columns* that describe another column of the list.

    "<base>_U" / "<base>_limit" is a companion only when "<base>" itself is
    in *columns*; a real column that merely ends in "_U" is not swallowed.
    A companion is never a composition (x) or property (y) candidate: it
    carries no property declaration (inspect_block's contract check fails on
    it) and must not be fitted or aligned as a variable.  It stays in
    ColumnMatch.other_columns, in BlockData.arrays and in column_ranges, so
    the uncertainty is still shown.
    """
    present = set(columns)
    out = set()
    for col in columns:
        for suffix in _COMPANION_SUFFIXES:
            if suffix and col.endswith(suffix) and col[: -len(suffix)] in present:
                out.add(col)
                break
    return out


def identify_columns(
    columns: List[str],
    property_hint: str = "",
    composition_hint: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> ColumnMatch:
    """Identify composition (x) and property (y) columns from a column list.

    Uses metadata.compound_map (if available) to resolve DOIcomp_N references
    to actual compound names in column_compound_map.

    Parameters
    ----------
    columns : list[str]
        Column names from a block extraction.
    property_hint : str
        Substring hint for the property column (e.g. "density", "viscosity").
    composition_hint : str
        Substring hint for the composition column (e.g. "mole_fraction").
    metadata : dict, optional
        Block metadata dict (from extract_block_arrays).  If present,
        metadata["compound_map"] is used to resolve DOIcomp_N references
        in column names to actual compound names.

    Returns
    -------
    ColumnMatch
    """
    if not columns:
        return ColumnMatch(x_column="", y_column="", error="No columns available")

    compound_map = (metadata or {}).get("compound_map", {})

    # Never a candidate: the point ids and the companion columns ("<col>_U",
    # "<col>_limit") that describe another column of this block.  They are
    # listed under other_columns instead.
    skip = {"BLKpoint_id", "_source"} | companion_columns(columns)

    # --- Find ALL composition columns ---
    x_candidates = []
    comp_hint = _normalize_hint_text(composition_hint) if composition_hint else ""
    for col in columns:
        if col in skip:
            continue
        cl = _normalize_hint_text(col)
        if comp_hint and comp_hint in cl:
            x_candidates.append(col)
        elif any(h in cl for h in _COMPOSITION_HINTS):
            x_candidates.append(col)

    # --- Find ALL property columns ---
    y_candidates = []
    prop_hint = _normalize_hint_text(property_hint) if property_hint else ""
    for col in columns:
        if col in skip:
            continue
        cl = _normalize_hint_text(col)
        if prop_hint and prop_hint in cl:
            y_candidates.append(col)
        elif not any(h in cl for h in _COMPOSITION_HINTS + ["temperature", "pressure"]):
            y_candidates.append(col)

    # If property_hint matches a known category, look for aliases too
    if prop_hint:
        for category, aliases in _PROPERTY_HINTS.items():
            if any(a in prop_hint for a in aliases) or prop_hint in category:
                for col in columns:
                    if col in skip:
                        continue
                    cl = _normalize_hint_text(col)
                    if any(a in cl for a in aliases) and col not in y_candidates:
                        y_candidates.append(col)

    # --- Build column → compound name mapping ---
    # Column names now use <compound_name> format:
    #   "mole_fraction_<ethanol>" → compound = "ethanol"
    #   "activity_coefficient_<hexane>" → compound = "hexane"
    _ANGLE_RE = re.compile(r'<([^>]+)>')
    col_comp_map: Dict[str, str] = {}
    for col in x_candidates + y_candidates:
        m = _ANGLE_RE.search(col)
        if m:
            col_comp_map[col] = m.group(1)

    x_col = x_candidates[0] if x_candidates else ""
    y_col = y_candidates[0] if y_candidates else ""
    others = [
        column for column in columns
        if column not in {"BLKpoint_id", "_source"}
        and column not in x_candidates
        and column not in y_candidates
    ]

    if not x_col:
        return ColumnMatch(
            x_column="", y_column=y_col,
            x_columns=[], y_columns=y_candidates,
            other_columns=others,
            column_compound_map=col_comp_map,
            error=f"No composition column found in {columns}"
        )
    if not y_col:
        return ColumnMatch(
            x_column=x_col, y_column="",
            x_columns=x_candidates, y_columns=[],
            other_columns=others,
            column_compound_map=col_comp_map,
            error=f"No property column found in {columns}"
        )

    return ColumnMatch(
        x_column=x_col,
        y_column=y_col,
        x_columns=x_candidates,
        y_columns=y_candidates,
        other_columns=others,
        column_compound_map=col_comp_map,
    )


# ═══════════════════════════════════════════════════════════════
#  Mixture point filtering
# ═══════════════════════════════════════════════════════════════

def filter_mixture_points(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove pure-component endpoints (x≈0 or x≈1) and NaN rows.

    The default ``eps=0.02`` matches the default edge-detection threshold
    used by ``extract_pure_from_edges``.  This prevents near-pure points
    that were used for pure-value extraction from leaking into the
    mixture-only RK fit, which previously caused large residuals in the
    dilute regime.

    Parameters
    ----------
    x : ndarray — composition
    y : ndarray — property values
    eps : float
        Composition margin; points with x < eps or x > 1-eps are excluded.
        Should match the edge_threshold used for pure-value extraction.

    Returns
    -------
    (x_mix, y_mix) — arrays with only mixture data (eps < x < 1-eps), no NaN.
    """
    valid = (
        ~np.isnan(x) & ~np.isnan(y) &
        (x > eps) & (x < 1.0 - eps)
    )
    return x[valid], y[valid]


# ═══════════════════════════════════════════════════════════════
#  Main extraction functions (Public API)
# ═══════════════════════════════════════════════════════════════

def extract_block_arrays(
    doi: str,
    block_number: str,
    *,
    BLKsubsys_id: str | None = None,
    property_filter: str | None = None,
) -> BlockData:
    """Extract a ThermoML block and return typed numpy arrays.

    This is the ONLY function external code should call to get data.
    All CSV parsing is handled internally.

    Parameters
    ----------
    doi : str
        Paper DOI.
    block_number : str
        Typed block identifier such as ``PROPblock_2`` or ``RXNblock_1``.
    BLKsubsys_id : str, optional
        Exact block-local composition subsystem to extract. ``None`` selects
        the declared parent block.
    property_filter : str, optional
        Only include property columns matching this substring.

    Returns
    -------
    BlockData — with .arrays dict of numpy arrays, .columns list, .metadata dict.
    """
    try:
        typed_block_id = require_block_id(block_number)
    except ValueError as exc:
        return BlockData(
            doi=doi,
            block_number=str(block_number),
            columns=[], arrays={}, n_rows=0, metadata={},
            error=(
                "ID_REFINEMENT_REQUIRED: block_number must be "
                "PROPblock_<positive integer> or RXNblock_<positive integer>; "
                f"received {block_number!r} ({exc})"
            ),
        )

    subsystem_id = (
        require_block_local_id("subsys", BLKsubsys_id)
        if BLKsubsys_id is not None else None
    )
    if _vbr.is_virtual(doi):
        if subsystem_id is not None:
            raise ValueError("Virtual blocks do not define composition subsystems")
        # Agent-built block — served from the in-memory registry.
        # property_filter is not applied here (virtual blocks keep all their
        # columns; identify_columns hints handle selection downstream).
        raw = _vbr.lookup(doi, typed_block_id)
    else:
        raw = _raw_extract_block_csv(
            doi,
            typed_block_id,
            BLKsubsys_id=subsystem_id,
            property_filter=property_filter,
        )

    required_result_fields = {"csv_text", "columns", "n_rows", "metadata", "error"}
    missing_result_fields = required_result_fields - raw.keys()
    if missing_result_fields:
        raise ValueError(
            "extract_block_csv result missing required fields: "
            f"{sorted(missing_result_fields)}"
        )
    if raw["error"]:
        return BlockData(
            doi=doi,
            block_number=typed_block_id,
            columns=[],
            arrays={},
            n_rows=0,
            metadata=raw["metadata"],
            error=raw["error"],
            BLKsubsys_id=subsystem_id,
        )

    csv_text = raw["csv_text"]
    columns, rows = parse_csv_text(csv_text)
    if columns != raw["columns"]:
        raise ValueError(
            "extract_block_csv columns disagree with the CSV header: "
            f"declared={raw['columns']!r}, parsed={columns!r}"
        )
    if raw["n_rows"] != len(rows):
        raise ValueError(
            "extract_block_csv n_rows disagrees with the CSV payload: "
            f"declared={raw['n_rows']!r}, parsed={len(rows)}"
        )
    if not isinstance(raw["metadata"], dict):
        raise TypeError("extract_block_csv metadata must be an object")
    arrays = _rows_to_arrays(columns, rows)

    return BlockData(
        doi=doi,
        block_number=typed_block_id,
        columns=columns,
        arrays=arrays,
        n_rows=len(rows),
        metadata=raw["metadata"],
        BLKsubsys_id=subsystem_id,
    )


def extract_multi_block_arrays(
    blocks: List[Dict[str, str]],
    *,
    property_filter: str | None = None,
) -> List[BlockData]:
    """Extract multiple blocks and return a list of BlockData.

    Parameters
    ----------
    blocks : list[dict]
        Each dict must have ``doi``, ``block_number``, and nullable
        ``BLKsubsys_id``.

    Returns
    -------
    list[BlockData]
    """
    results = []
    for index, spec in enumerate(blocks):
        if not isinstance(spec, dict) or set(spec) != {
            "doi", "block_number", "BLKsubsys_id"
        }:
            raise ValueError(
                f"blocks[{index}] must contain exactly doi, block_number, "
                "and BLKsubsys_id"
            )
        doi = spec["doi"]
        if not isinstance(doi, str) or not doi:
            raise TypeError(f"blocks[{index}].doi must be a non-empty string")
        bn = require_block_id(spec["block_number"])
        bd = extract_block_arrays(
            doi,
            bn,
            BLKsubsys_id=spec["BLKsubsys_id"],
            property_filter=property_filter,
        )
        results.append(bd)
    return results


# ═══════════════════════════════════════════════════════════════
#  Condition-variable grouping
# ═══════════════════════════════════════════════════════════════
#  ThermoML blocks often contain data at multiple temperatures AND
#  pressures.  Downstream analysis (baseline, RK fit) requires data
#  at a SINGLE set of conditions — i.e. a unique (T, P, …) tuple.
#  The functions below detect which condition columns vary and
#  group/filter data accordingly.

# Ordered list of known condition columns (checked in this order)
CONDITION_COLUMNS = ["temperature_k", "pressure_kpa"]


def identify_varying_conditions(
    columns: List[str],
    arrays: Dict[str, np.ndarray],
    x_col: str,
    y_col: str,
) -> List[str]:
    """Return condition columns that have >1 unique value in *arrays*.

    Only considers columns that are NOT the composition (x) or property (y).
    """
    varying = []
    for col in CONDITION_COLUMNS:
        if col not in columns or col == x_col or col == y_col:
            continue
        vals = arrays[col]
        unique = np.unique(vals[~np.isnan(vals)])
        if len(unique) > 1:
            varying.append(col)
    return varying


def unique_condition_groups(
    arrays: Dict[str, np.ndarray],
    cond_cols: List[str],
) -> List[Tuple[float, ...]]:
    """Return sorted list of unique condition-value tuples.

    If *cond_cols* is empty, returns ``[(None,)]`` — a single group
    representing "all data".
    """
    if not cond_cols:
        return [(None,)]
    n = len(next(iter(arrays.values())))
    combos: set = set()
    for i in range(n):
        key = tuple(float(arrays[c][i]) for c in cond_cols)
        if not any(np.isnan(v) for v in key):
            combos.add(key)
    return sorted(combos)


def filter_to_condition(
    arrays: Dict[str, np.ndarray],
    cond_cols: List[str],
    cond_values: Tuple[float, ...],
    tol: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Filter *arrays* to rows matching a specific condition tuple.

    Returns a new dict of arrays containing only the matching rows.
    If ``cond_values == (None,)`` (no conditions), returns *arrays* unchanged.
    """
    if cond_values == (None,):
        return arrays
    n = len(next(iter(arrays.values())))
    mask = np.ones(n, dtype=bool)
    for col, target in zip(cond_cols, cond_values):
        mask &= np.abs(arrays[col] - target) < tol
    return {col: arr[mask] for col, arr in arrays.items()}


def condition_label(
    cond_cols: List[str],
    cond_values: Tuple[float, ...],
) -> str:
    """Human-readable label, e.g. 'T=298.15 K, P=101.325 kPa'."""
    if cond_values == (None,):
        return ""
    parts = []
    for col, val in zip(cond_cols, cond_values):
        if "temperature" in col:
            parts.append(f"T={val} K")
        elif "pressure" in col:
            parts.append(f"P={val} kPa")
        else:
            parts.append(f"{col}={val}")
    return ", ".join(parts)


def condition_file_tag(
    cond_cols: List[str],
    cond_values: Tuple[float, ...],
) -> str:
    """File-safe tag, e.g. '_298.15K_101.325kPa'."""
    if cond_values == (None,):
        return ""
    parts = []
    for col, val in zip(cond_cols, cond_values):
        if "temperature" in col:
            parts.append(f"{val}K")
        elif "pressure" in col:
            parts.append(f"{val}kPa")
        else:
            parts.append(f"{val}")
    return "_" + "_".join(parts)


def count_unique_x(
    arrays: Dict[str, np.ndarray],
    x_col: str,
) -> int:
    """Return number of unique non-NaN values in the x column."""
    x = arrays[x_col]
    return len(np.unique(x[~np.isnan(x)]))
