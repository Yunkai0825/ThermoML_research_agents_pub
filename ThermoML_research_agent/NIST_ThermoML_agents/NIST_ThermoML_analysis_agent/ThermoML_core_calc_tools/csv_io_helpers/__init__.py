"""
csv_io_helpers — Self-contained CSV parsing and data marshalling.
=================================================================
ALL CSV reading, writing, column matching, and array extraction lives
here.  Nothing in this package uses an LLM — every function is pure
deterministic Python.

**No other module should import csv, io.StringIO, or csv.DictReader
for ThermoML data handling.  This is the single source of truth.**

Public API
----------
- extract_block_arrays(doi, block_number, ...) → BlockData
- extract_multi_block_arrays(blocks, ...) → list[BlockData]
- parse_csv_text(csv_text) → list[dict]
- arrays_to_csv(columns, data_dict) → str
- identify_columns(columns, property_hint, composition_hint) → ColumnMatch
- companion_columns(columns) → set  ("<col>_U" / "<col>_limit" beside <col>)
- filter_mixture_points(x, y, eps=0.02) → (x_mix, y_mix)
"""

from .block_data_io import (
    BlockData,
    ColumnMatch,
    extract_block_arrays,
    extract_multi_block_arrays,
    parse_csv_text,
    arrays_to_csv,
    identify_columns,
    companion_columns,
    filter_mixture_points,
)
from . import virtual_block_registry

__all__ = [
    "BlockData",
    "ColumnMatch",
    "extract_block_arrays",
    "extract_multi_block_arrays",
    "parse_csv_text",
    "arrays_to_csv",
    "identify_columns",
    "companion_columns",
    "filter_mixture_points",
    "virtual_block_registry",
]
