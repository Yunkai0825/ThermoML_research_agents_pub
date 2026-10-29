# Local ThermoML data

The assembled workspace uses the original databases under `ThermoML_research_agent/`. Five large databases are distributed as ordered ZIP byte chunks in [`_database_archives/`](../_database_archives/manifest.json); the other seven database files are stored directly in Git. [Automatic startup preparation](INSTALLATION.md#prepare-the-databases) restores and verifies the large files without regenerating data. The inventory below was read from the supplied SQLite files on 2026-09-18; no databases were rebuilt.

## Included files

After preparation, the 12 nonempty database files occupy **4,875,669,504 bytes** (about 4.54 GiB). They contain original SQLite data. The five reconstructed files total **4,687,310,848 bytes**. Their chunk order, destinations, sizes, and hashes are recorded in [`_database_archives/manifest.json`](../_database_archives/manifest.json). [RELEASE_ASSETS.json](RELEASE_ASSETS.json) retains original hashes and legacy release provenance for the packaging builder, not a separate download requirement.

| File, relative to `ThermoML_research_agent/` | Bytes | Observed contents |
| --- | ---: | --- |
| `ThermoML.v2020-09-30.db/thermoml_raw.db` | 2,014,683,136 | 11,923 papers; raw JSON and derived compound/block indexes |
| `ThermoML.v2020-09-30.db/thermoml_raw_corpus.db` | 388,104,192 | 11,923 paired compressed JSON/XML documents |
| `card_databases_storage/Individual_cards_dbs/CCS_ID_DK.db` | 35,622,912 | 8,526 compound identity cards |
| `card_databases_storage/Individual_cards_dbs/CCS_INDIV.db` | 65,069,056 | 11,923 paper-specific compound cards |
| `card_databases_storage/Individual_cards_dbs/MTDKS_ID_DK.db` | 6,156,288 | 2,302 measurement identity/knowledge cards |
| `card_databases_storage/Individual_cards_dbs/MTDKS_INDIV.db` | 19,341,312 | 11,923 paper-specific measurement cards |
| `card_databases_storage/Individual_cards_dbs/PCS_ID_DK.db` | 577,536 | 110 property identity/knowledge cards |
| `card_databases_storage/Individual_cards_dbs/PCS_INDIV.db` | 1,381,466,112 | 11,923 property cards containing 123,727 blocks |
| `card_databases_storage/Individual_cards_dbs/RMS_INDIV.db` | 56,520,704 | 11,923 reference cards |
| `card_databases_storage/PureOrMixtureData_registry.db` | 491,024,384 | 122,481 declared blocks and 57,741 composition subsystems |
| `card_databases_storage/ReactionData_registry.db` | 5,070,848 | 1,246 reaction blocks |
| `card_databases_storage/ThermoML_index.db` | 412,033,024 | 123,727 indexed blocks and joined identity registries |

The raw database has `papers`, `compounds`, `blocks`, `block_properties`, and `block_variables` tables. The corpus archive has `documents` and `metadata`; its format is `thermoml-zlib-sqlite-v1`. Each document stores independent zlib-compressed JSON and XML. Loose DOI-named JSON/XML directories are absent from this checkout. The results browser can read the compressed corpus directly.

## Packaged large databases

The five large databases are supplied as **53 ZIP files totaling 702,470,714
bytes**. The largest ZIP is **50,719,928 bytes**. Packaging verified every
original database against its recorded SHA-256; reconstruction restores the
original **4,687,310,848 bytes**.

| Database | ZIP parts |
| --- | ---: |
| `thermoml_raw.db` | 22 |
| `PCS_INDIV.db` | 15 |
| `PureOrMixtureData_registry.db` | 6 |
| `ThermoML_index.db` | 5 |
| `thermoml_raw_corpus.db` | 5 |

## Provenance and derived data

The supplied [raw database guide](../ThermoML_research_agent/ThermoML.v2020-09-30.db/thermoml_raw_db_README.md) and [database architecture](../ThermoML_research_agent/ThermoML.v2020-09-30.db/DATABASE_ARCHITECTURE.md) identify the source snapshot as the NIST ThermoML archive, version `2020-09-30`. `papers.json_data` holds the archived JSON. Its `md5_checksum` column preserves the source's `THERMOML_MD5_CHECKSUM` value; this inventory did not independently verify every document against the upstream archive.

[Parser modules](../ThermoML_research_agent/ThermoML_raw_json_to_card_db_parsers/) derive reference, compound, measurement, and property cards from raw JSON. [Canonical CSV catalogs](../ThermoML_research_agent/card_databases_storage/Canonicalized_ID_name_lists_csvs/) provide the identities used by cards and searches. The current derived databases use the `prefixed-v2` identifier schema. Property, variable, and constraint identities are linked by a hash-pinned translation CSV whose SHA-256 is recorded in the property database metadata.

The pipeline is:

```text
raw paper JSON + canonical identity catalogs + authored knowledge cards
    -> RMS / CCS / MTDKS / PCS card databases
    -> pure/mixture and reaction registries
    -> joined ThermoML search index
    -> search tools, compactors, and property-screening pipeline
```

The current property database records a display cap of 50 data points per block. Complete observations remain in the raw database; numerical extraction and screening can reconstruct uncapped blocks. The cap does not mean the source measurements have been discarded.

Some existing `source_db` metadata values contain the former `N:\_Code_Maintenance` build location. These values describe the original build; current runtime readers resolve the supplied files relative to the relocated project.

## Rebuild boundaries

Rebuilding data is a separate maintenance operation and is not part of this local migration. Included builders can replace card databases, registry databases, indexes, and canonical CSVs. Do not execute them merely to start the application.

The supplied sources include 129 authored measurement knowledge JSON files and 105 authored property knowledge JSON files, alongside the existing raw database and canonical catalogs. Card, registry, and index builder entrypoints are present under [the parser package](../ThermoML_research_agent/ThermoML_raw_json_to_card_db_parsers/). Their availability does not establish that a complete rebuild has been validated in this relocated checkout.

[The corpus archive builder](../ThermoML_research_agent/build_raw_corpus_archive.py) reads JSON from `thermoml_raw.db` but requires original loose XML files beneath `ThermoML.v2020-09-30.db/`. Those XML files are not present as loose files here, so its default inputs are incomplete even though the existing compressed archive is usable. Historical raw-database documentation names `build_thermoml_sqlite.py`; that script is not included in this checkout. Follow those older recovery instructions only after checking that the required source files and tools are available.

## Runtime output locations

Database input files remain under `ThermoML_research_agent/`. Benchmark runs belong under the workspace's [`_benchmark/`](../_benchmark/); freeform runs, caches, and diagnostics belong under [`_output/`](../_output/). Parser and compactor diagnostics use `_output/Query/Diagnostics/`. Standalone property-screening runs use `_output/Query/property_screening_runs/`, with shared deterministic caches under `_output/Query/_cache/property_screening/`; an active agent session receives its screening artifacts in that session's data directory.

Existing benchmark and freeform ZIPs contain historical run artifacts and are stored in Git. Benchmark collections are grouped by agent and full-capability/ablation mode, with whole examples packed into independently readable ZIP parts smaller than 100,000,000 bytes. Exact archive membership, sizes, and SHA-256 hashes are recorded in [`_benchmark/ARCHIVES.json`](../_benchmark/ARCHIVES.json); see [BENCHMARKS.md](BENCHMARKS.md) for the layout. The browser discovers these nested ZIP parts and reads answer Markdown and saved evaluation JSON without extraction. All required database payloads are also in Git and are [prepared automatically](INSTALLATION.md#prepare-the-databases). The saved artifacts do not establish that a fresh model-backed run or a database rebuild has been performed during setup.
