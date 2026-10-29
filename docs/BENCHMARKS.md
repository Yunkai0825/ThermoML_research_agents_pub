# Benchmarks and saved outputs

[Documentation home](../README.md) · [Development](DEVELOPMENT.md)

The saved benchmark collections are distributed in Git as ordinary ZIP files
under project-root `_benchmark/`. They contain existing local run artifacts;
repackaging them does not produce new evaluation results. Freeform sessions
remain under `_output/`.

## Archive layout

Each agent separates full-capability and ablation collections:

```text
_benchmark/
├── ARCHIVES.json
├── Main/
│   ├── test_run_full_capability/
│   │   └── <collection>[__partNN__<first>--<last>].zip
│   └── test_run_ablation/
│       └── <collection>[__partNN__<first>--<last>].zip
├── Query/
│   ├── test_run_full_capability/
│   │   └── <collection>[__partNN__<first>--<last>].zip
│   └── test_run_ablation/
│       └── <collection>[__partNN__<first>--<last>].zip
└── Analysis/
    ├── test_run_full_capability/
    │   └── <collection>[__partNN__<first>--<last>].zip
    └── test_run_ablation/
        └── <collection>[__partNN__<first>--<last>].zip
```

A collection that fits in one file keeps `<collection>.zip`. Larger collections
use names such as
`<collection>__part01__Q<id>_run_<timestamp>--Q<id>_run_<timestamp>.zip`.
The two boundary names identify the first and last examples in that part;
slashes in nested example paths become underscores in the filename. Exact
membership is recorded in the manifest rather than inferred from a range.

Every archive is smaller than **100,000,000 bytes** and is independently
readable by an ordinary ZIP reader. These are separate ZIP archives, so no
concatenation or split-volume utility is required. Each example is kept whole
within one part, with its saved file contents and source-relative member paths
preserved. The source collection directory itself is omitted from ZIP member paths; nested campaign and example directories remain intact. Collection-level shared metadata is repeated in each part.

[`_benchmark/ARCHIVES.json`](../_benchmark/ARCHIVES.json) is the schema-version-1
archive manifest. Its `max_archive_bytes` is `100000000`. Each `archives` row
records the archive `path` relative to `_benchmark`, `source_collection`, `part`,
`parts`, `first_example`, `last_example`, exact `examples` and `shared_members`
lists, `file_count`, `bytes`, and the archive's `sha256`. Use these records to
locate an example and verify a ZIP's contents, size, and hash.

All benchmark ZIP parts are supplied in Git; no release download or Git LFS
pointer is required. Database ZIP chunks are also included, with their automatic
reassembly described in [installation](INSTALLATION.md#prepare-the-databases).
Unlike database chunks, benchmark parts hold whole examples and need no reassembly.
The browser discovers collections and ZIP parts inside these mode folders and
reads answer artifacts directly from ZIP members, including `result.md`,
`eval/evaluation.json`, and `eval/claim_review.json`. Keep the archives compressed;
no extraction is needed for results or evaluation views.

The benchmark question table has its own vertical and horizontal scroll area
with sticky column headers. Select an execution to open its Result panel.
**Eval** starts on when saved numerical validation or claim review is available
and is disabled when neither is available. Switch it off to read the plain
answer. Claim categories and numerical outcomes use separate highlights;
select a highlight to inspect its review, compared values, source evidence,
and related DOI/block links in the sidebar. Displaying a saved evaluation does
not rerun it or make a model request.

The supplied benchmark outputs use the corrected prompt numbering in the
current catalogs. Saved run timestamps and recorded tool results retain their
provenance. Use the archive manifest to locate a specific execution.

## Repacking saved outputs

The [packaging script](../scripts/package_benchmarks.py) accepts `--source`
pointing to the local ThermoML benchmark root and `--destination` pointing to a
new staging directory. It refuses an existing destination, leaves source files
unchanged, and verifies every archived file against the source snapshot with
SHA-256. A collection is split only between complete example directories.
If one example cannot fit within the configured limit, the build stops rather
than dividing that example. Install the completed staging contents under
`_benchmark/` after reviewing its manifest.

## Commands

From the project root:

```shell
python ThermoML_research_agent/run_debug_benchmark.py --offline
```

This is the local import/configuration preflight and run plan. It must not be
used as evidence that provider credentials or scientific responses work.
To deliberately launch the configured live suites:

```shell
python ThermoML_research_agent/run_debug_benchmark.py --go
```

The batch launcher has a selected Main cohort and the Query/Analysis selections
declared in its source. Use its printed plan and the current prompt files when
reporting selected cases. Do not infer the cohort from an old run directory's
file count. Use `--resume TIMESTAMP` only with a matching existing campaign and
compatible prompts/settings.

The synchronized prompt catalogs contain **24 Main, 31 Query, and 21 Analysis**
prompts. The batch launcher selects **17 Main** cases and all Query and
Analysis cases. Main prompts `1.1.5`, `3.7`, and `4.5` are available in the
catalog but are absent from its selected Main cohort. The launcher uses the
corrected IDs while preserving the selected question texts.

The batch launcher still writes `_benchmark/<Agent>/test_run_<timestamp>`;
changing its output routing to the archive mode folders remains deferred.
New runs retain their own settings and source evidence.

The per-agent runners can also be invoked from `ThermoML_research_agent/`:

```shell
python -m NIST_ThermoML_agents._NIST_ThermoML_main_agent._DEBUG_script.run_tests
python -m NIST_ThermoML_agents.NIST_ThermoML_query_agent._DEBUG_script.run_tests
python -m NIST_ThermoML_agents.NIST_ThermoML_analysis_agent._DEBUG_script.run_tests
```

These commands call the model provider despite the `_DEBUG_script` name. They
are distinct from the mocked/deterministic tests in DEVELOPMENT.md. Consult each
runner's `--help` for prompt selection and output overrides.

For direct Anthropic benchmark requests, configure `ANTHROPIC_API_KEY` in the
launching environment and add `--anthropic` to the batch command. Without `--go`,
the launcher reports its plan. Provider/model availability is account dependent.

## Statistics from saved evaluations

The statistics script, accepted-cohort selection, charts, tables and complete
evaluation workflow are published in
[`eval_stats.zip`](../_benchmark/_eval_stats/eval_stats.zip), inside
`_benchmark/_eval_stats/`. The archive is below 100,000,000 bytes. Its
[manifest](../_benchmark/_eval_stats/manifest.json) records sizes and hashes.
The browser extracts this package into the same folder on first launch.
To prepare it without starting the app and then regenerate statistics:

```shell
python eval_stats_archives.py
python _benchmark/_eval_stats/eval_stats.py
```

The script defaults to this checkout's `_benchmark/` root and reads saved
`eval/evaluation.json` records directly from ZIP parts, without extraction or
model requests. The extracted `_benchmark/_eval_stats/selection.json` identifies the
**157 accepted answers to 76 prompts** in the full-capability and no-ledger
cohort. Answer and evaluation hashes are checked before counting; missing or
changed selected files stop the run. Other saved ablation collections remain
browsable and are outside this default statistical cohort.

For the six selected agent/mode groups the script writes **14 files** under
`_benchmark/_eval_stats/numeric_outcomes/`: one CSV and PNG for each group, plus
`summary.csv` and `summary.png`. Pipeline artifacts have their own row and
contribute once to the combined property total; all-types totals count each
native category once. Saved comparisons include the default four-significant-
digit precision rule for continuous fitting outputs. The statistics script
reads those judgments rather than re-evaluating numbers.

Use `--root` or `--out` to select another benchmark root or output directory,
`--selection FILE` for another declared cohort, `--all-answers` to scan all saved
answers, or `--archive-only` to omit pipeline-artifact comparisons. A root without
a selection manifest scans all saved answers. Use a separate output directory
when comparing a different cohort.

After extraction, start with
`_benchmark/_eval_stats/eval_workflow/reproduce-evidence-evaluation/SKILL.md`.
This self-contained workflow includes the instructions, system-prompt
templates, ThermoML adapter, numerical
rubric, human interpretations, tools and regression tests needed for a new
evaluation. Prepared evidence accompanies each relevant saved answer.
Private review packets, operational caches and raw evaluation working data are
outside the publication package.

## Interpreting a campaign

Keep the exact prompt IDs, prompt text, code/data snapshot, model settings, and
artifacts with any reported cohort. Resume logic may skip files already written;
a partial or failed response can require inspection before retrying a campaign.
An operational success status is not a chemistry-accuracy score. Live comparison
runs are separate from the local rebuild validation.
