# ThermoML Agents

This local workspace combines the Main, Query, and Analysis agents with the
ThermoML results browser and the supplied NIST ThermoML databases. It uses the
`v2020-09-30` archive snapshot. Source records, generated cards, and saved run
artifacts provide the context for inspecting an answer.

## Start locally

A complete Git clone or source-code ZIP includes all required database payloads;
no separate release download or Git LFS setup is needed. The browser launcher
reconstructs the five large databases from [`_database_archives/`](_database_archives/manifest.json)
on first use, verifies their original bytes, and then starts the server.
Later launches use a local verification receipt and file statistics to avoid
reassembling or hashing unchanged databases. See [installation](docs/INSTALLATION.md#prepare-the-databases).

Benchmark run ZIPs stay compressed: the browser reads their results and saved
evaluations directly. The evaluation-statistics and workflow package in
`_benchmark/_eval_stats/eval_stats.zip` is extracted automatically on first
launch. See [benchmarks](docs/BENCHMARKS.md) for the Eval controls and statistics.

Run from this folder (`ThermoML_research_agents_pub`):

```shell
python -m pip install -r requirements.txt
python launch_thermoml_browser.py
```

Use the URL printed by the launcher. It binds to localhost, tries port 5000,
and selects a free port if that port is occupied. `--port 0` always selects a
free port; `--no-browser` suppresses opening a tab. Keep the process running
and press Ctrl+C to stop it.

Python 3.13 is the local validation environment. The existing databases are the
starting point; routine setup does not regenerate them. Browsing data and saved
results needs no model-provider credentials. Live agent requests need the
server's provider configuration described in [installation](docs/INSTALLATION.md).

## Layout

```text
ThermoML_research_agents_pub/
├── launch_thermoml_browser.py
├── database_archives.py            # Prepare the original database files
├── _database_archives/             # Database chunk ZIPs and reconstruction manifest
├── ThermoML_results_browser/       # Browser application and analysis tools
├── ThermoML_research_agent/         # Agent framework, search tools and databases
├── _benchmark/                     # Main, Query and Analysis benchmark artifacts
├── _output/                        # Shared freeform sessions and diagnostics
└── docs/                           # Setup, use, data and validation
```

Freeform sessions use `_output/{Main,Query,Analysis}` for everyone. Historical
Main sessions in `_output/freeform_runs.zip` remain browsable without extraction.
Saved benchmark archives use full-capability and ablation folders under
`_benchmark/{Main,Query,Analysis}`. The unchanged runners write their existing
run directories beneath those agent roots. Components declare these paths relative
to their source files; there is no output-routing module. Explicit API session
paths and benchmark output arguments remain available.

The browser provides shared access to saved runs and their detailed artifacts.
It has no browser-account or debug-access gates. Read-only database access,
scientific validation, and artifact path checks remain part of the application.

## Documentation

- [Installation and provider configuration](docs/INSTALLATION.md)
- [Browser and Python API usage](docs/USAGE.md)
- [Data inventory and provenance](docs/DATA.md)
- [Database archive manifest](_database_archives/manifest.json)
- [Original database hashes and legacy release provenance](docs/RELEASE_ASSETS.json)
- [Benchmark commands and recorded outputs](docs/BENCHMARKS.md)
- [Evaluation statistics and reproducible workflow (ZIP)](_benchmark/_eval_stats/eval_stats.zip)
- [Development and local validation](docs/DEVELOPMENT.md)
- [Publication policy and no-pointer checks](docs/PUBLISHING.md)
- [Workspace architecture](MASTER_ARCHITECTURE.md)
- [Browser reference](ThermoML_results_browser/README.md)
- [Agent framework and call hierarchy](ThermoML_research_agent/README.md)

## License and citation

The software uses the [MIT license](LICENSE). Keep the [third-party data notice](NOTICE)
with the NIST-derived data. [CITATION.cff](CITATION.cff) retains the manuscript,
authors, and data citations from the published project. The `v1.0.0(R)` tag identifies this revised software publication.
Software archive DOIs are separate from the manuscript DOI.
