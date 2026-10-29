# Local installation

[Documentation home](../README.md) · [Data](DATA.md) · [Usage](USAGE.md)

## Prepare the databases

A complete Git clone or source-code ZIP includes the database payloads in
`_database_archives/`. No separate release download or Git LFS setup is required.
The browser launcher prepares the five large databases automatically before
starting the server. To prepare them explicitly, or before using Python APIs:

```shell
python database_archives.py
```

On first use, the helper reads
[`_database_archives/manifest.json`](../_database_archives/manifest.json), decodes
each database's ordered ZIP chunks, and writes a temporary file. It verifies the
original byte count and SHA-256 before installing the complete database
atomically at its normal path. This restores the supplied files; it does not
run database builders or regenerate scientific data.

| Reconstructed database | Local destination |
| --- | --- |
| `thermoml_raw.db` | `ThermoML_research_agent/ThermoML.v2020-09-30.db/thermoml_raw.db` |
| `PCS_INDIV.db` | `ThermoML_research_agent/card_databases_storage/Individual_cards_dbs/PCS_INDIV.db` |
| `PureOrMixtureData_registry.db` | `ThermoML_research_agent/card_databases_storage/PureOrMixtureData_registry.db` |
| `ThermoML_index.db` | `ThermoML_research_agent/card_databases_storage/ThermoML_index.db` |
| `thermoml_raw_corpus.db` | `ThermoML_research_agent/ThermoML.v2020-09-30.db/thermoml_raw_corpus.db` |

Verification receipts and an operating-system lock live in `.database_cache/`.
Subsequent launches check the manifest and file statistics and skip unchanged,
previously verified databases. Existing files are verified before adoption;
a differing database is preserved and preparation stops. Move that file aside
manually after reviewing it, then rerun `python database_archives.py` to restore
the packaged original.

Each database part is an ordinary ZIP below 100,000,000 bytes containing a byte
chunk. Files sit directly in `_database_archives/` with short names such as
`thermoml_raw_part1.zip` and `PCS_INDIV_part2.zip`: the database name followed by
an unpadded part number. Byte offsets and lengths are recorded only in the
manifest. The helper concatenates the **decoded payloads** in manifest order;
do not concatenate the ZIP containers or
treat an individual chunk as a complete SQLite database.
[RELEASE_ASSETS.json](RELEASE_ASSETS.json) retains the original database hashes
and historical release provenance and serves as the packaging builder's source
inventory. Its download links are not a current installation requirement.

Benchmark ZIPs use a different layout: each part holds complete examples and
remains compressed for direct browser access. Their exact members, sizes, and
SHA-256 hashes are in [`_benchmark/ARCHIVES.json`](../_benchmark/ARCHIVES.json).
See [BENCHMARKS.md](BENCHMARKS.md) for the layout and Eval UI.

The separate [evaluation-statistics ZIP](../_benchmark/_eval_stats/eval_stats.zip)
contains the statistics tools, saved charts/tables and reproducible evaluation
workflow. Both browser startup paths prepare it automatically on first use,
verifying the package and restoring its files under `_benchmark/_eval_stats/`.
Run `python eval_stats_archives.py` to prepare it without starting the browser.
Extracted files are local generated copies; publish the ZIP and its manifest.

## Environment and files

Start in `ThermoML_research_agents_pub`, the folder containing both
`ThermoML_research_agent/` and `ThermoML_results_browser/`. Python 3.13 is the
local validation environment. Other Python/platform combinations are not claimed
as tested. Preserve the archive layout and database locations listed in
[DATA.md](DATA.md); the launcher prepares missing database files automatically.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe launch_thermoml_browser.py
```

Bash with an appropriate Python interpreter:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python launch_thermoml_browser.py
```

The remaining examples use `python` for the interpreter with these packages.
The root `requirements.txt` is the single dependency file for the browser, agents,
and supporting tools.
The requirement bounds describe supported dependencies, not an exact environment
lockfile. Existing SQLite files are used directly; installation does not invoke
parser/database builders.

## Browser

```shell
python launch_thermoml_browser.py
```

The launcher opens the actual bound URL. Default port 5000 falls back to a free
port when unavailable. Use `--port 0` to request a free port, `--port 5001` to
require a particular port, or `--no-browser` to print the URL without opening it.
An explicit occupied port fails with an explanation. Ctrl+C stops the server.

An old browser may still be running on port 5000 from another workspace. If you
see a username badge, personal-history labels, or missing shared runs, use the
URL printed by this workspace's launcher.

## Provider configuration

Live agent calls use the server process's Argo configuration. Set its identity
before launching:

```powershell
$env:ARGO_API_USER = "your-argo-username"
python launch_thermoml_browser.py
```

On Bash, use `export ARGO_API_USER="your-argo-username"`. The identity is for
provider requests; it does not select a personal output directory or browser
account. All freeform outputs are shared.

The agent configs accept `ARGO_API_URL` for an alternate compatible endpoint.
The default service is an internal Argo endpoint and requires appropriate
network access and provider authorization. Model labels in the UI do not imply
that every account can access them. Set environment variables in the launching
shell; the application does not automatically load `.env` files.

The benchmark launcher also supports direct Anthropic mode with
`ANTHROPIC_API_KEY` and `--anthropic`; see [BENCHMARKS.md](BENCHMARKS.md).
That benchmark option does not change the browser's provider automatically.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing Python package | Install requirements with the same interpreter used to launch. |
| Missing database or archive part | Keep the complete `_database_archives/` directory and run `python database_archives.py`. If a supplied chunk is missing, restore it from the same repository snapshot. |
| Existing database differs | The original file is preserved. Review and move it aside manually, then rerun `python database_archives.py`. |
| Agent import error | Keep the sibling layout and use the root launcher or project-directory module commands. |
| Browser tab does not open | Open the printed URL manually. |
| Styling or interactive figures fail offline | The frontend uses CDN assets; local data does not make those assets available offline. |
| Provider request fails | Check server identity, endpoint, model availability, and network access. |
