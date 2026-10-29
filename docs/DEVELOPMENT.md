# Development and validation

[Documentation home](../README.md)

Install the shared requirements from the project root:

```shell
python -m pip install -r requirements.txt
```

The source is a Python workspace rather than a native-code build. Import checks,
syntax checks, and browser smoke checks are relevant local checks. Database
builders and live agent prompt runners are separate operations. Database ZIP
packaging uses `scripts/package_databases.py` and the original hashes in
`docs/RELEASE_ASSETS.json`; it is separate from normal startup preparation.

Check browser imports and registered routes from the project root:

```shell
python -B -c "from ThermoML_results_browser.app import app; print(app.url_map)"
```

Application startup prepares missing large databases from the included ZIP
chunks and uses receipts for unchanged verified files. To run that preparation
explicitly, use `python database_archives.py`. It restores original bytes and
does not run database builders. Receipts and the startup lock are local files
under `.database_cache/`.

The import check does not launch agents or start a server. Use
`python launch_thermoml_browser.py` to inspect local pages and saved ZIP-backed
sessions. In benchmark history, check the table's scrollbars and sticky headers,
then select a saved answer with evaluation records and exercise its Eval switch,
highlights, and sidebar. An answer without those records should keep Eval disabled.

Check Python syntax without creating bytecode caches:

```shell
python -B -c "import ast; from pathlib import Path; files = [p for p in Path('.').rglob('*.py') if '__tmp__' not in p.parts]; [ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p)) for p in files]; print(f'{len(files)} files parsed')"
```

Run this command from the repository root. Syntax and import checks do not
validate scientific accuracy or provider access.

Paths must be anchored to the component's source location. Keep freeform outputs
under `_output`, benchmark artifacts under `_benchmark`, and explicit child
session overrides working. Use the strict identifiers defined in
[ID_architecture.md](../ThermoML_research_agent/card_databases_storage/ID_architecture.md).

Agent workflow Markdown is consumed at runtime. Changes to that Markdown can
change model behavior; distinguish it from explanatory README changes.

See [Publishing original files](PUBLISHING.md) for the committed-object checks
and pre-push hook activation. Those checks inspect local Git objects and do not
publish the workspace.
