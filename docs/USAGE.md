# Using the workspace

[Documentation home](../README.md) · [Installation](INSTALLATION.md)

## Browser

From a complete clone or source-code ZIP, launch `python launch_thermoml_browser.py`
from the project root. The launcher [prepares the databases](INSTALLATION.md#prepare-the-databases)
from the included ZIP chunks before opening the browser. No separate release
download is needed. Search the
archive, inspect source papers and data blocks, or open History to inspect
saved results. Select Main, Query, or Analysis for the corresponding run tree.
The supplied legacy Main sessions are in `_output/freeform_runs.zip`, with run
folders directly at the archive root. The browser reads them without extraction.

Benchmark history discovers ZIP parts under each agent's
`test_run_full_capability/` and `test_run_ablation/` folders. The question table
scrolls vertically and horizontally with sticky headers. Open an execution to
view its result and saved evaluation directly from the ZIP, without extraction.
The Result panel's **Eval** switch starts on when numerical validation or claim
review is available; otherwise it is disabled. Select highlighted claims or
values to inspect their review, comparison details, sources, and related
DOI/block links in the sidebar. Turn Eval off for the plain answer. These views
use saved records and do not launch an evaluation or model request.

The Agents page submits a live request to the configured model service. The
browser currently manages one active agent run at a time. This is a process
execution limit, independent of the shared output layout.

| Location relative to the project root | Contents |
| --- | --- |
| `_output/freeform_runs.zip` | Archived legacy Main freeform sessions |
| `_output/Main` | New Main freeform sessions |
| `_output/Query` | Query freeform sessions and diagnostic subfolders |
| `_output/Analysis` | Analysis freeform sessions |
| `_benchmark/{Main,Query,Analysis}` | Full-capability/ablation ZIP collections and ordinary benchmark campaigns |

Session artifacts can include result Markdown, working memory, histories,
statistics, CSVs, plots, and workflow figures. Their presence depends on the
run and version that produced it. Closing a browser tab does not delete a run.
Source values, fitted values, and model interpretations should remain
identifiable when reusing an answer.

## Python APIs

First run `python database_archives.py` from the repository root if the database
files have not been prepared. Then run Python from `ThermoML_research_agent/`
so the packages are importable.
These examples send live model requests; they are not offline smoke checks.

Main:

```python
from NIST_ThermoML_agents._NIST_ThermoML_main_agent.ThermoML_main_api import ThermoML_main_run
result = ThermoML_main_run("Find reported density measurements for water.")
```

Query requires an explicit memory path or session directory:

```python
from pathlib import Path
from uuid import uuid4
from NIST_ThermoML_agents.NIST_ThermoML_query_agent.ThermoML_query_api import ThermoML_query_run
session = Path("../_output/Query") / f"run_{uuid4().hex}"
result = ThermoML_query_run("Find reported density measurements for water.", session_dir=session)
print(result.answer)
```

Analysis:

```python
from NIST_ThermoML_agents.NIST_ThermoML_analysis_agent.ThermoML_analysis_api import ThermoML_analysis_run
result = ThermoML_analysis_run("Inspect available liquid-density measurements for water.")
```

All three APIs accept a `session_dir` override. Child-agent work can therefore
remain nested inside a parent session. Retain the returned response together
with the session directory when using these APIs directly.

For a reproducible result, retain the prompt, provider/model settings, source
identifiers, data snapshot, and output artifacts. Historical files alone do not
establish the behavior or performance of a newly modified agent.
