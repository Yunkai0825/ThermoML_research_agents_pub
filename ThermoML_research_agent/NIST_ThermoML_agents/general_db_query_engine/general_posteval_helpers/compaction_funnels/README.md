# Compaction funnels — registry-driven session Sankeys

**Goal:** for every agent session, draw where the context volume went —
in four currencies (**chars**, unique **IDs**, qualified **blocks**,
datapoint **pts**) — from raw tool results through compaction, agent
contexts, synthesis, and envelopes, down to the final answer.

Rebuilt 2026-08-15 as a declarative, registry-driven package. Legacy
sessions (without exhaustive `**Nest:**` histories and the
`## Tool Compaction Pipeline` table) are **refused** with a clear
error — no re-parsing heuristics, no fabricated reconstruction.

## Architecture (four subpackages)

```
framework_registry/   WHAT the framework is (static, no session IO)
  tool_registry.py      ToolSpec per tool: owning catalog, det compactor?,
                        agentic triage?, exempt, inspection, dispatcher →
                        child agent kind, menu access, SUBSTEP_KEYS
  agent_registry.py     AgentSpec per layer (MAIN, Q_L0, Q_L1, A_L0,
                        A_L1_ALIGN, A_L1_QUERY, L2_COMP/MEAS/REF/PROP):
                        callable tools, dispatchables, envelope schema
                        (ANSWER/WRITTEN/WM/HYDRATED/LEDGER/VERDICT/
                        SCAFFOLD field classes), gate config, support
                        sources
  stage_templates.py    StageBlockSpec grammar: roles, upstream/downstream
                        adjacency, identity applicability, agentic_group /
                        pooled_group merge keys, sink families

funnel_evidence/      WHAT actually happened (logged artifacts only)
  history_evidence.py   exhaustive run_history.md: Nest breadcrumbs, step
                        blocks + verbatim results, pipeline table,
                        subagent summary, errors (LegacySessionError)
  event_evidence.py     run_history_detailed.md event log: per-tool triage
                        argo sizes, doubled-Tool# nesting, turn sizes
  stats_evidence.py     reference_stats §4/§6 + run windows
  envelope_evidence.py  result.md envelope partition + Data Inspections
                        ledger + INSP anchors
  memory_evidence.py    working_memory.md payload records
  measures.py           Measure {chars, ids, blocks, pts} + id extraction
                        + block→pts DB metadata hydration
  reexec.py             SANCTIONED re-execution: pre-delivery stage
                        text/ids for pure DB tools, FROM VERBATIM LOGGED
                        ARGS only; sizes reconciled to the pipeline table
  delegation_tree.py    AgentInstance/ToolUse tree — type-agnostic
                        parenthood (parent = actual dispatcher), menu
                        unwrap, full dispatch kwargs on every node

funnel_assembly/      HOW it is drawn
  skeleton_build.py     recursive emit(instance): canonical DETAILED
                        skeleton (chars+ids+blocks+pts+args+provenance
                        on every node/edge)
  graph_model.py        Skeleton/nodes/edges + merged(group_key)
  projections.py        identity projections of one skeleton
  render_export.py      plotly figures + edges/nodes TSVs
  build.py              orchestrator: detailed → agentic → pooled →
                        per-agent sub_ figures → session figures

workflow_audits/      registry_audits.py — conservation, template
                      adjacency, evidence coverage, envelope partition,
                      reexec drift → registry_audit_report.tsv
```

Entry points: `api.py` (`generate_workflow`, engine seam
`maybe_generate_workflow` — unchanged signatures) and
`DEBUG_runner.py` (fixture regression).

## Doctrine

1. **Canonical detailed skeleton.** One graph per session, built by
   recursively instantiating each agent's stage template over the
   actual delegation tree. `chars_detailed` and `ids_detailed` are its
   canonical projections; `blocks`/`pts` derive from the same
   skeleton's id sets (bare ids canonicalized onto their unique
   qualified twin; pts weighted by DB datapoint counts).
2. **Derived levels.** `agentic` = node-merge of detailed by
   (instance × tool family × stage). `pooled` = the LAYER STACK:
   same-type nodes of each delegation layer (main / Q / A / L1 / L2)
   merge — one context, answer, ONE merged envelope node and one tool
   rail per layer — plus consolidated pools: supporting content
   (briefs · seeds · scaffolding · relay wrappers · utility/error
   returns · generic rendering expansion · agent-written LLM
   additions, which keep their red `introduced` flow edges), memory
   carry (WM render ·
   transient; the WM archive stays a separate store — merging it
   with the render would close a context→memory→context cycle),
   inspection evidence and DB hydration collapsed to ONE
   deterministic-evidence pool across all agents (inspection tool
   stages + the ledger + workflow enrichment + the hardcoded_data
   deterministic DB-read rail + their rendering growth; each
   inspected id exits once). Merges only union ids / sum chars and
   parallel contributions stay separate edges, so in-flows ADD
   per-instance copies (L1 tool #1 + L2 returns + …); ids/blocks/pts
   deduplicate ONLY at each layer's agent context, whose forward flow
   carries the deduplicated union (the pooled L1 answer = union of
   all L1 answers). Cross-layer sinks: hardcoded compaction cut,
   agentic triage cut, condensed-in-synthesis, not-cited-downstream,
   dedup (re-confirmed known IDs) and memory-only carry (WM retained
   + uncited memory carry + uncited ledger) each merge into ONE sink
   across all agents; the remaining sinks stay per-layer.
   Conservation re-validated after every pass.
3. **Context merge = dedup layer.** Every tool chain and every child
   relay targets the DISPATCHING agent's context block; ids/blocks/pts
   dedup there (duplicates exit via the dedup sink), chars are
   additive.
4. **Type-agnostic parenthood.** Parent = whoever actually dispatched
   (an L1 under Main's menu is Main's child; an L2's tools belong to
   the L2 instance, never to L1 directly).
5. **Menu expansion.** `run_subagent_tool(tool_name=X)` renders as the
   CALLER's tool ("Main · X (menu)") running the owning catalog's
   det + triage pipeline inside the caller's lane.
6. **Logged-first evidence.** Stage sizes come from the logged
   pipeline table. Verbatim text exists only for the delivered stage,
   so pre-delivery id content is re-executed from the verbatim logged
   arguments (pure DB read tools only; stateful fit tools stay
   chars-only and inherit det ids). Re-exec sizes are audited against
   the logged table.
7. **Support semantics.** BRIEF (from dispatch kwargs) / SEED /
   WM_RENDER / SCHEMA_SCAFFOLD / RELAY_WRAPPER / LEDGER are
   first-class support blocks; the inspection ledger feeds the
   envelope directly (deterministic side-channel), never through
   context. System prompts are nodes-TSV metadata, not Sankey mass.
8. **Strict conservation.** Every block is minted with declared
   `source`/`sink` flags: sources (tool emissions — raw or
   delivered-as-is exempt/error calls — and supportive contexts:
   seed, briefs, WM render, scaffolding, hydration, expansion, relay
   wrappers, declared LLM additions) may emit without inflow but
   never receive; sinks (cut/dedup/not-cited/retained families, the
   terminal answer, a parentless root envelope) absorb but never
   emit. Every other node balances EXACTLY — in == out per metric.
   Downstream lanes DRAW from the caller's
   context budget (chars capped, each id exits once — the answer
   lane draws first); duplicate copies converging on an envelope or
   batch collector exit via the duplicate-canonical dedup sink.
   WM recording is a chars-only copy drawn out of the context into
   the archive store (unshipped payload exits via the WM-retained
   sink); the WM render into the prompt is a declared
   supportive-context root. Set metrics additionally pass a
   topological single-exit pass, and surplus copies that set-edges
   cannot carry (3rd+ duplicates, alias pairs riding one lane) flush
   from aggregation nodes into the explicit "duplicate copies
   collapsed" sink — an id that would vanish entirely (zero exits)
   is NOT flushed and fails the audit.
   `audit_flow_conservation` (per node, exact) and
   `audit_layer_balance` (Σ source emission == Σ sink intake per
   layer × metric; no stranded ids) enforce all of this on the exact
   graphs that render.

## Outputs per session (`<session>/workflow/`)

- 12 matrix figures: `{chars|ids|blocks|pts}_{detailed|agentic|pooled}_sankey_<case>.{png,html}`
- 2 whole-session figures: `session_{chars|ids}_sankey_<case>.*`
- per-agent subgraphs: `sub_<agent>_{chars|ids}_detailed_{query|analysis}_sankey_<case>.*`
- per figure: `*_edges.tsv` (Source/Target/Value, source_full/
  target_full = NODE KEYS, flow_type, ids, provenance, edge_order,
  appearance_order/appearance_key = the deduplicated node-key sequence
  in first-seen source_full-then-target_full order; if that sequence is
  longer than the edge list, metadata-only rows have blank edge fields)
  and `*_nodes.tsv` (key, label, kind, role, **is_source/is_sink
  declared endpoint flags**, value, n_ids, **full id and block
  lists**, order, **artifact provenance anchors**, **full dispatch
  args JSON**, meta)
- `registry_audit_report.tsv` (FAIL/WARN/INFO)
- `png_skipped.tsv` — only when a figure of this build got no PNG
  (figure, png, reason = timeout / error / breaker / switched off,
  detail, older_png = an earlier build's PNG is still in the folder);
  a later build that exports every PNG removes it

Node keys are instance-path scoped
(`Main/A1/L1#2:search_blocks#3:raw`); provenance anchors point into
the artifacts (`run_history.md::Step 12 [main - A_1 - L1_1 · c14]`,
`result.md::data_inspections::INSP_…`).

## Regression fixtures

`run_library/fixtures.py` — the four 2026-08-15 hharb campaign mains
(copies under `_benchmark/Main/
hharb_campaign_20260815/`) plus one query child and one analysis child
as standalone roots.

```
python -B DEBUG_runner.py debug              # full render, 6 fixtures
python -B DEBUG_runner.py debug --no-render  # TSVs + audits only
```

Set `TEMP`/`TMP` to a drive with space before rendering (kaleido).

PNG export (kaleido) is bounded per figure by
`THERMOML_WORKFLOW_PNG_TIMEOUT` seconds (default 45);
`THERMOML_WORKFLOW_PNG=0` turns it off. Both are read when a workflow
build starts. The first timeout trips the breaker of THAT workflow
only: its later figures skip PNG at once, the next workflow build
starts fresh. The HTML is always written. Every skipped PNG is logged
on the `workflow-png` logger (so it lands in the run's `run_log.txt`)
and listed in `png_skipped.tsv` in the workflow folder.

## Engine seam

The three L0 orchestrators call `maybe_generate_workflow` after
`finalize_tracking` for every completed root session. Saved child sessions
are skipped because their root workflow includes them. The browser's workflow panel reads the stems and audit
reports directly.

Historical docs (`SANKEY_ABSTRACTION_LEVELS.md`,
`CONTEXT_FUNNEL_TREES.md`) describe the pre-2026-08-15 architecture
and are retained for reference only.
