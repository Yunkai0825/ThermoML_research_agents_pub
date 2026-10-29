# Grounding gate activation

The gate compares block-anchored and DOI-cited numerical claims with the
inspection ledger, including verbatim inspections received through child
handovers.

## Automatic nested-tool eligibility

Inspection tools reachable through the actual available delegation/menu tree
activate the same gate as directly exposed inspection tools. Dispatchers declare
their child tool maps through `bind_reachable_tools`; shared traversal follows
wrappers and handles cycles without executing tools. There are no agent-name
exemptions. The current profile is the default.

`THERMOML_GROUNDING_GATE=off` remains the existing master override for gate-off
runs. Passive evidence reports are still written. Removing the compatibility
setting does not rewrite historical answers or evaluation scores.

## Enforcement and diagnostics

The same capability check governs ordinary final-answer repair and the
forced-final rewrite/flag path. Normal repairs allow two retries; persistent
violations carry explicit `UNGROUNDED DATA FLAGS`. Numerical agreement thresholds
and the retry policy are unchanged.

The ReAct loop inserts a grounding repair message before accepting a failing
answer. The later passive `grounding_evidence` writer records diagnostics and
does not itself re-enter the model. Logs record `active_for_this_tool_tree`.
Regression tests cover actual nested Query tool maps, repair feedback, persistent
flags, forced-final handling, the master override, and provider cycles.

## Source coverage

A DOI-only answer is checked against the blocks for that DOI already present in
the inherited inspection records. Multiple blocks remain a candidate set; the
checker does not silently choose one block. Ambiguous source scope or missing
inspection produces a repair request. DOI-cited text before a closing block
citation is also checked, preventing a pass with no numerical claims examined.

Qualified block references retain their literature identity. Another paper's
`PROPblock_N` cannot supply inspection evidence for the cited paper. Explicit
Celsius temperatures are compared with inspected Kelvin cells, and grouped
thousands are parsed as single numbers. These corrections are normal behavior in the current profile.

## Evidence and subsequent handovers

Exported inspections preserve optional `grounding_lines`: verbatim statistics,
brackets, and block constraints already shown by the tool. They remain available
through repeated handovers without exposing unseen measurement rows.

Main, Query, and Analysis use the same post-answer checker for newly generated
core claims, summaries, source descriptions, and structured factual outputs.
Unsupported claims receive field-specific flags. Deterministic ledgers, request
metadata, identifiers, and proposed follow-up tasks are not new measurement
claims. Core claims may preserve existing source citations, never invent them.

Source inference ignores machine-appended diagnostics. While the gate is active,
the passive writer also excludes those appended sections so warnings do not
create duplicate claims. The master-off passive path retains its historical
reporting behavior.

## Limits

The checker matches numerical evidence within its source scope. It does not
fully bind every property value to the claimed temperature, pressure, or
composition row, and it does not establish the correctness of qualitative
chemical explanations or thermodynamic calculations.

## Historical grounding toggle

In the entry agent's `ThermoML_main_argo_config.py`,
`ThermoML_query_argo_config.py`, or `ThermoML_analysis_argo_config.py`, set:

```python
GROUNDING_GATE_LEGACY_MODE: bool = True
```

The default is `False` (current grounding). The outermost run captures this
setting for its complete child-agent tree: a Main run's setting governs its
Query/Analysis/L1 children even if their standalone settings differ. Standalone
Query or Analysis uses its own setting. Restart the server after editing these
Python configuration files. The existing `THERMOML_GROUNDING_GATE=off` still
disables active repair in either profile; historical mode does **not** mean off.

Historical mode selects `historical_september_2026`: frozen grounding and
inspection-harvest sources from commit
`bdf50547fb4cb8f1b5037dd55b4878ccb1cb21bf`. Their repair instruction matches
59 saved September 5 Main/Analysis repair messages. The source commit was
recorded September 8; no complete September 4/5 working-tree manifest survives.
The source files and hashes are in `historical_september_2026/manifest.json`.

This restores immediate-tool eligibility, original block-scoped numeric
extraction/matching, repair hints and prompts, original inspection handovers,
and the passive full-answer rescan. It intentionally retains the historical
nested-Query bypass, DOI-only/closing-citation coverage gaps, older Celsius and
number parsing, and block-number source ambiguity. It omits the later answer
continuity bounce and post-answer factual-field checker. Normal and forced-final
repair limits retain their September behavior.

The mode changes grounding only. Today's archive readers, inspection tools
(including uncertainty output), prompts, models, compaction and other tool
implementations remain current. It is not a full historical build or a promise
of identical generated answers. No saved benchmark answers are rewritten.
Run logs record the profile; generated `grounding_evidence.json` and `.md`
record it alongside the verdict when evidence is available.
