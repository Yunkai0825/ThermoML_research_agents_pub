# Test Prompts for the ThermoML Query Agent

Researcher-style queries for validating the agentic system.
Sections are ordered by reasoning type, from concrete retrieval to
abstract planning and graceful failure.

**Design rule**: prompts are written the way a chemist would actually
ask — short, goal-stated, no embedded step lists or interpretation
checklists. A brief justification tail ("and why?", "justify your
estimations") is fine; step-by-step instructions are not. They never
reveal the retrieval plan: the decomposition is the agent's job, and a
prompt that enumerates the sub-searches does not test that category.

---

## 1 — Direct Lookup (baseline)

Single fact, single system. Should succeed with 1-2 L1 tool calls
(name resolution + one search). Name resolution (DMF, IPA, EG, ACN)
is exercised implicitly throughout all sections.

| # | Prompt |
|---|--------|
| 1.1 | I am preparing an equimolar ethanol + water liquid mixture at 25 °C and atmospheric pressure. What experimental density should I expect? |
| 1.2 | For an equimolar liquid mixture of hexane + ethanol near 25 °C, which thermophysical properties have been measured? |
| 1.3 | What density has been measured for an equimolar propan-2-ol + water liquid mixture at 25 °C and atmospheric pressure? |
| 1.4 | Have activity coefficients been measured for aqueous lithium chloride near 25 °C, and over what concentration range? |
| 1.5 | How has the viscosity of pure liquid ethanol near 25 °C been measured, and what values were reported by the different methods? |
| 1.6 | What is the measured isobaric heat capacity of pure liquid water at 25 °C and atmospheric pressure? |
| 1.7 | Using ThermoML, report the surface tension of 1-butyl-3-methylimidazolium tetrafluoroborate ([C4mim][BF4]) at or near 298.15 K from every source, each with its reported expanded uncertainty and block identifier, and say whether the sources agree within those uncertainties. |
| 1.8 | What critical temperature, critical pressure and critical density of 1,1,1,2-tetrafluoroethane (R134a) does ThermoML hold? Give each value with its reported uncertainty and source, and say whether the sources are independent determinations. |

---

## 2 — Comparison

Two or more retrievals reduced to a single verdict. The prompt names
the systems but NOT the searches to run or the report format; the agent
must pick comparable metrics and commit to a conclusion.

| # | Prompt |
|---|--------|
| 2.1 | I need a binary liquid for testing a mixture-property model near room temperature. Between benzene + toluene and hexane + ethanol, which would let me test the model over a wider composition range and against more types of measured properties? |
| 2.2 | I am deciding whether to validate a liquid-mixture viscosity model first on binary or ternary systems. Which class has enough experimental measurements across composition to provide the stronger test? |
| 2.3 | For validating a liquid-density model near room temperature, should I use ethanol + water or methanol + water? Which system has measurements over the broader composition and temperature ranges? |
| 2.4 | I want to validate an acoustic model for aqueous mixtures. Is the available speed-of-sound evidence broad enough for this purpose, or is it substantially more limited than the corresponding density evidence? |

---

## 3 — Multi-Step Retrieval (dependent hops)

Later steps depend on earlier results — the agent must chain searches,
not just fan out independent ones. Requires L1 + L2 coordination.

| # | Prompt |
|---|--------|
| 3.1 | I need three binary liquid systems for jointly testing density and viscosity predictions. Which systems have both properties measured for the same mixture, and which papers should I start with? |
| 3.2 | I am building an activity-coefficient correlation for ethanol + water over the widest possible temperature range. What range can the experimental literature support, and which studies provide the coldest and hottest measurements? |
| 3.3 | For aqueous propan-2-ol, which study provides the most complete experimental characterization? What properties were measured, which methods were used, and what purities were reported for the compounds? |
| 3.4 | I need an ethanol + water study suitable for cross-checking several measured properties from the same experimental source. Which paper is the best candidate, and what other chemical systems does it report? |

---

## 4 — Hypothesis / Open Goal (agent designs the plan)

Each prompt is a plain researcher question — an estimate, a trend, a
recommendation. Nothing in the prompt says how to get there: a genuine
pass requires the agent to invent the decomposition (ternary → three
binary subsystems, homologous series → per-member searches, missing
compound → fingerprint-similarity fallback) and answer the science
question from the data it retrieves.

| # | Prompt |
|---|--------|
| 4.1 | What viscosity should I expect for an equimolar DMF + water + acetonitrile liquid mixture at 298 K? If no direct measurement is available, estimate it from the closest relevant measurements and explain the chemical basis of the estimate. |
| 4.2 | Does the nonideality of alcohol + water mixtures grow with the alcohol chain length? |
| 4.3 | What molar volume should I expect for an equimolar DMF + water + methanol liquid mixture at 298 K? If it has not been measured directly, estimate it from the closest relevant measurements and explain the chemical basis of the estimate. |
| 4.4 | I need a substitute for DMF in aqueous mixtures whose viscosity with water is well characterized. What are my options and why? |
| 4.5 | How strongly does water + ethylene glycol deviate from ideal mixing in its transport properties, and what's the chemical reason behind? |
| 4.6 | Among aqueous mixtures of organic solvents, how does viscosity change with the molecular structure of the organic component, and which intermolecular interactions explain the trend? |

---

## 5 — Ambiguous / Underspecified

Names or scope are ambiguous. A genuine pass = the agent resolves the
most likely interpretation AND states the assumption, or reports the
alternatives — it never silently guesses or fails.

| # | Prompt |
|---|--------|
| 5.1 | Do you have viscosity data for IPA + water? |
| 5.2 | What data is available for glycol + water mixtures? |
| 5.3 | Find density data for salt water. |
| 5.4 | I'm looking for heat capacity data for butanol. |

---

## 6 — Impossible / Out-of-Scope

The requested primary target is unavailable or out of scope. A genuine pass
states that limit without fabricated values or external-source citations.
When a prompt explicitly asks about related archive data, report that
archive-supported fallback. 6.4 directly probes the no-external-sources
guardrail.

| # | Prompt |
|---|--------|
| 6.1 | Find data for the binary system propane-1,2,4-triol + water. |
| 6.2 | What is the Gibbs free energy of formation for diamond at 1000 K? |
| 6.3 | What is the viscosity of ethanol + water mixtures at 2000 K? |
| 6.4 | I need the density of pure liquid DMSO at 25 °C for a calculation. If ThermoML does not contain it, use an accepted value from another source. |
| 6.5 | Does ThermoML contain Joule–Thomson coefficients for methane? If so, report them with their uncertainties and conditions; if not, say what Joule–Thomson data the archive does hold. |
