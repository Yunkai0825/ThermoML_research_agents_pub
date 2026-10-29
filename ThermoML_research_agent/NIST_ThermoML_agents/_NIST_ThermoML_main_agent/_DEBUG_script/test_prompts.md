# Test Prompts for the ThermoML Main Agent

Science-level queries that require orchestrating across the query and
analysis agents.  The main agent must decide which subagent(s) to call,
combine results, and produce a coherent scientific answer.

**Data availability notes** (validated against the card database):
- Ethanol-water binary: density (29 blocks), viscosity (14), HE (6), speed of sound (5)
- Methanol-water binary: density (18), viscosity (6), VE (2), HE (2)
- DMSO-water binary: density (9), viscosity (3), speed of sound (3)
- 1-Propanol-water binary: density data available
- Acetonitrile-DMF binary: viscosity (2 blocks, 2 papers, 86 observations)
- Methanol-ethanol binary: viscosity (1 block, 1 paper, 12 observations)
- Water-ethylene-glycol binary: viscosity (7 blocks, 7 papers, 302 observations)
- Volumetric comparison: methanol-water has direct excess molar volume
  (2 blocks, 2 papers, 240 observations) plus density (18 blocks, 17 papers,
  1,610 observations); ethanol-water has no direct excess-volume block but
  has density (29 blocks, 27 papers, 2,895 observations); acetonitrile-DMF
  likewise has no direct excess-volume block but has density (3 blocks,
  3 papers, 188 observations). These are system-wide counts before state
  filtering.

---

## 1 — Simple Direct Tasks

### 1.1 — Chemistry Property Retrieval (query agent only)

| # | Prompt |
|---|--------|
| 1.1.1 | I am preparing an equimolar ethanol + water liquid mixture at 25 °C and atmospheric pressure. What experimental density should I expect? |
| 1.1.2 | What is the measured viscosity of an equimolar methanol + water liquid mixture at 25 °C and atmospheric pressure? |
| 1.1.3 | What surface tensions have been reported for liquid methanol + acetonitrile mixtures near room temperature, and how do they vary with composition? |
| 1.1.4 | For an equimolar acetonitrile + N,N-dimethylformamide liquid mixture at 25 °C, what experimental viscosity values have been reported? |
| 1.1.5 | Using ThermoML, which binary mixtures of ethylene carbonate, dimethyl carbonate and propylene carbonate remain liquid at 253.15 K, and over what composition range? Report the pure-component melting temperatures and the eutectic temperatures with their uncertainties and sources, and say what the data cannot determine. |

---

### 1.2 — Organized Property Fits and Coefficients (analysis agent only)

| # | Prompt |
|---|--------|
| 1.2.1 | At 25 °C, give me a Redlich–Kister correlation for the density of liquid ethanol + water as a function of ethanol mole fraction. Report the fitted equation, polynomial order, coefficients with units, R², RMSE, and composition range. |
| 1.2.2 | At 25 °C, what Redlich–Kister correlation best represents the density of liquid methanol + water? Select the polynomial order by BIC and report the fitted equation, coefficients with units, R², RMSE, BIC, and composition range. |
| 1.2.3 | At 303.15 K and 81.5 kPa, give me a Redlich–Kister correlation for the viscosity of liquid methanol + ethanol as a function of methanol mole fraction. Define the fitted viscosity basis, then report the equation, polynomial order, coefficients with units, R², RMSE, and composition range. |
| 1.2.4 | Near 25 °C, give me a Redlich–Kister correlation for the viscosity of liquid water + ethylene glycol using a single state-aligned composition series. State the selected temperature, pressure, composition basis, and viscosity basis, then report the equation, polynomial order, coefficients with units, R², RMSE, and composition range. |

---

## 2 — Context-Rich Direct Chemistry Tasks

These are still simple direct tasks, but the user supplies the experimental or
literature-review purpose behind the request.  The prompts do not name agent
tools; the main agent must recognize that direct identity resolution, block
search, pure-value lookup, or block inspection is sufficient.

| # | Prompt |
|---|--------|
| 2.1 | I am combining aqueous-alcohol measurements reported under the names “ethanol,” “ethyl alcohol,” “water,” and “aqua.” Before I compare the measurements, which canonical ThermoML compounds do these names refer to? |
| 2.2 | I am planning density measurements for aqueous 1-propanol at 25 °C. What mixture-density measurements near 298.15 K are already available, and what experimental density is reported for pure water at the same temperature for comparison? |
| 2.3 | I want to reuse the measurements in block 5 of DOI 10.1016/j.jct.2006.09.015. Before fitting them, what chemical system and property does the block contain, which variables and constraints define the measurements, and what temperature, pressure, composition, and property ranges were reported? |

---

## 3 — Multi-Step Chemistry Tasks

These questions combine evidence retrieval, property correlation or
comparison, and molecular interpretation.  They are phrased in terms of the
chemical behavior of interest; the main agent determines the required query,
analysis, and parallel-comparison steps.

| # | Prompt |
|---|--------|
| 3.1 | If equal mole amounts of ethanol and water are mixed at 25 °C, is their viscosity farther from ideal mixing than that of an equimolar methanol + water mixture? What intermolecular interactions explain the difference? |
| 3.2 | At 25 °C, which alcohol—methanol, ethanol, or 1-propanol—produces the greatest volume contraction when mixed with water, at what composition does it occur, and why does the behavior change with alcohol chain length? |
| 3.3 | At 30 °C, which mixture—methanol + ethanol, acetonitrile + N,N-dimethylformamide, or water + ethylene glycol—departs most strongly from ideal viscosity across composition, and what does the ordering reveal about molecular interactions in the three liquids? |
| 3.4 | At 25 °C and atmospheric pressure, which mixture—methanol + water, ethanol + water, or acetonitrile + N,N-dimethylformamide—contracts most on mixing, at what composition is the contraction greatest, and how can molecular packing and intermolecular attractions account for the result? |
| 3.5 | When ethanol and water are mixed at 25 °C, is heat released or absorbed? At what composition is the heat effect largest, and what molecular rearrangements produce its composition dependence? |
| 3.6 | Water and dimethyl sulfoxide are known to interact strongly. Near 25 °C, at what DMSO composition is this interaction most evident in the measured thermodynamic or transport behavior, and what molecular picture explains it? |
| 3.7 | For trans-1,3,3,3-tetrafluoropropene (R1234ze(E)), use the vapour-pressure data in ThermoML to estimate the vapour pressure and the enthalpy of vaporization at 273.15 K, and estimate the critical pressure at the reported critical temperature; give uncertainties, state your approximations, and say which data sets agree. |

---

## 4 — Edge Cases and Error Handling

| # | Prompt |
|---|--------|
| 4.1 | Estimate the viscosity of ethylene trioxide-water system using ThermoML data. |
| 4.2 | Find density data for ethanol-water at 1000 K. If no data exists at that temperature, report the closest available temperature range. |
| 4.3 | What viscosity data are available for rubbing alcohol mixed with water at room temperature? |
| 4.4 | What happens when ethanol is mixed with water near room temperature? |
| 4.5 | What does ThermoML report for the energy of combustion of D-glucose? Derive the standard enthalpy of formation of crystalline glucose at 298.15 K and propagate the uncertainties, using ΔfH°(CO2, g) = −393.51 ± 0.13 kJ/mol and ΔfH°(H2O, l) = −285.830 ± 0.040 kJ/mol (CODATA key values; treat these uncertainties as expanded, k = 2, like the ThermoML values). |

Prompts 4.3 and 4.4 intentionally leave different information unresolved.
Prompt 4.3 has a clear property target but an ambiguous common chemical name;
Prompt 4.4 identifies the chemical system but not the property or scientific
outcome of interest.  The agent should clarify the unresolved meaning rather
than silently choosing a compound or target property.
