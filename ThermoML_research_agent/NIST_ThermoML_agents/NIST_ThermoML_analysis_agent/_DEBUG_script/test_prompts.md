# Test Prompts for the ThermoML Analysis Agent

Analysis-focused queries for validating the full pipeline:
data retrieval → ideal baseline → excess property → RK fitting → prediction.

---

## 1 — Binary System Fitting (end-to-end baseline)

| # | Prompt |
|---|--------|
| 1.1 | At 25 °C, how does the viscosity of liquid ethanol + water depart from ideal mixing across composition? Give me a Redlich–Kister correlation with the fitted equation, coefficients and units, polynomial order, fit quality, and valid composition range. |
| 1.2 | At 25 °C, what Redlich–Kister correlation best represents the density of liquid methanol + water? Select the polynomial order by BIC and report the equation, coefficients and units, R², RMSE, BIC, and composition range. |
| 1.3 | At 25 °C, what is the excess molar volume of liquid acetone + water across composition? Give me a Redlich–Kister correlation with its equation, coefficients and units, polynomial order, R², RMSE, and valid composition range. |
| 1.4 | Fit a Redlich–Kister expansion to the excess molar enthalpy of benzene + cyclohexane at 298.2 K and 100 kPa from ThermoML, weighting the points by their reported uncertainties; choose the order, report the coefficients with uncertainties and say whether the residuals are within the reported uncertainties. |

---

## 2 — Ternary System Estimation (agent must decompose)

The prompts state only the ternary goal — discovering the
binary-subsystem decomposition is part of the test.

| # | Prompt |
|---|--------|
| 2.1 | I am preparing an equimolar DMF + water + acetonitrile liquid mixture at 25 °C. What viscosity should I expect? State whether the value is directly measured or estimated, and explain how the three binary interactions support the result. |
| 2.2 | What molar volume should I expect for an equimolar DMF + water + methanol liquid mixture at 25 °C? State whether the value is directly measured or estimated, and explain how the component and binary-mixture behavior supports it. |
| 2.3 | In an equimolar ethylene glycol + water + methanol liquid mixture near 25 °C, which binary interaction should contribute most strongly to the viscosity nonideality, and why? |

---

## 3 — Property-Specific Analysis

| # | Prompt |
|---|--------|
| 3.1 | At 25 °C, which liquid mixture—ethanol + water or methanol + water—shows the larger viscosity departure from ideal mixing, at what composition is it greatest, and what intermolecular interactions explain the difference? |
| 3.2 | At 25 °C, does the speed of sound in liquid ethanol + water deviate positively or negatively from ideal mixing, where is the deviation largest, and what molecular behavior explains its sign? |
| 3.3 | As temperature rises, does the volume change on mixing propan-2-ol + water become more or less ideal, and what does the density evidence imply about the underlying liquid structure? |

---

## 4 — Similarity Search + Data Gap Analysis

| # | Prompt |
|---|--------|
| 4.1 | I need to estimate the viscosity of aqueous N-methyl-2-pyrrolidone near 25 °C, but direct measurements may be unavailable. Which chemically similar aqueous solvent system offers the most defensible surrogate, and what similarities and measured behavior justify that choice? |
| 4.2 | I am planning measurements on liquid DMSO + water near room temperature. Which thermophysical properties and composition–temperature regions are already well characterized, and where would new measurements add the most useful chemical information? |

---

## 5 — Hypothesis / Open Goal (agent designs the plan)

Plain researcher questions with no workflow hints: the agent must
design the fitting campaign itself — which systems, which blocks,
single fit vs temperature sweep vs multi-system, and whether to chain
fitted RK coefficients into predict_from_rk. Architectural boundary:
all fits draw data points from DB blocks by DOI + block_number; the
only agent-supplied numbers are pure-component values and RK
coefficients, so prompts here must be answerable that way.

| # | Prompt |
|---|--------|
| 5.1 | At 25 °C, does the nonideality of aqueous alcohol mixtures become stronger from methanol to ethanol to 1-propanol, by how much, and what molecular changes produce the trend? |
| 5.2 | At 25 °C, at what ethanol mole fraction is the viscosity of liquid ethanol + water greatest, and how does that composition shift as temperature rises? |
| 5.3 | What density should I expect for a liquid mixture containing one mole of ethanol per three moles of water at 310 K? Explain the measured evidence and correlation used for the prediction and give its applicable range. |
| 5.4 | Near 25 °C, at what composition does liquid DMSO + water depart most strongly from ideal mixing, and does the answer change when volume, enthalpy, viscosity, or another measured property is considered? |
| 5.5 | What viscosity should I expect for an equimolar DMF + ethylene glycol + water liquid mixture at 25 °C? If it has not been measured directly, give the most defensible estimate, explain the chemical and mathematical basis of every inferred contribution, and identify the measurements supporting it. |

---

## 6 — Edge Cases

| # | Prompt |
|---|--------|
| 6.1 | I need a Redlich–Kister density correlation for liquid ethanol + water at 350 K. Can the available measurements support that temperature directly? If not, what is the nearest defensible temperature and why should the 350 K prediction be withheld? |
| 6.2 | What viscosity should I expect for an equimolar 1,2-dimethanol + water liquid mixture at 25 °C? |
| 6.3 | From the temperature dependence of the Henry's law constants of carbon dioxide in 1-butyl-3-methylimidazolium bis(trifluoromethylsulfonyl)imide in ThermoML, estimate the enthalpy of solution with its uncertainty, and check whether the low-pressure solubility measurements of other papers are consistent with those Henry constants. |
| 6.4 | Estimate the enthalpy of sublimation of naphthalene at 298.15 K from the crystal vapour-pressure data in ThermoML, with an uncertainty; compare it with any directly reported enthalpy of sublimation, and say which data sets you excluded and why. |
