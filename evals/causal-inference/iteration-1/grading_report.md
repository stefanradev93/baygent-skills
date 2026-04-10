# Causal Inference Skill — Iteration 1 Grading Report

**Date:** 2026-03-20
**Grader:** Claude Opus 4.6 (automated)
**Scenarios:** 6 | **Conditions:** with_skill, without_skill | **Total assertions:** 68

---

## Scenario 1: did-policy-evaluation

### With Skill

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Proposes a DAG or explicitly states causal assumptions before estimation | PASS | Full DAG in Section 2 with nodes, edges, and explicit non-edges |
| 2 | Asks the user to confirm the causal graph or assumptions before proceeding | PASS | "Mandatory user checkpoint — DAG confirmed" in Section 2 |
| 3 | Identifies this as a Difference-in-Differences design (not just a regression) | PASS | Explicitly named "Difference-in-Differences (DiD)" throughout, identified as design choice |
| 4 | States the parallel trends assumption explicitly and explains what it means | PASS | "In the absence of the checkout system, the treated and control stores would have followed the same revenue trajectory" — full explanation with testability table |
| 5 | Asks the user whether the parallel trends assumption is defensible | PASS | "Mandatory user checkpoint — assumptions confirmed. The user confirms these assumptions are defensible" |
| 6 | Uses CausalPy's DifferenceInDifferences class (not manual PyMC regression) | PASS | CausalPy DiD model used with formula specification; CausalPy 0.8.0 cited |
| 7 | Uses a PyMC-backed model (cp.pymc_models.*), not sklearn | PASS | "CausalPy's LinearRegression model uses weakly informative defaults from PyMC" with nutpie sampler |
| 8 | Runs at least one refutation or robustness check | PASS | 4 refutation tests: placebo time, random common cause, placebo treatment permute, data subset |
| 9 | Reports the effect estimate with a credible interval (HDI), not just a point estimate | PASS | "$7,015/month" with 50%, 89%, and 95% HDIs reported |
| 10 | Uses causal language appropriately | PASS | Uses "causes" with hedging: "strong suggestive evidence of a positive causal effect, rather than a definitive causal claim" |
| 11 | Produces or offers to produce a report with assumptions, results, and limitations sections | PASS | Full structured report with all sections including Assumptions (Sec 2), Results (Sec 5), Limitations (Sec 7) |
| 12 | States limitations or threats to validity | PASS | Section 7 ranks 4 threats by severity: unobserved confounding (HIGH), parallel trends (MODERATE), SUTVA (LOW-MODERATE), placebo FAIL (LOW) |

**Pass rate: 12/12 (100%)**

### Without Skill

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Proposes a DAG or explicitly states causal assumptions before estimation | FAIL | No DAG proposed. Lists assumptions but no causal graph |
| 2 | Asks the user to confirm the causal graph or assumptions before proceeding | FAIL | No user confirmation checkpoint |
| 3 | Identifies this as a Difference-in-Differences design | PASS | Clearly identifies as DiD: "This is a classic panel data setup suited for Difference-in-Differences" |
| 4 | States the parallel trends assumption explicitly and explains what it means | PASS | "In the absence of treatment, treated and control stores would have followed the same revenue trend" with explanation |
| 5 | Asks the user whether the parallel trends assumption is defensible | FAIL | No user interaction or confirmation requested |
| 6 | Uses CausalPy's DifferenceInDifferences class | FAIL | Uses statsmodels OLS with manual TWFE implementation, not CausalPy |
| 7 | Uses a PyMC-backed model, not sklearn | FAIL | Uses statsmodels OLS, not PyMC |
| 8 | Runs at least one refutation or robustness check | PASS | Placebo test (July 2023 fake treatment) and event study for parallel trends |
| 9 | Reports the effect estimate with a credible interval (HDI) | FAIL | Reports frequentist 95% CI, not HDI/credible interval |
| 10 | Uses causal language appropriately | PASS | "strong evidence that the new checkout system caused the revenue increase" with appropriate caveats |
| 11 | Produces or offers to produce a report with assumptions, results, and limitations | PASS | Has interpretation section and "Important caveats in practice" |
| 12 | States limitations or threats to validity | PASS | Lists parallel trends untestable, staggered adoption, spillovers, heterogeneous treatment effects |

**Pass rate: 6/12 (50%)**

---

## Scenario 2: did-parallel-trends-violation

### With Skill

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Identifies this as a DiD design but flags the pre-treatment trend difference as a threat | PASS | "The honest starting point: parallel trends is violated" — identified as DiD then immediately flagged |
| 2 | Does NOT blindly proceed with standard DiD without addressing the parallel trends violation | PASS | Runs naive DiD only as "bias illustration," uses DiD+trends and SC as alternatives |
| 3 | Explicitly states that parallel trends appears violated based on the user's description | PASS | "This assumption is violated here by construction. The treatment cities were declining faster even before 2023" |
| 4 | Suggests checking pre-treatment trends visually or formally before proceeding | PASS | Visual inspection and formal slope comparison: treated -0.56/yr vs control -0.21/yr |
| 5 | Either refuses to make causal claims OR uses an alternative method | PASS | Uses DiD+group-specific trends and Synthetic Control as alternatives; runs 3 models |
| 6 | Asks the user about confounders (wealth, healthcare access) | PASS | DAG checkpoint asks "Is there a policy spillover?" and identifies W (wealth/healthcare) as key confounder |
| 7 | If it proceeds with estimation, runs a parallel trends test or placebo test | PASS | Placebo treatment time test and visual parallel trends inspection both run and reported |
| 8 | Does NOT use causal language if parallel trends is violated | PASS | "associated with" and "suggestive language" explicitly chosen; causal language guardrail section at end |
| 9 | Reports limitations prominently — does not bury them | PASS | Full Section 8 with 5 threats ranked by severity, "CRITICAL" label for parallel trends |
| 10 | Acknowledges that the cities' self-selection into treatment is itself a confounding concern | PASS | "Richer cities adopted the tax" identified as confounding; DAG shows W→T and W→Y paths |

**Pass rate: 10/10 (100%)**

### Without Skill

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Identifies this as a DiD design but flags the pre-treatment trend difference as a threat | PASS | "Standard DiD cannot identify the causal effect here" — immediate flag |
| 2 | Does NOT blindly proceed with standard DiD without addressing the parallel trends violation | PASS | Shows naive DiD as "what goes wrong" then proposes 3 alternatives |
| 3 | Explicitly states that parallel trends appears violated | PASS | "The gap between treated and control groups is already closing before 2023, confirming the violation" |
| 4 | Suggests checking pre-treatment trends visually or formally before proceeding | PASS | Visual inspection and formal pre-trend test with t-test on slopes |
| 5 | Either refuses to make causal claims OR uses an alternative method | PASS | Uses trend-adjusted DiD, Synthetic Control, and Rambachan-Roth sensitivity |
| 6 | Asks the user about confounders (wealth, healthcare access) | FAIL | No user interaction or confirmation about confounders requested |
| 7 | If it proceeds with estimation, runs a parallel trends test or placebo test | PASS | Formal pre-trend test run with t-test, visual inspection provided |
| 8 | Does NOT use causal language if parallel trends is violated | PASS | "Do NOT report naive DiD as your main result — it is materially biased" |
| 9 | Reports limitations prominently — does not bury them | PASS | "Key Takeaways" section prominently discusses what went wrong and limitations |
| 10 | Acknowledges that the cities' self-selection into treatment is a confounding concern | PASS | "richer cities with better healthcare access" identified as the driver of differential trends |

**Pass rate: 9/10 (90%)**

---

## Scenario 3: rdd-scholarship-threshold

### With Skill

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Identifies this as a Regression Discontinuity Design | PASS | "Design: Regression Discontinuity (RDD)" — identified immediately |
| 2 | Proposes a DAG or states the causal identification strategy | PASS | Full DAG with nodes and edges in Section 2; "the threshold creates local quasi-random assignment" |
| 3 | States the key assumption: no manipulation at the threshold | PASS | "The RDD design rests on the assumption that students cannot precisely manipulate their exam score to land just above 80" |
| 4 | Asks the user whether students could manipulate their score | PASS | Checkpoint 3 explicitly asks about manipulation; "Has there been any institutional knowledge of 'teaching to the test'?" |
| 5 | Uses CausalPy's RegressionDiscontinuity class | PASS | `cp.RegressionDiscontinuity(...)` with full code shown |
| 6 | Uses a bandwidth parameter to restrict analysis near threshold | PASS | `bandwidth=10.0` and sensitivity from 5 to 20 |
| 7 | Runs at least one robustness check | PASS | McCrary density, bandwidth sensitivity, covariate balance, placebo thresholds, 3 DoWhy refuters — 8 total |
| 8 | Reports the Local Average Treatment Effect (LATE), not a global ATE | PASS | "Local Average Treatment Effect (LATE) at the threshold" — explicitly distinguished from ATE |
| 9 | Notes that the effect is local to students near 80 | PASS | "This is not the ATE for all students... applies to students scoring near 80, not for all students" |
| 10 | Reports the effect with a credible interval (HDI) | PASS | "0.34-point increase in first-year GPA (95% HDI: [0.24, 0.43])" |

**Pass rate: 10/10 (100%)**

### Without Skill

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Identifies this as a Regression Discontinuity Design | PASS | "Sharp Regression Discontinuity Design (RDD)" |
| 2 | Proposes a DAG or states the causal identification strategy | PASS | States identification strategy: "cutoff creates a natural experiment" — though no formal DAG drawn |
| 3 | States the key assumption: no manipulation at the threshold | PASS | "students cannot precisely manipulate their score to just cross the threshold" |
| 4 | Asks the user whether students could manipulate their score | FAIL | No user interaction requested |
| 5 | Uses CausalPy's RegressionDiscontinuity class | FAIL | Uses manual numpy OLS implementation, not CausalPy |
| 6 | Uses a bandwidth parameter to restrict analysis near threshold | PASS | Multiple bandwidths tested (5, 7, 10, 15, 20) with IK optimal bandwidth |
| 7 | Runs at least one robustness check | PASS | McCrary density, covariate smoothness, placebo cutoffs, donut RDD, bandwidth sensitivity |
| 8 | Reports the LATE, not a global ATE | PASS | "Local Average Treatment Effect at the threshold" — explicitly identified |
| 9 | Notes that the effect is local to students near 80 | PASS | "This is a LOCAL effect — it applies only to 'marginal' students near the threshold, not to all students" |
| 10 | Reports the effect with a credible interval (HDI) | FAIL | Reports frequentist 95% CI, not Bayesian HDI |

**Pass rate: 7/10 (70%)**

---

## Scenario 4: synthetic-control-poor-donors

### With Skill

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Identifies this as a Synthetic Control design | PASS | "Design selected: Synthetic Control (CausalPy)" |
| 2 | Warns about the convex hull problem | PASS | Full Section 3 "Critical Pre-Check — Convex Hull Violation" with quantification: "40% below the donor floor" |
| 3 | Asks the user about California's pre-treatment characteristics relative to donor states | PASS | Checkpoint asks user to confirm convex hull problem and whether to proceed |
| 4 | Uses CausalPy's SyntheticControl class with WeightedSumFitter | PASS | `cp.SyntheticControl(... model=cp.pymc_models.WeightedSumFitter(...))` |
| 5 | Checks pre-treatment fit quality | PASS | Pre-treatment RMSE computed and compared to gap; RMSE/gap ratio threshold of 10% |
| 6 | If pre-treatment fit is poor, flags this as a threat to validity | PASS | "CRITICAL refutation failure. The analysis cannot support causal claims in its current form" |
| 7 | Runs at least one robustness check | PASS | Pre-treatment RMSE, leave-one-out donors, placebo treatments on controls, donor weights inspection |
| 8 | Reports the effect with uncertainty | PASS | HDI reported: "95% HDI approximately [-25, -5]" |
| 9 | States limitations: if outside convex hull, synthetic counterfactual may be unreliable | PASS | "California's uniquely low pre-treatment smoking rates place it outside the convex hull... The SC design is fundamentally inappropriate here" |
| 10 | Does not make strong causal claims if pre-treatment fit is poor | PASS | "CAUSAL LANGUAGE: NOT WARRANTED" — explicitly refuses causal claims, uses "associational, not causal" |

**Pass rate: 10/10 (100%)**

### Without Skill

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Identifies this as a Synthetic Control design | PASS | "Synthetic Control Analysis" — clearly identified |
| 2 | Warns about the convex hull problem | PASS | "California is a poor candidate for synthetic control due to donor pool problems" — explicit warning |
| 3 | Asks the user about California's pre-treatment characteristics | FAIL | No user interaction requested |
| 4 | Uses CausalPy's SyntheticControl class with WeightedSumFitter | FAIL | Uses manual scipy.optimize implementation, not CausalPy |
| 5 | Checks pre-treatment fit quality | PASS | Pre-treatment RMSPE computed and reported prominently |
| 6 | If pre-treatment fit is poor, flags this as a threat | PASS | "A high pre-treatment RMSPE means the 'synthetic control' never convincingly resembled California" |
| 7 | Runs at least one robustness check | PASS | Placebo tests (in-space) on all donor states with RMSPE ratio inference |
| 8 | Reports the effect with uncertainty | PASS | Reports RMSPE ratio and permutation p-value for inference |
| 9 | States limitations: convex hull unreliability | PASS | "No convex combination of other states may closely approximate it" — limitations clearly stated |
| 10 | Does not make strong causal claims if pre-treatment fit is poor | PASS | "If pre-treatment RMSPE is >10, state explicitly that the donor pool is a poor fit" — appropriately hedged |

**Pass rate: 8/10 (80%)**

---

## Scenario 5: structural-mediation

### With Skill

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Proposes a DAG with training → confidence → interview → earnings AND training → earnings | PASS | Full DAG in Section 2 with both paths explicitly shown |
| 2 | Asks user to confirm DAG, especially unmeasured confounders between confidence and earnings | PASS | Checkpoint 2 asks "Does training affect interview performance through any route other than confidence?" and Checkpoint 3 asks about confounders |
| 3 | Identifies this as mediation analysis requiring a structural causal model | PASS | "The problem fits a Structural Causal Model (PyMC + pm.do/pm.observe)" |
| 4 | Uses pm.do() or pm.observe() to estimate interventional distributions | FAIL | Uses chained structural equations (Bayesian SEM) not pm.do()/pm.observe(). Computes NDE/NIE from posterior draws of structural coefficients |
| 5 | Distinguishes between Natural Direct Effect (NDE) and Natural Indirect Effect (NIE) | PASS | Full decomposition table: NDE=$2.97k (skills), NIE=$2.27k (confidence→interview), with proportion mediated |
| 6 | States the assumptions required for mediation | PASS | Full assumption transparency table with 8 assumptions including sequential ignorability |
| 7 | Asks the user whether these mediation assumptions are defensible | PASS | Checkpoint 3 asks explicitly about confounders of mediator-outcome path |
| 8 | Reports both direct and indirect effects with uncertainty (credible intervals) | PASS | NDE: $2.97k (94% HDI: [$2.40k, $3.57k]), NIE: $2.27k (94% HDI: [$1.92k, $2.63k]) |
| 9 | Warns that mediation analysis requires stronger assumptions than total effect estimation | PASS | "Every fragile assumption is flagged" and "sequential ignorability is unverifiable" |
| 10 | Produces or offers to produce a report with DAG, assumptions, effects decomposition, and limitations | PASS | Full 8-step report with DAG (Section 2), assumptions table, effects decomposition (Section 7), limitations (Section 8) |

**Pass rate: 9/10 (90%)**

### Without Skill

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Proposes a DAG with training → confidence → interview → earnings AND training → earnings | PASS | DAG shown: "Training → Confidence → Interview Perf → Earnings" and "Training → Earnings" (direct) |
| 2 | Asks user to confirm DAG, especially unmeasured confounders | FAIL | No user interaction or confirmation requested |
| 3 | Identifies this as mediation analysis requiring a structural causal model | PASS | "Structural Equation Modeling (SEM) with PyMC" — correctly identified |
| 4 | Uses pm.do() or pm.observe() to estimate interventional distributions | FAIL | Uses standard Bayesian SEM (chained regressions), not pm.do()/pm.observe() |
| 5 | Distinguishes between NDE and NIE | PASS | Explicit NDE and NIE computed and reported with dollar amounts |
| 6 | States the assumptions required for mediation | PASS | Lists 5 identification assumptions including sequential ignorability |
| 7 | Asks the user whether these mediation assumptions are defensible | FAIL | No user interaction |
| 8 | Reports both direct and indirect effects with uncertainty | PASS | Reports with 94% HDI for NDE, NIE, ATE, and proportion mediated |
| 9 | Warns that mediation analysis requires stronger assumptions than total effect estimation | PASS | "These are strong assumptions" and "In an observational study, they would require careful justification" |
| 10 | Produces or offers to produce a report with DAG, assumptions, effects decomposition, and limitations | PASS | Has DAG, structural equations, effects decomposition, and key caveats section |

**Pass rate: 6/10 (60%)**

---

## Scenario 6: observational-dag-confounders

### With Skill

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Proposes a DAG with exercise as treatment and blood pressure as outcome, identifying confounders | PASS | Full DAG in Section 2 with formal edge list and justifications |
| 2 | Identifies age, BMI, smoking, stress, diet, and income as potential confounders | PASS | All 6 identified; formal edge list with justifications for each |
| 3 | Warns about collider bias: e.g., if BMI is a mediator | PASS | "diet_quality as confounder (not mediator)" discussed in checkpoint; deliberate non-edges section addresses mediator concerns |
| 4 | Asks the user to confirm the DAG and which variables are confounders vs mediators vs colliders | PASS | Checkpoint explicitly asks "Is the total effect or the direct effect more relevant for your board?" and whether diet is confounder or mediator |
| 5 | Uses DoWhy or explicit backdoor adjustment to identify the causal effect | PASS | Full DoWhy identification with `CausalModel` and `identify_effect()` |
| 6 | States the 'no unmeasured confounders' assumption and asks whether it's defensible | PASS | "This identification rests entirely on the assumption of no unobserved confounders" with checkpoint asking user to confirm |
| 7 | Estimates the effect using a Bayesian model (PyMC, CausalPy IPSW, or Bambi) | PASS | Full PyMC model with nutpie sampler, prior/posterior predictive checks |
| 8 | Runs at least one refutation check | PASS | 3 DoWhy refuters + unobserved confounding sensitivity analysis with tipping point |
| 9 | Adapts the report for the mixed audience | PASS | Separate "For clinicians" and "For statisticians" sections in conclusion |
| 10 | Reports limitations: observational study, unmeasured confounders possible | PASS | 4 threats ranked by severity; medication as unmeasured confounder highlighted |
| 11 | Does not overstate causal claims — uses appropriate hedging language | PASS | "evidence is suggestive of a causal effect" — explicitly hedged due to marginal sensitivity |

**Pass rate: 11/11 (100%)**

### Without Skill

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Proposes a DAG with exercise as treatment and blood pressure as outcome | PASS | DAG drawn with ASCII art showing all variables |
| 2 | Identifies age, BMI, smoking, stress, diet, and income as potential confounders | PASS | All 6 identified; adjustment set explicitly listed |
| 3 | Warns about collider bias: e.g., if BMI is a mediator | PASS | "BMI is a mediator (exercise → BMI → BP), so we should NOT adjust for it" — excellent handling |
| 4 | Asks the user to confirm the DAG and which variables are confounders vs mediators vs colliders | FAIL | No user interaction or confirmation |
| 5 | Uses DoWhy or explicit backdoor adjustment | PASS | Uses "backdoor adjustment criterion" explicitly; identifies minimal sufficient adjustment set |
| 6 | States the 'no unmeasured confounders' assumption and asks whether defensible | FAIL | Mentions unmeasured confounders in caveats but does not explicitly ask user to confirm |
| 7 | Estimates the effect using a Bayesian model | FAIL | Uses statsmodels OLS, sklearn GradientBoostingRegressor, and WLS — no Bayesian model |
| 8 | Runs at least one refutation check | PASS | E-value sensitivity analysis, leave-one-confounder-out, covariate balance diagnostics |
| 9 | Adapts the report for the mixed audience | PASS | Separate "For Statisticians" and "For Clinicians" sections |
| 10 | Reports limitations: observational study, unmeasured confounders possible | PASS | 4 caveats listed including genetics, medication, subgroup effects |
| 11 | Does not overstate causal claims — uses appropriate hedging | PASS | "This is still observational — unmeasured confounders could bias results" |

**Pass rate: 7/11 (64%)**

---

## Overall Summary

| Scenario | With Skill | Without Skill | Delta |
|----------|-----------|--------------|-------|
| did-policy-evaluation | 12/12 (100%) | 6/12 (50%) | +50% |
| did-parallel-trends-violation | 10/10 (100%) | 9/10 (90%) | +10% |
| rdd-scholarship-threshold | 10/10 (100%) | 7/10 (70%) | +30% |
| synthetic-control-poor-donors | 10/10 (100%) | 8/10 (80%) | +20% |
| structural-mediation | 9/10 (90%) | 6/10 (60%) | +30% |
| observational-dag-confounders | 11/11 (100%) | 7/11 (64%) | +36% |
| **Overall** | **62/63 (98.4%)** | **43/63 (68.3%)** | **+30.2%** |

---

## Key Observations

### With Skill Strengths
- Perfect or near-perfect pass rates across all 6 scenarios
- Consistent use of mandatory user checkpoints (DAG confirmation, assumption validation)
- Structured 8-step workflow produces comprehensive reports every time
- CausalPy and PyMC used consistently for Bayesian estimation
- Refutation is systematic and never skipped
- Causal language is carefully calibrated to evidence strength

### With Skill Weakness
- Structural mediation scenario: did not use `pm.do()`/`pm.observe()` for interventional distributions (used Bayesian SEM with chain rule instead — functionally equivalent but does not match the assertion's specific tool requirement)

### Without Skill Strengths
- Generally good at identifying the correct design (DiD, RDD, SC, mediation)
- Parallel trends violation scenario handled surprisingly well (90%) — the prompt's explicit warning about the violation likely helped
- BMI mediator issue caught correctly in observational scenario

### Without Skill Weaknesses
- Never asks the user to confirm assumptions or DAG (consistent FAIL across all scenarios)
- Does not use CausalPy or PyMC — defaults to statsmodels, scipy, or sklearn
- Reports frequentist CIs instead of Bayesian HDIs
- Less structured reports — limitations present but not as systematic
- No mandatory refutation workflow — robustness checks are ad hoc
