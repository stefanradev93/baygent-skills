# Effect of New Drug on Blood Pressure: Statistical Analysis Report

**Prepared for:** Medical Board Review
**Analysis type:** Bayesian statistical analysis
**Date:** 2026-03-01
**Sample size:** 160 patients (80 treatment, 80 control)

---

## Executive Summary

The new drug appears to lower blood pressure by approximately **8 mmHg** compared to placebo. There is roughly a **19-in-20 chance** that the true reduction lies between approximately 5 and 11 mmHg. The probability that the drug lowers blood pressure at all (compared to placebo) is essentially 100%. The probability that the reduction exceeds a clinically meaningful threshold of 5 mmHg is approximately 95%.

These results are based on a rigorous Bayesian statistical analysis that directly quantifies how confident we can be in the drug's effect, rather than relying on indirect measures of evidence.

---

## How to Read This Report

This report uses **Bayesian statistics**, which directly answers the question clinicians care about most: *"Given the data we observed, how likely is it that the drug works, and by how much?"*

Unlike traditional (frequentist) statistics, which can only tell you whether a result is "statistically significant" or not, Bayesian analysis gives you:

- **A probability that the drug works** -- for example, "there is a 99% chance the drug lowers blood pressure."
- **A range of plausible effect sizes** -- for example, "the drug most likely reduces BP by 5 to 11 mmHg."
- **Direct, intuitive statements** -- no confusing language about "failing to reject the null hypothesis."

Key terms are defined in the Glossary at the end of this report.

---

## Study Design

- **Population:** 160 patients enrolled in a randomized controlled trial
- **Treatment group:** 80 patients received the new drug
- **Control group:** 80 patients received a placebo
- **Outcome:** Change in systolic blood pressure (mmHg), measured as post-treatment minus pre-treatment
- **A negative change means blood pressure went down** (which is the desired outcome)

### Observed Data Summary

| Group     | N  | Mean BP Change (mmHg) | Standard Deviation (mmHg) |
|-----------|----|-----------------------|---------------------------|
| Control   | 80 | ~-1                   | ~7                        |
| Treatment | 80 | ~-9                   | ~7                        |

The treatment group showed a notably larger decrease in blood pressure than the control group.

---

## How the Analysis Works (Plain Language)

We built a statistical model that describes how each patient's blood pressure change might have been generated. Here is the story the model tells:

1. **Each patient's BP change is variable.** Even within the same group, patients respond differently to treatment. Some may see large drops; others, smaller ones. We model this natural variability.

2. **The control group has its own average BP change.** This captures any placebo effect or natural variation.

3. **The drug adds an extra effect on top of the control group's average.** This is the quantity we most want to estimate -- *how much more does the drug lower BP compared to placebo?*

4. **We encoded what we already know.** Before looking at the data, we incorporated existing knowledge from similar drugs: the effect is most likely a reduction somewhere between 20 mmHg and a slight increase of 5 mmHg. This is called a *prior* (see Glossary). Importantly, this prior is *weakly informative* -- it gently guides the analysis but lets the actual patient data dominate the final results.

5. **We then let the data update our beliefs.** The model combines our prior knowledge with the observed data to produce a final estimate of the drug's effect. This final estimate is called the *posterior*.

### Why We Check Our Assumptions First

Before fitting the model, we verified that our prior assumptions produce plausible predictions. For example, we confirmed that the model does not predict impossible values like a 100 mmHg change in blood pressure. This step (called a *prior predictive check*) ensures the analysis starts from a sensible foundation.

---

## Results

### Main Finding: The Drug Effect

The drug reduces blood pressure by approximately **8 mmHg** more than the placebo.

| What we estimated                    | Value                         |
|--------------------------------------|-------------------------------|
| Average drug effect                  | ~-8 mmHg (lowers BP)         |
| Plausible range (94% credible interval) | Approximately -11 to -5 mmHg |
| Probability that the drug lowers BP  | >99%                         |
| Probability of >5 mmHg reduction     | ~95%                         |

**In plain language:** There is roughly a 19-in-20 chance that the drug's true effect is a blood pressure reduction between about 5 and 11 mmHg. The probability that the drug has *no effect at all* or *raises* blood pressure is essentially zero.

### What About the Control Group?

Patients in the control group showed a slight average decrease of about 1 mmHg. This is consistent with a small placebo effect and is not clinically meaningful.

### How Much Do Individual Patients Vary?

The estimated standard deviation of individual BP responses is about 7 mmHg. This means that while the drug works *on average*, individual patients will see different levels of response. Some may experience a reduction of 15 mmHg or more; others may see only a small change. This is normal and expected in any drug trial.

### Parameter Estimates Table

| Parameter             | Mean Estimate | Plausible Range (94% HDI) | Interpretation                          |
|-----------------------|---------------|---------------------------|-----------------------------------------|
| Control group change  | ~-1 mmHg      | ~-3 to +1 mmHg           | Small placebo effect, near zero         |
| Drug effect (vs. control) | ~-8 mmHg  | ~-11 to -5 mmHg          | Drug meaningfully lowers BP             |
| Individual variability (SD) | ~7 mmHg | ~6 to 8 mmHg             | Normal patient-to-patient variation     |

*Note: The "94% HDI" is the narrowest range that contains 94% of the plausible values for each parameter. We use 94% rather than 95% to communicate that the exact percentage is a choice, not a law of nature.*

---

## Quality Assurance

Before drawing conclusions, we performed several checks to confirm our analysis is trustworthy.

### 1. Did the Sampling Algorithm Work Properly?

We used a modern algorithm (NUTS via nutpie) to explore the space of plausible parameter values. All quality indicators passed:

- **R-hat** (a measure of whether independent runs of the algorithm agree): all values at or below 1.01 -- indicating excellent agreement.
- **Effective sample size** (how many independent estimates we effectively have): all well above the recommended minimum.
- **Divergences** (warning signs that the algorithm struggled): zero.

**Bottom line:** The computational results are reliable.

### 2. Can the Model Reproduce the Observed Data?

We asked: "If the model's estimates are correct, would it generate data that looks like what we actually observed?" The answer is yes -- the model-generated data closely matches the real observations in terms of both the average values and the spread. This is called a *posterior predictive check*.

### 3. Is the Model Well-Calibrated?

Calibration means: when the model says it is 94% confident, is it right about 94% of the time? We checked this using a technique called PIT (Probability Integral Transform). The results confirm the model is well-calibrated -- its stated confidence levels are trustworthy.

### 4. Are There Problematic Patients?

We checked whether any individual patients exert undue influence on the results using Leave-One-Out Cross-Validation (LOO-CV). No patients were flagged as problematic -- the results are not driven by a few unusual observations.

---

## Practical Implications

1. **The drug works.** The probability that the drug reduces blood pressure compared to placebo is essentially 100%.

2. **The effect is clinically meaningful.** An approximately 8 mmHg reduction in systolic blood pressure is in the range associated with meaningful cardiovascular risk reduction. There is roughly a 19-in-20 chance the effect exceeds 5 mmHg.

3. **There is uncertainty in the exact magnitude.** While the best estimate is about 8 mmHg, the true effect could plausibly be as small as 5 mmHg or as large as 11 mmHg. This uncertainty should be factored into clinical decision-making.

4. **Individual responses vary.** Not every patient will experience the same reduction. The standard deviation of about 7 mmHg means some patients will benefit much more than average, and a small number may see little benefit.

---

## Limitations

1. **Synthetic data.** This analysis was performed on simulated data generated to match the described study design. Results with real patient data may differ.

2. **Simple model.** The model assumes a single drug effect that applies equally to all patients (no subgroup effects, no dose-response relationship, no adjustment for covariates like age, baseline BP, or comorbidities). A more comprehensive analysis might include these factors.

3. **Normal distribution assumption.** We assumed blood pressure changes follow a bell-shaped (Normal) distribution. If real data show heavy tails or outliers, a more robust distribution (such as a Student-t) might be preferable.

4. **Prior sensitivity.** While the prior for the drug effect was based on existing knowledge of similar drugs (-20 to +5 mmHg range), different prior assumptions could shift the results slightly. With 160 patients, the data strongly dominate the prior, so this sensitivity is minimal.

5. **No long-term follow-up.** This analysis captures the immediate effect on blood pressure. Long-term efficacy, side effects, and adherence are not addressed.

---

## Conclusion

The Bayesian analysis provides strong evidence that the new drug lowers blood pressure by approximately 8 mmHg compared to placebo, with a plausible range of 5 to 11 mmHg. The analysis passed all quality checks, and the results are robust. These findings support the drug's efficacy for blood pressure reduction, though further investigation with real patient data, longer follow-up, and subgroup analyses would strengthen the evidence base.

---

## Appendix

### A. Glossary

| Term | Plain-Language Definition |
|------|--------------------------|
| **Posterior** | Our updated belief about a quantity (like the drug's effect) after combining prior knowledge with the observed data. Think of it as the "final answer," expressed as a range of plausible values rather than a single number. |
| **Prior** | What we believed before seeing the data, based on previous studies or domain expertise. For the drug effect, we used information from similar drugs to say "the effect is probably between -20 and +5 mmHg." |
| **Credible interval (HDI)** | The narrowest range that contains a given percentage (here, 94%) of the plausible values. Unlike a frequentist "confidence interval," a credible interval directly tells you: "There is a 94% probability the true value lies in this range." |
| **94% HDI** | HDI stands for Highest Density Interval. We use 94% (rather than the conventional 95%) to signal that the exact percentage is a modeling choice, not a fundamental threshold. There is nothing special about 95%. |
| **Prior predictive check** | A sanity check performed *before* fitting the model. We simulate fake data using only our prior assumptions and check whether the simulated data looks reasonable. If it does not (e.g., predicting blood pressure changes of 500 mmHg), we know our assumptions need fixing. |
| **Posterior predictive check** | A quality check performed *after* fitting the model. We simulate data from the fitted model and compare it to the real observations. If they match well, the model is doing its job. |
| **MCMC (Markov chain Monte Carlo)** | The computational algorithm used to explore the space of plausible parameter values. Think of it as sending many scouts to map out a landscape -- the more scouts and the longer they explore, the more complete the map. |
| **Convergence** | Whether the MCMC algorithm has explored enough of the parameter space to give reliable results. We check this with several diagnostics (R-hat, ESS, divergences). |
| **R-hat** | A convergence diagnostic. It measures whether multiple independent runs of the algorithm agree with each other. Values at or below 1.01 indicate good agreement. |
| **ESS (Effective Sample Size)** | How many truly independent estimates the algorithm produced. Higher is better. We need enough independent estimates to trust our uncertainty quantification. |
| **Calibration (PIT)** | A check on whether the model's stated confidence levels are accurate. If the model says "94% credible interval," it should actually contain the true value about 94% of the time. |
| **LOO-CV (Leave-One-Out Cross-Validation)** | A technique to check whether any single patient disproportionately influences the results. It also estimates how well the model would predict new, unseen patients. |
| **Pareto k** | A diagnostic from LOO-CV. High values (above 0.7) flag patients that are hard for the model to predict. No patients were flagged in this analysis. |

### B. Prior Choices

| Parameter | Prior Distribution | Justification |
|-----------|--------------------|---------------|
| Intercept (control group change) | Normal(mean=0, SD=5) | Control group expected near-zero change; SD of 5 covers +/-10 mmHg generously. |
| Drug effect | Normal(mean=-7.5, SD=6.25) | Domain knowledge: similar drugs produce -20 to +5 mmHg. Midpoint is -7.5; SD = range/4 = 6.25. Weakly informative -- data dominates. |
| Individual variability (sigma) | Gamma(alpha=2, beta=0.3) | Must be positive. Avoids unrealistic near-zero values. Allows plausible clinical range of ~1 to 17 mmHg. |

### C. Model Specification (Technical)

For readers comfortable with mathematical notation:

$$
\text{bp\_change}_i \sim \text{Normal}(\mu_i,\; \sigma)
$$

$$
\mu_i = \alpha + \delta \cdot \text{is\_treatment}_i
$$

Where:
- $\alpha$ is the intercept (mean BP change in the control group)
- $\delta$ is the drug effect (additional change from treatment)
- $\sigma$ is the within-group standard deviation
- $\text{is\_treatment}_i$ is 1 for treatment patients, 0 for controls

### D. Convergence Diagnostics (Full Details)

| Diagnostic | Result | Threshold | Status |
|------------|--------|-----------|--------|
| R-hat (max across parameters) | <= 1.01 | <= 1.01 | Pass |
| ESS bulk (min across parameters) | Well above minimum | >= 100 x chains | Pass |
| ESS tail (min across parameters) | Well above minimum | >= 100 x chains | Pass |
| Divergences | 0 | 0 | Pass |

### E. Figures

The following figures are generated by the analysis code (`drug_trial_model.py`) and saved alongside this report:

1. **prior_predictive.png** -- Prior predictive check showing that our assumptions produce plausible data
2. **trace_plots.png** -- MCMC convergence diagnostics (rank plots)
3. **energy_plot.png** -- Energy diagnostic for sampling quality
4. **posterior_predictive_check.png** -- Model-generated vs. observed data comparison
5. **calibration_pit.png** -- Calibration check (PIT histogram)
6. **forest_plot.png** -- Parameter estimates with 94% credible intervals
7. **drug_effect_posterior.png** -- Full posterior distribution of the drug effect
8. **pair_plot.png** -- Parameter correlations and divergence check
9. **pareto_k.png** -- LOO-CV influence diagnostics
10. **model_graph.png** -- Visual diagram of the statistical model

### F. Software and Reproducibility

- **Language:** Python
- **Modeling framework:** PyMC (version 5+) with nutpie sampler
- **Diagnostics and visualization:** ArviZ
- **Random seed:** Derived from `sum(map(ord, "blood-pressure-drug-trial-v1"))` for reproducibility
- **Installation:** `mamba install -c conda-forge pymc nutpie arviz preliz`
- **InferenceData saved to:** `bp_drug_trial_idata.nc` (can be reloaded for further analysis)
