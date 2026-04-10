# Bayesian Model Comparison: House Price Models

## Methodology Overview

We are comparing two models for predicting house prices:

| Model | Structure | Parameters |
|-------|-----------|------------|
| **Simple** | `price ~ sqft + bedrooms` | Intercept, beta_sqft, beta_beds, sigma |
| **Hierarchical** | `price ~ sqft + bedrooms + (1 | neighborhood)` | All of the above + neighborhood-level intercepts with shared hyperprior |

The hierarchical model adds partial-pooling intercepts for each neighborhood. This allows each neighborhood to have its own baseline price level while sharing information across neighborhoods (shrinkage toward the grand mean).

---

## Comparison Tools Used

### 1. LOO-CV (Leave-One-Out Cross-Validation via PSIS)

**What it is:** LOO-CV estimates how well each model would predict a single held-out observation, averaged across all observations. ArviZ uses Pareto-Smoothed Importance Sampling (PSIS) to approximate the leave-one-out posterior efficiently without actually refitting the model N times.

**Key metric:** ELPD (Expected Log Pointwise Predictive Density). Higher is better.

**Why we use it:** LOO-CV is the gold standard for Bayesian predictive model comparison. Unlike information criteria that only penalize based on parameter counts, LOO-CV directly estimates predictive performance and naturally accounts for the effective complexity of hierarchical models.

### 2. WAIC (Widely Applicable Information Criterion)

**What it is:** WAIC uses the full posterior to estimate out-of-sample prediction error. It consists of a goodness-of-fit term minus a penalty for effective number of parameters (p_waic).

**Why we include it:** As a cross-check against LOO. In most cases they agree. When they disagree, LOO is more trustworthy.

### 3. Pareto-k Diagnostics

**What they are:** Each observation gets a Pareto-k value that measures how influential it is to the LOO estimate. High k values indicate that removing that observation substantially changes the posterior, making importance sampling unreliable.

**Thresholds:**
- k < 0.5: Reliable
- 0.5 <= k < 0.7: Acceptable, but worth noting
- 0.7 <= k < 1.0: Problematic -- consider moment-matching (`az.loo(..., pointwise=True)`) or refitting without that observation (`az.reloo()`)
- k >= 1.0: Very unreliable -- LOO approximation breaks down for this observation

**Why it matters:** If many observations have high k values, the LOO comparison itself becomes untrustworthy and you should not rely on it for model selection without remediation.

### 4. Posterior Predictive Checks (PPC)

**What they are:** We simulate new datasets from the posterior and overlay them against the observed data. A well-fitting model produces simulated data that looks like the real data.

**What to look for:**
- Do the simulated distributions cover the observed data?
- Are there systematic mismatches (e.g., the model cannot reproduce the tails, or misses multimodality)?
- Does one model's PPC look qualitatively better than the other's?

### 5. LOO-PIT (Probability Integral Transform)

**What it is:** For each observation, we compute the probability that the LOO-predictive distribution produces a value less than or equal to the observed value. If the model is well-calibrated, these probabilities should be uniformly distributed.

**What to look for:**
- **Uniform (flat histogram):** Model is well-calibrated
- **U-shaped:** Model is underdispersed (uncertainty is too narrow)
- **Inverted-U:** Model is overdispersed (uncertainty is too wide)
- **Skewed:** Model predictions are systematically biased

### 6. Pointwise ELPD Differences

**What it is:** For each observation, we compute how much better (or worse) the hierarchical model predicts it compared to the simple model.

**What to look for:**
- Are improvements concentrated in certain neighborhoods (especially small ones where partial pooling helps)?
- Are there observations where the hierarchical model is actually worse?
- Is the improvement broadly distributed or driven by a handful of outliers?

### 7. Stacking Weights

**What they are:** Stacking weights solve an optimization problem: "If I had to combine the predictive distributions of these models to maximize LOO-CV, what weights would I use?" Weights sum to 1.

**Interpretation:**
- Weight ~1.0 for one model: That model dominates
- Roughly equal weights: Both models contribute unique predictive information; consider a model that combines their strengths
- These are NOT posterior probabilities of models being "true" -- they are optimal predictive combination weights

---

## Interpretation Guide

### Step-by-step Decision Process

1. **Check Pareto-k diagnostics first.** If either model has many observations with k > 0.7, the LOO comparison may be unreliable. Address this before trusting the ELPD comparison.

2. **Look at the LOO comparison table.** The models are ranked by ELPD (best first). Look at:
   - `elpd_loo`: The estimated ELPD (higher is better)
   - `d_loo`: The ELPD difference from the best model (0 for the best)
   - `dse`: The standard error of the ELPD difference
   - `weight`: Stacking weight

3. **Assess whether the difference is meaningful.** Compute |ELPD difference| / SE of difference:
   - Ratio < 2: The models predict similarly well. The difference is within noise.
   - Ratio 2-5: Moderate evidence favoring the better model.
   - Ratio > 5: Strong evidence favoring the better model.

4. **Examine PPC plots.** Even if LOO favors one model, check that its posterior predictive checks look reasonable. A model with better LOO but terrible PPC plots may have issues.

5. **Check LOO-PIT calibration.** A model with better LOO but poor calibration (non-uniform LOO-PIT) may be overfitting or misspecified in ways that LOO does not fully capture.

6. **Inspect the pointwise ELPD differences.** Understanding WHERE one model improves over the other provides substantive insight. For the house price problem, we expect the hierarchical model to help most for:
   - Observations in small neighborhoods (where pooling matters most)
   - Neighborhoods with prices far from the overall mean

7. **Consider domain context.** If neighborhood effects are scientifically important (e.g., you are advising buyers about neighborhood premiums), the hierarchical model may be preferred even if the predictive difference is small, because it provides interpretable neighborhood-level estimates with proper uncertainty.

### Common Scenarios for This Problem

**Scenario A: Hierarchical model clearly wins (ELPD ratio > 5)**
- Neighborhoods genuinely differ in baseline prices
- The extra complexity pays for itself in predictive accuracy
- Use the hierarchical model

**Scenario B: Models are similar (ELPD ratio < 2)**
- Neighborhood effects may be small relative to sqft/bedrooms
- OR you have very few neighborhoods and partial pooling does not help much
- Choose based on your goals:
  - Need neighborhood-level estimates? Use hierarchical.
  - Want the simplest adequate model? Use simple.
  - Want best predictions? Consider stacking both.

**Scenario C: Simple model wins**
- This is unusual but can happen if:
  - All neighborhoods have similar prices after controlling for sqft/bedrooms
  - You have many neighborhoods with very few observations each and the hierarchical parameterization struggles
  - The hierarchical model has convergence issues that were not fully resolved
- Double-check the hierarchical model specification and diagnostics

**Scenario D: Pareto-k values are high for one or both models**
- Do not trust the LOO comparison at face value
- Try `az.reloo()` to refit the model leaving out problematic observations
- Consider whether the problematic observations are outliers that suggest a heavier-tailed likelihood (e.g., Student-t instead of Normal)

---

## Reporting Template

Use the following template to report your findings:

---

### Model Comparison Results: House Price Prediction

**Models compared:**
- Model A (Simple): Linear regression with square footage and number of bedrooms
- Model B (Hierarchical): Adds neighborhood-level partial-pooling intercepts

**Predictive comparison (LOO-CV):**

| Model | ELPD | SE | Weight |
|-------|------|----|--------|
| [Best model] | [value] | [value] | [value] |
| [Other model] | [value] | [value] | [value] |

ELPD difference: [value] (SE: [value])

**Pareto-k diagnostic:** [X] of [N] observations had k > 0.7 for [model name]. [State whether this is a concern.]

**Posterior predictive checks:** [Describe qualitative assessment. E.g., "Both models reproduce the overall distribution of prices well, but the hierarchical model better captures the bimodal pattern visible in the data."]

**LOO-PIT calibration:** [Describe. E.g., "The simple model shows a slight U-shape indicating underdispersion, while the hierarchical model is approximately uniform."]

**Key finding from pointwise comparison:** [E.g., "The hierarchical model improves predictions primarily for homes in the downtown and suburban neighborhoods, which have the most distinctive price levels."]

**Recommendation:** Based on [LOO-CV / domain considerations / both], we recommend the **[chosen model]** because [reason].

[If the difference is small]: The predictive difference between models is small (|diff|/SE = [ratio] < 2). We recommend the [hierarchical/simple] model because [interpretability / neighborhood estimates are needed / parsimony].

[If the difference is large]: The [winning model] provides meaningfully better out-of-sample predictions (|diff|/SE = [ratio]). [Additional context about why.]

---

## Requirements and Assumptions

For the comparison code to work, both `InferenceData` objects must contain:

1. **`posterior`** -- from `pm.sample()`. Contains the MCMC draws.
2. **`log_likelihood`** -- required for LOO and WAIC. Include `idata_kwargs={"log_likelihood": True}` in `pm.sample()`, or call `pm.compute_log_likelihood(idata)` after sampling.
3. **`posterior_predictive`** -- from `pm.sample_posterior_predictive()`. Required for PPC and LOO-PIT plots.
4. **`observed_data`** -- automatically included by PyMC when you sample.

Both models must have converged (R-hat < 1.01, adequate ESS, no divergences). The comparison code does not check convergence -- that should be verified separately before model comparison.

---

## References

- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413-1432.
- Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking to average Bayesian predictive distributions. *Bayesian Analysis*, 13(3), 917-1007.
- Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A. (2019). Visualization in Bayesian workflow. *Journal of the Royal Statistical Society: Series A*, 182(2), 389-402.
- ArviZ documentation: https://python.arviz.org/en/stable/
