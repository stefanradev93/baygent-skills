# Model Comparison: Simple vs. Hierarchical House Price Models

## Comparison Methodology

### What we are comparing

Two Bayesian models for predicting house prices:

1. **Simple Linear Regression**: `price ~ Normal(alpha + beta_sqft * sqft + beta_bed * bedrooms, sigma)`
2. **Hierarchical Neighborhood Model**: Same linear predictors, but with neighborhood-level varying intercepts partially pooled toward a global mean. This allows each neighborhood to have its own baseline price level while borrowing strength across neighborhoods.

### Why compare these models?

These represent genuinely different modeling assumptions -- the simple model assumes all neighborhoods share the same intercept, while the hierarchical model allows neighborhood-level variation. This is exactly the kind of comparison LOO-CV is designed for (as opposed to variable selection, where tools like BART or kulprit would be more appropriate).

### Comparison tool: PSIS-LOO-CV

We use **Pareto-Smoothed Importance Sampling Leave-One-Out Cross-Validation** (PSIS-LOO) via ArviZ's `az.compare()`. This is the recommended primary comparison tool for the following reasons:

- It estimates out-of-sample predictive accuracy without refitting the model.
- It provides the **Pareto k diagnostic**, which tells us when to trust (or distrust) the LOO estimate for each observation.
- It is preferred over WAIC because WAIC can silently give unreliable results with no diagnostic warning.

### What PSIS-LOO measures

LOO-CV answers: **"Which model predicts unseen data better?"** Specifically, it estimates the **expected log pointwise predictive density (ELPD)** -- the expected log probability that the model assigns to a new, unseen observation. Higher (less negative) ELPD means better predictive accuracy.

This is a measure of *predictive* performance, not a test of which model is "true." It does NOT establish causation.

---

## Interpretation Guide

### Reading the comparison table

The output of `az.compare()` produces a table with these columns:

| Column | Meaning |
|--------|---------|
| `elpd_loo` | Expected log pointwise predictive density. Higher (less negative) is better. |
| `se` | Standard error of the ELPD estimate. |
| `elpd_diff` | Difference in ELPD from the best model. The best model has `elpd_diff = 0`. |
| `dse` | Standard error of the ELPD *difference*. This is NOT simply `se_model1 - se_model2` -- it accounts for correlation between the pointwise estimates. |
| `weight` | Stacking weight -- the optimal weight for combining model predictions. |
| `warning` | `True` if any observations have high Pareto k values (> 0.7). |
| `p_loo` | Effective number of parameters. If much larger than actual parameter count, suspect model misspecification or weak priors. |

### Decision rules for ELPD differences

These thresholds use the ratio of `|elpd_diff|` to `dse`:

| Ratio | Evidence | Recommendation |
|-------|----------|----------------|
| < 2 | Models are practically indistinguishable | Prefer the simpler model (parsimony), unless domain knowledge strongly favors the complex one. |
| 2 -- 4 | Moderate evidence for the better model | Consider domain knowledge. For house prices, if neighborhoods are known to matter (they almost certainly do), lean toward the hierarchical model. |
| > 4 | Strong evidence for the better model | The better model is clearly preferred on predictive grounds. |

### Pareto k diagnostics

Each observation gets a Pareto k value indicating how influential it is for the LOO estimate:

| Pareto k | Reliability | Action |
|----------|-------------|--------|
| < 0.5 | Reliable | Trust the LOO estimate for this observation. |
| 0.5 -- 0.7 | Marginally reliable | Investigate the flagged observations -- they may be unusual but not necessarily problematic. |
| > 0.7 | Unreliable | The LOO estimate for this observation is not trustworthy. Options: (1) investigate for outliers or data errors, (2) use K-fold CV instead, (3) try moment matching. |

If many observations (say > 5%) have Pareto k > 0.7, the LOO comparison itself is unreliable and you should switch to K-fold CV.

### Stacking weights

When no single model is clearly dominant, **stacking weights** provide the optimal way to combine predictions from multiple models. They minimize the expected log predictive density loss of the combined prediction.

- A weight of 1.0 for one model means it completely dominates.
- Weights of 0.7 / 0.3 mean a 70/30 mix would predict best.
- Stacking weights give a more nuanced picture than a binary "pick one model" decision.

Report stacking weights alongside ELPD differences in all comparison reports.

### Posterior predictive checks (PPC)

Before trusting any comparison, verify that **both** models can reproduce the observed data. Use `az.plot_ppc()` and check:

- Does the posterior predictive distribution envelop the observed data?
- Are the shape, spread, and key features (skewness, multimodality, tails) captured?
- Systematic departures indicate model misspecification -- a model that cannot reproduce the data should not be trusted regardless of its ELPD.

### What LOO-CV does NOT tell you

- **Causation**: A better ELPD does not mean the model's causal story is correct. It only means the model predicts unseen data better.
- **Correctness**: Both models could be wrong. LOO-CV picks the better of the candidates, not the "true" model.
- **Practical significance**: A model might predict marginally better but provide no additional actionable insight. Consider what you need the model *for*.

---

## Specific considerations for this comparison

### Why the hierarchical model might win

- **Neighborhoods matter**: House prices are strongly influenced by location. A hierarchical model captures this structured variation that the simple model ignores.
- **Partial pooling**: Neighborhoods with few observations get estimates shrunk toward the global mean, avoiding overfitting. Neighborhoods with many observations retain their own estimate.
- **Better uncertainty quantification**: The hierarchical model provides neighborhood-specific uncertainty, which is more honest than pretending all neighborhoods are identical.

### Why the simple model might still be competitive

- **Small ELPD differences**: If neighborhood effects are weak relative to sqft and bedrooms, the added complexity may not help much with prediction.
- **Few neighborhoods**: If there are very few neighborhoods, the hierarchical structure adds parameters without much gain.
- **Lots of data per neighborhood**: If every neighborhood has abundant data, the "borrowing strength" advantage of partial pooling is less pronounced.

### When to prefer the hierarchical model even if ELPD is similar

- You need neighborhood-specific predictions or uncertainty estimates.
- You plan to predict for neighborhoods with few historical sales (partial pooling handles this gracefully).
- Domain knowledge strongly supports neighborhood effects on price.
- The residual analysis from the simple model shows clustering by neighborhood.

---

## Reporting Template

Use this template when writing up the comparison results. Fill in the bracketed values from the script output.

```markdown
## Model Comparison

We compared two Bayesian models for house price prediction using
PSIS-LOO cross-validation (Vehtari, Gelman, & Gabry, 2017):

1. **Simple model**: price ~ sqft + bedrooms (Normal likelihood)
2. **Hierarchical model**: price ~ sqft + bedrooms + (1 | neighborhood)
   with neighborhood intercepts partially pooled toward a global mean

Both models converged successfully (all R-hat <= 1.01, ESS bulk and
tail >= [minimum threshold], zero divergences).

### Comparison results

| Model | ELPD (LOO) | SE | Delta ELPD | Delta SE | Weight |
|-------|------------|-----|------------|----------|--------|
| [Best model] | [value] | [value] | 0.0 | -- | [value] |
| [Second model] | [value] | [value] | [value] | [value] | [value] |

[Best model] is [slightly/moderately/strongly] preferred by LOO-CV
(Delta ELPD = [value], [ratio] times the SE of the difference),
suggesting [weak/moderate/strong] evidence for better out-of-sample
predictive accuracy.

[X] observations had Pareto k > 0.7 [or: No observations had
Pareto k > 0.7, indicating reliable LOO estimates].

### Posterior predictive checks

Both models [adequately capture / show systematic departures from]
the distribution of observed house prices. [Describe any specific
misfit: e.g., "The simple model underestimates variance in
high-price neighborhoods" or "Both models capture the overall
distribution well."]

### Interpretation

The hierarchical model accounts for neighborhood-level variation in
baseline house prices through partial pooling. This [does / does not]
meaningfully improve predictive accuracy over the simpler model.

[If hierarchical wins]: The neighborhood structure captures meaningful
price variation not explained by square footage and bedrooms alone.
Neighborhoods with fewer observations benefit from borrowing strength
across the dataset.

[If models are similar]: While the hierarchical model does not
substantially improve overall prediction, it provides neighborhood-
specific estimates with appropriate uncertainty, which may be
valuable for decision-making.

**Important**: This comparison measures predictive accuracy, not
causal effects. The hierarchical model's better fit does not imply
that neighborhood *causes* price differences -- it may proxy for
omitted variables (school quality, amenities, proximity to transit,
etc.).
```

---

## Checklist before reporting

- [ ] Both models have passed convergence diagnostics (R-hat, ESS, divergences).
- [ ] Posterior predictive checks have been run and inspected for both models.
- [ ] LOO-CV comparison table includes ELPD, SE, Delta ELPD, Delta SE, and stacking weights.
- [ ] Pareto k diagnostics have been checked and any warnings are noted.
- [ ] The interpretation discusses what the better model implies substantively (not just "model X has higher ELPD").
- [ ] No causal claims are made based on the comparison.
- [ ] Credible intervals (94% HDI) are reported for key parameters, not just point estimates.
- [ ] Stacking weights are reported alongside ELPD differences.
- [ ] Limitations are discussed (what the comparison does NOT tell us).

---

## References

- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413-1432.
- Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024). Pareto smoothed importance sampling. *Journal of Machine Learning Research*, 25(72), 1-58.
- Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking to average Bayesian predictive distributions. *Bayesian Analysis*, 13(3), 917-1007.
