# Bayesian Logistic Regression for Customer Churn

## Executive summary

We build a Bayesian logistic regression to predict customer churn probability from four predictors: age, tenure (months), monthly spend, and number of support tickets in the last 90 days. The model uses weakly informative priors on standardized predictors and produces full posterior distributions over churn probabilities, allowing us to quantify uncertainty in every prediction. The analysis follows a complete Bayesian workflow: prior predictive checks, NUTS sampling, convergence diagnostics, posterior predictive checks, LOO-CV, and calibration assessment.

## Data description

- **Source**: Synthetic data generated to mirror plausible customer churn patterns (replace with real data in production).
- **Sample size**: N = 5,000 customers.
- **Outcome**: Binary -- churned (1) or not (0).
- **Predictors**:
  - `age`: Customer age in years (range: 18--70).
  - `tenure_months`: Months since account creation (range: ~1--120, right-skewed).
  - `monthly_spend`: Monthly expenditure in dollars (log-normal, median ~$55).
  - `support_tickets_last_90d`: Count of support tickets in the last 90 days (Poisson distributed).

## Model specification

### Generative story

Each customer has a latent propensity to churn, determined by a linear combination of their characteristics. This propensity is mapped through the logistic (inverse-logit) function to produce a churn probability. The observed churn outcome is then a Bernoulli draw from that probability.

This story captures the intuition that churn is a probabilistic event influenced by measurable customer attributes, while acknowledging irreducible randomness (two identical customers may make different decisions).

### Mathematical notation

```
x_i^* = (x_i - mean(x)) / sd(x)           # standardization

logit(p_i) = alpha + beta' * x_i^*

churned_i ~ Bernoulli(p_i)
```

Where `p_i = logit^{-1}(alpha + beta' * x_i^*)`.

### Prior choices

| Parameter | Prior | Justification |
|-----------|-------|---------------|
| `alpha` (intercept) | Normal(0, 1.5) | On the logit scale, 0 corresponds to 50% churn. sigma=1.5 lets the baseline rate range from roughly 5% to 95%, covering all plausible base rates without strongly favoring any particular value. |
| `beta` (coefficients) | Normal(0, 1.5) each | Weakly informative on standardized predictors. A 1-SD change in a predictor shifts the log-odds by beta. sigma=1.5 concentrates most prior mass on moderate effects (log-odds shifts < 3) while allowing for large effects if the data support them. This follows the standard recommendation for logistic regression from Gelman et al. (2008). |

**Why Normal(0, 1.5) and not Normal(0, 2.5)?**

The skill guide suggests Normal(0, 1.5) specifically for binary outcomes. The logit link amplifies coefficient effects, so priors on the logit scale should be somewhat tighter than for linear regression. A sigma of 2.5 on the logit scale would allow implausibly extreme shifts (e.g., a 1-SD change turning a 5% probability into a 99% probability), which is rarely realistic in customer churn contexts.

**Why standardize predictors?**

Without standardization, the four predictors live on wildly different scales (age in decades, tenure in months, spend in dollars, tickets in single digits). A shared prior like Normal(0, 1.5) would mean very different things for each. Standardization makes the prior appropriately weakly informative for all coefficients simultaneously and also improves the sampling geometry.

### Prior predictive check

The code runs `pm.sample_prior_predictive()` and examines the distribution of prior-implied churn rates. With Normal(0, 1.5) priors:

- The prior predictive churn rate should spread broadly across the [0, 1] interval.
- It should not concentrate on extreme values (near 0 or 1), which would indicate overly diffuse priors on the logit scale.
- It should not be too narrow (e.g., concentrated around 50%), which would be overly informative.

**Decision rule applied**: If more than 10% of prior predictive samples are clearly implausible, we tighten priors. If predictions are extremely narrow, we loosen. If reasonable, we proceed.

## Inference details

- **Sampler**: NUTS via nutpie (faster than the default PyMC sampler).
- **Chains**: 4.
- **Draws**: 2,000 per chain (8,000 total posterior samples).
- **Tuning**: 2,000 per chain.
- **Target accept**: 0.9 (slightly above default; logistic models can benefit from a higher acceptance rate to handle the curvature introduced by the logit link).
- **Random seed**: 42 (for reproducibility of synthetic data and sampling).

## Convergence diagnostics interpretation

The code checks all of the following before interpreting results:

1. **R-hat (split, rank-normalized)**: Must be <= 1.01 for all parameters. Values above this indicate chains have not converged and results should not be trusted.

2. **ESS (bulk and tail)**: Must be >= 100 * number of chains (= 400 for 4 chains). ESS bulk measures reliability of central tendency estimates; ESS tail measures reliability of credible interval endpoints. For a simple logistic regression with 5 parameters on 5,000 observations, we expect ESS to be comfortably high (typically > 2,000).

3. **Divergences**: Must be 0. Even a handful of divergent transitions can bias the posterior. If divergences occur, the first remedy is to increase `target_accept` (up to 0.99). For this simple model (no hierarchical structure, no funnel geometry), divergences would be surprising.

4. **Trace/rank plots**: Rank plots should appear uniform (no spikes or gaps). All chains should be well-mixed and overlapping.

5. **Energy diagnostic**: The marginal energy and energy transition distributions should overlap. A large gap would indicate incomplete exploration of the posterior.

**Expected outcome for this model**: A 5-parameter logistic regression on 5,000 observations is a well-behaved problem. We expect all diagnostics to pass comfortably. If they do not, something unexpected is happening with the data (e.g., complete separation, extreme collinearity) and would need investigation.

## Model criticism interpretation

### Posterior predictive checks

The code generates posterior predictive samples and compares them to the observed data. For a binary outcome, the key check is whether the posterior predictive churn rate matches the observed churn rate and whether the distribution of predicted probabilities has the right shape.

We also produce a targeted PPC by computing the posterior predictive churn rate across all observations and comparing it to the observed rate. The observed rate should fall comfortably within the 94% HDI of the posterior predictive rate.

### LOO-CV

Leave-one-out cross-validation via PSIS-LOO estimates out-of-sample predictive accuracy. Key quantities:

- **ELPD (expected log pointwise predictive density)**: Higher (less negative) is better. This gives the model's estimated predictive performance.
- **p_loo (effective number of parameters)**: Should be close to the actual number of parameters (5). If p_loo >> 5, the model may be misspecified or priors may be too weak.
- **Pareto k values**: Per-observation reliability diagnostic. Observations with k > 0.7 have unreliable LOO estimates and deserve investigation (they may be outliers or observations the model handles poorly).

### Calibration

For binary outcomes, we assess calibration by:

1. **Calibration plot**: Binning predicted probabilities and comparing the mean predicted probability in each bin to the observed churn rate. Points should fall near the diagonal.
2. **Separation plot**: Showing the distribution of predicted probabilities for churned vs. non-churned customers. Good discrimination means these distributions are well-separated.

A well-calibrated model should show predicted probabilities that match observed frequencies. Systematic departures (e.g., always overestimating churn for low-risk customers) would indicate model misspecification.

## Results summary

### What the code produces

1. **Parameter summary table** (`parameter_summary.csv`): Mean, SD, 94% HDI, ESS, and R-hat for the intercept and all four coefficients on the standardized scale.

2. **Back-transformed coefficients**: Coefficients transformed to the original predictor scales for direct interpretability (e.g., "each additional support ticket changes the log-odds by X").

3. **Odds ratios**: Exponentiated standardized coefficients. An odds ratio of 1.5 for a predictor means a 1-SD increase in that predictor multiplies the odds of churn by 1.5.

4. **Probability of direction**: For each predictor, the posterior probability that the effect is positive vs. negative. This replaces the frequentist notion of "statistical significance" with a direct probability statement.

5. **Individual predictions with uncertainty**: For any customer profile, the model produces a full posterior distribution over their churn probability -- not just a point estimate, but a range reflecting our uncertainty.

### Expected findings (based on synthetic data)

Based on the data-generating process:

- **Age**: Weak negative effect (older customers slightly less likely to churn). The 94% HDI for the odds ratio may include 1, indicating uncertainty about whether this effect is meaningful.
- **Tenure**: Strong negative effect (longer-tenured customers much less likely to churn). This should be the strongest protective factor.
- **Monthly spend**: Moderate negative effect (higher spenders somewhat less likely to churn). Engaged customers who spend more are more invested.
- **Support tickets**: Strong positive effect (more tickets = higher churn risk). This should be the strongest risk factor.

### Interpretation guidance

- **Never report point estimates alone**. Always include the 94% HDI. The uncertainty IS the result.
- **Use probability language**: "There is a 94% probability that the effect of tenure on log-odds of churn lies in [a, b]" -- not "the effect is significant."
- **Discuss practical significance**: A coefficient whose HDI excludes zero is not automatically important. Consider the effect size in the context of the business problem. A 0.1% change in churn probability, even if "certain," may not be worth acting on.
- **Uncertainty in predictions**: When predicting churn for an individual customer, report the full posterior distribution. A prediction of "65% churn probability with a 94% HDI of [45%, 82%]" is far more useful than just "65%."

## Limitations

1. **Linear effects on logit scale**: The model assumes each predictor has a linear relationship with the log-odds of churn. Nonlinear effects (e.g., U-shaped relationship with age) would be missed. Consider splines or Gaussian processes for nonlinear effects.

2. **No interactions**: The model assumes predictors act independently. In reality, the effect of support tickets might be stronger for new customers than for long-tenured ones. Adding interaction terms (e.g., `tenure * tickets`) is a natural extension.

3. **No hierarchical structure**: If customers are nested within regions, product lines, or account managers, a hierarchical model with group-level effects would better capture this structure and provide partial pooling.

4. **Synthetic data**: The synthetic data was generated from the same model form we are fitting, which means the model will appear better-calibrated than it would on real data where the true data-generating process is unknown and likely more complex.

5. **No time dynamics**: Churn is treated as a static snapshot. In reality, churn risk evolves over time and a survival/hazard model might be more appropriate for modeling time-to-churn.

6. **Prior sensitivity**: For controversial or high-stakes findings, it is good practice to show how conclusions change under alternative priors (e.g., tighter or wider Normal priors on coefficients). This analysis uses standard weakly informative priors, which should not dominate the posterior with N = 5,000, but a sensitivity check is still worthwhile.

## Files produced

| File | Description |
|------|-------------|
| `churn_model.py` | Complete analysis code, runnable end-to-end |
| `analysis_notes.md` | This file -- approach, prior choices, diagnostics, interpretation |
| `churn_model_idata.nc` | InferenceData (NetCDF) with all posterior, prior predictive, and posterior predictive samples |
| `parameter_summary.csv` | Parameter estimates table |
| `model_graph.png` | PyMC model DAG |
| `prior_predictive_check.png` | Prior predictive distributions |
| `trace_rank_plots.png` | Trace and rank plots for convergence |
| `energy_plot.png` | Energy diagnostic |
| `posterior_predictive_check.png` | Posterior predictive check |
| `pareto_k_plot.png` | LOO-CV Pareto k diagnostic |
| `calibration_plots.png` | Calibration curve and predicted probability distributions |
| `forest_plot.png` | Forest plot of parameter posteriors with 94% HDI |
| `ridgeplot.png` | Ridge plot of coefficient posteriors |
| `pair_plot.png` | Pair plot showing posterior correlations |
| `individual_prediction.png` | Example individual prediction with uncertainty |

## Next steps

1. **Run on real data**: Replace the synthetic data generation block with real data loading. Keep standardization and everything downstream unchanged.
2. **Add interactions**: Test whether `tenure * support_tickets` or `age * monthly_spend` improve predictive performance using LOO-CV comparison.
3. **Consider nonlinearity**: Use splines (e.g., `patsy` B-splines) for predictors where nonlinear effects are plausible.
4. **Hierarchical extension**: If customers belong to groups (regions, product types), add group-level intercepts or slopes.
5. **Model comparison**: Fit alternative models (e.g., with interactions, with StudentT-distributed random effects for robustness) and compare using `az.compare()` on LOO-CV results.
