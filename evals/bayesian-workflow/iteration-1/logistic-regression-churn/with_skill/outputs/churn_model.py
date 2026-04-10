"""
Bayesian Logistic Regression for Customer Churn Prediction
==========================================================

Full Bayesian workflow:
1. Formulate the generative story
2. Specify priors (with justifications)
3. Implement in PyMC 5+
4. Prior predictive checks
5. Inference (NUTS via nutpie)
6. Convergence diagnostics
7. Model criticism (PPC, LOO-CV, calibration)
8. Report results with uncertainty

Requires: pymc >= 5.0, arviz >= 0.15, nutpie, numpy, pandas,
          matplotlib, scikit-learn (for standardization)
"""

import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# Reproducibility
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. DATA GENERATION (synthetic -- replace with real data in production)
# ---------------------------------------------------------------------------
# Generative story for synthetic data:
#   - Age: uniformly distributed 18-70
#   - Tenure: exponentially distributed (many short tenures, fewer long)
#   - Monthly spend: log-normal (right-skewed positive values)
#   - Support tickets: Poisson (count data, higher rates -> more churn)
#   - Churn probability: logistic function of a linear combination

N = 5000

age = rng.uniform(18, 70, size=N)
tenure_months = rng.exponential(scale=24, size=N).clip(1, 120)
monthly_spend = rng.lognormal(mean=4.0, sigma=0.5, size=N)  # median ~$55
support_tickets_last_90d = rng.poisson(lam=1.5, size=N)

# True data-generating coefficients (on standardized scale)
# Higher age -> slightly less churn
# Longer tenure -> much less churn (loyal customers stay)
# Higher spend -> slightly less churn (engaged customers)
# More support tickets -> much more churn (frustrated customers)
TRUE_INTERCEPT = -0.8  # baseline: ~30% churn rate at average predictor values
TRUE_BETA_AGE = -0.2
TRUE_BETA_TENURE = -0.7
TRUE_BETA_SPEND = -0.3
TRUE_BETA_TICKETS = 0.6

# Standardize for data generation (same transform we'll use in modeling)
scaler = StandardScaler()
X_raw = np.column_stack([age, tenure_months, monthly_spend, support_tickets_last_90d])
X_std = scaler.fit_transform(X_raw)

logit_p = (
    TRUE_INTERCEPT
    + TRUE_BETA_AGE * X_std[:, 0]
    + TRUE_BETA_TENURE * X_std[:, 1]
    + TRUE_BETA_SPEND * X_std[:, 2]
    + TRUE_BETA_TICKETS * X_std[:, 3]
)
churn_prob = 1 / (1 + np.exp(-logit_p))
churned = rng.binomial(n=1, p=churn_prob, size=N)

# Assemble dataframe
df = pd.DataFrame(
    {
        "age": age,
        "tenure_months": tenure_months,
        "monthly_spend": monthly_spend,
        "support_tickets_last_90d": support_tickets_last_90d,
        "churned": churned,
    }
)

print("=== Data Summary ===")
print(df.describe().round(2))
print(f"\nChurn rate: {df['churned'].mean():.1%}")
print(f"N = {len(df)}")

# ---------------------------------------------------------------------------
# 2. DATA PREPARATION
# ---------------------------------------------------------------------------
# Standardize predictors: makes priors comparable across predictors and
# improves sampling geometry. We store the scaler to back-transform later.

predictor_names = ["age", "tenure_months", "monthly_spend", "support_tickets_last_90d"]
scaler = StandardScaler()
X_std = scaler.fit_transform(df[predictor_names].values)

# Store standardization parameters for later interpretation
predictor_means = scaler.mean_
predictor_sds = scaler.scale_

print("\n=== Standardization Parameters ===")
for name, m, s in zip(predictor_names, predictor_means, predictor_sds):
    print(f"  {name}: mean={m:.2f}, sd={s:.2f}")

# Outcome
y = df["churned"].values

# ---------------------------------------------------------------------------
# 3. MODEL SPECIFICATION
# ---------------------------------------------------------------------------
# Generative story:
#   Each customer i has a latent propensity to churn, driven by a linear
#   combination of their (standardized) age, tenure, spending, and support
#   ticket history. This propensity is mapped to a churn probability via
#   the logistic (inverse-logit) link function. The observed churn outcome
#   is then a Bernoulli draw from this probability.
#
# Mathematical notation:
#   p_i = logit^{-1}(alpha + beta' * x_i)
#   churned_i ~ Bernoulli(p_i)

coords = {
    "predictors": predictor_names,
    "obs_id": np.arange(N),
}

with pm.Model(coords=coords) as churn_model:

    # --- Data containers ---
    X_data = pm.Data("X", X_std, dims=("obs_id", "predictors"))
    y_data = pm.Data("y", y, dims="obs_id")

    # --- Priors ---
    # Intercept: Normal(0, 1.5)
    # On the logit scale, 0 corresponds to 50% churn. sigma=1.5 gives
    # the intercept enough room to place the baseline churn rate anywhere
    # from ~5% to ~95%, which is weakly informative for a binary outcome.
    alpha = pm.Normal("alpha", mu=0, sigma=1.5)

    # Coefficients: Normal(0, 1.5) on standardized predictors
    # Following the skill guide recommendation for binary outcomes.
    # On the logit scale, a 1-SD change in a predictor shifts the log-odds
    # by beta. sigma=1.5 allows effect sizes up to ~3 on the logit scale
    # (which would be very large effects), while concentrating mass on
    # moderate effects. This is a standard weakly informative prior for
    # logistic regression on standardized predictors (see Gelman et al. 2008).
    beta = pm.Normal("beta", mu=0, sigma=1.5, dims="predictors")

    # --- Linear predictor and link function ---
    logit_p = pm.math.dot(X_data, beta) + alpha

    # --- Likelihood ---
    # Bernoulli with logit link: canonical choice for binary outcomes.
    # Using logit_p directly avoids numerical issues from computing
    # sigmoid separately.
    pm.Bernoulli("churned", logit_p=logit_p, observed=y_data, dims="obs_id")

# Visualize the model graph
print("\n=== Model Graph ===")
print(churn_model.point_logps())
try:
    graph = pm.model_to_graphviz(churn_model)
    graph.render("outputs/model_graph", format="png", cleanup=True)
    print("Model graph saved to outputs/model_graph.png")
except Exception:
    print("graphviz not available; skipping model graph rendering.")

# ---------------------------------------------------------------------------
# 4. PRIOR PREDICTIVE CHECK
# ---------------------------------------------------------------------------
print("\n=== Prior Predictive Check ===")

with churn_model:
    prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=RANDOM_SEED)

# Check: what churn rates does the prior imply?
prior_churn_rates = prior_pred.prior_predictive["churned"].mean(dim="obs_id")
print(f"Prior predictive churn rate: "
      f"mean={float(prior_churn_rates.mean()):.2f}, "
      f"sd={float(prior_churn_rates.std()):.2f}")
print(f"Prior predictive churn rate 94% HDI: "
      f"{az.hdi(prior_churn_rates.values.flatten(), hdi_prob=0.94)}")

# Visualize prior predictive distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Prior predictive churn rate distribution
axes[0].hist(prior_churn_rates.values.flatten(), bins=50, density=True, alpha=0.7)
axes[0].axvline(y.mean(), color="red", linestyle="--", label=f"Observed rate: {y.mean():.2f}")
axes[0].set_xlabel("Predicted churn rate")
axes[0].set_ylabel("Density")
axes[0].set_title("Prior Predictive: Churn Rate Distribution")
axes[0].legend()

# Prior distributions of coefficients
prior_beta = prior_pred.prior["beta"].values.reshape(-1, len(predictor_names))
for j, name in enumerate(predictor_names):
    axes[1].hist(prior_beta[:, j], bins=50, alpha=0.5, density=True, label=name)
axes[1].set_xlabel("Coefficient value (logit scale)")
axes[1].set_ylabel("Density")
axes[1].set_title("Prior Distributions of Coefficients")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/prior_predictive_check.png", dpi=150, bbox_inches="tight")
plt.show()
print("Prior predictive check figure saved.")

# Decision: with Normal(0, 1.5) priors on standardized predictors, the prior
# predictive churn rate should spread across the full [0, 1] range without
# concentrating on extremes. This is the desired behavior for a weakly
# informative prior on a logistic regression.

# ---------------------------------------------------------------------------
# 5. INFERENCE
# ---------------------------------------------------------------------------
print("\n=== Running Inference ===")

with churn_model:
    idata = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        nuts_sampler="nutpie",  # faster sampling via nutpie
        random_seed=RANDOM_SEED,
        target_accept=0.9,  # slightly above default for logistic models
    )
    # Extend with prior predictive samples
    idata.extend(prior_pred)

print("Sampling complete.")

# ---------------------------------------------------------------------------
# 6. CONVERGENCE DIAGNOSTICS
# ---------------------------------------------------------------------------
print("\n=== Convergence Diagnostics ===")

# 6a. Summary table with R-hat and ESS
summary = az.summary(idata, var_names=["alpha", "beta"], round_to=3)
print(summary)

# 6b. Check R-hat
rhat_ok = (summary["r_hat"] <= 1.01).all()
print(f"\nR-hat OK (all <= 1.01): {rhat_ok}")

# 6c. Check ESS (bulk and tail) -- threshold: 100 * num_chains
num_chains = len(idata.posterior.chain)
ess_threshold = 100 * num_chains
ess_bulk_ok = (summary["ess_bulk"] >= ess_threshold).all()
ess_tail_ok = (summary["ess_tail"] >= ess_threshold).all()
print(f"ESS bulk OK (all >= {ess_threshold}): {ess_bulk_ok}")
print(f"ESS tail OK (all >= {ess_threshold}): {ess_tail_ok}")
print(f"ESS bulk min: {summary['ess_bulk'].min():.0f}")
print(f"ESS tail min: {summary['ess_tail'].min():.0f}")

# 6d. Check divergences
n_div = int(idata.sample_stats["diverging"].sum())
print(f"Divergences: {n_div}")
divergences_ok = n_div == 0
print(f"Divergences OK: {divergences_ok}")

# 6e. Overall diagnostic verdict
all_diagnostics_ok = rhat_ok and ess_bulk_ok and ess_tail_ok and divergences_ok
print(f"\n>>> ALL DIAGNOSTICS PASSED: {all_diagnostics_ok} <<<")

if not all_diagnostics_ok:
    print("WARNING: Some diagnostics failed. Results may be unreliable.")
    print("Do NOT interpret results until diagnostics pass.")

# 6f. Trace and rank plots
fig = az.plot_trace(idata, var_names=["alpha", "beta"], kind="rank_vlines")
plt.tight_layout()
plt.savefig("outputs/trace_rank_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Trace/rank plots saved.")

# 6g. Energy diagnostic
fig = az.plot_energy(idata)
plt.savefig("outputs/energy_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Energy plot saved.")

# ---------------------------------------------------------------------------
# 7. MODEL CRITICISM
# ---------------------------------------------------------------------------
print("\n=== Model Criticism ===")

# 7a. Posterior predictive check
with churn_model:
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=RANDOM_SEED))

# PPC: compare observed churn rate to posterior predictive churn rates
pp_churn = idata.posterior_predictive["churned"]
pp_churn_rates = pp_churn.mean(dim="obs_id")

fig, ax = plt.subplots(figsize=(8, 4))
az.plot_ppc(idata, num_pp_samples=100, ax=ax)
ax.set_title("Posterior Predictive Check")
plt.tight_layout()
plt.savefig("outputs/posterior_predictive_check.png", dpi=150, bbox_inches="tight")
plt.show()
print("Posterior predictive check figure saved.")

# Observed vs posterior predictive churn rate
observed_rate = y.mean()
pp_rates_flat = pp_churn_rates.values.flatten()
print(f"\nObserved churn rate: {observed_rate:.3f}")
print(f"Posterior predictive churn rate: "
      f"mean={pp_rates_flat.mean():.3f}, "
      f"94% HDI={az.hdi(pp_rates_flat, hdi_prob=0.94)}")

# 7b. LOO-CV (Leave-One-Out Cross-Validation)
print("\n--- LOO-CV ---")
loo = az.loo(idata, pointwise=True)
print(loo)

# Check Pareto k diagnostics
pareto_k = loo.pareto_k.values
n_bad_k = (pareto_k > 0.7).sum()
n_marginal_k = ((pareto_k > 0.5) & (pareto_k <= 0.7)).sum()
print(f"\nPareto k diagnostics:")
print(f"  Observations with k > 0.7 (unreliable): {n_bad_k}")
print(f"  Observations with 0.5 < k <= 0.7 (marginal): {n_marginal_k}")
print(f"  Observations with k <= 0.5 (reliable): {(pareto_k <= 0.5).sum()}")

fig = az.plot_khat(loo)
plt.savefig("outputs/pareto_k_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Pareto k plot saved.")

if n_bad_k > 0:
    bad_obs_indices = np.where(pareto_k > 0.7)[0]
    print(f"\nWARNING: {n_bad_k} observations with high Pareto k.")
    print(f"Indices: {bad_obs_indices[:20]}...")  # show first 20
    print("Consider investigating these observations or using K-fold CV.")

# 7c. Calibration: PIT histogram via separation plots for binary data
# For binary outcomes, a separation plot is more informative than PIT
# We compute predicted probabilities and assess calibration

# Extract posterior mean predicted probabilities
alpha_post = idata.posterior["alpha"].values  # (chains, draws)
beta_post = idata.posterior["beta"].values  # (chains, draws, predictors)

# Reshape for broadcasting: (chains * draws, 1) and (chains * draws, predictors)
n_chains, n_draws = alpha_post.shape
alpha_flat = alpha_post.flatten()
beta_flat = beta_post.reshape(n_chains * n_draws, len(predictor_names))

# Compute predicted probabilities for each posterior draw
# logit_p = alpha + X @ beta.T  -> shape (n_obs, n_posterior_samples)
logit_p_post = X_std @ beta_flat.T + alpha_flat[np.newaxis, :]
p_post = 1 / (1 + np.exp(-logit_p_post))  # (n_obs, n_posterior_samples)

# Mean predicted probability per observation
p_mean = p_post.mean(axis=1)

# Calibration plot: bin predictions and compare to observed rates
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
observed_rates = np.zeros(n_bins)
predicted_rates = np.zeros(n_bins)
bin_counts = np.zeros(n_bins)

for i in range(n_bins):
    mask = (p_mean >= bin_edges[i]) & (p_mean < bin_edges[i + 1])
    if mask.sum() > 0:
        observed_rates[i] = y[mask].mean()
        predicted_rates[i] = p_mean[mask].mean()
        bin_counts[i] = mask.sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Calibration curve
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
axes[0].scatter(predicted_rates, observed_rates, s=bin_counts / 5, alpha=0.7,
                label="Model calibration")
axes[0].set_xlabel("Mean predicted probability")
axes[0].set_ylabel("Observed churn rate")
axes[0].set_title("Calibration Plot")
axes[0].legend()
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

# Distribution of predicted probabilities by actual outcome
axes[1].hist(p_mean[y == 0], bins=50, alpha=0.5, density=True, label="Not churned")
axes[1].hist(p_mean[y == 1], bins=50, alpha=0.5, density=True, label="Churned")
axes[1].set_xlabel("Predicted churn probability")
axes[1].set_ylabel("Density")
axes[1].set_title("Predicted Probabilities by Outcome")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/calibration_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Calibration plots saved.")

# ---------------------------------------------------------------------------
# 8. RESULTS AND INTERPRETATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("=== RESULTS ===")
print("=" * 60)

# 8a. Parameter estimates with 94% HDI
print("\n--- Parameter Estimates (logit scale, standardized predictors) ---")
summary_full = az.summary(
    idata,
    var_names=["alpha", "beta"],
    hdi_prob=0.94,
    round_to=3,
)
print(summary_full)

# 8b. Forest plot
fig = az.plot_forest(
    idata,
    var_names=["alpha", "beta"],
    combined=True,
    hdi_prob=0.94,
    figsize=(8, 4),
)
plt.title("Posterior Distributions (94% HDI)")
plt.tight_layout()
plt.savefig("outputs/forest_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Forest plot saved.")

# 8c. Posterior distributions (ridgeplot)
fig = az.plot_forest(
    idata,
    var_names=["beta"],
    kind="ridgeplot",
    combined=True,
    hdi_prob=0.94,
    figsize=(8, 5),
)
plt.title("Posterior Distributions of Coefficients")
plt.tight_layout()
plt.savefig("outputs/ridgeplot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Ridgeplot saved.")

# 8d. Pair plot to check posterior correlations
fig = az.plot_pair(
    idata,
    var_names=["alpha", "beta"],
    divergences=True,
    figsize=(10, 10),
)
plt.tight_layout()
plt.savefig("outputs/pair_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Pair plot saved.")

# 8e. Back-transform to original scale for interpretability
print("\n--- Coefficients on Original (unstandardized) Scale ---")
alpha_samples = idata.posterior["alpha"].values.flatten()
beta_samples = idata.posterior["beta"].values.reshape(-1, len(predictor_names))

# beta_original = beta_standardized / sd_x
# alpha_original = alpha_standardized - sum(beta_standardized * mean_x / sd_x)
beta_original = beta_samples / predictor_sds[np.newaxis, :]
alpha_original = alpha_samples - (beta_samples * predictor_means[np.newaxis, :] / predictor_sds[np.newaxis, :]).sum(axis=1)

print(f"\nIntercept (original scale):")
print(f"  Mean: {alpha_original.mean():.3f}")
print(f"  94% HDI: {az.hdi(alpha_original, hdi_prob=0.94)}")

for j, name in enumerate(predictor_names):
    print(f"\n{name} (original scale):")
    print(f"  Mean: {beta_original[:, j].mean():.4f}")
    print(f"  94% HDI: {az.hdi(beta_original[:, j], hdi_prob=0.94)}")

# 8f. Odds ratios (exponentiated coefficients on standardized scale)
print("\n--- Odds Ratios (per 1-SD change in predictor) ---")
for j, name in enumerate(predictor_names):
    or_samples = np.exp(beta_samples[:, j])
    print(f"{name}:")
    print(f"  Median OR: {np.median(or_samples):.3f}")
    print(f"  94% HDI: {az.hdi(or_samples, hdi_prob=0.94)}")

# 8g. Probability of direction (is the effect positive or negative?)
print("\n--- Probability of Direction ---")
for j, name in enumerate(predictor_names):
    prob_positive = (beta_samples[:, j] > 0).mean()
    prob_negative = 1 - prob_positive
    direction = "positive" if prob_positive > 0.5 else "negative"
    confidence = max(prob_positive, prob_negative)
    print(f"{name}: P(beta > 0) = {prob_positive:.3f}, "
          f"P(beta < 0) = {prob_negative:.3f} "
          f"[{confidence:.1%} confident the effect is {direction}]")

# ---------------------------------------------------------------------------
# 9. INDIVIDUAL PREDICTIONS WITH UNCERTAINTY
# ---------------------------------------------------------------------------
print("\n=== Individual Prediction Example ===")

# Example: predict churn for a specific customer profile
example_customer = {
    "age": 35,
    "tenure_months": 6,
    "monthly_spend": 45.0,
    "support_tickets_last_90d": 4,
}

# Standardize the example customer
x_example = np.array([[
    example_customer["age"],
    example_customer["tenure_months"],
    example_customer["monthly_spend"],
    example_customer["support_tickets_last_90d"],
]])
x_example_std = (x_example - predictor_means) / predictor_sds

# Compute posterior predictive probability
logit_example = x_example_std @ beta_samples.T + alpha_samples
p_example = 1 / (1 + np.exp(-logit_example.flatten()))

print(f"Customer profile: {example_customer}")
print(f"Predicted churn probability:")
print(f"  Mean: {p_example.mean():.3f}")
print(f"  Median: {np.median(p_example):.3f}")
print(f"  94% HDI: {az.hdi(p_example, hdi_prob=0.94)}")
print(f"  Std Dev: {p_example.std():.3f}")

# Visualize the uncertainty in this prediction
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(p_example, bins=80, density=True, alpha=0.7, color="steelblue")
hdi = az.hdi(p_example, hdi_prob=0.94)
ax.axvline(p_example.mean(), color="red", linestyle="-", label=f"Mean: {p_example.mean():.3f}")
ax.axvspan(hdi[0], hdi[1], alpha=0.2, color="red", label=f"94% HDI: [{hdi[0]:.3f}, {hdi[1]:.3f}]")
ax.set_xlabel("Churn Probability")
ax.set_ylabel("Density")
ax.set_title(f"Posterior Predictive Churn Probability\n"
             f"(Age={example_customer['age']}, Tenure={example_customer['tenure_months']}mo, "
             f"Spend=${example_customer['monthly_spend']}, Tickets={example_customer['support_tickets_last_90d']})")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/individual_prediction.png", dpi=150, bbox_inches="tight")
plt.show()
print("Individual prediction plot saved.")

# ---------------------------------------------------------------------------
# 10. SAVE INFERENCE DATA
# ---------------------------------------------------------------------------
idata.to_netcdf("outputs/churn_model_idata.nc")
print("\nInference data saved to outputs/churn_model_idata.nc")

# Save the summary table
summary_full.to_csv("outputs/parameter_summary.csv")
print("Parameter summary saved to outputs/parameter_summary.csv")

print("\n=== Analysis Complete ===")
print("All outputs saved to outputs/ directory.")
print("Review the analysis_notes.md file for detailed interpretation.")
