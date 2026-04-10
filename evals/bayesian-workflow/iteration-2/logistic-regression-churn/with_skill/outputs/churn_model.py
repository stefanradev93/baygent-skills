"""
Bayesian Logistic Regression for Customer Churn Prediction
==========================================================

Generative story:
    Each customer has a latent propensity to churn, driven by their age, tenure
    with the company, monthly spending, and recent support ticket volume. We model
    the log-odds of churning as a linear function of these (standardized) predictors.
    The observed churn outcome is a Bernoulli draw from the resulting probability.

Workflow follows the Bayesian modeling skill:
    1. Formulate generative story
    2. Specify priors with justification
    3. Implement in PyMC 5+
    4. Prior predictive checks
    5. Inference (nutpie)
    6. Convergence diagnostics
    7. Model criticism (PPC, LOO-CV, calibration)
    8. Report results
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import arviz_plots as azp
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

# Reproducible seeds: derived from a descriptive analysis name, never magic numbers.
RANDOM_SEED = sum(map(ord, "churn-logistic-v1"))
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Generate synthetic data
# ---------------------------------------------------------------------------

N = 5000

age = rng.normal(loc=40, scale=12, size=N).clip(18, 80)
tenure_months = rng.exponential(scale=24, size=N).clip(1, 120)
monthly_spend = rng.normal(loc=70, scale=25, size=N).clip(5, 200)
support_tickets = rng.poisson(lam=1.5, size=N)

# True data-generating process (on standardized scale for interpretability):
# Higher support tickets and lower tenure increase churn; age and spend have
# moderate effects.
age_z = (age - age.mean()) / age.std()
tenure_z = (tenure_months - tenure_months.mean()) / tenure_months.std()
spend_z = (monthly_spend - monthly_spend.mean()) / monthly_spend.std()
tickets_z = (support_tickets - support_tickets.mean()) / support_tickets.std()

true_intercept = -1.0  # baseline ~27% churn on inv-logit scale
true_beta_age = 0.15
true_beta_tenure = -0.6  # longer tenure -> less churn
true_beta_spend = -0.3   # higher spend -> less churn
true_beta_tickets = 0.7  # more tickets -> more churn

logit_p = (
    true_intercept
    + true_beta_age * age_z
    + true_beta_tenure * tenure_z
    + true_beta_spend * spend_z
    + true_beta_tickets * tickets_z
)
prob_churn = 1 / (1 + np.exp(-logit_p))
churned = rng.binomial(n=1, p=prob_churn, size=N)

df = pd.DataFrame({
    "age": age,
    "tenure_months": tenure_months,
    "monthly_spend": monthly_spend,
    "support_tickets_last_90d": support_tickets,
    "churned": churned,
})

# Store standardization parameters for back-transformation later
standardization_params = {
    "age": {"mean": age.mean(), "std": age.std()},
    "tenure_months": {"mean": tenure_months.mean(), "std": tenure_months.std()},
    "monthly_spend": {"mean": monthly_spend.mean(), "std": monthly_spend.std()},
    "support_tickets_last_90d": {
        "mean": support_tickets.mean(),
        "std": support_tickets.std(),
    },
}

# Build standardized predictor matrix
predictor_names = ["age", "tenure_months", "monthly_spend", "support_tickets_last_90d"]
X_raw = df[predictor_names].values
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X = (X_raw - X_mean) / X_std  # standardize predictors for shared priors
y = df["churned"].values

print(f"Dataset: {N} customers, churn rate = {y.mean():.1%}")
print(f"Predictors: {predictor_names}")

# ---------------------------------------------------------------------------
# 2. Model specification
# ---------------------------------------------------------------------------

coords = {
    "predictors": predictor_names,
    "obs_id": np.arange(N),
}

with pm.Model(coords=coords) as churn_model:
    # --- Data containers ---
    X_data = pm.Data("X", X, dims=("obs_id", "predictors"))
    y_data = pm.Data("y", y, dims="obs_id")

    # --- Priors ---
    # Intercept: Normal(0, 1.5) -- on logit scale this covers baseline churn
    # rates from ~5% to ~80%, which is a plausible range before seeing data.
    intercept = pm.Normal("intercept", mu=0, sigma=1.5)

    # Coefficients: Normal(0, 1.5) on standardized predictors -- weakly
    # informative. On logit scale, +/- 3 units covers odds ratios from ~0.05
    # to ~20, allowing strong effects but ruling out absurd ones.
    # Reference: Gelman et al. recommend Normal(0, 2.5) as a default for
    # logistic regression on standardized predictors. We use 1.5 because we
    # expect moderate (not extreme) effects in a churn context.
    beta = pm.Normal("beta", mu=0, sigma=1.5, dims="predictors")

    # --- Linear predictor ---
    logit_p = pm.Deterministic(
        "logit_p",
        intercept + pm.math.dot(X_data, beta),
        dims="obs_id",
    )

    # --- Likelihood ---
    # Bernoulli with logit link for binary churn outcome
    pm.Bernoulli("churned", logit_p=logit_p, observed=y_data, dims="obs_id")

    # --- Prior predictive check ---
    prior_pred = pm.sample_prior_predictive(random_seed=rng)

# ---------------------------------------------------------------------------
# 3. Prior predictive assessment
# ---------------------------------------------------------------------------

print("\n--- Prior Predictive Check ---")
prior_churn_rate = prior_pred.prior_predictive["churned"].mean(dim="obs_id")
print(
    f"Prior predictive churn rates: "
    f"mean = {float(prior_churn_rate.mean()):.2f}, "
    f"sd = {float(prior_churn_rate.std()):.2f}"
)
print(
    "Assessment: With Normal(0, 1.5) priors on standardized predictors, "
    "prior predictive churn rates span a wide but plausible range (roughly "
    "5%-95%), which is appropriate -- we let the data determine the actual rate."
)

# Visualize prior predictive
fig, ax = plt.subplots(figsize=(8, 4))
az.plot_ppc(prior_pred, group="prior", num_pp_samples=100, ax=ax)
ax.set_title("Prior Predictive Check")
fig.savefig(OUTPUT_DIR / "prior_predictive_check.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Inference
# ---------------------------------------------------------------------------

with churn_model:
    # Don't hardcode the number of chains -- let PyMC / nutpie pick the best
    # default for the user's platform (usually matches CPU cores).
    idata = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    # Extend with prior predictive samples
    idata.extend(prior_pred)

    # --- Posterior predictive check ---
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))

# --- Save immediately after sampling ---
# Late crashes can destroy valid results. Save to disk before any post-processing.
idata.to_netcdf(str(OUTPUT_DIR / "churn_model_output.nc"))
print("\nInferenceData saved to churn_model_output.nc")

# ---------------------------------------------------------------------------
# 5. Convergence diagnostics
# ---------------------------------------------------------------------------

print("\n--- Convergence Diagnostics ---")

# 5a. Summary table (R-hat + ESS at a glance)
summary = az.summary(idata, var_names=["intercept", "beta"], round_to=3)
print(summary)

# 5b. Check R-hat
rhat_ok = (summary["r_hat"] <= 1.01).all()
print(f"R-hat OK (all <= 1.01): {rhat_ok}")

# 5c. Check ESS (bulk and tail) -- threshold is 100 * number of chains
num_chains = idata.posterior.sizes["chain"]
ess_bulk_ok = (summary["ess_bulk"] >= 100 * num_chains).all()
ess_tail_ok = (summary["ess_tail"] >= 100 * num_chains).all()
print(f"ESS bulk OK (>= {100 * num_chains}): {ess_bulk_ok}")
print(f"ESS tail OK (>= {100 * num_chains}): {ess_tail_ok}")

# 5d. Check divergences
n_div = idata.sample_stats["diverging"].sum().item()
print(f"Divergences: {n_div}")

# 5e. Visual check -- rank plots (preferred over raw trace plots)
axes = az.plot_trace(idata, var_names=["intercept", "beta"], kind="rank_vlines")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "trace_rank_plots.png", dpi=150, bbox_inches="tight")
plt.close()

# 5f. Energy diagnostics
fig, ax = plt.subplots(figsize=(8, 4))
az.plot_energy(idata, ax=ax)
fig.savefig(OUTPUT_DIR / "energy_plot.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Model criticism
# ---------------------------------------------------------------------------

print("\n--- Model Criticism ---")

# 6a. Posterior predictive check
fig, ax = plt.subplots(figsize=(8, 4))
az.plot_ppc(idata, num_pp_samples=100, ax=ax)
ax.set_title("Posterior Predictive Check")
fig.savefig(OUTPUT_DIR / "posterior_predictive_check.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 6b. LOO-CV
# nutpie silently ignores idata_kwargs={"log_likelihood": True}, so compute
# log-likelihood explicitly after sampling.
pm.compute_log_likelihood(idata, model=churn_model)
loo = az.loo(idata, pointwise=True)
print(f"\nLOO-CV results:\n{loo}")

# Pareto k diagnostic
pareto_k = loo.pareto_k.values
n_bad_k = (pareto_k > 0.7).sum()
print(f"Observations with Pareto k > 0.7: {n_bad_k}")

fig, ax = plt.subplots(figsize=(8, 4))
az.plot_khat(loo, ax=ax)
fig.savefig(OUTPUT_DIR / "pareto_k_plot.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 6c. Calibration -- mandatory for every model
# Use ArviZ's plot_ppc_pit which handles binary data correctly out of the box.
fig = azp.plot_ppc_pit(idata)
fig.savefig(OUTPUT_DIR / "calibration_ppc_pit.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# LOO-PIT calibration (more robust, preferred when LOO is available)
fig = azp.plot_ppc_pit(idata, loo_pit=True)
fig.savefig(OUTPUT_DIR / "calibration_loo_pit.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Results summary
# ---------------------------------------------------------------------------

print("\n--- Parameter Estimates (94% HDI) ---")
summary_full = az.summary(
    idata,
    var_names=["intercept", "beta"],
    hdi_prob=0.94,
    round_to=3,
)
print(summary_full)

# Forest plot
fig, ax = plt.subplots(figsize=(8, 5))
az.plot_forest(
    idata,
    var_names=["intercept", "beta"],
    combined=True,
    hdi_prob=0.94,
    ax=ax,
)
ax.set_title("Posterior Estimates (94% HDI)")
ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
fig.savefig(OUTPUT_DIR / "forest_plot.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Pair plot for coefficient correlations
axes = az.plot_pair(idata, var_names=["intercept", "beta"], figsize=(10, 10))
plt.savefig(OUTPUT_DIR / "pair_plot.png", dpi=150, bbox_inches="tight")
plt.close()

# Model graph
graph = pm.model_to_graphviz(churn_model)
graph.render(str(OUTPUT_DIR / "model_graph"), format="png", cleanup=True)

# ---------------------------------------------------------------------------
# 8. Prediction utility
# ---------------------------------------------------------------------------


def predict_churn_probability(
    idata,
    age_val: float,
    tenure_val: float,
    spend_val: float,
    tickets_val: int,
) -> dict:
    """
    Compute posterior predictive churn probability for a single customer.

    Returns a dict with mean, median, and 94% HDI of the churn probability.
    All uncertainty from the posterior is propagated through.
    """
    # Standardize inputs using the same parameters as training
    x_new = np.array([
        (age_val - standardization_params["age"]["mean"])
        / standardization_params["age"]["std"],
        (tenure_val - standardization_params["tenure_months"]["mean"])
        / standardization_params["tenure_months"]["std"],
        (spend_val - standardization_params["monthly_spend"]["mean"])
        / standardization_params["monthly_spend"]["std"],
        (tickets_val - standardization_params["support_tickets_last_90d"]["mean"])
        / standardization_params["support_tickets_last_90d"]["std"],
    ])

    # Extract posterior samples
    intercept_samples = idata.posterior["intercept"].values.flatten()
    beta_samples = idata.posterior["beta"].values.reshape(-1, 4)

    # Compute logit and probability for each posterior draw
    logit_vals = intercept_samples + beta_samples @ x_new
    prob_vals = 1 / (1 + np.exp(-logit_vals))

    hdi = az.hdi(prob_vals, hdi_prob=0.94)

    return {
        "mean": float(prob_vals.mean()),
        "median": float(np.median(prob_vals)),
        "hdi_94_low": float(hdi[0]),
        "hdi_94_high": float(hdi[1]),
    }


# Example prediction
example = predict_churn_probability(
    idata,
    age_val=35,
    tenure_val=6,
    spend_val=50,
    tickets_val=4,
)
print(f"\nExample prediction (age=35, tenure=6mo, spend=$50, tickets=4):")
print(f"  Churn probability: {example['mean']:.1%} (94% HDI: [{example['hdi_94_low']:.1%}, {example['hdi_94_high']:.1%}])")

print("\nAnalysis complete. All outputs saved to:", OUTPUT_DIR)
