"""
Bayesian Logistic Regression for Customer Churn Prediction
==========================================================

Generative story:
    Each customer has a latent propensity to churn, driven by their age,
    tenure with the company, monthly spending, and recent support interactions.
    On the logit scale, these predictors combine linearly to form a log-odds
    of churning. The observed binary outcome (churned / not churned) is a
    Bernoulli draw from this probability.

    churn_i ~ Bernoulli(p_i)
    logit(p_i) = alpha + beta_age * age_z_i + beta_tenure * tenure_z_i
                       + beta_spend * spend_z_i + beta_tickets * tickets_z_i

    where *_z denotes standardized predictors (zero mean, unit variance).

Workflow steps:
    1. Generate synthetic data
    2. Standardize predictors
    3. Specify priors with justifications
    4. Prior predictive check
    5. Inference via nutpie
    6. Convergence diagnostics
    7. Posterior predictive checks
    8. Calibration (PIT)
    9. LOO-CV
    10. Save results and produce plots
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import arviz_plots as azp
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Reproducible seed -- derived from analysis name (never magic numbers)
# ---------------------------------------------------------------------------
RANDOM_SEED = sum(map(ord, "churn-logistic-v1"))
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Generate synthetic data
# ---------------------------------------------------------------------------
N = 5000

age = rng.normal(loc=40, scale=12, size=N).clip(18, 80)
tenure_months = rng.exponential(scale=24, size=N).clip(1, 120)
monthly_spend = rng.normal(loc=70, scale=25, size=N).clip(10, 200)
support_tickets = rng.poisson(lam=1.5, size=N)

# True data-generating coefficients (on standardized scale)
TRUE_ALPHA = -1.0        # base rate: ~27 % churn in the population
TRUE_BETA_AGE = -0.3     # older customers slightly less likely to churn
TRUE_BETA_TENURE = -0.6  # longer tenure strongly protects against churn
TRUE_BETA_SPEND = -0.2   # higher spenders slightly less likely to churn
TRUE_BETA_TICKETS = 0.7  # more support tickets strongly increases churn

# Standardize for generating probabilities
age_z = (age - age.mean()) / age.std()
tenure_z = (tenure_months - tenure_months.mean()) / tenure_months.std()
spend_z = (monthly_spend - monthly_spend.mean()) / monthly_spend.std()
tickets_z = (support_tickets - support_tickets.mean()) / support_tickets.std()

logit_p = (
    TRUE_ALPHA
    + TRUE_BETA_AGE * age_z
    + TRUE_BETA_TENURE * tenure_z
    + TRUE_BETA_SPEND * spend_z
    + TRUE_BETA_TICKETS * tickets_z
)
p_churn = 1 / (1 + np.exp(-logit_p))
churned = rng.binomial(n=1, p=p_churn)

df = pd.DataFrame({
    "age": age,
    "tenure_months": tenure_months,
    "monthly_spend": monthly_spend,
    "support_tickets": support_tickets,
    "churned": churned,
})

print(f"Dataset: {N} customers, {churned.sum()} churned ({churned.mean():.1%})")
print(df.describe().round(2))

# ---------------------------------------------------------------------------
# 2. Standardize predictors
# ---------------------------------------------------------------------------
# Store means and sds for back-transformation later
predictor_cols = ["age", "tenure_months", "monthly_spend", "support_tickets"]
means = df[predictor_cols].mean()
sds = df[predictor_cols].std()

df_z = pd.DataFrame({
    "age_z": (df["age"] - means["age"]) / sds["age"],
    "tenure_z": (df["tenure_months"] - means["tenure_months"]) / sds["tenure_months"],
    "spend_z": (df["monthly_spend"] - means["monthly_spend"]) / sds["monthly_spend"],
    "tickets_z": (df["support_tickets"] - means["support_tickets"]) / sds["support_tickets"],
})

# ---------------------------------------------------------------------------
# 3. Model specification
# ---------------------------------------------------------------------------
predictor_names = ["age", "tenure", "spend", "tickets"]
coords = {
    "predictor": predictor_names,
    "obs_id": np.arange(N),
}

X = df_z.values  # (N, 4) matrix of standardized predictors

with pm.Model(coords=coords) as churn_model:
    # --- Data containers ---
    X_data = pm.Data("X", X, dims=("obs_id", "predictor"))
    y_data = pm.Data("y", df["churned"].values, dims="obs_id")

    # --- Priors ---
    # Intercept: Normal(0, 1.5) -- on the logit scale this covers base rates
    # from ~5% to ~80% churn within +/-1 SD, which is a plausible range for
    # customer churn across industries.
    alpha = pm.Normal("alpha", mu=0, sigma=1.5)

    # Coefficients: Normal(0, 1.5) on standardized predictors.
    # On the logit scale, sigma=1.5 means a 1-SD change in a predictor can
    # shift log-odds by roughly +/-3 (within 2 SD of the prior), which maps
    # to moving probability from ~5% to ~95%. This is weakly informative:
    # it rules out implausibly huge effects while allowing substantial ones.
    beta = pm.Normal("beta", mu=0, sigma=1.5, dims="predictor")

    # --- Linear predictor ---
    # Using dot product for clean matrix-vector multiplication
    mu = alpha + pm.math.dot(X_data, beta)

    # --- Likelihood ---
    # Bernoulli with logit link (uses logit_p parameterization for numerical
    # stability -- avoids computing sigmoid then feeding to Bernoulli).
    pm.Bernoulli("churn_obs", logit_p=mu, observed=y_data, dims="obs_id")

    # --- Model graph ---
    print("\nModel graph:")
    print(pm.model_to_graphviz(churn_model))

# ---------------------------------------------------------------------------
# 4. Prior predictive check
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Prior Predictive Check")
print("=" * 60)

with churn_model:
    prior_pred = pm.sample_prior_predictive(random_seed=rng)

# Check the implied churn rates from the prior
prior_churn_rates = prior_pred.prior_predictive["churn_obs"].mean(dim="obs_id")
print(f"\nPrior predictive churn rates:")
print(f"  Mean: {float(prior_churn_rates.mean()):.3f}")
print(f"  Std:  {float(prior_churn_rates.std()):.3f}")
print(f"  5th percentile:  {float(prior_churn_rates.quantile(0.05)):.3f}")
print(f"  95th percentile: {float(prior_churn_rates.quantile(0.95)):.3f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(
    prior_churn_rates.values.flatten(),
    bins=50,
    density=True,
    alpha=0.7,
    color="steelblue",
)
ax.axvline(
    df["churned"].mean(), color="black", linestyle="--", linewidth=2,
    label=f"Observed churn rate = {df['churned'].mean():.2%}",
)
ax.set_xlabel("Churn rate (proportion)")
ax.set_ylabel("Density")
ax.set_title("Prior Predictive: Distribution of Implied Churn Rates")
ax.legend()
fig.tight_layout()
fig.savefig("prior_predictive_churn_rates.png", dpi=150)
plt.close(fig)
print("Saved: prior_predictive_churn_rates.png")

# ArviZ PPC plot for prior
fig_ppc_prior = az.plot_ppc(prior_pred, group="prior", num_pp_samples=100)
plt.gcf().savefig("prior_ppc.png", dpi=150)
plt.close("all")
print("Saved: prior_ppc.png")

# Decision: priors cover a wide but plausible range of churn rates
# (roughly 0% to 100%), with the bulk of mass in reasonable territory.
# The observed rate is well within the prior predictive range. Proceed.

# ---------------------------------------------------------------------------
# 5. Inference
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Inference (nutpie sampler)")
print("=" * 60)

with churn_model:
    # Don't hardcode number of chains -- let nutpie choose the best default.
    idata = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    idata.extend(prior_pred)

# ---------------------------------------------------------------------------
# 6. Convergence diagnostics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Convergence Diagnostics")
print("=" * 60)

# 6a. Summary table
summary = az.summary(idata, round_to=3)
print("\n--- Parameter Summary ---")
print(summary)

# 6b. R-hat check
rhat_ok = (summary["r_hat"] <= 1.01).all()
print(f"\nR-hat OK (all <= 1.01): {rhat_ok}")
print(f"  Max R-hat: {summary['r_hat'].max():.4f}")

# 6c. ESS check
num_chains = idata.posterior.sizes["chain"]
ess_threshold = 100 * num_chains
ess_bulk_ok = (summary["ess_bulk"] >= ess_threshold).all()
ess_tail_ok = (summary["ess_tail"] >= ess_threshold).all()
print(f"ESS bulk OK (all >= {ess_threshold}): {ess_bulk_ok}")
print(f"  Min ESS bulk: {summary['ess_bulk'].min():.0f}")
print(f"ESS tail OK (all >= {ess_threshold}): {ess_tail_ok}")
print(f"  Min ESS tail: {summary['ess_tail'].min():.0f}")

# 6d. Divergences
n_div = int(idata.sample_stats["diverging"].sum().item())
print(f"Divergences: {n_div}")

# 6e. Overall convergence assessment
all_converged = rhat_ok and ess_bulk_ok and ess_tail_ok and (n_div == 0)
print(f"\nOverall convergence: {'PASS' if all_converged else 'FAIL'}")

if not all_converged:
    print("WARNING: Convergence issues detected. Results may be unreliable.")
    print("Consider: increasing draws, reparameterizing, or adjusting target_accept.")

# 6f. Trace / rank plots
axes_trace = az.plot_trace(idata, kind="rank_vlines")
plt.gcf().savefig("trace_rank_plots.png", dpi=150, bbox_inches="tight")
plt.close("all")
print("Saved: trace_rank_plots.png")

# 6g. Energy diagnostic
ax_energy = az.plot_energy(idata)
plt.gcf().savefig("energy_plot.png", dpi=150)
plt.close("all")
print("Saved: energy_plot.png")

# ---------------------------------------------------------------------------
# 7. Posterior predictive check
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: Posterior Predictive Check")
print("=" * 60)

with churn_model:
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))

# Save immediately after all sampling is complete -- late crashes can
# destroy valid MCMC results.
idata.to_netcdf("churn_model_output.nc")
print("Saved InferenceData: churn_model_output.nc")

# PPC plot
fig_ppc = az.plot_ppc(idata, num_pp_samples=200)
plt.gcf().savefig("posterior_ppc.png", dpi=150)
plt.close("all")
print("Saved: posterior_ppc.png")

# Check observed vs predicted churn rates
pp_churn_rates = idata.posterior_predictive["churn_obs"].mean(dim="obs_id")
print(f"\nObserved churn rate: {df['churned'].mean():.4f}")
print(f"Posterior predictive churn rate (mean): {float(pp_churn_rates.mean()):.4f}")
print(f"Posterior predictive churn rate (94% HDI): "
      f"[{float(pp_churn_rates.quantile(0.03)):.4f}, "
      f"{float(pp_churn_rates.quantile(0.97)):.4f}]")

# ---------------------------------------------------------------------------
# 8. Calibration (PIT)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8: Calibration Assessment (PIT)")
print("=" * 60)

# Use ArviZ's plot_ppc_pit which handles binary data correctly
try:
    fig_pit = azp.plot_ppc_pit(idata)
    plt.gcf().savefig("ppc_pit.png", dpi=150)
    plt.close("all")
    print("Saved: ppc_pit.png")
except Exception as e:
    print(f"PPC-PIT plot note: {e}")
    print("Attempting LOO-PIT calibration instead...")

# LOO-PIT calibration (more robust, preferred when LOO is available)
try:
    fig_loo_pit = azp.plot_ppc_pit(idata, loo_pit=True)
    plt.gcf().savefig("loo_pit.png", dpi=150)
    plt.close("all")
    print("Saved: loo_pit.png")
except Exception as e:
    print(f"LOO-PIT note: {e}")
    print("Will rely on LOO-CV diagnostics for calibration assessment.")

# ---------------------------------------------------------------------------
# 9. LOO-CV
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 9: Leave-One-Out Cross-Validation (LOO-CV)")
print("=" * 60)

# nutpie does not auto-store log-likelihood -- compute it explicitly
with churn_model:
    pm.compute_log_likelihood(idata, model=churn_model)

loo = az.loo(idata, pointwise=True)
print(loo)

# Pareto k diagnostics
pareto_k = loo.pareto_k.values
n_bad_05 = (pareto_k > 0.5).sum()
n_bad_07 = (pareto_k > 0.7).sum()
print(f"\nPareto k diagnostics:")
print(f"  k > 0.5 (marginally reliable): {n_bad_05} observations")
print(f"  k > 0.7 (unreliable): {n_bad_07} observations")

if n_bad_07 > 0:
    bad_obs = np.where(pareto_k > 0.7)[0]
    print(f"  Problematic observation indices: {bad_obs[:20]}...")  # Show first 20
    print("  These observations are influential or poorly fit. Investigate them.")

# khat plot
ax_khat = az.plot_khat(loo)
plt.gcf().savefig("pareto_k_plot.png", dpi=150)
plt.close("all")
print("Saved: pareto_k_plot.png")

# ---------------------------------------------------------------------------
# 10. Results and interpretation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 10: Results and Interpretation")
print("=" * 60)

# Forest plot
ax_forest = az.plot_forest(
    idata,
    var_names=["alpha", "beta"],
    combined=True,
    hdi_prob=0.94,
)
plt.gcf().savefig("forest_plot.png", dpi=150, bbox_inches="tight")
plt.close("all")
print("Saved: forest_plot.png")

# Posterior distributions
ax_post = az.plot_posterior(
    idata,
    var_names=["alpha", "beta"],
    hdi_prob=0.94,
)
plt.gcf().savefig("posterior_distributions.png", dpi=150, bbox_inches="tight")
plt.close("all")
print("Saved: posterior_distributions.png")

# Pair plot to inspect posterior correlations
ax_pair = az.plot_pair(
    idata,
    var_names=["alpha", "beta"],
    divergences=True,
    kind="kde",
)
plt.gcf().savefig("pair_plot.png", dpi=150, bbox_inches="tight")
plt.close("all")
print("Saved: pair_plot.png")

# ---------------------------------------------------------------------------
# 11. Predicted churn probabilities with uncertainty
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 11: Churn Probability Predictions with Uncertainty")
print("=" * 60)

# Extract posterior samples for coefficients using xarray
alpha_samples = idata.posterior["alpha"]  # shape: (chain, draw)
beta_samples = idata.posterior["beta"]    # shape: (chain, draw, predictor)

# Compute predicted log-odds for each observation using xarray operations
import xarray as xr

X_xr = xr.DataArray(X, dims=("obs_id", "predictor"), coords={"obs_id": np.arange(N), "predictor": predictor_names})
logit_p_pred = alpha_samples + xr.dot(X_xr, beta_samples, dim="predictor")

# Convert to probability (sigmoid)
p_pred = 1 / (1 + np.exp(-logit_p_pred))

# Summarize per observation: mean, and 94% HDI bounds
p_mean = p_pred.mean(dim=["chain", "draw"]).values
p_hdi_low = p_pred.quantile(0.03, dim=["chain", "draw"]).values
p_hdi_high = p_pred.quantile(0.97, dim=["chain", "draw"]).values

# Show a few example predictions
print("\nExample predictions (first 10 customers):")
print(f"{'Customer':>10} {'Observed':>10} {'P(churn) mean':>14} {'94% HDI':>20}")
for i in range(10):
    print(
        f"{i:>10d} {df['churned'].iloc[i]:>10d} "
        f"{p_mean[i]:>14.3f} "
        f"[{p_hdi_low[i]:.3f}, {p_hdi_high[i]:.3f}]"
    )

# Uncertainty plot: predicted probability vs observed outcome
fig, ax = plt.subplots(figsize=(10, 5))
order = np.argsort(p_mean)
ax.fill_between(
    range(N),
    p_hdi_low[order],
    p_hdi_high[order],
    alpha=0.3,
    color="steelblue",
    label="94% HDI",
)
ax.plot(range(N), p_mean[order], color="steelblue", linewidth=0.5, label="Mean P(churn)")
ax.scatter(
    range(N),
    df["churned"].values[order],
    s=1,
    alpha=0.3,
    color="black",
    label="Observed (0/1)",
)
ax.set_xlabel("Customers (sorted by predicted churn probability)")
ax.set_ylabel("P(churn)")
ax.set_title("Predicted Churn Probabilities with 94% Credible Intervals")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("churn_predictions_uncertainty.png", dpi=150)
plt.close(fig)
print("Saved: churn_predictions_uncertainty.png")

# ---------------------------------------------------------------------------
# 12. Effect sizes on probability scale
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 12: Effect Sizes on Probability Scale")
print("=" * 60)

# For interpretability: what does a 1-SD change in each predictor do to
# churn probability, starting from the baseline (average customer)?
alpha_flat = alpha_samples.values.flatten()
beta_flat = beta_samples.values.reshape(-1, 4)  # (n_samples, 4)

baseline_logit = alpha_flat  # average customer (all z-scores = 0)
baseline_p = 1 / (1 + np.exp(-baseline_logit))

print(f"\nBaseline churn probability (average customer):")
print(f"  Mean: {baseline_p.mean():.3f}")
print(f"  94% HDI: [{np.quantile(baseline_p, 0.03):.3f}, {np.quantile(baseline_p, 0.97):.3f}]")

print(f"\nEffect of +1 SD change in each predictor on churn probability:")
print(f"  (1 SD of age = {sds['age']:.1f} years)")
print(f"  (1 SD of tenure = {sds['tenure_months']:.1f} months)")
print(f"  (1 SD of monthly spend = {sds['monthly_spend']:.1f} dollars)")
print(f"  (1 SD of support tickets = {sds['support_tickets']:.2f} tickets)")

for j, name in enumerate(predictor_names):
    shifted_logit = alpha_flat + beta_flat[:, j]
    shifted_p = 1 / (1 + np.exp(-shifted_logit))
    diff = shifted_p - baseline_p
    print(f"\n  {name}:")
    print(f"    Change in P(churn): {diff.mean():+.3f} "
          f"[{np.quantile(diff, 0.03):+.3f}, {np.quantile(diff, 0.97):+.3f}] (94% HDI)")

# ---------------------------------------------------------------------------
# 13. Model graph visualization
# ---------------------------------------------------------------------------
try:
    graph = pm.model_to_graphviz(churn_model)
    graph.render("model_graph", format="png", cleanup=True)
    print("\nSaved: model_graph.png")
except Exception as e:
    print(f"\nModel graph rendering note: {e}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("WORKFLOW COMPLETE")
print("=" * 60)
print("""
Output files:
  - churn_model_output.nc         : Full InferenceData (reload with az.from_netcdf)
  - prior_predictive_churn_rates.png : Prior predictive check
  - prior_ppc.png                 : Prior PPC (ArviZ)
  - trace_rank_plots.png          : Trace / rank plots
  - energy_plot.png               : Energy diagnostic
  - posterior_ppc.png             : Posterior predictive check
  - ppc_pit.png                   : PPC-PIT calibration
  - loo_pit.png                   : LOO-PIT calibration
  - pareto_k_plot.png             : Pareto k diagnostic
  - forest_plot.png               : Forest plot of coefficients
  - posterior_distributions.png   : Posterior distributions
  - pair_plot.png                 : Pair plot with divergences
  - churn_predictions_uncertainty.png : Predictions with uncertainty
  - model_graph.png               : Model DAG
""")
