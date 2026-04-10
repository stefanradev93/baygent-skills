"""
Model Comparison: Simple Linear Regression vs. Hierarchical Model for House Prices
====================================================================================

This script builds two models for predicting house prices:
  1. Simple linear regression (square footage + bedrooms)
  2. Hierarchical model adding neighborhood as a varying (random) intercept

Both models are fit, diagnosed, criticized, and compared using LOO-CV, Pareto k
diagnostics, and stacking weights. Results are saved as InferenceData (.nc) files.

See companion file `comparison_notes.md` for a detailed interpretation of all results.
"""

import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Reproducible seed derived from analysis name (skill rule: no magic numbers)
# ---------------------------------------------------------------------------
RANDOM_SEED = sum(map(ord, "house-price-comparison"))
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Generate synthetic data
# ---------------------------------------------------------------------------
n_obs = 400
n_neighborhoods = 8
neighborhood_names = [f"neighborhood_{i}" for i in range(n_neighborhoods)]

# Assign observations to neighborhoods (unbalanced on purpose)
neighborhood_sizes = rng.multinomial(n_obs, np.ones(n_neighborhoods) / n_neighborhoods)
neighborhood_idx = np.concatenate(
    [np.full(size, i) for i, size in enumerate(neighborhood_sizes)]
)
rng.shuffle(neighborhood_idx)

# True parameters
true_intercept = 200_000  # base price in dollars
true_beta_sqft = 150  # dollars per sqft
true_beta_bedrooms = 10_000  # dollars per bedroom
true_sigma = 25_000  # observation noise

# True neighborhood offsets (the hierarchical effect we want to detect)
true_neigh_sigma = 30_000
true_neigh_offsets = rng.normal(0, true_neigh_sigma, size=n_neighborhoods)

# Generate predictors
sqft = rng.normal(1500, 400, size=n_obs).clip(600, 4000)
bedrooms = rng.choice([1, 2, 3, 4, 5], size=n_obs, p=[0.05, 0.2, 0.4, 0.25, 0.1])

# Generate prices (with neighborhood effect)
mu_true = (
    true_intercept
    + true_beta_sqft * sqft
    + true_beta_bedrooms * bedrooms
    + true_neigh_offsets[neighborhood_idx]
)
price = rng.normal(mu_true, true_sigma)

df = pd.DataFrame(
    {
        "price": price,
        "sqft": sqft,
        "bedrooms": bedrooms,
        "neighborhood": [neighborhood_names[i] for i in neighborhood_idx],
        "neighborhood_idx": neighborhood_idx,
    }
)

print("=== Data summary ===")
print(df.describe())
print(f"\nNeighborhood counts:\n{df['neighborhood'].value_counts().sort_index()}")

# ---------------------------------------------------------------------------
# 2. Standardize continuous predictors (skill rule: always standardize)
# ---------------------------------------------------------------------------
sqft_mean, sqft_std = df["sqft"].mean(), df["sqft"].std()
bedrooms_mean, bedrooms_std = df["bedrooms"].mean(), df["bedrooms"].std()
price_mean, price_std = df["price"].mean(), df["price"].std()

sqft_z = ((df["sqft"] - sqft_mean) / sqft_std).values
bedrooms_z = ((df["bedrooms"] - bedrooms_mean) / bedrooms_std).values
price_z = ((df["price"] - price_mean) / price_std).values

# ---------------------------------------------------------------------------
# 3. Model 1 -- Simple linear regression
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL 1: Simple Linear Regression")
print("=" * 60)

coords_m1 = {"obs": np.arange(n_obs), "predictor": ["sqft", "bedrooms"]}

with pm.Model(coords=coords_m1) as model_simple:
    # --- Data containers ---
    X_data = pm.Data(
        "X", np.column_stack([sqft_z, bedrooms_z]), dims=("obs", "predictor")
    )
    y_data = pm.Data("y", price_z, dims="obs")

    # --- Priors ---
    # Intercept: centered on zero because data is standardized
    intercept = pm.Normal("intercept", mu=0, sigma=1)  # Standardized scale; ~1 SD range

    # Regression coefficients: weakly informative on standardized scale
    beta = pm.Normal(
        "beta", mu=0, sigma=2.5, dims="predictor"
    )  # Allows moderately large effects

    # Residual SD: must be positive; Gamma(2,2) avoids near-zero region
    sigma = pm.Gamma("sigma", alpha=2, beta=2)  # Mode at 0.5, wide tail

    # --- Likelihood ---
    mu = intercept + pm.math.dot(X_data, beta)
    pm.Normal("obs_like", mu=mu, sigma=sigma, observed=y_data, dims="obs")

    # --- Prior predictive check (skill rule: always run before sampling) ---
    prior_pred_m1 = pm.sample_prior_predictive(random_seed=rng)

    # --- Inference (skill rule: use nutpie, don't hardcode chains) ---
    idata_m1 = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    idata_m1.extend(prior_pred_m1)

    # --- Posterior predictive check (skill rule: always run) ---
    idata_m1.extend(pm.sample_posterior_predictive(idata_m1, random_seed=rng))

    # --- Compute log-likelihood (skill gotcha: nutpie doesn't auto-store it) ---
    pm.compute_log_likelihood(idata_m1)

    # --- Save immediately (skill rule: save before post-processing) ---
    idata_m1.to_netcdf("model_simple.nc")

# ---------------------------------------------------------------------------
# 4. Model 1 diagnostics
# ---------------------------------------------------------------------------
print("\n--- Model 1: Convergence diagnostics ---")
summary_m1 = az.summary(idata_m1, round_to=3)
print(summary_m1)

num_chains_m1 = idata_m1.posterior.sizes["chain"]
rhat_ok_m1 = (summary_m1["r_hat"] <= 1.01).all()
ess_bulk_ok_m1 = (summary_m1["ess_bulk"] >= 100 * num_chains_m1).all()
ess_tail_ok_m1 = (summary_m1["ess_tail"] >= 100 * num_chains_m1).all()
n_div_m1 = int(idata_m1.sample_stats["diverging"].sum().item())
print(f"R-hat OK: {rhat_ok_m1}")
print(f"ESS bulk OK: {ess_bulk_ok_m1}, ESS tail OK: {ess_tail_ok_m1}")
print(f"Divergences: {n_div_m1}")

# Trace / rank plots
az.plot_trace(idata_m1, kind="rank_vlines")
plt.suptitle("Model 1 (Simple) -- Rank Plots", y=1.02)
plt.tight_layout()
plt.savefig("m1_trace_rank.png", dpi=150, bbox_inches="tight")
plt.close()

# PPC
az.plot_ppc(idata_m1, num_pp_samples=100)
plt.title("Model 1 -- Posterior Predictive Check")
plt.tight_layout()
plt.savefig("m1_ppc.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 5. Model 2 -- Hierarchical model (neighborhood varying intercept)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL 2: Hierarchical (Neighborhood Varying Intercept)")
print("=" * 60)

coords_m2 = {
    "obs": np.arange(n_obs),
    "predictor": ["sqft", "bedrooms"],
    "neighborhood": neighborhood_names,
}

with pm.Model(coords=coords_m2) as model_hier:
    # --- Data containers ---
    X_data = pm.Data(
        "X", np.column_stack([sqft_z, bedrooms_z]), dims=("obs", "predictor")
    )
    y_data = pm.Data("y", price_z, dims="obs")
    neigh_idx = pm.Data("neigh_idx", neighborhood_idx.astype(int), dims="obs")

    # --- Priors ---
    # Global intercept: standardized scale
    intercept = pm.Normal("intercept", mu=0, sigma=1)  # Standardized; ~1 SD range

    # Regression coefficients: same weakly informative priors as Model 1
    beta = pm.Normal(
        "beta", mu=0, sigma=2.5, dims="predictor"
    )  # Allows moderately large effects

    # Neighborhood-level SD: Gamma avoids near-zero funnel (skill gotcha)
    sigma_neigh = pm.Gamma(
        "sigma_neigh", alpha=2, beta=2
    )  # Positive, avoids funnel near zero

    # Non-centered parameterization (skill rule: start non-centered)
    neigh_offset = pm.Normal(
        "neigh_offset", mu=0, sigma=1, dims="neighborhood"
    )  # Raw offsets for non-centered param.
    neigh_effect = pm.Deterministic(
        "neigh_effect",
        neigh_offset * sigma_neigh,
        dims="neighborhood",
    )  # Actual neighborhood deviation

    # Residual SD
    sigma = pm.Gamma("sigma", alpha=2, beta=2)  # Same as Model 1; positive, wide tail

    # --- Likelihood ---
    mu = intercept + pm.math.dot(X_data, beta) + neigh_effect[neigh_idx]
    pm.Normal("obs_like", mu=mu, sigma=sigma, observed=y_data, dims="obs")

    # --- Prior predictive check ---
    prior_pred_m2 = pm.sample_prior_predictive(random_seed=rng)

    # --- Inference ---
    idata_m2 = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    idata_m2.extend(prior_pred_m2)

    # --- Posterior predictive check ---
    idata_m2.extend(pm.sample_posterior_predictive(idata_m2, random_seed=rng))

    # --- Compute log-likelihood (nutpie gotcha) ---
    pm.compute_log_likelihood(idata_m2)

    # --- Save immediately ---
    idata_m2.to_netcdf("model_hierarchical.nc")

# ---------------------------------------------------------------------------
# 6. Model 2 diagnostics
# ---------------------------------------------------------------------------
print("\n--- Model 2: Convergence diagnostics ---")
summary_m2 = az.summary(idata_m2, round_to=3)
print(summary_m2)

num_chains_m2 = idata_m2.posterior.sizes["chain"]
rhat_ok_m2 = (summary_m2["r_hat"] <= 1.01).all()
ess_bulk_ok_m2 = (summary_m2["ess_bulk"] >= 100 * num_chains_m2).all()
ess_tail_ok_m2 = (summary_m2["ess_tail"] >= 100 * num_chains_m2).all()
n_div_m2 = int(idata_m2.sample_stats["diverging"].sum().item())
print(f"R-hat OK: {rhat_ok_m2}")
print(f"ESS bulk OK: {ess_bulk_ok_m2}, ESS tail OK: {ess_tail_ok_m2}")
print(f"Divergences: {n_div_m2}")

# Trace / rank plots
az.plot_trace(idata_m2, kind="rank_vlines")
plt.suptitle("Model 2 (Hierarchical) -- Rank Plots", y=1.02)
plt.tight_layout()
plt.savefig("m2_trace_rank.png", dpi=150, bbox_inches="tight")
plt.close()

# PPC
az.plot_ppc(idata_m2, num_pp_samples=100)
plt.title("Model 2 -- Posterior Predictive Check")
plt.tight_layout()
plt.savefig("m2_ppc.png", dpi=150, bbox_inches="tight")
plt.close()

# Shrinkage plot (hierarchical-specific diagnostic)
neigh_means_posterior = (
    idata_m2.posterior["neigh_effect"]
    .mean(dim=["chain", "draw"])
    .values
)
neigh_means_obs = np.array(
    [price_z[neighborhood_idx == g].mean() for g in range(n_neighborhoods)]
)

plt.figure(figsize=(7, 6))
plt.scatter(neigh_means_obs, neigh_means_posterior, s=80, zorder=5)
lims = [
    min(neigh_means_obs.min(), neigh_means_posterior.min()) - 0.1,
    max(neigh_means_obs.max(), neigh_means_posterior.max()) + 0.1,
]
plt.plot(lims, lims, "r--", label="No pooling (identity)")
plt.axhline(0, color="gray", linestyle=":", label="Complete pooling (global mean=0)")
for i, name in enumerate(neighborhood_names):
    plt.annotate(
        name.replace("neighborhood_", "N"),
        (neigh_means_obs[i], neigh_means_posterior[i]),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=8,
    )
plt.xlabel("Observed neighborhood mean (standardized)")
plt.ylabel("Posterior neighborhood effect")
plt.legend()
plt.title("Shrinkage toward global mean")
plt.tight_layout()
plt.savefig("m2_shrinkage.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 7. LOO-CV for each model individually (model criticism)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("LOO-CV: Individual model criticism")
print("=" * 60)

loo_m1 = az.loo(idata_m1, pointwise=True)
loo_m2 = az.loo(idata_m2, pointwise=True)

print("\nModel 1 (Simple):")
print(loo_m1)

print("\nModel 2 (Hierarchical):")
print(loo_m2)

# Pareto k diagnostics
pareto_k_m1 = loo_m1.pareto_k.values
pareto_k_m2 = loo_m2.pareto_k.values

bad_obs_m1 = np.where(pareto_k_m1 > 0.7)[0]
bad_obs_m2 = np.where(pareto_k_m2 > 0.7)[0]

print(f"\nModel 1 -- observations with Pareto k > 0.7: {bad_obs_m1}")
print(f"Model 2 -- observations with Pareto k > 0.7: {bad_obs_m2}")

# Pareto k hat plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
az.plot_khat(loo_m1, ax=axes[0])
axes[0].set_title("Model 1 (Simple) -- Pareto k")
az.plot_khat(loo_m2, ax=axes[1])
axes[1].set_title("Model 2 (Hierarchical) -- Pareto k")
plt.tight_layout()
plt.savefig("pareto_k_both.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 8. Model comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

models = {"simple_linear": idata_m1, "hierarchical_neigh": idata_m2}

# az.compare uses stacking weights by default
comparison = az.compare(models)
print("\n--- Comparison table ---")
print(comparison)

# Visualize comparison
az.plot_compare(comparison)
plt.title("Model Comparison (LOO-CV)")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 9. Interpret the magnitude of the difference
# ---------------------------------------------------------------------------
print("\n--- Interpreting ELPD difference ---")
# The first row in the sorted comparison table is the best model
best_model = comparison.index[0]
second_model = comparison.index[1]

elpd_diff = comparison.loc[second_model, "elpd_diff"]
dse = comparison.loc[second_model, "dse"]

ratio = abs(elpd_diff) / dse if dse > 0 else float("inf")

print(f"Best model: {best_model}")
print(f"ELPD difference: {elpd_diff:.2f}")
print(f"SE of difference: {dse:.2f}")
print(f"|ELPD_diff| / dse = {ratio:.2f}")

if ratio < 2:
    strength = "weak (models practically indistinguishable -- prefer simpler)"
elif ratio < 4:
    strength = "moderate (consider domain knowledge)"
else:
    strength = "strong (clear preference for best model)"
print(f"Evidence strength: {strength}")

# Stacking weights
print("\n--- Stacking weights ---")
print(comparison[["weight"]])

# ---------------------------------------------------------------------------
# 10. Calibration (PIT) -- skill rule: mandatory for every model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CALIBRATION (LOO-PIT)")
print("=" * 60)

try:
    import arviz_plots as azp

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    azp.plot_ppc_pit(idata_m1, ax=axes[0])
    axes[0].set_title("Model 1 (Simple) -- LOO-PIT")
    azp.plot_ppc_pit(idata_m2, ax=axes[1])
    axes[1].set_title("Model 2 (Hierarchical) -- LOO-PIT")
    plt.tight_layout()
    plt.savefig("calibration_pit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Calibration PIT plots saved to calibration_pit.png")
except ImportError:
    # Fallback: use az.plot_loo_pit from older ArviZ if arviz_plots unavailable
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    az.plot_loo_pit(idata_m1, y="obs_like", ax=axes[0])
    axes[0].set_title("Model 1 (Simple) -- LOO-PIT")
    az.plot_loo_pit(idata_m2, y="obs_like", ax=axes[1])
    axes[1].set_title("Model 2 (Hierarchical) -- LOO-PIT")
    plt.tight_layout()
    plt.savefig("calibration_pit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Calibration LOO-PIT plots saved (via az.plot_loo_pit fallback)")

# ---------------------------------------------------------------------------
# 11. Residual analysis for both models
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESIDUAL ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (name, idata, mod_label) in enumerate(
    [
        ("simple_linear", idata_m1, "Model 1 (Simple)"),
        ("hierarchical_neigh", idata_m2, "Model 2 (Hierarchical)"),
    ]
):
    pp_mean = idata.posterior_predictive["obs_like"].mean(dim=["chain", "draw"]).values
    residuals = price_z - pp_mean

    # Residuals vs fitted
    axes[idx, 0].scatter(pp_mean, residuals, alpha=0.3, s=10)
    axes[idx, 0].axhline(0, color="red", linestyle="--")
    axes[idx, 0].set_xlabel("Fitted values")
    axes[idx, 0].set_ylabel("Residuals")
    axes[idx, 0].set_title(f"{mod_label} -- Residuals vs Fitted")

    # Residuals vs sqft (check for missed nonlinearity)
    axes[idx, 1].scatter(sqft_z, residuals, alpha=0.3, s=10)
    axes[idx, 1].axhline(0, color="red", linestyle="--")
    axes[idx, 1].set_xlabel("Square footage (standardized)")
    axes[idx, 1].set_ylabel("Residuals")
    axes[idx, 1].set_title(f"{mod_label} -- Residuals vs Sqft")

plt.tight_layout()
plt.savefig("residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Residual plots saved to residuals.png")

# ---------------------------------------------------------------------------
# 12. Summary printout
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"\nModel 1 (Simple Linear Regression):")
print(f"  ELPD_LOO = {comparison.loc['simple_linear', 'elpd_loo']:.1f}")
print(f"  SE       = {comparison.loc['simple_linear', 'se']:.1f}")
print(f"  Weight   = {comparison.loc['simple_linear', 'weight']:.3f}")
print(f"  Warning  = {comparison.loc['simple_linear', 'warning']}")

print(f"\nModel 2 (Hierarchical):")
print(f"  ELPD_LOO = {comparison.loc['hierarchical_neigh', 'elpd_loo']:.1f}")
print(f"  SE       = {comparison.loc['hierarchical_neigh', 'se']:.1f}")
print(f"  Weight   = {comparison.loc['hierarchical_neigh', 'weight']:.3f}")
print(f"  Warning  = {comparison.loc['hierarchical_neigh', 'warning']}")

print(f"\nELPD difference: {elpd_diff:.2f} (SE = {dse:.2f})")
print(f"Evidence strength: {strength}")
print(f"\nBest model by LOO: {best_model}")

print("\nAll plots and InferenceData files saved. See comparison_notes.md for interpretation.")
