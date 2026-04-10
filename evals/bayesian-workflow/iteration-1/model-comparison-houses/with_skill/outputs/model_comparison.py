"""
Model Comparison: Simple Linear Regression vs. Hierarchical Neighborhood Model
for House Price Prediction.

This script compares two Bayesian models:
  - Model 1 (simple): price ~ Normal(alpha + beta_sqft * sqft + beta_bed * bedrooms, sigma)
  - Model 2 (hierarchical): Same as Model 1, but with neighborhood-level
    varying intercepts (and optionally varying slopes), partially pooled
    toward a global mean.

Both models have already converged. This script takes their InferenceData objects
and runs the full comparison workflow per the Bayesian Workflow skill:
  1. Verify convergence diagnostics for both models
  2. Posterior predictive checks for both models
  3. LOO-CV comparison (ELPD, Pareto k diagnostics)
  4. Stacking weights
  5. Visualization of comparison results
  6. Residual comparison

Prerequisites:
  - Both models must have been sampled with `idata_kwargs={"log_likelihood": True}`
    OR you must compute log-likelihoods after sampling (see Step 0 below).
  - Both InferenceData objects should contain posterior_predictive groups
    (from `pm.sample_posterior_predictive`).

Usage:
  Adapt the loading section (Step 0) to point to your actual InferenceData objects,
  then run the script end to end.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import warnings

# ---------------------------------------------------------------------------
# Step 0: Load InferenceData objects
# ---------------------------------------------------------------------------
# Replace these with your actual InferenceData objects or file paths.
# If loading from disk:
#   idata_simple = az.from_netcdf("path/to/simple_model.nc")
#   idata_hier = az.from_netcdf("path/to/hierarchical_model.nc")
#
# If you already have them in memory, just assign them:
#   idata_simple = <your simple model InferenceData>
#   idata_hier = <your hierarchical model InferenceData>

# PLACEHOLDER -- replace with your actual objects:
idata_simple = None  # <-- your simple linear regression InferenceData
idata_hier = None  # <-- your hierarchical neighborhood InferenceData

# If log-likelihood was NOT stored during sampling, compute it now.
# You need the original PyMC model contexts for this:
#
# with simple_model:
#     pm.compute_log_likelihood(idata_simple)
# with hierarchical_model:
#     pm.compute_log_likelihood(idata_hier)

# Also, if posterior predictive was NOT stored during sampling, compute it now:
#
# with simple_model:
#     pm.sample_posterior_predictive(idata_simple, extend_inferencedata=True)
# with hierarchical_model:
#     pm.sample_posterior_predictive(idata_hier, extend_inferencedata=True)

# Observed data (needed for residual analysis). Replace with your actual data.
# y_obs = df["price"].to_numpy()
# sqft = df["sqft"].to_numpy()
# bedrooms = df["bedrooms"].to_numpy()
# neighborhoods = df["neighborhood"].to_numpy()

# ---------------------------------------------------------------------------
# Step 1: Verify convergence diagnostics for both models
# ---------------------------------------------------------------------------
# Per the Bayesian Workflow skill: "Always check convergence before interpreting
# results. R-hat > 1.01 or ESS < 100 * nbr_chains means results are unreliable."
# The user states both models have converged, but we verify anyway.


def run_diagnostics(idata, model_name="model"):
    """
    Run all convergence diagnostics for a single model.
    Returns a dict summarizing convergence status.
    """
    summary = az.summary(idata, round_to=3)
    num_chains = idata.posterior.dims["chain"]

    rhat_max = float(summary["r_hat"].max())
    rhat_ok = bool((summary["r_hat"] <= 1.01).all())

    ess_bulk_min = int(summary["ess_bulk"].min())
    ess_tail_min = int(summary["ess_tail"].min())
    ess_ok = bool(
        (summary["ess_bulk"] >= 100 * num_chains).all()
        and (summary["ess_tail"] >= 100 * num_chains).all()
    )

    n_divergences = int(idata.sample_stats["diverging"].sum())
    divergences_ok = n_divergences == 0

    all_ok = rhat_ok and ess_ok and divergences_ok

    results = {
        "model_name": model_name,
        "rhat_max": rhat_max,
        "rhat_ok": rhat_ok,
        "ess_bulk_min": ess_bulk_min,
        "ess_tail_min": ess_tail_min,
        "ess_ok": ess_ok,
        "n_divergences": n_divergences,
        "divergences_ok": divergences_ok,
        "all_ok": all_ok,
    }

    print(f"\n{'='*60}")
    print(f"Convergence Diagnostics: {model_name}")
    print(f"{'='*60}")
    print(f"  R-hat max:       {rhat_max:.4f}  {'OK' if rhat_ok else 'FAIL'}")
    print(f"  ESS bulk min:    {ess_bulk_min}  {'OK' if ess_ok else 'FAIL'}")
    print(f"  ESS tail min:    {ess_tail_min}  {'OK' if ess_ok else 'FAIL'}")
    print(f"  Divergences:     {n_divergences}  {'OK' if divergences_ok else 'FAIL'}")
    print(f"  Overall:         {'PASS' if all_ok else 'FAIL -- do NOT interpret results'}")

    if not all_ok:
        warnings.warn(
            f"Model '{model_name}' has convergence issues. "
            "Do NOT interpret results until these are resolved.",
            stacklevel=2,
        )

    return results


print("=" * 60)
print("STEP 1: Convergence Diagnostics")
print("=" * 60)

diag_simple = run_diagnostics(idata_simple, "Simple Linear Regression")
diag_hier = run_diagnostics(idata_hier, "Hierarchical Neighborhood Model")

# Trace / rank plots for visual inspection
fig_trace_simple = az.plot_trace(idata_simple, kind="rank_vlines")
plt.suptitle("Simple Model: Rank Plots", y=1.02)
plt.tight_layout()
plt.savefig("outputs/trace_rank_simple.png", dpi=150, bbox_inches="tight")
plt.show()

fig_trace_hier = az.plot_trace(idata_hier, kind="rank_vlines")
plt.suptitle("Hierarchical Model: Rank Plots", y=1.02)
plt.tight_layout()
plt.savefig("outputs/trace_rank_hierarchical.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Step 2: Posterior predictive checks for both models
# ---------------------------------------------------------------------------
# Per the skill: "Always run posterior predictive checks. A model that fits well
# numerically but cannot reproduce the data is useless."

print("\n" + "=" * 60)
print("STEP 2: Posterior Predictive Checks")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Simple model PPC
az.plot_ppc(idata_simple, num_pp_samples=100, ax=axes[0])
axes[0].set_title("Simple Model: Posterior Predictive Check")

# Hierarchical model PPC
az.plot_ppc(idata_hier, num_pp_samples=100, ax=axes[1])
axes[1].set_title("Hierarchical Model: Posterior Predictive Check")

plt.tight_layout()
plt.savefig("outputs/ppc_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print(
    "Inspect the PPC plots above. The posterior predictive distribution should "
    "envelop the observed data. Check shape, spread, skewness, and tails."
)

# ---------------------------------------------------------------------------
# Step 3: LOO-CV Comparison
# ---------------------------------------------------------------------------
# This is the primary comparison tool, per the skill's model-comparison reference.
# Uses PSIS-LOO via ArviZ.

print("\n" + "=" * 60)
print("STEP 3: LOO-CV Model Comparison")
print("=" * 60)

# Build the model dictionary
models = {
    "Simple (sqft + bedrooms)": idata_simple,
    "Hierarchical (+ neighborhood)": idata_hier,
}

# Run the comparison -- default method is stacking for weights
comparison = az.compare(models)
print("\nFull comparison table:")
print(comparison)
print()

# ---------------------------------------------------------------------------
# Step 4: Interpret ELPD differences
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Interpreting ELPD Differences")
print("=" * 60)

# Extract key values from the comparison table
# The table is sorted by elpd_loo (best model first)
best_model_name = comparison.index[0]
second_model_name = comparison.index[1]

elpd_diff = comparison.loc[second_model_name, "elpd_diff"]
dse = comparison.loc[second_model_name, "dse"]

# Interpret per the skill's guidelines:
# - elpd_diff < 2 * dse --> models practically indistinguishable, prefer simpler
# - elpd_diff > 4 * dse --> strong evidence for the better model
# - between 2-4 * dse --> moderate evidence, consider domain knowledge
ratio = abs(elpd_diff) / dse if dse > 0 else float("inf")

print(f"Best model:            {best_model_name}")
print(f"Second model:          {second_model_name}")
print(f"ELPD difference:       {elpd_diff:.2f}")
print(f"SE of difference:      {dse:.2f}")
print(f"|ELPD_diff| / dSE:     {ratio:.2f}")
print()

if ratio < 2:
    interpretation = (
        "The models are practically indistinguishable in predictive accuracy. "
        "Prefer the simpler model (Simple Linear Regression) unless domain "
        "knowledge strongly favors the hierarchical structure."
    )
elif ratio < 4:
    interpretation = (
        f"There is moderate evidence favoring '{best_model_name}'. "
        "Consider domain knowledge: if neighborhoods are known to matter for "
        "house prices (they almost certainly do), the hierarchical model is "
        "preferred. If the difference is small, the simpler model may still be "
        "adequate for some use cases."
    )
else:
    interpretation = (
        f"There is strong evidence favoring '{best_model_name}'. "
        "The ELPD difference is large relative to its standard error, "
        "indicating meaningfully better out-of-sample predictive accuracy."
    )

print(f"Interpretation: {interpretation}")

# ---------------------------------------------------------------------------
# Step 5: Stacking weights
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Stacking Weights")
print("=" * 60)

# Stacking weights are already computed in az.compare (default method="stacking")
print("\nStacking weights (optimal combination for prediction):")
for model_name in comparison.index:
    weight = comparison.loc[model_name, "weight"]
    print(f"  {model_name}: {weight:.3f}")

print(
    "\nStacking weights minimize expected log predictive density loss. "
    "If no single model dominates, consider using stacking to combine "
    "predictions from both models."
)

# ---------------------------------------------------------------------------
# Step 6: Pareto k diagnostics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Pareto k Diagnostics")
print("=" * 60)

# Compute pointwise LOO for each model to inspect Pareto k values
loo_simple = az.loo(idata_simple, pointwise=True)
loo_hier = az.loo(idata_hier, pointwise=True)

for name, loo_result in [
    ("Simple", loo_simple),
    ("Hierarchical", loo_hier),
]:
    pareto_k = loo_result.pareto_k.values
    n_bad_05 = int(np.sum(pareto_k > 0.5))
    n_bad_07 = int(np.sum(pareto_k > 0.7))
    n_total = len(pareto_k)

    print(f"\n{name} model:")
    print(f"  Pareto k > 0.5 (marginally reliable): {n_bad_05} / {n_total}")
    print(f"  Pareto k > 0.7 (unreliable):          {n_bad_07} / {n_total}")
    print(f"  p_loo (effective parameters):          {loo_result.p_loo:.2f}")

    if n_bad_07 > 0:
        bad_obs = np.where(pareto_k > 0.7)[0]
        print(
            f"  WARNING: {n_bad_07} observations have Pareto k > 0.7. "
            f"LOO estimates for these observations are unreliable."
        )
        print(f"  Problematic observation indices: {bad_obs}")
        print(
            "  Consider: (1) investigating these observations for outliers, "
            "(2) using K-fold CV, or (3) moment matching via az.loo(..., method='moment_matching')."
        )
    else:
        print("  All Pareto k values are acceptable.")

# Pareto k plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

az.plot_khat(loo_simple, ax=axes[0])
axes[0].set_title("Simple Model: Pareto k")

az.plot_khat(loo_hier, ax=axes[1])
axes[1].set_title("Hierarchical Model: Pareto k")

plt.tight_layout()
plt.savefig("outputs/pareto_k_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Step 7: Visual comparison (ELPD plot)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: Visual Comparison")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 4))
az.plot_compare(comparison, ax=ax)
ax.set_title("Model Comparison (LOO-CV)")
plt.tight_layout()
plt.savefig("outputs/elpd_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Step 8: Residual comparison (if observed data is available)
# ---------------------------------------------------------------------------
# Uncomment and adapt once you have the observed data loaded.

# print("\n" + "=" * 60)
# print("STEP 8: Residual Comparison")
# print("=" * 60)
#
# # Compute posterior predictive means for each model
# pp_mean_simple = (
#     idata_simple.posterior_predictive["obs"]
#     .mean(dim=["chain", "draw"])
#     .to_numpy()
# )
# pp_mean_hier = (
#     idata_hier.posterior_predictive["obs"]
#     .mean(dim=["chain", "draw"])
#     .to_numpy()
# )
#
# resid_simple = y_obs - pp_mean_simple
# resid_hier = y_obs - pp_mean_hier
#
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#
# # Simple model: residuals vs fitted
# axes[0, 0].scatter(pp_mean_simple, resid_simple, alpha=0.4, s=10)
# axes[0, 0].axhline(0, color="red", linestyle="--")
# axes[0, 0].set_xlabel("Fitted values")
# axes[0, 0].set_ylabel("Residuals")
# axes[0, 0].set_title("Simple Model: Residuals vs. Fitted")
#
# # Hierarchical model: residuals vs fitted
# axes[0, 1].scatter(pp_mean_hier, resid_hier, alpha=0.4, s=10)
# axes[0, 1].axhline(0, color="red", linestyle="--")
# axes[0, 1].set_xlabel("Fitted values")
# axes[0, 1].set_ylabel("Residuals")
# axes[0, 1].set_title("Hierarchical Model: Residuals vs. Fitted")
#
# # Simple model: residuals vs square footage (check for missed nonlinearity)
# axes[1, 0].scatter(sqft, resid_simple, alpha=0.4, s=10)
# axes[1, 0].axhline(0, color="red", linestyle="--")
# axes[1, 0].set_xlabel("Square Footage")
# axes[1, 0].set_ylabel("Residuals")
# axes[1, 0].set_title("Simple Model: Residuals vs. Sqft")
#
# # Hierarchical model: residuals by neighborhood (check if grouping is captured)
# for nbhd in np.unique(neighborhoods):
#     mask = neighborhoods == nbhd
#     axes[1, 1].scatter(
#         pp_mean_hier[mask], resid_hier[mask], alpha=0.4, s=10, label=nbhd
#     )
# axes[1, 1].axhline(0, color="red", linestyle="--")
# axes[1, 1].set_xlabel("Fitted values")
# axes[1, 1].set_ylabel("Residuals")
# axes[1, 1].set_title("Hierarchical Model: Residuals by Neighborhood")
# axes[1, 1].legend(fontsize=7, ncol=2)
#
# plt.tight_layout()
# plt.savefig("outputs/residual_comparison.png", dpi=150, bbox_inches="tight")
# plt.show()
#
# # Look for: trends (missed nonlinearity), fans (heteroscedasticity),
# # clusters (missing grouping variable in the simple model)

# ---------------------------------------------------------------------------
# Step 9: Shrinkage plot for the hierarchical model
# ---------------------------------------------------------------------------
# This is specific to the hierarchical model -- visualize how much each
# neighborhood is pulled toward the global mean.

# Uncomment and adapt once you have the data:
#
# print("\n" + "=" * 60)
# print("STEP 9: Shrinkage Plot (Hierarchical Model)")
# print("=" * 60)
#
# # Posterior neighborhood intercepts
# nbhd_intercepts_post = (
#     idata_hier.posterior["neighborhood_intercept"]
#     .mean(dim=["chain", "draw"])
#     .to_numpy()
# )
#
# # Observed neighborhood means (no-pooling estimates)
# unique_nbhds = np.unique(neighborhoods)
# nbhd_means_obs = np.array([
#     y_obs[neighborhoods == n].mean() for n in unique_nbhds
# ])
#
# # Global mean (complete-pooling estimate)
# global_mean = y_obs.mean()
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(nbhd_means_obs, nbhd_intercepts_post, s=40, zorder=3)
#
# # Identity line (no pooling)
# lims = [
#     min(nbhd_means_obs.min(), nbhd_intercepts_post.min()),
#     max(nbhd_means_obs.max(), nbhd_intercepts_post.max()),
# ]
# ax.plot(lims, lims, "r--", label="No pooling (identity)")
# ax.axhline(global_mean, color="gray", linestyle=":", label="Complete pooling")
#
# ax.set_xlabel("Observed Neighborhood Mean Price")
# ax.set_ylabel("Posterior Neighborhood Intercept")
# ax.set_title("Shrinkage: Neighborhood Intercepts toward Global Mean")
# ax.legend()
# plt.tight_layout()
# plt.savefig("outputs/shrinkage_plot.png", dpi=150, bbox_inches="tight")
# plt.show()

# ---------------------------------------------------------------------------
# Step 10: Summary and recommendation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 10: Summary and Recommendation")
print("=" * 60)

print(f"""
Model Comparison Summary
========================

Models compared:
  1. Simple Linear Regression: price ~ sqft + bedrooms
  2. Hierarchical Model: price ~ sqft + bedrooms + (1 | neighborhood)

Comparison method: PSIS-LOO (Pareto-smoothed importance sampling LOO-CV)

Results:
  Best model by ELPD:    {best_model_name}
  ELPD difference:       {elpd_diff:.2f} (SE = {dse:.2f})
  |ELPD_diff| / dSE:     {ratio:.2f}

Stacking weights:""")
for model_name in comparison.index:
    w = comparison.loc[model_name, "weight"]
    print(f"  {model_name}: {w:.3f}")

print(f"""
Interpretation:
  {interpretation}

IMPORTANT CAVEATS:
  - This comparison tells us about PREDICTIVE ACCURACY, not causation.
  - The hierarchical model captures neighborhood-level variation through
    partial pooling, which is especially valuable when some neighborhoods
    have few observations.
  - Even if the simple model has similar ELPD, the hierarchical model
    provides richer inference (neighborhood-specific estimates with
    appropriate uncertainty).
  - If the goal is prediction for new houses in known neighborhoods,
    the hierarchical model is likely more useful regardless of small
    ELPD differences.
  - If the goal is prediction for new houses in UNKNOWN neighborhoods,
    both models are similar (the hierarchical model would use the
    population-level intercept).

Next steps:
  1. Inspect PPC plots for systematic misfit in either model.
  2. Check Pareto k values -- if any > 0.7, investigate those observations.
  3. If residual plots show patterns (heteroscedasticity, nonlinearity),
     consider adding complexity (e.g., log-transform price, add interactions,
     use StudentT likelihood for heavier tails).
  4. Report results using the reporting template from the Bayesian Workflow
     skill (see comparison_notes.md for the template).
""")

# ---------------------------------------------------------------------------
# Print the formatted comparison table for reporting
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Formatted Comparison Table (for reports)")
print("=" * 60)
print()
print("| Model | ELPD (LOO) | SE | Delta ELPD | Delta SE | Weight |")
print("|-------|------------|-----|------------|----------|--------|")
for model_name in comparison.index:
    row = comparison.loc[model_name]
    elpd = row["elpd_loo"]
    se = row["se"]
    d_elpd = row["elpd_diff"]
    d_se = row["dse"]
    w = row["weight"]
    # Format delta columns -- the best model has diff=0
    d_elpd_str = f"{d_elpd:.1f}" if d_elpd != 0.0 else "0.0"
    d_se_str = f"{d_se:.1f}" if d_se != 0.0 else "--"
    print(
        f"| {model_name} | {elpd:.1f} | {se:.1f} | "
        f"{d_elpd_str} | {d_se_str} | {w:.2f} |"
    )

print("\nDone. All comparison outputs saved to the outputs/ directory.")
