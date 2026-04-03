"""
Model comparison: Simple linear regression vs. hierarchical neighborhood model
for house price prediction.

Assumes both models have already been fit and their InferenceData objects
(idata_simple, idata_hierarchical) are available in the current session,
with convergence already verified.
"""

import pymc as pm
import arviz as az
import arviz_plots as azp
import numpy as np
import matplotlib.pyplot as plt

# Reproducible seed derived from analysis name -- never use magic numbers like 42
RANDOM_SEED = sum(map(ord, "house-price-comparison"))
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Step 1: Ensure log-likelihood is stored in both InferenceData objects
# ---------------------------------------------------------------------------
# nutpie silently ignores idata_kwargs={"log_likelihood": True}, so we
# must compute log-likelihood explicitly after sampling.

with simple_model:
    pm.compute_log_likelihood(idata_simple)

with hierarchical_model:
    pm.compute_log_likelihood(idata_hierarchical)


# ---------------------------------------------------------------------------
# Step 2: LOO-CV for each model individually
# ---------------------------------------------------------------------------
# Compute pointwise LOO so we can inspect Pareto k diagnostics per model
# before comparing. This catches problematic observations early.

loo_simple = az.loo(idata_simple, pointwise=True)
print("=== Simple model LOO ===")
print(loo_simple)
print()

loo_hierarchical = az.loo(idata_hierarchical, pointwise=True)
print("=== Hierarchical model LOO ===")
print(loo_hierarchical)
print()

# ---------------------------------------------------------------------------
# Step 3: Check Pareto k diagnostics for each model
# ---------------------------------------------------------------------------
# High Pareto k (> 0.7) means LOO is unreliable for those observations.
# If many are flagged, consider K-fold CV instead.

for name, loo_obj in [("Simple", loo_simple), ("Hierarchical", loo_hierarchical)]:
    pareto_k = loo_obj.pareto_k.values
    bad_obs = np.where(pareto_k > 0.7)[0]
    marginal_obs = np.where((pareto_k > 0.5) & (pareto_k <= 0.7))[0]
    print(f"{name} model:")
    print(f"  Observations with Pareto k > 0.7 (unreliable): {len(bad_obs)}")
    if len(bad_obs) > 0:
        print(f"    Indices: {bad_obs}")
    print(f"  Observations with Pareto k 0.5-0.7 (marginal): {len(marginal_obs)}")
    print()

# Visualize Pareto k values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
az.plot_khat(loo_simple, ax=axes[0])
axes[0].set_title("Simple Model: Pareto k")
az.plot_khat(loo_hierarchical, ax=axes[1])
axes[1].set_title("Hierarchical Model: Pareto k")
plt.tight_layout()
plt.savefig("pareto_k_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Step 4: Model comparison with az.compare
# ---------------------------------------------------------------------------
# az.compare ranks models by ELPD and computes stacking weights.
# Stacking weights give a more nuanced picture than picking a single winner.

models = {
    "simple_linear": idata_simple,
    "hierarchical_neighborhood": idata_hierarchical,
}

comparison = az.compare(models)
print("=== Model Comparison Table ===")
print(comparison)
print()

# ---------------------------------------------------------------------------
# Step 5: Interpret the ELPD difference
# ---------------------------------------------------------------------------
# Rule of thumb from reference/model-comparison.md:
#   - elpd_diff < 2 * dse  --> models practically indistinguishable, prefer simpler
#   - elpd_diff > 4 * dse  --> strong evidence for the better model
#   - 2-4 * dse            --> moderate evidence, consider domain knowledge

elpd_diff = comparison["elpd_diff"].iloc[1]  # difference for the second-ranked model
dse = comparison["dse"].iloc[1]  # standard error of the difference

if dse > 0:
    ratio = abs(elpd_diff) / dse
else:
    ratio = float("inf")

print(f"ELPD difference: {elpd_diff:.2f}")
print(f"SE of difference: {dse:.2f}")
print(f"|ELPD_diff| / dse ratio: {ratio:.2f}")
print()

if ratio < 2:
    print(
        "Interpretation: Models are practically indistinguishable in predictive "
        "accuracy. Prefer the simpler model (simple linear regression) unless "
        "domain knowledge strongly favors the hierarchical structure."
    )
elif ratio < 4:
    print(
        "Interpretation: Moderate evidence favoring the top-ranked model. "
        "Consider domain knowledge -- does neighborhood grouping make "
        "substantive sense for this housing market?"
    )
else:
    print(
        "Interpretation: Strong evidence favoring the top-ranked model. "
        "The predictive improvement is meaningful."
    )
print()

# ---------------------------------------------------------------------------
# Step 6: Visualize the comparison
# ---------------------------------------------------------------------------
az.plot_compare(comparison)
plt.title("LOO-CV Model Comparison")
plt.tight_layout()
plt.savefig("model_comparison_elpd.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Step 7: Stacking weights
# ---------------------------------------------------------------------------
# When no single model is clearly best, stacking weights tell us how to
# optimally combine predictions from both models.

print("=== Stacking Weights ===")
for model_name in comparison.index:
    weight = comparison.loc[model_name, "weight"]
    print(f"  {model_name}: {weight:.3f}")
print()
print(
    "Stacking weights minimize expected log predictive density loss. "
    "If neither model dominates, consider using these weights to combine "
    "predictions rather than selecting a single model."
)

# ---------------------------------------------------------------------------
# Step 8: Calibration check for both models
# ---------------------------------------------------------------------------
# Calibration is mandatory for every model (per skill rules). A model with
# better ELPD but poor calibration is not trustworthy.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PPC-PIT calibration for simple model
azp.plot_ppc_pit(idata_simple, ax=axes[0])
axes[0].set_title("Simple Model: PPC-PIT Calibration")

# PPC-PIT calibration for hierarchical model
azp.plot_ppc_pit(idata_hierarchical, ax=axes[1])
axes[1].set_title("Hierarchical Model: PPC-PIT Calibration")

plt.tight_layout()
plt.savefig("calibration_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Step 9: Posterior predictive checks for both models
# ---------------------------------------------------------------------------
# A model that wins on ELPD but cannot reproduce the data is useless.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

az.plot_ppc(idata_simple, ax=axes[0])
axes[0].set_title("Simple Model: Posterior Predictive Check")

az.plot_ppc(idata_hierarchical, ax=axes[1])
axes[1].set_title("Hierarchical Model: Posterior Predictive Check")

plt.tight_layout()
plt.savefig("ppc_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Step 10: Summary of key parameters from both models
# ---------------------------------------------------------------------------
# Report full posteriors with 94% HDI -- never point estimates alone.

print("=== Simple Model Parameter Summary ===")
print(az.summary(idata_simple, hdi_prob=0.94))
print()

print("=== Hierarchical Model Parameter Summary ===")
print(az.summary(idata_hierarchical, hdi_prob=0.94))
print()

# ---------------------------------------------------------------------------
# Step 11: Print final recommendation
# ---------------------------------------------------------------------------
best_model = comparison.index[0]
second_model = comparison.index[1]
best_weight = comparison.loc[best_model, "weight"]

print("=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print(f"Top-ranked model by ELPD: {best_model}")
print(f"Stacking weight: {best_weight:.3f}")
print()

if ratio < 2:
    print(
        f"The models are practically indistinguishable (|ELPD_diff|/dse = {ratio:.1f}).\n"
        f"The simpler model (simple linear regression) is preferred by parsimony.\n"
        f"Only adopt the hierarchical model if domain knowledge strongly suggests\n"
        f"that neighborhood effects are important for this housing market."
    )
elif ratio < 4:
    print(
        f"There is moderate evidence favoring {best_model} (|ELPD_diff|/dse = {ratio:.1f}).\n"
        f"Consider whether neighborhood effects are substantively meaningful.\n"
        f"If neighborhoods have genuinely different price dynamics, the hierarchical\n"
        f"model is the better choice. Otherwise, the simpler model may suffice."
    )
else:
    print(
        f"There is strong evidence favoring {best_model} (|ELPD_diff|/dse = {ratio:.1f}).\n"
        f"The predictive improvement is substantial and supports using this model.\n"
        f"Check calibration and posterior predictive plots above to confirm the\n"
        f"winning model also passes all model criticism checks."
    )

print()
print(
    "IMPORTANT: This comparison tells us about predictive accuracy only.\n"
    "Do NOT interpret it as evidence for causal effects of neighborhood on price."
)
