"""
Bayesian Model Comparison: House Price Models
==============================================

Compares two models for predicting house prices:
  - Model 1 (simple): Linear regression with square footage + bedrooms
  - Model 2 (hierarchical): Adds neighborhood as a hierarchical (partial-pooling) effect

Assumes both models have been fit with PyMC and that ArviZ InferenceData
objects (`idata_simple` and `idata_hierarchical`) are available in the
calling scope.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = (
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-1/"
    "model-comparison-houses/without_skill/outputs"
)

MODEL_NAMES = {
    "simple": "Simple Linear (sqft + beds)",
    "hierarchical": "Hierarchical (+ neighborhood)",
}


# ---------------------------------------------------------------------------
# 1. Helper: build a comparison dict expected by ArviZ
# ---------------------------------------------------------------------------
def build_model_dict(idata_simple, idata_hierarchical):
    """Return a dict mapping human-readable names to InferenceData objects."""
    return {
        MODEL_NAMES["simple"]: idata_simple,
        MODEL_NAMES["hierarchical"]: idata_hierarchical,
    }


# ---------------------------------------------------------------------------
# 2. LOO-CV (PSIS-LOO) comparison
# ---------------------------------------------------------------------------
def compare_loo(idata_simple, idata_hierarchical):
    """
    Compute and return the LOO-CV comparison table.

    LOO (Leave-One-Out cross-validation via Pareto-Smoothed Importance
    Sampling) estimates the expected log pointwise predictive density (ELPD)
    for new data.  Higher ELPD = better out-of-sample predictive accuracy.

    Returns
    -------
    df_loo : pd.DataFrame
        ArviZ comparison table sorted by ELPD (best model first).
    """
    model_dict = build_model_dict(idata_simple, idata_hierarchical)
    df_loo = az.compare(model_dict, ic="loo", method="stacking", scale="log")
    return df_loo


# ---------------------------------------------------------------------------
# 3. WAIC comparison
# ---------------------------------------------------------------------------
def compare_waic(idata_simple, idata_hierarchical):
    """
    Compute and return the WAIC comparison table.

    WAIC (Widely Applicable Information Criterion) is another estimate of
    out-of-sample predictive accuracy.  It is asymptotically equivalent to
    LOO-CV but can be less robust when there are influential observations.

    Returns
    -------
    df_waic : pd.DataFrame
        ArviZ comparison table sorted by ELPD (best model first).
    """
    model_dict = build_model_dict(idata_simple, idata_hierarchical)
    df_waic = az.compare(model_dict, ic="waic", method="stacking", scale="log")
    return df_waic


# ---------------------------------------------------------------------------
# 4. Pareto-k diagnostic check
# ---------------------------------------------------------------------------
def check_pareto_k(idata, model_label="model"):
    """
    Compute LOO for a single model and report Pareto-k diagnostics.

    Pareto-k values flag observations that are hard to leave out:
      - k < 0.5  : good
      - 0.5-0.7  : acceptable
      - 0.7-1.0  : problematic — consider moment-matching or refit
      - k > 1.0  : very bad — LOO estimate unreliable for this point

    Returns
    -------
    loo_result : az.ELPDData
        The full LOO result object.
    summary : dict
        Counts of observations in each Pareto-k category.
    """
    loo_result = az.loo(idata, pointwise=True)
    pareto_k = loo_result.pareto_k.values

    summary = {
        "good (k < 0.5)": int(np.sum(pareto_k < 0.5)),
        "acceptable (0.5 <= k < 0.7)": int(np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))),
        "problematic (0.7 <= k < 1.0)": int(np.sum((pareto_k >= 0.7) & (pareto_k < 1.0))),
        "very bad (k >= 1.0)": int(np.sum(pareto_k >= 1.0)),
    }

    print(f"\n--- Pareto-k diagnostics for {model_label} ---")
    for category, count in summary.items():
        print(f"  {category}: {count}")

    return loo_result, summary


# ---------------------------------------------------------------------------
# 5. Posterior predictive checks (visual)
# ---------------------------------------------------------------------------
def plot_ppc(idata, model_label="model", observed_var="y"):
    """
    Plot posterior predictive check overlay for a single model.

    This answers the question: "Can the model reproduce the observed data?"
    A model that fits well will show simulated data distributions that
    closely match the observed data distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    az.plot_ppc(idata, observed_rug=True, ax=ax)
    ax.set_title(f"Posterior Predictive Check — {model_label}")
    ax.set_xlabel("House Price")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/ppc_{model_label.replace(' ', '_').lower()}.png", dpi=150)
    plt.close(fig)
    print(f"Saved PPC plot for {model_label}")


# ---------------------------------------------------------------------------
# 6. LOO-PIT (Probability Integral Transform) calibration
# ---------------------------------------------------------------------------
def plot_loo_pit(idata, model_label="model"):
    """
    Plot LOO-PIT for a single model.

    If the model is well-calibrated, the LOO-PIT values should be
    approximately uniform.  Deviations indicate systematic misfit:
      - U-shaped: underdispersed (too narrow posteriors)
      - Inverted-U: overdispersed (too wide posteriors)
      - Skewed: biased predictions
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    az.plot_loo_pit(idata, y="y", ax=ax)
    ax.set_title(f"LOO-PIT — {model_label}")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/loo_pit_{model_label.replace(' ', '_').lower()}.png", dpi=150)
    plt.close(fig)
    print(f"Saved LOO-PIT plot for {model_label}")


# ---------------------------------------------------------------------------
# 7. Compare plot (forest plot of ELPD differences)
# ---------------------------------------------------------------------------
def plot_comparison(idata_simple, idata_hierarchical, ic="loo"):
    """
    Create the ArviZ model comparison plot.

    This is a forest-style plot showing ELPD point estimates and standard
    errors.  The best model is at the top; non-overlapping error bars
    indicate a meaningful difference.
    """
    model_dict = build_model_dict(idata_simple, idata_hierarchical)
    fig, ax = plt.subplots(figsize=(10, 4))
    az.plot_compare(az.compare(model_dict, ic=ic, scale="log"), ax=ax)
    ax.set_title(f"Model Comparison ({ic.upper()})")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/model_compare_{ic}.png", dpi=150)
    plt.close(fig)
    print(f"Saved model comparison plot ({ic.upper()})")


# ---------------------------------------------------------------------------
# 8. Pointwise ELPD difference plot
# ---------------------------------------------------------------------------
def plot_pointwise_elpd_diff(idata_simple, idata_hierarchical):
    """
    Plot the observation-level ELPD difference between the two models.

    Positive values = hierarchical model predicts that observation better.
    Negative values = simple model predicts that observation better.

    This helps identify *which* data points benefit from the hierarchical
    structure (typically observations in small neighborhoods where partial
    pooling helps most).
    """
    loo_simple = az.loo(idata_simple, pointwise=True)
    loo_hier = az.loo(idata_hierarchical, pointwise=True)

    elpd_diff = loo_hier.loo_i.values - loo_simple.loo_i.values

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(range(len(elpd_diff)), elpd_diff, alpha=0.5, s=15, color="steelblue")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("ELPD difference (hierarchical − simple)")
    ax.set_title("Pointwise ELPD Difference")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/pointwise_elpd_diff.png", dpi=150)
    plt.close(fig)
    print("Saved pointwise ELPD difference plot")

    return elpd_diff


# ---------------------------------------------------------------------------
# 9. Stacking weights
# ---------------------------------------------------------------------------
def compute_stacking_weights(idata_simple, idata_hierarchical):
    """
    Compute Bayesian stacking weights.

    Stacking weights indicate how much weight an optimal prediction
    combination would place on each model.  Weights sum to 1.

    A weight of ~1.0 for one model means it dominates.
    Roughly equal weights suggest both models contribute unique
    predictive information.
    """
    model_dict = build_model_dict(idata_simple, idata_hierarchical)
    df_compare = az.compare(model_dict, ic="loo", method="stacking", scale="log")
    print("\n--- Stacking Weights ---")
    for model_name in df_compare.index:
        print(f"  {model_name}: {df_compare.loc[model_name, 'weight']:.3f}")
    return df_compare["weight"]


# ---------------------------------------------------------------------------
# 10. Summary statistics side-by-side
# ---------------------------------------------------------------------------
def summarize_models(idata_simple, idata_hierarchical):
    """
    Print summary statistics for both models side by side.

    Useful for comparing shared parameters (e.g., the coefficient on
    square footage) across models.
    """
    print("\n" + "=" * 70)
    print("SIMPLE MODEL SUMMARY")
    print("=" * 70)
    print(az.summary(idata_simple, round_to=3))

    print("\n" + "=" * 70)
    print("HIERARCHICAL MODEL SUMMARY")
    print("=" * 70)
    print(az.summary(idata_hierarchical, round_to=3))


# ---------------------------------------------------------------------------
# 11. Full comparison pipeline
# ---------------------------------------------------------------------------
def run_full_comparison(idata_simple, idata_hierarchical):
    """
    Execute the complete model comparison workflow.

    Steps:
      1. Print parameter summaries
      2. LOO-CV comparison table
      3. WAIC comparison table
      4. Pareto-k diagnostics for both models
      5. Posterior predictive check plots
      6. LOO-PIT calibration plots
      7. Model comparison forest plot
      8. Pointwise ELPD difference plot
      9. Stacking weights

    All plots are saved to the output directory.
    """
    print("=" * 70)
    print("  BAYESIAN MODEL COMPARISON: HOUSE PRICE MODELS")
    print("=" * 70)

    # --- Parameter summaries ---
    summarize_models(idata_simple, idata_hierarchical)

    # --- LOO-CV ---
    print("\n" + "=" * 70)
    print("LOO-CV COMPARISON (PSIS-LOO)")
    print("=" * 70)
    df_loo = compare_loo(idata_simple, idata_hierarchical)
    print(df_loo)

    # --- WAIC ---
    print("\n" + "=" * 70)
    print("WAIC COMPARISON")
    print("=" * 70)
    df_waic = compare_waic(idata_simple, idata_hierarchical)
    print(df_waic)

    # --- Pareto-k ---
    print("\n" + "=" * 70)
    print("PARETO-K DIAGNOSTICS")
    print("=" * 70)
    loo_simple, pk_simple = check_pareto_k(
        idata_simple, MODEL_NAMES["simple"]
    )
    loo_hier, pk_hier = check_pareto_k(
        idata_hierarchical, MODEL_NAMES["hierarchical"]
    )

    # --- PPC ---
    print("\n" + "=" * 70)
    print("POSTERIOR PREDICTIVE CHECKS")
    print("=" * 70)
    plot_ppc(idata_simple, MODEL_NAMES["simple"])
    plot_ppc(idata_hierarchical, MODEL_NAMES["hierarchical"])

    # --- LOO-PIT ---
    print("\n" + "=" * 70)
    print("LOO-PIT CALIBRATION")
    print("=" * 70)
    plot_loo_pit(idata_simple, MODEL_NAMES["simple"])
    plot_loo_pit(idata_hierarchical, MODEL_NAMES["hierarchical"])

    # --- Comparison plot ---
    print("\n" + "=" * 70)
    print("MODEL COMPARISON PLOTS")
    print("=" * 70)
    plot_comparison(idata_simple, idata_hierarchical, ic="loo")

    # --- Pointwise ELPD ---
    elpd_diff = plot_pointwise_elpd_diff(idata_simple, idata_hierarchical)
    n_favor_hier = np.sum(elpd_diff > 0)
    n_total = len(elpd_diff)
    print(
        f"\nHierarchical model predicts better for {n_favor_hier}/{n_total} "
        f"observations ({100 * n_favor_hier / n_total:.1f}%)"
    )

    # --- Stacking weights ---
    print("\n" + "=" * 70)
    print("STACKING WEIGHTS")
    print("=" * 70)
    weights = compute_stacking_weights(idata_simple, idata_hierarchical)

    # --- Final recommendation logic ---
    print("\n" + "=" * 70)
    print("DECISION GUIDANCE")
    print("=" * 70)

    df_loo_final = compare_loo(idata_simple, idata_hierarchical)
    best_model = df_loo_final.index[0]
    elpd_diff_total = df_loo_final["elpd_loo"].iloc[0] - df_loo_final["elpd_loo"].iloc[1]
    se_diff = df_loo_final["dse"].iloc[1]  # SE of the difference (second row)

    if se_diff > 0:
        ratio = abs(elpd_diff_total) / se_diff
    else:
        ratio = float("inf")

    print(f"\n  Best model by LOO: {best_model}")
    print(f"  ELPD difference:   {elpd_diff_total:.2f}")
    print(f"  SE of difference:  {se_diff:.2f}")
    print(f"  |diff| / SE:       {ratio:.2f}")

    if ratio < 2:
        print(
            "\n  --> The difference is NOT decisive (|diff|/SE < 2)."
            "\n      Both models predict similarly well."
            "\n      Consider choosing based on interpretability or domain goals."
        )
    elif ratio < 5:
        print(
            f"\n  --> There is moderate evidence favoring {best_model}."
            "\n      Check Pareto-k and PPC plots to confirm."
        )
    else:
        print(
            f"\n  --> There is strong evidence favoring {best_model}."
            "\n      This model provides meaningfully better predictions."
        )

    # Check for Pareto-k warnings
    bad_k_simple = pk_simple["problematic (0.7 <= k < 1.0)"] + pk_simple["very bad (k >= 1.0)"]
    bad_k_hier = pk_hier["problematic (0.7 <= k < 1.0)"] + pk_hier["very bad (k >= 1.0)"]

    if bad_k_simple > 0 or bad_k_hier > 0:
        print(
            "\n  WARNING: Some Pareto-k values are problematic."
            "\n  The LOO estimates may be unreliable for affected observations."
            "\n  Consider using az.reloo() or moment-matching for those points."
        )

    print("\n" + "=" * 70)
    print("  COMPARISON COMPLETE — see saved plots in output directory")
    print("=" * 70)

    return {
        "loo_table": df_loo_final,
        "waic_table": df_waic,
        "pareto_k_simple": pk_simple,
        "pareto_k_hierarchical": pk_hier,
        "elpd_diff_pointwise": elpd_diff,
        "stacking_weights": weights,
        "best_model": best_model,
        "elpd_diff_total": elpd_diff_total,
        "se_diff": se_diff,
    }


# ---------------------------------------------------------------------------
# Entry point (example usage)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Replace these with your actual InferenceData objects:
    #
    #   import pymc as pm
    #
    #   with simple_model:
    #       idata_simple = pm.sample(...)
    #       pm.sample_posterior_predictive(idata_simple, extend_inferencedata=True)
    #
    #   with hierarchical_model:
    #       idata_hierarchical = pm.sample(...)
    #       pm.sample_posterior_predictive(idata_hierarchical, extend_inferencedata=True)
    #
    # IMPORTANT: Both InferenceData objects must contain:
    #   - posterior (from pm.sample)
    #   - log_likelihood (add log_likelihood=True to pm.sample, or use
    #     pm.compute_log_likelihood)
    #   - posterior_predictive (from pm.sample_posterior_predictive)
    #   - observed_data (automatically included by PyMC)

    print("Load your idata_simple and idata_hierarchical, then call:")
    print("  results = run_full_comparison(idata_simple, idata_hierarchical)")
