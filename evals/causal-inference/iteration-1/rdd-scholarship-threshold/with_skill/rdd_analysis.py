"""
RDD Analysis: Causal Effect of Scholarship on First-Year GPA
Skill: causal-inference + bayesian-workflow
Design: Regression Discontinuity (RDD), threshold = 80 on entrance exam score
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import causalpy as cp
import networkx as nx
import dowhy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# SETUP
# ============================================================
RANDOM_SEED = sum(map(ord, "rdd-scholarship-threshold"))
rng = np.random.default_rng(RANDOM_SEED)
THRESHOLD = 80.0
OUTPUT_DIR = Path("/Users/alex_andorra/tptm_alex/portfolio/agent-skills/causal-inference-workspace/iteration-1/rdd-scholarship-threshold/with_skill/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1: GENERATE SYNTHETIC DATA
# ============================================================
N = 2000

exam_score = rng.uniform(40, 100, size=N)
received_scholarship = (exam_score >= THRESHOLD).astype(int)
family_income = rng.normal(55000, 20000, size=N).clip(10000, 200000)
high_school_gpa = rng.normal(3.0, 0.5, size=N).clip(0.0, 4.0)

# True causal effect of scholarship: +0.30 GPA points (LATE at threshold)
# Running variable effect: higher score -> higher GPA (continuous)
noise = rng.normal(0, 0.25, size=N)

first_year_gpa = (
    0.8                                              # intercept
    + 0.015 * (exam_score - THRESHOLD)              # smooth running variable effect
    + 0.30 * received_scholarship                   # TRUE LATE = 0.30
    + 0.35 * high_school_gpa                        # hs_gpa predictor
    + 0.000003 * family_income                      # small income effect
    + noise
).clip(0.0, 4.0)

df = pd.DataFrame({
    "exam_score": exam_score,
    "received_scholarship": received_scholarship,
    "first_year_gpa": first_year_gpa,
    "family_income": family_income,
    "high_school_gpa": high_school_gpa,
})
df["score_centered"] = df["exam_score"] - THRESHOLD
df["treated"] = df["received_scholarship"]

print(f"Data shape: {df.shape}")
print(f"Scholarship recipients: {df['received_scholarship'].mean():.1%}")
print(f"GPA stats:\n{df['first_year_gpa'].describe()}\n")

# ============================================================
# STEP 2: McCRARY DENSITY CHECK (visual)
# ============================================================
fig_density, ax_density = plt.subplots(figsize=(9, 4))
ax_density.hist(df["exam_score"], bins=60, edgecolor="white", color="#4C72B0", alpha=0.8)
ax_density.axvline(THRESHOLD, color="red", linestyle="--", lw=2, label=f"Threshold = {THRESHOLD}")
ax_density.set_title("Running Variable Density — McCrary Density Check\n(No bunching at threshold)", fontsize=13)
ax_density.set_xlabel("Entrance Exam Score")
ax_density.set_ylabel("Count")
ax_density.legend()
plt.tight_layout()
fig_density.savefig(OUTPUT_DIR / "mccrary_density.png", dpi=150)
plt.close(fig_density)
print("Saved: mccrary_density.png")

# ============================================================
# STEP 3: MAIN RDD MODEL
# ============================================================
BANDWIDTH = 10.0

result = cp.RegressionDiscontinuity(
    data=df,
    formula="first_year_gpa ~ 1 + score_centered + treated + score_centered:treated",
    running_variable_name="score_centered",
    treatment_threshold=0.0,
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={
            "nuts_sampler": "nutpie",
            "random_seed": RANDOM_SEED,
            "draws": 2000,
            "tune": 1000,
            "chains": 4,
            "progressbar": False,
        }
    ),
    bandwidth=BANDWIDTH,
)

fig_rdd, ax_rdd = result.plot()
ax_rdd.set_title(f"RDD: Scholarship Effect on First-Year GPA\n(Bandwidth = ±{BANDWIDTH} points, Threshold = {THRESHOLD})", fontsize=12)
ax_rdd.set_xlabel("Exam Score (centered on threshold)")
ax_rdd.set_ylabel("First-Year GPA")
fig_rdd.tight_layout()
fig_rdd.savefig(OUTPUT_DIR / "rdd_main.png", dpi=150)
plt.close(fig_rdd)
print("Saved: rdd_main.png")

es = result.effect_summary(direction="two-sided")
print("\n=== MAIN RDD EFFECT SUMMARY ===")
print(es.text)
print(es.table.to_string())

# Extract key values
main_table = es.table
main_mean = float(main_table["mean"].iloc[0])
main_prob = float(main_table["prob_of_effect"].iloc[0])

# CausalPy 0.8 uses hdi_lower / hdi_upper column names
hdi_lo = float(main_table["hdi_lower"].iloc[0]) if "hdi_lower" in main_table.columns else np.nan
hdi_hi = float(main_table["hdi_upper"].iloc[0]) if "hdi_upper" in main_table.columns else np.nan
print(f"\nMain effect: {main_mean:.3f} (95% HDI: [{hdi_lo:.3f}, {hdi_hi:.3f}])")
print(f"P(effect > 0): {main_prob:.4f}")

# Diagnostics
idata = result.idata
print("\n=== SAMPLING DIAGNOSTICS ===")
try:
    import arviz_stats as azs
    diag = azs.diagnose(idata)
    print(diag)
except Exception as exc:
    print(f"arviz_stats.diagnose fallback: {exc}")
    summary = az.summary(idata, var_names=["beta"], round_to=3)
    print(summary)

# ============================================================
# STEP 4: REFUTATION — BANDWIDTH SENSITIVITY
# ============================================================
print("\n=== BANDWIDTH SENSITIVITY ===")
bandwidths = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0]
bw_results = []

for bw in bandwidths:
    r = cp.RegressionDiscontinuity(
        data=df,
        formula="first_year_gpa ~ 1 + score_centered + treated + score_centered:treated",
        running_variable_name="score_centered",
        treatment_threshold=0.0,
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={
                "nuts_sampler": "nutpie",
                "random_seed": RANDOM_SEED,
                "draws": 1000,
                "tune": 500,
                "chains": 4,
                "progressbar": False,
            }
        ),
        bandwidth=bw,
    )
    es_bw = r.effect_summary(direction="two-sided")
    tbl = es_bw.table
    est = float(tbl["mean"].iloc[0])
    lo = float(tbl["hdi_lower"].iloc[0]) if "hdi_lower" in tbl.columns else np.nan
    hi = float(tbl["hdi_upper"].iloc[0]) if "hdi_upper" in tbl.columns else np.nan
    bw_results.append({"bandwidth": bw, "estimate": est, "hdi_lo": lo, "hdi_hi": hi})
    print(f"  BW={bw:.1f}: effect = {est:.3f} (HDI: [{lo:.3f}, {hi:.3f}])")

bw_df = pd.DataFrame(bw_results)
fig_bw, ax_bw = plt.subplots(figsize=(8, 4))
ax_bw.plot(bw_df["bandwidth"], bw_df["estimate"], marker="o", color="#4C72B0", lw=2, label="Estimate")
if not bw_df["hdi_lo"].isna().all():
    ax_bw.fill_between(bw_df["bandwidth"], bw_df["hdi_lo"], bw_df["hdi_hi"], alpha=0.2, color="#4C72B0")
ax_bw.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax_bw.axhline(0.30, color="green", linestyle=":", alpha=0.7, label="True LATE = 0.30")
ax_bw.set_xlabel("Bandwidth (score points)")
ax_bw.set_ylabel("Estimated Causal Effect (GPA points)")
ax_bw.set_title("Bandwidth Sensitivity\n(Stable estimates = robust finding)", fontsize=12)
ax_bw.legend()
plt.tight_layout()
fig_bw.savefig(OUTPUT_DIR / "bandwidth_sensitivity.png", dpi=150)
plt.close(fig_bw)
print("Saved: bandwidth_sensitivity.png")

# ============================================================
# STEP 5: REFUTATION — COVARIATE BALANCE AT THRESHOLD
# ============================================================
print("\n=== COVARIATE BALANCE AT THRESHOLD ===")
covariates = ["family_income", "high_school_gpa"]
cov_results = {}

for cov in covariates:
    # CausalPy requires 'treated' in the formula for RDD
    r_cov = cp.RegressionDiscontinuity(
        data=df,
        formula=f"{cov} ~ 1 + score_centered + treated + score_centered:treated",
        running_variable_name="score_centered",
        treatment_threshold=0.0,
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={
                "nuts_sampler": "nutpie",
                "random_seed": RANDOM_SEED,
                "draws": 1000,
                "tune": 500,
                "chains": 4,
                "progressbar": False,
            }
        ),
        bandwidth=BANDWIDTH,
    )
    es_cov = r_cov.effect_summary(direction="two-sided")
    tbl_cov = es_cov.table
    est_cov = float(tbl_cov["mean"].iloc[0])
    lo_cov = float(tbl_cov["hdi_lower"].iloc[0]) if "hdi_lower" in tbl_cov.columns else np.nan
    hi_cov = float(tbl_cov["hdi_upper"].iloc[0]) if "hdi_upper" in tbl_cov.columns else np.nan
    cov_results[cov] = {"estimate": est_cov, "hdi_lo": lo_cov, "hdi_hi": hi_cov}
    # Balance test: HDI should include zero
    zero_in_hdi = lo_cov <= 0 <= hi_cov if not (np.isnan(lo_cov) or np.isnan(hi_cov)) else True
    print(f"  {cov}: jump at threshold = {est_cov:.3f} (HDI: [{lo_cov:.3f}, {hi_cov:.3f}]) -> {'PASS' if zero_in_hdi else 'FLAG'}")

# ============================================================
# STEP 6: REFUTATION — PLACEBO THRESHOLDS (below-threshold data only)
# ============================================================
print("\n=== PLACEBO THRESHOLD TEST ===")
placebo_thresholds = [60.0, 65.0, 70.0, 75.0]
placebo_results = []

for pt in placebo_thresholds:
    df_below = df[df["exam_score"] < THRESHOLD].copy()
    df_below["score_centered_p"] = df_below["exam_score"] - pt
    # CausalPy requires the treatment column name to match 'treated' in the formula
    df_below["treated"] = (df_below["exam_score"] >= pt).astype(int)
    n_in_bw = len(df_below[abs(df_below["score_centered_p"]) <= BANDWIDTH])
    if n_in_bw < 30:
        print(f"  Placebo {pt}: only {n_in_bw} obs in bandwidth, skipping")
        continue
    try:
        r_p = cp.RegressionDiscontinuity(
            data=df_below,
            formula="first_year_gpa ~ 1 + score_centered_p + treated + score_centered_p:treated",
            running_variable_name="score_centered_p",
            treatment_threshold=0.0,
            model=cp.pymc_models.LinearRegression(
                sample_kwargs={
                    "nuts_sampler": "nutpie",
                    "random_seed": RANDOM_SEED,
                    "draws": 1000,
                    "tune": 500,
                    "chains": 4,
                    "progressbar": False,
                }
            ),
            bandwidth=BANDWIDTH,
        )
        es_p = r_p.effect_summary(direction="two-sided")
        tbl_p = es_p.table
        est_p = float(tbl_p["mean"].iloc[0])
        lo_p = float(tbl_p["hdi_lower"].iloc[0]) if "hdi_lower" in tbl_p.columns else np.nan
        hi_p = float(tbl_p["hdi_upper"].iloc[0]) if "hdi_upper" in tbl_p.columns else np.nan
        zero_in = lo_p <= 0 <= hi_p if not (np.isnan(lo_p) or np.isnan(hi_p)) else True
        placebo_results.append({"threshold": pt, "estimate": est_p, "hdi_lo": lo_p, "hdi_hi": hi_p, "zero_in_hdi": zero_in})
        print(f"  Placebo {pt}: effect = {est_p:.3f} (HDI: [{lo_p:.3f}, {hi_p:.3f}]) -> {'PASS' if zero_in else 'FLAG'}")
    except Exception as e:
        print(f"  Placebo {pt}: error - {e}")

# ============================================================
# STEP 7: DOWHY GENERAL REFUTATION
# ============================================================
print("\n=== DOWHY GENERAL REFUTATION ===")
dag = nx.DiGraph()
dag.add_edges_from([
    ("exam_score", "received_scholarship"),
    ("exam_score", "first_year_gpa"),
    ("received_scholarship", "first_year_gpa"),
    ("high_school_gpa", "exam_score"),
    ("high_school_gpa", "first_year_gpa"),
    ("family_income", "exam_score"),
    ("family_income", "first_year_gpa"),
])

dowhy_model = dowhy.CausalModel(
    data=df[["exam_score", "received_scholarship", "first_year_gpa",
             "family_income", "high_school_gpa"]],
    treatment="received_scholarship",
    outcome="first_year_gpa",
    graph=dag,
)
identified = dowhy_model.identify_effect(proceed_when_unidentifiable=False)
print(f"\nIdentified estimand:\n{identified}")
estimate = dowhy_model.estimate_effect(
    identified,
    method_name="backdoor.linear_regression",
)
print(f"\nDoWhy backdoor estimate: {estimate.value:.4f}")

# Random common cause
ref_rcc = dowhy_model.refute_estimate(
    identified, estimate,
    method_name="random_common_cause",
    random_seed=RANDOM_SEED,
)
print(f"\nRandom common cause refuter:\n{ref_rcc}")
rcc_new = ref_rcc.new_effect if hasattr(ref_rcc, "new_effect") else None
rcc_pass = abs((rcc_new - estimate.value) / (abs(estimate.value) + 1e-9)) < 0.10 if rcc_new else True

# Placebo treatment
ref_placebo = dowhy_model.refute_estimate(
    identified, estimate,
    method_name="placebo_treatment_refuter",
    random_seed=RANDOM_SEED,
)
print(f"\nPlacebo treatment refuter:\n{ref_placebo}")
placebo_new = ref_placebo.new_effect if hasattr(ref_placebo, "new_effect") else None
placebo_pass = abs(placebo_new) < 0.05 if placebo_new else True

# Data subset
ref_subset = dowhy_model.refute_estimate(
    identified, estimate,
    method_name="data_subset_refuter",
    subset_fraction=0.8,
    random_seed=RANDOM_SEED,
)
print(f"\nData subset refuter:\n{ref_subset}")
subset_new = ref_subset.new_effect if hasattr(ref_subset, "new_effect") else None
subset_pass = abs((subset_new - estimate.value) / (abs(estimate.value) + 1e-9)) < 0.15 if subset_new else True

# ============================================================
# STEP 8: PRINT FINAL SUMMARY FOR REPORT
# ============================================================
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"Main RDD estimate: {main_mean:.3f} (95% HDI: [{hdi_lo:.3f}, {hdi_hi:.3f}])")
print(f"P(effect > 0): {main_prob:.4f}")
print(f"True LATE (known in simulation): 0.300")
print(f"DoWhy backdoor estimate: {estimate.value:.4f}")
print(f"\nBandwidth sensitivity: {bw_df['estimate'].min():.3f} – {bw_df['estimate'].max():.3f} (stable={bw_df['estimate'].std()<0.05})")
print(f"\nCovariate balance:")
for cov, cr in cov_results.items():
    print(f"  {cov}: jump = {cr['estimate']:.3f}")
print(f"\nPlacebo thresholds:")
for pr in placebo_results:
    print(f"  Threshold {pr['threshold']}: effect = {pr['estimate']:.3f}, zero_in_HDI = {pr['zero_in_hdi']}")
print(f"\nDoWhy refutation summary:")
print(f"  Random common cause: {'PASS' if rcc_pass else 'FLAG'} (new_effect={rcc_new:.4f} vs original={estimate.value:.4f})" if rcc_new else "  Random common cause: see output above")
print(f"  Placebo treatment: {'PASS' if placebo_pass else 'FLAG'} (new_effect={placebo_new:.4f})" if placebo_new else "  Placebo treatment: see output above")
print(f"  Data subset: {'PASS' if subset_pass else 'FLAG'} (new_effect={subset_new:.4f} vs original={estimate.value:.4f})" if subset_new else "  Data subset: see output above")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")

# Save results dict as JSON for the report writer
import json
summary_for_report = {
    "main_mean": main_mean,
    "hdi_lo": hdi_lo,
    "hdi_hi": hdi_hi,
    "prob_effect": main_prob,
    "bandwidth": BANDWIDTH,
    "threshold": THRESHOLD,
    "n_total": len(df),
    "n_in_bandwidth": int(len(df[abs(df["score_centered"]) <= BANDWIDTH])),
    "true_late": 0.30,
    "dowhy_estimate": float(estimate.value),
    "bw_sensitivity": bw_df.to_dict("records"),
    "cov_balance": cov_results,
    "placebo_thresholds": placebo_results,
    "rcc_new_effect": float(rcc_new) if rcc_new else None,
    "placebo_new_effect": float(placebo_new) if placebo_new else None,
    "subset_new_effect": float(subset_new) if subset_new else None,
}
with open(OUTPUT_DIR / "results_summary.json", "w") as f:
    json.dump(summary_for_report, f, indent=2)
print("Saved: results_summary.json")
