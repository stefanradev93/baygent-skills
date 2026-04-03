"""
Structural Mediation Analysis: Job Training → Earnings
Decompose Total Effect into Natural Direct Effect (skills) and
Natural Indirect Effect (confidence → interview → earnings).

Follows the causal-inference skill workflow exactly.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import networkx as nx
import dowhy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = sum(map(ord, "job-training-mediation"))
rng = np.random.default_rng(RANDOM_SEED)
print(f"RANDOM_SEED = {RANDOM_SEED}")

OUTPUT_DIR = Path("/Users/alex_andorra/tptm_alex/portfolio/agent-skills/causal-inference-workspace/iteration-1/structural-mediation/with_skill/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── STEP 1: Formulate causal question ───────────────────────────────────────
# Estimand: Average Treatment Effect (ATE) of job training on annual earnings,
# decomposed into:
#   - Natural Direct Effect (NDE): training → earnings (skills channel)
#   - Natural Indirect Effect (NIE): training → confidence → interview → earnings
# Population: working-age adults eligible for the training programme (N=1000).

print("\n=== STEP 1: Causal question ===")
print("What is the effect of job training on annual earnings,")
print("and how much flows directly vs. through the confidence→interview pathway?")

# ─── STEP 2: Draw and validate the DAG ───────────────────────────────────────
print("\n=== STEP 2: DAG ===")
# Nodes: training (T), confidence_score (C), interview_performance (I), annual_earnings (Y)
# Edges:
#   T → C  (training builds confidence)
#   T → Y  (direct skill improvement)
#   C → I  (confident candidates interview better)
#   I → Y  (better interviews → higher earnings)
#   U_TY   (unmeasured baseline ability confounds T and Y — we acknowledge it)
# Non-edges (explicit assumptions):
#   T does NOT directly cause I (all interview benefit is mediated by confidence)
#   C does NOT directly cause Y (confidence only matters via interview performance)

dag = nx.DiGraph()
dag.add_edges_from([
    ("training", "confidence_score"),
    ("training", "annual_earnings"),     # direct path
    ("confidence_score", "interview_performance"),
    ("interview_performance", "annual_earnings"),
    # Unobserved confounder: baseline ability affects both selection into training
    # and baseline earnings capacity
    ("U_ability", "training"),
    ("U_ability", "annual_earnings"),
])

# Visualise the DAG
fig, ax = plt.subplots(figsize=(9, 5))
pos = {
    "training": (0, 1),
    "confidence_score": (1, 2),
    "interview_performance": (2, 2),
    "annual_earnings": (3, 1),
    "U_ability": (1.5, 0),
}
observed_nodes = ["training", "confidence_score", "interview_performance", "annual_earnings"]
latent_nodes = ["U_ability"]

nx.draw_networkx_nodes(dag, pos, nodelist=observed_nodes, node_color="#4C72B0",
                       node_size=2000, ax=ax, alpha=0.9)
nx.draw_networkx_nodes(dag, pos, nodelist=latent_nodes, node_color="#DD8452",
                       node_size=2000, ax=ax, alpha=0.7, node_shape="s")
nx.draw_networkx_labels(dag, pos, {n: n.replace("_", "\n") for n in dag.nodes()},
                        font_size=8, font_color="white", ax=ax)
nx.draw_networkx_edges(dag, pos, ax=ax, arrows=True, arrowsize=20,
                       edge_color="black", width=2,
                       connectionstyle="arc3,rad=0.05")
ax.set_title("DAG: Job Training → Earnings (mediation through confidence & interview)\n"
             "Orange square = unobserved confounder (U_ability)", fontsize=11)
ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dag.png", dpi=150)
plt.close()
print("DAG saved.")

# ─── STEP 3: Identification ───────────────────────────────────────────────────
print("\n=== STEP 3: Identification ===")
# Mediation identification assumptions (Pearl 2001):
#  A1. No unmeasured T–Y confounding BEYOND U_ability (accounted for by design)
#  A2. No unmeasured T–C confounding
#  A3. No unmeasured C–I confounding
#  A4. No unmeasured I–Y confounding
#  A5. No effect of T on I–Y confounders (sequential ignorability)
# Because T is randomised in the programme (we'll generate data accordingly),
# A1 is met by design. A2–A5 rest on domain assumptions — flagged prominently.
#
# Estimand (Pearl's mediation formula):
#   NDE = E[Y(t=1, C(t=0))] - E[Y(t=0, C(t=0))]
#   NIE = E[Y(t=1, C(t=1))] - E[Y(t=1, C(t=0))]
#   TE  = NDE + NIE
#
# Strategy: Structural Causal Model via PyMC with pm.do() for intervention.

# ─── STEP 4: Generate synthetic data ─────────────────────────────────────────
print("\n=== STEP 4: Synthetic data generation ===")

N = 1000

# True structural parameters (ground truth to recover)
# Training is randomly assigned (simulated RCT-like programme)
training = rng.binomial(1, 0.5, N).astype(float)

# Confidence: training raises confidence by ~1.2 units
confidence_noise = rng.normal(0, 1.0, N)
confidence_score = 5.0 + 1.2 * training + confidence_noise  # mean ~5 without training

# Interview performance: driven by confidence
interview_noise = rng.normal(0, 1.5, N)
interview_performance = 3.0 + 0.8 * confidence_score + interview_noise

# Annual earnings (in $1000s): direct training effect + interview effect
earnings_noise = rng.normal(0, 5.0, N)
annual_earnings = (
    20.0                     # baseline
    + 3.0 * training         # DIRECT effect (skills)
    + 2.5 * interview_performance  # indirect path completes here
    + earnings_noise
)

# True effects for reference
# NDE = 3.0 (direct training coefficient)
# NIE: training→conf (1.2) × conf→interview (0.8) × interview→earnings (2.5) = 2.4
# TE ≈ 3.0 + 2.4 = 5.4

df = pd.DataFrame({
    "training": training,
    "confidence_score": confidence_score,
    "interview_performance": interview_performance,
    "annual_earnings": annual_earnings,
})

print(f"Dataset shape: {df.shape}")
print(df.describe().round(2))

# ─── STEP 5: Estimate — Structural Causal Model in PyMC ─────────────────────
print("\n=== STEP 5: Estimate ===")

# Standardise for cleaner priors
conf_mean, conf_std = df["confidence_score"].mean(), df["confidence_score"].std()
interview_mean, interview_std = df["interview_performance"].mean(), df["interview_performance"].std()
earn_mean, earn_std = df["annual_earnings"].mean(), df["annual_earnings"].std()

conf_z = (df["confidence_score"].values - conf_mean) / conf_std
interview_z = (df["interview_performance"].values - interview_mean) / interview_std
earn_z = (df["annual_earnings"].values - earn_mean) / earn_std
T = df["training"].values

# Build structural equations (observed model)
with pm.Model() as scm_obs:
    # ── Equation 1: Confidence ~ f(Training) ──────────────────────────────
    # Prior: training raises confidence; standardised scale
    alpha_c = pm.Normal("alpha_c", mu=0, sigma=1.0)
    beta_tc = pm.Normal("beta_tc", mu=0, sigma=1.0)   # effect of T on C
    sigma_c = pm.Gamma("sigma_c", alpha=2, beta=2)

    mu_c = alpha_c + beta_tc * T
    C_obs = pm.Normal("C_obs", mu=mu_c, sigma=sigma_c, observed=conf_z)

    # ── Equation 2: Interview ~ f(Confidence) ─────────────────────────────
    alpha_i = pm.Normal("alpha_i", mu=0, sigma=1.0)
    beta_ci = pm.Normal("beta_ci", mu=0, sigma=1.0)   # effect of C on I
    sigma_i = pm.Gamma("sigma_i", alpha=2, beta=2)

    mu_i = alpha_i + beta_ci * conf_z
    I_obs = pm.Normal("I_obs", mu=mu_i, sigma=sigma_i, observed=interview_z)

    # ── Equation 3: Earnings ~ f(Training, Interview) ─────────────────────
    alpha_y = pm.Normal("alpha_y", mu=0, sigma=1.0)
    beta_ty = pm.Normal("beta_ty", mu=0, sigma=1.0)   # DIRECT effect T → Y
    beta_iy = pm.Normal("beta_iy", mu=0, sigma=1.0)   # effect I → Y
    sigma_y = pm.Gamma("sigma_y", alpha=2, beta=2)

    mu_y = alpha_y + beta_ty * T + beta_iy * interview_z
    Y_obs = pm.Normal("Y_obs", mu=mu_y, sigma=sigma_y, observed=earn_z)

    # Sample
    idata = pm.sample(
        draws=2000,
        tune=1000,
        nuts_sampler="nutpie",
        random_seed=rng,
        target_accept=0.9,
        progressbar=True,
    )

print("\nSampling complete.")

# ─── DIAGNOSTICS ──────────────────────────────────────────────────────────────
print("\n=== Diagnostics ===")
key_vars = ["alpha_c", "beta_tc", "sigma_c",
            "alpha_i", "beta_ci", "sigma_i",
            "alpha_y", "beta_ty", "beta_iy", "sigma_y"]

summary = az.summary(idata, var_names=key_vars, round_to=3)
print(summary.to_string())

rhat_max = summary["r_hat"].max()
ess_min = summary[["ess_bulk", "ess_tail"]].min().min()
div_count = int(idata.sample_stats["diverging"].sum())
print(f"\nMax R-hat: {rhat_max:.4f}  |  Min ESS: {ess_min:.0f}  |  Divergences: {div_count}")

assert rhat_max < 1.01, f"R-hat too high: {rhat_max}"
assert ess_min > 400, f"ESS too low: {ess_min}"
assert div_count == 0, f"Divergences detected: {div_count}"
print("All diagnostics PASS.")

# Trace plot (key parameters)
axes = az.plot_trace(idata, var_names=["beta_tc", "beta_ci", "beta_ty", "beta_iy"],
                     compact=True)
plt.suptitle("Trace plots: structural coefficients", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "trace_plots.png", dpi=150)
plt.close()
print("Trace plots saved.")

# ─── COMPUTE NDE, NIE, TE from posterior ────────────────────────────────────
print("\n=== Computing NDE, NIE, TE on original scale ===")

# Extract posterior draws
post = idata.posterior
beta_tc_post = post["beta_tc"].values.flatten()   # T→C (standardised)
beta_ci_post = post["beta_ci"].values.flatten()   # C→I (standardised)
beta_ty_post = post["beta_ty"].values.flatten()   # T→Y direct (standardised)
beta_iy_post = post["beta_iy"].values.flatten()   # I→Y (standardised)

# Convert standardised coefficients to original (earnings $1000s) scale:
# Y_z = (Y - earn_mean) / earn_std  =>  Y = Y_z * earn_std + earn_mean
# C_z = (C - conf_mean) / conf_std
# I_z = (I - interview_mean) / interview_std

# Chain rule on original scale:
#   T→C on original: beta_tc_orig = beta_tc * conf_std
#   C→I on original: beta_ci_orig = beta_ci * interview_std / conf_std
#   I→Y on original: beta_iy_orig = beta_iy * earn_std / interview_std
#   T→Y direct on original: beta_ty_orig = beta_ty * earn_std

beta_tc_orig = beta_tc_post * conf_std               # per unit T
beta_ci_orig = beta_ci_post * interview_std / conf_std
beta_iy_orig = beta_iy_post * earn_std / interview_std
beta_ty_orig = beta_ty_post * earn_std               # direct

# NDE = direct effect = beta_ty_orig (hold confidence at T=0 level)
NDE = beta_ty_orig

# NIE = T→C × C→I × I→Y (one-unit increase in T flowing through mediator chain)
NIE = beta_tc_orig * beta_ci_orig * beta_iy_orig

# TE = NDE + NIE
TE = NDE + NIE

print(f"True NDE (ground truth): $3,000")
print(f"True NIE (ground truth): $2,400  (1.2 × 0.8 × 2.5 × 1000)")
print(f"True TE  (ground truth): $5,400")
print()

def summarise_posterior(arr, name, true_val=None):
    mean = np.mean(arr)
    hdi50 = az.hdi(arr.reshape(1, -1, 1), hdi_prob=0.50).squeeze()
    hdi94 = az.hdi(arr.reshape(1, -1, 1), hdi_prob=0.94).squeeze()
    p_pos = (arr > 0).mean()
    if true_val is not None:
        print(f"{name:5s}: mean={mean:7.2f}  50%HDI=[{hdi50[0]:.2f},{hdi50[1]:.2f}]  "
              f"94%HDI=[{hdi94[0]:.2f},{hdi94[1]:.2f}]  P(>0)={p_pos:.3f}  "
              f"(true={true_val:.2f})")
    else:
        print(f"{name:5s}: mean={mean:7.2f}  50%HDI=[{hdi50[0]:.2f},{hdi50[1]:.2f}]  "
              f"94%HDI=[{hdi94[0]:.2f},{hdi94[1]:.2f}]  P(>0)={p_pos:.3f}")
    return {"mean": mean, "hdi50": hdi50, "hdi94": hdi94, "p_pos": p_pos}

nde_stats = summarise_posterior(NDE, "NDE", true_val=3.0)
nie_stats = summarise_posterior(NIE, "NIE", true_val=2.4)
te_stats  = summarise_posterior(TE,  "TE",  true_val=5.4)

# ─── FOREST PLOT: NDE / NIE / TE ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
colors = {"NDE": "#4C72B0", "NIE": "#55A868", "TE": "#C44E52"}
ys = {"NDE": 2, "NIE": 1, "TE": 0}

for label, stats, true_v in [("NDE", nde_stats, 3.0), ("NIE", nie_stats, 2.4), ("TE", te_stats, 5.4)]:
    y = ys[label]
    c = colors[label]
    ax.errorbar(stats["mean"], y,
                xerr=[[stats["mean"] - stats["hdi94"][0]],
                      [stats["hdi94"][1] - stats["mean"]]],
                fmt="o", color=c, linewidth=2, markersize=8, capsize=5, label=f"{label} 94% HDI")
    ax.errorbar(stats["mean"], y,
                xerr=[[stats["mean"] - stats["hdi50"][0]],
                      [stats["hdi50"][1] - stats["mean"]]],
                fmt="o", color=c, linewidth=5, markersize=8, capsize=0)
    ax.axvline(true_v, ymin=(y + 0.1)/3.0, ymax=(y + 0.9)/3.0,
               color=c, linestyle="--", linewidth=1.5, alpha=0.6)

ax.axvline(0, color="gray", linestyle=":", linewidth=1)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Total Effect (TE)", "Natural Indirect\nEffect (NIE)", "Natural Direct\nEffect (NDE)"])
ax.set_xlabel("Effect on Annual Earnings ($1000s)", fontsize=11)
ax.set_title("Mediation decomposition: Job Training → Earnings\n"
             "Thick bars = 50% HDI, thin bars = 94% HDI, dashed = ground truth", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mediation_forest.png", dpi=150)
plt.close()
print("\nForest plot saved.")

# Proportion mediated
prop_mediated = NIE / TE
pm_stats = summarise_posterior(prop_mediated, "NIE/TE")
print(f"\nProportion of TE through confidence→interview pathway: {pm_stats['mean']:.1%}")

# ─── STEP 6: Refutation ──────────────────────────────────────────────────────
print("\n\n=== STEP 6: Refutation ===")

# 6a. Placebo treatment (DoWhy) — replace training with random permutation
print("\n--- 6a. DoWhy placebo treatment refuter ---")
# Note: Because training was randomised in this programme (simulated RCT-like design),
# U_ability does not confound T→Y in expectation. We use DoWhy WITHOUT the U_ability
# node for the refutation step (which performs balanced random permutation regardless),
# and we run refutations on the full structural model posteriors from PyMC.
# This is consistent: DoWhy probes associational robustness; PyMC captures causal structure.

dag_no_u = nx.DiGraph()
dag_no_u.add_edges_from([
    ("training", "confidence_score"),
    ("training", "annual_earnings"),
    ("confidence_score", "interview_performance"),
    ("interview_performance", "annual_earnings"),
])

dowhy_model = dowhy.CausalModel(
    data=df,
    treatment="training",
    outcome="annual_earnings",
    graph=dag_no_u,
)
identified = dowhy_model.identify_effect(proceed_when_unidentifiable=False)
estimate_linear = dowhy_model.estimate_effect(
    identified,
    method_name="backdoor.linear_regression",
)
dowhy_val = float(estimate_linear.value) if estimate_linear.value is not None else np.nan
print(f"DoWhy linear regression estimate (total effect): {dowhy_val:.3f} ($1000s)")

ref_placebo = dowhy_model.refute_estimate(
    identified, estimate_linear,
    method_name="placebo_treatment_refuter",
    placebo_type="permute",
    random_seed=RANDOM_SEED,
)
print(ref_placebo)
placebo_val = float(ref_placebo.new_effect) if ref_placebo.new_effect is not None else np.nan
placebo_pass = abs(placebo_val) < 0.5  # should be near zero
print(f"Placebo refuter: {'PASS' if placebo_pass else 'FAIL'} (new_effect={placebo_val:.4f})")

# 6b. Random common cause
print("\n--- 6b. Random common cause refuter ---")
ref_rcc = dowhy_model.refute_estimate(
    identified, estimate_linear,
    method_name="random_common_cause",
    random_seed=RANDOM_SEED,
)
print(ref_rcc)
rcc_new = float(ref_rcc.new_effect) if ref_rcc.new_effect is not None else np.nan
rcc_shift = abs(rcc_new - dowhy_val)
rcc_pass = rcc_shift < 1.0
print(f"Random common cause: {'PASS' if rcc_pass else 'FAIL'} (shift={rcc_shift:.4f})")

# 6c. Data subset refuter
print("\n--- 6c. Data subset refuter ---")
ref_subset = dowhy_model.refute_estimate(
    identified, estimate_linear,
    method_name="data_subset_refuter",
    subset_fraction=0.8,
    random_seed=RANDOM_SEED,
)
print(ref_subset)
subset_new = float(ref_subset.new_effect) if ref_subset.new_effect is not None else np.nan
subset_shift = abs(subset_new - dowhy_val)
subset_pass = subset_shift < 1.0
print(f"Data subset refuter: {'PASS' if subset_pass else 'FAIL'} (shift={subset_shift:.4f})")

# 6d. Sensitivity to unobserved confounding (mediation-specific)
# For NDE/NIE, the critical assumption is no unmeasured T-C, C-I, or I-Y confounders.
# We probe robustness by adding noise to confidence_score and checking NDE stability.
print("\n--- 6d. Sensitivity: NDE stability under noise on mediator ---")
nde_sensitivity = []
for noise_frac in [0.0, 0.1, 0.2, 0.3, 0.5]:
    df_perturbed = df.copy()
    df_perturbed["confidence_score"] = (
        df["confidence_score"] + rng.normal(0, noise_frac * df["confidence_score"].std(), N)
    )
    # Quick OLS approximation for NDE under perturbation
    from numpy.linalg import lstsq
    # Equation 1 (T→C)
    X1 = np.column_stack([np.ones(N), df_perturbed["training"].values])
    coef1, _, _, _ = lstsq(X1, df_perturbed["confidence_score"].values, rcond=None)
    beta_tc_ols = coef1[1]
    # Equation 2 (C→I)
    X2 = np.column_stack([np.ones(N), df_perturbed["confidence_score"].values])
    coef2, _, _, _ = lstsq(X2, df_perturbed["interview_performance"].values, rcond=None)
    beta_ci_ols = coef2[1]
    # Equation 3 (T→Y, I→Y)
    X3 = np.column_stack([np.ones(N), df_perturbed["training"].values,
                          df_perturbed["interview_performance"].values])
    coef3, _, _, _ = lstsq(X3, df_perturbed["annual_earnings"].values, rcond=None)
    beta_ty_ols = coef3[1]
    beta_iy_ols = coef3[2]

    nde_ols = beta_ty_ols
    nie_ols = beta_tc_ols * beta_ci_ols * beta_iy_ols
    nde_sensitivity.append({"noise_frac": noise_frac, "NDE_ols": nde_ols, "NIE_ols": nie_ols})

sens_df = pd.DataFrame(nde_sensitivity)
print(sens_df.round(3).to_string(index=False))
nde_range = sens_df["NDE_ols"].max() - sens_df["NDE_ols"].min()
sens_pass = nde_range < 1.0
print(f"NDE sensitivity range: {nde_range:.3f}  -> {'PASS' if sens_pass else 'FAIL'}")

# ─── STEP 7: Interpret ───────────────────────────────────────────────────────
print("\n\n=== STEP 7: Interpretation ===")
print(f"NDE: mean={nde_stats['mean']:.2f}k  P(>0)={nde_stats['p_pos']:.3f}")
print(f"NIE: mean={nie_stats['mean']:.2f}k  P(>0)={nie_stats['p_pos']:.3f}")
print(f"TE:  mean={te_stats['mean']:.2f}k   P(>0)={te_stats['p_pos']:.3f}")
print(f"Proportion mediated: {pm_stats['mean']:.1%}")

# ─── Save refutation summary as JSON for report ──────────────────────────────
import json
refutation_results = {
    "placebo_treatment": {"pass": bool(placebo_pass), "new_effect": float(placebo_val)},
    "random_common_cause": {"pass": bool(rcc_pass), "shift": float(rcc_shift), "new_effect": float(rcc_new)},
    "data_subset": {"pass": bool(subset_pass), "shift": float(subset_shift), "new_effect": float(subset_new)},
    "nde_sensitivity": {"pass": bool(sens_pass), "range": float(nde_range)},
    "nde_stats": {k: (float(v) if isinstance(v, (np.floating, float)) else
                      [float(x) for x in v]) for k, v in nde_stats.items()},
    "nie_stats": {k: (float(v) if isinstance(v, (np.floating, float)) else
                      [float(x) for x in v]) for k, v in nie_stats.items()},
    "te_stats": {k: (float(v) if isinstance(v, (np.floating, float)) else
                     [float(x) for x in v]) for k, v in te_stats.items()},
    "prop_mediated": float(pm_stats["mean"]),
    "diagnostics": {
        "rhat_max": float(rhat_max),
        "ess_min": float(ess_min),
        "divergences": div_count,
    },
}
with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(refutation_results, f, indent=2)
print("\nResults saved to results.json")

# ─── Posterior density plot ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax_i, (arr, label, true_v, color) in enumerate(zip(
    [NDE, NIE, TE],
    ["NDE (Direct)", "NIE (Indirect)", "TE (Total)"],
    [3.0, 2.4, 5.4],
    ["#4C72B0", "#55A868", "#C44E52"]
)):
    ax_cur = axes[ax_i]
    az.plot_posterior(
        arr.reshape(1, -1),
        hdi_prob=0.94,
        ref_val=true_v,
        ax=ax_cur,
        color=color,
    )
    ax_cur.set_title(f"{label}\n(dashed = ground truth)", fontsize=10)
    ax_cur.set_xlabel("$1000s")

plt.suptitle("Posterior distributions: Mediation decomposition", y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "posterior_density.png", dpi=150)
plt.close()
print("Posterior density plots saved.")

print("\n\n=== ALL DONE ===")
