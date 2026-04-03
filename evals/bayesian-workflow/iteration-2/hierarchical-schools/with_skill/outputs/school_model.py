"""
Hierarchical (Multilevel) Model for Student Test Scores Across Schools
======================================================================

Goal: Understand how much of the variation in test scores is between schools
vs. within schools, and produce school-level estimates that account for
different sample sizes through partial pooling (shrinkage).

Generative story:
    1. A population of schools exists, each with a "true" average test score.
    2. School means are drawn from Normal(mu_global, sigma_school) -- schools
       vary around a global average.
    3. Each student's observed score within school j is drawn from
       Normal(mu_school[j], sigma_student) -- individual variation within a school.

This is a varying-intercepts model. Partial pooling lets small schools
(as few as 5 students) borrow strength from the population, shrinking their
estimates toward the global mean. Large schools retain estimates closer to
their observed sample means.

Workflow (per Bayesian Workflow skill):
    1. Formulate -- define the generative story (above)
    2. Specify priors -- with justifications
    3. Implement in PyMC -- non-centered parameterization
    4. Prior predictive checks -- verify priors produce plausible data
    5. Inference -- nutpie sampler
    6. Convergence diagnostics -- R-hat, ESS, divergences, trace/rank plots
    7. Model criticism -- PPC, calibration (plot_ppc_pit), LOO-CV
    8. Report results -- ICC, shrinkage plot, school-level estimates with HDI
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import arviz_plots as azp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Reproducible seed: derived from a descriptive name, never magic numbers.
# This makes the seed self-documenting and tied to the analysis.
# ---------------------------------------------------------------------------
RANDOM_SEED = sum(map(ord, "hierarchical-schools-test-scores-v2"))
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Generate synthetic data
# ---------------------------------------------------------------------------
# 15 schools with varying sample sizes (some as small as 5 students).
# True generative process:
#   - Global mean score ~ 65
#   - Between-school SD ~ 8 (schools genuinely differ)
#   - Within-school SD ~ 12 (individual variation within a school)

N_SCHOOLS = 15
TOTAL_STUDENTS = 200

# Assign students to schools with imbalanced sizes.
# Force some schools to have exactly 5 students to stress-test partial pooling.
school_sizes_raw = rng.integers(5, 25, size=N_SCHOOLS)
school_sizes_raw[:3] = 5  # Ensure a few very small schools
# Rescale so total = 200
school_sizes = (school_sizes_raw / school_sizes_raw.sum() * TOTAL_STUDENTS).astype(int)
diff = TOTAL_STUDENTS - school_sizes.sum()
school_sizes[-1] += diff  # Adjust rounding to hit exactly 200

# True parameters
TRUE_GLOBAL_MEAN = 65.0
TRUE_BETWEEN_SCHOOL_SD = 8.0
TRUE_WITHIN_SCHOOL_SD = 12.0

true_school_means = rng.normal(TRUE_GLOBAL_MEAN, TRUE_BETWEEN_SCHOOL_SD, size=N_SCHOOLS)

# Generate student-level data
school_ids = []
scores = []
for j in range(N_SCHOOLS):
    n_j = school_sizes[j]
    school_ids.extend([j] * n_j)
    scores.extend(rng.normal(true_school_means[j], TRUE_WITHIN_SCHOOL_SD, size=n_j))

school_ids = np.array(school_ids)
scores = np.array(scores)

df = pd.DataFrame({
    "school_id": school_ids,
    "score": scores,
})

school_names = [f"School_{j:02d}" for j in range(N_SCHOOLS)]

print("=== Data Summary ===")
print(f"Total students: {len(df)}")
print(f"Number of schools: {N_SCHOOLS}")
print(f"Students per school:\n{df.groupby('school_id').size()}")
print(f"\nObserved global mean: {scores.mean():.2f}")
print(f"Observed global SD:   {scores.std():.2f}")
print(f"Observed school means:\n{df.groupby('school_id')['score'].mean()}")

# ---------------------------------------------------------------------------
# 2. Formulate the generative story (documented in docstring above)
# ---------------------------------------------------------------------------
# y_{ij} ~ Normal(mu_j, sigma_student)         [student i in school j]
# mu_j   ~ Normal(mu_global, sigma_school)     [school j's true mean]
#
# Partial pooling: small schools shrink toward the global mean;
# large schools retain their own estimate.

# ---------------------------------------------------------------------------
# 3. Specify priors (with justifications in code comments)
# ---------------------------------------------------------------------------
# See analysis_notes.md for extended discussion.
#
# mu_global     ~ Normal(50, 20)
#   Test scores typically range 0-100. Centering at 50 with SD=20 puts
#   95% prior mass roughly in [10, 90] -- weakly informative, lets data dominate.
#
# sigma_school  ~ Gamma(2, 0.5)
#   Between-school SD. Mean=4, mode=2, allows up to ~15-20.
#   Gamma avoids near-zero region that creates funnel geometry in hierarchical
#   models. If there is no school-level variation, we don't need the hierarchy.
#
# sigma_student ~ Gamma(2, 0.2)
#   Within-school SD. Mean=10, mode=5. Student-level variation is expected
#   to be larger than school-level variation. Allows a wide range of spreads.
#
# Non-centered parameterization (mu_raw ~ Normal(0, 1)):
#   Used because some schools have as few as 5 students, which would cause
#   funnel divergences with centered parameterization.

# ---------------------------------------------------------------------------
# 4. Build the PyMC model
# ---------------------------------------------------------------------------
coords = {
    "school": school_names,
    "obs": np.arange(len(df)),
}

with pm.Model(coords=coords) as school_model:
    # --- Data containers ---
    school_idx = pm.Data("school_idx", df["school_id"].to_numpy(), dims="obs")
    y = pm.Data("y", df["score"].to_numpy(), dims="obs")

    # --- Hyper-priors (population level) ---
    # Global mean test score: weakly informative, centered on mid-range of 0-100 scores.
    # SD=20 gives 95% prior mass in roughly [10, 90], letting the data dominate.
    mu_global = pm.Normal("mu_global", mu=50, sigma=20)

    # Between-school SD: how much schools differ from each other.
    # Gamma(2, 0.5) has mean=4, mode=2, avoids near-zero to prevent funnel geometry.
    # If schools don't truly differ, the data will push this down, but we don't
    # force it toward zero a priori.
    sigma_school = pm.Gamma("sigma_school", alpha=2, beta=0.5)

    # --- School-level means (non-centered parameterization) ---
    # Non-centered eliminates funnel geometry. Essential with small groups (n=5).
    # Rule of thumb: start non-centered, switch to centered only if poor ESS AND
    # groups all have 50+ observations.
    mu_raw = pm.Normal("mu_raw", mu=0, sigma=1, dims="school")
    mu_school = pm.Deterministic(
        "mu_school", mu_global + mu_raw * sigma_school, dims="school"
    )

    # --- Within-school SD: individual student variation ---
    # Gamma(2, 0.2) has mean=10, mode=5. Student-level spread is expected
    # to be larger than school-level variation.
    sigma_student = pm.Gamma("sigma_student", alpha=2, beta=0.2)

    # --- Likelihood ---
    pm.Normal(
        "likelihood",
        mu=mu_school[school_idx],
        sigma=sigma_student,
        observed=y,
        dims="obs",
    )

    # ------------------------------------------------------------------
    # Step 4: Prior predictive check -- verify priors produce plausible data
    # ------------------------------------------------------------------
    prior_pred = pm.sample_prior_predictive(random_seed=rng)

# ---------------------------------------------------------------------------
# 5. Visualize prior predictive checks
# ---------------------------------------------------------------------------
print("\n=== Prior Predictive Check ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

prior_scores = prior_pred.prior_predictive["likelihood"].values.flatten()
axes[0].hist(prior_scores, bins=80, density=True, alpha=0.6, color="steelblue")
axes[0].axvline(scores.mean(), color="red", linestyle="--", label="Observed mean")
axes[0].set_title("Prior Predictive: Student Scores")
axes[0].set_xlabel("Score")
axes[0].set_ylabel("Density")
axes[0].legend()

prior_school_means = prior_pred.prior["mu_school"].values.flatten()
axes[1].hist(prior_school_means, bins=80, density=True, alpha=0.6, color="darkorange")
axes[1].set_title("Prior Predictive: School Means")
axes[1].set_xlabel("School Mean Score")
axes[1].set_ylabel("Density")

plt.tight_layout()
plt.savefig("outputs/prior_predictive_check.png", dpi=150, bbox_inches="tight")
plt.close()

print(
    "Prior predictive scores range: "
    f"[{np.percentile(prior_scores, 2.5):.0f}, {np.percentile(prior_scores, 97.5):.0f}]"
)
print("Check: do these cover plausible test score ranges (roughly 0-100)?")

# ---------------------------------------------------------------------------
# 6. Inference -- nutpie sampler, do NOT hardcode number of chains
# ---------------------------------------------------------------------------
with school_model:
    idata = pm.sample(
        nuts_sampler="nutpie",
        random_seed=rng,
        # Don't hardcode chains -- let PyMC / nutpie pick the best default
        # for the user's platform (usually matches CPU cores).
    )
    # Merge prior predictive samples
    idata.extend(prior_pred)

    # --- Posterior predictive check ---
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))

# --- Save InferenceData IMMEDIATELY after sampling ---
# Late crashes or kernel restarts can destroy valid MCMC results.
# Save before any post-processing.
idata.to_netcdf("outputs/school_model_idata.nc")
print("\nInferenceData saved to outputs/school_model_idata.nc")

# ---------------------------------------------------------------------------
# 7. Convergence Diagnostics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 60)

# 7a. Summary table (R-hat + ESS at a glance)
summary = az.summary(idata, round_to=3)
print("\n--- ArviZ Summary ---")
print(summary)

# 7b. Check R-hat
rhat_max = summary["r_hat"].max()
rhat_ok = (summary["r_hat"] <= 1.01).all()
print(f"\nR-hat max: {rhat_max:.4f} -- OK: {rhat_ok}")

# 7c. Check ESS -- threshold is 100 * number of chains
num_chains = idata.posterior.sizes["chain"]
ess_bulk_min = summary["ess_bulk"].min()
ess_tail_min = summary["ess_tail"].min()
ess_bulk_ok = (summary["ess_bulk"] >= 100 * num_chains).all()
ess_tail_ok = (summary["ess_tail"] >= 100 * num_chains).all()
print(f"ESS bulk min: {ess_bulk_min:.0f} -- OK (>= {100 * num_chains}): {ess_bulk_ok}")
print(f"ESS tail min: {ess_tail_min:.0f} -- OK (>= {100 * num_chains}): {ess_tail_ok}")

# 7d. Check divergences
n_div = idata.sample_stats["diverging"].sum().item()
divergences_ok = n_div == 0
print(f"Divergences: {n_div} -- OK: {divergences_ok}")

all_diagnostics_ok = rhat_ok and ess_bulk_ok and ess_tail_ok and divergences_ok
print(f"\nAll diagnostics pass: {all_diagnostics_ok}")
if not all_diagnostics_ok:
    print("WARNING: Fix convergence issues before interpreting results!")

# 7e. Trace / rank plots
az.plot_trace(
    idata, var_names=["mu_global", "sigma_school", "sigma_student"], kind="rank_vlines"
)
plt.tight_layout()
plt.savefig("outputs/trace_rank_plots_global.png", dpi=150, bbox_inches="tight")
plt.close()

az.plot_trace(idata, var_names=["mu_school"], kind="rank_vlines")
plt.tight_layout()
plt.savefig("outputs/trace_rank_plots_schools.png", dpi=150, bbox_inches="tight")
plt.close()

# 7f. Energy diagnostics
az.plot_energy(idata)
plt.savefig("outputs/energy_plot.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 8. Model Criticism
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL CRITICISM")
print("=" * 60)

# --- 8a. Posterior predictive check ---
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_ppc(idata, num_pp_samples=100, ax=ax)
ax.set_title("Posterior Predictive Check")
ax.set_xlabel("Score")
plt.tight_layout()
plt.savefig("outputs/posterior_predictive_check.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 8b. Calibration check using ArviZ plot_ppc_pit ---
# MANDATORY per skill rules. ArviZ handles all data types (continuous, binary,
# count) correctly out of the box. Don't write custom calibration code.
azp.plot_ppc_pit(idata)
plt.tight_layout()
plt.savefig("outputs/calibration_ppc_pit.png", dpi=150, bbox_inches="tight")
plt.close()
print("PPC-PIT calibration plot saved.")

# --- 8c. LOO-CV ---
# nutpie silently ignores idata_kwargs={"log_likelihood": True}.
# Must compute log_likelihood explicitly after sampling.
with school_model:
    pm.compute_log_likelihood(idata, model=school_model)

loo = az.loo(idata, pointwise=True)
print("\n--- LOO-CV ---")
print(loo)

# Pareto k diagnostics
pareto_k = loo.pareto_k.values
n_bad = (pareto_k > 0.7).sum()
print(f"\nObservations with Pareto k > 0.7: {n_bad}")
if n_bad > 0:
    bad_obs = np.where(pareto_k > 0.7)[0]
    print(f"  Indices: {bad_obs}")
    print("  Investigate these observations -- they may be outliers or poorly fit.")

# az.plot_khat requires the LOO object, not InferenceData
az.plot_khat(loo)
plt.savefig("outputs/pareto_k_plot.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 8d. LOO-PIT calibration (more robust when LOO is available) ---
azp.plot_ppc_pit(idata, loo_pit=True)
plt.tight_layout()
plt.savefig("outputs/calibration_loo_pit.png", dpi=150, bbox_inches="tight")
plt.close()
print("LOO-PIT calibration plot saved.")

# ---------------------------------------------------------------------------
# 9. Results
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

# --- 9a. Variance Partition Coefficient (VPC / ICC) ---
# VPC = sigma_school^2 / (sigma_school^2 + sigma_student^2)
# This is the central quantity: what fraction of total variation is between schools?
sigma_school_post = idata.posterior["sigma_school"].values.flatten()
sigma_student_post = idata.posterior["sigma_student"].values.flatten()

icc_samples = sigma_school_post**2 / (sigma_school_post**2 + sigma_student_post**2)
icc_mean = icc_samples.mean()
icc_hdi = az.hdi(icc_samples, hdi_prob=0.94)

print("\n--- Intraclass Correlation Coefficient (ICC / VPC) ---")
print(f"ICC mean: {icc_mean:.3f}")
print(f"ICC 94% HDI: [{icc_hdi[0]:.3f}, {icc_hdi[1]:.3f}]")
print(
    f"Interpretation: ~{icc_mean * 100:.1f}% of the total variation in test scores "
    f"is between schools; ~{(1 - icc_mean) * 100:.1f}% is within schools."
)

# Plot ICC posterior distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(icc_samples, bins=60, density=True, alpha=0.7, color="steelblue", edgecolor="navy")
ax.axvline(icc_mean, color="red", linestyle="--", linewidth=2, label=f"Mean = {icc_mean:.3f}")
ax.axvline(icc_hdi[0], color="orange", linestyle=":", linewidth=1.5,
           label=f"94% HDI lower = {icc_hdi[0]:.3f}")
ax.axvline(icc_hdi[1], color="orange", linestyle=":", linewidth=1.5,
           label=f"94% HDI upper = {icc_hdi[1]:.3f}")
ax.set_xlabel("ICC (proportion of variance between schools)")
ax.set_ylabel("Density")
ax.set_title("Posterior Distribution of Intraclass Correlation Coefficient")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/icc_posterior.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 9b. Population-level parameters (always with credible intervals, never point estimates alone) ---
print("\n--- Population-Level Estimates (94% HDI) ---")
for param in ["mu_global", "sigma_school", "sigma_student"]:
    post = idata.posterior[param].values.flatten()
    hdi = az.hdi(post, hdi_prob=0.94)
    print(f"  {param:<16}: mean={post.mean():.2f}, SD={post.std():.2f}, 94% HDI=[{hdi[0]:.2f}, {hdi[1]:.2f}]")

# --- 9c. School-level estimates (forest plot with 94% HDI) ---
print("\n--- School-Level Estimates (mu_school, 94% HDI) ---")
school_summary = az.summary(idata, var_names=["mu_school"], hdi_prob=0.94, round_to=2)
print(school_summary)

az.plot_forest(
    idata,
    var_names=["mu_school"],
    combined=True,
    hdi_prob=0.94,
    figsize=(8, 6),
)
plt.title("School-Level Mean Estimates (94% HDI)")
plt.tight_layout()
plt.savefig("outputs/school_forest_plot.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 9d. Shrinkage plot -- the hallmark of partial pooling ---
# Shows how school estimates are pulled toward the global mean,
# especially for schools with fewer students.
mu_school_post = idata.posterior["mu_school"].mean(dim=["chain", "draw"]).values
observed_school_means = df.groupby("school_id")["score"].mean().values
school_n = df.groupby("school_id").size().values
global_mean_post = idata.posterior["mu_global"].mean().item()

fig, ax = plt.subplots(figsize=(10, 7))

# Draw arrows from observed mean to posterior mean for each school
for j in range(N_SCHOOLS):
    ax.annotate(
        "",
        xy=(mu_school_post[j], j),
        xytext=(observed_school_means[j], j),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )

# Observed means (no-pooling estimates)
ax.scatter(
    observed_school_means,
    range(N_SCHOOLS),
    s=school_n * 8,
    c="steelblue",
    alpha=0.7,
    edgecolors="navy",
    label="Observed mean (no pooling)",
    zorder=3,
)

# Posterior means (partial-pooling estimates)
ax.scatter(
    mu_school_post,
    range(N_SCHOOLS),
    s=school_n * 8,
    c="darkorange",
    alpha=0.7,
    edgecolors="saddlebrown",
    label="Posterior mean (partial pooling)",
    zorder=3,
)

# Global mean (complete-pooling reference)
ax.axvline(
    global_mean_post, color="red", linestyle=":", linewidth=2,
    label="Global mean (complete pooling)"
)

ax.set_yticks(range(N_SCHOOLS))
ax.set_yticklabels([f"{school_names[j]} (n={school_n[j]})" for j in range(N_SCHOOLS)])
ax.set_xlabel("Mean Test Score")
ax.set_title("Shrinkage Plot: Partial Pooling Pulls Small Schools Toward Global Mean")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/shrinkage_plot.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nShrinkage summary:")
print("Schools with fewer students show more shrinkage toward the global mean.")
print("Schools with more students retain estimates closer to their observed means.")

# --- 9e. Variance components posteriors ---
# Check that sigma_school does not pile up near zero (which would mean
# no school-level variation, i.e. complete pooling).
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(sigma_school_post, bins=60, density=True, alpha=0.7, color="steelblue")
axes[0].set_title("Posterior: sigma_school (Between-School SD)")
axes[0].set_xlabel("sigma_school")

axes[1].hist(sigma_student_post, bins=60, density=True, alpha=0.7, color="darkorange")
axes[1].set_title("Posterior: sigma_student (Within-School SD)")
axes[1].set_xlabel("sigma_student")

plt.tight_layout()
plt.savefig("outputs/variance_components_posterior.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 9f. Model graph (optional, requires graphviz) ---
try:
    graph = pm.model_to_graphviz(school_model)
    graph.render("outputs/model_graph", format="png", cleanup=True)
    print("\nModel graph saved to outputs/model_graph.png")
except Exception as e:
    print(f"\nCould not render model graph (graphviz may not be installed): {e}")

print("\nAnalysis complete. All outputs saved to outputs/ directory.")
