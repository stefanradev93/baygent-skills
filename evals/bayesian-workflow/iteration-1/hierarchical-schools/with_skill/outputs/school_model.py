"""
Hierarchical (Multilevel) Model for Student Test Scores Across Schools
======================================================================

Goal: Understand how much variation in test scores is between schools vs.
within schools, and obtain school-level estimates that account for different
sample sizes (partial pooling / shrinkage).

Workflow follows the Bayesian Workflow skill:
  1. Formulate the generative story
  2. Specify and justify priors
  3. Implement in PyMC
  4. Prior predictive checks
  5. Inference
  6. Convergence diagnostics
  7. Model criticism (PPC, LOO, calibration)
  8. Report results (shrinkage plot, variance partition, summaries)
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

RANDOM_SEED = 42
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

# Assign students to schools with imbalanced sizes
# We force some small schools (size ~5) and let the rest fill up
school_sizes_raw = rng.integers(5, 25, size=N_SCHOOLS)
# Ensure a few schools have exactly 5 students
school_sizes_raw[:3] = 5
# Rescale so total = 200
school_sizes = (school_sizes_raw / school_sizes_raw.sum() * TOTAL_STUDENTS).astype(int)
# Adjust rounding to hit exactly 200
diff = TOTAL_STUDENTS - school_sizes.sum()
school_sizes[-1] += diff

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

# Build a DataFrame for convenience
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
# 2. Formulate the generative story
# ---------------------------------------------------------------------------
# Each school j has a "true" average score mu_j drawn from a population of
# schools with global mean mu_global and between-school spread sigma_between.
# Each student i within school j then has an observed score drawn from a
# Normal centered on mu_j with within-school spread sigma_within.
#
# y_{ij} ~ Normal(mu_j, sigma_within)
# mu_j   ~ Normal(mu_global, sigma_between)
#
# This is a varying-intercepts model. Partial pooling lets small schools
# borrow strength from the population, shrinking their estimates toward the
# global mean. Large schools keep estimates closer to their raw sample mean.

# ---------------------------------------------------------------------------
# 3. Specify priors (with justifications)
# ---------------------------------------------------------------------------
# See analysis_notes.md for extended discussion.
#
# mu_global     ~ Normal(50, 20)
#   Test scores typically range 0-100. Centering at 50 with SD=20 puts
#   95% prior mass in [10, 90] -- wide enough to let the data speak.
#
# sigma_between ~ Gamma(2, 0.5)
#   Between-school SD. Gamma(2, 0.5) has mean 4, mode 2, and allows
#   values up to ~15-20. Avoids near-zero (which would collapse to
#   complete pooling) while staying moderate. This is weakly informative.
#
# sigma_within  ~ Gamma(2, 0.2)
#   Within-school SD. Gamma(2, 0.2) has mean 10, mode 5, allowing the
#   typical spread of student scores within a school. Individual test
#   score variation is expected to be larger than school-level variation.
#
# Non-centered parameterization for school means:
#   mu_raw_j ~ Normal(0, 1)
#   mu_j = mu_global + mu_raw_j * sigma_between
#   Rule of thumb: start non-centered when some groups have few data
#   points (we have groups with only 5 students).

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
    # Global mean test score: weakly informative, centered on mid-range
    mu_global = pm.Normal(
        "mu_global", mu=50, sigma=20
    )  # 95% prior mass in [10, 90]; lets data dominate

    # Between-school SD: how much schools differ from each other
    sigma_between = pm.Gamma(
        "sigma_between", alpha=2, beta=0.5
    )  # Mean=4, avoids 0, allows up to ~20

    # --- School-level means (non-centered parameterization) ---
    # Non-centered to avoid funnel geometry; essential with small groups
    mu_raw = pm.Normal("mu_raw", mu=0, sigma=1, dims="school")
    mu_school = pm.Deterministic(
        "mu_school", mu_global + mu_raw * sigma_between, dims="school"
    )

    # --- Within-school SD: individual student variation ---
    sigma_within = pm.Gamma(
        "sigma_within", alpha=2, beta=0.2
    )  # Mean=10, mode=5; student-level spread

    # --- Likelihood ---
    pm.Normal(
        "likelihood",
        mu=mu_school[school_idx],
        sigma=sigma_within,
        observed=y,
        dims="obs",
    )

    # --- Prior predictive check (Step 4) ---
    prior_pred = pm.sample_prior_predictive(random_seed=RANDOM_SEED)

# ---------------------------------------------------------------------------
# 5. Visualize prior predictive checks
# ---------------------------------------------------------------------------
print("\n=== Prior Predictive Check ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Prior predictive distribution of observed scores
prior_scores = prior_pred.prior_predictive["likelihood"].values.flatten()
axes[0].hist(prior_scores, bins=80, density=True, alpha=0.6, color="steelblue")
axes[0].axvline(scores.mean(), color="red", linestyle="--", label="Observed mean")
axes[0].set_title("Prior Predictive: Student Scores")
axes[0].set_xlabel("Score")
axes[0].set_ylabel("Density")
axes[0].legend()

# Prior predictive school means
prior_school_means = prior_pred.prior["mu_school"].values.flatten()
axes[1].hist(prior_school_means, bins=80, density=True, alpha=0.6, color="darkorange")
axes[1].set_title("Prior Predictive: School Means")
axes[1].set_xlabel("School Mean Score")
axes[1].set_ylabel("Density")

plt.tight_layout()
plt.savefig("outputs/prior_predictive_check.png", dpi=150, bbox_inches="tight")
plt.show()
print(
    "Prior predictive scores range: "
    f"[{np.percentile(prior_scores, 2.5):.0f}, {np.percentile(prior_scores, 97.5):.0f}]"
)
print("Check: do these cover plausible test score ranges (roughly 0-100)?")

# ---------------------------------------------------------------------------
# 6. Inference (Step 5)
# ---------------------------------------------------------------------------
with school_model:
    idata = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        nuts_sampler="nutpie",  # Faster sampling when available
        random_seed=RANDOM_SEED,
    )
    # Extend with prior predictive samples
    idata.extend(prior_pred)

    # --- Posterior predictive check (Step 7) ---
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=RANDOM_SEED))

# ---------------------------------------------------------------------------
# 7. Convergence Diagnostics (Step 6)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 60)

# 7a. Summary table
summary = az.summary(idata, round_to=3)
print("\n--- ArviZ Summary ---")
print(summary)

# 7b. Check R-hat
rhat_max = summary["r_hat"].max()
rhat_ok = (summary["r_hat"] <= 1.01).all()
print(f"\nR-hat max: {rhat_max:.4f} -- OK: {rhat_ok}")

# 7c. Check ESS
num_chains = idata.posterior.sizes["chain"]
ess_bulk_min = summary["ess_bulk"].min()
ess_tail_min = summary["ess_tail"].min()
ess_bulk_ok = (summary["ess_bulk"] >= 100 * num_chains).all()
ess_tail_ok = (summary["ess_tail"] >= 100 * num_chains).all()
print(f"ESS bulk min: {ess_bulk_min:.0f} -- OK: {ess_bulk_ok}")
print(f"ESS tail min: {ess_tail_min:.0f} -- OK: {ess_tail_ok}")

# 7d. Check divergences
n_div = idata.sample_stats["diverging"].sum().item()
divergences_ok = n_div == 0
print(f"Divergences: {n_div} -- OK: {divergences_ok}")

all_diagnostics_ok = rhat_ok and ess_bulk_ok and ess_tail_ok and divergences_ok
print(f"\nAll diagnostics pass: {all_diagnostics_ok}")
if not all_diagnostics_ok:
    print("WARNING: Fix convergence issues before interpreting results!")

# 7e. Trace / rank plots
az.plot_trace(idata, var_names=["mu_global", "sigma_between", "sigma_within"], kind="rank_vlines")
plt.tight_layout()
plt.savefig("outputs/trace_rank_plots_global.png", dpi=150, bbox_inches="tight")
plt.show()

az.plot_trace(idata, var_names=["mu_school"], kind="rank_vlines")
plt.tight_layout()
plt.savefig("outputs/trace_rank_plots_schools.png", dpi=150, bbox_inches="tight")
plt.show()

# 7f. Energy diagnostics
az.plot_energy(idata)
plt.savefig("outputs/energy_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# 8. Model Criticism (Step 7)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL CRITICISM")
print("=" * 60)

# 8a. Posterior predictive check
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_ppc(idata, num_pp_samples=100, ax=ax)
ax.set_title("Posterior Predictive Check")
ax.set_xlabel("Score")
plt.tight_layout()
plt.savefig("outputs/posterior_predictive_check.png", dpi=150, bbox_inches="tight")
plt.show()

# 8b. LOO-CV
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

az.plot_khat(loo)
plt.savefig("outputs/pareto_k_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# 9. Key Results (Step 9)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

# 9a. Variance partition coefficient (VPC / ICC)
# VPC = sigma_between^2 / (sigma_between^2 + sigma_within^2)
# This tells us the proportion of total variance attributable to schools.
sigma_between_post = idata.posterior["sigma_between"].values.flatten()
sigma_within_post = idata.posterior["sigma_within"].values.flatten()

vpc_samples = sigma_between_post**2 / (sigma_between_post**2 + sigma_within_post**2)
vpc_mean = vpc_samples.mean()
vpc_hdi = az.hdi(vpc_samples, hdi_prob=0.94)

print("\n--- Variance Partition Coefficient (VPC / ICC) ---")
print(f"VPC mean: {vpc_mean:.3f}")
print(f"VPC 94% HDI: [{vpc_hdi[0]:.3f}, {vpc_hdi[1]:.3f}]")
print(
    f"Interpretation: ~{vpc_mean * 100:.1f}% of the total variation in test scores "
    f"is between schools; ~{(1 - vpc_mean) * 100:.1f}% is within schools."
)

# 9b. Population-level parameters
print("\n--- Population-Level Estimates ---")
for param in ["mu_global", "sigma_between", "sigma_within"]:
    post = idata.posterior[param].values.flatten()
    hdi = az.hdi(post, hdi_prob=0.94)
    print(f"  {param}: mean={post.mean():.2f}, SD={post.std():.2f}, 94% HDI=[{hdi[0]:.2f}, {hdi[1]:.2f}]")

# 9c. School-level estimates (forest plot)
print("\n--- School-Level Estimates (mu_school) ---")
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
plt.show()

# 9d. Shrinkage plot -- the hallmark of partial pooling
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

# Plot observed means (no-pooling estimates)
scatter_obs = ax.scatter(
    observed_school_means,
    range(N_SCHOOLS),
    s=school_n * 8,  # Size proportional to sample size
    c="steelblue",
    alpha=0.7,
    edgecolors="navy",
    label="Observed mean (no pooling)",
    zorder=3,
)

# Plot posterior means (partial-pooling estimates)
scatter_post = ax.scatter(
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
ax.axvline(global_mean_post, color="red", linestyle=":", linewidth=2, label="Global mean (complete pooling)")

ax.set_yticks(range(N_SCHOOLS))
ax.set_yticklabels([f"{school_names[j]} (n={school_n[j]})" for j in range(N_SCHOOLS)])
ax.set_xlabel("Mean Test Score")
ax.set_title("Shrinkage Plot: Partial Pooling Pulls Small Schools Toward Global Mean")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/shrinkage_plot.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nShrinkage summary:")
print("Schools with fewer students show more shrinkage toward the global mean.")
print("Schools with more students retain estimates closer to their observed means.")

# 9e. VPC posterior distribution plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(vpc_samples, bins=60, density=True, alpha=0.7, color="steelblue", edgecolor="navy")
ax.axvline(vpc_mean, color="red", linestyle="--", linewidth=2, label=f"Mean = {vpc_mean:.3f}")
ax.axvline(vpc_hdi[0], color="orange", linestyle=":", linewidth=1.5, label=f"94% HDI lower = {vpc_hdi[0]:.3f}")
ax.axvline(vpc_hdi[1], color="orange", linestyle=":", linewidth=1.5, label=f"94% HDI upper = {vpc_hdi[1]:.3f}")
ax.set_xlabel("VPC (proportion of variance between schools)")
ax.set_ylabel("Density")
ax.set_title("Posterior Distribution of Variance Partition Coefficient")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/vpc_posterior.png", dpi=150, bbox_inches="tight")
plt.show()

# 9f. sigma_between posterior -- check it is not collapsing to zero
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(sigma_between_post, bins=60, density=True, alpha=0.7, color="steelblue")
axes[0].set_title("Posterior: sigma_between (Between-School SD)")
axes[0].set_xlabel("sigma_between")

axes[1].hist(sigma_within_post, bins=60, density=True, alpha=0.7, color="darkorange")
axes[1].set_title("Posterior: sigma_within (Within-School SD)")
axes[1].set_xlabel("sigma_within")

plt.tight_layout()
plt.savefig("outputs/variance_components_posterior.png", dpi=150, bbox_inches="tight")
plt.show()

# 9g. Model graph
try:
    graph = pm.model_to_graphviz(school_model)
    graph.render("outputs/model_graph", format="png", cleanup=True)
    print("\nModel graph saved to outputs/model_graph.png")
except Exception as e:
    print(f"\nCould not render model graph (graphviz may not be installed): {e}")

# ---------------------------------------------------------------------------
# 10. Save InferenceData for later use
# ---------------------------------------------------------------------------
idata.to_netcdf("outputs/school_model_idata.nc")
print("\nInferenceData saved to outputs/school_model_idata.nc")
print("\nAnalysis complete.")
