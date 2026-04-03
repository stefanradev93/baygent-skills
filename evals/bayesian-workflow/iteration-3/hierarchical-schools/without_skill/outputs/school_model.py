"""
Hierarchical (Multilevel) Model for School Test Scores
======================================================

Goal: Understand how much variation in student test scores is between schools
vs within schools, and obtain school-level estimates that properly account
for different sample sizes (partial pooling).

Model:
    y_ij ~ Normal(mu_j, sigma_within)
    mu_j ~ Normal(mu_global, sigma_between)
    mu_global ~ Normal(50, 25)
    sigma_between ~ HalfNormal(20)
    sigma_within ~ HalfNormal(20)

where y_ij is the test score for student i in school j.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# 1. GENERATE SYNTHETIC DATA
# ============================================================

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

N_SCHOOLS = 15
N_STUDENTS_TOTAL = 200

# True parameters (for later comparison with estimates)
TRUE_MU_GLOBAL = 55.0
TRUE_SIGMA_BETWEEN = 10.0
TRUE_SIGMA_WITHIN = 12.0

# Generate unequal school sizes summing to 200
# Some schools are small (5 students), some are large
school_sizes_raw = rng.integers(5, 30, size=N_SCHOOLS)
# Force a few schools to have exactly 5 students to match the problem description
school_sizes_raw[0] = 5
school_sizes_raw[1] = 5
school_sizes_raw[2] = 5
# Scale the rest so total is 200
remaining = N_STUDENTS_TOTAL - 15  # 15 from the three small schools
remaining_schools = N_SCHOOLS - 3
school_sizes_raw[3:] = rng.integers(8, 25, size=remaining_schools)
# Adjust to hit exactly 200
current_sum = school_sizes_raw.sum()
diff = N_STUDENTS_TOTAL - current_sum
school_sizes_raw[-1] += diff  # adjust last school
school_sizes = school_sizes_raw.copy()

print(f"School sizes: {school_sizes}")
print(f"Total students: {school_sizes.sum()}")

# Generate true school means from the between-school distribution
true_school_means = rng.normal(TRUE_MU_GLOBAL, TRUE_SIGMA_BETWEEN, size=N_SCHOOLS)

# Generate student scores within each school
school_ids = []
scores = []
school_labels = []

for j in range(N_SCHOOLS):
    n_j = school_sizes[j]
    school_scores = rng.normal(true_school_means[j], TRUE_SIGMA_WITHIN, size=n_j)
    scores.extend(school_scores)
    school_ids.extend([j] * n_j)
    school_labels.extend([f"School {j+1}"] * n_j)

data = pd.DataFrame({
    "student_id": range(N_STUDENTS_TOTAL),
    "school_id": school_ids,
    "school_label": school_labels,
    "score": scores,
})

print("\n--- Data Summary ---")
print(f"Total students: {len(data)}")
print(f"Number of schools: {data['school_id'].nunique()}")
print(f"\nStudents per school:")
print(data.groupby("school_label")["score"].agg(["count", "mean", "std"]).round(2))


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 2a. Distribution of all scores
axes[0].hist(data["score"], bins=25, edgecolor="white", alpha=0.8, color="steelblue")
axes[0].set_xlabel("Test Score")
axes[0].set_ylabel("Count")
axes[0].set_title("Overall Score Distribution")
axes[0].axvline(data["score"].mean(), color="red", linestyle="--", label=f'Mean = {data["score"].mean():.1f}')
axes[0].legend()

# 2b. School-level means with sample sizes
school_stats = data.groupby("school_id")["score"].agg(["mean", "std", "count"]).reset_index()
school_stats.columns = ["school_id", "mean", "std", "count"]
school_stats = school_stats.sort_values("mean")

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(school_stats)))
bars = axes[1].barh(
    range(len(school_stats)),
    school_stats["mean"],
    xerr=school_stats["std"] / np.sqrt(school_stats["count"]),
    color=colors,
    edgecolor="white",
    capsize=3,
)
axes[1].set_yticks(range(len(school_stats)))
axes[1].set_yticklabels([f'School {sid+1} (n={cnt})' for sid, cnt in zip(school_stats["school_id"], school_stats["count"])])
axes[1].set_xlabel("Mean Test Score (+/- SE)")
axes[1].set_title("School Means (Raw)")
axes[1].axvline(data["score"].mean(), color="red", linestyle="--", alpha=0.5)

# 2c. Box plots by school
school_order = school_stats["school_id"].values
data_sorted = data.copy()
data_sorted["school_id_ordered"] = data_sorted["school_id"].map(
    {sid: i for i, sid in enumerate(school_order)}
)
boxes = []
for sid in school_order:
    boxes.append(data[data["school_id"] == sid]["score"].values)
axes[2].boxplot(boxes, vert=True, patch_artist=True)
axes[2].set_xticklabels([f"S{sid+1}" for sid in school_order], rotation=45, fontsize=8)
axes[2].set_ylabel("Test Score")
axes[2].set_title("Score Distributions by School")

plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-1/hierarchical-schools/without_skill/outputs/eda_plots.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("\nEDA plots saved.")


# ============================================================
# 3. PRIOR PREDICTIVE CHECK
# ============================================================

with pm.Model() as prior_model:
    # Hyperpriors
    mu_global = pm.Normal("mu_global", mu=50, sigma=25)
    sigma_between = pm.HalfNormal("sigma_between", sigma=20)
    sigma_within = pm.HalfNormal("sigma_within", sigma=20)

    # School-level means (non-centered parameterization for sampling efficiency)
    school_offset = pm.Normal("school_offset", mu=0, sigma=1, shape=N_SCHOOLS)
    mu_school = pm.Deterministic("mu_school", mu_global + sigma_between * school_offset)

    # Likelihood
    y_obs = pm.Normal(
        "y_obs",
        mu=mu_school[data["school_id"].values],
        sigma=sigma_within,
        observed=data["score"].values,
    )

    # Prior predictive
    prior_pred = pm.sample_prior_predictive(samples=500, random_seed=RANDOM_SEED)

# Visualize prior predictive
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Prior predictive for y
prior_y = prior_pred.prior_predictive["y_obs"].values.flatten()
axes[0].hist(prior_y, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="white")
axes[0].set_xlabel("Test Score")
axes[0].set_ylabel("Density")
axes[0].set_title("Prior Predictive Distribution for Scores")
axes[0].set_xlim(-100, 200)

# Prior for school means
prior_mu_school = prior_pred.prior["mu_school"].values.flatten()
axes[1].hist(prior_mu_school, bins=80, density=True, alpha=0.7, color="coral", edgecolor="white")
axes[1].set_xlabel("School Mean")
axes[1].set_ylabel("Density")
axes[1].set_title("Prior Distribution for School Means")

plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-1/hierarchical-schools/without_skill/outputs/prior_predictive.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("Prior predictive plots saved.")


# ============================================================
# 4. FIT THE MODEL (POSTERIOR SAMPLING)
# ============================================================

with pm.Model() as hierarchical_model:
    # --- Hyperpriors ---
    # Global mean: centered on 50, wide enough to cover typical test score ranges
    mu_global = pm.Normal("mu_global", mu=50, sigma=25)

    # Between-school SD: how much do school means vary?
    # HalfNormal(20) puts most prior mass below ~40, allowing for substantial
    # between-school variation but making extreme values unlikely
    sigma_between = pm.HalfNormal("sigma_between", sigma=20)

    # Within-school SD: how much do students within a school vary?
    sigma_within = pm.HalfNormal("sigma_within", sigma=20)

    # --- Non-centered parameterization ---
    # This is crucial for hierarchical models to avoid the "funnel" geometry
    # that makes centered parameterizations hard to sample
    school_offset = pm.Normal("school_offset", mu=0, sigma=1, shape=N_SCHOOLS)
    mu_school = pm.Deterministic("mu_school", mu_global + sigma_between * school_offset)

    # --- Derived quantities ---
    # Intraclass Correlation Coefficient (ICC): proportion of total variance
    # that is between schools
    icc = pm.Deterministic(
        "icc", sigma_between**2 / (sigma_between**2 + sigma_within**2)
    )

    # --- Likelihood ---
    y_obs = pm.Normal(
        "y_obs",
        mu=mu_school[data["school_id"].values],
        sigma=sigma_within,
        observed=data["score"].values,
    )

    # --- Sample ---
    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.95,
        random_seed=RANDOM_SEED,
        return_inferencedata=True,
    )

    # --- Posterior predictive ---
    post_pred = pm.sample_posterior_predictive(trace, random_seed=RANDOM_SEED)


# ============================================================
# 5. DIAGNOSTICS
# ============================================================

print("\n" + "=" * 60)
print("MODEL DIAGNOSTICS")
print("=" * 60)

# 5a. Summary statistics with convergence diagnostics
summary = az.summary(
    trace,
    var_names=["mu_global", "sigma_between", "sigma_within", "icc", "mu_school"],
    round_to=2,
)
print("\n--- Posterior Summary ---")
print(summary)

# 5b. Check R-hat values
rhat_vars = ["mu_global", "sigma_between", "sigma_within", "school_offset"]
print("\n--- R-hat Check ---")
rhat = az.rhat(trace, var_names=rhat_vars)
for var in rhat_vars:
    vals = rhat[var].values
    if np.isscalar(vals):
        max_rhat = float(vals)
    else:
        max_rhat = float(np.max(vals))
    status = "OK" if max_rhat < 1.01 else "WARNING"
    print(f"  {var}: max R-hat = {max_rhat:.4f} [{status}]")

# 5c. Check effective sample size
print("\n--- Effective Sample Size Check ---")
ess = az.ess(trace, var_names=rhat_vars)
for var in rhat_vars:
    vals = ess[var].values
    if np.isscalar(vals):
        min_ess = float(vals)
    else:
        min_ess = float(np.min(vals))
    status = "OK" if min_ess > 400 else "WARNING"
    print(f"  {var}: min ESS = {min_ess:.0f} [{status}]")

# 5d. Check for divergences
n_divergences = trace.sample_stats["diverging"].sum().values
print(f"\n--- Divergences: {n_divergences} ---")
if n_divergences > 0:
    print("  WARNING: Divergences detected. Consider reparameterization or higher target_accept.")
else:
    print("  No divergences detected. Good.")

# 5e. Trace plots
az.plot_trace(
    trace,
    var_names=["mu_global", "sigma_between", "sigma_within", "icc"],
    figsize=(14, 10),
)
plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-1/hierarchical-schools/without_skill/outputs/trace_plots.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("Trace plots saved.")

# 5f. Rank plots (more informative than trace plots for convergence)
az.plot_rank(
    trace,
    var_names=["mu_global", "sigma_between", "sigma_within"],
    figsize=(14, 4),
)
plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-1/hierarchical-schools/without_skill/outputs/rank_plots.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("Rank plots saved.")

# 5g. Energy plot (checks for pathological behavior in HMC)
az.plot_energy(trace, figsize=(6, 4))
plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-1/hierarchical-schools/without_skill/outputs/energy_plot.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("Energy plot saved.")


# ============================================================
# 6. POSTERIOR PREDICTIVE CHECK
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 6a. Overall distribution: observed vs predicted
az.plot_ppc(
    post_pred,
    observed_rug=True,
    ax=axes[0],
    num_pp_samples=100,
)
axes[0].set_title("Posterior Predictive Check: Overall")
axes[0].set_xlabel("Test Score")

# 6b. School-level means: observed vs predicted
posterior_mu_school = trace.posterior["mu_school"]
posterior_means = posterior_mu_school.mean(dim=["chain", "draw"]).values
posterior_hdi = az.hdi(trace, var_names=["mu_school"])["mu_school"].values

observed_means = data.groupby("school_id")["score"].mean().values

axes[1].errorbar(
    range(N_SCHOOLS),
    posterior_means,
    yerr=[
        posterior_means - posterior_hdi[:, 0],
        posterior_hdi[:, 1] - posterior_means,
    ],
    fmt="o",
    color="steelblue",
    capsize=4,
    label="Posterior (94% HDI)",
    markersize=6,
)
axes[1].scatter(
    range(N_SCHOOLS),
    observed_means,
    marker="x",
    color="red",
    s=80,
    zorder=5,
    label="Observed Mean",
)
axes[1].scatter(
    range(N_SCHOOLS),
    true_school_means,
    marker="D",
    color="green",
    s=40,
    zorder=5,
    alpha=0.7,
    label="True Mean",
)
axes[1].axhline(
    data["score"].mean(), color="gray", linestyle="--", alpha=0.5, label="Grand Mean"
)
axes[1].set_xlabel("School ID")
axes[1].set_ylabel("Mean Score")
axes[1].set_title("School-Level Estimates: Shrinkage Effect")
axes[1].legend(fontsize=8)
axes[1].set_xticks(range(N_SCHOOLS))
axes[1].set_xticklabels([f"S{i+1}\n(n={school_sizes[i]})" for i in range(N_SCHOOLS)], fontsize=7)

plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-1/hierarchical-schools/without_skill/outputs/posterior_predictive.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("Posterior predictive plots saved.")


# ============================================================
# 7. KEY RESULTS: VARIANCE DECOMPOSITION & SHRINKAGE
# ============================================================

print("\n" + "=" * 60)
print("KEY RESULTS")
print("=" * 60)

# 7a. Variance decomposition
sigma_between_post = trace.posterior["sigma_between"].values.flatten()
sigma_within_post = trace.posterior["sigma_within"].values.flatten()
icc_post = trace.posterior["icc"].values.flatten()

print("\n--- Variance Decomposition ---")
print(f"  sigma_between (school-level SD):  {np.mean(sigma_between_post):.2f} "
      f"[94% HDI: {np.percentile(sigma_between_post, 3):.2f}, {np.percentile(sigma_between_post, 97):.2f}]")
print(f"  sigma_within  (student-level SD): {np.mean(sigma_within_post):.2f} "
      f"[94% HDI: {np.percentile(sigma_within_post, 3):.2f}, {np.percentile(sigma_within_post, 97):.2f}]")
print(f"  ICC (proportion between-school):  {np.mean(icc_post):.3f} "
      f"[94% HDI: {np.percentile(icc_post, 3):.3f}, {np.percentile(icc_post, 97):.3f}]")
print(f"\n  True sigma_between: {TRUE_SIGMA_BETWEEN}")
print(f"  True sigma_within:  {TRUE_SIGMA_WITHIN}")
true_icc = TRUE_SIGMA_BETWEEN**2 / (TRUE_SIGMA_BETWEEN**2 + TRUE_SIGMA_WITHIN**2)
print(f"  True ICC:           {true_icc:.3f}")

# 7b. Shrinkage analysis
print("\n--- Shrinkage Analysis ---")
print("  Schools with fewer students are shrunk more toward the grand mean.")
print("  This is the key benefit of partial pooling.\n")

mu_global_post_mean = trace.posterior["mu_global"].mean().values

print(f"  {'School':<12} {'n':>4} {'Obs Mean':>10} {'Post Mean':>10} {'Shrinkage':>10} {'True Mean':>10}")
print("  " + "-" * 60)

for j in range(N_SCHOOLS):
    obs_mean = data[data["school_id"] == j]["score"].mean()
    post_mean = posterior_means[j]
    # Shrinkage = how far the posterior moved from observed toward grand mean
    # as a fraction of the distance from observed to grand mean
    distance_obs_to_grand = obs_mean - mu_global_post_mean
    distance_post_to_grand = post_mean - mu_global_post_mean
    if abs(distance_obs_to_grand) > 0.01:
        shrinkage = 1 - distance_post_to_grand / distance_obs_to_grand
    else:
        shrinkage = 0.0
    print(f"  School {j+1:<4} {school_sizes[j]:>4} {obs_mean:>10.2f} {post_mean:>10.2f} "
          f"{shrinkage:>9.1%} {true_school_means[j]:>10.2f}")


# ============================================================
# 8. SHRINKAGE VISUALIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))

for j in range(N_SCHOOLS):
    obs_mean = data[data["school_id"] == j]["score"].mean()
    post_mean = posterior_means[j]
    size_marker = school_sizes[j] * 3

    # Draw arrow from observed to posterior
    ax.annotate(
        "",
        xy=(post_mean, j),
        xytext=(obs_mean, j),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )
    # Observed mean (size proportional to n)
    ax.scatter(obs_mean, j, s=size_marker, color="red", zorder=5, alpha=0.7)
    # Posterior mean
    ax.scatter(post_mean, j, s=size_marker, color="steelblue", zorder=5, alpha=0.9)
    # True mean
    ax.scatter(true_school_means[j], j, s=30, color="green", marker="D", zorder=6, alpha=0.7)

# Grand mean line
ax.axvline(mu_global_post_mean, color="black", linestyle="--", alpha=0.4, label="Grand Mean (posterior)")

ax.set_yticks(range(N_SCHOOLS))
ax.set_yticklabels([f"School {j+1} (n={school_sizes[j]})" for j in range(N_SCHOOLS)])
ax.set_xlabel("Mean Test Score")
ax.set_title("Shrinkage Plot: Observed (red) -> Posterior (blue), True (green diamond)")
ax.legend(loc="lower right")

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="Observed Mean"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", markersize=8, label="Posterior Mean"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="green", markersize=6, label="True Mean"),
    Line2D([0], [0], color="black", linestyle="--", alpha=0.4, label="Grand Mean"),
]
ax.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-1/hierarchical-schools/without_skill/outputs/shrinkage_plot.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("Shrinkage plot saved.")


# ============================================================
# 9. ICC POSTERIOR VISUALIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(8, 4))
az.plot_posterior(
    trace,
    var_names=["icc"],
    ref_val=true_icc,
    ax=ax,
    hdi_prob=0.94,
)
ax.set_title("Posterior Distribution of ICC (Intraclass Correlation Coefficient)")
ax.set_xlabel("ICC = sigma_between^2 / (sigma_between^2 + sigma_within^2)")

plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-1/hierarchical-schools/without_skill/outputs/icc_posterior.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("ICC posterior plot saved.")


# ============================================================
# 10. FOREST PLOT OF SCHOOL MEANS
# ============================================================

az.plot_forest(
    trace,
    var_names=["mu_school"],
    combined=True,
    figsize=(8, 6),
    hdi_prob=0.94,
)
plt.title("School-Level Mean Estimates (94% HDI)")
plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/"
    "iteration-1/hierarchical-schools/without_skill/outputs/forest_plot.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("Forest plot saved.")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nAll outputs saved to:")
print("  /Users/alex_andorra/tptm_alex/portfolio/agent-skills/bayesian-workflow-workspace/")
print("  iteration-1/hierarchical-schools/without_skill/outputs/")
