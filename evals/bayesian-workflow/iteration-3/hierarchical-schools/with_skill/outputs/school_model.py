"""
Hierarchical (Multilevel) Model for Student Test Scores Across Schools
======================================================================

Generative story:
    Students are nested within schools. Each school has its own "true" average
    performance (mu_school), which is itself drawn from a population of schools.
    Schools with more students contribute more information about their own mean;
    schools with fewer students are partially pooled toward the grand mean.

    y_ij ~ Normal(mu_school[j], sigma_within)
    mu_school[j] ~ Normal(mu_global, sigma_between)
    mu_global ~ Normal(65, 15)           # Centered on plausible avg test score
    sigma_between ~ Gamma(2, 0.5)        # Between-school SD
    sigma_within ~ Gamma(2, 0.1)         # Within-school SD

We use a non-centered parameterization because some schools have as few
as 5 students -- the centered form is prone to funnel divergences in this
low-data-per-group regime.

Outputs:
    - Prior predictive check figure
    - Trace / rank plots
    - Posterior predictive check figure
    - Shrinkage plot
    - Calibration (PIT) plot
    - LOO diagnostics
    - Saved InferenceData (.nc)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

try:
    import arviz_plots as azp

    HAS_AZP = True
except ImportError:
    HAS_AZP = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = sum(map(ord, "hierarchical-schools-v1"))
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Generate synthetic data
# ---------------------------------------------------------------------------
N_SCHOOLS = 15
N_STUDENTS = 200

# True parameters (known because synthetic -- used later for calibration)
TRUE_MU_GLOBAL = 68.0
TRUE_SIGMA_BETWEEN = 8.0
TRUE_SIGMA_WITHIN = 12.0

# Generate unequal school sizes: some schools have as few as 5 students
# We use a Dirichlet-like approach to get varying sizes summing to N_STUDENTS
raw_sizes = rng.dirichlet(np.ones(N_SCHOOLS)) * N_STUDENTS
sizes = np.maximum(raw_sizes.astype(int), 5)  # minimum 5 per school
# Adjust to hit exactly N_STUDENTS
diff = N_STUDENTS - sizes.sum()
if diff > 0:
    # Add extra students to random schools
    for _ in range(diff):
        idx = rng.integers(N_SCHOOLS)
        sizes[idx] += 1
elif diff < 0:
    # Remove students from largest schools
    for _ in range(-diff):
        idx = sizes.argmax()
        sizes[idx] -= 1

# True school means
true_school_means = rng.normal(TRUE_MU_GLOBAL, TRUE_SIGMA_BETWEEN, size=N_SCHOOLS)

# Generate student scores
school_ids = []
scores = []
for j in range(N_SCHOOLS):
    n_j = sizes[j]
    school_ids.extend([j] * n_j)
    scores.extend(rng.normal(true_school_means[j], TRUE_SIGMA_WITHIN, size=n_j))

school_ids = np.array(school_ids)
scores = np.array(scores)

school_names = [f"School_{j:02d}" for j in range(N_SCHOOLS)]

df = pd.DataFrame({
    "student_id": np.arange(len(scores)),
    "school_id": school_ids,
    "school_name": [school_names[j] for j in school_ids],
    "score": scores,
})

print("=" * 60)
print("DATA SUMMARY")
print("=" * 60)
print(f"Total students: {len(df)}")
print(f"Number of schools: {N_SCHOOLS}")
print(f"\nStudents per school:")
school_counts = df.groupby("school_name")["score"].agg(["count", "mean", "std"])
print(school_counts.to_string())
print(f"\nOverall mean: {scores.mean():.1f}")
print(f"Overall SD:   {scores.std():.1f}")

# ---------------------------------------------------------------------------
# 2. Model specification (non-centered parameterization)
# ---------------------------------------------------------------------------
coords = {
    "school": school_names,
    "obs": np.arange(len(df)),
}

with pm.Model(coords=coords) as school_model:
    # --- Data containers ---
    school_idx = pm.Data("school_idx", df["school_id"].to_numpy(), dims="obs")
    y = pm.Data("y", df["score"].to_numpy(), dims="obs")

    # --- Hyperpriors (population-level) ---
    # Grand mean: centered on a plausible average test score (~65),
    # with SD=15 allowing a wide range of plausible averages (roughly 35-95).
    mu_global = pm.Normal("mu_global", mu=65, sigma=15)

    # Between-school SD: Gamma(2, 0.5) gives mean=4, but allows values
    # up to ~15-20. Avoids near-zero (no funnel) and implausibly large values.
    sigma_between = pm.Gamma("sigma_between", alpha=2, beta=0.5)

    # --- Non-centered group-level effects ---
    # Raw offsets on unit scale; avoids the funnel geometry
    # that arises when sigma_between is small and some schools have few data.
    school_offset = pm.Normal("school_offset", mu=0, sigma=1, dims="school")
    mu_school = pm.Deterministic(
        "mu_school",
        mu_global + school_offset * sigma_between,
        dims="school",
    )

    # --- Within-school SD ---
    # Gamma(2, 0.1): mean=20, allowing a broad range of residual noise.
    # Test score SDs of 8-15 are common; this prior comfortably covers that.
    sigma_within = pm.Gamma("sigma_within", alpha=2, beta=0.1)

    # --- Likelihood ---
    pm.Normal("likelihood", mu=mu_school[school_idx], sigma=sigma_within, observed=y, dims="obs")

    # -----------------------------------------------------------------
    # 3. Prior predictive check
    # -----------------------------------------------------------------
    prior_pred = pm.sample_prior_predictive(random_seed=rng)

print("\n" + "=" * 60)
print("PRIOR PREDICTIVE CHECK")
print("=" * 60)
fig, ax = plt.subplots(figsize=(8, 4))
az.plot_ppc(prior_pred, group="prior", num_pp_samples=100, ax=ax)
ax.set_title("Prior Predictive Check — Are priors plausible?")
ax.set_xlabel("Test score")
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
    "with_skill/outputs/prior_predictive.png",
    dpi=150,
)
plt.close(fig)

prior_scores = prior_pred.prior_predictive["likelihood"].values.flatten()
print(f"Prior predictive range: [{prior_scores.min():.0f}, {prior_scores.max():.0f}]")
print(f"Prior predictive mean:  {prior_scores.mean():.1f}")
print(f"Prior predictive SD:    {prior_scores.std():.1f}")

# -----------------------------------------------------------------
# 4. Inference
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("SAMPLING")
print("=" * 60)

with school_model:
    idata = pm.sample(nuts_sampler="nutpie", random_seed=rng)
    idata.extend(prior_pred)

    # Posterior predictive check
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))

    # Compute log-likelihood (nutpie does not auto-store it)
    pm.compute_log_likelihood(idata)

# Save immediately after sampling -- protect against late crashes
SAVE_PATH = (
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
    "with_skill/outputs/school_model_output.nc"
)
idata.to_netcdf(SAVE_PATH)
print(f"\nInferenceData saved to {SAVE_PATH}")

# -----------------------------------------------------------------
# 5. Convergence diagnostics
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 60)

summary = az.summary(idata, round_to=3)
print(summary)

# Automated checks
num_chains = idata.posterior.sizes["chain"]
rhat_ok = (summary["r_hat"] <= 1.01).all()
ess_bulk_ok = (summary["ess_bulk"] >= 100 * num_chains).all()
ess_tail_ok = (summary["ess_tail"] >= 100 * num_chains).all()
n_div = int(idata.sample_stats["diverging"].sum().item())

print(f"\nR-hat OK (all <= 1.01):       {rhat_ok}")
print(f"ESS bulk OK (>= {100 * num_chains}):    {ess_bulk_ok}  (min: {int(summary['ess_bulk'].min())})")
print(f"ESS tail OK (>= {100 * num_chains}):    {ess_tail_ok}  (min: {int(summary['ess_tail'].min())})")
print(f"Divergences:                  {n_div}")

if not (rhat_ok and ess_bulk_ok and ess_tail_ok and n_div == 0):
    print("\n*** WARNING: Some diagnostics failed. Results may be unreliable. ***")

# Trace / rank plots
fig = az.plot_trace(idata, var_names=["mu_global", "sigma_between", "sigma_within"], kind="rank_vlines")
plt.suptitle("Trace & Rank Plots — Key Parameters", y=1.02)
plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
    "with_skill/outputs/trace_rank_plots.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# Trace / rank plots for school-level means
fig = az.plot_trace(idata, var_names=["mu_school"], kind="rank_vlines")
plt.suptitle("Trace & Rank Plots — School Means", y=1.02)
plt.tight_layout()
plt.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
    "with_skill/outputs/trace_rank_school_means.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# Energy plot
fig, ax = plt.subplots(figsize=(6, 4))
az.plot_energy(idata, ax=ax)
ax.set_title("Energy Diagnostic")
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
    "with_skill/outputs/energy_plot.png",
    dpi=150,
)
plt.close(fig)

# -----------------------------------------------------------------
# 6. Model criticism
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL CRITICISM")
print("=" * 60)

# 6a. Posterior predictive check
fig, ax = plt.subplots(figsize=(8, 4))
az.plot_ppc(idata, num_pp_samples=100, ax=ax)
ax.set_title("Posterior Predictive Check")
ax.set_xlabel("Test score")
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
    "with_skill/outputs/posterior_predictive.png",
    dpi=150,
)
plt.close(fig)

# 6b. LOO-CV
print("\n--- LOO-CV ---")
loo = az.loo(idata, pointwise=True)
print(loo)

# Pareto k diagnostics
pareto_k_vals = loo.pareto_k.values
bad_obs = np.where(pareto_k_vals > 0.7)[0]
print(f"\nObservations with Pareto k > 0.7: {len(bad_obs)}")
if len(bad_obs) > 0:
    print(f"Indices: {bad_obs}")

fig, ax = plt.subplots(figsize=(8, 4))
az.plot_khat(loo, ax=ax)
ax.set_title("Pareto k Diagnostic (LOO)")
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
    "with_skill/outputs/pareto_k.png",
    dpi=150,
)
plt.close(fig)

# 6c. Calibration (PIT)
print("\n--- Calibration (PIT) ---")
if HAS_AZP:
    fig, ax = plt.subplots(figsize=(6, 5))
    azp.plot_ppc_pit(idata, ax=ax)
    ax.set_title("PPC-PIT Calibration")
    fig.tight_layout()
    fig.savefig(
        "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
        "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
        "with_skill/outputs/calibration_pit.png",
        dpi=150,
    )
    plt.close(fig)

    # Also LOO-PIT if available
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        azp.plot_ppc_pit(idata, loo_pit=True, ax=ax)
        ax.set_title("LOO-PIT Calibration")
        fig.tight_layout()
        fig.savefig(
            "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
            "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
            "with_skill/outputs/calibration_loo_pit.png",
            dpi=150,
        )
        plt.close(fig)
        print("LOO-PIT calibration plot saved.")
    except Exception as e:
        print(f"LOO-PIT plot skipped: {e}")
else:
    # Fallback: use az.plot_loo_pit
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        az.plot_loo_pit(idata, y="likelihood", ax=ax)
        ax.set_title("LOO-PIT Calibration")
        fig.tight_layout()
        fig.savefig(
            "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
            "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
            "with_skill/outputs/calibration_loo_pit.png",
            dpi=150,
        )
        plt.close(fig)
        print("LOO-PIT calibration plot saved (ArviZ fallback).")
    except Exception as e:
        print(f"Calibration plot skipped: {e}")

# -----------------------------------------------------------------
# 7. Hierarchical-specific diagnostics
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("HIERARCHICAL DIAGNOSTICS")
print("=" * 60)

# 7a. Shrinkage plot
# Observed (no-pooling) school means vs. posterior (partial-pooling) means
observed_means = np.array([
    scores[school_ids == j].mean() for j in range(N_SCHOOLS)
])
posterior_means = (
    idata.posterior["mu_school"]
    .mean(dim=["chain", "draw"])
    .to_numpy()
)
global_mean_posterior = float(
    idata.posterior["mu_global"].mean(dim=["chain", "draw"]).item()
)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(observed_means, posterior_means, s=sizes * 5, alpha=0.7, edgecolors="k", zorder=5)
for j in range(N_SCHOOLS):
    ax.annotate(
        f"n={sizes[j]}",
        (observed_means[j], posterior_means[j]),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=7,
    )
lims = [
    min(observed_means.min(), posterior_means.min()) - 3,
    max(observed_means.max(), posterior_means.max()) + 3,
]
ax.plot(lims, lims, "r--", linewidth=1, label="No pooling (y=x)")
ax.axhline(global_mean_posterior, color="gray", linestyle=":", linewidth=1, label="Complete pooling (grand mean)")
ax.set_xlabel("Observed school mean (no pooling)")
ax.set_ylabel("Posterior school mean (partial pooling)")
ax.set_title("Shrinkage Plot: Partial Pooling Toward Grand Mean")
ax.legend()
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
    "with_skill/outputs/shrinkage_plot.png",
    dpi=150,
)
plt.close(fig)

# 7b. ICC (Intraclass Correlation Coefficient)
sigma_between_draws = idata.posterior["sigma_between"].values.flatten()
sigma_within_draws = idata.posterior["sigma_within"].values.flatten()
icc_draws = sigma_between_draws ** 2 / (sigma_between_draws ** 2 + sigma_within_draws ** 2)

icc_mean = icc_draws.mean()
icc_hdi = az.hdi(icc_draws, hdi_prob=0.94)

print(f"\nIntraclass Correlation Coefficient (ICC):")
print(f"  Mean:    {icc_mean:.3f}")
print(f"  94% HDI: [{icc_hdi[0]:.3f}, {icc_hdi[1]:.3f}]")
print(
    f"\nInterpretation: ~{icc_mean * 100:.0f}% of the total variance in test scores "
    f"is attributable to differences between schools."
)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(icc_draws, bins=50, density=True, alpha=0.7, edgecolor="k")
ax.axvline(icc_mean, color="red", linestyle="--", label=f"Mean = {icc_mean:.3f}")
ax.axvspan(icc_hdi[0], icc_hdi[1], alpha=0.15, color="red", label=f"94% HDI [{icc_hdi[0]:.3f}, {icc_hdi[1]:.3f}]")
ax.set_xlabel("ICC")
ax.set_ylabel("Density")
ax.set_title("Posterior Distribution of ICC (Between-School / Total Variance)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
    "with_skill/outputs/icc_posterior.png",
    dpi=150,
)
plt.close(fig)

# 7c. Forest plot of school means
fig, ax = plt.subplots(figsize=(8, 6))
az.plot_forest(idata, var_names=["mu_school"], combined=True, ax=ax)
ax.set_title("School-Level Mean Estimates (94% HDI)")
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
    "with_skill/outputs/forest_school_means.png",
    dpi=150,
)
plt.close(fig)

# 7d. sigma_between posterior -- check it's not piling up at zero
fig, ax = plt.subplots(figsize=(6, 4))
az.plot_posterior(idata, var_names=["sigma_between"], ax=ax)
ax.set_title("Posterior: Between-School SD")
fig.tight_layout()
fig.savefig(
    "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
    "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
    "with_skill/outputs/sigma_between_posterior.png",
    dpi=150,
)
plt.close(fig)

# -----------------------------------------------------------------
# 8. Summary table of school estimates
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("SCHOOL-LEVEL ESTIMATES (Partial Pooling)")
print("=" * 60)

school_summary = az.summary(idata, var_names=["mu_school"], round_to=2)
school_summary["school"] = school_names
school_summary["n_students"] = sizes
school_summary["observed_mean"] = observed_means.round(2)
school_summary["true_mean"] = true_school_means.round(2)
school_summary["shrinkage"] = (
    1 - (posterior_means - global_mean_posterior) / (observed_means - global_mean_posterior + 1e-10)
).round(3)
print(school_summary.to_string())

# -----------------------------------------------------------------
# 9. Population-level estimates
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("POPULATION-LEVEL ESTIMATES")
print("=" * 60)

pop_summary = az.summary(
    idata,
    var_names=["mu_global", "sigma_between", "sigma_within"],
    round_to=3,
)
print(pop_summary)

print(f"\nTrue values for comparison:")
print(f"  mu_global:      {TRUE_MU_GLOBAL}")
print(f"  sigma_between:  {TRUE_SIGMA_BETWEEN}")
print(f"  sigma_within:   {TRUE_SIGMA_WITHIN}")

# -----------------------------------------------------------------
# 10. Model graph
# -----------------------------------------------------------------
try:
    graph = pm.model_to_graphviz(school_model)
    graph.render(
        "/Users/alex_andorra/tptm_alex/portfolio/agent-skills/"
        "bayesian-workflow-workspace/iteration-3/hierarchical-schools/"
        "with_skill/outputs/model_graph",
        format="png",
        cleanup=True,
    )
    print("\nModel graph saved.")
except Exception as e:
    print(f"\nModel graph skipped: {e}")

print("\n" + "=" * 60)
print("DONE. All outputs saved.")
print("=" * 60)
