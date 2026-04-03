"""
Amortized Bayesian inference for a Gaussian location-scale model.

Generative model:
    mu    ~ Normal(0, 5)
    sigma ~ HalfNormal(2)       [equivalently: |Normal(0, 2)|]
    x_i   ~ Normal(mu, sigma)   for i = 1, ..., 50  (i.i.d.)

We train a BayesFlow amortized posterior estimator so that, given any
new dataset of 50 observations, we can obtain approximate posterior
samples for (mu, sigma) in a single forward pass.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json
import sys

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 0. Paths and reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = sum(map(ord, "gaussian-location-scale-v1"))
rng = np.random.default_rng(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_SCRIPTS = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    os.pardir, os.pardir, os.pardir, os.pardir,
    "baygent-skills", "amortized-workflow", "scripts",
)
SKILL_SCRIPTS = os.path.normpath(SKILL_SCRIPTS)
sys.path.insert(0, SKILL_SCRIPTS)

from check_diagnostics import check_diagnostics
from inspect_training import inspect_history

# ---------------------------------------------------------------------------
# 1. Define prior + observation model
# ---------------------------------------------------------------------------

def prior(rng=None):
    """Prior over (mu, sigma)."""
    rng = rng or np.random.default_rng()
    mu = rng.normal(0.0, 5.0)
    sigma = np.abs(rng.normal(0.0, 2.0))
    return {"mu": np.array(mu), "sigma": np.array(sigma)}


def observation_model(mu, sigma, rng=None):
    """Likelihood: 50 i.i.d. draws from Normal(mu, sigma)."""
    rng = rng or np.random.default_rng()
    x = rng.normal(mu, sigma, size=50)
    return {"x": x}


simulator = bf.make_simulator([prior, observation_model])

# ---------------------------------------------------------------------------
# 2. Simulation sanity check
# ---------------------------------------------------------------------------

print("=== Simulation sanity check ===")
test_sims = simulator.sample(5, rng=rng)
for key, val in test_sims.items():
    print(f"  {key}: shape={np.shape(val)}, dtype={np.asarray(val).dtype}")
# Spot-check a few simulations
for i in range(3):
    mu_i = float(test_sims["mu"][i])
    sigma_i = float(test_sims["sigma"][i])
    x_i = test_sims["x"][i]
    print(f"  sim {i}: mu={mu_i:.2f}, sigma={sigma_i:.2f}, "
          f"x_mean={np.mean(x_i):.2f}, x_std={np.std(x_i):.2f}")
print()

# ---------------------------------------------------------------------------
# 3. Architecture — Small config (2 parameters => summary_dim=4)
# ---------------------------------------------------------------------------

summary_net = bf.networks.SetTransformer(
    summary_dim=4,
    embed_dims=(32, 32),
    num_heads=(2, 2),
    mlp_depths=(1, 1),
    mlp_widths=(64, 64),
)

inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"widths": (128, 128), "time_embedding_dim": 16},
)

# ---------------------------------------------------------------------------
# 4. Adapter
# ---------------------------------------------------------------------------

adapter = (
    bf.Adapter()
    .as_set(["x"])                          # (50,) -> (50, 1) for SetTransformer
    .constrain("sigma", lower=0)            # sigma > 0
    .convert_dtype("float64", "float32")
    .concatenate(["x"], into="summary_variables")
    .concatenate(["mu", "sigma"], into="inference_variables")
)

# ---------------------------------------------------------------------------
# 5. Workflow
# ---------------------------------------------------------------------------

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath=os.path.join(SCRIPT_DIR, "checkpoints"),
    checkpoint_name="gaussian_location_scale",
)

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------

print("=== Training ===")
history = workflow.fit_online(
    epochs=100,
    batch_size=32,
    num_batches_per_epoch=100,
    validation_data=300,
    verbose=2,
)

# --- Save and inspect training history ---
history_path = os.path.join(SCRIPT_DIR, "history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)
print(f"\nTraining history saved to {history_path}")

training_report = inspect_history(history.history)
print("\n=== Training inspection ===")
print(json.dumps(training_report, indent=2))

if not training_report["overall"]["ok"]:
    print("\nTRAINING ISSUES -- address before continuing:")
    for issue in training_report["overall"]["issues"]:
        print(f"  - {issue}")

# ---------------------------------------------------------------------------
# 7. In-silico diagnostics
# ---------------------------------------------------------------------------

print("\n=== In-silico diagnostics ===")
test_data = workflow.simulate(300)

# Visual diagnostics
figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig_path = os.path.join(SCRIPT_DIR, f"diagnostics_{name}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: diagnostics_{name}.png")

# Numerical diagnostics
metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
metrics_path = os.path.join(SCRIPT_DIR, "metrics.csv")
metrics.to_csv(metrics_path)
print(f"\nDiagnostics table:\n{metrics}\n")

# --- Check against house thresholds ---
diag_report = check_diagnostics(metrics)
print("=== Diagnostics check ===")
print(json.dumps(diag_report, indent=2))

if diag_report["overall"]["decision"] == "STOP":
    raise RuntimeError(
        "Diagnostics STOP: " + diag_report["overall"]["recommendation"]
    )
elif diag_report["overall"]["decision"] == "WARN":
    print("\nWARNING: " + diag_report["overall"]["recommendation"])

# ---------------------------------------------------------------------------
# 8. Amortized inference on a test dataset
# ---------------------------------------------------------------------------

print("\n=== Inference on test dataset ===")

# Generate a test dataset with known ground truth
true_mu = 2.5
true_sigma = 1.3
x_obs = rng.normal(true_mu, true_sigma, size=50)
print(f"Ground truth: mu={true_mu}, sigma={true_sigma}")
print(f"Observed: mean={np.mean(x_obs):.3f}, std={np.std(x_obs):.3f}")

# Amortized posterior sampling (instant)
real_data = {"x": x_obs}
samples = workflow.sample(conditions=real_data, num_samples=1000)

# The adapter reverse-transforms: keys are "mu" and "sigma", NOT "inference_variables"
mu_samples = samples["mu"][0, :, 0]    # (1, 1000, 1) -> (1000,)
sigma_samples = samples["sigma"][0, :, 0]

print(f"\nPosterior mu:    mean={np.mean(mu_samples):.3f}, "
      f"std={np.std(mu_samples):.3f}, "
      f"95% CI=[{np.percentile(mu_samples, 2.5):.3f}, {np.percentile(mu_samples, 97.5):.3f}]")
print(f"Posterior sigma: mean={np.mean(sigma_samples):.3f}, "
      f"std={np.std(sigma_samples):.3f}, "
      f"95% CI=[{np.percentile(sigma_samples, 2.5):.3f}, {np.percentile(sigma_samples, 97.5):.3f}]")

# ---------------------------------------------------------------------------
# 9. Posterior predictive checks (reusing simulator functions)
# ---------------------------------------------------------------------------

print("\n=== Posterior predictive checks ===")
n_ppc = 50
ppc_means = []
ppc_stds = []

for s in range(n_ppc):
    theta_s = {
        "mu": float(samples["mu"][0, s, 0]),
        "sigma": float(samples["sigma"][0, s, 0]),
    }
    x_rep = observation_model(**theta_s)["x"]
    ppc_means.append(np.mean(x_rep))
    ppc_stds.append(np.std(x_rep))

obs_mean = np.mean(x_obs)
obs_std = np.std(x_obs)

print(f"Observed mean: {obs_mean:.3f}")
print(f"PPC means:     median={np.median(ppc_means):.3f}, "
      f"95% interval=[{np.percentile(ppc_means, 2.5):.3f}, {np.percentile(ppc_means, 97.5):.3f}]")
print(f"Observed std:  {obs_std:.3f}")
print(f"PPC stds:      median={np.median(ppc_stds):.3f}, "
      f"95% interval=[{np.percentile(ppc_stds, 2.5):.3f}, {np.percentile(ppc_stds, 97.5):.3f}]")

# --- PPC figure ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(ppc_means, bins=20, alpha=0.6, label="PPC means")
axes[0].axvline(obs_mean, color="red", linewidth=2, label=f"Observed ({obs_mean:.2f})")
axes[0].set_xlabel("Sample mean")
axes[0].set_title("PPC: Sample Mean")
axes[0].legend()

axes[1].hist(ppc_stds, bins=20, alpha=0.6, label="PPC stds")
axes[1].axvline(obs_std, color="red", linewidth=2, label=f"Observed ({obs_std:.2f})")
axes[1].set_xlabel("Sample std")
axes[1].set_title("PPC: Sample Std")
axes[1].legend()

plt.tight_layout()
ppc_path = os.path.join(SCRIPT_DIR, "ppc_checks.png")
fig.savefig(ppc_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPPC figure saved to {ppc_path}")

print("\n=== Done ===")
