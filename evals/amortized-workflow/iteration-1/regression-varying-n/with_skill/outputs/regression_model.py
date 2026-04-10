"""
Amortized Bayesian inference for simple linear regression with variable N.

Model: y = alpha + beta * x + noise(sigma)
       N ~ Uniform{10, ..., 200}  (varies across datasets)

The network learns to infer (alpha, beta, sigma) from any dataset with
10 <= N <= 200, without retraining.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import sys
import json
import numpy as np
import bayesflow as bf
import matplotlib.pyplot as plt

# Add scripts to path for inspect_training and check_diagnostics
SKILL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..",
    "baygent-skills", "amortized-workflow",
)
sys.path.insert(0, os.path.abspath(SKILL_DIR))

from scripts.inspect_training import inspect_history
from scripts.check_diagnostics import check_diagnostics

# ── Output directory ─────────────────────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

RANDOM_SEED = sum(map(ord, "linear-regression-variable-n"))
rng = np.random.default_rng(RANDOM_SEED)

# ------------------------------------------------------------------
# 1. Define prior + observation model + meta function
# ------------------------------------------------------------------


def prior():
    alpha = rng.normal(0, 5)
    beta = rng.normal(0, 3)
    sigma = rng.gamma(2, 1)  # shape=2, scale=1 => mean=2, mode=1
    return dict(alpha=alpha, beta=beta, sigma=sigma)


def observation_model(alpha, beta, sigma, N):
    x = rng.normal(0, 1, size=N)
    y = rng.normal(alpha + beta * x, sigma, size=N)
    return dict(x=x, y=y)


def meta_fn():
    """Sample N uniformly from [10, 200] — varies between batches."""
    return dict(N=rng.integers(10, 201))  # upper is exclusive


simulator = bf.make_simulator([prior, observation_model], meta_fn=meta_fn)

# ------------------------------------------------------------------
# 2. Simulation sanity check
# ------------------------------------------------------------------

print("=== Simulation sanity check ===")
test_sim = simulator.sample(5)
for key, val in test_sim.items():
    if isinstance(val, np.ndarray):
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
              f"min={val.min():.3f}, max={val.max():.3f}")
    else:
        print(f"  {key}: {val}")
print()

# ------------------------------------------------------------------
# 3. Choose architecture — SMALL config (3 params => summary_dim=6)
# ------------------------------------------------------------------

# SetTransformer Small: embed_dims=(32,32), num_heads=(2,2),
#   mlp_depths=(1,1), mlp_widths=(64,64)
# summary_dim = 2 * num_params = 2 * 3 = 6
summary_net = bf.networks.SetTransformer(
    summary_dim=6,
    embed_dims=(32, 32),
    num_heads=(2, 2),
    mlp_depths=(1, 1),
    mlp_widths=(64, 64),
)

# FlowMatching Small: widths=(128,128), time_embedding_dim=16
inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"widths": (128, 128), "time_embedding_dim": 16}
)

# ------------------------------------------------------------------
# 4. Build adapter
# ------------------------------------------------------------------
# Simulator returns: {alpha, beta, sigma, N, x, y}
#
# Routing:
#   - (x, y) are exchangeable pairs => summary_variables via SetTransformer
#   - N is a scalar context variable => broadcast, sqrt, then inference_conditions
#   - (alpha, beta, sigma) => inference_variables; sigma constrained > 0

adapter = (
    bf.Adapter()
    .broadcast("N", to="x")           # replicate N along batch dim
    .as_set(["x", "y"])               # (N,) -> (N, 1) for SetTransformer
    .constrain("sigma", lower=0)      # softplus bijection for positivity
    .sqrt("N")                        # sqrt(N) is a smoother signal for the network
    .convert_dtype("float64", "float32")
    .concatenate(["alpha", "beta", "sigma"], into="inference_variables")
    .concatenate(["x", "y"], into="summary_variables")
    .rename("N", "inference_conditions")
)

# ------------------------------------------------------------------
# 5. Create workflow
# ------------------------------------------------------------------

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath=os.path.join(OUTPUT_DIR, "checkpoints"),
    checkpoint_name="regression_variable_n",
)

# ------------------------------------------------------------------
# 6. Train
# ------------------------------------------------------------------

history = workflow.fit_online(
    epochs=100,
    batch_size=32,
    num_batches_per_epoch=100,
    validation_data=300,
    verbose=2,
)

# ── Save and inspect training history ────────────────────────────
history_path = os.path.join(OUTPUT_DIR, "history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)

training_report = inspect_history(history.history)
print("\n=== Training inspection ===")
print(json.dumps(training_report, indent=2))

if not training_report["overall"]["ok"]:
    print("\nTRAINING ISSUES — address before continuing:")
    for issue in training_report["overall"]["issues"]:
        print(f"  - {issue}")

# ------------------------------------------------------------------
# 7. In-silico diagnostics on held-out simulations
# ------------------------------------------------------------------

test_data = workflow.simulate(300)

# ── Visual diagnostics ───────────────────────────────────────────
figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"diagnostics_{name}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

# ── Numerical diagnostics ───────────────────────────────────────
metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)

metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
metrics.to_csv(metrics_path)
print("\n=== Diagnostics table ===")
print(metrics)

diag_report = check_diagnostics(metrics)
print("\n=== Diagnostics check ===")
print(json.dumps(diag_report, indent=2))

if diag_report["overall"]["decision"] == "STOP":
    raise RuntimeError(diag_report["overall"]["recommendation"])
elif diag_report["overall"]["decision"] == "WARN":
    print("\nWARNING:", diag_report["overall"]["recommendation"])

# ------------------------------------------------------------------
# 8. Example: amortized inference on a synthetic "real" dataset
# ------------------------------------------------------------------

# Generate a single observed dataset with known ground truth
TRUE_ALPHA, TRUE_BETA, TRUE_SIGMA = 2.5, -1.3, 0.8
N_OBS = 50
x_obs = rng.normal(0, 1, size=N_OBS)
y_obs = rng.normal(TRUE_ALPHA + TRUE_BETA * x_obs, TRUE_SIGMA, size=N_OBS)

# Package as the simulator would — keys must match adapter expectations
real_data = {
    "x": x_obs,
    "y": y_obs,
    "N": N_OBS,
}

samples = workflow.sample(conditions=real_data, num_samples=1000)

# samples has original parameter names: alpha, beta, sigma
print("\n=== Posterior summary (ground truth: alpha=2.5, beta=-1.3, sigma=0.8) ===")
for param in ["alpha", "beta", "sigma"]:
    draws = samples[param].squeeze()
    print(
        f"  {param}: mean={draws.mean():.3f}, "
        f"std={draws.std():.3f}, "
        f"95% CI=[{np.percentile(draws, 2.5):.3f}, {np.percentile(draws, 97.5):.3f}]"
    )

# ------------------------------------------------------------------
# 9. Posterior predictive checks (reuse observation_model)
# ------------------------------------------------------------------

n_ppc = 50
ppc_y_means = []
ppc_y_stds = []

for s in range(n_ppc):
    alpha_s = float(samples["alpha"][0, s])
    beta_s = float(samples["beta"][0, s])
    sigma_s = float(samples["sigma"][0, s])
    # Reuse the observation model — never re-implement
    y_rep = rng.normal(alpha_s + beta_s * x_obs, sigma_s, size=N_OBS)
    ppc_y_means.append(y_rep.mean())
    ppc_y_stds.append(y_rep.std())

print("\n=== Posterior predictive checks ===")
print(f"  Observed y: mean={y_obs.mean():.3f}, std={y_obs.std():.3f}")
print(
    f"  PPC y means: [{np.percentile(ppc_y_means, 2.5):.3f}, "
    f"{np.percentile(ppc_y_means, 97.5):.3f}]"
)
print(
    f"  PPC y stds:  [{np.percentile(ppc_y_stds, 2.5):.3f}, "
    f"{np.percentile(ppc_y_stds, 97.5):.3f}]"
)

print("\nDone. All outputs saved to:", OUTPUT_DIR)
