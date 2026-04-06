"""
Amortized Bayesian inference for an SIR compartmental model with BayesFlow.

Parameters:
    beta  -- infection rate (positive)
    gamma -- recovery rate (positive)
    i0    -- initial infected fraction (0, 1)
    p     -- reporting probability (0, 1)

Observations:
    Daily reported case counts for 30 days (time series).

Architecture:
    - TimeSeriesTransformer (Small config) as summary network
    - FlowMatching (Small config) as inference network
    - Explicit bf.Adapter with .constrain() for all bounded parameters
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json
import sys

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np

# Reproducible seed
RANDOM_SEED = sum(map(ord, "compartmental-sir-v1"))
rng = np.random.default_rng(RANDOM_SEED)

# Add skill scripts to path for inspect_training and check_diagnostics
SKILL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..",
    "baygent-skills", "amortized-workflow",
)
sys.path.insert(0, os.path.abspath(SKILL_DIR))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================================================================
# 1. Define prior + observation model (online training)
# ==================================================================

POPULATION = 10_000
N_DAYS = 30


def prior():
    """Sample parameters from the prior."""
    beta = rng.gamma(shape=2.0, scale=0.3)       # infection rate, positive
    gamma = rng.gamma(shape=2.0, scale=0.15)      # recovery rate, positive
    i0 = rng.beta(a=2.0, b=50.0)                  # initial infected fraction, (0, 1)
    p = rng.beta(a=5.0, b=2.0)                    # reporting probability, (0, 1)
    return dict(beta=beta, gamma=gamma, i0=i0, p=p)


def observation_model(beta, gamma, i0, p):
    """Simulate daily reported cases from a discrete-time SIR model."""
    S = 1.0 - i0
    I = i0
    R = 0.0

    daily_cases = np.zeros(N_DAYS)
    for t in range(N_DAYS):
        new_infections = beta * S * I
        new_recoveries = gamma * I

        # Clamp to valid ranges
        new_infections = min(new_infections, S)
        new_recoveries = min(new_recoveries, I)

        S -= new_infections
        I += new_infections - new_recoveries
        R += new_recoveries

        # Reported cases = true new infections * reporting probability * population
        daily_cases[t] = rng.poisson(lam=max(new_infections * p * POPULATION, 1e-8))

    return dict(daily_cases=daily_cases)


simulator = bf.make_simulator([prior, observation_model])


# ==================================================================
# 2. Choose architecture -- Small config from model-sizes.md
# ==================================================================

# 4 parameters -> summary_dim = 2 * 4 = 8
# Daily cases are a TIME SERIES (30 days, ordered), not a set.
# Use TimeSeriesTransformer, Small config.
summary_net = bf.networks.TimeSeriesTransformer(
    summary_dim=8,
    embed_dims=(32, 32),
    num_heads=(2, 2),
    mlp_depths=(1, 1),
    mlp_widths=(64, 64),
    time_embed_dim=4,
    time_axis=1,
)

# FlowMatching, Small config
inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"widths": (128, 128), "time_embedding_dim": 16}
)


# ==================================================================
# 3. Build the adapter (MUST use bf.Adapter)
# ==================================================================

# Order: structural transforms -> constraints -> dtype -> concatenation
adapter = (
    bf.Adapter()
    # Structural: mark daily_cases as time series (30,) -> (30, 1)
    .as_time_series(["daily_cases"])
    # Parameter constraints: beta > 0, gamma > 0, i0 in (0,1), p in (0,1)
    .constrain("beta", lower=0)
    .constrain("gamma", lower=0)
    .constrain("i0", lower=0, upper=1)
    .constrain("p", lower=0, upper=1)
    # Dtype conversion (NumPy float64 -> float32 for JAX)
    .convert_dtype("float64", "float32")
    # Concatenate into reserved BayesFlow keys
    .concatenate(["beta", "gamma", "i0", "p"], into="inference_variables")
    .concatenate(["daily_cases"], into="summary_variables")
)


# ==================================================================
# 4. Create workflow
# ==================================================================

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath=os.path.join(OUTPUT_DIR, "checkpoints"),
    checkpoint_name="sir_compartmental",
)


# ==================================================================
# 5. Simulation sanity checks
# ==================================================================

print("=== Simulation sanity checks ===")
test_sims = workflow.simulate(10)
for key in test_sims:
    arr = np.asarray(test_sims[key])
    print(f"  {key}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, "
          f"mean={arr.mean():.4f}")
print()


# ==================================================================
# 6. Train the amortizer
# ==================================================================

history = workflow.fit_online(
    epochs=50,
    batch_size=32,
    num_batches_per_epoch=100,
    validation_data=300,
    verbose=2,
)

# --- Mandatory: save history and inspect training convergence ---
history_path = os.path.join(OUTPUT_DIR, "history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)

from scripts.inspect_training import inspect_history

training_report = inspect_history(history.history)
print("\n=== Training Inspection Report ===")
print(json.dumps(training_report, indent=2))

if not training_report["overall"]["ok"]:
    print("\nTRAINING ISSUES -- address before continuing:")
    for issue in training_report["overall"]["issues"]:
        print(f"  - {issue}")


# ==================================================================
# 7. In-silico diagnostics on held-out simulations
# ==================================================================

# MUST use workflow.simulate() -- NEVER loop over simulator() manually.
test_data = workflow.simulate(300)

# plot_default_diagnostics ALWAYS returns a dict[str, Figure].
figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig_path = os.path.join(OUTPUT_DIR, f"diagnostics_{name}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)

# --- Mandatory: save diagnostics and check house thresholds ---
from scripts.check_diagnostics import check_diagnostics

metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
metrics.to_csv(metrics_path)
print("\n=== Diagnostics Table ===")
print(metrics)

diag_report = check_diagnostics(metrics)
print("\n=== Diagnostics Report ===")
print(json.dumps(diag_report, indent=2))

if diag_report["overall"]["decision"] == "STOP":
    raise RuntimeError(
        "Diagnostics STOP: " + diag_report["overall"]["recommendation"]
    )
elif diag_report["overall"]["decision"] == "WARN":
    print("\nWARNING: " + diag_report["overall"]["recommendation"])
else:
    print("\nDiagnostics GO -- safe to proceed to real-data inference.")


# ==================================================================
# 8. Posterior predictive checks (template -- requires real data)
# ==================================================================

# Uncomment when real observed data is available:
#
# x_obs = np.array([...])  # shape (30,) -- daily reported case counts
# real_data = {"daily_cases": x_obs}
# samples = workflow.sample(conditions=real_data, num_samples=1000)
#
# # samples["beta"].shape  == (1, 1000, 1)
# # samples["gamma"].shape == (1, 1000, 1)
# # samples["i0"].shape    == (1, 1000, 1)
# # samples["p"].shape     == (1, 1000, 1)
#
# # PPC: reuse the observation_model -- NEVER re-implement
# n_ppc = 50
# ppc_curves = []
# for s in range(n_ppc):
#     theta_s = {
#         "beta": float(samples["beta"][0, s]),
#         "gamma": float(samples["gamma"][0, s]),
#         "i0": float(samples["i0"][0, s]),
#         "p": float(samples["p"][0, s]),
#     }
#     x_rep = observation_model(**theta_s)
#     ppc_curves.append(x_rep["daily_cases"])
#
# ppc_curves = np.array(ppc_curves)
# fig, ax = plt.subplots(figsize=(10, 5))
# for curve in ppc_curves:
#     ax.plot(range(N_DAYS), curve, alpha=0.15, color="steelblue")
# ax.plot(range(N_DAYS), x_obs, color="black", linewidth=2, label="Observed")
# ax.set_xlabel("Day")
# ax.set_ylabel("Reported cases")
# ax.set_title("Posterior Predictive Check")
# ax.legend()
# fig.savefig(os.path.join(OUTPUT_DIR, "ppc.png"), dpi=150, bbox_inches="tight")
# plt.close(fig)

print("\n=== Workflow complete ===")
