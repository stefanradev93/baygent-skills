"""
Offline amortized Bayesian inference from a pre-generated simulation bank.

The simulator is proprietary and unavailable. We train a BayesFlow amortized
posterior estimator from 20,000 pre-simulated (parameter, observable) pairs
stored as .npz files on disk.

Each file contains:
  - 'parameters': shape (3,)
  - 'observables': shape (50,) -- 50 i.i.d. measurements

We hold out data for validation (loss tracking) and test (diagnostics).
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json
from pathlib import Path

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

DATA_DIR = Path("simulation_bank")  # directory containing .npz files
OUTPUT_DIR = Path(".")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

N_TOTAL = 20_000
N_VAL = 1_000
N_TEST = 1_000
N_TRAIN = N_TOTAL - N_VAL - N_TEST  # 18,000

NUM_PARAMS = 3
NUM_OBS = 50

# ---------------------------------------------------------------------------
# 1. Load simulation bank
# ---------------------------------------------------------------------------

print(f"Loading simulation bank from {DATA_DIR} ...")

npz_files = sorted(DATA_DIR.glob("*.npz"))
if len(npz_files) < N_TOTAL:
    raise FileNotFoundError(
        f"Expected {N_TOTAL} .npz files but found {len(npz_files)}."
    )

all_parameters = np.empty((N_TOTAL, NUM_PARAMS), dtype=np.float64)
all_observables = np.empty((N_TOTAL, NUM_OBS), dtype=np.float64)

for i, fpath in enumerate(npz_files[:N_TOTAL]):
    data = np.load(fpath)
    all_parameters[i] = data["parameters"]
    all_observables[i] = data["observables"]

print(f"Loaded {N_TOTAL} simulations.")
print(f"  parameters  shape: {all_parameters.shape}")
print(f"  observables shape: {all_observables.shape}")

# Quick sanity check
for j in range(NUM_PARAMS):
    col = all_parameters[:, j]
    print(f"  param[{j}]: mean={col.mean():.4f}, std={col.std():.4f}, "
          f"range=[{col.min():.4f}, {col.max():.4f}]")

assert not np.any(np.isnan(all_parameters)), "NaN in parameters!"
assert not np.any(np.isnan(all_observables)), "NaN in observables!"
print("  No NaN detected.\n")

# ---------------------------------------------------------------------------
# 2. Train / validation / test split
# ---------------------------------------------------------------------------

indices = rng.permutation(N_TOTAL)
train_idx = indices[:N_TRAIN]
val_idx = indices[N_TRAIN : N_TRAIN + N_VAL]
test_idx = indices[N_TRAIN + N_VAL :]

train_data = {
    "parameters": all_parameters[train_idx],
    "observables": all_observables[train_idx],
}
val_data = {
    "parameters": all_parameters[val_idx],
    "observables": all_observables[val_idx],
}
test_data = {
    "parameters": all_parameters[test_idx],
    "observables": all_observables[test_idx],
}

print(f"Split: train={N_TRAIN}, val={N_VAL}, test={N_TEST}")

# ---------------------------------------------------------------------------
# 3. Adapter
# ---------------------------------------------------------------------------

# Observables are 50 i.i.d. measurements (exchangeable), so we treat them
# as set data for the summary network.
adapter = (
    bf.Adapter()
    .as_set(["observables"])
    .convert_dtype("float64", "float32")
    .concatenate(["observables"], into="summary_variables")
    .concatenate(["parameters"], into="inference_variables")
)

# ---------------------------------------------------------------------------
# 4. Networks
# ---------------------------------------------------------------------------

# SetTransformer to summarize the 50 exchangeable observations.
# summary_dim = 2 * num_params = 6 is a reasonable starting point.
summary_net = bf.networks.SetTransformer(
    summary_dim=2 * NUM_PARAMS,
    embed_dims=(64, 64),
    num_heads=(4, 4),
    mlp_depths=(2, 2),
    mlp_widths=(128, 128),
)

# Flow matching for the conditional density estimator.
inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"widths": (256, 256), "time_embedding_dim": 32},
)

# ---------------------------------------------------------------------------
# 5. Workflow (offline -- no simulator)
# ---------------------------------------------------------------------------

workflow = bf.BasicWorkflow(
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath=str(CHECKPOINT_DIR),
    checkpoint_name="offline_estimator",
)

# ---------------------------------------------------------------------------
# 6. Train offline
# ---------------------------------------------------------------------------

print("\nStarting offline training ...")
history = workflow.fit_offline(
    data=train_data,
    epochs=150,
    batch_size=64,
    validation_data=val_data,
    verbose=2,
)

# Save training history
history_path = OUTPUT_DIR / "history.json"
with open(history_path, "w") as f:
    json.dump(history.history, f)
print(f"\nTraining history saved to {history_path}")

# Basic convergence check
train_loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

if train_loss:
    print(f"  Final train loss: {train_loss[-1]:.4f}")
if val_loss:
    print(f"  Final val loss:   {val_loss[-1]:.4f}")
    if val_loss[-1] > val_loss[-10] * 1.05:
        print("  WARNING: Validation loss may be increasing -- possible overfitting.")

# Plot loss curves
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_loss, label="Train")
if val_loss:
    ax.plot(val_loss, label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Curves")
ax.legend()
plt.tight_layout()
fig.savefig(str(OUTPUT_DIR / "training_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Training curves saved.")

# ---------------------------------------------------------------------------
# 7. Diagnostics on held-out test set
# ---------------------------------------------------------------------------

print("\nRunning diagnostics on held-out test set ...")

# Visual diagnostics
figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig_path = OUTPUT_DIR / f"diagnostics_{name}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")

# Numerical diagnostics
metrics = workflow.compute_default_diagnostics(
    test_data=test_data, as_data_frame=True
)
metrics_path = OUTPUT_DIR / "metrics.csv"
metrics.to_csv(str(metrics_path))
print(f"\nDiagnostics:\n{metrics}\n")

# ---------------------------------------------------------------------------
# 8. Example: inference on new data
# ---------------------------------------------------------------------------

# When new real observations become available:
#
#   real_obs = np.load("new_data.npz")["observables"]  # shape (50,)
#   conditions = {"observables": real_obs[np.newaxis, :]}  # add batch dim
#   samples = workflow.sample(conditions=conditions, num_samples=2000)
#
# `samples` is a dict with reverse-transformed parameter names.

# ---------------------------------------------------------------------------
# 9. Note on posterior predictive checks
# ---------------------------------------------------------------------------

# PPCs require running data through the original simulator with posterior
# parameter draws. Since the simulator is proprietary and unavailable,
# PPCs cannot be performed. This is a limitation of offline-only training.
# If/when the simulator becomes available, PPCs should be run immediately.

print("NOTE: Posterior predictive checks cannot be performed without the simulator.")
print("\nDone.")
