"""
Offline amortized Bayesian inference from a pre-generated simulation bank.

The simulator is proprietary and unavailable. We train BayesFlow from 20,000
pre-simulated (parameter, observable) pairs stored as .npz files on disk.
Each file contains:
  - 'parameters': shape (3,)
  - 'observables': shape (50,) — 50 i.i.d. measurements

Training mode: offline (all data loaded into memory).
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json
import sys
from pathlib import Path

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = sum(map(ord, "offline-simulation-bank-v1"))
rng = np.random.default_rng(RANDOM_SEED)

DATA_DIR = Path("simulation_bank")  # directory of .npz files
OUTPUT_DIR = Path(".")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

N_TOTAL = 20_000
N_TEST = 500       # held-out for diagnostics — never seen during training
N_VAL = 500        # held-out for validation loss tracking
N_TRAIN = N_TOTAL - N_TEST - N_VAL  # 19,000

NUM_PARAMS = 3
NUM_OBS = 50

# Skill: MUST start with Small config from references/model-sizes.md
# summary_dim heuristic: 2x number of parameters = 6
SUMMARY_DIM = 2 * NUM_PARAMS  # 6

# ---------------------------------------------------------------------------
# 1. Load simulation bank from .npz files
# ---------------------------------------------------------------------------

print(f"Loading {N_TOTAL} simulations from {DATA_DIR} ...")

npz_files = sorted(DATA_DIR.glob("*.npz"))
if len(npz_files) < N_TOTAL:
    print(
        f"WARNING: Found {len(npz_files)} files but expected {N_TOTAL}. "
        f"Using all {len(npz_files)} files."
    )
    N_TOTAL = len(npz_files)
    N_TEST = min(500, N_TOTAL // 10)
    N_VAL = min(500, N_TOTAL // 10)
    N_TRAIN = N_TOTAL - N_TEST - N_VAL

all_parameters = np.empty((N_TOTAL, NUM_PARAMS), dtype=np.float64)
all_observables = np.empty((N_TOTAL, NUM_OBS), dtype=np.float64)

for i, fpath in enumerate(npz_files[:N_TOTAL]):
    data = np.load(fpath)
    all_parameters[i] = data["parameters"]
    all_observables[i] = data["observables"]

print(f"Loaded {N_TOTAL} simulations.")
print(f"  parameters  shape: {all_parameters.shape}, range: [{all_parameters.min():.3f}, {all_parameters.max():.3f}]")
print(f"  observables shape: {all_observables.shape}, range: [{all_observables.min():.3f}, {all_observables.max():.3f}]")

# ---------------------------------------------------------------------------
# 2. Simulation sanity checks (mandatory before training)
# ---------------------------------------------------------------------------

print("\n=== Simulation Sanity Checks ===")
for j in range(NUM_PARAMS):
    col = all_parameters[:, j]
    print(f"  param[{j}]: mean={col.mean():.4f}, std={col.std():.4f}, "
          f"min={col.min():.4f}, max={col.max():.4f}")

obs_means = all_observables.mean(axis=1)
obs_stds = all_observables.std(axis=1)
print(f"  obs per-sim mean: mean={obs_means.mean():.4f}, std={obs_means.std():.4f}")
print(f"  obs per-sim std:  mean={obs_stds.mean():.4f}, std={obs_stds.std():.4f}")

# Check for NaN / Inf
assert not np.any(np.isnan(all_parameters)), "NaN found in parameters!"
assert not np.any(np.isnan(all_observables)), "NaN found in observables!"
assert not np.any(np.isinf(all_parameters)), "Inf found in parameters!"
assert not np.any(np.isinf(all_observables)), "Inf found in observables!"
print("  No NaN or Inf detected.")
print("================================\n")

# ---------------------------------------------------------------------------
# 3. Split into train / validation / test
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

print(f"Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")

# ---------------------------------------------------------------------------
# 4. Build adapter (MUST use bf.Adapter — never mix with naming shorthand)
# ---------------------------------------------------------------------------

# Observables are 50 i.i.d. measurements — exchangeable set data.
# Skill: route through summary_variables with SetTransformer. NEVER inference_conditions.
# .as_set converts (50,) -> (50, 1) so the SetTransformer gets (batch, 50, 1).
adapter = (
    bf.Adapter()
    .as_set(["observables"])
    .convert_dtype("float64", "float32")
    .concatenate(["observables"], into="summary_variables")
    .concatenate(["parameters"], into="inference_variables")
)

# ---------------------------------------------------------------------------
# 5. Choose architecture — MUST start Small
# ---------------------------------------------------------------------------

# SetTransformer Small config from references/model-sizes.md
summary_net = bf.networks.SetTransformer(
    summary_dim=SUMMARY_DIM,
    embed_dims=(32, 32),
    num_heads=(2, 2),
    mlp_depths=(1, 1),
    mlp_widths=(64, 64),
)

# FlowMatching Small config
inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"widths": (128, 128), "time_embedding_dim": 16}
)

# ---------------------------------------------------------------------------
# 6. Create workflow (offline — no simulator needed)
# ---------------------------------------------------------------------------

workflow = bf.BasicWorkflow(
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath=str(CHECKPOINT_DIR),
    checkpoint_name="offline_estimator",
)

# ---------------------------------------------------------------------------
# 7. Train with fit_offline (MUST pass validation_data)
# ---------------------------------------------------------------------------

print("\nStarting offline training ...")
history = workflow.fit_offline(
    data=train_data,
    epochs=100,
    batch_size=32,
    validation_data=val_data,
    verbose=2,
)

# --- Mandatory: save history and inspect training convergence ---
history_path = OUTPUT_DIR / "history.json"
with open(history_path, "w") as f:
    json.dump(history.history, f)
print(f"\nTraining history saved to {history_path}")

# Add scripts directory to path for imports
SKILL_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "baygent-skills" / "amortized-workflow"
sys.path.insert(0, str(SKILL_DIR))
from scripts.inspect_training import inspect_history

training_report = inspect_history(history.history)
print("\n=== Training Convergence Report ===")
print(json.dumps(training_report, indent=2))

if not training_report["overall"]["ok"]:
    print("\nTRAINING ISSUES — address before continuing:")
    for issue in training_report["overall"]["issues"]:
        print(f"  - {issue}")

# ---------------------------------------------------------------------------
# 8. In-silico diagnostics on HELD-OUT test set (never training data)
# ---------------------------------------------------------------------------

# Skill: MUST use built-in diagnostics — NEVER hand-roll coverage/bias/calibration.
# Since we have no simulator, we use our pre-split test_data dict directly.
print("\nRunning in-silico diagnostics on held-out test set ...")

# plot_default_diagnostics ALWAYS returns a dict[str, Figure].
figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig_path = OUTPUT_DIR / f"diagnostics_{name}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")

metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)

# --- Mandatory: save diagnostics and check house thresholds ---
metrics_path = OUTPUT_DIR / "metrics.csv"
metrics.to_csv(str(metrics_path))
print(f"\nDiagnostics saved to {metrics_path}")
print("\n=== Diagnostics Table ===")
print(metrics.to_string())
print("=========================\n")

from scripts.check_diagnostics import check_diagnostics

diag_report = check_diagnostics(metrics)
print("=== Diagnostics Report ===")
print(json.dumps(diag_report, indent=2))

if diag_report["overall"]["decision"] == "STOP":
    raise RuntimeError(
        "Diagnostics FAILED — do not proceed to real-data inference. "
        + diag_report["overall"]["recommendation"]
    )
elif diag_report["overall"]["decision"] == "WARN":
    print("\nWARNING: Some diagnostics are marginal. Review before proceeding.")
else:
    print("\nDiagnostics passed. Safe to proceed to real-data inference.")

# ---------------------------------------------------------------------------
# 9. Amortized inference on new observations (example usage)
# ---------------------------------------------------------------------------

# When new real observations become available, run inference like this:
#
# real_obs = np.load("new_observation.npz")["observables"]  # shape (50,)
# real_data = {"observables": real_obs[np.newaxis, :]}       # add batch dim -> (1, 50)
# samples = workflow.sample(conditions=real_data, num_samples=1000)
#
# The returned dict has the ORIGINAL parameter names from the adapter's
# reverse transform — NOT "inference_variables". Each value has shape
# (batch, num_samples, d).

# ---------------------------------------------------------------------------
# 10. Posterior predictive checks — NOT POSSIBLE
# ---------------------------------------------------------------------------

# PPCs require re-simulating data from posterior draws through the original
# observation model. Since the simulator is proprietary and unavailable,
# PPCs cannot be performed. This is a known limitation of this analysis.
# If the simulator becomes available again, PPCs should be the first
# follow-up step.

print("\nNOTE: Posterior predictive checks are NOT possible without the original simulator.")
print("This is a known limitation. If the simulator becomes available, run PPCs immediately.")
print("\nDone.")
