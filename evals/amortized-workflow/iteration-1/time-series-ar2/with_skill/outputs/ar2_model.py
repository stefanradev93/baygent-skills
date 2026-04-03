import os

os.environ["KERAS_BACKEND"] = "jax"

import json
import sys

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = sum(map(ord, "ar2-amortized-inference-v1"))
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Make the scripts importable
# ---------------------------------------------------------------------------
SKILL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "..",
    "baygent-skills", "amortized-workflow",
)
sys.path.insert(0, os.path.abspath(SKILL_DIR))

from scripts.check_diagnostics import check_diagnostics
from scripts.inspect_training import inspect_history

# ---------------------------------------------------------------------------
# 1. Prior + observation model (online training)
# ---------------------------------------------------------------------------
T = 100  # time series length


def prior():
    """
    Prior over AR(2) parameters.
    phi1, phi2 ~ Uniform(-1, 1) — deliberately wide; stationarity is NOT
    enforced so the network learns to handle both stationary and
    non-stationary regimes.
    sigma ~ Gamma(2, 1) — weakly informative, keeps noise scale positive.
    """
    phi1 = rng.uniform(-1.0, 1.0)
    phi2 = rng.uniform(-1.0, 1.0)
    sigma = rng.gamma(2.0, 1.0)
    return dict(phi1=phi1, phi2=phi2, sigma=sigma)


def observation_model(phi1, phi2, sigma):
    """
    Generate a length-T AR(2) time series:
        y_t = phi1 * y_{t-1} + phi2 * y_{t-2} + N(0, sigma)
    First two values drawn from N(0, sigma).
    """
    y = np.zeros(T)
    y[0] = rng.normal(0.0, sigma)
    y[1] = rng.normal(phi1 * y[0], sigma)
    for t in range(2, T):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + rng.normal(0.0, sigma)
    return dict(y=y)


simulator = bf.make_simulator([prior, observation_model])

# ---------------------------------------------------------------------------
# 2. Simulation sanity check — inspect a handful of draws
# ---------------------------------------------------------------------------
print("=== Simulation sanity check ===")
sanity = simulator.sample(8)
for key in sanity:
    arr = np.asarray(sanity[key])
    print(f"  {key}: shape={arr.shape}, min={arr.min():.3f}, max={arr.max():.3f}")
print()

# ---------------------------------------------------------------------------
# 3. Architecture — Small config (skill rule: always start Small)
# ---------------------------------------------------------------------------
# 3 parameters -> summary_dim = 2 * 3 = 6
# TimeSeriesTransformer Small config from model-sizes.md:
#   embed_dims=(32,32), num_heads=(2,2), mlp_depths=(1,1),
#   mlp_widths=(64,64), time_embed_dim=4, time_axis=1
summary_net = bf.networks.TimeSeriesTransformer(
    summary_dim=6,
    embed_dims=(32, 32),
    num_heads=(2, 2),
    mlp_depths=(1, 1),
    mlp_widths=(64, 64),
    time_embed_dim=4,
    time_axis=1,
)

# Inference network — FlowMatching, Small subnet
inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"widths": (128, 128), "time_embedding_dim": 16},
)

# ---------------------------------------------------------------------------
# 4. Adapter (explicit — MUST use bf.Adapter)
# ---------------------------------------------------------------------------
# y is shape (T,) from the simulator; .as_time_series makes it (T, 1).
# sigma is strictly positive -> .constrain(lower=0).
# phi1, phi2 are unconstrained (support is all of R in the network's space,
# even though the prior is bounded — no hard boundary needed for the network).
adapter = (
    bf.Adapter()
    .as_time_series(["y"])
    .constrain("sigma", lower=0)
    .convert_dtype("float64", "float32")
    .concatenate(["phi1", "phi2", "sigma"], into="inference_variables")
    .concatenate(["y"], into="summary_variables")
)

# ---------------------------------------------------------------------------
# 5. Build workflow
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath=os.path.join(OUTPUT_DIR, "checkpoints"),
    checkpoint_name="ar2_model",
)

# ---------------------------------------------------------------------------
# 6. Train (online, with validation_data — mandatory)
# ---------------------------------------------------------------------------
history = workflow.fit_online(
    epochs=50,
    batch_size=32,
    num_batches_per_epoch=100,
    validation_data=300,
    verbose=2,
)

# --- Mandatory: save and inspect training history ---
history_path = os.path.join(OUTPUT_DIR, "history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)

training_report = inspect_history(history.history)
print("\n=== Training inspection ===")
print(json.dumps(training_report, indent=2))

if not training_report["overall"]["ok"]:
    print("TRAINING ISSUES — address before continuing:")
    for issue in training_report["overall"]["issues"]:
        print(f"  - {issue}")

# ---------------------------------------------------------------------------
# 7. In-silico diagnostics (MUST use workflow.simulate, NEVER a for-loop)
# ---------------------------------------------------------------------------
test_data = workflow.simulate(300)

# --- Numerical diagnostics ---
metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
metrics.to_csv(metrics_path)
print("\n=== Diagnostic metrics ===")
print(metrics)

diag_report = check_diagnostics(metrics)
print("\n=== Diagnostics gate ===")
print(json.dumps(diag_report, indent=2))

if diag_report["overall"]["decision"] == "STOP":
    raise RuntimeError(diag_report["overall"]["recommendation"])

# --- Visual diagnostics ---
figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"diagnostics_{name}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

print("\nDiagnostic plots saved.")

# ---------------------------------------------------------------------------
# 8. Amortized inference on a synthetic "real" observation (demo)
# ---------------------------------------------------------------------------
# Generate one ground-truth sequence for demonstration
true_params = dict(phi1=0.6, phi2=-0.3, sigma=0.5)
obs = observation_model(**true_params)
real_data = {"y": obs["y"]}

samples = workflow.sample(conditions=real_data, num_samples=1000)

print("\n=== Posterior summary (demo observation) ===")
print(f"  Ground truth: phi1={true_params['phi1']}, phi2={true_params['phi2']}, sigma={true_params['sigma']}")
for pname in ["phi1", "phi2", "sigma"]:
    s = np.asarray(samples[pname]).flatten()
    print(f"  {pname}: mean={s.mean():.3f}, std={s.std():.3f}, "
          f"95% CI=[{np.percentile(s, 2.5):.3f}, {np.percentile(s, 97.5):.3f}]")

# ---------------------------------------------------------------------------
# 9. Posterior predictive checks — MUST reuse observation_model
# ---------------------------------------------------------------------------
n_ppc = 50
ppc_series = []
for s in range(n_ppc):
    theta_s = {
        "phi1": float(np.asarray(samples["phi1"]).flatten()[s]),
        "phi2": float(np.asarray(samples["phi2"]).flatten()[s]),
        "sigma": float(np.asarray(samples["sigma"]).flatten()[s]),
    }
    y_rep = observation_model(**theta_s)["y"]
    ppc_series.append(y_rep)

ppc_series = np.array(ppc_series)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: overlay replicated series
ax = axes[0]
for i in range(min(20, n_ppc)):
    ax.plot(ppc_series[i], color="steelblue", alpha=0.15, linewidth=0.7)
ax.plot(obs["y"], color="black", linewidth=1.5, label="observed")
ax.set_title("Posterior predictive overlay")
ax.set_xlabel("Time step")
ax.set_ylabel("y")
ax.legend()

# Right: compare ACF at lags 1 and 2
obs_acf1 = np.corrcoef(obs["y"][1:], obs["y"][:-1])[0, 1]
obs_acf2 = np.corrcoef(obs["y"][2:], obs["y"][:-2])[0, 1]
rep_acf1 = [np.corrcoef(s[1:], s[:-1])[0, 1] for s in ppc_series]
rep_acf2 = [np.corrcoef(s[2:], s[:-2])[0, 1] for s in ppc_series]

ax = axes[1]
ax.hist(rep_acf1, bins=20, alpha=0.5, color="steelblue", label="rep lag-1")
ax.axvline(obs_acf1, color="black", linestyle="--", linewidth=2, label="obs lag-1")
ax.hist(rep_acf2, bins=20, alpha=0.5, color="coral", label="rep lag-2")
ax.axvline(obs_acf2, color="red", linestyle="--", linewidth=2, label="obs lag-2")
ax.set_title("PPC: autocorrelation at lags 1 & 2")
ax.set_xlabel("Autocorrelation")
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "ppc_ar2.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nPPC plot saved. Done.")
