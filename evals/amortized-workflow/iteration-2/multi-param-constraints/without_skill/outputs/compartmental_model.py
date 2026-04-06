"""
Amortized Bayesian inference for an SIR compartmental model using BayesFlow.

Generative model:
    beta  ~ Gamma(2, 0.3)      -- infection rate (positive)
    gamma ~ Gamma(2, 0.15)     -- recovery rate (positive)
    i0    ~ Beta(2, 50)        -- initial infected fraction (0, 1)
    p     ~ Beta(5, 2)         -- reporting probability (0, 1)

    Discrete-time SIR dynamics over 30 days, with daily reported cases
    drawn from Poisson(new_infections * p * N).

We train an amortized posterior estimator so that, given any 30-day
case-count time series, posterior samples for (beta, gamma, i0, p) are
available via a single forward pass.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 0. Reproducibility and output paths
# ---------------------------------------------------------------------------

RANDOM_SEED = sum(map(ord, "compartmental-sir-model"))
rng = np.random.default_rng(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

POPULATION = 10_000
N_DAYS = 30


# ---------------------------------------------------------------------------
# 1. Generative model -- prior + observation model
# ---------------------------------------------------------------------------

def prior(rng=None):
    """Sample epidemiological parameters from the prior."""
    rng = rng or np.random.default_rng()
    beta = rng.gamma(shape=2.0, scale=0.3)        # infection rate > 0
    gamma = rng.gamma(shape=2.0, scale=0.15)       # recovery rate > 0
    i0 = rng.beta(a=2.0, b=50.0)                   # initial infected fraction in (0, 1)
    p = rng.beta(a=5.0, b=2.0)                     # reporting probability in (0, 1)
    return {"beta": np.array(beta), "gamma": np.array(gamma),
            "i0": np.array(i0), "p": np.array(p)}


def observation_model(beta, gamma, i0, p, rng=None):
    """Simulate daily reported case counts from a discrete-time SIR model."""
    rng = rng or np.random.default_rng()

    S = 1.0 - i0
    I = i0
    R = 0.0

    daily_cases = np.zeros(N_DAYS)
    for t in range(N_DAYS):
        new_infections = beta * S * I
        new_recoveries = gamma * I

        # Clamp to valid compartment sizes
        new_infections = min(new_infections, S)
        new_recoveries = min(new_recoveries, I)

        S -= new_infections
        I += new_infections - new_recoveries
        R += new_recoveries

        # Reported cases: true new infections * reporting probability * population
        expected = max(new_infections * p * POPULATION, 1e-8)
        daily_cases[t] = rng.poisson(lam=expected)

    return {"daily_cases": daily_cases}


simulator = bf.make_simulator([prior, observation_model])


# ---------------------------------------------------------------------------
# 2. Simulation sanity checks
# ---------------------------------------------------------------------------

print("=" * 60)
print("SIMULATION SANITY CHECK")
print("=" * 60)

test_sims = simulator.sample(5, rng=rng)
for key, val in test_sims.items():
    arr = np.asarray(val)
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

for i in range(3):
    b_i = float(test_sims["beta"][i])
    g_i = float(test_sims["gamma"][i])
    i0_i = float(test_sims["i0"][i])
    p_i = float(test_sims["p"][i])
    cases = test_sims["daily_cases"][i]
    print(
        f"  sim {i}: beta={b_i:.3f}, gamma={g_i:.3f}, i0={i0_i:.4f}, p={p_i:.3f}, "
        f"total_cases={np.sum(cases):.0f}, peak_cases={np.max(cases):.0f}"
    )
print()


# ---------------------------------------------------------------------------
# 3. Neural network architecture
# ---------------------------------------------------------------------------
# 4 parameters -> summary_dim = 2 * num_params = 8
# Daily case counts are an ordered time series (30 days), not an
# exchangeable set, so we use TimeSeriesTransformer.

summary_net = bf.networks.TimeSeriesTransformer(
    summary_dim=8,
    embed_dims=(32, 32),
    num_heads=(2, 2),
    mlp_depths=(1, 1),
    mlp_widths=(64, 64),
    time_embed_dim=4,
    time_axis=1,
)

inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"widths": (128, 128), "time_embedding_dim": 16},
)


# ---------------------------------------------------------------------------
# 4. Adapter -- parameter constraints and data reshaping
# ---------------------------------------------------------------------------
# Constraints map bounded parameters to unconstrained space for the network:
#   - beta, gamma > 0: log-transform (lower=0)
#   - i0, p in (0, 1): logit-transform (lower=0, upper=1)

adapter = (
    bf.Adapter()
    # Structural: daily_cases is a time series (30,) -> (30, 1)
    .as_time_series(["daily_cases"])
    # Enforce parameter constraints via bijective transforms
    .constrain("beta", lower=0)
    .constrain("gamma", lower=0)
    .constrain("i0", lower=0, upper=1)
    .constrain("p", lower=0, upper=1)
    # NumPy float64 -> float32 for JAX
    .convert_dtype("float64", "float32")
    # Pack into BayesFlow's reserved keys
    .concatenate(["beta", "gamma", "i0", "p"], into="inference_variables")
    .concatenate(["daily_cases"], into="summary_variables")
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
    checkpoint_name="sir_compartmental",
)


# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------

print("=" * 60)
print("TRAINING")
print("=" * 60)

history = workflow.fit_online(
    epochs=50,
    batch_size=32,
    num_batches_per_epoch=100,
    validation_data=300,
    verbose=2,
)

# Save training history
history_path = os.path.join(SCRIPT_DIR, "training_history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)
print(f"\nTraining history saved to {history_path}")

# Inspect training curve
train_loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

if train_loss:
    print(f"  Final train loss: {train_loss[-1]:.4f}")
    print(f"  Min train loss:   {min(train_loss):.4f} (epoch {np.argmin(train_loss) + 1})")
if val_loss:
    print(f"  Final val loss:   {val_loss[-1]:.4f}")
    print(f"  Min val loss:     {min(val_loss):.4f} (epoch {np.argmin(val_loss) + 1})")

    # Check for overfitting
    if len(val_loss) > 20:
        recent_val = val_loss[-10:]
        if recent_val[-1] > recent_val[0]:
            print("  WARNING: Validation loss increasing in last 10 epochs -- possible overfitting")

# Plot training curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_loss, label="Train loss")
if val_loss:
    ax.plot(val_loss, label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Curve -- SIR Compartmental Model")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "training_curve.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: training_curve.png")


# ---------------------------------------------------------------------------
# 7. In-silico diagnostics
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("IN-SILICO DIAGNOSTICS")
print("=" * 60)

test_data = workflow.simulate(300)

# Built-in diagnostic plots (recovery, calibration, etc.)
figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig_path = os.path.join(SCRIPT_DIR, f"diag_{name}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: diag_{name}.png")

# Numerical diagnostic metrics
metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
metrics_path = os.path.join(SCRIPT_DIR, "diagnostics_metrics.csv")
metrics.to_csv(metrics_path)
print(f"\nDiagnostic metrics:\n{metrics}\n")


# ---------------------------------------------------------------------------
# 8. Inference on synthetic test data with known ground truth
# ---------------------------------------------------------------------------

print("=" * 60)
print("INFERENCE ON SYNTHETIC TEST DATA")
print("=" * 60)

# Generate one synthetic dataset with known parameters
true_beta = 0.5
true_gamma = 0.2
true_i0 = 0.01
true_p = 0.7

obs = observation_model(true_beta, true_gamma, true_i0, true_p, rng=rng)
x_obs = obs["daily_cases"]

print(f"Ground truth: beta={true_beta}, gamma={true_gamma}, i0={true_i0}, p={true_p}")
print(f"Observed: total_cases={np.sum(x_obs):.0f}, peak_day={np.argmax(x_obs)}, "
      f"peak_cases={np.max(x_obs):.0f}")

# Amortized inference -- single forward pass
samples = workflow.sample(conditions={"daily_cases": x_obs}, num_samples=2000)

param_names = ["beta", "gamma", "i0", "p"]
true_vals = [true_beta, true_gamma, true_i0, true_p]

for name, true_val in zip(param_names, true_vals):
    post = samples[name][0, :, 0]  # (1, num_samples, 1) -> (num_samples,)
    lo, hi = np.percentile(post, [2.5, 97.5])
    covers = lo <= true_val <= hi
    print(
        f"  {name:>5s}: mean={np.mean(post):.4f}, std={np.std(post):.4f}, "
        f"95% CI=[{lo:.4f}, {hi:.4f}] {'(covers truth)' if covers else '(MISSES truth)'}"
    )

# Posterior marginals plot
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
colors = ["steelblue", "darkorange", "forestgreen", "mediumpurple"]

for ax, name, true_val, color in zip(axes, param_names, true_vals, colors):
    post = samples[name][0, :, 0]
    ax.hist(post, bins=40, alpha=0.7, color=color, edgecolor="white")
    ax.axvline(true_val, color="red", linewidth=2, linestyle="--", label=f"True ({true_val})")
    ax.set_xlabel(name)
    ax.set_title(f"Posterior {name}")
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "posterior_test.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: posterior_test.png")

# Joint posterior scatter: beta vs gamma (key epidemiological trade-off)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(
    samples["beta"][0, :, 0], samples["gamma"][0, :, 0],
    alpha=0.1, s=5, color="gray",
)
ax.axvline(true_beta, color="red", linewidth=1, linestyle="--", alpha=0.7)
ax.axhline(true_gamma, color="red", linewidth=1, linestyle="--", alpha=0.7)
ax.set_xlabel(r"$\beta$ (infection rate)")
ax.set_ylabel(r"$\gamma$ (recovery rate)")
ax.set_title(r"Joint Posterior: $\beta$ vs $\gamma$")
plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "joint_posterior_beta_gamma.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: joint_posterior_beta_gamma.png")


# ---------------------------------------------------------------------------
# 9. Posterior predictive checks
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("POSTERIOR PREDICTIVE CHECKS")
print("=" * 60)

n_ppc = 200
ppc_curves = []

for s in range(n_ppc):
    theta_s = {
        "beta": float(samples["beta"][0, s, 0]),
        "gamma": float(samples["gamma"][0, s, 0]),
        "i0": float(samples["i0"][0, s, 0]),
        "p": float(samples["p"][0, s, 0]),
    }
    x_rep = observation_model(**theta_s, rng=rng)
    ppc_curves.append(x_rep["daily_cases"])

ppc_curves = np.array(ppc_curves)

fig, ax = plt.subplots(figsize=(10, 5))
for curve in ppc_curves:
    ax.plot(range(N_DAYS), curve, alpha=0.08, color="steelblue")
ax.plot(range(N_DAYS), x_obs, color="black", linewidth=2, label="Observed")
ppc_median = np.median(ppc_curves, axis=0)
ax.plot(range(N_DAYS), ppc_median, color="red", linewidth=1.5, linestyle="--", label="PPC median")
ax.set_xlabel("Day")
ax.set_ylabel("Reported cases")
ax.set_title("Posterior Predictive Check")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "ppc_check.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: ppc_check.png")

# Summary statistics for PPC
ppc_totals = ppc_curves.sum(axis=1)
obs_total = x_obs.sum()
ppc_peaks = ppc_curves.max(axis=1)
obs_peak = x_obs.max()

print(f"  Observed total cases: {obs_total:.0f}")
print(
    f"  PPC total cases: median={np.median(ppc_totals):.0f}, "
    f"95% interval=[{np.percentile(ppc_totals, 2.5):.0f}, {np.percentile(ppc_totals, 97.5):.0f}]"
)
print(f"  Observed peak cases: {obs_peak:.0f}")
print(
    f"  PPC peak cases: median={np.median(ppc_peaks):.0f}, "
    f"95% interval=[{np.percentile(ppc_peaks, 2.5):.0f}, {np.percentile(ppc_peaks, 97.5):.0f}]"
)

# Check constraint satisfaction in posterior samples
print("\n" + "=" * 60)
print("CONSTRAINT VERIFICATION")
print("=" * 60)

for name, true_val in zip(param_names, true_vals):
    post = samples[name][0, :, 0]
    if name in ("beta", "gamma"):
        violations = np.sum(post <= 0)
        print(f"  {name}: min={np.min(post):.6f}, violations (<=0): {violations}/{len(post)}")
    elif name in ("i0", "p"):
        violations_lo = np.sum(post <= 0)
        violations_hi = np.sum(post >= 1)
        print(
            f"  {name}: min={np.min(post):.6f}, max={np.max(post):.6f}, "
            f"violations (<0 or >1): {violations_lo + violations_hi}/{len(post)}"
        )

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
