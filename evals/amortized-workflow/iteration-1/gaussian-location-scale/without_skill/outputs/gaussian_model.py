"""
Amortized Bayesian inference for a Gaussian location-scale model using BayesFlow.

Generative model:
    mu    ~ Normal(0, 5)
    sigma ~ HalfNormal(2)
    x_i   ~ Normal(mu, sigma)   for i = 1, ..., 50  (i.i.d.)

We train an amortized posterior estimator so that, given any new dataset
of 50 observations, posterior samples for (mu, sigma) are available via
a single forward pass — no MCMC required at inference time.
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

RANDOM_SEED = sum(map(ord, "gaussian-location-scale"))
rng = np.random.default_rng(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. Generative model — prior + likelihood
# ---------------------------------------------------------------------------

def prior(rng=None):
    """Sample from the prior: mu ~ N(0, 5), sigma ~ HalfNormal(2)."""
    rng = rng or np.random.default_rng()
    mu = rng.normal(loc=0.0, scale=5.0)
    sigma = np.abs(rng.normal(loc=0.0, scale=2.0))
    return {"mu": np.array(mu), "sigma": np.array(sigma)}


def observation_model(mu, sigma, rng=None):
    """Simulate 50 i.i.d. observations from Normal(mu, sigma)."""
    rng = rng or np.random.default_rng()
    x = rng.normal(loc=mu, scale=sigma, size=50)
    return {"x": x}


simulator = bf.make_simulator([prior, observation_model])


# ---------------------------------------------------------------------------
# 2. Simulation sanity checks
# ---------------------------------------------------------------------------

print("=" * 60)
print("SIMULATION SANITY CHECK")
print("=" * 60)

test_sims = simulator.sample(5, rng=rng)
for key, val in test_sims.items():
    print(f"  {key}: shape={np.shape(val)}, dtype={np.asarray(val).dtype}")

for i in range(3):
    mu_i = float(test_sims["mu"][i])
    sigma_i = float(test_sims["sigma"][i])
    x_i = test_sims["x"][i]
    print(
        f"  sim {i}: mu={mu_i:.2f}, sigma={sigma_i:.2f}, "
        f"x_mean={np.mean(x_i):.2f}, x_std={np.std(x_i):.2f}"
    )
print()


# ---------------------------------------------------------------------------
# 3. Neural network architecture
# ---------------------------------------------------------------------------
# With only 2 parameters and a simple exchangeable likelihood, we keep
# the networks small to avoid overfitting.

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
# 4. Adapter — tells BayesFlow how to reshape and transform variables
# ---------------------------------------------------------------------------

adapter = (
    bf.Adapter()
    .as_set(["x"])                          # (50,) -> (50, 1) for SetTransformer
    .constrain("sigma", lower=0)            # log-transform for sigma > 0
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
# 6. Training
# ---------------------------------------------------------------------------

print("=" * 60)
print("TRAINING")
print("=" * 60)

history = workflow.fit_online(
    epochs=100,
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

# Quick inspection of training curve
train_loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

if train_loss:
    print(f"  Final train loss: {train_loss[-1]:.4f}")
    print(f"  Min train loss:   {min(train_loss):.4f} (epoch {np.argmin(train_loss) + 1})")
if val_loss:
    print(f"  Final val loss:   {val_loss[-1]:.4f}")
    print(f"  Min val loss:     {min(val_loss):.4f} (epoch {np.argmin(val_loss) + 1})")

    # Check for overfitting: val loss increasing while train loss still decreasing
    if len(val_loss) > 20:
        recent_val = val_loss[-10:]
        if recent_val[-1] > recent_val[0]:
            print("  WARNING: Validation loss increasing in last 10 epochs — possible overfitting")

# Plot training curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_loss, label="Train loss")
if val_loss:
    ax.plot(val_loss, label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Curve")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "training_curve.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: training_curve.png")


# ---------------------------------------------------------------------------
# 7. Diagnostics — simulation-based calibration
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
# 8. Inference on a test dataset with known ground truth
# ---------------------------------------------------------------------------

print("=" * 60)
print("INFERENCE ON TEST DATASET")
print("=" * 60)

true_mu = 2.5
true_sigma = 1.3
x_obs = rng.normal(true_mu, true_sigma, size=50)

print(f"Ground truth:  mu={true_mu}, sigma={true_sigma}")
print(f"Observed data: mean={np.mean(x_obs):.3f}, std={np.std(x_obs):.3f}")

# Amortized inference — single forward pass
samples = workflow.sample(conditions={"x": x_obs}, num_samples=2000)

mu_post = samples["mu"][0, :, 0]       # shape: (1, num_samples, 1) -> (num_samples,)
sigma_post = samples["sigma"][0, :, 0]

for name, true_val, post in [("mu", true_mu, mu_post), ("sigma", true_sigma, sigma_post)]:
    lo, hi = np.percentile(post, [2.5, 97.5])
    covers = lo <= true_val <= hi
    print(
        f"  {name:>5s}: mean={np.mean(post):.3f}, std={np.std(post):.3f}, "
        f"95% CI=[{lo:.3f}, {hi:.3f}] {'(covers truth)' if covers else '(MISSES truth)'}"
    )

# Posterior corner plot
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].hist(mu_post, bins=40, alpha=0.7, color="steelblue", edgecolor="white")
axes[0].axvline(true_mu, color="red", linewidth=2, linestyle="--", label=f"True ({true_mu})")
axes[0].set_xlabel(r"$\mu$")
axes[0].set_title(r"Posterior $\mu$")
axes[0].legend()

axes[1].hist(sigma_post, bins=40, alpha=0.7, color="darkorange", edgecolor="white")
axes[1].axvline(true_sigma, color="red", linewidth=2, linestyle="--", label=f"True ({true_sigma})")
axes[1].set_xlabel(r"$\sigma$")
axes[1].set_title(r"Posterior $\sigma$")
axes[1].legend()

axes[2].scatter(mu_post, sigma_post, alpha=0.1, s=5, color="gray")
axes[2].axvline(true_mu, color="red", linewidth=1, linestyle="--", alpha=0.7)
axes[2].axhline(true_sigma, color="red", linewidth=1, linestyle="--", alpha=0.7)
axes[2].set_xlabel(r"$\mu$")
axes[2].set_ylabel(r"$\sigma$")
axes[2].set_title("Joint Posterior")

plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "posterior_test.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: posterior_test.png")


# ---------------------------------------------------------------------------
# 9. Posterior predictive checks
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("POSTERIOR PREDICTIVE CHECKS")
print("=" * 60)

n_ppc = 200
ppc_means = []
ppc_stds = []

for s in range(n_ppc):
    x_rep = rng.normal(
        loc=float(mu_post[s]),
        scale=float(sigma_post[s]),
        size=50,
    )
    ppc_means.append(np.mean(x_rep))
    ppc_stds.append(np.std(x_rep))

obs_mean = np.mean(x_obs)
obs_std = np.std(x_obs)

print(f"  Observed mean: {obs_mean:.3f}")
print(
    f"  PPC means:     median={np.median(ppc_means):.3f}, "
    f"95% interval=[{np.percentile(ppc_means, 2.5):.3f}, {np.percentile(ppc_means, 97.5):.3f}]"
)
print(f"  Observed std:  {obs_std:.3f}")
print(
    f"  PPC stds:      median={np.median(ppc_stds):.3f}, "
    f"95% interval=[{np.percentile(ppc_stds, 2.5):.3f}, {np.percentile(ppc_stds, 97.5):.3f}]"
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(ppc_means, bins=25, alpha=0.6, color="steelblue", edgecolor="white")
axes[0].axvline(obs_mean, color="red", linewidth=2, label=f"Observed ({obs_mean:.2f})")
axes[0].set_xlabel("Replicated sample mean")
axes[0].set_title("PPC: Sample Mean")
axes[0].legend()

axes[1].hist(ppc_stds, bins=25, alpha=0.6, color="darkorange", edgecolor="white")
axes[1].axvline(obs_std, color="red", linewidth=2, label=f"Observed ({obs_std:.2f})")
axes[1].set_xlabel("Replicated sample std")
axes[1].set_title("PPC: Sample Std Dev")
axes[1].legend()

plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "ppc_checks.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: ppc_checks.png")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
