"""
Amortized Bayesian inference for an AR(2) time series model using BayesFlow 2.x.

Model:  y_t = phi1 * y_{t-1} + phi2 * y_{t-2} + N(0, sigma)
Params: phi1, phi2 (AR coefficients), sigma (noise scale)

We train a neural posterior estimator that, once trained, can produce
posterior samples for *any* observed length-100 AR(2) series in a single
forward pass — no MCMC needed at inference time.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = sum(map(ord, "ar2-amortized"))
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Simulator: prior + observation model
# ---------------------------------------------------------------------------
T = 100  # fixed sequence length


def prior():
    """
    Prior over AR(2) parameters.

    phi1, phi2 ~ Uniform(-1, 1)
        Wide prior that covers stationary and non-stationary regimes.
        We deliberately do not enforce stationarity constraints — the
        network should learn to handle both cases.

    sigma ~ Gamma(2, 1)
        Weakly informative prior on the noise scale. Ensures positivity
        and concentrates mass around reasonable noise levels.
    """
    phi1 = rng.uniform(-1.0, 1.0)
    phi2 = rng.uniform(-1.0, 1.0)
    sigma = rng.gamma(2.0, 1.0)
    return dict(phi1=phi1, phi2=phi2, sigma=sigma)


def observation_model(phi1, phi2, sigma):
    """
    Simulate a length-T AR(2) time series given parameters.

    y_0 ~ N(0, sigma)
    y_1 ~ N(phi1 * y_0, sigma)
    y_t = phi1 * y_{t-1} + phi2 * y_{t-2} + N(0, sigma)  for t >= 2
    """
    y = np.zeros(T)
    y[0] = rng.normal(0.0, sigma)
    y[1] = rng.normal(phi1 * y[0], sigma)
    for t in range(2, T):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + rng.normal(0.0, sigma)
    return dict(y=y)


simulator = bf.make_simulator([prior, observation_model])

# ---------------------------------------------------------------------------
# 2. Quick sanity check on simulated data
# ---------------------------------------------------------------------------
print("=== Simulation sanity check ===")
test_sims = simulator.sample(8)
for key in test_sims:
    arr = np.asarray(test_sims[key])
    print(f"  {key}: shape={arr.shape}, min={arr.min():.3f}, max={arr.max():.3f}")
print()

# ---------------------------------------------------------------------------
# 3. Network architecture
# ---------------------------------------------------------------------------
# The key design choice: since observations are temporally ordered (not
# exchangeable), we must use a sequence-aware summary network. A DeepSet
# or SetTransformer would discard temporal ordering, which is essential
# for identifying AR coefficients.
#
# TimeSeriesTransformer uses Time2Vec positional embeddings internally
# to encode temporal position. We set time_axis=1 so it generates a
# uniform time grid along the sequence dimension.
#
# summary_dim = 8: slightly above the 2x-num-params heuristic (6) to
# give the network some breathing room for learning useful summaries.

summary_network = bf.networks.TimeSeriesTransformer(
    summary_dim=8,
    embed_dims=(64, 64),
    num_heads=(4, 4),
    mlp_depths=(2, 2),
    mlp_widths=(128, 128),
    dropout=0.05,
    time_embed_dim=8,
    time_axis=1,
)

# FlowMatching as the inference network — continuous normalizing flow
# trained with the flow matching objective. Good default for low-dimensional
# parameter spaces like ours (3 params).
inference_network = bf.networks.FlowMatching(
    subnet_kwargs=dict(widths=(256, 256), time_embedding_dim=16),
)

# ---------------------------------------------------------------------------
# 4. Adapter
# ---------------------------------------------------------------------------
# The adapter handles data reshaping and parameter transforms:
# - as_time_series: reshapes y from (T,) to (T, 1) for the transformer
# - constrain sigma > 0: applies softplus bijection so network works
#   in unconstrained space but sigma is always positive
# - concatenate parameters into inference_variables
# - concatenate observations into summary_variables
adapter = (
    bf.Adapter()
    .as_time_series(["y"])
    .constrain("sigma", lower=0)
    .convert_dtype("float64", "float32")
    .concatenate(["phi1", "phi2", "sigma"], into="inference_variables")
    .concatenate(["y"], into="summary_variables")
)

# ---------------------------------------------------------------------------
# 5. Workflow
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_network,
    summary_network=summary_network,
    adapter=adapter,
    checkpoint_filepath=os.path.join(OUTPUT_DIR, "checkpoints"),
    checkpoint_name="ar2_model",
)

# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------
# Online training: simulator is cheap (pure NumPy loop), so we generate
# fresh data each epoch rather than pre-simulating a fixed dataset.
#
# 50 epochs x 100 batches x 32 samples = 160,000 simulated series total.
# validation_data=300 for monitoring overfitting.
print("=== Starting training ===")
history = workflow.fit_online(
    epochs=50,
    batch_size=32,
    num_batches_per_epoch=100,
    validation_data=300,
    verbose=2,
)

# Save and inspect training history
train_loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

print(f"\n=== Training summary ===")
print(f"  Final training loss: {train_loss[-1]:.4f}")
if val_loss:
    print(f"  Final validation loss: {val_loss[-1]:.4f}")
    # Check for overfitting: val loss should not be much worse than train loss
    if val_loss[-1] > train_loss[-1] * 1.5:
        print("  WARNING: possible overfitting (val_loss >> train_loss)")
    # Check for convergence: loss should have decreased
    if len(train_loss) > 5 and train_loss[-1] > train_loss[5]:
        print("  WARNING: training loss not decreasing — check learning rate")

# Plot training curves
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_loss, label="Training loss")
if val_loss:
    ax.plot(val_loss, label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training history")
ax.legend()
fig.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Diagnostics on held-out simulations
# ---------------------------------------------------------------------------
print("\n=== Running diagnostics ===")
test_data = workflow.simulate(300)

# Numerical diagnostics
metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
metrics.to_csv(metrics_path)
print(metrics)

# Visual diagnostics (recovery, calibration, etc.)
figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"diagnostics_{name}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

print("Diagnostic plots saved.")

# ---------------------------------------------------------------------------
# 8. Demo: inference on a synthetic observation
# ---------------------------------------------------------------------------
true_params = dict(phi1=0.6, phi2=-0.3, sigma=0.5)
obs = observation_model(**true_params)
real_data = {"y": obs["y"]}

samples = workflow.sample(conditions=real_data, num_samples=1000)

print("\n=== Posterior summary (demo observation) ===")
print(f"  Ground truth: phi1={true_params['phi1']}, phi2={true_params['phi2']}, "
      f"sigma={true_params['sigma']}")
for pname in ["phi1", "phi2", "sigma"]:
    s = np.asarray(samples[pname]).flatten()
    print(f"  {pname}: mean={s.mean():.3f}, std={s.std():.3f}, "
          f"95% CI=[{np.percentile(s, 2.5):.3f}, {np.percentile(s, 97.5):.3f}]")

# ---------------------------------------------------------------------------
# 9. Posterior predictive checks
# ---------------------------------------------------------------------------
n_ppc = 50
ppc_series = []
for i in range(n_ppc):
    theta = {
        "phi1": float(np.asarray(samples["phi1"]).flatten()[i]),
        "phi2": float(np.asarray(samples["phi2"]).flatten()[i]),
        "sigma": float(np.asarray(samples["sigma"]).flatten()[i]),
    }
    y_rep = observation_model(**theta)["y"]
    ppc_series.append(y_rep)

ppc_series = np.array(ppc_series)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: overlay replicated series on observed
ax = axes[0]
for i in range(min(20, n_ppc)):
    ax.plot(ppc_series[i], color="steelblue", alpha=0.15, linewidth=0.7)
ax.plot(obs["y"], color="black", linewidth=1.5, label="Observed")
ax.set_title("Posterior predictive overlay")
ax.set_xlabel("Time step")
ax.set_ylabel("y")
ax.legend()

# Right panel: compare autocorrelation at lags 1 and 2
obs_acf1 = np.corrcoef(obs["y"][1:], obs["y"][:-1])[0, 1]
obs_acf2 = np.corrcoef(obs["y"][2:], obs["y"][:-2])[0, 1]
rep_acf1 = [np.corrcoef(s[1:], s[:-1])[0, 1] for s in ppc_series]
rep_acf2 = [np.corrcoef(s[2:], s[:-2])[0, 1] for s in ppc_series]

ax = axes[1]
ax.hist(rep_acf1, bins=20, alpha=0.5, color="steelblue", label="Replicated lag-1")
ax.axvline(obs_acf1, color="black", linestyle="--", linewidth=2, label="Observed lag-1")
ax.hist(rep_acf2, bins=20, alpha=0.5, color="coral", label="Replicated lag-2")
ax.axvline(obs_acf2, color="red", linestyle="--", linewidth=2, label="Observed lag-2")
ax.set_title("PPC: autocorrelation at lags 1 & 2")
ax.set_xlabel("Autocorrelation")
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "ppc_ar2.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nPPC plot saved. Done.")
