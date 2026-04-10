"""
Amortized Bayesian Inference for Linear Regression with Variable Sample Sizes
==============================================================================

Uses BayesFlow 2.x to train a neural posterior estimator that handles
datasets with N in [10, 200]. Once trained, posterior inference for any
new dataset in that range is a single forward pass (~milliseconds).

Model:
    y_i = alpha + beta * x_i + eps_i,   eps_i ~ Normal(0, sigma)

Priors:
    alpha ~ Normal(0, 5)
    beta  ~ Normal(0, 5)
    sigma ~ HalfNormal(2)
"""

import numpy as np
import matplotlib.pyplot as plt
import bayesflow as bf

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = sum(map(ord, "linear-regression-varying-n"))
RNG = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Simulator components
# ---------------------------------------------------------------------------
PARAM_NAMES = ["alpha", "beta", "sigma"]
N_MIN, N_MAX = 10, 200


@bf.simulators.make_simulator
def prior(batch_size, rng=None):
    """Draw from the joint prior over (alpha, beta, sigma)."""
    rng = rng or np.random.default_rng()
    alpha = rng.normal(loc=0.0, scale=5.0, size=(batch_size, 1))
    beta = rng.normal(loc=0.0, scale=5.0, size=(batch_size, 1))
    sigma = np.abs(rng.normal(loc=0.0, scale=2.0, size=(batch_size, 1)))
    # Clip sigma away from zero for numerical safety
    sigma = np.clip(sigma, 1e-4, None)
    return {"parameters": np.concatenate([alpha, beta, sigma], axis=-1)}


@bf.simulators.make_simulator
def likelihood(parameters, batch_size, rng=None):
    """
    Generate (x, y) pairs for a linear regression.

    Each simulated dataset gets a random N ~ Uniform{N_MIN, ..., N_MAX}.
    To enable batching with variable N, we pad to N_MAX and return a
    boolean mask indicating which observations are real.
    """
    rng = rng or np.random.default_rng()

    all_obs = np.zeros((batch_size, N_MAX, 2), dtype=np.float32)
    all_masks = np.zeros((batch_size, N_MAX, 1), dtype=np.float32)

    for i in range(batch_size):
        n = rng.integers(N_MIN, N_MAX + 1)
        x = rng.normal(loc=0.0, scale=1.0, size=n)
        alpha, beta, sigma = parameters[i]
        y = alpha + beta * x + rng.normal(0.0, sigma, size=n)

        # Stack (x, y) as a 2-column matrix
        all_obs[i, :n, 0] = x
        all_obs[i, :n, 1] = y
        all_masks[i, :n, 0] = 1.0

    return {
        "observables": all_obs,
        "mask": all_masks,
    }


simulator = bf.simulators.TwoLevelSimulator(prior, likelihood)

# ---------------------------------------------------------------------------
# 2. Adapter — bridge simulator outputs to network inputs
# ---------------------------------------------------------------------------
# The adapter tells BayesFlow how to rename / reshape simulator outputs
# into the dict keys the network expects.
adapter = (
    bf.Adapter()
    .rename("parameters", "inference_variables")
    .rename("observables", "summary_variables")
    .rename("mask", "summary_mask")
)

# ---------------------------------------------------------------------------
# 3. Summary network — handles variable-length sets
# ---------------------------------------------------------------------------
# A SetTransformer (DeepSets variant) naturally handles variable-size inputs
# via masking.  It produces a fixed-dimensional summary vector regardless of N.
summary_net = bf.networks.SetTransformer(
    input_dim=2,          # each observation is (x, y)
    summary_dim=32,       # output summary vector length
    num_attention_heads=4,
    num_dense_layers=2,
    dense_units=128,
)

# ---------------------------------------------------------------------------
# 4. Inference network — conditional normalizing flow
# ---------------------------------------------------------------------------
inference_net = bf.networks.CouplingFlow(
    num_params=len(PARAM_NAMES),
    num_coupling_layers=6,
    coupling_settings={"dense_units": 128, "num_dense_layers": 2},
)

# ---------------------------------------------------------------------------
# 5. Approximator — ties everything together
# ---------------------------------------------------------------------------
workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    summary_network=summary_net,
    inference_network=inference_net,
)

# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------
EPOCHS = 50
BATCH_SIZE = 64
SIMULATIONS_PER_EPOCH = 1000

print("Starting training...")
history = workflow.fit(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    simulations_per_epoch=SIMULATIONS_PER_EPOCH,
)

# Plot training loss
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history["loss"])
ax.set(xlabel="Epoch", ylabel="Loss", title="Training loss")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("training_loss.png", dpi=150)
plt.close(fig)
print("Saved training_loss.png")

# ---------------------------------------------------------------------------
# 7. Simulation-Based Calibration (SBC)
# ---------------------------------------------------------------------------
print("Running SBC diagnostics (500 datasets)...")
num_sbc = 500
sbc_results = workflow.simulate_and_infer(num_datasets=num_sbc, num_samples=1000)

fig = bf.diagnostics.plot_sbc_histograms(
    sbc_results,
    param_names=PARAM_NAMES,
)
fig.savefig("sbc_histograms.png", dpi=150)
plt.close(fig)
print("Saved sbc_histograms.png")

# ECDF-based calibration check
fig = bf.diagnostics.plot_sbc_ecdf(
    sbc_results,
    param_names=PARAM_NAMES,
)
fig.savefig("sbc_ecdf.png", dpi=150)
plt.close(fig)
print("Saved sbc_ecdf.png")

# ---------------------------------------------------------------------------
# 8. Posterior z-score / contraction diagnostic
# ---------------------------------------------------------------------------
fig = bf.diagnostics.plot_z_score_contraction(
    sbc_results,
    param_names=PARAM_NAMES,
)
fig.savefig("z_score_contraction.png", dpi=150)
plt.close(fig)
print("Saved z_score_contraction.png")

# ---------------------------------------------------------------------------
# 9. Inference on a specific observed dataset
# ---------------------------------------------------------------------------
print("\n--- Inference on a test dataset ---")
TRUE_ALPHA, TRUE_BETA, TRUE_SIGMA = 2.0, -1.5, 0.8
N_OBS = 50

rng_test = np.random.default_rng(42)
x_obs = rng_test.normal(0, 1, size=N_OBS)
y_obs = TRUE_ALPHA + TRUE_BETA * x_obs + rng_test.normal(0, TRUE_SIGMA, size=N_OBS)

# Pad to N_MAX for the network
obs_padded = np.zeros((1, N_MAX, 2), dtype=np.float32)
mask_padded = np.zeros((1, N_MAX, 1), dtype=np.float32)
obs_padded[0, :N_OBS, 0] = x_obs
obs_padded[0, :N_OBS, 1] = y_obs
mask_padded[0, :N_OBS, 0] = 1.0

# Draw posterior samples
posterior_samples = workflow.infer(
    {"summary_variables": obs_padded, "summary_mask": mask_padded},
    num_samples=4000,
)

# Print summary
samples = posterior_samples[0]  # shape (num_samples, num_params)
print(f"\nTrue values:  alpha={TRUE_ALPHA}, beta={TRUE_BETA}, sigma={TRUE_SIGMA}")
print(f"Posterior means: alpha={samples[:, 0].mean():.3f}, "
      f"beta={samples[:, 1].mean():.3f}, sigma={samples[:, 2].mean():.3f}")
print(f"Posterior stds:  alpha={samples[:, 0].std():.3f}, "
      f"beta={samples[:, 1].std():.3f}, sigma={samples[:, 2].std():.3f}")

# Posterior pair plot
fig = bf.diagnostics.plot_posterior(
    samples,
    param_names=PARAM_NAMES,
    true_values=[TRUE_ALPHA, TRUE_BETA, TRUE_SIGMA],
)
fig.savefig("posterior_pairplot.png", dpi=150)
plt.close(fig)
print("Saved posterior_pairplot.png")

# ---------------------------------------------------------------------------
# 10. Test across different sample sizes to verify generalization
# ---------------------------------------------------------------------------
print("\n--- Checking posterior quality across sample sizes ---")
test_sizes = [10, 25, 50, 100, 200]
for n in test_sizes:
    x_test = rng_test.normal(0, 1, size=n)
    y_test = TRUE_ALPHA + TRUE_BETA * x_test + rng_test.normal(0, TRUE_SIGMA, size=n)

    obs_pad = np.zeros((1, N_MAX, 2), dtype=np.float32)
    mask_pad = np.zeros((1, N_MAX, 1), dtype=np.float32)
    obs_pad[0, :n, 0] = x_test
    obs_pad[0, :n, 1] = y_test
    mask_pad[0, :n, 0] = 1.0

    post = workflow.infer(
        {"summary_variables": obs_pad, "summary_mask": mask_pad},
        num_samples=2000,
    )
    s = post[0]
    print(f"  N={n:3d} | alpha={s[:, 0].mean():+.2f} (sd {s[:, 0].std():.2f}) | "
          f"beta={s[:, 1].mean():+.2f} (sd {s[:, 1].std():.2f}) | "
          f"sigma={s[:, 2].mean():.2f} (sd {s[:, 2].std():.2f})")

print("\nDone.")
