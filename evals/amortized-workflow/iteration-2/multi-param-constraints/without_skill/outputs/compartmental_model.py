"""
Amortized Bayesian inference for a compartmental SIR-like model using BayesFlow 2.x.
"""

import numpy as np
import bayesflow as bf
import tensorflow as tf

RANDOM_SEED = sum(map(ord, "sir-amortized-inference"))
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

N_DAYS = 30
POPULATION = 1_000_000

def prior(batch_size):
    beta = np.random.lognormal(mean=-1.0, sigma=0.5, size=(batch_size, 1))
    gamma = np.random.lognormal(mean=-1.5, sigma=0.5, size=(batch_size, 1))
    i0 = np.random.beta(a=2.0, b=20.0, size=(batch_size, 1))
    p = np.random.beta(a=5.0, b=5.0, size=(batch_size, 1))
    return np.concatenate([beta, gamma, i0, p], axis=-1).astype(np.float32)

def sir_simulator(params):
    batch_size = params.shape[0]
    all_counts = np.zeros((batch_size, N_DAYS, 1), dtype=np.float32)
    for b in range(batch_size):
        beta_b, gamma_b, i0_b, p_b = params[b]
        I = float(i0_b) * POPULATION
        S = POPULATION - I
        R = 0.0
        for t in range(N_DAYS):
            new_infections = beta_b * S * I / POPULATION
            new_recoveries = gamma_b * I
            S = max(S - new_infections, 0.0)
            I = max(I + new_infections - new_recoveries, 0.0)
            R = R + new_recoveries
            observed = np.random.binomial(n=max(int(round(new_infections)), 0), p=float(p_b))
            all_counts[b, t, 0] = observed
    return {"summary_variables": all_counts}

adapter = (
    bf.Adapter()
    .constrain("beta", lower=0)
    .constrain("gamma", lower=0)
    .constrain("i0", lower=0, upper=1)
    .constrain("p", lower=0, upper=1)
    .concatenate(["beta", "gamma", "i0", "p"], into="inference_variables")
)

SUMMARY_DIM = 8
NUM_PARAMS = 4

summary_net = bf.networks.TimeSeriesTransformer(
    summary_dim=SUMMARY_DIM,
    num_attention_heads=2,
    num_transformer_blocks=2,
    dropout=0.05,
)

inference_net = bf.networks.CouplingFlow(
    num_coupling_layers=4,
    subnet_kwargs={"units": [64, 64]},
)

approximator = bf.approximators.ContinuousApproximator(
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
)

workflow = bf.BasicWorkflow(
    prior=prior,
    simulator=sir_simulator,
    approximator=approximator,
)

train_data = workflow.simulate(num_samples=10_000)
val_data = workflow.simulate(num_samples=1_000)

history = workflow.fit_dataset(
    data=train_data, validation_data=val_data, epochs=50, batch_size=64,
)

PARAM_NAMES = ["beta", "gamma", "i0", "p"]
test_data = workflow.simulate(num_samples=500)
posterior_samples = workflow.sample_posterior(data=test_data, num_samples=1_000)

true_params = test_data["inference_variables"]

for i, name in enumerate(PARAM_NAMES):
    true_vals = true_params[:, i]
    post_samples = posterior_samples[:, :, i]
    post_means = post_samples.mean(axis=-1)
    rmse = np.sqrt(np.mean((post_means - true_vals) ** 2))
    norm_rmse = rmse / (true_vals.std() + 1e-8)
    ss_res = np.sum((post_means - true_vals) ** 2)
    ss_tot = np.sum((true_vals - true_vals.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    lo = np.percentile(post_samples, 5, axis=-1)
    hi = np.percentile(post_samples, 95, axis=-1)
    coverage = np.mean((true_vals >= lo) & (true_vals <= hi))
    print(f"  {name}: norm_RMSE={norm_rmse:.3f} R2={r2:.3f} coverage={coverage:.2f}")
