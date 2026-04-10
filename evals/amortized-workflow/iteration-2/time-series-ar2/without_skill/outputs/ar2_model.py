"""
AR(2) Amortized Bayesian Inference with BayesFlow 2.x
"""

import json
import numpy as np
import bayesflow as bf
from bayesflow.simulators import make_simulator

def prior():
    phi1 = np.random.uniform(-1.5, 1.5)
    phi2 = np.random.uniform(-1.0, 1.0)
    log_sigma = np.random.uniform(-2.0, 1.5)
    sigma = np.exp(log_sigma)
    return {"phi1": phi1, "phi2": phi2, "sigma": sigma}

def simulator(phi1, phi2, sigma, T=100):
    y = np.zeros(T)
    for t in range(2, T):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + np.random.normal(0.0, sigma)
    return {"summary_variables": y.reshape(T, 1).astype(np.float32)}

ar2_simulator = make_simulator([prior, simulator])

summary_network = bf.networks.TimeSeriesTransformer(
    input_dim=1, embed_dim=64, num_heads=4, num_layers=2, dropout=0.05,
)

inference_network = bf.networks.CouplingFlow(
    num_layers=6, subnet_kwargs={"hidden_dim": 128, "num_layers": 2},
)

approximator = bf.Approximator(
    inference_network=inference_network,
    summary_network=summary_network,
)
approximator.constrain("sigma", lower=0)

RANDOM_SEED = sum(map(ord, "ar2-amortized-inference"))
np.random.seed(RANDOM_SEED)

NUM_TRAIN = 10_000
NUM_VAL = 1_000

training_data = ar2_simulator.sample(NUM_TRAIN)
validation_data = ar2_simulator.sample(NUM_VAL)

history = approximator.fit_online(
    simulator=ar2_simulator, epochs=50, iterations_per_epoch=200,
    batch_size=64, validation_data=validation_data,
    optimizer_kwargs={"learning_rate": 5e-4},
)

history_dict = {
    "train_loss": [float(v) for v in history.history.get("loss", [])],
    "val_loss": [float(v) for v in history.history.get("val_loss", [])],
}
with open("ar2_training_history.json", "w") as f:
    json.dump(history_dict, f, indent=2)

NUM_TEST = 500
test_data = ar2_simulator.sample(NUM_TEST)

posterior_samples = approximator.sample(conditions=test_data, num_samples=1000)

diagnostics = bf.diagnostics.run_sbc(
    prior_samples={"phi1": test_data["phi1"], "phi2": test_data["phi2"], "sigma": test_data["sigma"]},
    posterior_samples={"phi1": posterior_samples["phi1"], "phi2": posterior_samples["phi2"], "sigma": posterior_samples["sigma"]},
)

for param in ["phi1", "phi2", "sigma"]:
    truth = test_data[param]
    post = posterior_samples[param]
    lo, hi = np.percentile(post, [5, 95], axis=-1)
    coverage = np.mean((truth >= lo) & (truth <= hi))
    print(f"  {param}: 90% CI coverage = {coverage:.3f}")

single_obs = ar2_simulator.sample(1)
post_single = approximator.sample(conditions=single_obs, num_samples=4000)

for param in ["phi1", "phi2", "sigma"]:
    samples = post_single[param].squeeze()
    print(f"  {param}: mean={samples.mean():.3f} std={samples.std():.3f}")
