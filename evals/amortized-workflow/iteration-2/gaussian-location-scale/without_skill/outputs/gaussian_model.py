"""
Amortized Bayesian inference for a Gaussian location-scale model using BayesFlow 2.x.
"""

import json
import numpy as np
import bayesflow as bf

RANDOM_SEED = sum(map(ord, "gaussian-location-scale"))
rng = np.random.default_rng(RANDOM_SEED)

def prior():
    mu = rng.normal(loc=0.0, scale=1.0)
    sigma = abs(rng.normal(loc=0.0, scale=1.0))
    return {"mu": mu, "sigma": sigma}

def observation_model(mu, sigma, n_obs=50):
    x = rng.normal(loc=mu, scale=sigma, size=(n_obs, 1))
    return {"x": x}

simulator = bf.make_simulator([prior, observation_model])

adapter = (
    bf.Adapter()
    .constrain("sigma", lower=0)
    .as_set("x")
    .convert_dtype("float64", "float32")
)

summary_network = bf.networks.SetTransformer(
    num_seeds=4, num_heads=4, embed_dim=64,
)

inference_network = bf.networks.FlowMatching(
    subnet_kwargs={"hidden_units": [128, 128]},
)

workflow = bf.AmortizedWorkflow(
    simulator=simulator,
    adapter=adapter,
    summary_network=summary_network,
    inference_network=inference_network,
)

validation_data = workflow.simulate(500)

history = workflow.fit_online(
    epochs=30, num_batches_per_epoch=100, batch_size=32,
    validation_data=validation_data,
)

with open("training_history.json", "w") as f:
    json.dump(history.history, f, indent=2)

workflow.inspect_training(history)

test_data = workflow.simulate(300)
diagnostics = workflow.compute_default_diagnostics(test_data)
print(diagnostics)

ece = diagnostics.get("ece", diagnostics.get("ECE", None))
nrmse = diagnostics.get("nrmse", diagnostics.get("NRMSE", None))

if ece is not None:
    print(f"ECE = {ece:.4f} [{'PASS' if ece < 0.05 else 'FAIL'}]")
if nrmse is not None:
    print(f"NRMSE = {nrmse:.4f} [{'PASS' if nrmse < 0.15 else 'FAIL'}]")

workflow.plot_default_diagnostics(test_data)

new_obs_batch = workflow.simulate(1)
posterior_samples = workflow.sample_posterior(new_obs_batch, num_samples=2_000)

mu_samples = posterior_samples["mu"]
sigma_samples = posterior_samples["sigma"]

print(f"mu    | mean={mu_samples.mean():.3f}  std={mu_samples.std():.3f}")
print(f"sigma | mean={sigma_samples.mean():.3f}  std={sigma_samples.std():.3f}")
