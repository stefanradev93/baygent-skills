"""
Non-identifiable Gaussian Mixture: Amortized Bayesian Inference with BayesFlow 2.x
"""

import numpy as np
import matplotlib.pyplot as plt
import bayesflow as bf

RANDOM_SEED = sum(map(ord, "non-identifiable-mixture"))
rng = np.random.default_rng(RANDOM_SEED)

def prior_fn(batch_size, rng=rng):
    w = rng.beta(2, 2, size=batch_size)
    mu1 = rng.normal(0.0, 3.0, size=batch_size)
    mu2 = rng.normal(0.0, 3.0, size=batch_size)
    sigma = np.abs(rng.normal(0.0, 1.0, size=batch_size)) + 1e-6
    return {"w": w, "mu1": mu1, "mu2": mu2, "sigma": sigma}

def likelihood_fn(params, n_obs=50, rng=rng):
    w, mu1, mu2, sigma = params["w"], params["mu1"], params["mu2"], params["sigma"]
    batch_size = w.shape[0]
    assignments = rng.binomial(1, w[:, None], size=(batch_size, n_obs))
    x = np.where(
        assignments == 1,
        rng.normal(mu1[:, None], sigma[:, None], size=(batch_size, n_obs)),
        rng.normal(mu2[:, None], sigma[:, None], size=(batch_size, n_obs)),
    )
    return {"x": x[:, :, None].astype(np.float32)}

adapter = (
    bf.Adapter()
    .constrain("w", method="sigmoid")
    .constrain("sigma", method="softplus")
)

summary_net = bf.networks.SetTransformer(input_dim=1, summary_dim=32)
inference_net = bf.networks.CouplingFlow(target_dim=4)

approximator = bf.Approximator(
    summary_network=summary_net,
    inference_network=inference_net,
    adapter=adapter,
)

dataset = bf.datasets.OnlineDataset(
    prior=prior_fn, likelihood=likelihood_fn, batch_size=64, num_batches=500,
)

history = approximator.fit(dataset=dataset, epochs=30, learning_rate=5e-4)

val_params = prior_fn(batch_size=1000)
val_data = likelihood_fn(val_params, n_obs=50)

diagnostics = bf.diagnostics.compute_default_diagnostics(
    approximator=approximator, prior_samples=val_params, data=val_data,
    parameter_names=["w", "mu1", "mu2", "sigma"], num_posterior_samples=500,
)

print("DIAGNOSTIC SUMMARY")
for param, metrics in diagnostics.items():
    print(f"[{param}]")
    for metric_name, value in metrics.items():
        if np.isscalar(value):
            print(f"  {metric_name}: {value:.4f}")

# Contraction check for label-switching cases near w ~ 0.5
near_half_mask = (val_params["w"] > 0.4) & (val_params["w"] < 0.6)
idx_near_half = np.where(near_half_mask)[0][:5]

param_names = ["w", "mu1", "mu2", "sigma"]
for idx in idx_near_half:
    test_data = {k: v[idx:idx+1] for k, v in val_data.items()}
    post_samples = approximator.sample(data=test_data, num_samples=500)
    for pname in param_names:
        samples = post_samples[pname].squeeze()
        contraction = 1.0 - (samples.std() / (3.0 if pname in ("mu1", "mu2") else 1.0))
        print(f"    {pname}: posterior_std={samples.std():.3f}, contraction={contraction:.3f}")
