"""
Amortized Bayesian inference for linear regression with variable sample size N.
"""

import numpy as np
import bayesflow as bf

def meta_fn():
    N = np.random.randint(10, 201)
    return {"N": N}

def prior_fn():
    alpha = np.random.normal(0.0, 5.0)
    beta = np.random.normal(0.0, 2.0)
    sigma = np.abs(np.random.normal(0.0, 1.0)) + 1e-3
    return {"alpha": alpha, "beta": beta, "sigma": sigma}

def observation_model(alpha, beta, sigma, N):
    x = np.random.uniform(-3.0, 3.0, size=(N,))
    y = alpha + beta * x + np.random.normal(0.0, sigma, size=(N,))
    return {"x": x, "y": y, "N": N}

simulator = bf.make_simulator([prior_fn, observation_model], meta_fn=meta_fn)

adapter = (
    bf.Adapter()
    .concatenate(["x", "y"], into="summary_variables", axis=-1)
    .as_set("summary_variables")
    .apply(np.sqrt, to="N", resulting_in="sqrt_N")
    .rename("sqrt_N", "inference_conditions")
    .constrain("sigma", lower=0.0)
    .rename(["alpha", "beta", "sigma"], "inference_variables")
    .drop(["x", "y", "N"])
)

summary_network = bf.networks.SetTransformer(input_dim=2, output_dim=64)
inference_network = bf.networks.CouplingFlow(num_layers=6, subnet_kwargs={"units": 128, "activation": "relu"})

workflow = bf.BasicWorkflow(
    simulator=simulator, adapter=adapter,
    summary_network=summary_network, inference_network=inference_network,
    inference_conditions_dim=1,
)

RANDOM_SEED = sum(map(ord, "regression-varying-n"))
np.random.seed(RANDOM_SEED)

train_data = simulator.sample(2000)
val_data = simulator.sample(400)

history = workflow.fit_offline(train_data, epochs=50, batch_size=32, validation_data=val_data)

diagnostics_data = simulator.sample(500)
diagnostics = workflow.compute_default_diagnostics(diagnostics_data)
print("Default diagnostics:", diagnostics)

new_meta = meta_fn()
new_params = prior_fn()
new_obs = observation_model(**new_params, **new_meta)

posterior_samples = workflow.sample_posterior(num_samples=2000, conditions=new_obs)

for param in ["alpha", "beta", "sigma"]:
    samples = posterior_samples[param]
    print(f"  {param}: {samples.mean():.3f}  (true: {new_params[param]:.3f})")

def posterior_predictive_check(posterior_samples, x_new, n_ppc=200):
    n_draws = min(n_ppc, len(posterior_samples["alpha"]))
    ppc_ys = []
    N_new = len(x_new)
    for i in range(n_draws):
        alpha_i = posterior_samples["alpha"][i]
        beta_i = posterior_samples["beta"][i]
        sigma_i = posterior_samples["sigma"][i]
        y_i = alpha_i + beta_i * x_new + np.random.normal(0, sigma_i, size=N_new)
        ppc_ys.append(y_i)
    return np.array(ppc_ys)

x_test = np.linspace(-3, 3, 50)
ppc = posterior_predictive_check(posterior_samples, x_test)
print(f"PPC shape: {ppc.shape}")
