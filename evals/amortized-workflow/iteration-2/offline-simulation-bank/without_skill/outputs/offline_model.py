"""
Offline amortized Bayesian inference with BayesFlow 2.x
"""

import os
import glob
import numpy as np
import bayesflow as bf
from sklearn.model_selection import train_test_split

BANK_DIR = "./simulation_bank"
NPZ_FILES = sorted(glob.glob(os.path.join(BANK_DIR, "*.npz")))

if not NPZ_FILES:
    raise FileNotFoundError(f"No .npz files found in {BANK_DIR}")

all_parameters, all_observables = [], []
for fpath in NPZ_FILES:
    data = np.load(fpath)
    all_parameters.append(data["parameters"])
    all_observables.append(data["observables"])

parameters = np.stack(all_parameters, axis=0)
observables = np.stack(all_observables, axis=0)

RANDOM_SEED = sum(map(ord, "offline-simulation-bank"))

idx = np.arange(len(parameters))
train_idx, temp_idx = train_test_split(idx, test_size=0.20, random_state=RANDOM_SEED, shuffle=True)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED)

train_data = {"parameters": parameters[train_idx], "observables": observables[train_idx, :, np.newaxis]}
val_data = {"parameters": parameters[val_idx], "observables": observables[val_idx, :, np.newaxis]}
test_data = {"parameters": parameters[test_idx], "observables": observables[test_idx, :, np.newaxis]}

summary_net = bf.networks.SetTransformer(input_dim=1, output_dim=32)
inference_net = bf.networks.CouplingFlow(target_dim=3)

workflow = bf.AmortizedPosterior(
    inference_network=inference_net,
    summary_network=summary_net,
    inference_variables=["parameters"],
    summary_variables=["observables"],
)

history = workflow.fit_offline(
    simulations=train_data, epochs=50, batch_size=256, validation_data=val_data,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 4))
plt.plot(train_loss, label="Train loss")
plt.plot(val_loss, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_history.png", dpi=150)
plt.close()

print(f"Final train loss: {train_loss[-1]:.4f}")
print(f"Final val loss: {val_loss[-1]:.4f}")

workflow.save("offline_amortized_posterior")

diagnostics = workflow.compute_default_diagnostics(simulations=test_data, num_posterior_samples=500)

print("Diagnostics summary:")
print(diagnostics)

passed = True
if "posterior_z_scores" in diagnostics:
    mean_z = np.abs(diagnostics["posterior_z_scores"]).mean()
    ok = mean_z < 2.0
    print(f"Mean |z-score|: {mean_z:.3f} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok
if "posterior_contraction" in diagnostics:
    min_contraction = diagnostics["posterior_contraction"].min()
    ok = min_contraction > 0.1
    print(f"Min contraction: {min_contraction:.3f} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

print(f"Overall: {'PASSED' if passed else 'FAILED'}")

single_obs = test_data["observables"][[0]]
single_true = test_data["parameters"][[0]]

posterior_samples = workflow.sample(conditions={"observables": single_obs}, num_samples=2000)

print("Posterior means:", posterior_samples.mean(axis=1))
print("True parameters:", single_true)

print("NOTE: PPCs skipped — simulator unavailable.")
