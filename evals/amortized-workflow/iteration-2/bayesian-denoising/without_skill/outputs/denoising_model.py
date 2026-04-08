"""
Bayesian Denoising with BayesFlow 2.x
Fashion MNIST: clean images as targets, blurry images as conditions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import keras
(x_train_full, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()

N_TRAIN = 8000
N_VAL = 1000
N_TEST = 8

x_train_clean = x_train_full[:N_TRAIN]
x_val_clean = x_train_full[N_TRAIN:N_TRAIN + N_VAL]
x_test_clean = x_test[:N_TEST]

def blur_images(images, sigma=2.0):
    blurred = np.stack([gaussian_filter(img.astype(np.float64), sigma=sigma) for img in images])
    return blurred

x_train_blurry = blur_images(x_train_clean)
x_val_blurry = blur_images(x_val_clean)
x_test_blurry = blur_images(x_test_clean)

x_train_clean = x_train_clean[..., np.newaxis].astype(np.float64)
x_val_clean = x_val_clean[..., np.newaxis].astype(np.float64)
x_test_clean = x_test_clean[..., np.newaxis].astype(np.float64)
x_train_blurry = x_train_blurry[..., np.newaxis]
x_val_blurry = x_val_blurry[..., np.newaxis]
x_test_blurry = x_test_blurry[..., np.newaxis]

train_data = {"inference_variables": x_train_clean, "inference_conditions": x_train_blurry}
val_data = {"inference_variables": x_val_clean, "inference_conditions": x_val_blurry}

import bayesflow as bf
from bayesflow.networks import FlowMatching
from bayesflow.networks.unet import UNet

adapter = bf.Adapter().convert_dtype("float64", "float32")

unet = UNet(spatial_dims=2)
approximator = FlowMatching(subnet=unet)

workflow = bf.BasicWorkflow(approximator=approximator, adapter=adapter)

history = workflow.fit_offline(data=train_data, epochs=20, batch_size=64, validation_data=val_data)

history_path = "denoising_history.json"
with open(history_path, "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

losses = history.history.get("loss", [])
val_losses = history.history.get("val_loss", [])
print(f"Final train loss: {losses[-1]:.4f} | Final val loss: {val_losses[-1]:.4f}")

if abs(losses[-1] - val_losses[-1]) / (abs(val_losses[-1]) + 1e-8) > 0.3:
    print("WARNING: possible overfitting")
else:
    print("Convergence looks healthy")

N_SAMPLES = 4
test_conditions = {"inference_conditions": x_test_blurry.astype(np.float32)}

posterior_samples = workflow.sample(conditions=test_conditions, num_samples=N_SAMPLES)

ps = np.array(posterior_samples)
if ps.shape[0] == N_SAMPLES and ps.shape[1] == N_TEST:
    ps = ps.transpose(1, 0, 2, 3, 4)
ps = ps[..., 0]

fig, axes = plt.subplots(N_TEST, N_SAMPLES + 2, figsize=(2 * (N_SAMPLES + 2), 2 * N_TEST))
for i in range(N_TEST):
    axes[i, 0].imshow(x_test_blurry[i, ..., 0], cmap="gray", vmin=0, vmax=255)
    axes[i, 0].axis("off")
    for j in range(N_SAMPLES):
        axes[i, j + 1].imshow(ps[i, j], cmap="gray")
        axes[i, j + 1].axis("off")
    axes[i, -1].imshow(x_test_clean[i, ..., 0], cmap="gray", vmin=0, vmax=255)
    axes[i, -1].axis("off")

fig.suptitle("Posterior clean images — Fashion MNIST denoising")
fig.tight_layout()
fig.savefig("denoising_posterior_grid.png", dpi=150)
plt.close(fig)
print("Saved denoising_posterior_grid.png")
