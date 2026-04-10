# Analysis Notes — Bayesian Denoising (Fashion MNIST)

## Architecture
DiffusionModel with UNet subnet — follows references/image-generation.md exactly. No summary network (condition is already spatial).

## Adapter
Explicit bf.Adapter() chain: .convert_dtype -> .rename clean_image to inference_variables -> .rename blurry_image to inference_conditions.

## Training
fit_offline with Fashion MNIST. Blurry images created via Gaussian blur (sigma=2). 5000 train, 500 val.

## Diagnostics
Visual sample grid (not check_diagnostics.py — that script is for low-dimensional parameter targets). Posterior samples inspected as image grids.

## Key point
Condition (blurry image) is (B, 28, 28, 1), same spatial shape as target — directly channel-concatenable inside UNet.
