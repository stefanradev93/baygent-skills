# Analysis Notes — Gaussian Location-Scale

## Problem setup
Textbook 2-parameter Gaussian location-scale model: N=50 i.i.d. observations per dataset, inferring mu (unconstrained mean) and sigma (positive standard deviation).

## Key design decisions (all mandated by the skill)

**Data routing — SetTransformer required.** The 50 observations are exchangeable (i.i.d.), so they constitute a set, not a vector. They go through summary_variables + SetTransformer.

**Adapter chain:** .as_set(["x"]) → .constrain("sigma", lower=0) → .convert_dtype("float64", "float32") → .concatenate routing.

**summary_dim=4** — heuristic: 2 × number of parameters = 2 × 2 = 4.

**Small network sizes** from model-sizes.md. SetTransformer Small: embed_dims=(32,32), num_heads=(2,2). FlowMatching Small: widths=(128,128,128).

**validation_data=300** passed as integer to fit_online.

**Posterior accessed by original names "mu" and "sigma"** — not "inference_variables".

**PPCs reuse observation_model** — not a reimplementation.
