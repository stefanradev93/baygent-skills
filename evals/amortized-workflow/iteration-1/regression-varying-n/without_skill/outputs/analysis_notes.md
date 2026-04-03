# Analysis Notes: Amortized Linear Regression with Variable N

## Model

Simple linear regression: `y = alpha + beta * x + Normal(0, sigma)`.

Priors:
- `alpha ~ Normal(0, 5)` — weakly informative, centered at zero
- `beta ~ Normal(0, 5)` — weakly informative
- `sigma ~ HalfNormal(2)` — positive-only, weakly informative

## Variable Sample Size Strategy

The key challenge is that neural networks require fixed-dimensional inputs, but N varies from 10 to 200.

**Approach: Padding + Masking with SetTransformer**

1. Each simulated dataset draws `N ~ Uniform{10, ..., 200}`.
2. Observations are padded to `N_MAX = 200` with zeros.
3. A boolean mask indicates which entries are real vs. padding.
4. A **SetTransformer** summary network processes the variable-length set, using the mask to ignore padded positions. It outputs a fixed 32-dimensional summary vector regardless of input size.

This is the standard approach in BayesFlow for set-valued data with variable cardinality. The SetTransformer learns permutation-invariant summaries via attention, which naturally respects masking.

## Architecture Choices

- **Summary network**: SetTransformer with 4 attention heads, 2 dense layers of 128 units, producing a 32-dim summary. Attention-based pooling is preferred over DeepSets for variable-N problems because it can weight observations adaptively.
- **Inference network**: CouplingFlow with 6 coupling layers (128 units, 2 dense layers each). This is a conditional normalizing flow that maps the summary to a posterior over 3 parameters.
- **Training**: 50 epochs, 1000 simulations/epoch, batch size 64 (50k total simulated datasets).

## Diagnostics

1. **Training loss curve** — should decrease and plateau.
2. **SBC histograms** — rank statistics should be uniform if the posterior is well-calibrated.
3. **SBC ECDF** — empirical CDF of ranks vs. theoretical uniform; deviations indicate miscalibration.
4. **Z-score vs. contraction** — checks that posteriors are both accurate (low z-score) and informative (high contraction relative to prior).

## Generalization Check

The script tests inference at N = 10, 25, 50, 100, 200 with the same true parameters. Expected behavior:
- Posterior means should be close to true values at all N.
- Posterior standard deviations should shrink as N increases (more data = more precision).
- Even at N = 10, the network should return reasonable (if wide) posteriors.

## Limitations

- The network only generalizes to N in [10, 200] — extrapolation beyond this range is unreliable.
- Training budget (50 epochs, 50k sims) is moderate. For production use, increase to 100+ epochs or use early stopping.
- The HalfNormal prior on sigma is implemented via `abs(Normal)` + clipping; a log-normal or truncated normal would be cleaner but this is standard practice.
- No explicit sufficient statistics are provided — the network must learn them from raw (x, y) pairs. For linear regression this is fine; for more complex models, hand-crafted summaries can help.
