# AR(2) Amortized Inference -- Design Notes

## Architecture choices

- **Summary network**: `TimeSeriesTransformer` — the observations are temporally ordered, not exchangeable. Using a SetTransformer or DeepSet would destroy the ordering information critical for identifying AR coefficients. The TimeSeriesTransformer includes Time2Vec positional embeddings via `time_axis=1` to encode temporal position.
- **Inference network**: `FlowMatching` — continuous normalizing flow trained with the flow matching objective. Good default for low-dimensional parameter spaces (3 parameters here).
- **summary_dim = 8**: slightly above the common 2x-num-params heuristic (which gives 6) to provide extra capacity for learning useful summary statistics.
- Used default-sized architecture (embed_dims=64, num_heads=4, mlp_widths=128) since the problem is moderately complex — temporal dependencies across 100 time steps with 3 parameters to infer.

## Adapter design

- `.as_time_series(["y"])` reshapes the flat `(T,)` vector to `(T, 1)` so the transformer receives a proper 3D input `(batch, T, 1)`.
- `.constrain("sigma", lower=0)` applies a softplus bijection so the network operates in unconstrained space but sigma remains positive after inverse transform.
- `phi1` and `phi2` left unconstrained — the Uniform(-1,1) prior bounds them during simulation, but no hard bijection is needed for the network.
- Observations routed through `summary_variables` (processed by summary network), parameters through `inference_variables`.

## Prior specification

- `phi1, phi2 ~ Uniform(-1, 1)` — covers both stationary and non-stationary AR(2) processes.
- `sigma ~ Gamma(2, 1)` — weakly informative, ensures positivity, mode near 1.

## Training

- Online training (`fit_online`) since the AR(2) simulator is cheap (pure NumPy loop).
- 50 epochs, 100 batches/epoch, batch_size=32 = 160,000 total simulated series.
- Validation set of 300 simulations for overfitting monitoring.

## Diagnostics

- 300 held-out test simulations for calibration and recovery checks.
- Default BayesFlow diagnostics: ECDF calibration, posterior z-score, contraction, recovery plots.
- Manual posterior predictive checks comparing lag-1 and lag-2 autocorrelation between observed and replicated data — these are the most informative test quantities for AR(2) since they directly relate to the AR coefficients.

## Limitations

- Prior does not enforce stationarity (characteristic roots inside unit circle). This means the network sees some explosive trajectories during training, which is intentional for generality but may reduce efficiency.
- Fixed sequence length T=100. Variable-length series would require padding/truncation or passing T as an inference condition.
- Gaussian noise is hard-coded. Heavy-tailed or heteroskedastic noise would require modifying the simulator.
