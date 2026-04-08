# Analysis Notes — AR(2) Time Series

## Summary network
TimeSeriesTransformer (Small) — NOT SetTransformer. AR(2) data is temporally ordered, not exchangeable.

## Adapter
.as_time_series → .constrain("sigma", lower=0) → .convert_dtype → .concatenate routing.

## Architecture
Small config. summary_dim=6 (2 × 3 params). TimeSeriesTransformer Small: embed_dims=(32,32), num_heads=(2,2).

## Prior
Rejection-sampling enforces AR(2) stationarity to avoid explosive series during training.

## PPCs
Reuses observation_model directly with posterior draws.
