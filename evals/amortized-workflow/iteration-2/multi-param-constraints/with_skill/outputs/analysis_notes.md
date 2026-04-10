# Analysis Notes — Multi-Parameter Constraints (SIR)

## Routing decision
Daily case counts are an ordered time series — not exchangeable. Uses TimeSeriesTransformer via summary_variables.

## Constraints
- beta, gamma: .constrain(lower=0) — softplus bijection
- i0, p: .constrain(lower=0, upper=1) — sigmoid bijection
- All constraints applied before .concatenate per adapter.md ordering rule.

## Architecture
Small config throughout. summary_dim=8 (2 × 4 parameters). TimeSeriesTransformer Small: embed_dims=(32,32), num_heads=(2,2). FlowMatching Small: widths=(128,128,128).

## Diagnostics
All 4 parameters checked individually via check_diagnostics(). STOP gate enforced.

## PPCs
Reuses sir_observation_model directly — no reimplementation.
