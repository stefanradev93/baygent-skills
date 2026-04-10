# Analysis Notes (Without Skill)

## Approach
Used TimeSeriesTransformer + CouplingFlow. Adapter handles 4 parameter constraints.

## Network Configuration
Ad hoc sizes (summary_dim=8, 2 attention heads, 2 transformer blocks). No reference to model-sizes.md.

## Diagnostics
Manual per-parameter checks with RMSE, R-squared, and coverage thresholds. Does not use ECE/NRMSE house thresholds or check_diagnostics script.
