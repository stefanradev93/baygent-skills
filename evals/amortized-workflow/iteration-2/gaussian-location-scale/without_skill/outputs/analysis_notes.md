# Analysis Notes (Without Skill)

## Approach
Used bf.AmortizedWorkflow with SetTransformer and FlowMatching. Adapter chain with .constrain, .as_set, .convert_dtype.

## Network Configuration
Ad hoc SetTransformer (num_seeds=4, num_heads=4, embed_dim=64) and FlowMatching (hidden_units=[128, 128]). No reference to standardized model sizes.

## Diagnostics
Manual ECE and NRMSE checks against thresholds (ECE < 0.05, NRMSE < 0.15). No structured check_diagnostics script.
