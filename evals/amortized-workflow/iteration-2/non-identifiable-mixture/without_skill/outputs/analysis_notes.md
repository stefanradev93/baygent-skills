# Analysis Notes (Without Skill)

## Label switching
Discusses non-identifiability extensively. Notes that mu1/mu2 will show poor recovery near w=0.5.

## Remediation
Suggests ordering constraint (mu1 < mu2) or accepting non-identifiability.

## Issues
- Uses fabricated bf.Approximator, bf.datasets.OnlineDataset, bf.diagnostics.compute_default_diagnostics()
- Uses .constrain(method="sigmoid") instead of .constrain(lower=0, upper=1) — wrong kwarg
- Does not check ECE per parameter
- No reference to model-sizes.md
