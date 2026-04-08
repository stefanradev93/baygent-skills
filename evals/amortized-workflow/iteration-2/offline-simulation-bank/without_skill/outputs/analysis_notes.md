# Analysis Notes (Without Skill)

## Approach
Fully offline with fit_offline(). 80/10/10 split.

## Issues
- Uses fabricated bf.AmortizedPosterior class
- Doesn't save history.history as JSON (only PNG plot)
- Uses very lenient diagnostic thresholds (contraction > 0.1 vs skill's 0.80)
- No structured convergence check

## Good
- Correctly notes PPC limitation
- Uses SetTransformer for i.i.d. data
- Splits data properly
