# Analysis Notes — Non-Identifiable Gaussian Mixture

## Non-identifiability
The two-component mixture has label-switching symmetry: swapping (mu1, mu2) and replacing w with 1-w leaves the likelihood unchanged.

## Expected diagnostics
- sigma: good recovery (identifiable from spread)
- w: moderate (partially identifiable)
- mu1, mu2: poor recovery near w~0.5 (label switching)

## Interpretation
Poor mu1/mu2 diagnostics are NOT a training failure — they reflect model symmetry. "Recovery bad / Calibration good" means the network honestly spreads mass over both modes.

## Remediation options
1. Ordering constraint (mu1 < mu2) — breaks symmetry
2. Accept non-identifiability — report marginals and derived quantities
3. Do NOT proceed to real-data inference with poor mu1/mu2 diagnostics unfixed
