# Analysis Notes (Without Skill)

## Approach
Uses FlowMatching with UNet subnet (not DiffusionModel as recommended by image-generation.md). Offline training on Fashion MNIST with Gaussian blur.

## Good
- No summary network (correct for image-to-image)
- Images not flattened
- Spatial shapes match for conditioning
- Visual sample grid for diagnostics
- Saves history as JSON

## Issues
- Adapter only does dtype conversion — routing done via dict key naming convention, not explicit .rename
- Uses FlowMatching instead of DiffusionModel (image-generation.md recommends DiffusionModel for image targets)
- Does not follow references/image-generation.md guidance
