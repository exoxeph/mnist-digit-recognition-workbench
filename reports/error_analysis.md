# Error Analysis

Overall test accuracy: 0.9845

## Most common confusions

- True `8` predicted as `5`: 8 times
- True `8` predicted as `3`: 5 times
- True `8` predicted as `9`: 4 times

## Why this matters

Confusions between visually similar digit shapes indicate the model relies on stroke geometry that overlaps between certain digit pairs (e.g. closed loops, similar curvature, or shared vertical strokes).

## Possible future improvements

- Data augmentation (rotation/elastic distortion) targeted at the most-confused pairs
- A deeper CNN or additional regularization (dropout, batch norm)
- Ensembling the V1 baseline and V2 CNN predictions