# Slice-presence baseline

The task is **does this axial CT slice contain any labelled spleen?**
All labelled dataset volumes contain spleen, so whole-volume yes/no detection
would not provide the needed negative examples. Slices without a spleen label
are negatives. A positive target needs at least one native labelled voxel.

## Design

```text
CT -> fixed features -> training-only scaler -> logistic score
Mask -> native per-slice target -----------------> evaluation
Case IDs -> frozen train / validation / test membership
```

- Features use CT intensities only. No mask crops, masked statistics, slice
  indices, case IDs, or label-derived location enter the model.
- Input features are 12x12 field-of-view tile means and standard deviations,
  a 16-bin intensity histogram, and two mean absolute image gradients.
- Intensities use a fixed [-200, 300] HU window. Tile coordinates normalize the
  field of view; they do not register anatomy or equalize physical spacing.
- A StandardScaler and class-balanced logistic regression fit training cases
  only. C=1, maximum 2000 iterations, one numerical-library thread.
- Default seed is 42. About 60/20/20 percent of cases go to train/validation/test.
  For 41 cases this gives 25/8/8. At least five cases are required.
- Cases are split before slice examples are collected. Dataset case IDs are
  the grouping unit; unique patient identity has not been independently verified.
- Validation is reported separately. There is no tuning or model selection.
  The classification threshold stays fixed at 0.5.
- Class-balanced logistic scores are not calibrated clinical probabilities.
  Cell fractions in the resolution viewer are a different quantity entirely.

## Resolution comparison

Default training uses native resolution (factor 1). The same frozen model is
evaluated at factors 1, 2 and 4 on the same cases and native slice targets.

A factor 4 input averages 4x4x4 CT blocks. Its coarse slab is mapped back to
each corresponding native slice. This mixes neighboring slice information;
it is not purely an in-plane blur. Original targets remain unchanged,
including very small labelled edge slices. Partial edge blocks use their
actual counts, without zero padding.

This protocol measures a fixed model's response to lower-resolution inputs.
It does **not** measure the best achievable accuracy after retraining.
For a separate matched-resolution experiment, choose a new run folder:

```bash
python -m ct_poc.cli train --cases all --train-factor 4 --output outputs/train-at-4
```

Do not choose a resolution using test results and then claim an unbiased
test score. Use training/validation for future choices and reserve fresh
cases if earlier test results have already influenced those choices.

## Commands and outputs

Training is explicit and independent from the viewer:

```bash
python -m ct_poc.cli train --cases all --output outputs/baseline
python -m ct_poc.cli view --results-dir outputs/baseline
```

The run folder must be new or empty. Prior results are never overwritten.

| File | Content |
| --- | --- |
| manifest.json | Case split, seed, settings, hashes and library versions |
| model.npz | Numeric scaler and logistic arrays; no pickle |
| report.md | Held-out results and protocol notes |
| metrics.csv / metrics.json | Pooled metrics and majority comparator |
| per-case.csv | Metrics for each case and resolution |
| predictions.csv | Native slice target, score and decision |

Sensitivity measures the share of positive slices found. Specificity measures
the share of negative slices rejected. Balanced accuracy averages them.
Undefined metrics (such as sensitivity without any positive targets) are
null rather than fabricated zeros. Macro-case balanced accuracy averages
defined case values equally; pooled slice metrics weight longer scans more.
No slice-level confidence intervals are reported: neighboring slices are
correlated, and the held-out sample is small.

The majority comparator chooses its constant prediction from training labels.
Feature caches live under DATA_DIR/features-v1, keyed by source hashes, factor,
feature settings and algorithm version. A changed scan or mask rebuilds the
affected cache. Model manifests contain hashes, not machine-specific paths.

## Predict an image without labels

No spleen mask or ML training dependency is needed for saved-model inference:

```bash
python -m ct_poc.cli predict --image data/raw/imagesTr/spleen_3.nii.gz --model outputs/baseline/model.npz --output outputs/predictions.csv
```

Prediction requires an axis-aligned 3D NIfTI CT with mm units. It uses the same
orientation and geometry checks as the viewer. Native slice indices are in
canonical RAS orientation, which may differ from the input file's array order.
An image used in training is useful for software checks but not an independent
test. Predictions do not supply a pixel-level segmentation.

## Limits and next branch

This is a transparent CPU baseline, not a trained segmentation network.
It may learn acquisition or anatomy shortcuts and may perform poorly.
Synthetic demo accuracy checks plumbing only. Real test accuracy is exploratory,
not clinical validation or a demonstrated research contribution.
Future routing should compare against fixed-spacing and random slice choices
under the same budget. Routing and reconstruction are not implemented here.

The training loss weights slices, with class balancing; longer volumes therefore
contribute more examples. The majority row is a hard classifier comparator.
Its Brier value is not a prevalence-based probability baseline. Standalone
prediction exports include CT and model hashes for traceability.
