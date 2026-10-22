# Project Overview Report: Tomato Growth-Stage Classification

**Repository:** `kosalanayanajithdeshapriya/Resnet--Research-2026-sequence-`, branch `organized-repo-structure`
**Related repository:** `kosalanayanajithdeshapriya/Unet_Research_2026` (U-Net training and cross-validation)
**Date:** 22 October 2026

## 1. Purpose

The project classifies tomato plant photos into four growth stages (**seeding, developing, flowering, fruiting**) with ResNet50. It then asks whether the classifier uses the plant or the background to decide. The final system chains a segmenter to a masked classifier so that it works on new, unannotated photos:

```
photo -> segmenter (plant vs. background) -> mask out background -> ResNet50 classifier -> growth stage
```

## 2. Research questions and where each is answered

| # | Question | Notebook |
|---|---|---|
| 1 | Can a plain ResNet50 distinguish growth stages? | `01_resnet50_baseline` |
| 2 | Does the model rely on background context? | `02_resnet50_masked_comparison` (Grad-CAM and a 2x2 masked/original test) |
| 3 | Can predicted masks replace ground-truth annotations? | `03_deeplabv3_segmentation_pipeline` |
| 4 | Is U-Net or DeepLabV3 the better segmenter for the pipeline? | `04_unet_vs_deeplabv3_comparison`, `05_deeplabv3_pipeline_evaluation` |
| 5 | How stable are the results under 5-fold cross-validation? | `03` (sections 9-10), reports |
| - | Extra experiment: ResNet101 on U-Net-masked images (Colab) | `06_resnet101_colab_training` |

Dependency chain: `01 -> 02 -> 03 -> {04, 05}`. Later notebooks load checkpoints from earlier ones and never retrain them in place.

## 3. Data

- Annotated set ("tomato final dataset", COCO polygons): 979 images, split 684 train / 135 validation / 160 test.
- Test supports: developing 41, flowering 60, fruiting 21, seeding 38.
- The baseline uses a separate, unmasked dataset folder (`dataset/`).
- Datasets and model weights are **not** stored in git (too large); the notebooks regenerate them.

## 4. Repository structure

```
notebooks/     six numbered notebooks (training and evaluation)
src/           dataset helper modules imported by the notebooks
scripts/       masked-dataset generator, report and figure builders
results/       metrics (CSV/JSON) and figures produced by the notebooks
reports/       written reports (md/pdf) and organized thesis figures
paper_report/  paper-ready report (md/docx/pdf), figures, recompute scripts
docs/          per-notebook documentation and this report
export_masked/ exported masked classifier
```

The notebooks and scripts were originally flat in the repo root. They are now organized into these folders, and the file paths inside them were updated. Only path and configuration lines changed; no model code or saved outputs were altered.

## 5. Key results (test set, 160 images unless stated)

| Experiment | Result |
|---|---|
| Baseline ResNet50, original images | 93.1% accuracy (96.9% in the comparison run) |
| Original-trained model, ground-truth-masked test images | 76.9% |
| Masked-trained model, original images | 35.6% |
| Masked-trained model, ground-truth masks (oracle upper bound) | 91.9% |
| Pipeline, DeepLabV3 segmenter | 84.4% (87.5% in the standalone re-evaluation) |
| Pipeline, U-Net native preprocessing | 78.8% |
| Pipeline, U-Net crop-aligned preprocessing | 89.4% |

Segmentation quality (fixed split, plant class):

| Segmenter | Foreground IoU | Foreground Dice |
|---|---|---|
| DeepLabV3-ResNet50 | 0.623 | 0.757 |
| U-Net native | 0.665 | 0.784 |
| U-Net crop-aligned | 0.667 | 0.783 |

DeepLabV3, 5-fold cross-validation (mean over folds):

| Measure | Value |
|---|---|
| Full pipeline accuracy | 82.2% |
| Full pipeline macro F1 | 0.741 |
| Segmentation IoU inside the pipeline | 0.749 |
| Segmentation-only CV IoU | 0.747 |

**Main findings**
1. The original classifier depends on background context: accuracy falls from 96.9% to 76.9% once the background is removed.
2. A classifier trained on masked images reaches 91.9% with ground-truth masks but 35.6% on unmasked images. It must be paired with a segmenter.
3. With predicted masks, the pipeline reaches 82-89% depending on the segmenter and preprocessing, close to the oracle upper bound.
4. Preprocessing geometry matters as much as the segmenter: the same U-Net moves from 78.8% to 89.4% when fed DeepLabV3-style cropped input.

## 6. Caveats to be aware of

- The fixed-split and cross-validation numbers differ (for example DeepLabV3 pipeline 84.4% or 87.5% versus 82.2%). The epoch budgets also differ, so the gap cannot be attributed to split luck alone.
- The DeepLabV3 confusion-matrix PNG is written by two notebooks under the same filename. The saved file currently corresponds to the 87.5% standalone run.
- IoU and Dice are averaged per image in the comparison notebooks but per batch in the training notebook, so the two sets of values are not strictly identical aggregations.
- The earlier thesis summary quoted 0.7565 as foreground IoU. It is foreground Dice; the IoU is 0.623.
- Overlapping standard deviations across folds are not a significance test, and no significance tests were run.
- The U-Net models were trained and cross-validated in the separate `Unet_Research_2026` repository. This repo only loads those checkpoints, so the U-Net CV results cannot be reproduced from this repo alone.
- Notebook 06 (Colab ResNet101) has no results stored in this repo.
- Notebooks set `PROJECT_ROOT` to `d:\Reasearch\Resnet50test`; this must be changed on another machine.

## 7. Reports in the repo

| File | Content |
|---|---|
| `reports/FULL_RESULTS_UNET_DEEPLABV3.md` / `.pdf` | Full tables, per-class reports, confusion matrices, fold-level CV |
| `paper_report/PAPER_RESULTS_REPORT.*` | Paper-ready results with figures |
| `paper_report/SEQUENTIAL_PIPELINE_RESULTS.*` | Sequential-pipeline results |
| `reports/thesis_figures/` | Organized thesis figures (see `INDEX.txt`) |

## 8. Reproducing

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

1. Place the datasets in the repo root.
2. Set `PROJECT_ROOT` in each notebook's config cell.
3. Run `python scripts/generate_masked_dataset.py`.
4. Run notebooks 01 to 05 in order; outputs go to `results/`.
5. Rebuild reports with `scripts/build_full_results_report.py` and `scripts/build_paper_figures.py`.

## 9. Suggested next steps

- Add the U-Net notebooks to this repo or link them from the README, so the full pipeline can be reproduced.
- Replace the hard-coded absolute paths with relative paths.
- Add significance tests or confidence intervals for the model comparisons.
