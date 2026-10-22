# Full results report: U-Net and DeepLabV3

Segmentation models, complete growth-stage pipelines, and cross-validation. Compiled 30 September 2026 from saved evidence in this workspace. No training or inference was rerun.

This report includes all available aggregate and per-stage segmentation results, per-class pipeline reports, confusion matrices, and fold-level CV metrics found in the local result files and relevant notebooks. DeepLabV3 has completed segmentation-only and full-pipeline five-fold CV. U-Net has fixed-split results for two preprocessing variants; no U-Net CV results were found in this workspace. The external U-Net training project was not audited.

All metric values are proportions from 0 to 1 unless a percent sign is shown. For example, 0.8750 means 87.50%. Saved precision is preserved; two-decimal notebook classification reports are not silently presented as four-decimal measurements. CV +/- values are sample standard deviations over five folds, not confidence intervals.

The fixed annotated split has 684 training, 135 validation, and 160 test images. Test supports are developing 41, flowering 60, fruiting 21, and seeding 38. Segmentation predicts background versus plant; pipeline classification predicts these four growth stages. A per-stage segmentation score measures mask quality within images of that stage, not stage-classification accuracy.

## 1. Segmentation models only

### 1.1 Model and evaluation context

| Model / variant | Saved model context | Evaluation geometry |
| --- | --- | --- |
| DeepLabV3-ResNet50 | Pretrained segmenter, fine-tuned; 41,994,308 trainable parameters in training output | Resize then center crop to 224 x 224 |
| U-Net native | Externally trained custom U-Net; 7,765,985 parameters; threshold 0.5 | Direct resize to 224 x 224 |
| U-Net crop-aligned | Same U-Net checkpoint; inference preprocessing variant | Resize then center crop to 224 x 224 |

The comparison code uses Resize(int(224 * 1.14)), which is Resize(255), followed by CenterCrop(224). Some prose describes this as Resize(256); the executable code is the more precise record. Native U-Net and crop-aligned U-Net use different image geometry and matching ground-truth geometry. These are not three independently trained segmentation models.

Foreground IoU and Dice score plant pixels only. Both-class IoU averages background and plant IoU. The comparison notebooks compute these per image and average over images. The training/CV notebook computes both-class IoU per batch and averages batches; these aggregations should not be treated as identical. Pixel-level macro precision/recall/F1 in the training notebook come from the accumulated background/plant confusion matrix.

### 1.2 Fixed-split segmentation comparison: 160 test images

| Segmenter / preprocessing | Plant IoU | Plant Dice | Both-class IoU |
| --- | --- | --- | --- |
| DeepLabV3-ResNet50 | 0.6230 | 0.7565 | 0.7507 |
| U-Net (custom, from scratch) | 0.6646 | 0.7844 | 0.7926 |
| U-Net (crop-aligned, DeepLabV3-style preprocessing) | 0.6674 | 0.7827 | 0.7753 |

Source: [unet_vs_deeplabv3_comparison_v2.csv](unet_vs_deeplabv3_comparison_v2.csv).

### 1.3 DeepLabV3: comparison run

| Stage | Plant IoU | Plant Dice | Both-class IoU | Images |
| --- | --- | --- | --- | --- |
| developing | 0.6883 | 0.7973 | 0.7675 | 41 |
| flowering | 0.5994 | 0.7433 | 0.7212 | 60 |
| fruiting | 0.5059 | 0.6582 | 0.6797 | 21 |
| seeding | 0.6548 | 0.7877 | 0.8185 | 38 |

Source: [unet_leaf_pipeline_comparison.ipynb](unet_leaf_pipeline_comparison.ipynb), notebook cell 17 (counting all cells from 1).

### 1.4 U-Net: native resize

| Stage | Plant IoU | Plant Dice | Both-class IoU | Images |
| --- | --- | --- | --- | --- |
| developing | 0.6865 | 0.7992 | 0.7848 | 41 |
| flowering | 0.6307 | 0.7657 | 0.7682 | 60 |
| fruiting | 0.5181 | 0.6539 | 0.7115 | 21 |
| seeding | 0.7755 | 0.8702 | 0.8842 | 38 |

Source: [unet_leaf_pipeline_comparison.ipynb](unet_leaf_pipeline_comparison.ipynb), notebook cell 15 (counting all cells from 1).

### 1.5 U-Net: crop-aligned

| Stage | Plant IoU | Plant Dice | Both-class IoU | Images |
| --- | --- | --- | --- | --- |
| developing | 0.6730 | 0.7816 | 0.7594 | 41 |
| flowering | 0.6362 | 0.7696 | 0.7453 | 60 |
| fruiting | 0.4998 | 0.6318 | 0.6738 | 21 |
| seeding | 0.8031 | 0.8881 | 0.8962 | 38 |

Source: [unet_leaf_pipeline_comparison.ipynb](unet_leaf_pipeline_comparison.ipynb), notebook cell 31 (counting all cells from 1).

### 1.6 DeepLabV3 standalone recomputation: separate saved run

| Plant IoU | Plant Dice | Both-class IoU |
| --- | --- | --- |
| 0.6204 | 0.7536 | 0.7475 |

| Stage | Plant IoU | Plant Dice | Both-class IoU | Images |
| --- | --- | --- | --- | --- |
| developing | 0.6840 | 0.7936 | 0.7588 | 41 |
| flowering | 0.5906 | 0.7353 | 0.7166 | 60 |
| fruiting | 0.4904 | 0.6444 | 0.6711 | 21 |
| seeding | 0.6706 | 0.7996 | 0.8266 | 38 |

Source: [deeplabv3_pipeline_evaluation.ipynb](deeplabv3_pipeline_evaluation.ipynb), notebook cell 11 (counting all cells from 1).

This output differs from the comparison run (0.6230 / 0.7565 / 0.7507). The stored JSON below reports both-class IoU 0.747557..., which rounds to 0.7476, whereas this notebook prints 0.7475. Both records are retained; their provenance does not establish that they are a single identical execution.

### 1.7 DeepLabV3 fixed-split pixel classification metrics

| Level / class | Pixel accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Overall / macro | 0.9010 | 0.8466 | 0.8626 | 0.8542 |
| Background | N/A | 0.9450 | 0.9289 | 0.9368 |
| Plant | N/A | 0.7482 | 0.7963 | 0.7715 |

Source: [leaf_segmentation.ipynb](leaf_segmentation.ipynb), notebook cell 35 (counting all cells from 1).

No corresponding U-Net pixel-accuracy or pixel precision/recall summary is saved in the inspected comparison outputs. These values cannot be recovered from IoU and Dice alone without additional counts or predictions.

### 1.8 DeepLabV3 detailed saved JSON: validation, test, and stages

The JSON stores a pair for each metric, reproduced as value +/- stored spread below. Its generating code and spread convention were not found in the inspected notebooks, so the second number is not labeled as a verified standard deviation or confidence interval. These are separate image-level summaries, not CV results. The IoU key corresponds to foreground IoU; precision/recall/F1 appear to describe foreground scores, but their exact averaging implementation is not independently documented in this file.

| Split / test stage | Images | Plant IoU | Plant Dice | Both-class IoU |
| --- | --- | --- | --- | --- |
| Validation | 135 | 0.6433 +/- 0.1165 | 0.7764 +/- 0.0926 | 0.7823 +/- 0.0834 |
| Test | 160 | 0.6204 +/- 0.1446 | 0.7536 +/- 0.1389 | 0.7476 +/- 0.0960 |
| developing | 41 | 0.6840 +/- 0.1789 | 0.7936 +/- 0.1786 | 0.7588 +/- 0.1218 |
| flowering | 60 | 0.5906 +/- 0.1111 | 0.7353 +/- 0.1062 | 0.7166 +/- 0.0650 |
| fruiting | 21 | 0.4904 +/- 0.1307 | 0.6444 +/- 0.1580 | 0.6711 +/- 0.0770 |
| seeding | 38 | 0.6706 +/- 0.0836 | 0.7996 +/- 0.0650 | 0.8266 +/- 0.0416 |

| Split / test stage | Images | Pixel accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Validation | 135 | 0.9363 +/- 0.0640 | 0.7775 +/- 0.0991 | 0.7865 +/- 0.1121 | 0.7764 +/- 0.0926 |
| Test | 160 | 0.9007 +/- 0.0650 | 0.7528 +/- 0.1509 | 0.7798 +/- 0.1371 | 0.7536 +/- 0.1389 |
| developing | 41 | 0.8772 +/- 0.0728 | 0.7483 +/- 0.1991 | 0.9014 +/- 0.0524 | 0.7936 +/- 0.1786 |
| flowering | 60 | 0.8743 +/- 0.0353 | 0.7373 +/- 0.1235 | 0.7443 +/- 0.1090 | 0.7353 +/- 0.1062 |
| fruiting | 21 | 0.8726 +/- 0.0445 | 0.7267 +/- 0.1898 | 0.5918 +/- 0.1620 | 0.6444 +/- 0.1580 |
| seeding | 38 | 0.9833 +/- 0.0066 | 0.7963 +/- 0.0801 | 0.8085 +/- 0.0680 | 0.7996 +/- 0.0650 |

Source: [deeplabv3_report_tables.json](deeplabv3_report_tables.json).

### 1.9 DeepLabV3 saved training history and other recorded evaluations

| Epoch | Train loss | Train IoU | Validation loss | Validation IoU |
| --- | --- | --- | --- | --- |
| 01 | 0.6877 | 0.5756 | 0.3900 | 0.6587 |
| 02 | 0.4101 | 0.6813 | 0.2383 | 0.7216 |
| 03 | 0.3311 | 0.7176 | 0.2209 | 0.7550 |
| 04 | 0.2959 | 0.7318 | 0.1857 | 0.7534 |
| 05 | 0.2715 | 0.7424 | 0.1728 | 0.7759 |
| 06 | 0.2564 | 0.7485 | 0.1702 | 0.7721 |
| 07 | 0.2400 | 0.7582 | 0.1664 | 0.7844 |
| 08 | 0.2309 | 0.7622 | 0.1570 | 0.7851 |
| 09 | 0.2248 | 0.7625 | 0.1541 | 0.7875 |
| 10 | 0.2158 | 0.7704 | 0.1626 | 0.7868 |
| 11 | 0.2101 | 0.7737 | 0.1565 | 0.7932 |
| 12 | 0.2035 | 0.7776 | 0.1544 | 0.7965 |
| 13 | 0.1961 | 0.7849 | 0.1452 | 0.7940 |
| 14 | 0.1912 | 0.7905 | 0.1473 | 0.7959 |
| 15 | 0.1865 | 0.7940 | 0.1472 | 0.7970 |

This saved training run completed in 12 minutes 9 seconds with best validation IoU 0.7970. The saved test evaluation cell reports loss 0.2273 and mean IoU 0.7548. A later CV aggregation output retains a single-run reference IoU 0.7523. Historical annotations mention approximately 0.7570. These records must not be averaged or treated as one precisely reproduced run. U-Net training history and validation metrics are not saved in this workspace comparison notebook.

Source: [leaf_segmentation.ipynb](leaf_segmentation.ipynb), notebook cell 12 (counting all cells from 1).

Source: [leaf_segmentation.ipynb](leaf_segmentation.ipynb), notebook cell 18 (counting all cells from 1).

Source: [leaf_segmentation.ipynb](leaf_segmentation.ipynb), notebook cell 34 (counting all cells from 1).

The U-Net comparison output also quotes external historical values of foreground IoU 0.6650 and Dice 0.7572. The locally recomputed native-U-Net values are 0.6646 and 0.7844. The external values are references only; their original training/evaluation records were not verified here.

## 2. Complete segmentation-to-classification pipelines

Each pipeline predicts a plant mask, blacks out background pixels, applies classifier normalization, and runs the masked-image ResNet50 growth-stage classifier. Both U-Net variants reuse the same trained U-Net; crop alignment changes inference preprocessing. The U-Net path loads the masked-classifier checkpoint; the DeepLabV3 comparison path loads its exported classifier weights. Their intended classifier identity is documented, but checkpoint hashes were not compared in this report.

### 2.1 Fixed-split pipeline summaries

| Pipeline / saved run | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- | --- |
| DeepLabV3: comparison notebook | 0.8438 | 0.83 | 0.84 | 0.83 |
| U-Net: native resize | 0.7875 | 0.80 | 0.83 | 0.78 |
| U-Net: crop-aligned | 0.8938 | 0.87 | 0.88 | 0.88 |
| DeepLabV3: standalone / final summary | 0.8750 | 0.8546 | 0.8857 | 0.8612 |

The first three macro rows retain the two-decimal precision printed by the comparison notebook. The final DeepLabV3 row uses the four-decimal summary in leaf_segmentation.ipynb. The older 0.8500 DeepLabV3 accuracy is a historical annotation in evaluation outputs, not an additional verified experiment with a complete saved report.

### 2.2 DeepLabV3 comparison-run per-class results

| Class / average | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| developing | 0.97 | 0.88 | 0.92 | 41 |
| flowering | 0.81 | 0.77 | 0.79 | 60 |
| fruiting | 0.54 | 0.71 | 0.61 | 21 |
| seeding | 1.00 | 1.00 | 1.00 | 38 |
| macro avg | 0.83 | 0.84 | 0.83 | 160 |
| weighted avg | 0.86 | 0.84 | 0.85 | 160 |

Source: [unet_leaf_pipeline_comparison.ipynb](unet_leaf_pipeline_comparison.ipynb), notebook cell 25 (counting all cells from 1).

### 2.3 U-Net native per-class results

| Class / average | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| developing | 0.95 | 0.95 | 0.95 | 41 |
| flowering | 0.89 | 0.52 | 0.65 | 60 |
| fruiting | 0.40 | 0.86 | 0.55 | 21 |
| seeding | 0.97 | 1.00 | 0.99 | 38 |
| macro avg | 0.80 | 0.83 | 0.78 | 160 |
| weighted avg | 0.86 | 0.79 | 0.79 | 160 |

Source: [unet_leaf_pipeline_comparison.ipynb](unet_leaf_pipeline_comparison.ipynb), notebook cell 23 (counting all cells from 1).

### 2.4 U-Net crop-aligned per-class results

| Class / average | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| developing | 0.95 | 0.95 | 0.95 | 41 |
| flowering | 0.86 | 0.85 | 0.86 | 60 |
| fruiting | 0.68 | 0.71 | 0.70 | 21 |
| seeding | 1.00 | 1.00 | 1.00 | 38 |
| macro avg | 0.87 | 0.88 | 0.88 | 160 |
| weighted avg | 0.89 | 0.89 | 0.89 | 160 |

Source: [unet_leaf_pipeline_comparison.ipynb](unet_leaf_pipeline_comparison.ipynb), notebook cell 32 (counting all cells from 1).

### 2.5 DeepLabV3 standalone per-class results

| Class / average | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| developing | 0.95 | 1.00 | 0.98 | 41 |
| flowering | 0.92 | 0.73 | 0.81 | 60 |
| fruiting | 0.55 | 0.81 | 0.65 | 21 |
| seeding | 1.00 | 1.00 | 1.00 | 38 |
| macro avg | 0.85 | 0.89 | 0.86 | 160 |
| weighted avg | 0.90 | 0.88 | 0.88 | 160 |

Source: [deeplabv3_pipeline_evaluation.ipynb](deeplabv3_pipeline_evaluation.ipynb), notebook cell 13 (counting all cells from 1).

The comparison notebook additionally prints more precise class F1 scores:

| Stage | DeepLabV3 comparison F1 | U-Net native F1 |
| --- | --- | --- |
| developing | 0.9231 | 0.9512 |
| flowering | 0.7863 | 0.6526 |
| fruiting | 0.6122 | 0.5455 |
| seeding | 1.0000 | 0.9870 |

Source: [unet_leaf_pipeline_comparison.ipynb](unet_leaf_pipeline_comparison.ipynb), notebook cell 29 (counting all cells from 1).

The final DeepLabV3 standalone-compatible summary prints:

| Stage | Precision | Recall | F1 |
| --- | --- | --- | --- |
| developing | 0.9535 | 1.0000 | 0.9762 |
| flowering | 0.9167 | 0.7333 | 0.8148 |
| fruiting | 0.5484 | 0.8095 | 0.6538 |
| seeding | 1.0000 | 1.0000 | 1.0000 |

Source: [leaf_segmentation.ipynb](leaf_segmentation.ipynb), notebook cell 35 (counting all cells from 1).

### 2.6 Saved confusion matrices

Rows are true classes; columns are predicted classes. These three matrices were visually checked against the current PNG artifacts. Each totals 160 images. The DeepLabV3 PNG currently corresponds to 87.50%, not the 84.38% comparison run; both notebooks write the same PNG filename.

**U-Net native**

| True / predicted | Developing | Flowering | Fruiting | Seeding |
| --- | --- | --- | --- | --- |
| Developing | 39 | 2 | 0 | 0 |
| Flowering | 2 | 31 | 27 | 0 |
| Fruiting | 0 | 2 | 18 | 1 |
| Seeding | 0 | 0 | 0 | 38 |

Source: [unet_pipeline_confusion_matrix.png](unet_pipeline_confusion_matrix.png).

**U-Net crop-aligned**

| True / predicted | Developing | Flowering | Fruiting | Seeding |
| --- | --- | --- | --- | --- |
| Developing | 39 | 2 | 0 | 0 |
| Flowering | 2 | 51 | 7 | 0 |
| Fruiting | 0 | 6 | 15 | 0 |
| Seeding | 0 | 0 | 0 | 38 |

Source: [unet_crop_aligned_pipeline_confusion_matrix.png](unet_crop_aligned_pipeline_confusion_matrix.png).

**DeepLabV3 standalone**

| True / predicted | Developing | Flowering | Fruiting | Seeding |
| --- | --- | --- | --- | --- |
| Developing | 41 | 0 | 0 | 0 |
| Flowering | 2 | 44 | 14 | 0 |
| Fruiting | 0 | 4 | 17 | 0 |
| Seeding | 0 | 0 | 0 | 38 |

Source: [deeplabv3_pipeline_confusion_matrix_recomputed.png](deeplabv3_pipeline_confusion_matrix_recomputed.png).

The existing Thesis_Results_Summary.pdf also records the earlier DeepLabV3 comparison matrix below. It totals 160 images and reproduces 135/160 = 0.84375 accuracy and the saved comparison classification report; its provenance is the older PDF, not the current overwritten PNG.

| True / predicted | Developing | Flowering | Fruiting | Seeding |
| --- | --- | --- | --- | --- |
| Developing | 36 | 5 | 0 | 0 |
| Flowering | 1 | 46 | 13 | 0 |
| Fruiting | 0 | 6 | 15 | 0 |
| Seeding | 0 | 0 | 0 | 38 |

Source: [Thesis_Results_Summary.pdf](Thesis_Results_Summary.pdf).

### 2.7 Oracle-mask and original-image reference experiments

| Classifier training / evaluation | Accuracy |
| --- | --- |
| Original baseline on dataset/ | 0.9313 |
| Original-trained / original images, comparison | 0.9688 |
| Original-trained / ground-truth masks | 0.7688 |
| Masked-trained / original images | 0.3563 |
| Masked-trained / ground-truth masks | 0.9187 |

These are classifier references, not additional segmentation models. Ground-truth-mask accuracy requires test annotations. The initial baseline uses a separate dataset directory and should not automatically be treated as a matched dataset comparison. Using displayed scores, the oracle-reference gaps are 4.37 percentage points for standalone DeepLabV3 and 2.49 points for crop-aligned U-Net. These are descriptive differences between fixed-split records, not causal estimates or statistical significance tests.

Source: [masked_vs_original_comparison.csv](masked_vs_original_comparison.csv).

## 3. Cross-validated results

### 3.1 Coverage: what has and has not been completed

| Model / experiment | CV evidence in this workspace |
| --- | --- |
| DeepLabV3 segmenter only | Completed five-fold KFold; segmenter retrained; no pipeline classification scoring in this experiment |
| DeepLabV3 complete pipeline | Completed five-fold StratifiedKFold; segmenter and classifier retrained in every fold |
| U-Net native segmenter and pipeline | No completed CV outputs found |
| U-Net crop-aligned segmenter and pipeline | No completed CV outputs found |
| Original ResNet50 classifier | CV cell exists but saved execution fails: train_ds is not defined |
| Masked ResNet50 alone | No separate classifier-only CV results found |
| ResNet101 | No saved performance results found |

U-Net CV accuracy, IoU, precision, recall, F1, fold scores, and mean +/- SD are unavailable. Fixed-split U-Net scores must not be entered into a CV comparison table as if they were cross-validated.

### 3.2 CV protocol and fold sizes

Both DeepLabV3 experiments pool 979 annotated images across the original train, validation, and test folders. Segmenter-only CV uses shuffled KFold; full-pipeline CV uses shuffled StratifiedKFold by growth stage, with seed 42. Each experiment carves out 15% of each outer training partition for inner validation. Checkpoints are selected using inner validation, and final scoring uses the outer test fold. These two CV experiments use different fold assignments and training runs.

| Fold | Training | Inner validation | Outer test |
| --- | --- | --- | --- |
| 1 | 665 | 118 | 196 |
| 2 | 665 | 118 | 196 |
| 3 | 665 | 118 | 196 |
| 4 | 665 | 118 | 196 |
| 5 | 666 | 118 | 195 |

Each CV segmenter is freshly initialized from pretrained weights and trained for 10 epochs. Full-pipeline CV additionally initializes an ImageNet-pretrained ResNet50 and trains it for 12 epochs on ground-truth-masked training images; outer-fold classification uses predicted masks. The fixed-split schedules were 15 segmenter epochs and 20 classifier epochs. Thus fixed-split versus CV differences include both partitioning and training schedule.

### 3.3 DeepLabV3 segmenter-only CV: all held-out metrics

| Fold | Test IoU | Pixel accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.7565 | 0.9177 | 0.8507 | 0.8620 | 0.8562 |
| 2 | 0.7390 | 0.9119 | 0.8600 | 0.8385 | 0.8486 |
| 3 | 0.7390 | 0.9039 | 0.8507 | 0.8315 | 0.8406 |
| 4 | 0.7341 | 0.9130 | 0.8472 | 0.8317 | 0.8391 |
| 5 | 0.7645 | 0.9130 | 0.8549 | 0.8609 | 0.8578 |
| Mean +/- SD | 0.7466 +/- 0.0131 | 0.9119 +/- 0.0050 | 0.8527 +/- 0.0049 | 0.8449 +/- 0.0154 | 0.8485 +/- 0.0086 |

Source: [checkpoints_segmentation/deeplabv3_kfold_cv_results.csv](checkpoints_segmentation/deeplabv3_kfold_cv_results.csv).

### 3.4 Full-pipeline CV: segmentation stage

| Fold | Test IoU | Pixel accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.7464 | 0.9075 | 0.8412 | 0.8496 | 0.8453 |
| 2 | 0.7490 | 0.9107 | 0.8455 | 0.8482 | 0.8469 |
| 3 | 0.7502 | 0.9100 | 0.8559 | 0.8368 | 0.8459 |
| 4 | 0.7448 | 0.9128 | 0.8670 | 0.8234 | 0.8428 |
| 5 | 0.7543 | 0.9143 | 0.8625 | 0.8347 | 0.8476 |
| Mean +/- SD | 0.7489 +/- 0.0037 | 0.9111 +/- 0.0026 | 0.8544 +/- 0.0110 | 0.8385 +/- 0.0108 | 0.8457 +/- 0.0018 |

Source: [deeplabv3_pipeline_kfold_cv_results.csv](deeplabv3_pipeline_kfold_cv_results.csv).

### 3.5 Full-pipeline CV: growth-stage classification

| Fold | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- | --- |
| 1 | 0.8316 | 0.7887 | 0.8320 | 0.7916 |
| 2 | 0.8316 | 0.7628 | 0.7893 | 0.7690 |
| 3 | 0.8214 | 0.7632 | 0.7250 | 0.7188 |
| 4 | 0.7857 | 0.6765 | 0.7000 | 0.6813 |
| 5 | 0.8410 | 0.8648 | 0.7438 | 0.7433 |
| Mean +/- SD | 0.8223 +/- 0.0216 | 0.7712 +/- 0.0674 | 0.7580 +/- 0.0527 | 0.7408 +/- 0.0430 |

Source: [deeplabv3_pipeline_kfold_cv_results.csv](deeplabv3_pipeline_kfold_cv_results.csv).

### 3.6 Validation scores used for checkpoint selection

These scores are model-selection measurements and are not outer-test performance.

| Fold | Segmenter-only best validation IoU |
| --- | --- |
| 1 | 0.7442 |
| 2 | 0.7268 |
| 3 | 0.7511 |
| 4 | 0.7541 |
| 5 | 0.7516 |
| Mean +/- SD | 0.7456 +/- 0.0111 |

Source: [checkpoints_segmentation/deeplabv3_kfold_cv_results.csv](checkpoints_segmentation/deeplabv3_kfold_cv_results.csv).

| Fold | Full-CV best segmenter validation IoU | Full-CV best classifier validation accuracy |
| --- | --- | --- |
| 1 | 0.7415 | 0.9068 |
| 2 | 0.7546 | 0.9237 |
| 3 | 0.7516 | 0.9237 |
| 4 | 0.7434 | 0.9237 |
| 5 | 0.7585 | 0.8814 |
| Mean +/- SD | 0.7499 +/- 0.0073 | 0.9119 +/- 0.0185 |

Source: [deeplabv3_pipeline_kfold_cv_results.csv](deeplabv3_pipeline_kfold_cv_results.csv).

### 3.7 CV reporting limits

The saved CSVs contain fold-level aggregate metrics, not per-image predictions, per-fold confusion matrices, per-class CV reports, or CV foreground Dice. Therefore these missing results are not invented or substituted with fixed-split class scores. Means weight folds equally, and standard deviations are computed from the saved four-decimal fold values. Pixel accuracy measures segmentation pixels, whereas pipeline accuracy measures correctly classified images.

The CV is internal and image-level. The inspected implementation does not establish grouping by plant, acquisition session, camera, or farm, nor a duplicate-image audit. Since the original test partition is pooled into CV, it is not an untouched external test set for this CV study. Without matched U-Net CV runs, no cross-validated superiority claim or paired significance comparison between U-Net and DeepLabV3 is supported.

## 4. Interpretation and reconciliation of saved records

The fixed-split comparison records higher plant IoU for both U-Net variants than for DeepLabV3. However, native U-Net has lower downstream accuracy (78.75%) than the comparison DeepLabV3 pipeline (84.38%). Crop-aligned U-Net reaches 89.38%, a 10.63-point increase over native U-Net using the displayed rounded values. This associates classifier-compatible preprocessing with improved pipeline performance; the comparison does not isolate all possible causes or establish an architecture-wide advantage.

The native U-Net implementation applies its mask to the same direct-resized image. It is therefore internally aligned; its preprocessing differs from the classifier training convention. The results do not justify claiming that a mask was necessarily applied to the wrong spatial frame. Fruiting has the lowest foreground IoU among the four stages for all three comparison variants and low pipeline F1, but the available metrics alone do not prove why.

DeepLabV3 has saved fixed-split pipeline scores of 84.38% and 87.50%, plus a historical 85.00% annotation. The relevant current comparison and standalone source code is very similar, while saved outputs differ. Checkpoint version, execution state, or other causes were not established; do not attribute the discrepancy to nondeterminism without evidence. The full-pipeline CV result is separately 82.23% +/- 2.16 percentage points accuracy and 74.08% +/- 4.30 points macro F1.

Corrections relative to the older Thesis_Results_Summary.pdf: 0.7565 is DeepLabV3 foreground Dice in the comparison, not foreground IoU (which is 0.6230); segmenter-only CV uses KFold and scores only segmentation, rather than stratified pipeline evaluation with a fixed classifier; the current standalone notebook reports 87.50%, while the comparison notebook reports 84.38%; differences between CV and fixed-split runs cannot be attributed solely to a favorable test split because epoch budgets also differ. Overlapping standard deviations are not a significance test. The older PDF is retained unchanged.

## 5. Source inventory

- [leaf_segmentation.ipynb](leaf_segmentation.ipynb)

- [unet_leaf_pipeline_comparison.ipynb](unet_leaf_pipeline_comparison.ipynb)

- [deeplabv3_pipeline_evaluation.ipynb](deeplabv3_pipeline_evaluation.ipynb)

- [unet_vs_deeplabv3_comparison.csv](unet_vs_deeplabv3_comparison.csv)

- [unet_vs_deeplabv3_comparison.json](unet_vs_deeplabv3_comparison.json)

- [unet_vs_deeplabv3_comparison_v2.csv](unet_vs_deeplabv3_comparison_v2.csv)

- [unet_vs_deeplabv3_comparison_v2.json](unet_vs_deeplabv3_comparison_v2.json)

- [deeplabv3_report_tables.json](deeplabv3_report_tables.json)

- [checkpoints_segmentation/deeplabv3_kfold_cv_results.csv](checkpoints_segmentation/deeplabv3_kfold_cv_results.csv)

- [deeplabv3_pipeline_kfold_cv_results.csv](deeplabv3_pipeline_kfold_cv_results.csv)

- [masked_vs_original_comparison.csv](masked_vs_original_comparison.csv)

- [RESEARCH_REPORT_DEEPLABV3_CV.md](RESEARCH_REPORT_DEEPLABV3_CV.md)

- [Thesis_Results_Summary.pdf](Thesis_Results_Summary.pdf)

The original two-row comparison CSV/JSON duplicates the DeepLabV3 and native U-Net rows in v2; v2 adds crop-aligned U-Net. Duplicate exports are not separate experimental replications. This report prioritizes executable notebook methods and saved numerical outputs over older prose summaries.
