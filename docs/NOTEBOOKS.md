# Notebooks in this project

Six notebooks (01-05 are the main chain; 06 is a separate Colab experiment), in the order they were built. Each one is additive — later notebooks load
checkpoints produced by earlier ones but never retrain or modify them in place.

## 1. `notebooks/01_resnet50_baseline.ipynb` — baseline classifier

Fine-tunes ImageNet-pretrained ResNet50 to classify tomato plant photos into 4 growth
stages (**seeding, developing, flowering, fruiting**), using the original unmasked images
in `dataset/<class>/{train,valid,test}/`.

- **Trains:** the production classifier, saved to `checkpoints/`.
- **Exports:** `export/` — the deployable classifier.
- **Answers:** "can a plain ResNet50 tell growth stages apart at all?"

## 2. `notebooks/02_resnet50_masked_comparison.ipynb` — does the model cheat by looking at the background?

Retrains the same ResNet50 architecture, but on `tomato final dataset/` (COCO-polygon
annotated) with the background blacked out using the *ground-truth* polygon masks. Runs a
four-way comparison (masked/original model × masked/original test data) to check whether
the original model was relying on background context instead of the plant itself.

- **Trains:** the masked classifier, saved to `checkpoints_masked/resnet50_masked_best.pt`
  (91.87% oracle-mask accuracy — this is the classifier every later pipeline reuses).
- **Exports:** `export_masked/` (comparison-only, not a production replacement).
- **Answers:** "is the plant region alone enough to classify correctly?"

## 3. `notebooks/03_deeplabv3_segmentation_pipeline.ipynb` — train the segmenter + build the real deployable pipeline

Ground-truth polygon masks don't exist for brand-new, unannotated photos. This notebook
trains **DeepLabV3-ResNet50** as a binary (plant vs. background) segmenter, so masks can be
*predicted* instead of relying on annotations. It then chains that segmenter to the masked
classifier from notebook 2 (segment → mask → classify) and evaluates the realistic,
deployable pipeline (predicted masks, not oracle ones).

- **Trains:** the segmenter, saved to `checkpoints_segmentation/deeplabv3_leaf_seg_best.pt`.
- **Exports:** `export_leaf_pipeline/` — the full standalone segment→mask→classify pipeline
  (two `.pth` files + `leaf_pipeline.py` exposing `predict_growth_stage()`).
- **Answers:** "what accuracy do we actually get on a brand-new photo with no annotations?"
  (Originally reported: ~75.7% segmentation IoU, ~85% pipeline accuracy.)

## 4. `notebooks/04_unet_vs_deeplabv3_comparison.ipynb` — DeepLabV3 vs. an alternative segmenter (U-Net)

Loads a U-Net segmenter trained in a **separate** project (`d:\Reasearch\Unet`) and runs it
through the same segment→mask→classify pipeline as notebook 3, using identical metrics, so
the two segmenters are compared apples-to-apples on the same held-out test set. Also tests
a crop-aligned U-Net variant (fed DeepLabV3's own preprocessing) to remove a geometry
mismatch between the two models' native conventions.

- **Trains:** nothing — only loads existing checkpoints from both projects and re-scores them.
- **Produces:** `results/unet_vs_deeplabv3_comparison*.csv/json`, confusion-matrix PNGs.
- **Answers:** "is U-Net a better/worse segmenter than DeepLabV3 for this pipeline?"

## 5. `notebooks/05_deeplabv3_pipeline_evaluation.ipynb` — standalone DeepLabV3 re-evaluation

A trimmed-down extraction of just the DeepLabV3-specific cells from notebook 4 (all U-Net
loading/comparison code removed), so the DeepLabV3 pipeline can be re-run and re-scored on
its own without needing the separate U-Net project to be present.

- **Trains:** nothing — re-scores the checkpoints from notebooks 2 and 3.
- **Produces:** `results/deeplabv3_pipeline_confusion_matrix_recomputed.png`.
- **Answers:** the same question as notebook 3's evaluation section, in isolation.

Helper modules (`dataset_utils`, `masked_comparison_utils`, `segmentation_dataset_utils`) live in `src/`; metrics and figures are written to `results/`.

## 6. `notebooks/06_resnet101_colab_training.ipynb` - ResNet101 on U-Net-predicted-mask images

Google Colab notebook (Drive-mounted dataset) that trains a ResNet101 growth-stage classifier on images masked by a U-Net.

## Dependency chain

```
notebooks/01_resnet50_baseline.ipynb
      |
      v (background-reliance question)
notebooks/02_resnet50_masked_comparison.ipynb --> checkpoints_masked/resnet50_masked_best.pt
      |
      v (masked classifier reused)
notebooks/03_deeplabv3_segmentation_pipeline.ipynb --> checkpoints_segmentation/deeplabv3_leaf_seg_best.pt
      |                 --> export_leaf_pipeline/
      v (DeepLabV3 pipeline reused)
      +--> notebooks/04_unet_vs_deeplabv3_comparison.ipynb   (+ external U-Net project)
      +--> notebooks/05_deeplabv3_pipeline_evaluation.ipynb   (DeepLabV3-only, standalone)
```
