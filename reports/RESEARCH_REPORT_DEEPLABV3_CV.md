# Tomato growth-stage classification with plant segmentation: research report

This report summarizes results recorded in this repository. It is a draft for a paper's Methods, Results, and Discussion sections. The numerical results are from saved notebook outputs and CSV files; no models were retrained for this report. Four growth-stage labels are used: developing, flowering, fruiting, and seeding.

## Study aim and model sequence

The study examined whether removing image background changes tomato growth-stage classification and whether a predicted plant mask can support inference on unannotated photos. The initial model was an ImageNet-pretrained ResNet50 trained on original images. A second ResNet50 was trained on images blacked out outside COCO polygon plant annotations. The deployable sequence was an ImageNet/COCO-pretrained DeepLabV3-ResNet50 binary segmenter followed by mask application and the masked-image ResNet50 classifier. At inference, the segmenter predicts plant/background pixels; the original image is multiplied by the plant mask, ImageNet-normalized, and classified into one of four stages.

The initial original-image ResNet50 used `dataset/`. The masked classifier, segmentation model, and full-pipeline CV used `tomato final dataset/` and its ground-truth-masked copy. Therefore, results from the first notebook should not be presented as a strictly matched head-to-head comparison with later pipeline results unless dataset identity is established.

## Data and training methods

The full-pipeline cross-validation pooled 979 annotated images from the original train, validation, and test folders. Images without usable COCO annotations were excluded. Five stratified outer folds were generated with shuffle and random seed 42. Each outer training partition contained 783 or 784 images; 15% of it was separated into an inner validation set, giving 665/118/196 training/validation/test images for folds 1–4 and 666/118/195 for fold 5. Outer-fold test images were used only for that fold's scoring. Both models were initialized and trained anew for every fold, with validation-selected weights. The segmenter ran for 10 epochs per fold and the classifier for 12 epochs per fold. These CV schedules differ from the fixed-split runs (15 and 20 epochs, respectively).

DeepLabV3 used a ResNet50 backbone with a two-output segmentation head for background and plant, 224 × 224 input, batch size 16, cross-entropy loss, Adam learning rate 1 × 10^-4, and a ReduceLROnPlateau scheduler. Training included an auxiliary-head loss weighted 0.4. The classifier used ImageNet-pretrained ResNet50 with a four-output fully connected head, 224 × 224 input, batch size 32, cross-entropy loss, Adam learning rate 1 × 10^-4, and the same scheduler type. Its backbone was fine-tuned. Training augmentation included random resized crop, horizontal flip, rotation up to 15°, and brightness/contrast/saturation jitter; evaluation used resize followed by center crop. Both networks used ImageNet normalization.

The classifier's training and inner-validation images used *ground-truth* polygon masks. Full-pipeline test images used *predicted* masks. This difference reflects the intended unannotated-photo inference route, but also introduces a train–test mask-quality shift.

Segmentation IoU was computed per batch as the mean IoU of background and plant, then averaged equally across batches. Segmentation accuracy, precision, recall, and F1 came from a pixel confusion matrix; precision, recall, and F1 were macro averages over the two pixel classes. Pipeline accuracy was the fraction of test images assigned the correct growth stage; pipeline precision, recall, and F1 were macro averages over the four stages. Values below are mean ± **sample standard deviation across five folds**, not confidence intervals. Each fold was weighted equally in the reported mean.

## Complete full-pipeline five-fold cross-validation results

| Fold | Train | Inner val. | Test | Best seg. val. IoU | Test seg. IoU | Test seg. pixel accuracy | Test seg. macro precision | Test seg. macro recall | Test seg. macro F1 | Best classifier val. accuracy | Pipeline test accuracy | Pipeline macro precision | Pipeline macro recall | Pipeline macro F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 665 | 118 | 196 | 0.7415 | 0.7464 | 0.9075 | 0.8412 | 0.8496 | 0.8453 | 0.9068 | 0.8316 | 0.7887 | 0.8320 | 0.7916 |
| 2 | 665 | 118 | 196 | 0.7546 | 0.7490 | 0.9107 | 0.8455 | 0.8482 | 0.8469 | 0.9237 | 0.8316 | 0.7628 | 0.7893 | 0.7690 |
| 3 | 665 | 118 | 196 | 0.7516 | 0.7502 | 0.9100 | 0.8559 | 0.8368 | 0.8459 | 0.9237 | 0.8214 | 0.7632 | 0.7250 | 0.7188 |
| 4 | 665 | 118 | 196 | 0.7434 | 0.7448 | 0.9128 | 0.8670 | 0.8234 | 0.8428 | 0.9237 | 0.7857 | 0.6765 | 0.7000 | 0.6813 |
| 5 | 666 | 118 | 195 | 0.7585 | 0.7543 | 0.9143 | 0.8625 | 0.8347 | 0.8476 | 0.8814 | 0.8410 | 0.8648 | 0.7438 | 0.7433 |
| Mean ± SD | — | — | — | 0.7499 ± 0.0073 | **0.7489 ± 0.0037** | **0.9111 ± 0.0026** | 0.8544 ± 0.0110 | 0.8385 ± 0.0108 | 0.8457 ± 0.0018 | 0.9119 ± 0.0185 | **0.8223 ± 0.0216** | **0.7712 ± 0.0674** | **0.7580 ± 0.0527** | **0.7408 ± 0.0430** |

Source: [`deeplabv3_pipeline_kfold_cv_results.csv`](deeplabv3_pipeline_kfold_cv_results.csv), produced by Section 10 of [`leaf_segmentation.ipynb`](leaf_segmentation.ipynb). Best validation metrics are model-selection measurements, not held-out performance.

## Separate DeepLabV3 segmenter-only five-fold CV

This experiment retrained only the segmenter. It used non-stratified KFold on the pooled annotated image set, with 10 training epochs and a 15% inner validation split. Because its fold assignments and training runs differ, its numbers should be reported separately from the segmentation half of the full-pipeline CV.

| Fold | Train | Inner val. | Test | Best val. IoU | Test IoU | Pixel accuracy | Macro precision | Macro recall | Macro F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 665 | 118 | 196 | 0.7442 | 0.7565 | 0.9177 | 0.8507 | 0.8620 | 0.8562 |
| 2 | 665 | 118 | 196 | 0.7268 | 0.7390 | 0.9119 | 0.8600 | 0.8385 | 0.8486 |
| 3 | 665 | 118 | 196 | 0.7511 | 0.7390 | 0.9039 | 0.8507 | 0.8315 | 0.8406 |
| 4 | 665 | 118 | 196 | 0.7541 | 0.7341 | 0.9130 | 0.8472 | 0.8317 | 0.8391 |
| 5 | 666 | 118 | 195 | 0.7516 | 0.7645 | 0.9130 | 0.8549 | 0.8609 | 0.8578 |
| Mean ± SD | — | — | — | 0.7456 ± 0.0111 | **0.7466 ± 0.0131** | **0.9119 ± 0.0050** | 0.8527 ± 0.0049 | 0.8449 ± 0.0154 | 0.8485 ± 0.0086 |

Source: [`checkpoints_segmentation/deeplabv3_kfold_cv_results.csv`](checkpoints_segmentation/deeplabv3_kfold_cv_results.csv), produced by Section 9 of [`leaf_segmentation.ipynb`](leaf_segmentation.ipynb).

## Fixed-split experiments and model comparisons

| Experiment | Recorded result | Interpretation |
|---|---:|---|
| Original-image ResNet50 on `dataset/`, fixed 160-image test | 0.9313 accuracy | Initial classification baseline; separate data directory. |
| Original-image model on original images in masked comparison | 0.9688 accuracy | Matched comparison on `tomato final dataset/`. |
| Original-image model on ground-truth-masked images | 0.7688 accuracy | Removing background reduced performance for this model. |
| Masked-image ResNet50 on original images | 0.3563 accuracy | Strong train–test preprocessing mismatch. |
| Masked-image ResNet50 on ground-truth-masked images | 0.9187 accuracy | Oracle-mask reference; requires annotations at test time. |
| DeepLabV3 fixed-split segmenter | 0.7548 mean IoU in saved notebook evaluation | Batch-averaged, both-class metric. |
| DeepLabV3 → predicted mask → masked ResNet50 | 0.8750 accuracy in saved standalone recomputation (160 images) | Single fixed-split checkpoint evaluation, not CV. Earlier saved runs report approximately 0.8438–0.8500; use 0.8750 only with the exact recomputation protocol identified. |
| U-Net native preprocessing → masked ResNet50 | 0.7875 accuracy | External U-Net checkpoint; fixed-split comparison. |
| U-Net crop-aligned preprocessing → masked ResNet50 | 0.8938 accuracy | Fixed-split variant; no matched full-pipeline CV in this repository. |

The U-Net comparison reported higher both-class segmentation IoU for U-Net (0.7926) than DeepLabV3 (0.7507) on that run, while native U-Net had lower downstream pipeline accuracy (0.7875 versus 0.8438). Aligning preprocessing increased U-Net pipeline accuracy to 0.8938. These findings suggest that input geometry and classifier compatibility matter alongside segmentation quality. The comparison reused separately trained checkpoints and does not establish a cross-validated ranking between U-Net and DeepLabV3.

For the standalone DeepLabV3 recomputation, the saved test classification report gives 0.8750 accuracy and macro F1 0.8612. Class F1 values were developing 0.9762, flowering 0.8148, fruiting 0.6538, and seeding 1.0000. The lower fruiting F1 indicates a remaining stage-specific error pattern in that fixed-split run. These class scores are not available per fold in the saved CV CSV and must not be described as CV class-wise results.

## Suggested paper-ready Results text

“In five-fold stratified cross-validation across 979 annotated tomato images, the DeepLabV3-ResNet50 segmenter and masked-image ResNet50 classifier were retrained within each fold and evaluated sequentially on the held-out fold. The segmentation stage achieved a mean batch-averaged background/plant IoU of 0.7489 ± 0.0037 and pixel accuracy of 0.9111 ± 0.0026. The complete predicted-mask growth-stage pipeline achieved accuracy of 0.8223 ± 0.0216 and macro F1 of 0.7408 ± 0.0430 across the five folds. A separate segmenter-only five-fold experiment yielded mean IoU of 0.7466 ± 0.0131. In a fixed-split oracle-mask comparison, the masked-image classifier scored 0.9187 accuracy on ground-truth-masked test images, indicating that downstream classification remained feasible when plant pixels were isolated. The lower full-pipeline CV accuracy reflects performance with predicted masks and fold-specific retraining rather than ground-truth masks.”

## Interpretation and reporting limits

The CV results support a completed **internal, image-level, five-fold evaluation of the full two-model pipeline**. They do not establish generalization to new farms, cameras, seasons, plants, or collection sessions. The notebook records do not show group-aware splitting by plant or acquisition session, nor a duplicate-image audit. If multiple photographs of the same plant or near-duplicate images exist, image-level folds could overestimate independence. The original 160-image test partition was pooled into CV, so it cannot also be called an untouched external test set for the CV study.

The CV classifier was trained using ground-truth masks and tested downstream using predicted masks. This is a valid description of the implemented pipeline, but the difference in mask quality is an important limitation. A classifier trained on out-of-fold predicted masks would be a separate experiment and has not been demonstrated here. The fixed-split 0.8750 and five-fold mean 0.8223 differ in dataset partition, training schedule, and model selection; they should not be treated as repeated measurements of the same protocol. Fixed-split DeepLabV3 pipeline accuracies also differ across saved notebook reruns; identify which checkpoint/evaluation protocol supplied any quoted fixed-split score.

The saved CV file contains per-fold summary metrics, not per-image predictions, fold-level confusion matrices, or per-class CV metrics. It supports mean and standard deviation reporting but does not support a formal paired significance test against U-Net, an external-validation claim, or confident class-specific CV conclusions. Hardware/software version details and a dataset provenance statement should be added from experiment records before manuscript submission. The separate ResNet101 Colab notebook contains code but no saved results in this repository; no ResNet101 performance claim is warranted.

## Reproducibility record

Primary implementation: [`leaf_segmentation.ipynb`](leaf_segmentation.ipynb), [`segmentation_dataset_utils.py`](segmentation_dataset_utils.py), and [`masked_comparison_utils.py`](masked_comparison_utils.py). Full-pipeline CV: [`deeplabv3_pipeline_kfold_cv_results.csv`](deeplabv3_pipeline_kfold_cv_results.csv). Segmenter-only CV: [`checkpoints_segmentation/deeplabv3_kfold_cv_results.csv`](checkpoints_segmentation/deeplabv3_kfold_cv_results.csv). Fixed-split comparison: [`masked_vs_original_comparison.csv`](masked_vs_original_comparison.csv). U-Net comparison: [`unet_vs_deeplabv3_comparison_v2.csv`](unet_vs_deeplabv3_comparison_v2.csv). Deployable pipeline: [`export_leaf_pipeline/leaf_pipeline.py`](export_leaf_pipeline/leaf_pipeline.py).
