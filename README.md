# Tomato Growth-Stage Classification — Sequential Segment → Mask → Classify Pipeline

Research project (2026) classifying tomato plant photos into four growth stages
(**seeding, developing, flowering, fruiting**) with ResNet50, and testing whether the model
relies on the plant or on the background. The final system chains a **segmenter**
(DeepLabV3-ResNet50 or U-Net) with a **masked ResNet50 classifier**, so it works on new,
unannotated photos.

```
photo ──► segmenter (plant vs. background) ──► black out background ──► ResNet50 classifier ──► growth stage
```

## Research questions

1. Can a plain ResNet50 tell growth stages apart? → notebook 01
2. Does it "cheat" by using background context? → notebook 02 (Grad-CAM + masked/original 2×2 test)
3. Can predicted masks replace ground-truth annotations? → notebook 03
4. Is U-Net or DeepLabV3 the better segmenter for this pipeline? → notebooks 04, 05
5. How stable are the results under 5-fold cross-validation? → notebook 03 (sections 9–10) and `reports/`

## Key results

| Experiment | Result |
|---|---|
| Baseline ResNet50, original images | 93.1% test accuracy (96.9% in the comparison run) |
| Original-trained model on ground-truth-masked images | 76.9% (drops → background dependence) |
| Masked-trained model on ground-truth masks (oracle) | 91.9% |
| Masked-trained model on original images | 35.6% |
| Pipeline, DeepLabV3 segmenter (fixed split, 160 test images) | 84.4% (87.5% in standalone re-evaluation) |
| Pipeline, U-Net native / crop-aligned | 78.8% / 89.4% |
| Pipeline, DeepLabV3, 5-fold CV | 82.2% accuracy, segmentation IoU 0.749 |

Full tables, per-class reports and caveats: [reports/FULL_RESULTS_UNET_DEEPLABV3.md](reports/FULL_RESULTS_UNET_DEEPLABV3.md)
and [paper_report/PAPER_RESULTS_REPORT.md](paper_report/PAPER_RESULTS_REPORT.md).

## Repository layout

```
.
├── notebooks/        Training & evaluation notebooks, numbered in the order they were built
├── src/              Reusable dataset helper modules imported by the notebooks
├── scripts/          Dataset masking + report/figure builders
├── results/          Metrics (CSV/JSON) and figures produced by the notebooks
├── reports/          Written results (markdown/PDF) and thesis figures
├── paper_report/     Paper-ready report (md/docx/pdf), figures, and data-recompute scripts
├── docs/             Detailed per-notebook documentation (NOTEBOOKS.md)
├── export_masked/    Exported masked classifier (weights + feature extractor)
└── requirements.txt
```

Not tracked in git (too large; regenerate with the notebooks): `dataset/`,
`tomato final dataset/`, `tomato final dataset_masked/`, `checkpoints*/`, `export/`,
`export_leaf_pipeline*/`.

## Notebooks

| # | Notebook | Trains | Purpose |
|---|---|---|---|
| 01 | [01_resnet50_baseline.ipynb](notebooks/01_resnet50_baseline.ipynb) | ResNet50 classifier | Baseline growth-stage classifier |
| 02 | [02_resnet50_masked_comparison.ipynb](notebooks/02_resnet50_masked_comparison.ipynb) | Masked ResNet50 | Background-dependence test, Grad-CAM |
| 03 | [03_deeplabv3_segmentation_pipeline.ipynb](notebooks/03_deeplabv3_segmentation_pipeline.ipynb) | DeepLabV3 segmenter | Segment → mask → classify pipeline, 5-fold CV |
| 04 | [04_unet_vs_deeplabv3_comparison.ipynb](notebooks/04_unet_vs_deeplabv3_comparison.ipynb) | — | U-Net vs. DeepLabV3 on the same pipeline |
| 05 | [05_deeplabv3_pipeline_evaluation.ipynb](notebooks/05_deeplabv3_pipeline_evaluation.ipynb) | — | Standalone DeepLabV3 pipeline re-evaluation |
| 06 | [06_resnet101_colab_training.ipynb](notebooks/06_resnet101_colab_training.ipynb) | ResNet101 | Google Colab experiment on U-Net-predicted-mask images |

Dependency chain: `01 → 02 → 03 → {04, 05}`. Details: [docs/NOTEBOOKS.md](docs/NOTEBOOKS.md).

## Reproducing

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

1. Place the datasets in the repo root (`dataset/`, `tomato final dataset/`).
2. Notebooks set `PROJECT_ROOT` in their config cell — change it to your local clone path
   (it is currently `d:\Reasearch\Resnet50test`). Helper modules are loaded from `src/`.
3. Generate the masked dataset: `python scripts/generate_masked_dataset.py`.
4. Run notebooks 01 → 05 in order. Outputs are written to `results/`.
5. Notebook 04 additionally needs a U-Net checkpoint from a separate project (`d:\Reasearch\Unet`).
6. Rebuild reports: `python scripts/build_full_results_report.py`, `python scripts/build_paper_figures.py`.

## License

See [LICENSE](LICENSE).
