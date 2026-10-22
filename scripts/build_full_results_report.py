"""Compile saved experiment evidence without loading or rerunning models."""
from pathlib import Path
import csv
import json
import re
import statistics
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
parts = []

def add(s):
    parts.append(s.strip() + '\n')

def table(headers, rows):
    add('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' +
        '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows))

def output(name, idx):
    n = json.loads((ROOT / name).read_text(encoding='utf-8'))
    return ''.join(''.join(o.get('text', [])) for o in n['cells'][idx].get('outputs', []))

def source(name, cell=None):
    add(f'Source: [{name}](../{name})' + (f', notebook cell {cell + 1} (counting all cells from 1).' if cell is not None else '.'))

def stage_table(name, cell):
    rows = re.findall(r'(developing|flowering|fruiting|seeding)\s+fg_iou=([\d.]+)\s+fg_dice=([\d.]+)\s+both_iou=([\d.]+)\s+n=(\d+)', output(name, cell))
    assert len(rows) == 4
    table(['Stage', 'Plant IoU', 'Plant Dice', 'Both-class IoU', 'Images'], rows)
    source(name, cell)

def class_report(name, cell):
    text = output(name, cell)
    rows = re.findall(r'^\s*(developing|flowering|fruiting|seeding|macro avg|weighted avg)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$', text, re.M)
    assert len(rows) == 6
    table(['Class / average', 'Precision', 'Recall', 'F1', 'Support'], rows)
    source(name, cell)

def cv_table(path, columns, labels):
    rows = list(csv.DictReader((ROOT / path).open(encoding='utf-8')))
    table(['Fold'] + labels, [[r['fold']] + [f'{float(r[c]):.4f}' for c in columns] for r in rows] +
          [['Mean +/- SD'] + [f'{statistics.mean(float(r[c]) for r in rows):.4f} +/- {statistics.stdev(float(r[c]) for r in rows):.4f}' for c in columns]])
    source(path)

U = 'notebooks/04_unet_vs_deeplabv3_comparison.ipynb'
D = 'notebooks/05_deeplabv3_pipeline_evaluation.ipynb'
L = 'notebooks/03_deeplabv3_segmentation_pipeline.ipynb'
FULL = 'results/deeplabv3_pipeline_kfold_cv_results.csv'
SEG = 'results/deeplabv3_kfold_cv_results.csv'

add('# Full results report: U-Net and DeepLabV3\n\nSegmentation models, complete growth-stage pipelines, and cross-validation. Compiled 30 September 2026 from saved evidence in this workspace. No training or inference was rerun.')
add('This report includes all available aggregate and per-stage segmentation results, per-class pipeline reports, confusion matrices, and fold-level CV metrics found in the local result files and relevant notebooks. DeepLabV3 has completed segmentation-only and full-pipeline five-fold CV. U-Net has fixed-split results for two preprocessing variants; no U-Net CV results were found in this workspace. The external U-Net training project was not audited.')
add('All metric values are proportions from 0 to 1 unless a percent sign is shown. For example, 0.8750 means 87.50%. Saved precision is preserved; two-decimal notebook classification reports are not silently presented as four-decimal measurements. CV +/- values are sample standard deviations over five folds, not confidence intervals.')
add('The fixed annotated split has 684 training, 135 validation, and 160 test images. Test supports are developing 41, flowering 60, fruiting 21, and seeding 38. Segmentation predicts background versus plant; pipeline classification predicts these four growth stages. A per-stage segmentation score measures mask quality within images of that stage, not stage-classification accuracy.')

add('## 1. Segmentation models only')
add('### 1.1 Model and evaluation context')
table(['Model / variant', 'Saved model context', 'Evaluation geometry'], [
    ['DeepLabV3-ResNet50', 'Pretrained segmenter, fine-tuned; 41,994,308 trainable parameters in training output', 'Resize then center crop to 224 x 224'],
    ['U-Net native', 'Externally trained custom U-Net; 7,765,985 parameters; threshold 0.5', 'Direct resize to 224 x 224'],
    ['U-Net crop-aligned', 'Same U-Net checkpoint; inference preprocessing variant', 'Resize then center crop to 224 x 224']])
add('The comparison code uses Resize(int(224 * 1.14)), which is Resize(255), followed by CenterCrop(224). Some prose describes this as Resize(256); the executable code is the more precise record. Native U-Net and crop-aligned U-Net use different image geometry and matching ground-truth geometry. These are not three independently trained segmentation models.')
add('Foreground IoU and Dice score plant pixels only. Both-class IoU averages background and plant IoU. The comparison notebooks compute these per image and average over images. The training/CV notebook computes both-class IoU per batch and averages batches; these aggregations should not be treated as identical. Pixel-level macro precision/recall/F1 in the training notebook come from the accumulated background/plant confusion matrix.')

add('### 1.2 Fixed-split segmentation comparison: 160 test images')
comparison = json.loads((ROOT / 'results/unet_vs_deeplabv3_comparison_v2.json').read_text())
table(['Segmenter / preprocessing', 'Plant IoU', 'Plant Dice', 'Both-class IoU'], [[r['segmenter']] + [f'{r[k]:.4f}' for k in ['seg_foreground_iou','seg_foreground_dice','seg_both_class_mean_iou']] for r in comparison])
source('results/unet_vs_deeplabv3_comparison_v2.csv')
for title, cell in [('DeepLabV3: comparison run',16), ('U-Net: native resize',14), ('U-Net: crop-aligned',30)]:
    add('### 1.3 ' + title if cell == 16 else '### ' + ('1.4 ' if cell == 14 else '1.5 ') + title)
    stage_table(U, cell)

add('### 1.6 DeepLabV3 standalone recomputation: separate saved run')
table(['Plant IoU', 'Plant Dice', 'Both-class IoU'], [['0.6204','0.7536','0.7475']])
stage_table(D,10)
add('This output differs from the comparison run (0.6230 / 0.7565 / 0.7507). The stored JSON below reports both-class IoU 0.747557..., which rounds to 0.7476, whereas this notebook prints 0.7475. Both records are retained; their provenance does not establish that they are a single identical execution.')

add('### 1.7 DeepLabV3 fixed-split pixel classification metrics')
table(['Level / class', 'Pixel accuracy', 'Precision', 'Recall', 'F1'], [
    ['Overall / macro', '0.9010','0.8466','0.8626','0.8542'],
    ['Background','N/A','0.9450','0.9289','0.9368'],
    ['Plant','N/A','0.7482','0.7963','0.7715']])
source(L,34)
add('No corresponding U-Net pixel-accuracy or pixel precision/recall summary is saved in the inspected comparison outputs. These values cannot be recovered from IoU and Dice alone without additional counts or predictions.')

add('### 1.8 DeepLabV3 detailed saved JSON: validation, test, and stages')
add('The JSON stores a pair for each metric, reproduced as value +/- stored spread below. Its generating code and spread convention were not found in the inspected notebooks, so the second number is not labeled as a verified standard deviation or confidence interval. These are separate image-level summaries, not CV results. The IoU key corresponds to foreground IoU; precision/recall/F1 appear to describe foreground scores, but their exact averaging implementation is not independently documented in this file.')
j = json.loads((ROOT / 'results/deeplabv3_report_tables.json').read_text())
groups = [('Validation',j['validation']),('Test',j['test'])] + list(j['per_stage_test'].items())
for keys,labels in [(['iou','dice','both_iou'],['Plant IoU','Plant Dice','Both-class IoU']), (['pixel_acc','precision','recall','f1'],['Pixel accuracy','Precision','Recall','F1'])]:
    table(['Split / test stage','Images']+labels, [[name,str(v['n'])] + [f'{v[k][0]:.4f} +/- {v[k][1]:.4f}' for k in keys] for name,v in groups])
source('results/deeplabv3_report_tables.json')

add('### 1.9 DeepLabV3 saved training history and other recorded evaluations')
epochs = re.findall(r'Epoch (\d+)/15 \| train_loss ([\d.]+) train_iou ([\d.]+) \| val_loss ([\d.]+) val_iou ([\d.]+)',output(L,11))
assert len(epochs)==15
table(['Epoch','Train loss','Train IoU','Validation loss','Validation IoU'],epochs)
add('This saved training run completed in 12 minutes 9 seconds with best validation IoU 0.7970. The saved test evaluation cell reports loss 0.2273 and mean IoU 0.7548. A later CV aggregation output retains a single-run reference IoU 0.7523. Historical annotations mention approximately 0.7570. These records must not be averaged or treated as one precisely reproduced run. U-Net training history and validation metrics are not saved in this workspace comparison notebook.')
source(L,11)
source(L,17)
source(L,33)
add('The U-Net comparison output also quotes external historical values of foreground IoU 0.6650 and Dice 0.7572. The locally recomputed native-U-Net values are 0.6646 and 0.7844. The external values are references only; their original training/evaluation records were not verified here.')

add('## 2. Complete segmentation-to-classification pipelines')
add('Each pipeline predicts a plant mask, blacks out background pixels, applies classifier normalization, and runs the masked-image ResNet50 growth-stage classifier. Both U-Net variants reuse the same trained U-Net; crop alignment changes inference preprocessing. The U-Net path loads the masked-classifier checkpoint; the DeepLabV3 comparison path loads its exported classifier weights. Their intended classifier identity is documented, but checkpoint hashes were not compared in this report.')
add('### 2.1 Fixed-split pipeline summaries')
table(['Pipeline / saved run','Accuracy','Macro precision','Macro recall','Macro F1'],[
    ['DeepLabV3: comparison notebook','0.8438','0.83','0.84','0.83'],
    ['U-Net: native resize','0.7875','0.80','0.83','0.78'],
    ['U-Net: crop-aligned','0.8938','0.87','0.88','0.88'],
    ['DeepLabV3: standalone / final summary','0.8750','0.8546','0.8857','0.8612']])
add('The first three macro rows retain the two-decimal precision printed by the comparison notebook. The final DeepLabV3 row uses the four-decimal summary in leaf_segmentation.ipynb. The older 0.8500 DeepLabV3 accuracy is a historical annotation in evaluation outputs, not an additional verified experiment with a complete saved report.')
for title, f, cell in [('2.2 DeepLabV3 comparison-run per-class results',U,24),('2.3 U-Net native per-class results',U,22),('2.4 U-Net crop-aligned per-class results',U,31),('2.5 DeepLabV3 standalone per-class results',D,12)]:
    add('### '+title)
    class_report(f,cell)
add('The comparison notebook additionally prints more precise class F1 scores:')
table(['Stage','DeepLabV3 comparison F1','U-Net native F1'],[['developing','0.9231','0.9512'],['flowering','0.7863','0.6526'],['fruiting','0.6122','0.5455'],['seeding','1.0000','0.9870']])
source(U,28)
add('The final DeepLabV3 standalone-compatible summary prints:')
table(['Stage','Precision','Recall','F1'],[['developing','0.9535','1.0000','0.9762'],['flowering','0.9167','0.7333','0.8148'],['fruiting','0.5484','0.8095','0.6538'],['seeding','1.0000','1.0000','1.0000']])
source(L,34)

add('### 2.6 Saved confusion matrices')
add('Rows are true classes; columns are predicted classes. These three matrices were visually checked against the current PNG artifacts. Each totals 160 images. The DeepLabV3 PNG currently corresponds to 87.50%, not the 84.38% comparison run; both notebooks write the same PNG filename.')
for title, fname, matrix in [
    ('U-Net native','results/unet_pipeline_confusion_matrix.png',[[39,2,0,0],[2,31,27,0],[0,2,18,1],[0,0,0,38]]),
    ('U-Net crop-aligned','results/unet_crop_aligned_pipeline_confusion_matrix.png',[[39,2,0,0],[2,51,7,0],[0,6,15,0],[0,0,0,38]]),
    ('DeepLabV3 standalone','results/deeplabv3_pipeline_confusion_matrix_recomputed.png',[[41,0,0,0],[2,44,14,0],[0,4,17,0],[0,0,0,38]])]:
    add('**'+title+'**')
    table(['True / predicted','Developing','Flowering','Fruiting','Seeding'],[[c]+r for c,r in zip(['Developing','Flowering','Fruiting','Seeding'],matrix)])
    source(fname)
add('The existing Thesis_Results_Summary.pdf also records the earlier DeepLabV3 comparison matrix below. It totals 160 images and reproduces 135/160 = 0.84375 accuracy and the saved comparison classification report; its provenance is the older PDF, not the current overwritten PNG.')
table(['True / predicted','Developing','Flowering','Fruiting','Seeding'],[['Developing',36,5,0,0],['Flowering',1,46,13,0],['Fruiting',0,6,15,0],['Seeding',0,0,0,38]])
source('reports/Thesis_Results_Summary.pdf')

add('### 2.7 Oracle-mask and original-image reference experiments')
table(['Classifier training / evaluation','Accuracy'],[['Original baseline on dataset/','0.9313'],['Original-trained / original images, comparison','0.9688'],['Original-trained / ground-truth masks','0.7688'],['Masked-trained / original images','0.3563'],['Masked-trained / ground-truth masks','0.9187']])
add('These are classifier references, not additional segmentation models. Ground-truth-mask accuracy requires test annotations. The initial baseline uses a separate dataset directory and should not automatically be treated as a matched dataset comparison. Using displayed scores, the oracle-reference gaps are 4.37 percentage points for standalone DeepLabV3 and 2.49 points for crop-aligned U-Net. These are descriptive differences between fixed-split records, not causal estimates or statistical significance tests.')
source('results/masked_vs_original_comparison.csv')

add('## 3. Cross-validated results')
add('### 3.1 Coverage: what has and has not been completed')
table(['Model / experiment','CV evidence in this workspace'],[['DeepLabV3 segmenter only','Completed five-fold KFold; segmenter retrained; no pipeline classification scoring in this experiment'],['DeepLabV3 complete pipeline','Completed five-fold StratifiedKFold; segmenter and classifier retrained in every fold'],['U-Net native segmenter and pipeline','No completed CV outputs found'],['U-Net crop-aligned segmenter and pipeline','No completed CV outputs found'],['Original ResNet50 classifier','CV cell exists but saved execution fails: train_ds is not defined'],['Masked ResNet50 alone','No separate classifier-only CV results found'],['ResNet101','No saved performance results found']])
add('U-Net CV accuracy, IoU, precision, recall, F1, fold scores, and mean +/- SD are unavailable. Fixed-split U-Net scores must not be entered into a CV comparison table as if they were cross-validated.')
add('### 3.2 CV protocol and fold sizes')
add('Both DeepLabV3 experiments pool 979 annotated images across the original train, validation, and test folders. Segmenter-only CV uses shuffled KFold; full-pipeline CV uses shuffled StratifiedKFold by growth stage, with seed 42. Each experiment carves out 15% of each outer training partition for inner validation. Checkpoints are selected using inner validation, and final scoring uses the outer test fold. These two CV experiments use different fold assignments and training runs.')
table(['Fold','Training','Inner validation','Outer test'],[[str(i),'665' if i<5 else '666','118','196' if i<5 else '195'] for i in range(1,6)])
add('Each CV segmenter is freshly initialized from pretrained weights and trained for 10 epochs. Full-pipeline CV additionally initializes an ImageNet-pretrained ResNet50 and trains it for 12 epochs on ground-truth-masked training images; outer-fold classification uses predicted masks. The fixed-split schedules were 15 segmenter epochs and 20 classifier epochs. Thus fixed-split versus CV differences include both partitioning and training schedule.')

add('### 3.3 DeepLabV3 segmenter-only CV: all held-out metrics')
cv_table(SEG,['test_iou','test_accuracy','test_precision_macro','test_recall_macro','test_f1_macro'],['Test IoU','Pixel accuracy','Macro precision','Macro recall','Macro F1'])
add('### 3.4 Full-pipeline CV: segmentation stage')
cv_table(FULL,['seg_test_iou','seg_test_accuracy','seg_test_precision_macro','seg_test_recall_macro','seg_test_f1_macro'],['Test IoU','Pixel accuracy','Macro precision','Macro recall','Macro F1'])
add('### 3.5 Full-pipeline CV: growth-stage classification')
cv_table(FULL,['pipeline_test_accuracy','pipeline_test_precision_macro','pipeline_test_recall_macro','pipeline_test_f1_macro'],['Accuracy','Macro precision','Macro recall','Macro F1'])
add('### 3.6 Validation scores used for checkpoint selection')
add('These scores are model-selection measurements and are not outer-test performance.')
cv_table(SEG,['best_val_iou'],['Segmenter-only best validation IoU'])
cv_table(FULL,['seg_val_iou_best','clf_val_acc_best'],['Full-CV best segmenter validation IoU','Full-CV best classifier validation accuracy'])
add('### 3.7 CV reporting limits')
add('The saved CSVs contain fold-level aggregate metrics, not per-image predictions, per-fold confusion matrices, per-class CV reports, or CV foreground Dice. Therefore these missing results are not invented or substituted with fixed-split class scores. Means weight folds equally, and standard deviations are computed from the saved four-decimal fold values. Pixel accuracy measures segmentation pixels, whereas pipeline accuracy measures correctly classified images.')
add('The CV is internal and image-level. The inspected implementation does not establish grouping by plant, acquisition session, camera, or farm, nor a duplicate-image audit. Since the original test partition is pooled into CV, it is not an untouched external test set for this CV study. Without matched U-Net CV runs, no cross-validated superiority claim or paired significance comparison between U-Net and DeepLabV3 is supported.')

add('## 4. Interpretation and reconciliation of saved records')
add('The fixed-split comparison records higher plant IoU for both U-Net variants than for DeepLabV3. However, native U-Net has lower downstream accuracy (78.75%) than the comparison DeepLabV3 pipeline (84.38%). Crop-aligned U-Net reaches 89.38%, a 10.63-point increase over native U-Net using the displayed rounded values. This associates classifier-compatible preprocessing with improved pipeline performance; the comparison does not isolate all possible causes or establish an architecture-wide advantage.')
add('The native U-Net implementation applies its mask to the same direct-resized image. It is therefore internally aligned; its preprocessing differs from the classifier training convention. The results do not justify claiming that a mask was necessarily applied to the wrong spatial frame. Fruiting has the lowest foreground IoU among the four stages for all three comparison variants and low pipeline F1, but the available metrics alone do not prove why.')
add('DeepLabV3 has saved fixed-split pipeline scores of 84.38% and 87.50%, plus a historical 85.00% annotation. The relevant current comparison and standalone source code is very similar, while saved outputs differ. Checkpoint version, execution state, or other causes were not established; do not attribute the discrepancy to nondeterminism without evidence. The full-pipeline CV result is separately 82.23% +/- 2.16 percentage points accuracy and 74.08% +/- 4.30 points macro F1.')
add('Corrections relative to the older Thesis_Results_Summary.pdf: 0.7565 is DeepLabV3 foreground Dice in the comparison, not foreground IoU (which is 0.6230); segmenter-only CV uses KFold and scores only segmentation, rather than stratified pipeline evaluation with a fixed classifier; the current standalone notebook reports 87.50%, while the comparison notebook reports 84.38%; differences between CV and fixed-split runs cannot be attributed solely to a favorable test split because epoch budgets also differ. Overlapping standard deviations are not a significance test. The older PDF is retained unchanged.')

add('## 5. Source inventory')
for name in [L,U,D,'results/unet_vs_deeplabv3_comparison.csv','results/unet_vs_deeplabv3_comparison.json','results/unet_vs_deeplabv3_comparison_v2.csv','results/unet_vs_deeplabv3_comparison_v2.json','results/deeplabv3_report_tables.json',SEG,FULL,'results/masked_vs_original_comparison.csv','reports/RESEARCH_REPORT_DEEPLABV3_CV.md','reports/Thesis_Results_Summary.pdf']:
    add('- ['+name+'](../'+name+')')
add('The original two-row comparison CSV/JSON duplicates the DeepLabV3 and native U-Net rows in v2; v2 adds crop-aligned U-Net. Duplicate exports are not separate experimental replications. This report prioritizes executable notebook methods and saved numerical outputs over older prose summaries.')

md = '\n'.join(parts)
path = ROOT / 'reports' / 'FULL_RESULTS_UNET_DEEPLABV3.md'
path.write_text(md, encoding='utf-8')

# Produce a printable companion without additional dependencies or network access.
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportBody',fontName='Helvetica',fontSize=9,leading=13,spaceAfter=7))
styles.add(ParagraphStyle(name='Cell',fontName='Helvetica',fontSize=7,leading=9,wordWrap='CJK'))
styles.add(ParagraphStyle(name='CellHeader',fontName='Helvetica-Bold',fontSize=7,leading=9,textColor=colors.white))

def markup(s):
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', s)
    s = escape(s)
    s = re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    return s

story=[]
lines=md.splitlines()
i=0
width=A4[0]-72
while i<len(lines):
    line=lines[i]
    if line.startswith('| '):
        data=[]
        while i<len(lines) and lines[i].startswith('| '):
            vals=[s.strip() for s in lines[i].strip('|').split('|')]
            if not all(v=='---' for v in vals):data.append(vals)
            i+=1
        n=len(data[0]); first=width*(0.29 if n>3 else 0.46)
        widths=[first]+[(width-first)/(n-1)]*(n-1)
        tab=Table([[Paragraph(markup(v),styles['CellHeader' if ri==0 else 'Cell']) for v in row] for ri,row in enumerate(data)],colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#183e57')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5f8'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),0.5,colors.white)]))
        story.extend([tab,Spacer(1,8)])
        continue
    if line.startswith('# '):style='Title';line=line[2:]
    elif line.startswith('## '):style='Heading1';line=line[3:]
    elif line.startswith('### '):style='Heading2';line=line[4:]
    else:style='ReportBody'
    if line:story.append(Paragraph(markup(line),styles[style]))
    i+=1

def footer(canvas,doc):
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#52616b'))
    canvas.drawString(36,22,'U-Net and DeepLabV3 | Saved results | 30 September 2026')
    canvas.drawRightString(A4[0]-36,22,str(doc.page))

pdf=ROOT/'reports'/'FULL_RESULTS_UNET_DEEPLABV3.pdf'
SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=38,title='Full Results: U-Net and DeepLabV3',author='').build(story,onFirstPage=footer,onLaterPages=footer)
print(f'Created {path.name} ({len(md.splitlines())} lines) and {pdf.name}')
