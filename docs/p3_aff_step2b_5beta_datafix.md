# Step 2B: 5beta subset50 Data Config Fix

Date: 2026-06-19

This step fixes the official 5beta subset50 data path for later P3AFF fixed-alpha experiments. It does not implement learnable gate, haze-aware gate, Relationship Reasoning, or formal 10e/50e/100e ablations.

## 1. Original Failure

Original YAML:

```text
datasets/VOC_hazy_5beta/VOC_hazy_subset50.yaml
```

Content:

```yaml
path: .
train: train_5beta_subset50.txt
val: images/val
test: images/test
clean: clean
```

The train list contains forward-slash relative image paths without `./`:

```text
images/train/000012_beta0.60.jpg
images/train/000012_beta1.00.jpg
...
```

Under the current direct `python ultralytics/yolo/v8/detect/train.py ...` workflow, `path: .` resolves to the project root. Then `train_5beta_subset50.txt` entries are read as plain relative paths and are not made relative to the dataset directory because `BaseDataset.get_img_files()` only expands paths that start with `./`.

Result:

```text
The loader looks for images/train/... under the project root.
It does not find datasets/VOC_hazy_5beta/images/train/...
Training fails before labels can be loaded.
```

## 2. Path and Label Checks

Original subset50 list:

```text
train_5beta_subset50.txt lines: 6250
```

Actual dataset files:

```text
images/train jpg count: 12505
labels/train txt count: 12505
images/val jpg count: 12550
labels/val txt count: 12550
```

After the fix, a direct path check found:

```text
train_lines: 6250
missing_train_images: 0
missing_train_labels: 0
val_images: 12550
missing_val_labels: 0
```

Example resolved paths:

```text
image: C:\Python\project\ultralytics-yolov8-official\datasets\VOC_hazy_5beta\images\train\000012_beta0.60.jpg
label: C:\Python\project\ultralytics-yolov8-official\datasets\VOC_hazy_5beta\labels\train\000012_beta0.60.txt
```

## 3. Final Fix

No dataloader code was changed.

Added local data YAML:

```text
datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml
```

Content:

```yaml
path: C:/Python/project/ultralytics-yolov8-official/datasets/VOC_hazy_5beta
train: train_5beta_subset50_local.txt
val: images/val
test: images/test
clean: clean
```

Added compatible train list:

```text
datasets/VOC_hazy_5beta/train_5beta_subset50_local.txt
```

It preserves the same 6250 subset50 entries but prefixes each line with `./`:

```text
./images/train/000012_beta0.60.jpg
./images/train/000012_beta1.00.jpg
...
```

Why this works:

```text
BaseDataset.get_img_files() expands ./ paths relative to the txt parent directory.
The resulting absolute Windows paths contain \images\, so img2label_paths() maps them to \labels\ correctly.
```

## 4. Files Added or Modified

Added:

```text
datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml
datasets/VOC_hazy_5beta/train_5beta_subset50_local.txt
docs/p3_aff_step2b_5beta_datafix.md
```

Modified:

```text
None for code.
```

Dataloader:

```text
Not modified.
```

## 5. alpha=0 Datacheck

Command summary:

```text
model=ultralytics/models/v8/yolov8-recovery-aff-p3-a000.yaml
data=datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml
epochs=1
batch=4
imgsz=640
workers=0
seed=0
recovery=True
recovery_fuse=none
recovery_loss_weight=0.0
recovery_debug_shapes=True
name=5beta_subset50_p3aff_a000_datacheck_1e
```

Data loading:

```text
train scan: 6250 images, 0 backgrounds, 0 corrupt
val scan: 12550 images, 0 backgrounds, 0 corrupt
```

Run status:

```text
training started: yes
validation completed: yes
results.csv generated: yes
args.yaml generated: yes
best.pt generated: yes
last.pt generated: yes
recovery_shape_debug.txt generated: yes
depends on C:\tmp YAML: no
```

Outputs:

```text
runs/detect/5beta_subset50_p3aff_a000_datacheck_1e/results.csv
runs/detect/5beta_subset50_p3aff_a000_datacheck_1e/args.yaml
runs/detect/5beta_subset50_p3aff_a000_datacheck_1e/recovery_shape_debug.txt
runs/detect/5beta_subset50_p3aff_a000_datacheck_1e/weights/best.pt
runs/detect/5beta_subset50_p3aff_a000_datacheck_1e/weights/last.pt
```

Shape debug:

```text
train: P3AFF_P3=(4, 64, 80, 80), P3AFF_R3=(4, 64, 80, 80), P3AFF_F3=(4, 64, 80, 80)
val:   P3AFF_P3=(8, 64, 20, 84), P3AFF_R3=(8, 64, 20, 84), P3AFF_F3=(8, 64, 20, 84)
```

Final results row:

```text
epoch=0
train/box_loss=3.3668
train/cls_loss=4.7019
train/dfl_loss=3.8939
train/dehaze_loss=0
train/recovery_loss=0
metrics/precision(B)=0.0027
metrics/recall(B)=0.07016
metrics/mAP50(B)=0.00222
metrics/mAP50-95(B)=0.00052
val/box_loss=3.1191
val/cls_loss=4.0971
val/dfl_loss=3.6035
```

These values are only a one-epoch datacheck and should not be used as performance conclusions.

## 6. alpha=0.05 Datacheck

Command summary:

```text
model=ultralytics/models/v8/yolov8-recovery-aff-p3-a005.yaml
data=datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml
epochs=1
batch=4
imgsz=640
workers=0
seed=0
recovery=True
recovery_fuse=none
recovery_loss_weight=0.0
recovery_debug_shapes=True
name=5beta_subset50_p3aff_a005_datacheck_1e
```

Data loading:

```text
The same fixed YAML/list was used.
Pre-run path check showed 6250 train images with 0 missing train labels and 12550 val images with 0 missing val labels.
```

Run status:

```text
training started: yes
validation completed: yes
results.csv generated: yes
args.yaml generated: yes
best.pt generated: yes
last.pt generated: yes
recovery_shape_debug.txt generated: yes
depends on C:\tmp YAML: no
```

Note:

```text
The shell command exceeded the tool wait window after the run had generated final artifacts.
The output files were checked directly afterward.
```

Outputs:

```text
runs/detect/5beta_subset50_p3aff_a005_datacheck_1e/results.csv
runs/detect/5beta_subset50_p3aff_a005_datacheck_1e/args.yaml
runs/detect/5beta_subset50_p3aff_a005_datacheck_1e/recovery_shape_debug.txt
runs/detect/5beta_subset50_p3aff_a005_datacheck_1e/weights/best.pt
runs/detect/5beta_subset50_p3aff_a005_datacheck_1e/weights/last.pt
```

Shape debug:

```text
train: P3AFF_P3=(4, 64, 80, 80), P3AFF_R3=(4, 64, 80, 80), P3AFF_F3=(4, 64, 80, 80)
val:   P3AFF_P3=(8, 64, 20, 84), P3AFF_R3=(8, 64, 20, 84), P3AFF_F3=(8, 64, 20, 84)
```

Final results row:

```text
epoch=0
train/box_loss=3.3644
train/cls_loss=4.7040
train/dfl_loss=3.9047
train/dehaze_loss=0
train/recovery_loss=0
metrics/precision(B)=0.00205
metrics/recall(B)=0.16567
metrics/mAP50(B)=0.00178
metrics/mAP50-95(B)=0.00045
val/box_loss=3.1602
val/cls_loss=5.3801
val/dfl_loss=3.6606
```

These values are only a one-epoch datacheck and should not be used as performance conclusions.

## 7. Can Move To 10e Fixed-Alpha Ablation?

Engineering status:

```text
official subset50-style data path fixed: yes
train labels found: yes
val labels found: yes
alpha=0 datacheck 1e: pass
alpha=0.05 datacheck 1e: pass
dataloader modified: no
C:\tmp smoke YAML dependency removed: yes
```

This is ready for a future 10e fixed-alpha screening after owner confirmation.

Do not start formal ablation automatically.

## 8. Data YAML For Future Formal Runs

Use:

```text
datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml
```

This is a local absolute-path YAML for this machine. For team-wide reproduction, a portable alternative would be:

```text
path: <repo>/datasets/VOC_hazy_5beta
train: train_5beta_subset50_local.txt
```

or keep `path` relative to the YAML parent only if the training entrypoint is confirmed to resolve it consistently.
