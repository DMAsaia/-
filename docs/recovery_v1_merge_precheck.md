# RecoveryBranch v1 Merge Precheck

Date: 2026-06-19

This document records the safe merge check for the teammate RecoveryBranch v1 handoff package at:

```text
C:\Users\卢治廷\Downloads\recovery_v1_handoff
```

The current project already had Step 1 P3AFF identity integration. This merge did not overwrite the existing P3AFF code, YAML, or documents.

## 1. Handoff Package Contents

Source and config files in the package:

```text
README_recovery_v1_handoff.md
tools/visualize_recovery_outputs.py
ultralytics/models/v8/yolov8n-recovery-v1.yaml
ultralytics/nn/recovery.py
ultralytics/nn/tasks.py
ultralytics/yolo/cfg/default.yaml
ultralytics/yolo/data/base.py
ultralytics/yolo/data/utils.py
ultralytics/yolo/utils/torch_utils.py
ultralytics/yolo/v8/detect/train.py
```

Run artifacts in the package:

```text
runs/dataset_check/recovery_smoke/hard_condition_report.md
runs/dataset_check/recovery_smoke/VOC_hazy_5beta_smoke.yaml
runs/detect/recovery_v1_aux_only_1e/
runs/detect/recovery_v1_p3_fixed_1e/
runs/recovery_vis/recovery_v1/
```

The package also contains `best.pt` and `last.pt` under the two teammate run directories. These large weights were not copied into the project.

The handoff prompt mentioned `recovery_v1_teammate_note.md`, but that file was not present in the extracted package.

## 2. Same-Name File Differences

High-level diff stats against the current project:

```text
ultralytics/nn/tasks.py: about 207 insertions / 50 deletions in teammate version
ultralytics/yolo/v8/detect/train.py: about 342 insertions / 78 deletions in teammate version
ultralytics/yolo/cfg/default.yaml: about 30 insertions / 7 deletions in teammate version
```

Important differences:

```text
tasks.py:
  teammate version adds RecoveryBranch mounting in DetectionModel.
  teammate version also imports AODNet from ultralytics.nn.aod, but this project has no ultralytics/nn/aod.py.
  teammate version does not include this project's P3AFF import and parse_model branch.

train.py:
  teammate version adds CLI entrypoint support for direct python train.py key=value usage.
  teammate version adds recovery clean L1 loss.
  teammate version contains broader AOD helper code that is not needed for this merge.
  current project already has dehaze auxiliary loss and dehaze_loss reporting.

default.yaml:
  teammate version adds recovery parameters.
  current project already has dehaze parameters.

data/base.py, data/utils.py, torch_utils.py:
  present in the package, but not part of the allowed source merge for this step.
```

## 3. Changes Merged

Added:

```text
ultralytics/nn/recovery.py
ultralytics/models/v8/yolov8n-recovery-v1.yaml
```

Modified:

```text
ultralytics/nn/tasks.py
ultralytics/yolo/v8/detect/train.py
ultralytics/yolo/cfg/default.yaml
```

Merged behavior:

```text
RecoveryBranch input: x [B, 3, H, W]
RecoveryBranch output:
  dehaze_img [B, 3, H, W]
  r3 [B, 64, H/8, W/8]

recovery_fuse=none:
  Detect receives original [P3, P4, P5].

recovery_fuse=p3_fixed:
  Detect receives [P3 + recovery_alpha * Conv1x1(R3), P4, P5].
```

## 4. Changes Not Merged

Not merged:

```text
teammate runs/ weights and visualizations
tools/visualize_recovery_outputs.py
ultralytics/yolo/data/base.py
ultralytics/yolo/data/utils.py
ultralytics/yolo/utils/torch_utils.py
AODNet-dependent tasks.py logic
```

Reason:

```text
The current task only requires RecoveryBranch v1 code needed for Step 2 smoke.
The project currently has no ultralytics/nn/aod.py.
P3AFF identity, DehazeFeatureFuse, V2/V3C code, and p3_aff docs must remain intact.
```

## 5. P3AFF Impact Check

Preserved:

```text
P3AFF import in ultralytics/nn/tasks.py
P3AFF parse_model branch
ultralytics/models/v8/yolov8-aff-p3.yaml
docs/p3_aff_*.md
DehazeFeatureFuse / DehazeFeatureFuseSkip / DehazeFeatureFuseSkipResidual parsing
```

Verified after merge:

```text
P3AFF layer index: 22
Detect input: [22, 18, 21]
P3AFF input:  (1, 64, 80, 80)
P3AFF output: (1, 64, 80, 80)
stride: [8.0, 16.0, 32.0]
forward smoke: PASS
```

## 6. train.py / default.yaml / tasks.py Status

`tasks.py` was modified to:

```text
import RecoveryBranch
mount RecoveryBranch in DetectionModel when recovery=True
store self.recovery_output = self.recovery(x)
inject p3_fixed before Detect only when recovery_fuse=p3_fixed
write optional recovery shape debug lines
use getattr(self, "recovery_enabled", False) for old weight compatibility
```

`train.py` was modified to:

```text
pass recovery args into DetectionModel
write recovery_shape_debug.txt when recovery_debug_shapes=True
add train/recovery_loss and val/recovery_loss fields
add clean L1 recovery loss only when recovery_loss_weight > 0
support direct python ultralytics/yolo/v8/detect/train.py key=value commands
```

`default.yaml` now includes:

```yaml
recovery: False
recovery_fuse: none
recovery_alpha: 0.1
recovery_loss_weight: 0.0
recovery_debug_shapes: False
```

Default behavior remains disabled: `recovery=False`.

## 7. Dataset and runs/ Handling

The official command with:

```text
data=datasets/VOC_hazy_5beta/VOC_hazy_subset50.yaml
```

was attempted first. It failed before training because the YAML uses `path: .`, which resolved to the project root for this direct train.py entrypoint. The loader scanned paths like:

```text
images\train\000012_beta0.60.jpg
```

from the wrong root and marked all subset50 images corrupt/missing.

No dataloader or dataset YAML was modified in this merge. For smoke only, a temporary debug YAML was generated outside the repo:

```text
C:\tmp\recovery_v1_merge_debug.yaml
C:\tmp\recovery_v1_merge_train_debug.txt
C:\tmp\recovery_v1_merge_val_debug.txt
```

It uses 8 train and 8 val absolute image paths and is only for engineering smoke verification. Metrics from these runs are not method conclusions.

New smoke run outputs were generated under:

```text
runs/detect/recovery_v1_aux_only_1e_mergecheck/
runs/detect/recovery_v1_p3_fixed_1e_mergecheck/
```

These run artifacts and weights should not be added to git.

## 8. Verification Summary

Syntax:

```text
python -m py_compile ultralytics/nn/recovery.py ultralytics/nn/tasks.py ultralytics/yolo/v8/detect/train.py
PASS
```

Recovery forward smoke:

```text
recovery_fuse=none:
  dehaze_img: (1, 3, 640, 640)
  R3: (1, 64, 80, 80)

recovery_fuse=p3_fixed:
  dehaze_img: (1, 3, 640, 640)
  R3: (1, 64, 80, 80)
  Detect forward: PASS
```

1 epoch smoke with temporary absolute-path YAML:

```text
aux-only: PASS
p3_fixed: PASS
```

Aux-only final row:

```text
train/box_loss=3.0659
train/cls_loss=5.0928
train/dfl_loss=4.5542
train/dehaze_loss=0
train/recovery_loss=0.00192
metrics/mAP50(B)=0
```

P3-fixed final row:

```text
train/box_loss=3.0634
train/cls_loss=5.0974
train/dfl_loss=4.5602
train/dehaze_loss=0
train/recovery_loss=0.00192
metrics/mAP50(B)=0
```

The metric values are from 8 train / 8 val smoke data and must not be used as performance evidence.
