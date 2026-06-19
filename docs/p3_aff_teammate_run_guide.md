# P3-AFF + RecoveryBranch fixed-alpha teammate run guide

This handoff is for fixed-alpha P3-AFF ablation only. It does not include
learnable gate, haze-aware gate, Relationship Reasoning, dataloader rewrites, or
formal 50e/100e experiments.

## Current model path

The implemented path is:

```text
RecoveryBranch -> r3
P3AFF([p3, r3]) -> F3 = P3 + alpha * R3
Detect([F3, P4, P5])
```

Important settings:

```text
recovery=True
recovery_fuse=none
recovery_loss_weight=0.0
```

Do not use `recovery_fuse=p3_fixed` as the P3-AFF result. That option performs
an internal recovery fusion path and would duplicate the P3-AFF fusion.

## Fixed-alpha model configs

Use these model YAML files:

```text
ultralytics/models/v8/yolov8-recovery-aff-p3-a000.yaml  alpha=0.00
ultralytics/models/v8/yolov8-recovery-aff-p3-a005.yaml  alpha=0.05
ultralytics/models/v8/yolov8-recovery-aff-p3-a010.yaml  alpha=0.10
ultralytics/models/v8/yolov8-recovery-aff-p3-a020.yaml  alpha=0.20
```

All four configs keep `recovery: True` and `recovery_fuse: none`.

## Dataset config

For this machine the verified data config is:

```text
datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml
```

It uses:

```text
train: train_5beta_subset50_local.txt
val: images/val
test: images/test
clean: clean
```

The local YAML contains an absolute `path` for this workstation. Teammates should
copy `datasets/VOC_hazy_5beta/VOC_hazy_subset50_template.yaml` or edit their own
local YAML so that `path` points to their clone's `datasets/VOC_hazy_5beta`
folder. Keep the data directory layout unchanged:

```text
datasets/VOC_hazy_5beta/images/train
datasets/VOC_hazy_5beta/images/val
datasets/VOC_hazy_5beta/labels/train
datasets/VOC_hazy_5beta/labels/val
datasets/VOC_hazy_5beta/clean
```

Dataset images, labels, clean images, checkpoints, and `runs/` outputs are not
intended to be committed to GitHub.

## Recommended 10e commands

The command list is stored in:

```text
scripts/run_p3_aff_fixed_alpha_10e_commands.md
```

Run the baseline and the four fixed-alpha variants from the repository root.
Use the same `batch`, `imgsz`, `workers`, `seed`, and dataset YAML across all
runs.

## Pre-run smoke checks

Before 10e ablation, teammates should run:

```powershell
python -m py_compile ultralytics/nn/modules.py ultralytics/nn/tasks.py ultralytics/nn/recovery.py
```

Then instantiate all four fixed-alpha YAMLs once to confirm parsing and forward
shape behavior in the target environment. No formal metrics should be concluded
from smoke runs.

## Result summary

After runs finish, summarize them with:

```powershell
python tools/summarize_p3_aff_results.py runs/detect/5beta_subset50_*_10e --output runs/detect/p3_aff_fixed_alpha_10e_summary.csv
```

See:

```text
docs/p3_aff_result_summary_usage.md
```

## Experiment boundary

Allowed in the next fixed-alpha ablation:

```text
baseline YOLOv8n 10e
alpha=0.00 10e
alpha=0.05 10e
alpha=0.10 10e
alpha=0.20 10e
```

Not part of this handoff:

```text
learnable gate
haze-aware gate
Relationship Reasoning
50e / 100e formal runs
changing loss, dataloader, or train entry
```
