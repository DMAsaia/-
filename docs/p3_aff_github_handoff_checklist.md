# P3-AFF GitHub handoff checklist

## Included code/config

- `ultralytics/nn/modules.py`: `P3AFF` supports `forward(p3)` identity and
  `forward([p3, r3])` fixed gate.
- `ultralytics/nn/tasks.py`: `DetectionModel._forward_once()` passes
  `RecoveryBranch` R3 into `P3AFF` when `recovery=True`.
- `ultralytics/nn/recovery.py`: lightweight RecoveryBranch v1 interface.
- `ultralytics/models/v8/yolov8-recovery-aff-p3-a000.yaml`
- `ultralytics/models/v8/yolov8-recovery-aff-p3-a005.yaml`
- `ultralytics/models/v8/yolov8-recovery-aff-p3-a010.yaml`
- `ultralytics/models/v8/yolov8-recovery-aff-p3-a020.yaml`
- `datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml`
- `datasets/VOC_hazy_5beta/VOC_hazy_subset50_template.yaml`
- `datasets/VOC_hazy_5beta/train_5beta_subset50_local.txt`
- `scripts/run_p3_aff_fixed_alpha_10e_commands.md`
- `tools/summarize_p3_aff_results.py`

## GitHub ignore policy

These should not be uploaded:

- `runs/`
- `weights/`
- `*.pt`, `*.pth`, `*.onnx`, `*.engine`
- `__pycache__/`, `*.pyc`, `.cache/`
- `datasets/VOC_hazy_5beta/images/`
- `datasets/VOC_hazy_5beta/labels/`
- `datasets/VOC_hazy_5beta/clean/`

These lightweight dataset config files are allowed:

- `datasets/VOC_hazy_5beta/*.yaml`
- `datasets/VOC_hazy_5beta/train_*subset*.txt`

## Required pre-push checks

Run from repository root:

```powershell
python -m py_compile ultralytics/nn/modules.py ultralytics/nn/tasks.py ultralytics/nn/recovery.py tools/summarize_p3_aff_results.py
```

Parse all fixed-alpha model YAMLs:

```text
alpha=0.00
alpha=0.05
alpha=0.10
alpha=0.20
```

No formal training is required for this GitHub handoff.

## Next valid experiment

After teammates clone the repo and place the dataset locally, they can run:

```text
baseline YOLOv8n 10e
P3-AFF fixed alpha=0.00 10e
P3-AFF fixed alpha=0.05 10e
P3-AFF fixed alpha=0.10 10e
P3-AFF fixed alpha=0.20 10e
```

Keep `recovery_fuse=none` for every P3-AFF run.

## Explicitly out of scope

- learnable gate
- haze-aware gate
- Relationship Reasoning
- official 50e / 100e runs
- changing loss, dataloader, or formal training scripts
