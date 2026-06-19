# P3-AFF Fixed-Alpha 10e Commands

Do not run these until the dataset path is configured.

Use the same data YAML for every run:

```text
data=datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml
```

If this file contains another machine's absolute path, edit `path:` first or copy `datasets/VOC_hazy_5beta/VOC_hazy_subset50_template.yaml` and set `path:` to your local dataset directory.

## Baseline

```powershell
python ultralytics/yolo/v8/detect/train.py model=ultralytics/models/v8/yolov8.yaml data=datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml epochs=10 batch=4 imgsz=640 workers=0 seed=0 device=0 save_period=1 name=5beta_subset50_yolov8n_baseline_10e
```

## P3AFF alpha=0

```powershell
python ultralytics/yolo/v8/detect/train.py model=ultralytics/models/v8/yolov8-recovery-aff-p3-a000.yaml data=datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml epochs=10 batch=4 imgsz=640 workers=0 seed=0 device=0 save_period=1 recovery=True recovery_fuse=none recovery_loss_weight=0.0 name=5beta_subset50_p3aff_fixed000_10e
```

## P3AFF alpha=0.05

```powershell
python ultralytics/yolo/v8/detect/train.py model=ultralytics/models/v8/yolov8-recovery-aff-p3-a005.yaml data=datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml epochs=10 batch=4 imgsz=640 workers=0 seed=0 device=0 save_period=1 recovery=True recovery_fuse=none recovery_loss_weight=0.0 name=5beta_subset50_p3aff_fixed005_10e
```

## P3AFF alpha=0.10

```powershell
python ultralytics/yolo/v8/detect/train.py model=ultralytics/models/v8/yolov8-recovery-aff-p3-a010.yaml data=datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml epochs=10 batch=4 imgsz=640 workers=0 seed=0 device=0 save_period=1 recovery=True recovery_fuse=none recovery_loss_weight=0.0 name=5beta_subset50_p3aff_fixed010_10e
```

## P3AFF alpha=0.20

```powershell
python ultralytics/yolo/v8/detect/train.py model=ultralytics/models/v8/yolov8-recovery-aff-p3-a020.yaml data=datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml epochs=10 batch=4 imgsz=640 workers=0 seed=0 device=0 save_period=1 recovery=True recovery_fuse=none recovery_loss_weight=0.0 name=5beta_subset50_p3aff_fixed020_10e
```

Important: do not use `recovery_fuse=p3_fixed` for P3AFF experiments. That path performs teammate-side fusion before Detect and would duplicate P3AFF fusion.
