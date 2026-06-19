# P3-AFF result summary script usage

Script:

```text
tools/summarize_p3_aff_results.py
```

Purpose:

```text
Read args.yaml and results.csv from completed Ultralytics run directories and
produce a compact CSV/table for fixed-alpha P3-AFF comparison.
```

Example after 10e runs:

```powershell
python tools/summarize_p3_aff_results.py runs/detect/5beta_subset50_*_10e --output runs/detect/p3_aff_fixed_alpha_10e_summary.csv
```

Example for printing to terminal only:

```powershell
python tools/summarize_p3_aff_results.py runs/detect/5beta_subset50_p3aff_fixed*_10e
```

Collected fields include:

```text
run_dir
name
model
data
alpha
epochs_arg
batch
imgsz
seed
recovery
recovery_fuse
recovery_loss_weight
final_epoch
best_epoch
best/final precision, recall, mAP50, mAP50-95
best.pt / last.pt existence
recovery_shape_debug.txt existence
```

The script is read-only except when `--output` is provided. It does not train,
validate, or modify checkpoints.

Interpretation rule:

```text
Use this script only to organize completed run outputs. Do not treat 1e smoke
runs as performance conclusions.
```
