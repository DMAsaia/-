# P3-AFF GitHub Commit File List

This document defines the intended GitHub commit surface for the P3AFF + RecoveryBranch fixed-alpha handoff.

## Recommended files to commit

### Core code

- `.gitignore`
- `ultralytics/nn/modules.py`
- `ultralytics/nn/tasks.py`
- `ultralytics/nn/recovery.py`
- `ultralytics/yolo/v8/detect/train.py`
- `ultralytics/yolo/cfg/default.yaml`

### Model YAMLs

- `ultralytics/models/v8/yolov8-aff-p3.yaml`
- `ultralytics/models/v8/yolov8-recovery-aff-p3-a000.yaml`
- `ultralytics/models/v8/yolov8-recovery-aff-p3-a005.yaml`
- `ultralytics/models/v8/yolov8-recovery-aff-p3-a010.yaml`
- `ultralytics/models/v8/yolov8-recovery-aff-p3-a020.yaml`

### Lightweight dataset config

- `datasets/VOC_hazy_5beta/VOC_hazy_subset50_template.yaml`
- `datasets/VOC_hazy_5beta/train_5beta_subset50_local.txt`

Do not commit real dataset images, labels, clean references, or machine-specific absolute-path dataset YAMLs unless explicitly required for a local-only handoff.

### Scripts and tools

- `scripts/run_p3_aff_fixed_alpha_10e_commands.md`
- `tools/summarize_p3_aff_results.py`

### Documentation

- `docs/p3_aff_step0_repo_check.md`
- `docs/p3_aff_step1_identity.md`
- `docs/p3_aff_r3_interface.md`
- `docs/p3_aff_next_ablation_plan.md`
- `docs/p3_aff_step2a_fixed_gate_smoke.md`
- `docs/p3_aff_step2b_5beta_datafix.md`
- `docs/p3_aff_teammate_run_guide.md`
- `docs/p3_aff_result_summary_usage.md`
- `docs/p3_aff_github_handoff_checklist.md`
- `docs/p3_aff_github_commit_filelist.md`

## Additional files to review before committing

These files are currently modified or untracked in the working tree, but are outside the minimal P3AFF fixed-alpha GitHub handoff surface unless the change is intentionally part of the final delivery:

- `course_design_notes/scheme2_framework_guide.md`
- `course_design_notes/scheme2_framework_handoff.md`
- `tools/prepare_voc_hazy.py`
- `ultralytics/yolo/data/utils.py`
- `datasets/VOC_hazy/`
- `datasets/VOC_hazy_5beta/VOC_hazy.yaml`
- `datasets/VOC_hazy_5beta/VOC_hazy_subset30.yaml`
- `datasets/VOC_hazy_5beta/VOC_hazy_subset50.yaml`
- `datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml`
- `datasets/VOC_hazy_5beta/train_5beta_subset30.txt`
- `datasets/VOC_hazy_5beta/train_5beta_subset50.txt`
- `datasets/VOC_hazy_5beta/train_beta0.60_subset30.txt`
- `datasets/VOC_hazy_5beta/train_beta0.60_subset50.txt`
- `datasets/VOC_hazy_5beta/train_beta0.80_subset30.txt`
- `datasets/VOC_hazy_5beta/train_beta0.80_subset50.txt`
- `datasets/VOC_hazy_5beta/train_beta1.00_subset30.txt`
- `datasets/VOC_hazy_5beta/train_beta1.00_subset50.txt`
- `datasets/VOC_hazy_5beta/train_beta1.35_subset30.txt`
- `datasets/VOC_hazy_5beta/train_beta1.35_subset50.txt`
- `datasets/VOC_hazy_5beta/train_beta1.80_subset30.txt`
- `datasets/VOC_hazy_5beta/train_beta1.80_subset50.txt`
- `docs/p3_aff_step2a_recovery_to_aff_plan.md`
- `docs/recovery_v1_merge_precheck.md`
- `ultralytics/models/v8/yolov8n-recovery-v1.yaml`

## Files and directories that should not be committed

- `runs/`
- `weights/`
- `*.pt`
- `*.pth`
- `*.onnx`
- `*.engine`
- `datasets/VOC_hazy_5beta/images/`
- `datasets/VOC_hazy_5beta/labels/`
- `datasets/VOC_hazy_5beta/clean/`
- `.cache/`
- `__pycache__/`

## Tracked artifact cleanup commands

If `git ls-files runs` prints tracked files, remove them from Git tracking without deleting local files:

```powershell
git rm --cached -r runs
```

If `git ls-files "*.pt"` prints tracked files, remove tracked weights from Git tracking without deleting local files:

```powershell
git rm --cached -- "*.pt"
```

At the time of this check, no tracked `*.pth`, `*.onnx`, or `*.engine` files were found. If they appear later, use:

```powershell
git rm --cached -- "*.pth"
git rm --cached -- "*.onnx"
git rm --cached -- "*.engine"
```
