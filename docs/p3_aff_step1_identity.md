# Step 1: P3AFF Identity Integration

Date: 2026-06-18

This step implements only an identity P3 Adaptive Feature Fusion placeholder. It verifies that a P3-AFF layer can be inserted into YOLOv8 and that Detect can consume `[P3_fused, P4, P5]`. It does not implement RecoveryBranch, fixed gate, learnable gate, haze-aware gate, loss changes, or dataloader changes.

## 1. Modified Files

Core code:

```text
ultralytics/nn/modules.py
ultralytics/nn/tasks.py
ultralytics/models/v8/yolov8-aff-p3.yaml
```

Documentation:

```text
docs/p3_aff_step1_identity.md
docs/p3_aff_r3_interface.md
docs/p3_aff_next_ablation_plan.md
```

Generated debug run:

```text
runs/detect/5beta_p3aff_identity_debug_1e
```

Temporary debug-only files outside the repo:

```text
C:\tmp\p3aff_5beta_debug.yaml
C:\tmp\p3aff_5beta_train_debug.txt
C:\tmp\p3aff_5beta_val_debug.txt
```

## 2. P3AFF Class

Location:

```text
ultralytics/nn/modules.py
```

Implementation:

```python
class P3AFF(nn.Module):
    """Identity P3 Adaptive Feature Fusion placeholder.

    Step 1 only verifies model integration.
    It does not perform real recovery feature fusion yet.
    """

    def __init__(self, c1=None, enable=False, *args, **kwargs):
        super().__init__()
        self.c1 = c1
        self.enable = enable

    def forward(self, x):
        return x
```

Behavior:

```text
Input shape = output shape
Input channel count = output channel count
No recovery feature is used
No feature value is changed
```

## 3. Parser Registration

Location:

```text
ultralytics/nn/tasks.py
```

Changes:

```text
1. Imported P3AFF from ultralytics.nn.modules.
2. Added parse_model branch:
   c1, c2 = ch[f], ch[f]
   args = [c1, *args]
```

This guarantees that P3AFF keeps the same output channel count as its input P3 feature.

## 4. YAML Structure Change

New YAML:

```text
ultralytics/models/v8/yolov8-aff-p3.yaml
```

Original Detect input:

```yaml
- [[15, 18, 21], 1, Detect, [nc]]
```

Step 1 P3AFF structure:

```yaml
- [15, 1, P3AFF, []]  # 22 P3_fused, identity in Step 1
- [[22, 18, 21], 1, Detect, [nc]]  # 23 Detect(P3_fused, P4, P5)
```

Confirmed layer meaning:

```text
layer 15: P3 original
layer 18: P4
layer 21: P5
layer 22: P3AFF identity output, used as P3_fused
layer 23: Detect
```

## 5. py_compile Result

Command:

```powershell
C:\Anaconda\envs\yolo8hazy\python.exe -m py_compile ultralytics\nn\modules.py ultralytics\nn\tasks.py
```

Result:

```text
Passed. No syntax errors.
```

## 6. Model Parse Result

Command summary:

```text
Build DetectionModel from ultralytics/models/v8/yolov8-aff-p3.yaml with nc=20.
```

Result:

```text
parse_ok: true
layers: 24
p3aff_index: [22]
detect_from: [22, 18, 21]
stride: [8.0, 16.0, 32.0]
```

The parser recognized P3AFF and Detect receives P3AFF output, P4, and P5.

## 7. Forward Smoke Test Result

Input:

```text
torch.zeros(1, 3, 640, 640)
```

Hook result:

```text
P3AFF.forward called: true
P3AFF input shape:  (1, 64, 80, 80)
P3AFF output shape: (1, 64, 80, 80)
Detect output type: tuple
Detect output[0] shape: (1, 24, 8400)
```

Conclusion:

```text
P3AFF preserves P3 shape and channel count.
Detect runs normally after receiving [P3_fused, P4, P5].
```

## 8. 1 Epoch Debug Result

### 8.1 Attempt With Official 5beta YAML

Command used:

```powershell
yolo detect train model=ultralytics/models/v8/yolov8-aff-p3.yaml data=datasets/VOC_hazy_5beta/VOC_hazy.yaml imgsz=640 epochs=1 batch=4 device=0 workers=0 seed=0 save_period=1 name=5beta_p3aff_identity_debug_1e
```

Result:

```text
Failed before training.
Reason: No labels found.
```

Diagnosis:

```text
datasets/VOC_hazy_5beta/VOC_hazy.yaml uses train_5beta_subset30.txt.
That txt contains paths such as images/train/000012_beta0.60.jpg.
On this old Windows Ultralytics loader, img2label_paths() searches for "\images\" to replace it with "\labels\".
The forward-slash relative txt paths are not converted to label paths correctly.
Image files and label files do exist, but the official txt path format is incompatible with the current Windows label-path conversion.
```

No dataloader or dataset file was modified in Step 1.

### 8.2 Temporary 5beta Debug YAML

To verify the training loop without modifying dataloader or dataset config, a temporary debug YAML was generated in `C:\tmp` with 8 train and 8 val samples using absolute paths.

Command used:

```powershell
yolo detect train model=C:/Python/project/ultralytics-yolov8-official/ultralytics/models/v8/yolov8-aff-p3.yaml data=C:/tmp/p3aff_5beta_debug.yaml project=C:/Python/project/ultralytics-yolov8-official/runs/detect imgsz=640 epochs=1 batch=4 device=0 workers=0 seed=0 save_period=1 name=5beta_p3aff_identity_debug_1e exist_ok=True
```

Result:

```text
Training started: yes
Loss NaN: no
Validation ran: yes
results.csv generated: yes
best.pt generated: yes
last.pt generated: yes
```

Final row from `results.csv`:

```text
epoch: 0
train/box_loss: 3.107
train/cls_loss: 4.744
train/dfl_loss: 4.2774
train/dehaze_loss: 0
metrics/precision(B): 0
metrics/recall(B): 0
metrics/mAP50(B): 0
metrics/mAP50-95(B): 0
val/box_loss: 2.7504
val/cls_loss: 4.8374
val/dfl_loss: 4.1586
val/dehaze_loss: 0
```

Interpretation:

```text
The 1 epoch debug validates engineering integration only.
The metric values are from a tiny temporary sample and must not be used as method performance.
```

## 9. Can Step 2 Start?

Engineering status:

```text
P3AFF identity integration: passed
Model parse: passed
Forward smoke test: passed
1 epoch training loop: passed with temporary absolute-path 5beta debug YAML
Official VOC_hazy_5beta/VOC_hazy.yaml training: blocked by txt path format on Windows loader
```

Step 2 can start only after deciding how to handle R3:

```text
1. Wait for RecoveryBranch / R3 from the Recovery owner; or
2. Use R3 = proj(P3) only for debug, clearly marked as placeholder.
```

Do not start formal fixed-gate / learnable-gate / haze-aware-gate ablation until real R3 is available.

## 10. Current Dependency On Recovery Owner

P3AFF still needs the Recovery owner to provide:

```text
module file path
module class name
module input
module output
R3 shape
whether R3 depends on clean target
whether RecoveryBranch can run 1 epoch debug
```

Expected R3 shape for YOLOv8n + `imgsz=640`:

```text
[B, 64, 80, 80]
```

If R3 differs in spatial size or channels, P3AFF will need an interpolation/projection adapter in the next implementation step.

