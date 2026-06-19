# Step 2A: Recovery R3 to P3AFF Fixed Gate Smoke

Date: 2026-06-19

This step connects RecoveryBranch v1 R3 into P3AFF and verifies only fixed-gate engineering paths. It does not implement learnable gate, haze-aware gate, Relationship Reasoning, or formal ablation training.

## 1. Modified Files

Core code:

```text
ultralytics/nn/modules.py
ultralytics/nn/tasks.py
```

New model YAMLs:

```text
ultralytics/models/v8/yolov8-recovery-aff-p3-a000.yaml
ultralytics/models/v8/yolov8-recovery-aff-p3-a005.yaml
```

Documentation:

```text
docs/p3_aff_step2a_fixed_gate_smoke.md
```

No loss, dataloader, train.py, or default.yaml changes were made in this step.

## 2. P3AFF Forward Logic

P3AFF now supports two input forms:

```python
P3AFF.forward(p3)
P3AFF.forward([p3, r3])
```

Single tensor input preserves Step 1 identity behavior:

```text
return p3
```

List/tuple input uses fixed gate:

```text
F3 = P3 + alpha * R3
```

Shape handling:

```text
If R3 spatial size differs from P3, R3 is bilinear-resized to P3.
If R3 channel count differs from P3, P3AFF raises RuntimeError.
```

Only `mode="fixed"` is implemented. Learnable and haze-aware modes are intentionally not implemented.

## 3. How R3 Is Read

RecoveryBranch still runs from the image input:

```python
self.recovery_output = self.recovery(x)
r3 = self.recovery_output["r3"]
```

In `DetectionModel._forward_once()`, when the current module is `P3AFF` and `recovery=True`, the model passes:

```python
x = m([p3, r3])
```

Otherwise it keeps the original path:

```python
x = m(p3)
```

This preserves `recovery=False` identity behavior.

## 4. Duplicate Fusion Avoidance

This step does not use teammate internal fusion:

```text
recovery_fuse=none
```

If a P3AFF YAML is used with `recovery_fuse=p3_fixed`, the code raises an error to avoid double fusion:

```text
P3AFF recovery fusion requires recovery_fuse=none to avoid duplicate fusion.
```

Therefore the tested path is:

```text
RecoveryBranch -> R3 only
P3AFF -> F3 fusion
Detect -> [F3, P4, P5]
```

## 5. YAML Configs

Two dedicated YAMLs were added:

```yaml
# alpha=0
- [15, 1, P3AFF, [fixed, 0.0]]
- [[22, 18, 21], 1, Detect, [nc]]
```

```yaml
# alpha=0.05
- [15, 1, P3AFF, [fixed, 0.05]]
- [[22, 18, 21], 1, Detect, [nc]]
```

Both set:

```yaml
recovery: True
recovery_fuse: none
```

## 6. py_compile

Command:

```powershell
C:\Anaconda\envs\yolo8hazy\python.exe -m py_compile ultralytics\nn\modules.py ultralytics\nn\tasks.py
```

Result:

```text
PASS
```

## 7. alpha=0 Forward Smoke

Model:

```text
ultralytics/models/v8/yolov8-recovery-aff-p3-a000.yaml
```

Random input:

```text
[1, 3, 640, 640]
```

Result:

```text
P3AFF index: 22
alpha: 0.0
P3: (1, 64, 80, 80)
R3: (1, 64, 80, 80)
F3: (1, 64, 80, 80)
Detect input: [22, 18, 21]
Detect output: [(1, 24, 8400), [(1, 84, 80, 80), (1, 84, 40, 40), (1, 84, 20, 20)]]
finite P3/R3/F3/Detect: True
max_abs_diff_F3_P3: 0.0
```

Conclusion:

```text
alpha=0 is identity-safe after routing R3 through P3AFF.
```

## 8. alpha=0.05 Forward Smoke

Model:

```text
ultralytics/models/v8/yolov8-recovery-aff-p3-a005.yaml
```

Random input:

```text
[1, 3, 640, 640]
```

Result:

```text
P3AFF index: 22
alpha: 0.05
P3: (1, 64, 80, 80)
R3: (1, 64, 80, 80)
F3: (1, 64, 80, 80)
Detect input: [22, 18, 21]
Detect output: [(1, 24, 8400), [(1, 84, 80, 80), (1, 84, 40, 40), (1, 84, 20, 20)]]
finite P3/R3/F3/Detect: True
max_abs_diff_F3_P3: 0.00031897786539047956
```

Conclusion:

```text
alpha=0.05 injects R3 through P3AFF without shape mismatch or NaN/inf.
```

## 9. 1 Epoch Debug

The official 5beta YAML still has a path resolution issue in this Windows/direct-train.py setup, so this step reused the temporary absolute-path smoke YAML:

```text
C:\tmp\recovery_v1_merge_debug.yaml
```

It contains 8 train and 8 val images and is only for engineering smoke. Metrics are not performance conclusions.

### alpha=0

Command summary:

```text
model=ultralytics/models/v8/yolov8-recovery-aff-p3-a000.yaml
data=C:/tmp/recovery_v1_merge_debug.yaml
epochs=1
recovery=True
recovery_fuse=none
recovery_loss_weight=0.0
name=5beta_recovery_p3aff_fixed000_debug_1e
```

Outputs:

```text
runs/detect/5beta_recovery_p3aff_fixed000_debug_1e/args.yaml
runs/detect/5beta_recovery_p3aff_fixed000_debug_1e/results.csv
runs/detect/5beta_recovery_p3aff_fixed000_debug_1e/recovery_shape_debug.txt
runs/detect/5beta_recovery_p3aff_fixed000_debug_1e/weights/best.pt
runs/detect/5beta_recovery_p3aff_fixed000_debug_1e/weights/last.pt
```

Shape debug:

```text
train: P3AFF_P3=(4, 64, 80, 80), P3AFF_R3=(4, 64, 80, 80), P3AFF_F3=(4, 64, 80, 80)
val:   P3AFF_P3=(8, 64, 64, 84), P3AFF_R3=(8, 64, 64, 84), P3AFF_F3=(8, 64, 64, 84)
```

Final row:

```text
epoch=0
train/box_loss=3.0659
train/cls_loss=5.0928
train/dfl_loss=4.5542
train/dehaze_loss=0
train/recovery_loss=0
metrics/mAP50(B)=0
val/box_loss=2.7504
val/cls_loss=4.8375
val/dfl_loss=4.1586
```

### alpha=0.05

Command summary:

```text
model=ultralytics/models/v8/yolov8-recovery-aff-p3-a005.yaml
data=C:/tmp/recovery_v1_merge_debug.yaml
epochs=1
recovery=True
recovery_fuse=none
recovery_loss_weight=0.0
name=5beta_recovery_p3aff_fixed005_debug_1e
```

Outputs:

```text
runs/detect/5beta_recovery_p3aff_fixed005_debug_1e/args.yaml
runs/detect/5beta_recovery_p3aff_fixed005_debug_1e/results.csv
runs/detect/5beta_recovery_p3aff_fixed005_debug_1e/recovery_shape_debug.txt
runs/detect/5beta_recovery_p3aff_fixed005_debug_1e/weights/best.pt
runs/detect/5beta_recovery_p3aff_fixed005_debug_1e/weights/last.pt
```

Shape debug:

```text
train: P3AFF_P3=(4, 64, 80, 80), P3AFF_R3=(4, 64, 80, 80), P3AFF_F3=(4, 64, 80, 80)
val:   P3AFF_P3=(8, 64, 64, 84), P3AFF_R3=(8, 64, 64, 84), P3AFF_F3=(8, 64, 64, 84)
```

Final row:

```text
epoch=0
train/box_loss=3.0567
train/cls_loss=5.0878
train/dfl_loss=4.5598
train/dehaze_loss=0
train/recovery_loss=0
metrics/mAP50(B)=0
val/box_loss=2.7504
val/cls_loss=4.8375
val/dfl_loss=4.1589
```

## 10. Can Move To 10e Fixed Alpha Ablation?

Engineering status:

```text
parse: PASS
forward smoke: PASS for alpha=0 and alpha=0.05
1 epoch debug: PASS for alpha=0 and alpha=0.05
duplicate teammate p3_fixed fusion avoided: YES
```

The code path is ready for a future fixed-alpha screening such as:

```text
alpha=0.05
alpha=0.10
alpha=0.20
```

However, do not start 10e ablation until confirmed by the project owner.

## 11. Remaining Issues

Still unresolved:

```text
Official datasets/VOC_hazy_5beta/VOC_hazy_subset50.yaml path resolution under current Windows/direct-train.py workflow.
Formal fixed-alpha runs should use a corrected official 5beta data config or a loader-safe invocation.
Current 1 epoch debug uses tiny temporary data and cannot support performance claims.
```
