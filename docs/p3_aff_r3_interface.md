# P3-AFF R3 Recovery Feature Interface

Date: 2026-06-18

This document defines what the RecoveryBranch owner needs to provide before P3-AFF can move from identity integration to real adaptive feature fusion.

## 1. Required Feature

The formal P3-AFF module needs a recovery feature:

```text
P3_original + R3 recovery feature -> P3AFF -> P3_fused
Detect([P3_fused, P4, P5])
```

Current Step 1 does not have a stable R3 source. The implemented `P3AFF` is identity-only and must not be treated as a complete Recovery + AFF method.

## 2. Expected P3 Shape

For the current YOLOv8n configuration and `imgsz=640`, P3 is expected to be:

```text
P3 shape: [B, 64, 80, 80]
```

Reason:

```text
P3 is layer 15.
YAML channel is 256.
YOLOv8n width multiplier is 0.25.
Actual P3 channels are 256 * 0.25 = 64.
P3 stride is 8, so 640 / 8 = 80.
```

## 3. Ideal R3 Shape

The ideal recovery feature should already match P3:

```text
R3 shape: [B, 64, 80, 80]
```

If R3 spatial size is different:

```text
R3 = interpolate(R3, size=P3.shape[-2:])
```

If R3 channel count is different:

```text
R3 = Conv1x1(R3_channels, P3_channels)(R3)
```

The projection should be inside P3-AFF or a small adapter module, not inside Detect.

## 4. Information Needed From Recovery Owner

The RecoveryBranch owner should provide:

```text
1. Module file path
2. Module class name
3. Module input
4. Module output
5. R3 tensor shape under YOLOv8n + imgsz=640
6. Whether the module depends on clean target
7. Whether it can run a 1 epoch debug
8. YAML insertion point and output layer index, if already known
```

## 5. Temporary Rule Before R3 Is Ready

Before a real R3 is provided:

```text
Do not run formal AFF ablations.
Do not claim Recovery + AFF is implemented.
Only identity P3AFF integration or R3 = proj(P3) debug is allowed.
Placeholder debug results cannot be used as final experimental conclusions.
```

## 6. Candidate Future YAML Pattern

When RecoveryBranch exists, the likely structure is:

```yaml
- [15, 1, RecoveryBranch, [...]]   # R3
- [[15, 22], 1, P3AFF, [...]]      # P3_original + R3 -> P3_fused
- [[23, 18, 21], 1, Detect, [nc]]
```

Layer numbers must be adjusted according to the final YAML.

