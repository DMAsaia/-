# Step 2a Plan: RecoveryBranch R3 to P3AFF

Date: 2026-06-19

This is a planning document only. It does not implement fixed gate, learnable gate, haze-aware gate, or Relationship Reasoning.

## Current Status

Step 1 P3AFF identity remains:

```text
P3AFF layer = 22
Detect input = [22, 18, 21]
P3AFF input/output = [B, 64, 80, 80] at imgsz=640
```

RecoveryBranch v1 is now available:

```python
recover = self.recovery(x)
dehaze_img = recover["dehaze_img"]
r3 = recover["r3"]
```

At `imgsz=640`:

```text
P3 = [B, 64, 80, 80]
R3 = [B, 64, 80, 80]
```

## Target Interface

Next step should make P3AFF consume both P3 and R3:

```python
r3 = self.recovery_output["r3"]
f3 = P3AFF([p3, r3])
```

Then Detect should consume:

```text
Detect([F3, P4, P5])
```

## Suggested Minimal Implementation

Keep the YAML-level P3AFF layer after layer 15:

```yaml
- [15, 1, P3AFF, [...]]
- [[22, 18, 21], 1, Detect, [nc]]
```

In `DetectionModel._forward_once()`, when the current module is `P3AFF` and `recovery=True`, read:

```python
r3 = self.recovery_output["r3"]
```

and pass:

```python
x = m([p3, r3])
```

instead of:

```python
x = m(p3)
```

`P3AFF.forward()` should accept both forms:

```python
P3AFF.forward(p3)        # identity fallback, Step 1 compatible
P3AFF.forward([p3, r3])  # Step 2a recovery feature path
```

This preserves old identity behavior when `recovery=False`.

## First Ablation Values

Only fixed alpha should be tried first:

```text
alpha = 0
alpha = 0.05
```

Meaning:

```text
alpha=0:
  F3 = P3
  Confirms that the new P3AFF([P3, R3]) path is identity-safe.

alpha=0.05:
  F3 = P3 + 0.05 * R3
  Minimal fixed recovery injection.
```

Do not implement learnable gate or haze-aware gate until this fixed-alpha path has passed parse, forward, and 1 epoch smoke.

## Shape Rules

Before fusion:

```text
assert R3 spatial size == P3 spatial size
```

If future R3 changes:

```text
R3 = interpolate(R3, size=P3.shape[-2:])
R3 = Conv1x1(R3_channels, P3_channels)(R3)
```

The adapter should live inside P3AFF or a small P3AFF adapter, not inside Detect.

## Evaluation Boundary

For Step 2a, run only:

```text
model parse
forward smoke
1 epoch debug
```

Do not run 10/50/100 epoch ablations yet. Debug metrics from tiny temporary samples must not be treated as conclusions.
