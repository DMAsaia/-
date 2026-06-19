# P3-AFF Next Ablation Plan

Date: 2026-06-18

This plan starts after Step 1 identity integration passes parse, forward, and 1 epoch debug. It does not implement any gate in Step 1.

## Preconditions

Before formal ablation:

```text
1. P3AFF identity module can be parsed.
2. Detect receives [P3_fused, P4, P5].
3. 1 epoch debug produces results.csv and weights.
4. RecoveryBranch provides a valid R3 recovery feature.
```

If R3 is still unavailable, only `R3 = proj(P3)` debug is allowed. Placeholder debug must not be reported as a final method result.

## Step 2: Fixed Gate

Goal:

```text
Verify whether a fixed recovery feature injection strength is useful.
```

Formula:

```text
P3_fused = P3 + alpha * R3
```

Ablation values:

```text
alpha = 0.05
alpha = 0.10
alpha = 0.20
```

Suggested experiment names:

```text
5beta_p3aff_fixed005
5beta_p3aff_fixed010
5beta_p3aff_fixed020
```

## Step 3: Learnable Gate

Goal:

```text
Let the model learn the recovery feature fusion strength.
```

Formula:

```text
P3_fused = P3 + sigmoid(gate) * R3
```

Recommended first implementation:

```text
scalar gate initialized around alpha=0.10
```

This keeps the first learnable version close to the fixed-gate setting that has been relatively stable in previous P3 fusion experiments.

## Step 4: Haze-Aware Gate

Goal:

```text
Adapt fusion strength according to haze severity.
```

Initial plan:

```text
Use a haze score to choose light / medium / heavy fusion strength.
```

Possible mapping:

```text
light haze:  lower fusion strength
medium haze: medium fusion strength
heavy haze:  higher fusion strength
```

The exact haze score source should be decided after RecoveryBranch / R3 is available.

## Evaluation Rules

Use 5beta data only for final comparison:

```text
data=datasets/VOC_hazy_5beta/VOC_hazy.yaml
or data=datasets/VOC_hazy_5beta/VOC_hazy_subset50.yaml
```

Report:

```text
Precision
Recall
mAP50
mAP50-95
best epoch
final epoch
gate settings
R3 source
```

Do not compare final conclusions against old single-beta VOC_hazy results except as historical reference.

