# Step 0: P3-AFF 工程检查与接入点定位

检查时间：2026-06-17

本轮只做工程检查、接口定位和下一步实现方案确认；未实现 P3AFF，未修改核心代码，未运行 10/50/100 epoch 训练。

## 1. 当前工程关键文件路径

目标工程：

```text
.
```

已确认存在：

```text
ultralytics\nn\tasks.py
ultralytics\nn\modules.py
ultralytics\yolo\v8\detect\train.py
ultralytics\yolo\cfg\default.yaml
ultralytics\models\v8
datasets\VOC_hazy_5beta
```

未发现或路径不同：

```text
ultralytics/nn/modules/    不存在；当前工程使用单文件 ultralytics/nn/modules.py
ultralytics/nn/aod.py      不存在
```

当前模型 YAML：

```text
ultralytics/models/v8/yolov8.yaml
ultralytics/models/v8/yolov8-dehaze.yaml
ultralytics/models/v8/yolov8-dehaze-v2.yaml
ultralytics/models/v8/yolov8-dehaze-v3.yaml
ultralytics/models/v8/yolov8-dehaze-v3b.yaml
ultralytics/models/v8/yolov8-dehaze-v3c.yaml
```

## 2. P3/P4/P5 的生成位置

以当前工程 `ultralytics/models/v8/yolov8.yaml` 为准：

```text
layer 15: C2f, 注释为 P3/8-small
layer 18: C2f, 注释为 P4/16-medium
layer 21: C2f, 注释为 P5/32-large
```

关键 YAML 片段：

```text
35: - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
36: - [-1, 3, C2f, [256]]        # 15 (P3/8-small)
38: - [-1, 1, Conv, [256, 3, 2]]
39: - [[-1, 12], 1, Concat, [1]] # cat head P4
40: - [-1, 3, C2f, [512]]        # 18 (P4/16-medium)
42: - [-1, 1, Conv, [512, 3, 2]]
43: - [[-1, 9], 1, Concat, [1]]  # cat head P5
44: - [-1, 3, C2f, [1024]]       # 21 (P5/32-large)
```

P3 的来源不是 backbone 原始 layer 4，而是 head 中融合 backbone P3 后得到的 layer 15。P3-AFF 第一阶段应接在 layer 15 后面。

## 3. Detect Head 的输入位置

原始 YOLOv8 Detect 输入：

```text
ultralytics/models/v8/yolov8.yaml
46: - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```

已有 V2 融合版本 Detect 输入：

```text
ultralytics/models/v8/yolov8-dehaze-v2.yaml
46: - [15, 1, DehazeFeatureFuse, [3, True, 0.1]]  # 22 fused P3 + dehaze image
47: - [[22, 18, 21], 1, Detect, [nc]]             # 23 Detect(P3_fused, P4, P5)
```

这说明当前工程已经验证过一种最小接入方式：在 P3 layer 15 后新增一个融合模块，输出 layer 22，再把 Detect 的第一个输入从 15 改为 22。

## 4. P3 的通道数和 shape 获取方式

### 4.1 通道数

YAML 中 P3 layer 15 写作 `C2f [256]`，但当前工程使用 YOLOv8 `scales`：

```text
n: [0.33, 0.25, 1024]
```

`parse_model()` 在 `ultralytics/nn/tasks.py` 中会对 Conv/C2f 等模块执行：

```text
c2 = make_divisible(min(c2, max_channels) * width, 8)
```

因此 YOLOv8n 下实际通道应为：

```text
P3 layer 15: 256 * 0.25 = 64 channels
P4 layer 18: 512 * 0.25 = 128 channels
P5 layer 21: 1024 * 0.25 = 256 channels
```

### 4.2 feature map 尺寸

P3 是 stride 8 特征。输入 `imgsz=640` 时，通常为：

```text
P3: [B, 64, 80, 80]
P4: [B, 128, 40, 40]
P5: [B, 256, 20, 20]
```

本轮尝试用当前 yolo8hazy 环境的 `python.exe` 做一次只读 forward hook 形状检查，但 Python/Torch 导入或构建模型过程超时，未获得运行输出。因此当前 shape 结论来自实际 YAML、`parse_model()` 缩放逻辑和 stride 结构推导。Step 1 前建议再用模型 summary 或 hook 复核一次。

## 5. 现有 AOD/Dehaze/Recovery 模块梳理

### 5.1 AODNet

检查结果：

```text
未发现 ultralytics/nn/aod.py
未在 ultralytics 代码中发现 AODNet 类定义
```

当前工程中的 AOD-Net 更多出现在课程设计说明文档中，不是当前可直接接入的代码模块。

### 5.2 DehazeHead

位置：

```text
ultralytics/nn/modules.py
class DehazeHead(nn.Module)
```

输入：

```text
P3 feature，来自 layer 15
```

输出：

```text
RGB 去雾图像，经过多次上采样和 Sigmoid
```

是否能提供 R3：

```text
不能直接提供与 P3 对齐的 recovery feature。DehazeHead 只输出图像，不输出 P3 尺度特征。
```

### 5.3 DehazeFeatureFuse

位置：

```text
ultralytics/nn/modules.py
class DehazeFeatureFuse(nn.Module)
```

输入：

```text
x = P3 feature
```

内部中间特征：

```text
dehaze_feat = self.feat(x)
```

输出：

```text
fused = x + alpha * gate(dehaze_feat) * dehaze_feat
dehaze_img = self.img(dehaze_feat)
return fused, dehaze_img
```

是否能提供 R3：

```text
可以作为 R3 接口的经验参考，因为 dehaze_feat 与 P3 同尺度、同通道。
但当前 forward 只对外返回 fused 和 dehaze_img，不单独返回 recovery feature。
```

### 5.4 DehazeFeatureFuseSkip

位置：

```text
ultralytics/nn/modules.py
class DehazeFeatureFuseSkip(nn.Module)
```

输入：

```text
[p3, p2, p1]
```

输出：

```text
fused P3
skip-decoded dehaze image
```

是否能提供 R3：

```text
内部 dehaze_feat 与 P3 对齐，可作为 R3 参考；但当前接口仍不单独返回 R3。
```

### 5.5 DehazeFeatureFuseSkipResidual

位置：

```text
ultralytics/nn/modules.py
class DehazeFeatureFuseSkipResidual(nn.Module)
```

输入：

```text
[p3, p2, p1, hazy]
```

输出：

```text
fused P3
residual RGB dehaze image
```

是否能提供 R3：

```text
内部 dehaze_feat 与 P3 对齐，可作为 R3 参考；但当前接口仍不单独返回 R3。
```

### 5.6 RecoveryBranch / Recovery Subnet

检查结果：

```text
未发现 RecoveryBranch 类
未发现独立 Recovery Subnet 实现
```

结论：

```text
当前没有 DR-YOLO 风格的 Recovery Subnet 可直接提供 R3。下一步 P3AFF 应先设计 R3 输入接口，不应强行在 AFF 模块内实现 Recovery。
```

### 5.7 AOD gate / haze-aware gate / feature consistency / partial10

检查结果：

```text
AOD gate: 未发现明确实现
haze-aware gate: 未发现明确实现
feature consistency: 未发现明确实现
partial10: 未发现明确实现
```

当前已有 gate 形式：

```text
DehazeFeatureFuse.gate = Conv2d(c1, c1, 1) + Sigmoid
DehazeFeatureFuseSkip.gate = Conv2d(c1, c1, 1) + Sigmoid
DehazeFeatureFuseSkipResidual.gate = Conv2d(c1, c1, 1) + Sigmoid
```

它们是通道数为 `c1` 的卷积门控，不是后续计划中的 fixed / learnable scalar / haze-aware 三类标准消融接口。

## 6. 5beta 数据集配置检查

5beta 数据集目录：

```text
datasets\VOC_hazy_5beta
```

YAML 文件：

```text
datasets/VOC_hazy_5beta/VOC_hazy.yaml
datasets/VOC_hazy_5beta/VOC_hazy_subset30.yaml
datasets/VOC_hazy_5beta/VOC_hazy_subset50.yaml
```

默认 `VOC_hazy.yaml`：

```text
path: .
train: train_5beta_subset30.txt
val: images/val
test: images/test
clean: clean
```

subset50：

```text
path: .
train: train_5beta_subset50.txt
val: images/val
test: images/test
clean: clean
```

类别：

```text
nc = 20
names = VOC 20 类
```

数据目录：

```text
images/train, images/val, images/test
labels/train, labels/val, labels/test
clean/train, clean/val, clean/test
```

clean target：

```text
存在 clean 目录。
ultralytics/yolo/data/base.py 会把 images 路径替换为 sibling clean 路径，并要求同名 clean 文件存在。
例如 images/train/xxx_beta0.60.jpg 对应 clean/train/xxx_beta0.60.jpg。
```

是否需要 hazy-clean 配对：

```text
旧 Dehaze clean-supervision 路线需要。
纯 P3-AFF 如果只做特征融合且不加 clean loss，则不必依赖 clean target。
后续若接 RSM recon-only，则需要 hazy 图像本身，不应再使用 clean L1/SSIM/color 监督。
```

当前 baseline 结果：

```text
runs/detect/baseline_yolov8n_100e 存在，但 args.yaml 显示 data=datasets/VOC_hazy/VOC_hazy.yaml，是旧单数据集，不是 5beta。
未在 runs/detect 的 args.yaml 中确认到 VOC_hazy_5beta / 5beta / subset30 / subset50 训练记录。
```

因此：

```text
当前尚未确认本机已有 5beta baseline 结果。
旧 VOC_hazy baseline 和 V2/V3 结果只能作为结构经验参考，不能作为最终对照。
```

## 7. P3-AFF 最小实现方案

本轮不实现，只确认下一步最小改动方案。

### 7.1 新增或修改文件

建议 Step 1 最小改动：

```text
修改 ultralytics/nn/modules.py
  新增 P3AFF 或 AFFP3 模块类

修改 ultralytics/nn/tasks.py
  import 新模块
  在 parse_model() 中加入 P3AFF 通道解析逻辑

新增 ultralytics/models/v8/yolov8-aff-p3.yaml
  复制 yolov8.yaml 主体
  在 layer 15 后插入 P3AFF
  Detect 输入由 [15,18,21] 改为 [22,18,21]
```

可选但不建议 Step 1 立刻修改：

```text
ultralytics/yolo/cfg/default.yaml
ultralytics/yolo/cfg/__init__.py
ultralytics/yolo/v8/detect/train.py
```

原因：

```text
第一步可以先用 YAML args 固定默认行为，避免过早扩展全局 CLI 参数。
等 P3AFF 空模块跑通后，再增加 aff_p3 / aff_gate / aff_alpha 等配置。
```

### 7.2 P3AFF 模块放置位置

当前工程没有 `ultralytics/nn/modules/` 包目录，因此应放在：

```text
ultralytics/nn/modules.py
```

建议靠近现有 `DehazeFeatureFuse` 类，便于复用 P3 融合经验，但命名和职责应区别于去雾图像输出。

### 7.3 P3_original 如何取得

最小方式：

```text
在 YAML 中把 P3AFF 的 from 设置为 15。
P3AFF.forward(x) 中的 x 就是 P3_original。
```

对应结构：

```text
- [15, 1, P3AFF, [...]]     # 22 P3_fused
- [[22, 18, 21], 1, Detect, [nc]]
```

### 7.4 R3 recovery feature 从哪里取得

当前未发现独立 RecoveryBranch，因此 R3 暂时没有稳定来源。

建议接口预留：

```text
P3AFF.forward(x)
```

第一阶段可以先让模块内部生成一个 `r3 = proj(x)` 作为占位 recovery feature，用于验证接入链路；但要在报告中说明这不是完整 Recovery Subnet。

等 Recovery 负责人提供模块后，再切换为：

```text
P3AFF.forward([p3, r3])
```

或在 YAML 中：

```text
- [15, 1, RecoveryBranch, [...]]  # R3
- [[15, 22], 1, P3AFF, [...]]     # P3_original + R3 -> P3_fused
- [[23, 18, 21], 1, Detect, [nc]]
```

### 7.5 P3_fused 如何替换 Detect Head 的 P3

沿用 V2 已验证的 YAML 接入方式：

```text
原始:
- [[15, 18, 21], 1, Detect, [nc]]

P3-AFF:
- [15, 1, P3AFF, [...]]
- [[22, 18, 21], 1, Detect, [nc]]
```

这样不需要修改 Detect Head 本身，也不需要改 loss。

### 7.6 如何保证默认关闭 AFF 时原 YOLO 行为不变

建议：

```text
1. 原始 yolov8.yaml 不修改；
2. P3AFF 单独放在新 YAML，例如 yolov8-aff-p3.yaml；
3. P3AFF 内部支持 enable=False 时直接 return x；
4. fixed gate 初始 alpha 可设为 0 或小值，Step 1 空模块先 return x；
5. 不在 default.yaml 中把 AFF 默认打开。
```

这样使用 `model=yolov8n.pt` 或 `model=ultralytics/models/v8/yolov8.yaml` 时，原 YOLO 行为保持不变。

## 8. 下一步建议改哪些文件

建议 Step 1 只做最小空模块接入：

```text
ultralytics/nn/modules.py
  新增 P3AFF，forward 先 return x

ultralytics/nn/tasks.py
  import P3AFF
  parse_model 中加入 P3AFF 分支，使 c2 = ch[f]

ultralytics/models/v8/yolov8-aff-p3.yaml
  在 layer 15 后插入 P3AFF
  Detect 改为 [P3_fused, P4, P5]
```

Step 1 验证建议：

```text
只做 model parse / forward smoke test
不训练长 epoch
确认 P3AFF enable=False 或 identity 输出时，模型能正常构建和前向
```

Step 2 再做：

```text
fixed gate
learnable gate
haze-aware gate
gate 统计日志
短 epoch debug
```

## 9. 仍不确定，需要确认的问题

```text
1. Recovery 负责人是否会提供独立 RecoveryBranch，以及输出 R3 的 shape 是否为 [B, 64, 80, 80]。
2. 如果短期没有 R3，P3AFF 第一版是否允许使用内部 proj(P3) 作为占位 recovery feature。
3. 5beta 最终训练使用 subset30、subset50 还是 full train list；当前 VOC_hazy.yaml 默认是 subset30。
4. 5beta baseline 是否已经由队友完成；本机 runs/detect 中暂未确认到 5beta baseline。
5. 是否继续保留旧 clean target 加载逻辑；纯 AFF 不需要 clean supervision，但旧 dataloader 当前会强制寻找 clean 文件。
6. Step 1 是否需要把 AFF 参数接入 CLI/default.yaml，还是先只通过 YAML 固定配置跑通。
```

## 10. 可行性结论

```text
是否已经找到 P3 接入点：是，layer 15。
是否已经找到 Detect Head 输入点：是，原始 [15,18,21]，V2 已示例 [22,18,21]。
是否已经能确定 R3 来源：否。当前只有 DehazeFeatureFuse 内部 dehaze_feat 可作参考，未发现独立 RecoveryBranch。
下一步是否可以开始实现 P3AFF：可以，但建议 Step 1 先做 enable=False / identity 空模块接入。
如果不能完整实现，缺少的信息：RecoveryBranch/R3 的正式来源和 shape 约定，以及 5beta baseline 的最终对照配置。
```

## 11. 本轮代码状态说明

本轮计划只新增：

```text
docs/p3_aff_step0_repo_check.md
```

检查前 `git status --short` 已显示既有修改：

```text
 M tools/prepare_voc_hazy.py
```

该文件不是本轮修改，后续状态检查时应与本轮新增文档区分。
