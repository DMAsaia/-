# P3AFF fixed-alpha pretrained loading check

检查日期：2026-06-22

## 结论

四个 fixed-alpha P3AFF YAML 均可使用本地 `yolov8n.pt` 的可匹配预训练权重。现有 CLI、`DetectionTrainer.get_model()` 和 `DetectionModel.load()` 已支持自定义 YAML 加载预训练模型，无需修改模型结构。CLI 命令必须显式传入 `recovery=True`，否则默认配置会覆盖 YAML 中的 RecoveryBranch 开关。

- 官方 YOLOv8n Detect 位于 layer 22，自定义 P3AFF 模型的 Detect 位于 layer 23。
- `DetectionModel.load()` 会把官方 `model.22.*` Detect 键按名称和 shape 映射到自定义模型的 `model.23.*`。
- RecoveryBranch、`recovery_p3_adapter` 在官方权重中不存在，因此保持随机初始化。
- 当前 fixed gate `P3AFF` 没有可训练 state keys；`alpha` 来自 YAML，forward 正常。
- 全部 YAML 保持 `recovery=True`、`recovery_fuse=none`，未发生重复融合。

## 方案 A 路径确认

命令入口对字符串形式的 `pretrained` 执行以下路径：

1. CLI 创建 `YOLO(custom_yaml)`。
2. `pretrained=yolov8n.pt` 触发 `YOLO.load()`。
3. `YOLO.load()` 调用 `DetectionModel.load()`，迁移名称和 shape 匹配的权重。
4. 训练器按数据集 `nc` 重建模型时，再通过 `DetectionTrainer.get_model(cfg, weights)` 迁移一次可匹配权重。

对应代码位置：

- `ultralytics/yolo/cfg/__init__.py`：字符串 `pretrained` 调用 `model.load(...)`。
- `ultralytics/yolo/v8/detect/train.py`：`get_model()` 构建 `DetectionModel` 后调用 `model.load(weights)`。
- `ultralytics/nn/tasks.py`：`BaseModel.load()` 执行 shape 交集，并重映射 Detect 层索引。

没有执行 `epochs=0` 训练命令，因为该路径仍可能初始化 trainer、dataloader 和 `runs/`。采用了等价且更小的加载验证：

```python
m = YOLO("ultralytics/models/v8/yolov8-recovery-aff-p3-a005.yaml")
m.load("yolov8n.pt")
out = m.model(torch.zeros(1, 3, 64, 64))
```

结果：`Transferred 355/355 items from pretrained weights`，forward 输出 `(1, 84, 84)`，全部 finite，Detect layer 为 23。

### 方案 A 命令修正

对题目中未显式传入 Recovery 参数的命令进行配置解析，结果为：

```text
pretrained='yolov8n.pt'
recovery=False
recovery_fuse=None
recovery_loss_weight=0.0
```

原因是 `DetectionTrainer.get_model()` 将 CLI/default args 中的 `recovery=False` 显式传给 `DetectionModel`，覆盖了模型 YAML 的 `recovery=True`。因此题目中的原命令可以证明预训练权重路径可用，但不能作为 P3AFF+Recovery 实验命令。正确实验命令必须包含：

```powershell
python ultralytics/yolo/v8/detect/train.py model=ultralytics/models/v8/yolov8-recovery-aff-p3-a005.yaml pretrained=yolov8n.pt data=datasets/VOC_hazy_5beta/VOC_hazy_subset50_local.yaml recovery=True recovery_fuse=none recovery_loss_weight=0.0
```

本检查没有执行该训练命令，也没有添加任何 epoch 参数。

## 统计口径

- `loaded keys`：名称相同，或 Detect 层重映射后名称对应，且 shape 相同的 source state keys。
- `missing keys`：目标模型 state keys 中没有加载预训练值的键。
- `unexpected keys`：对过滤后的 state dict 执行 `strict=False` 加载时的 unexpected keys。
- `source unmatched keys`：源权重在目标模型中没有对应名称的键。
- `shape mismatch keys`：名称对应但 tensor shape 不同的源键。

## nc=80 初次加载

以 `a005` 验证 CLI 初次 `YOLO(custom_yaml).load(yolov8n.pt)` 的同类别数场景：

| 项目 | 数量 |
| --- | ---: |
| source keys | 355 |
| target keys | 395 |
| loaded keys | 355 |
| direct keys | 270 |
| Detect remapped keys | 85 |
| missing keys | 40 |
| unexpected keys | 0 |
| source unmatched keys | 0 |
| shape mismatch keys | 0 |

40 个 missing keys 全部属于 RecoveryBranch 和 `recovery_p3_adapter`。加载前后参数逐键比较一致，确认保持随机初始化。

## VOC nc=20 最终模型加载

实际 VOC 数据集覆盖 `nc=20` 后，四个 YAML 的统计一致：

| YAML | alpha | loaded | missing | unexpected | source unmatched | shape mismatch | forward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `yolov8-recovery-aff-p3-a000.yaml` | 0.00 | 319 | 76 | 0 | 0 | 36 | PASS |
| `yolov8-recovery-aff-p3-a005.yaml` | 0.05 | 319 | 76 | 0 | 0 | 36 | PASS |
| `yolov8-recovery-aff-p3-a010.yaml` | 0.10 | 319 | 76 | 0 | 0 | 36 | PASS |
| `yolov8-recovery-aff-p3-a020.yaml` | 0.20 | 319 | 76 | 0 | 0 | 36 | PASS |

每个模型的 319 个 loaded keys 包括：

- 270 个同名 backbone/neck keys。
- 49 个从官方 Detect layer 22 重映射到自定义 Detect layer 23 的 keys，包括可匹配的 box regression/DFL 权重。

每个模型的 76 个 missing keys 包括：

- 40 个 RecoveryBranch 和 `recovery_p3_adapter` keys，预期随机初始化。
- 36 个 Detect 分类分支 `model.23.cv3.*` keys，由 COCO `nc=80` 与 VOC `nc=20` 引起 shape mismatch，预期随机初始化。

36 个 shape mismatch keys 均来自官方 `model.22.cv3.{0,1,2}.*` 到目标 `model.23.cv3.{0,1,2}.*` 的分类分支映射。不存在无对应名称的 source keys，过滤后加载也不存在 unexpected keys。

## Forward smoke

四个 VOC `nc=20` 模型均使用输入 `(1, 3, 64, 64)` 完成 eval forward：

- Detect 输出：`(1, 24, 84)`。
- `P3 = R3 = F3 = (1, 64, 8, 8)`。
- 输出全部 finite，无 NaN/Inf。
- P3AFF layer 为 22，Detect layer 为 23。
- 四个 YAML 的运行时 alpha 分别为 `0.00`、`0.05`、`0.10`、`0.20`。
- 所有预期 loaded keys 均逐键验证与 `yolov8n.pt` 相同。
- RecoveryBranch 加载命中数为 0，加载前后随机初始化值保持不变。

## 最终判断

fixed-alpha P3AFF 实验可以使用 `pretrained=yolov8n.pt`。Backbone、neck 和 shape 匹配的 Detect 权重会成功迁移；RecoveryBranch、无参数的 fixed P3AFF，以及 VOC 类别数导致不匹配的 Detect 分类分支按预期使用初始化值。正式命令必须显式设置 `recovery=True recovery_fuse=none recovery_loss_weight=0.0`。本检查未启动训练，未运行 10e/50e/100e，也未修改模型结构。
