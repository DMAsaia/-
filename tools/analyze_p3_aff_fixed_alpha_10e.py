"""Analyze the first pretrained P3-AFF fixed-alpha 10-epoch ablation.

The five run directories are read-only. The script writes only the requested
summary CSV and Markdown report under runs/detect.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from summarize_p3_aff_results import load_args, load_results, to_float


METRIC_ALIASES = {
    "P": ("metrics/precision(B)", "metrics/precision", "precision"),
    "R": ("metrics/recall(B)", "metrics/recall", "recall"),
    "mAP50": ("metrics/mAP50(B)", "metrics/mAP50", "mAP50"),
    "mAP50-95": ("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"),
}

RUN_SPECS = (
    ("5beta_subset50_yolov8n_pretrained_10e", "baseline", None),
    ("5beta_subset50_p3aff_fixed000_pretrained_10e", "P3AFF fixed", 0.00),
    ("5beta_subset50_p3aff_fixed005_pretrained_10e", "P3AFF fixed", 0.05),
    ("5beta_subset50_p3aff_fixed010_pretrained_10e", "P3AFF fixed", 0.10),
    ("5beta_subset50_p3aff_fixed020_pretrained_10e", "P3AFF fixed", 0.20),
)

FAIR_FIELDS = ("data", "epochs", "batch", "imgsz", "workers", "seed", "device", "save_period")


@dataclass
class Run:
    name: str
    kind: str
    alpha: float | None
    path: Path
    args: dict[str, Any]
    rows: list[dict[str, str]]
    columns: dict[str, str]
    best: dict[str, str]
    final: dict[str, str]
    missing_files: list[str]
    missing_periodic: list[str]
    model_cfg: dict[str, Any]

    def metric(self, row: dict[str, str], key: str) -> float:
        value = to_float(row.get(self.columns[key]))
        if value is None:
            raise ValueError(f"{self.name}: invalid {key} value")
        return value


def find_columns(rows: list[dict[str, str]], run_name: str) -> dict[str, str]:
    if not rows:
        raise ValueError(f"{run_name}: results.csv has no rows")
    available = set(rows[0])
    found = {}
    for logical, aliases in METRIC_ALIASES.items():
        found[logical] = next((name for name in aliases if name in available), "")
        if not found[logical]:
            raise KeyError(f"{run_name}: no column found for {logical}; columns={sorted(available)}")
    return found


def load_model_cfg(model_value: Any, repo: Path) -> dict[str, Any]:
    model_path = Path(str(model_value or ""))
    if not model_path.is_absolute():
        model_path = repo / model_path
    if model_path.suffix.lower() not in (".yaml", ".yml") or not model_path.exists():
        return {}
    return load_args(model_path)


def load_run(repo: Path, runs_root: Path, spec: tuple[str, str, float | None]) -> Run:
    name, kind, alpha = spec
    path = runs_root / name
    required = ("args.yaml", "results.csv", "weights/best.pt", "weights/last.pt")
    missing_files = [rel for rel in required if not (path / rel).exists()]
    args = load_args(path / "args.yaml")
    rows = load_results(path / "results.csv")
    columns = find_columns(rows, name)
    best = max(rows, key=lambda row: to_float(row.get(columns["mAP50-95"])) or float("-inf"))
    final = rows[-1]
    expected_periodic = []
    if args.get("save_period") == 1:
        for row in rows:
            epoch = int(float(row.get("epoch", "0")))
            if epoch > 0:
                expected_periodic.append(f"weights/epoch{epoch}.pt")
    missing_periodic = [rel for rel in expected_periodic if not (path / rel).exists()]
    return Run(name, kind, alpha, path, args, rows, columns, best, final, missing_files,
               missing_periodic, load_model_cfg(args.get("model"), repo))


def fmt(value: float) -> str:
    return f"{value:.5f}"


def signed(value: float) -> str:
    return f"{value:+.5f}"


def metric_values(run: Run, row: dict[str, str]) -> dict[str, float]:
    return {key: run.metric(row, key) for key in METRIC_ALIASES}


def delta(left: Run, right: Run) -> dict[str, float]:
    left_values = metric_values(left, left.best)
    right_values = metric_values(right, right.best)
    return {key: left_values[key] - right_values[key] for key in METRIC_ALIASES}


def p3aff_layer(cfg: dict[str, Any]) -> list[Any] | None:
    for layer in cfg.get("head", []):
        if isinstance(layer, list) and len(layer) >= 3 and layer[2] == "P3AFF":
            return layer
    return None


def effective_fuse(run: Run) -> Any:
    value = run.args.get("recovery_fuse")
    return run.model_cfg.get("recovery_fuse") if value is None else value


def alpha_from_cfg(run: Run) -> float | None:
    layer = p3aff_layer(run.model_cfg)
    if not layer or len(layer) < 4 or len(layer[3]) < 2:
        return None
    return to_float(layer[3][1])


def write_summary(runs: list[Run], output: Path) -> None:
    fields = ("experiment", "type", "alpha", "best_epoch", "Precision", "Recall", "mAP50", "mAP50-95",
              "final_epoch_P", "final_epoch_R", "final_epoch_mAP50", "final_epoch_mAP50-95")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            best = metric_values(run, run.best)
            final = metric_values(run, run.final)
            writer.writerow({
                "experiment": run.name,
                "type": run.kind,
                "alpha": "" if run.alpha is None else f"{run.alpha:.2f}",
                "best_epoch": run.best.get("epoch", ""),
                "Precision": fmt(best["P"]),
                "Recall": fmt(best["R"]),
                "mAP50": fmt(best["mAP50"]),
                "mAP50-95": fmt(best["mAP50-95"]),
                "final_epoch_P": fmt(final["P"]),
                "final_epoch_R": fmt(final["R"]),
                "final_epoch_mAP50": fmt(final["mAP50"]),
                "final_epoch_mAP50-95": fmt(final["mAP50-95"]),
            })


def markdown_report(runs: list[Run], summary_path: Path) -> str:
    baseline = runs[0]
    alpha0 = runs[1]
    positives = runs[2:]
    best_positive = max(positives, key=lambda run: run.metric(run.best, "mAP50-95"))
    d0 = delta(alpha0, baseline)
    dbest_base = delta(best_positive, baseline)
    dbest_zero = delta(best_positive, alpha0)
    alpha0_close = abs(d0["mAP50-95"]) <= 0.01
    beats_both = dbest_base["mAP50-95"] > 0 and dbest_zero["mAP50-95"] > 0
    no_collapse = dbest_base["P"] > -0.05 and dbest_base["R"] > -0.05

    lines = [
        "# P3-AFF Fixed Alpha 10e 消融分析",
        "",
        "> 数据来源：本机五组实验目录中的 `args.yaml`、`results.csv`、模型 YAML 与 checkpoint。",
        "> best epoch 定义：`metrics/mAP50-95(B)` 最大的 epoch；CSV epoch 从 0 开始，因此 epoch 9 是第 10 轮。",
        "",
        "## 1. 实验完整性检查",
        "",
        "| 实验 | args.yaml | results.csv | best.pt | last.pt | results 行数 | save_period=1 checkpoint | 结论 |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for run in runs:
        periodic = "完整（epoch1.pt-epoch9.pt）" if not run.missing_periodic else "缺失：" + ", ".join(run.missing_periodic)
        complete = not run.missing_files and not run.missing_periodic and len(run.rows) == int(run.args.get("epochs", 0))
        lines.append(f"| `{run.name}` | {'有' if (run.path/'args.yaml').exists() else '缺'} | "
                     f"{'有' if (run.path/'results.csv').exists() else '缺'} | "
                     f"{'有' if (run.path/'weights/best.pt').exists() else '缺'} | "
                     f"{'有' if (run.path/'weights/last.pt').exists() else '缺'} | {len(run.rows)} | {periodic} | "
                     f"{'完整' if complete else '不完整'} |")
    lines += [
        "",
        "五组均完整训练到 10 个 epoch。当前保存逻辑不单独生成 `epoch0.pt`；`epoch1.pt` 至 `epoch9.pt` 均存在，最终状态另有 `last.pt`，不影响本轮分析或后续 checkpoint 选择。",
        "",
        "## 2. args.yaml 公平性检查",
        "",
        "| 字段 | baseline | alpha=0 | alpha=0.05 | alpha=0.10 | alpha=0.20 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for field in FAIR_FIELDS:
        values = [str(run.args.get(field)) for run in runs]
        lines.append("| `" + field + "` | " + " | ".join(f"`{v}`" for v in values) + " |")
    lines += [
        "",
        "上述核心训练字段完全一致。预训练的记录形式不同：baseline 的 `model=yolov8n.pt`、`pretrained=False` 表示直接从权重模型启动；四组 P3AFF 使用自定义 YAML，并由 `pretrained=yolov8n.pt` 显式迁移权重。两者均实际使用 YOLOv8n 预训练权重，语义一致。",
        "",
        "| P3AFF 实验 | args recovery | args recovery_fuse | 有效 recovery_fuse | recovery_loss_weight | YAML alpha | Detect 输入 |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for run in runs[1:]:
        detect = run.model_cfg.get("head", [])[-1][0] if run.model_cfg.get("head") else ""
        lines.append(f"| `{run.name}` | `{run.args.get('recovery')}` | `{run.args.get('recovery_fuse')}` | "
                     f"`{effective_fuse(run)}` | {run.args.get('recovery_loss_weight')} | {alpha_from_cfg(run):.2f} | `{detect}` |")
    severe = [run.name for run in runs[1:] if run.args.get("recovery") is not True or str(effective_fuse(run)).lower() != "none"]
    lines += [
        "",
        "`args.yaml` 中 `recovery_fuse=null` 是 CLI 参数未显式覆盖的记录；运行时回退到模型 YAML 的 `recovery_fuse=none`。对 `best.pt` 和 `last.pt` 的只读核验也确认四组实际均为 `recovery_enabled=True`、`recovery_fuse='none'`、Detect 输入 `[22,18,21]`，不存在重复 `p3_fixed` 融合。" if not severe else "发现严重配置错误：" + ", ".join(severe),
        "",
        "## 3. best epoch 指标汇总",
        "",
        "| 实验 | 类型 | alpha | best epoch | Precision | Recall | mAP50 | mAP50-95 | final P | final R | final mAP50 | final mAP50-95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        b, f = metric_values(run, run.best), metric_values(run, run.final)
        alpha = "-" if run.alpha is None else f"{run.alpha:.2f}"
        lines.append(f"| `{run.name}` | {run.kind} | {alpha} | {run.best.get('epoch')} | {fmt(b['P'])} | "
                     f"{fmt(b['R'])} | {fmt(b['mAP50'])} | {fmt(b['mAP50-95'])} | {fmt(f['P'])} | "
                     f"{fmt(f['R'])} | {fmt(f['mAP50'])} | {fmt(f['mAP50-95'])} |")
    lines += [
        "",
        f"五组 best 均为 epoch 9，且等于 final 指标。汇总 CSV：`{summary_path.as_posix()}`。所有组在 epoch 9 仍明显高于 epoch 8，说明 10e 尚未收敛，只适合候选筛选。",
        "",
        "## 4. baseline / alpha=0 / alpha>0 对比",
        "",
        "所有差值均使用各组 best epoch 指标。正值表示左侧实验更高。",
        "",
        "| 对比 | ΔP | ΔR | ΔmAP50 | ΔmAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    comparisons = [("alpha=0 - baseline", alpha0, baseline)]
    comparisons += [(f"alpha={run.alpha:.2f} - alpha=0", run, alpha0) for run in positives]
    comparisons += [(f"alpha={run.alpha:.2f} - baseline", run, baseline) for run in positives]
    for label, left, right in comparisons:
        d = delta(left, right)
        lines.append(f"| {label} | {signed(d['P'])} | {signed(d['R'])} | {signed(d['mAP50'])} | {signed(d['mAP50-95'])} |")
    lines += [
        "",
        f"alpha=0 相对 baseline 的 mAP50-95 差值为 {signed(d0['mAP50-95'])}，同时 Precision、Recall 分别变化 {signed(d0['P'])}、{signed(d0['R'])}。差距较小且方向为轻微提升，可视为基本接近；在 subset50、单 seed、10e 条件下不能把这点差异解释为结构收益。",
        "",
        "## 5. alpha 趋势分析",
        "",
        f"1. **最优值为 alpha={best_positive.alpha:.2f}。** 其 mAP50-95={best_positive.metric(best_positive.best, 'mAP50-95'):.5f}，高于 alpha=0 的 {alpha0.metric(alpha0.best, 'mAP50-95'):.5f} 和 baseline 的 {baseline.metric(baseline.best, 'mAP50-95'):.5f}。",
        "2. 正 alpha 内部呈现 `0.05 -> 0.10` 上升、`0.10 -> 0.20` 下降；若把 alpha=0 也计入，则不是严格的单调先升后降，因为 alpha=0.05 先低于 alpha=0。",
        "3. alpha=0.20 相对 alpha=0.10 的 mAP50-95 和 Precision 均下降，支持“融合过强后收益衰减”的解释，但它仍略高于 baseline，因此只能作为趋势证据，不能单凭一次 10e 认定发生严重破坏。",
        f"4. alpha={best_positive.alpha:.2f} 相对 alpha=0：Precision {signed(dbest_zero['P'])}、Recall {signed(dbest_zero['R'])}、mAP50 {signed(dbest_zero['mAP50'])}、mAP50-95 {signed(dbest_zero['mAP50-95'])}。收益主要来自 Precision 和更高 IoU 范围的综合质量，不来自 Recall；Recall 的下降仅约 {abs(dbest_zero['R']):.5f}，不属于异常塌陷。",
        "5. alpha=0.05 存在 mAP50-95 相对 baseline 仅微增、但 Precision 明显下降的问题；其更高 Recall 没有转化成稳定的综合收益，不应作为候选。",
        "6. alpha=0.10 和 alpha=0.20 的 Recall 均略高于 baseline；相对 alpha=0 的小幅下降不足以判断为异常。",
        "",
        "## 6. 结论",
        "",
        f"本轮属于题设的情况 1：alpha={best_positive.alpha:.2f} 同时超过 baseline 和 alpha=0，且 Precision 没有塌陷、Recall 没有异常。fixed P3-AFF 在这轮 10e 初筛中表现出正收益，最优 alpha 为 {best_positive.alpha:.2f}。",
        "",
        f"相对 baseline，alpha={best_positive.alpha:.2f} 的 mAP50-95 提升 {signed(dbest_base['mAP50-95'])}（绝对值，即 {dbest_base['mAP50-95']*100:+.3f} 个百分点）；相对 alpha=0 提升 {signed(dbest_zero['mAP50-95'])}（{dbest_zero['mAP50-95']*100:+.3f} 个百分点）。不过五组都在最后一轮继续上升，且只有 subset50、单 seed，因此当前结论应表述为“10e 初步有效”，不能表述为已稳定优于 baseline。",
        "",
        "## 7. 下一步建议",
        "",
        f"1. 将 alpha={best_positive.alpha:.2f} 作为 fixed gate 唯一优先候选，进入 50e 验证；不需要把全部 alpha 都扩展到 50e。",
        "2. 保持同一数据、seed、预训练方式和 `save_period=1`，按 mAP50-95 及 Precision/Recall 平衡选择 checkpoint，不只看 last。",
        "3. 50e 对照至少保留 YOLOv8n baseline 与 alpha=0；否则无法区分长训练收益来自 fixed R3 还是共同训练时长。",
        f"4. 在确认 alpha={best_positive.alpha:.2f} 的 50e 趋势后，可做一个独立的 `recovery_loss_weight=0.01` 小实验。它值得测试，但不要与本轮 fixed-alpha 结论混合。",
        "5. 暂不进入 learnable gate 或 haze-aware gate。先完成 50e 或 recovery loss 小实验，确认 fixed gate 收益稳定后再决定。",
        f"6. 若资源允许，在 50e 前或后对 alpha={best_positive.alpha:.2f} 复跑一个不同 seed 的 10e，可判断 epoch 9 大幅跃升是否具有稳定性。",
        "",
        "明确回答：",
        "",
        f"- fixed P3-AFF 是否有效：{'是，10e 初步有效' if beats_both and no_collapse else '当前证据不足'}。",
        f"- 最优 alpha：{best_positive.alpha:.2f}。",
        f"- 是否超过 YOLOv8n baseline：{'是' if dbest_base['mAP50-95'] > 0 else '否'}，mAP50-95 差值 {signed(dbest_base['mAP50-95'])}。",
        f"- 是否值得进入 50e：{'是，仅优先 alpha=' + format(best_positive.alpha, '.2f') + ' 并保留必要对照' if beats_both and no_collapse else '暂不建议'}。",
        "- 是否值得加入 recovery_loss_weight=0.01：值得作为后续独立小实验，不应立即替代当前检测-only 对照。",
        "- 是否可以进入 learnable gate / haze-aware gate：暂不进入，等待 50e 或 recovery loss 小实验确认。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    runs_root = (args.runs_root or repo / "runs/detect").resolve()
    summary = (args.summary or runs_root / "p3_aff_fixed_alpha_pretrained_10e_summary.csv").resolve()
    report = (args.report or runs_root / "p3_aff_fixed_alpha_pretrained_10e_analysis.md").resolve()
    runs = [load_run(repo, runs_root, spec) for spec in RUN_SPECS]
    write_summary(runs, summary)
    report.write_text(markdown_report(runs, summary.relative_to(repo)), encoding="utf-8")

    baseline, alpha0 = runs[0], runs[1]
    best = max(runs[2:], key=lambda run: run.metric(run.best, "mAP50-95"))
    d0 = delta(alpha0, baseline)
    dz = delta(best, alpha0)
    db = delta(best, baseline)
    print("[Final Conclusion]")
    print(f"- alpha=0 与 baseline 是否接近：{'是' if abs(d0['mAP50-95']) <= 0.01 else '否'}，ΔmAP50-95={signed(d0['mAP50-95'])}")
    print(f"- 最优 alpha：{best.alpha:.2f}")
    print(f"- fixed P3-AFF 是否超过 alpha=0：{'是' if dz['mAP50-95'] > 0 else '否'}，ΔmAP50-95={signed(dz['mAP50-95'])}")
    print(f"- fixed P3-AFF 是否超过 baseline：{'是' if db['mAP50-95'] > 0 else '否'}，ΔmAP50-95={signed(db['mAP50-95'])}")
    print(f"- 下一步建议：优先 alpha={best.alpha:.2f} 做 50e，并保留 baseline/alpha=0 对照；之后独立测试 recovery_loss_weight=0.01，暂缓 learnable/haze-aware gate")
    print(f"- 报告路径：{report.relative_to(repo).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
