"""Summarize P3-AFF fixed-alpha YOLO run directories.

This script is read-only except for the optional --output CSV path. It expects
Ultralytics run directories that contain args.yaml and results.csv.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - fallback for minimal environments
    yaml = None


RESULT_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def load_args(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    if yaml is not None:
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}

    data: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def load_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items() if k is not None})
        return rows


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_alpha(args: dict[str, Any], run_dir: Path) -> str:
    model = str(args.get("model", ""))
    name = str(args.get("name", run_dir.name))
    text = f"{model} {name}".lower()
    for tag, alpha in (("a000", "0.00"), ("fixed000", "0.00"), ("a005", "0.05"),
                       ("fixed005", "0.05"), ("a010", "0.10"), ("fixed010", "0.10"),
                       ("a020", "0.20"), ("fixed020", "0.20")):
        if tag in text:
            return alpha
    return ""


def summarize_run(run_dir: Path) -> dict[str, Any]:
    args = load_args(run_dir / "args.yaml")
    rows = load_results(run_dir / "results.csv")
    last = rows[-1] if rows else {}

    best = {}
    best_metric = None
    for row in rows:
        metric = to_float(row.get("metrics/mAP50-95(B)"))
        if metric is not None and (best_metric is None or metric > best_metric):
            best = row
            best_metric = metric

    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "name": args.get("name", run_dir.name),
        "model": args.get("model", ""),
        "data": args.get("data", ""),
        "alpha": infer_alpha(args, run_dir),
        "epochs_arg": args.get("epochs", ""),
        "batch": args.get("batch", ""),
        "imgsz": args.get("imgsz", ""),
        "seed": args.get("seed", ""),
        "recovery": args.get("recovery", ""),
        "recovery_fuse": args.get("recovery_fuse", ""),
        "recovery_loss_weight": args.get("recovery_loss_weight", ""),
        "final_epoch": last.get("epoch", ""),
        "best_epoch": best.get("epoch", ""),
        "has_results_csv": str((run_dir / "results.csv").exists()),
        "has_args_yaml": str((run_dir / "args.yaml").exists()),
        "has_best_pt": str((run_dir / "weights" / "best.pt").exists()),
        "has_last_pt": str((run_dir / "weights" / "last.pt").exists()),
        "has_recovery_shape_debug": str((run_dir / "recovery_shape_debug.txt").exists()),
    }

    for key in RESULT_KEYS:
        record[f"final_{key}"] = last.get(key, "")
        record[f"best_{key}"] = best.get(key, "")
    return record


def expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        matches = glob.glob(item)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(item))
    return sorted({p.resolve() for p in paths if p.exists() and p.is_dir()})


def write_csv(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) if records else ["run_dir"]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_table(records: list[dict[str, Any]]) -> None:
    if not records:
        print("No run directories found.")
        return
    cols = ["name", "alpha", "final_epoch", "best_metrics/mAP50(B)", "best_metrics/mAP50-95(B)",
            "has_best_pt", "has_last_pt"]
    print("\t".join(cols))
    for record in records:
        print("\t".join(str(record.get(col, "")) for col in cols))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run directories or glob patterns, e.g. runs/detect/5beta_*_10e")
    parser.add_argument("--output", "-o", type=Path, help="Optional output CSV path")
    args = parser.parse_args()

    run_dirs = expand_inputs(args.runs)
    records = [summarize_run(path) for path in run_dirs]

    if args.output:
        write_csv(records, args.output)
        print(f"Wrote {len(records)} rows to {args.output}")
    else:
        print_table(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
