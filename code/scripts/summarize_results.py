import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SUMMARY_FIELDS = [
    "run",
    "model",
    "dataset",
    "img_size",
    "epochs",
    "batch_size",
    "seed",
    "lr",
    "weight_decay",
    "drop_path",
    "attention",
    "no_large_kernel",
    "no_grn",
    "no_gate",
    "mixup",
    "cutmix",
    "random_erasing",
    "gram_weight",
    "parameters",
    "flops_estimate",
    "best_epoch",
    "best_val_acc1",
    "best_val_acc5",
    "best_val_loss",
    "final_train_acc1",
    "final_train_loss",
    "log_rows",
    "status",
]


GROUP_FIELDS = [
    "group",
    "runs",
    "model",
    "img_size",
    "lr",
    "weight_decay",
    "drop_path",
    "mixup",
    "cutmix",
    "gram_weight",
    "mean_best_val_acc1",
    "std_best_val_acc1",
    "max_best_val_acc1",
    "mean_best_val_acc5",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize GLF-MobileViT runs.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(value, default: Optional[float] = None) -> Optional[float]:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default: Optional[int] = None) -> Optional[int]:
    value = to_float(value, None)
    if value is None:
        return default
    return int(value)


def read_log(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def best_log_row(rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    valid = [row for row in rows if to_float(row.get("val_acc1")) is not None]
    if not valid:
        return rows[-1] if rows else None
    return max(valid, key=lambda row: to_float(row.get("val_acc1"), -math.inf))


def summarize_run(run_dir: Path) -> Dict:
    config = read_json(run_dir / "config.json")
    model_summary = read_json(run_dir / "model_summary.json")
    rows = read_log(run_dir / "log.csv")
    best = best_log_row(rows)
    last = rows[-1] if rows else {}
    status = "complete" if (run_dir / "best.pt").exists() else "partial"
    if not rows:
        status = "no_log"

    summary = {
        "run": run_dir.name,
        "model": config.get("model", ""),
        "dataset": config.get("dataset", ""),
        "img_size": config.get("img_size", ""),
        "epochs": config.get("epochs", ""),
        "batch_size": config.get("batch_size", ""),
        "seed": config.get("seed", ""),
        "lr": config.get("lr", ""),
        "weight_decay": config.get("weight_decay", ""),
        "drop_path": config.get("drop_path", ""),
        "attention": config.get("attention", ""),
        "no_large_kernel": config.get("no_large_kernel", ""),
        "no_grn": config.get("no_grn", ""),
        "no_gate": config.get("no_gate", ""),
        "mixup": config.get("mixup", ""),
        "cutmix": config.get("cutmix", ""),
        "random_erasing": config.get("random_erasing", ""),
        "gram_weight": config.get("gram_weight", ""),
        "parameters": model_summary.get("parameters", ""),
        "flops_estimate": model_summary.get("flops_estimate", ""),
        "best_epoch": best.get("epoch", "") if best else "",
        "best_val_acc1": best.get("val_acc1", "") if best else "",
        "best_val_acc5": best.get("val_acc5", "") if best else "",
        "best_val_loss": best.get("val_loss", "") if best else "",
        "final_train_acc1": last.get("train_acc1", ""),
        "final_train_loss": last.get("train_loss", ""),
        "log_rows": len(rows),
        "status": status,
    }
    return summary


def group_key(row: Dict) -> str:
    keys = [
        "model",
        "dataset",
        "img_size",
        "lr",
        "weight_decay",
        "drop_path",
        "attention",
        "no_large_kernel",
        "no_grn",
        "no_gate",
        "mixup",
        "cutmix",
        "random_erasing",
        "gram_weight",
    ]
    return "|".join(f"{key}={row.get(key, '')}" for key in keys)


def write_csv(path: Path, rows: List[Dict], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize_groups(rows: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for row in rows:
        if to_float(row.get("best_val_acc1")) is not None:
            groups[group_key(row)].append(row)

    group_rows = []
    for key, items in groups.items():
        acc1 = [to_float(item["best_val_acc1"]) for item in items]
        acc5 = [to_float(item["best_val_acc5"], 0.0) for item in items]
        first = items[0]
        group_rows.append(
            {
                "group": key,
                "runs": len(items),
                "model": first.get("model", ""),
                "img_size": first.get("img_size", ""),
                "lr": first.get("lr", ""),
                "weight_decay": first.get("weight_decay", ""),
                "drop_path": first.get("drop_path", ""),
                "mixup": first.get("mixup", ""),
                "cutmix": first.get("cutmix", ""),
                "gram_weight": first.get("gram_weight", ""),
                "mean_best_val_acc1": statistics.mean(acc1),
                "std_best_val_acc1": statistics.stdev(acc1) if len(acc1) > 1 else 0.0,
                "max_best_val_acc1": max(acc1),
                "mean_best_val_acc5": statistics.mean(acc5),
            }
        )
    group_rows.sort(key=lambda row: row["mean_best_val_acc1"], reverse=True)
    return group_rows


def svg_bar(path: Path, labels: List[str], values: List[float], title: str, ylabel: str) -> None:
    if not labels:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(900, 90 * len(labels))
    height = 560
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 60, 150
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_v = max(values) if values else 1.0
    min_v = min(values) if values else 0.0
    floor = max(0.0, min_v - max(1.0, (max_v - min_v) * 0.2))
    ceil = max_v + max(1.0, (max_v - min_v) * 0.2)
    bar_w = plot_w / len(labels) * 0.68
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20" font-family="Arial">{html.escape(title)}</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-size="13" font-family="Arial">{html.escape(ylabel)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top+plot_h}" x2="{margin_left+plot_w}" y2="{margin_top+plot_h}" stroke="#444"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}" stroke="#444"/>',
    ]
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + (idx + 0.16) * plot_w / len(labels)
        bar_h = (value - floor) / max(ceil - floor, 1e-9) * plot_h
        y = margin_top + plot_h - bar_h
        chunks.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#4C78A8"/>')
        chunks.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="11" font-family="Arial">{value:.2f}</text>')
        chunks.append(
            f'<text x="{x + bar_w/2:.1f}" y="{margin_top + plot_h + 18}" text-anchor="end" '
            f'transform="rotate(-45 {x + bar_w/2:.1f} {margin_top + plot_h + 18})" '
            f'font-size="11" font-family="Arial">{html.escape(label[:42])}</text>'
        )
    chunks.append("</svg>")
    path.write_text("\n".join(chunks), encoding="utf-8")


def svg_curve(path: Path, rows: List[Dict[str, str]], title: str) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    series = {
        "train_acc1": "#4C78A8",
        "val_acc1": "#F58518",
    }
    points = {}
    for name in series:
        pts = []
        for row in rows:
            epoch = to_float(row.get("epoch"))
            value = to_float(row.get(name))
            if epoch is not None and value is not None:
                pts.append((epoch, value))
        points[name] = pts
    all_pts = [pt for pts in points.values() for pt in pts]
    if not all_pts:
        return
    min_x, max_x = min(x for x, _ in all_pts), max(x for x, _ in all_pts)
    min_y, max_y = 0.0, max(100.0, max(y for _, y in all_pts))
    width, height = 900, 520
    ml, mr, mt, mb = 70, 30, 55, 65
    pw, ph = width - ml - mr, height - mt - mb

    def xy(epoch, value):
        x = ml + (epoch - min_x) / max(max_x - min_x, 1e-9) * pw
        y = mt + ph - (value - min_y) / max(max_y - min_y, 1e-9) * ph
        return x, y

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20" font-family="Arial">{html.escape(title)}</text>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#444"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#444"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="13" font-family="Arial">epoch</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-size="13" font-family="Arial">accuracy (%)</text>',
    ]
    legend_x = ml + 20
    for idx, (name, color) in enumerate(series.items()):
        pts = points[name]
        if not pts:
            continue
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in [xy(a, b) for a, b in pts])
        chunks.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{poly}"/>')
        chunks.append(f'<rect x="{legend_x}" y="{mt + 15 + idx * 22}" width="14" height="14" fill="{color}"/>')
        chunks.append(f'<text x="{legend_x + 20}" y="{mt + 27 + idx * 22}" font-size="13" font-family="Arial">{html.escape(name)}</text>')
    chunks.append("</svg>")
    path.write_text("\n".join(chunks), encoding="utf-8")


def write_conclusions(path: Path, rows: List[Dict], group_rows: List[Dict], charts_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = [row for row in rows if to_float(row.get("best_val_acc1")) is not None]
    complete.sort(key=lambda row: to_float(row["best_val_acc1"], -math.inf), reverse=True)
    best = complete[0] if complete else None
    robust = next((row for row in group_rows if int(row["runs"]) >= 2), None)
    lines = [
        "# 参数调优与可靠性结论",
        "",
        "本文件由 `scripts/summarize_results.py` 自动生成，用于课程设计中“模型参数调优、给出合理可靠结论”的实验记录。",
        "",
        "## 调优逻辑",
        "",
        "1. 先用 `glf_small + 160px + 80 epochs` 作为 proxy setting 搜索学习率、权重衰减和 DropPath。",
        "2. 再用 `glf_base + 224px + 120 epochs` 比较 Mixup/CutMix 和 Gram consistency 等正则配置。",
        "3. 最后用 `glf_base + 224px + 200 epochs` 对最终配置跑多个 seed，报告均值和标准差，避免只依据单次实验下结论。",
        "",
    ]
    if best:
        lines.extend(
            [
                "## 当前最佳单次实验",
                "",
                f"- Run: `{best['run']}`",
                f"- Model: `{best['model']}`",
                f"- Top-1: `{to_float(best['best_val_acc1'], 0.0):.2f}%`",
                f"- Top-5: `{to_float(best['best_val_acc5'], 0.0):.2f}%`",
                f"- Epoch: `{best['best_epoch']}`",
                f"- lr/weight_decay/drop_path: `{best['lr']} / {best['weight_decay']} / {best['drop_path']}`",
                f"- mixup/cutmix/gram_weight: `{best['mixup']} / {best['cutmix']} / {best['gram_weight']}`",
                "",
            ]
        )
    if robust:
        lines.extend(
            [
                "## 多 seed 可靠性结论",
                "",
                f"- 最可靠配置包含 `{robust['runs']}` 次重复。",
                f"- Mean Top-1: `{float(robust['mean_best_val_acc1']):.2f}%`",
                f"- Std Top-1: `{float(robust['std_best_val_acc1']):.2f}`",
                f"- Max Top-1: `{float(robust['max_best_val_acc1']):.2f}%`",
                "",
                "报告中应优先引用多 seed 均值与标准差；单次最高值只作为补充。",
                "",
            ]
        )
    lines.extend(
        [
            "## 输出文件",
            "",
            "- `summary.csv`：每个 run 的最优 epoch、Top-1/Top-5、loss、超参数、参数量、FLOPs。",
            "- `group_summary.csv`：按超参数组合聚合后的均值、标准差和最大值。",
            f"- `{charts_dir.name}/top1_by_run.svg`：Top-1 单次实验对比图。",
            f"- `{charts_dir.name}/top1_by_group.svg`：按配置聚合的 Top-1 对比图。",
            f"- `{charts_dir.name}/accuracy_curve_best.svg`：当前最佳 run 的训练/验证准确率曲线。",
            "",
        ]
    )
    if not complete:
        lines.append("> 当前没有发现可汇总的 `log.csv`，请先运行训练脚本。")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    charts_dir = output_dir / "charts"
    run_dirs = sorted([p for p in runs_dir.glob("*") if p.is_dir()])
    rows = [summarize_run(path) for path in run_dirs]
    rows.sort(key=lambda row: to_float(row.get("best_val_acc1"), -math.inf), reverse=True)
    groups = summarize_groups(rows)

    write_csv(output_dir / "summary.csv", rows, SUMMARY_FIELDS)
    write_csv(output_dir / "group_summary.csv", groups, GROUP_FIELDS)

    top_rows = [row for row in rows if to_float(row.get("best_val_acc1")) is not None][: args.top_k]
    svg_bar(
        charts_dir / "top1_by_run.svg",
        [row["run"] for row in top_rows],
        [to_float(row["best_val_acc1"], 0.0) for row in top_rows],
        "Best Top-1 by Run",
        "Top-1 (%)",
    )
    top_groups = groups[: args.top_k]
    svg_bar(
        charts_dir / "top1_by_group.svg",
        [f"{row['model']}_lr{row['lr']}_wd{row['weight_decay']}_g{row['gram_weight']}" for row in top_groups],
        [float(row["mean_best_val_acc1"]) for row in top_groups],
        "Mean Best Top-1 by Hyperparameter Group",
        "Mean Top-1 (%)",
    )

    best_run = top_rows[0]["run"] if top_rows else None
    if best_run:
        best_log = read_log(runs_dir / best_run / "log.csv")
        svg_curve(charts_dir / "accuracy_curve_best.svg", best_log, f"Accuracy Curve: {best_run}")

    write_conclusions(output_dir / "conclusions.md", rows, groups, charts_dir)
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'group_summary.csv'}")
    print(f"Wrote {output_dir / 'conclusions.md'}")


if __name__ == "__main__":
    main()
