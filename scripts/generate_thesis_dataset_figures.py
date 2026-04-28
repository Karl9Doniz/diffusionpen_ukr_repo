#!/usr/bin/env python3
"""
Generate data-driven thesis figures for the dataset chapter.

Outputs:
  - letter_frequency_pre_post.png
  - word_length_hist_balanced.png
  - pipeline_counts_ulcleannaf_v1.png (if summary JSON is provided)
  - training_curves_v3_v7_v8.png (if W&B curves are enabled and available)
"""

from __future__ import annotations

import argparse
import csv
import json
import importlib
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


OVERSAMPLED_LETTERS = {"ф", "щ", "Щ", "Є", "ґ", "ї"}


def read_transcriptions(meta_file: Path) -> list[str]:
    with meta_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [row["transcription"] for row in reader if row.get("transcription")]


def char_frequency(words: list[str]) -> Counter:
    freq: Counter = Counter()
    for word in words:
        for ch in word:
            if ch.isalpha():
                freq[ch] += 1
    return freq


def word_length_buckets(words: list[str]) -> tuple[list[str], list[int]]:
    buckets = [
        ("1", lambda n: n == 1),
        ("2", lambda n: n == 2),
        ("3", lambda n: n == 3),
        ("4-5", lambda n: 4 <= n <= 5),
        ("6-9", lambda n: 6 <= n <= 9),
        ("10+", lambda n: n >= 10),
    ]
    counts = [0] * len(buckets)
    for w in words:
        n = len(w)
        for i, (_, rule) in enumerate(buckets):
            if rule(n):
                counts[i] += 1
                break
    return [b[0] for b in buckets], counts


def plot_letter_frequency(pre: Counter, post: Counter, out_file: Path, dpi: int) -> None:
    chars = sorted(set(pre.keys()) | set(post.keys()), key=lambda c: post[c], reverse=True)
    x = list(range(len(chars)))
    pre_vals = [pre[c] for c in chars]
    post_vals = [post[c] for c in chars]

    fig, ax = plt.subplots(figsize=(18, 7))
    width = 0.44
    ax.bar([i - width / 2 for i in x], pre_vals, width=width, label="Before balancing", color="#93a1a1")
    ax.bar([i + width / 2 for i in x], post_vals, width=width, label="After balancing", color="#268bd2")

    ax.set_title("Letter Frequency Before/After Balancing")
    ax.set_xlabel("Character")
    ax.set_ylabel("Occurrences")
    ax.set_xticks(x)
    ax.set_xticklabels(chars, rotation=90)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    for tick in ax.get_xticklabels():
        if tick.get_text() in OVERSAMPLED_LETTERS:
            tick.set_color("#dc322f")
            tick.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_word_lengths(words: list[str], out_file: Path, dpi: int) -> None:
    labels, counts = word_length_buckets(words)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts, color="#2aa198")
    ax.set_title("Word Length Distribution (Balanced Metafile)")
    ax.set_xlabel("Word length bucket")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.25)

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def plot_pipeline_counts(summary: dict, out_file: Path, dpi: int) -> None:
    counts = summary.get("counts", {})
    extended_rows = None
    balanced_rows = None
    extended_label = "Extended filtered"

    for key, label in [
        ("metafile_extended_trocr_local3", "Extended filtered + TrOCR"),
        ("metafile_extended_trocr", "Extended filtered + TrOCR"),
        ("metafile_extended_skip_trocr", "Extended filtered"),
    ]:
        value = counts.get(key, {}).get("rows")
        if isinstance(value, int):
            extended_rows = value
            extended_label = label
            break

    for key in [
        "metafile_extended_balanced_trocr_local3",
        "metafile_extended_balanced_trocr",
        "metafile_extended_balanced_skip_trocr",
    ]:
        value = counts.get(key, {}).get("rows")
        if isinstance(value, int):
            balanced_rows = value
            break

    stages = [
        ("Cleaned lines", counts.get("line_images_cleaned")),
        ("Segmented words", counts.get("word_images_segmented")),
        (extended_label, extended_rows),
        ("Balanced output", balanced_rows),
    ]
    stages = [(n, v) for n, v in stages if isinstance(v, int)]
    if not stages:
        return

    names = [n for n, _ in stages]
    values = [v for _, v in stages]

    fig, ax = plt.subplots(figsize=(11, 3.8))
    y = list(range(len(names)))
    ax.barh(y, values, color=["#6c71c4", "#268bd2", "#2aa198", "#859900"])
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Samples")
    ax.set_title("ULCleanNAF-v1 Pipeline Counts")
    ax.grid(axis="x", alpha=0.25)

    for yi, val in zip(y, values):
        ax.text(val, yi, f"  {val:,}", va="center", ha="left", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def _import_wandb_sdk():
    """
    Import the real wandb package even when running from the repository root
    that also contains a local ./wandb folder.
    """
    project_root = Path(__file__).resolve().parents[1]
    original_sys_path = list(sys.path)
    try:
        cleaned_path = []
        for p in sys.path:
            resolved = Path(p or ".").resolve()
            if resolved == project_root:
                continue
            cleaned_path.append(p)
        sys.path = cleaned_path
        wb = importlib.import_module("wandb")
        if not hasattr(wb, "Api"):
            raise RuntimeError("Imported wandb module does not expose Api()")
        return wb
    finally:
        sys.path = original_sys_path


def fetch_wandb_curve(entity: str, project: str, run_id: str, metric_key: str) -> list[tuple[int, float]]:
    wb = _import_wandb_sdk()
    api = wb.Api(timeout=60)
    run = api.run(f"{entity}/{project}/{run_id}")

    by_epoch: dict[int, float] = {}
    for row in run.scan_history(keys=["epoch", metric_key]):
        epoch = row.get("epoch")
        metric = row.get(metric_key)
        if epoch is None or metric is None:
            continue
        try:
            e = int(epoch)
            m = float(metric)
        except (TypeError, ValueError):
            continue
        by_epoch[e] = m

    return sorted(by_epoch.items(), key=lambda x: x[0])


def plot_training_curves(curves: dict[str, list[tuple[int, float]]], out_file: Path, dpi: int, metric_label: str) -> None:
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {
        "v3 (Clean 67K)": "#586e75",
        "v7 (CC-Clean 99K)": "#268bd2",
        "v8 (CC-Clean 99K, 2 res blocks)": "#dc322f",
    }

    plotted_any = False
    for label, points in curves.items():
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = colors.get(label)
        ax.plot(xs, ys, linewidth=2.0, label=label, color=color)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_title("Validation MSE Curves for Representative Runs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--extended_meta", type=Path, required=True, help="Path to METAFILE_extended.tsv")
    p.add_argument("--balanced_meta", type=Path, required=True, help="Path to METAFILE_extended_balanced.tsv")
    p.add_argument("--summary_json", type=Path, default=None, help="Optional pipeline summary JSON")
    p.add_argument("--output_dir", type=Path, required=True, help="Output directory for figure PNGs")
    p.add_argument("--disable_training_curves", action="store_true", help="Skip W&B training curve plot")
    p.add_argument("--wandb_entity", type=str, default="andrei-agitolyev-ukrainian-catholic-university")
    p.add_argument("--wandb_project", type=str, default="DiffusionPen")
    p.add_argument("--run_v3", type=str, default="rbwlfdtn", help="W&B run id for v3 curve")
    p.add_argument("--run_v7", type=str, default="nwosrrjr", help="W&B run id for v7 curve")
    p.add_argument("--run_v8", type=str, default="qct1wr7c", help="W&B run id for v8 curve")
    p.add_argument("--curve_metric", type=str, default="val/mse", help="Metric key to plot from W&B history")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    words_pre = read_transcriptions(args.extended_meta)
    words_post = read_transcriptions(args.balanced_meta)

    pre_freq = char_frequency(words_pre)
    post_freq = char_frequency(words_post)

    plot_letter_frequency(pre_freq, post_freq, args.output_dir / "letter_frequency_pre_post.png", args.dpi)
    plot_word_lengths(words_post, args.output_dir / "word_length_hist_balanced.png", args.dpi)

    if args.summary_json and args.summary_json.exists():
        with args.summary_json.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        plot_pipeline_counts(summary, args.output_dir / "pipeline_counts_ulcleannaf_v1.png", args.dpi)

    if not args.disable_training_curves:
        curves: dict[str, list[tuple[int, float]]] = {}
        run_map = {
            "v3 (Clean 67K)": args.run_v3,
            "v7 (CC-Clean 99K)": args.run_v7,
            "v8 (CC-Clean 99K, 2 res blocks)": args.run_v8,
        }
        for label, run_id in run_map.items():
            try:
                points = fetch_wandb_curve(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    run_id=run_id,
                    metric_key=args.curve_metric,
                )
                curves[label] = points
                print(f"Loaded {len(points)} points for {label} from run {run_id}")
            except Exception as exc:
                curves[label] = []
                print(f"Warning: could not load curve for {label} ({run_id}): {exc}")
        plot_training_curves(
            curves=curves,
            out_file=args.output_dir / "training_curves_v3_v7_v8.png",
            dpi=args.dpi,
            metric_label=args.curve_metric,
        )

    print("Generated figures in", args.output_dir)


if __name__ == "__main__":
    main()
