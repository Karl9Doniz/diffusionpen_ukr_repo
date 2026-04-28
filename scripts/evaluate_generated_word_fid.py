#!/usr/bin/env python3
"""
Empirical image-distribution evaluation for generated Ukrainian word images.

Phase 1B:
- build a deterministic seen-writer manifest of real word crops
- copy/standardize sampled real images to a FID-ready folder
- generate matched synthetic images for the same writer/text pairs
- compute FID between the two folders with pytorch-fid
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from evaluate_generated_word_cer import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_EVAL_VERSION,
    DEFAULT_META_FILE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SD_PATH,
    DEFAULT_STYLE_PATH,
    DEFAULT_TROCR_MODEL,
    checkpoint_id,
    load_models,
    meta_id,
    precompute_style_payloads,
    read_records,
    stable_seed,
)
from generate_sentence import (
    build_writer_id_map,
    encode_text_context,
    generate_single_word,
    load_style_images,
    set_global_seed,
)


DEFAULT_FID_VERSION = "fid_v1_seen_words"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID for generated Ukrainian word images.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--meta_file", type=Path, default=DEFAULT_META_FILE)
    parser.add_argument("--style_path", type=Path, default=DEFAULT_STYLE_PATH)
    parser.add_argument("--stable_dif_path", type=Path, default=DEFAULT_SD_PATH)
    parser.add_argument("--trocr_model", type=str, default=DEFAULT_TROCR_MODEL)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--trocr_device", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--img_height", type=int, default=64)
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--text_max_len", type=int, default=40)
    parser.add_argument("--emb_dim", type=int, default=320)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--eval_version", type=str, default=DEFAULT_FID_VERSION)
    parser.add_argument("--sample_count", type=int, default=5000)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--manifest_path", type=Path, default=None)
    parser.add_argument("--manifest_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--audit_count", type=int, default=24)
    parser.add_argument("--fid_batch_size", type=int, default=32)
    return parser.parse_args()


def ensure_canvas(img: Image.Image, height: int, width: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if h <= 0:
        raise ValueError("Invalid image height.")
    new_w = max(1, int(round(w * height / float(h))))
    img = img.resize((new_w, height), Image.Resampling.BILINEAR)
    if new_w < width:
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(img, (0, 0))
        return canvas
    if new_w > width:
        return img.resize((width, height), Image.Resampling.BILINEAR)
    return img


def build_writer_records(records) -> Dict[str, List]:
    writer_records: Dict[str, List] = defaultdict(list)
    for record in records:
        writer_records[record.writer_id].append(record)
    return writer_records


def choose_real_samples(records, sample_count: int, eval_version: str) -> List[Dict[str, str]]:
    writer_records = build_writer_records(records)
    writer_ids = sorted(writer_records)
    writer_count = len(writer_ids)
    if writer_count == 0:
        raise ValueError("No writer records found.")

    base = sample_count // writer_count
    remainder = sample_count % writer_count

    rows: List[Dict[str, str]] = []
    for idx, writer_id in enumerate(writer_ids):
        quota = base + (1 if idx < remainder else 0)
        available = writer_records[writer_id]
        if len(available) < quota:
            raise ValueError(f"Writer {writer_id} has only {len(available)} usable records, need {quota}.")

        order = sorted(
            available,
            key=lambda rec: stable_seed(eval_version, writer_id, rec.filename),
        )
        chosen = order[:quota]

        writer_filenames = [rec.filename for rec in available]
        for record in chosen:
            style_refs = [fname for fname in writer_filenames if fname != record.filename][:5]
            if not style_refs:
                style_refs = [record.filename]
            while len(style_refs) < 5:
                style_refs.append(style_refs[0])
            rows.append(
                {
                    "writer_id": record.writer_id,
                    "filename": record.filename,
                    "target_text": record.transcription,
                    "target_norm": record.normalized,
                    "seed": str(stable_seed(eval_version, record.writer_id, record.filename)),
                    "style_ref_filenames": "|".join(style_refs),
                    "length_bucket": record.length_bucket,
                    "checkpoint_id": checkpoint_id(DEFAULT_CHECKPOINT),
                    "meta_file_id": meta_id(DEFAULT_META_FILE),
                }
            )

    rows.sort(key=lambda row: (row["writer_id"], row["filename"]))
    if len(rows) != sample_count:
        raise AssertionError(f"Expected {sample_count} rows, got {len(rows)}")
    return rows


def write_manifest(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fieldnames = [
        "writer_id",
        "filename",
        "target_text",
        "target_norm",
        "seed",
        "style_ref_filenames",
        "length_bucket",
        "checkpoint_id",
        "meta_file_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def select_audit_indices(rows: Sequence[Dict[str, str]], audit_count: int) -> set[int]:
    if audit_count <= 0:
        return set()
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[row["length_bucket"]].append(idx)
    selected: List[int] = []
    per_group = max(1, audit_count // max(len(groups), 1))
    for bucket in sorted(groups):
        selected.extend(groups[bucket][:per_group])
    if len(selected) < audit_count:
        seen = set(selected)
        for idx in range(len(rows)):
            if len(selected) >= audit_count:
                break
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
    return set(selected[:audit_count])


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def save_pair_audit(path: Path, real_image: Image.Image, gen_image: Image.Image, target_text: str, writer_id: str) -> None:
    real_img = real_image.convert("RGB")
    gen_img = gen_image.convert("RGB")
    font = load_font(14)
    pad = 12
    gap = 14
    label_h = 44
    width = real_img.width + gen_img.width + gap + pad * 2
    height = max(real_img.height, gen_img.height) + label_h + pad * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(real_img, (pad, pad))
    canvas.paste(gen_img, (pad + real_img.width + gap, pad))
    draw.text((pad, real_img.height + pad + 4), f"real | writer {writer_id}", fill=(20, 20, 20), font=font)
    draw.text((pad + real_img.width + gap, gen_img.height + pad + 4), "generated", fill=(20, 20, 20), font=font)
    draw.text((pad, real_img.height + pad + 22), f"text: {target_text}", fill=(20, 20, 20), font=font)
    canvas.save(path)


def build_audit_contact_sheet(audit_dir: Path, output_path: Path) -> None:
    files = sorted(audit_dir.glob("*.png"))
    if not files:
        return
    thumbs = [Image.open(path).convert("RGB") for path in files]
    columns = 4
    rows = math.ceil(len(thumbs) / columns)
    gap = 12
    cell_w = max(img.width for img in thumbs)
    cell_h = max(img.height for img in thumbs)
    canvas = Image.new(
        "RGB",
        (columns * cell_w + (columns + 1) * gap, rows * cell_h + (rows + 1) * gap),
        "white",
    )
    for idx, img in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        x = gap + col * (cell_w + gap)
        y = gap + row * (cell_h + gap)
        canvas.paste(img, (x, y))
    canvas.save(output_path)


def compute_fid_score(real_dir: Path, generated_dir: Path, device: str, batch_size: int) -> tuple[float, str]:
    cmd = [
        sys.executable,
        "-m",
        "pytorch_fid",
        str(real_dir),
        str(generated_dir),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    fid_value = None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if "FID:" in line:
            fid_value = float(line.split("FID:")[-1].strip())
            break
    if fid_value is None:
        raise RuntimeError(f"Could not parse FID from output:\n{text}")
    return fid_value, text


def plot_sample_counts(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    by_bucket: Dict[str, int] = defaultdict(int)
    for row in rows:
        by_bucket[row["length_bucket"]] += 1
    labels = ["1-3", "4-6", "7-9", "10+"]
    values = [by_bucket.get(label, 0) for label in labels]
    plt.figure(figsize=(7.2, 4.2))
    bars = plt.bar(labels, values, color="#4C72B0")
    plt.ylabel("Samples")
    plt.xlabel("Word-length bucket")
    plt.title("FID benchmark sample distribution")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.01, str(value), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.eval_version}_{timestamp}"
    output_dir = args.output_root / run_name
    real_dir = output_dir / "real"
    generated_dir = output_dir / "generated"
    artifacts_dir = output_dir / "artifacts"
    figures_dir = output_dir / "figures"
    audit_dir = output_dir / "audit"
    for path in [output_dir, real_dir, generated_dir, artifacts_dir, figures_dir, audit_dir]:
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest_path or artifacts_dir / "manifest.tsv"

    if not args.eval_only:
        records = read_records(args.meta_file)
        manifest_rows = choose_real_samples(records, args.sample_count, args.eval_version)
        write_manifest(manifest_path, manifest_rows)
        print(f"Manifest written to {manifest_path}")

    if args.manifest_only:
        return

    manifest_rows = read_manifest(manifest_path)
    writer_ids = sorted({row["writer_id"] for row in manifest_rows})
    writer_id_map = build_writer_id_map(str(args.meta_file))
    writer_indices = [writer_id_map[writer_id] for writer_id in writer_ids]
    style_ref_override = {
        writer_id: row["style_ref_filenames"].split("|")
        for writer_id, row in {row["writer_id"]: row for row in manifest_rows}.items()
    }

    models = load_models(args)
    style_refs, _ = load_style_images(
        str(args.dataset_root),
        str(args.meta_file),
        writer_indices,
        writer_id_map,
        img_height=args.img_height,
        img_width=args.img_width,
        style_ref_override=style_ref_override,
    )
    style_payloads = precompute_style_payloads(
        style_refs=style_refs,
        style_extractor=models["style_extractor"],
        device=models["device"],
        img_height=args.img_height,
        img_width=args.img_width,
    )
    null_context = (
        encode_text_context(
            unet=models["unet"],
            tokenizer=models["tokenizer"],
            texts=[""],
            device=models["device"],
            text_max_len=args.text_max_len,
        )
        if args.cfg_scale > 1.0
        else None
    )

    words_root = args.dataset_root / "words" / "words"
    audit_indices = select_audit_indices(manifest_rows, args.audit_count)
    start_time = time.time()

    pair_rows: List[Dict[str, str]] = []
    for idx, row in enumerate(tqdm(manifest_rows, desc="Preparing FID folders")):
        real_src = words_root / row["filename"]
        real_image = ensure_canvas(Image.open(real_src), args.img_height, args.img_width)
        pair_name = f"{idx:05d}_{row['writer_id']}_{row['filename'].replace('.png', '')}.png"
        real_path = real_dir / pair_name
        real_image.save(real_path)

        writer_idx = writer_id_map[row["writer_id"]]
        style_payload = style_payloads[writer_idx]
        set_global_seed(int(row["seed"]))
        text_context = encode_text_context(
            unet=models["unet"],
            tokenizer=models["tokenizer"],
            texts=[row["target_text"]],
            device=models["device"],
            text_max_len=args.text_max_len,
        )
        gen_image = generate_single_word(
            word=row["target_text"],
            unet=models["unet"],
            vae=models["vae"],
            style_extractor=models["style_extractor"],
            tokenizer=models["tokenizer"],
            noise_scheduler=models["noise_scheduler"],
            style_ref=style_refs[writer_idx],
            writer_idx=writer_idx,
            device=models["device"],
            cfg_scale=args.cfg_scale,
            img_height=args.img_height,
            img_width=args.img_width,
            text_max_len=args.text_max_len,
            style_features=style_payload["style_features"],
            label_tensor=style_payload["label_tensor"],
            null_context=null_context,
        ).convert("RGB")
        gen_path = generated_dir / pair_name
        gen_image.save(gen_path)

        pair_rows.append(
            {
                "index": str(idx),
                "writer_id": row["writer_id"],
                "filename": row["filename"],
                "target_text": row["target_text"],
                "target_norm": row["target_norm"],
                "length_bucket": row["length_bucket"],
                "seed": row["seed"],
                "real_path": str(real_path),
                "generated_path": str(gen_path),
                "style_ref_filenames": row["style_ref_filenames"],
            }
        )

        if idx in audit_indices:
            save_pair_audit(audit_dir / pair_name, real_image, gen_image, row["target_text"], row["writer_id"])

    generation_runtime = time.time() - start_time

    pair_manifest_path = artifacts_dir / "pair_manifest.csv"
    with pair_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    fid_start = time.time()
    fid_value, fid_stdout = compute_fid_score(real_dir, generated_dir, args.device, args.fid_batch_size)
    fid_runtime = time.time() - fid_start

    (artifacts_dir / "fid_stdout.txt").write_text(fid_stdout, encoding="utf-8")
    build_audit_contact_sheet(audit_dir, audit_dir / "audit_contact_sheet.png")
    plot_sample_counts(manifest_rows, figures_dir / "fid_sample_distribution.png")

    summary = {
        "total_samples": len(manifest_rows),
        "unique_writers": len(writer_ids),
        "fid": fid_value,
        "checkpoint": str(args.checkpoint),
        "checkpoint_id": checkpoint_id(args.checkpoint),
        "meta_file": str(args.meta_file),
        "meta_file_id": meta_id(args.meta_file),
        "dataset_root": str(args.dataset_root),
        "cfg_scale": args.cfg_scale,
        "eval_version": args.eval_version,
        "generation_runtime_sec": generation_runtime,
        "fid_runtime_sec": fid_runtime,
        "total_runtime_sec": generation_runtime + fid_runtime,
        "real_dir": str(real_dir),
        "generated_dir": str(generated_dir),
        "timestamp": timestamp,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "| Metric | Value |",
        "|---|---:|",
        f"| Samples | {len(manifest_rows)} |",
        f"| Writers | {len(writer_ids)} |",
        f"| FID | {fid_value:.6f} |",
        f"| Generation runtime (s) | {generation_runtime:.2f} |",
        f"| FID runtime (s) | {fid_runtime:.2f} |",
    ]
    (artifacts_dir / "fid_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
