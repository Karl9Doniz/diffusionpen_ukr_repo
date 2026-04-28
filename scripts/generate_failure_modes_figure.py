#!/usr/bin/env python3
"""
Generate a small thesis-ready failure-mode panel focused on:
1. rare-letter words
2. apostrophe words

The script searches a targeted pool of (writer, word, seed) combinations,
scores them with the project's Cyrillic TrOCR model, and assembles the
strongest representative failures into a compact grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont

from evaluate_generated_word_cer import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_META_FILE,
    DEFAULT_SD_PATH,
    DEFAULT_STYLE_PATH,
    cer_components,
    classify_length_bucket,
    load_models,
    normalize_text,
    precompute_style_payloads,
    rare_letter_string,
    run_trocr_batch,
)
from generate_sentence import (
    build_writer_id_map,
    encode_text_context,
    generate_single_word,
    load_style_images,
    set_global_seed,
)

DEFAULT_RARE_WORDS = [
    "ґрати",
    "щойно",
    "своїх",
    "ефективність",
]

DEFAULT_APOSTROPHE_WORDS = [
    "комп'ютер",
    "ім'я",
    "здоров'я",
    "м'який",
]


@dataclass
class CandidateResult:
    category: str
    writer_id: str
    target_text: str
    seed: int
    trocr_pred: str
    target_norm: str
    pred_norm: str
    char_errors: int
    cer: float
    rare_letter_set: str
    length_bucket: str
    image_relpath: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis failure-mode figure.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--meta_file", type=Path, default=DEFAULT_META_FILE)
    parser.add_argument("--style_path", type=Path, default=DEFAULT_STYLE_PATH)
    parser.add_argument("--stable_dif_path", type=Path, default=DEFAULT_SD_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--trocr_device", type=str, default=None)
    parser.add_argument("--trocr_model", type=str, default="cyrillic-trocr/trocr-handwritten-cyrillic")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--img_height", type=int, default=64)
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--text_max_len", type=int, default=40)
    parser.add_argument("--emb_dim", type=int, default=320)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--top_writers", type=int, default=6)
    parser.add_argument("--writers", nargs="*", default=None)
    parser.add_argument("--exclude_writers", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 123])
    parser.add_argument("--rare_words", nargs="+", default=DEFAULT_RARE_WORDS)
    parser.add_argument("--apostrophe_words", nargs="+", default=DEFAULT_APOSTROPHE_WORDS)
    parser.add_argument("--select_per_category", type=int, default=3)
    parser.add_argument(
        "--run_root",
        type=Path,
        default=ROOT / "generated" / "failure_modes_20260424",
    )
    parser.add_argument(
        "--figure_output",
        type=Path,
        default=ROOT / "thesis" / "Roadmap_Ahitoliev_Andrii" / "Figures" / "failure_modes_rare_apostrophe.png",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def top_writers_from_meta(meta_file: Path, top_k: int) -> List[str]:
    counter: Counter[str] = Counter()
    with meta_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            parts = row["filename"].split("-")
            if len(parts) >= 3:
                counter[parts[2]] += 1
    return [writer for writer, _ in counter.most_common(top_k)]


def ensure_dirs(run_root: Path) -> Dict[str, Path]:
    candidates = run_root / "candidates"
    artifacts = run_root / "artifacts"
    for path in (run_root, candidates, artifacts):
        path.mkdir(parents=True, exist_ok=True)
    return {"run_root": run_root, "candidates": candidates, "artifacts": artifacts}


def choose_diverse(results: Sequence[CandidateResult], count: int) -> List[CandidateResult]:
    picked: List[CandidateResult] = []
    seen_words: set[str] = set()
    seen_writers: set[str] = set()
    sorted_results = sorted(
        results,
        key=lambda row: (
            -row.cer,
            -row.char_errors,
            row.target_text.count("'") + row.target_text.count("’") + row.target_text.count("ʼ"),
            len(row.target_norm),
        ),
    )

    for row in sorted_results:
        if row.target_text in seen_words:
            continue
        if row.writer_id in seen_writers and len(seen_writers) < count:
            continue
        picked.append(row)
        seen_words.add(row.target_text)
        seen_writers.add(row.writer_id)
        if len(picked) >= count:
            return picked

    for row in sorted_results:
        if row in picked:
            continue
        picked.append(row)
        if len(picked) >= count:
            break
    return picked


def save_candidate_csv(path: Path, rows: Sequence[CandidateResult]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_panel(selected: Dict[str, List[CandidateResult]], run_root: Path, output_path: Path) -> None:
    categories = [
        ("rare", "Rare-letter failures"),
        ("apostrophe", "Apostrophe failures"),
    ]
    cols = max(len(selected.get(cat, [])) for cat, _ in categories)
    if cols == 0:
        raise RuntimeError("No selected failure cases to render.")

    title_font = load_font(28)
    section_font = load_font(24)
    text_font = load_font(18)
    small_font = load_font(16)

    cell_w = 360
    cell_h = 210
    left_pad = 40
    right_pad = 40
    top_pad = 40
    bottom_pad = 40
    section_gap = 26
    col_gap = 24
    row_gap = 40
    section_title_h = 32

    width = left_pad + cols * cell_w + (cols - 1) * col_gap + right_pad
    height = top_pad + 36 + row_gap
    height += len(categories) * (section_title_h + section_gap + cell_h)
    height += (len(categories) - 1) * row_gap
    height += bottom_pad

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = "Representative failure modes"
    tw, th = text_size(draw, title, title_font)
    draw.text(((width - tw) // 2, top_pad), title, fill="black", font=title_font)

    y = top_pad + th + row_gap
    for category_key, category_label in categories:
        draw.text((left_pad, y), category_label, fill="black", font=section_font)
        y += section_title_h + section_gap

        for idx, row in enumerate(selected.get(category_key, [])):
            x = left_pad + idx * (cell_w + col_gap)
            draw.rounded_rectangle(
                [x, y, x + cell_w, y + cell_h],
                radius=14,
                outline="#c8c8c8",
                width=2,
                fill="white",
            )

            target_line = f"Target: {row.target_text}"
            writer_line = f"Writer {row.writer_id}"

            draw.text((x + 16, y + 14), target_line, fill="black", font=text_font)
            draw.text((x + 16, y + 38), writer_line, fill="#555555", font=small_font)

            image_path = run_root / row.image_relpath
            image = Image.open(image_path).convert("RGB")
            max_w = cell_w - 32
            max_h = 96
            scale = min(max_w / image.width, max_h / image.height, 1.0)
            new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            image_x = x + (cell_w - image.width) // 2
            image_y = y + 72
            canvas.paste(image, (image_x, image_y))

        y += cell_h + row_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    paths = ensure_dirs(args.run_root)

    writers = args.writers or top_writers_from_meta(args.meta_file, args.top_writers + len(args.exclude_writers))
    if args.exclude_writers:
        excluded = set(args.exclude_writers)
        writers = [writer for writer in writers if writer not in excluded]
    if args.top_writers:
        writers = writers[: args.top_writers]
    if not writers:
        raise RuntimeError("No writers selected.")

    from types import SimpleNamespace

    model_args = SimpleNamespace(
        checkpoint=args.checkpoint,
        stable_dif_path=args.stable_dif_path,
        style_path=args.style_path,
        device=args.device,
        trocr_device=args.trocr_device,
        trocr_model=args.trocr_model,
        cfg_scale=args.cfg_scale,
        img_height=args.img_height,
        img_width=args.img_width,
        text_max_len=args.text_max_len,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        num_res_blocks=args.num_res_blocks,
        channels=args.channels,
    )
    models = load_models(model_args)
    writer_id_map = build_writer_id_map(str(args.meta_file))
    writer_indices = [writer_id_map[writer] for writer in writers]

    style_ref_override = {writer: [] for writer in writers}
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
    null_context = encode_text_context(
        unet=models["unet"],
        tokenizer=models["tokenizer"],
        texts=[""],
        device=models["device"],
        text_max_len=args.text_max_len,
    ) if args.cfg_scale > 1.0 else None

    tasks = []
    for category, words in (
        ("rare", args.rare_words),
        ("apostrophe", args.apostrophe_words),
    ):
        for writer_id in writers:
            for word in words:
                for seed in args.seeds:
                    tasks.append((category, writer_id, word, seed))

    print(f"Selected writers: {writers}")
    print(f"Generating {len(tasks)} targeted samples...")

    results: List[CandidateResult] = []
    for category, writer_id, word, seed in tasks:
        writer_idx = writer_id_map[writer_id]
        style_payload = style_payloads[writer_idx]
        set_global_seed(seed)
        image = generate_single_word(
            word=word,
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
        )
        trocr_pred = run_trocr_batch(
            images=[image.convert("RGB")],
            processor=models["trocr_processor"],
            model=models["trocr_model"],
            device=models["trocr_device"],
        )[0]
        target_norm, pred_norm, char_errors, cer = cer_components(word, trocr_pred)
        safe_word = "".join(ch if ch.isalnum() or ch in "-_'" else "_" for ch in normalize_text(word))
        image_relpath = Path("candidates") / category / f"{writer_id}_{safe_word}_seed{seed}.png"
        image_path = args.run_root / image_relpath
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        results.append(
            CandidateResult(
                category=category,
                writer_id=writer_id,
                target_text=word,
                seed=seed,
                trocr_pred=trocr_pred,
                target_norm=target_norm,
                pred_norm=pred_norm,
                char_errors=char_errors,
                cer=cer,
                rare_letter_set=rare_letter_string(word),
                length_bucket=classify_length_bucket(word),
                image_relpath=str(image_relpath),
            )
        )

    save_candidate_csv(paths["artifacts"] / "failure_mode_candidates.csv", results)
    (paths["artifacts"] / "failure_mode_candidates.json").write_text(
        json.dumps([asdict(row) for row in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    selected = {
        "rare": choose_diverse([row for row in results if row.category == "rare" and row.cer > 0.0], args.select_per_category),
        "apostrophe": choose_diverse([row for row in results if row.category == "apostrophe" and row.cer > 0.0], args.select_per_category),
    }
    (paths["artifacts"] / "failure_mode_selected.json").write_text(
        json.dumps({key: [asdict(row) for row in value] for key, value in selected.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    build_panel(selected, args.run_root, args.figure_output)
    print(f"Figure written to {args.figure_output}")
    print(f"Artifacts written to {paths['artifacts']}")


if __name__ == "__main__":
    main()
