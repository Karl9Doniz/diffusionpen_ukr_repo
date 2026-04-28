#!/usr/bin/env python3
"""
Generate thesis-ready style-transfer figures for one archival page and one
modern unseen-writer page.

Each figure has:
  - top row: 5 reference word crops from the source page
  - bottom row: generated Ukrainian words in that extracted style
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cross_lingual_style import (  # noqa: E402
    IMG_HEIGHT,
    IMG_WIDTH,
    build_ukr_embedding_index,
    build_ukr_writer_id_map,
    compute_style_embedding,
    find_nearest_ukr_writer,
    generate_word,
    load_images_from_paths,
)
from evaluate_generated_word_cer import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_META_FILE,
    DEFAULT_SD_PATH,
    DEFAULT_STYLE_PATH,
    DEFAULT_TROCR_MODEL,
    load_models,
    precompute_style_payloads,
)
from generate_sentence import encode_text_context, generate_single_word, set_global_seed  # noqa: E402


@dataclass
class SourceSpec:
    key: str
    page_path: Path
    display_boxes: List[List[int]]
    target_words: List[str]
    output_figure: Path


ARCHIVAL_BOXES = [
    [431, 728, 1093, 857],
    [1202, 739, 1543, 847],
    [643, 880, 802, 1003],
    [927, 882, 1155, 999],
    [1416, 1057, 1729, 1108],
]

MODERN_BOXES = [
    [228, 226, 366, 305],
    [428, 249, 582, 282],
    [632, 212, 815, 285],
    [848, 249, 1038, 280],
    [1079, 240, 1298, 284],
]

ARCHIVAL_TARGET_WORDS = ["вікно", "сторінка", "дорога", "природа"]
MODERN_TARGET_WORDS = ["книга", "привіт", "вечір", "діжка"]


def derive_word_seed(panel_seed: int, domain_key: str, word: str) -> int:
    digest = hashlib.sha256(f"{panel_seed}|{domain_key}|{word}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def parse_args() -> object:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--meta_file", type=Path, default=DEFAULT_META_FILE)
    p.add_argument("--style_path", type=Path, default=DEFAULT_STYLE_PATH)
    p.add_argument("--stable_dif_path", type=Path, default=DEFAULT_SD_PATH)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--trocr_device", type=str, default=None)
    p.add_argument("--trocr_model", type=str, default=DEFAULT_TROCR_MODEL)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--archival_writer_override", type=str, default=None)
    p.add_argument("--modern_writer_override", type=str, default=None)
    p.add_argument("--img_height", type=int, default=IMG_HEIGHT)
    p.add_argument("--img_width", type=int, default=IMG_WIDTH)
    p.add_argument("--text_max_len", type=int, default=40)
    p.add_argument("--emb_dim", type=int, default=320)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument(
        "--run_root",
        type=Path,
        default=ROOT / "generated" / "cross_domain_style_transfer_20260425",
    )
    return p.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def crop_with_margin(image: Image.Image, box: Sequence[int], margin: int = 12) -> Image.Image:
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image.width, x2 + margin)
    y2 = min(image.height, y2 + margin)
    return image.crop((x1, y1, x2, y2))


def archival_normalize(crop: Image.Image) -> Image.Image:
    gray = np.array(crop.convert("L"))
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=19, sigmaY=19)
    bg = np.maximum(bg, 1)
    norm = np.clip(gray.astype(np.float32) / bg.astype(np.float32) * 240.0, 0, 255).astype(np.uint8)
    return Image.fromarray(norm).convert("RGB")


def modern_normalize(crop: Image.Image) -> Image.Image:
    gray = np.array(crop.convert("L"))
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(norm).convert("RGB")


def save_reference_crops(spec: SourceSpec, source_image: Image.Image, run_root: Path) -> Dict[str, List[str]]:
    display_dir = run_root / spec.key / "display_refs"
    model_dir = run_root / spec.key / "model_refs"
    display_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    display_paths: List[str] = []
    model_paths: List[str] = []
    for idx, box in enumerate(spec.display_boxes, start=1):
        crop = crop_with_margin(source_image, box, margin=10)
        display_path = display_dir / f"ref_{idx:02d}.png"
        crop.save(display_path)
        display_paths.append(str(display_path))

        if spec.key == "archival":
            model_crop = archival_normalize(crop)
        else:
            model_crop = modern_normalize(crop)
        model_path = model_dir / f"ref_{idx:02d}.png"
        model_crop.save(model_path)
        model_paths.append(str(model_path))

    return {"display": display_paths, "model": model_paths}


def load_generation_stack(args) -> Dict[str, object]:
    # Reuse the same loading path as the CER evaluator so the thesis figures
    # are generated with the exact current model stack.
    stack = load_models(args)
    stack["scheduler"] = stack.pop("noise_scheduler")
    stack.pop("trocr_device", None)
    stack.pop("trocr_processor", None)
    stack.pop("trocr_model", None)
    return stack


def build_panel(display_ref_paths: Sequence[str], generated_word_paths: Sequence[str], target_words: Sequence[str], output_path: Path) -> None:
    label_col_w = 170
    ref_gap = 16
    word_gap = 18
    outer_pad = 24
    header_h = 46
    row_gap = 18
    ref_h = 64
    gen_h = 64

    font_header = load_font(22)
    font_small = load_font(18)

    ref_imgs = []
    for path in display_ref_paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        new_w = max(1, int(round(w * ref_h / max(h, 1))))
        ref_imgs.append(img.resize((new_w, ref_h), Image.Resampling.LANCZOS))

    gen_imgs = []
    word_widths = []
    for path in generated_word_paths:
        img = Image.open(path).convert("L")
        w, h = img.size
        new_w = max(1, int(round(w * gen_h / max(h, 1))))
        img = img.resize((new_w, gen_h), Image.Resampling.LANCZOS).convert("RGB")
        gen_imgs.append(img)
        word_widths.append(img.width)

    ref_total_w = sum(im.width for im in ref_imgs) + ref_gap * max(0, len(ref_imgs) - 1)
    gen_total_w = sum(word_widths) + word_gap * max(0, len(gen_imgs) - 1)
    body_w = max(ref_total_w, gen_total_w)

    width = outer_pad * 2 + label_col_w + body_w
    height = outer_pad * 2 + header_h + ref_h + row_gap + gen_h + 26

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((outer_pad + 8, outer_pad + 8), "Reference crops", fill=(25, 25, 25), font=font_header)
    draw.text((outer_pad + 8, outer_pad + header_h + ref_h + row_gap + 2), "Generated words", fill=(25, 25, 25), font=font_header)

    ref_x = outer_pad + label_col_w + (body_w - ref_total_w) // 2
    x = ref_x
    for img in ref_imgs:
        canvas.paste(img, (x, outer_pad + header_h))
        x += img.width + ref_gap

    gen_x = outer_pad + label_col_w + (body_w - gen_total_w) // 2
    x = gen_x
    label_y = outer_pad + header_h + ref_h + row_gap - 2
    img_y = outer_pad + header_h + ref_h + row_gap + 28
    for word, img in zip(target_words, gen_imgs):
        tw, _ = text_size(draw, word, font_small)
        draw.text((x + (img.width - tw) // 2, label_y), word, fill=(25, 25, 25), font=font_small)
        canvas.paste(img, (x, img_y))
        x += img.width + word_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)

    stack = load_generation_stack(args)
    writer_id_map = build_ukr_writer_id_map(str(args.meta_file))
    ukr_index_cache = args.run_root / "ukr_style_index.pt"
    if ukr_index_cache.exists():
        ukr_index = torch.load(ukr_index_cache, map_location="cpu")
    else:
        ukr_index = build_ukr_embedding_index(
            str(args.dataset_root),
            str(args.meta_file),
            writer_id_map,
            stack["style_extractor"],
            stack["device"],
        )
        torch.save(ukr_index, ukr_index_cache)

    # Cached indices are usually loaded on CPU; move embeddings onto the active
    # device so nearest-writer search can compare them with the source-page
    # embedding without device mismatch.
    for writer_str, payload in ukr_index.items():
        if isinstance(payload.get("embedding"), torch.Tensor):
            payload["embedding"] = payload["embedding"].to(stack["device"])

    specs = [
        SourceSpec(
            key="archival",
            page_path=ROOT / "archival" / "723996.jpg",
            display_boxes=ARCHIVAL_BOXES,
            target_words=ARCHIVAL_TARGET_WORDS,
            output_figure=ROOT / "thesis" / "Roadmap_Ahitoliev_Andrii" / "Figures" / "archival_transfer_selected.png",
        ),
        SourceSpec(
            key="modern",
            page_path=ROOT / "modern" / "2efee78e-0580-4604-b0b1-fa084de91814.jpg",
            display_boxes=MODERN_BOXES,
            target_words=MODERN_TARGET_WORDS,
            output_figure=ROOT / "thesis" / "Roadmap_Ahitoliev_Andrii" / "Figures" / "modern_transfer_selected.png",
        ),
    ]

    manifest = []
    for spec in specs:
        page_img = Image.open(spec.page_path).convert("RGB")
        ref_paths = save_reference_crops(spec, page_img, args.run_root)

        style_embedding = compute_style_embedding(
            ref_paths["model"],
            stack["style_extractor"],
            stack["device"],
            IMG_HEIGHT,
            IMG_WIDTH,
        )
        top_match = find_nearest_ukr_writer(style_embedding, ukr_index, top_k=1)[0]
        matched_writer_str, matched_writer_idx, cosine_sim = top_match

        override_writer = (
            args.archival_writer_override if spec.key == "archival" else args.modern_writer_override
        )
        if override_writer:
            if override_writer not in writer_id_map:
                raise KeyError(f"Unknown override writer ID: {override_writer}")
            matched_writer_str = override_writer
            matched_writer_idx = writer_id_map[override_writer]
            cosine_sim = None

        style_ref_tensor = load_images_from_paths(ref_paths["model"], IMG_HEIGHT, IMG_WIDTH)
        if style_ref_tensor is None:
            raise RuntimeError(f"Failed to load style refs for {spec.key}")
        style_ref_tensor = style_ref_tensor.to(stack["device"])
        style_payload = precompute_style_payloads(
            style_refs={matched_writer_idx: style_ref_tensor},
            style_extractor=stack["style_extractor"],
            device=stack["device"],
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
        )[matched_writer_idx]

        gen_dir = args.run_root / spec.key / "generated_words"
        gen_dir.mkdir(parents=True, exist_ok=True)
        generated_paths = []
        word_seeds = {}
        for word in spec.target_words:
            word_seed = derive_word_seed(args.seed, spec.key, word)
            word_seeds[word] = word_seed
            set_global_seed(word_seed)
            text_context = encode_text_context(
                unet=stack["unet"],
                tokenizer=stack["tokenizer"],
                texts=[word],
                device=stack["device"],
            )
            null_context = encode_text_context(
                unet=stack["unet"],
                tokenizer=stack["tokenizer"],
                texts=[""],
                device=stack["device"],
            ) if args.cfg_scale > 1.0 else None
            out_img = generate_single_word(
                word=word,
                unet=stack["unet"],
                vae=stack["vae"],
                style_extractor=stack["style_extractor"],
                tokenizer=stack["tokenizer"],
                noise_scheduler=stack["scheduler"],
                style_ref=style_ref_tensor,
                writer_idx=matched_writer_idx,
                device=stack["device"],
                cfg_scale=args.cfg_scale,
                img_height=IMG_HEIGHT,
                img_width=IMG_WIDTH,
                style_features=style_payload["style_features"],
                label_tensor=style_payload["label_tensor"],
                text_context=text_context,
                null_context=null_context,
            )
            out_path = gen_dir / f"{word}.png"
            out_img.save(out_path)
            generated_paths.append(str(out_path))

        build_panel(ref_paths["display"], generated_paths, spec.target_words, spec.output_figure)

        manifest.append(
            {
                "key": spec.key,
                "page_path": str(spec.page_path),
                "display_boxes": spec.display_boxes,
                "display_ref_paths": ref_paths["display"],
                "model_ref_paths": ref_paths["model"],
                "matched_writer": matched_writer_str,
                "matched_writer_idx": matched_writer_idx,
                "cosine_sim": cosine_sim,
                "panel_seed": args.seed,
                "word_seeds": word_seeds,
                "target_words": spec.target_words,
                "generated_paths": generated_paths,
                "figure_output": str(spec.output_figure),
            }
        )

    (args.run_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.run_root / "manifest.json")


if __name__ == "__main__":
    main()
