#!/usr/bin/env python3
"""
Generate a large, single-row cross-lingual style-transfer figure:
- 5 curated English IAM reference words from one writer
- 4 generated Ukrainian words in the extracted style
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Sequence

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
    crop_whitespace_h,
    find_nearest_ukr_writer,
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


DEFAULT_IAM_PATHS = [
    "/extra_space2/oles_new/iam_data/words/e01/e01-014/e01-014-00-01.png",  # know
    "/extra_space2/oles_new/iam_data/words/e01/e01-014/e01-014-00-02.png",  # from
    "/extra_space2/oles_new/iam_data/words/e01/e01-014/e01-014-01-04.png",  # great
    "/extra_space2/oles_new/iam_data/words/e01/e01-014/e01-014-03-01.png",  # mean
    "/extra_space2/oles_new/iam_data/words/e01/e01-014/e01-014-04-11.png",  # wife
]
DEFAULT_IAM_WORDS = ["know", "from", "great", "mean", "wife"]
DEFAULT_UA_WORDS = ["мова", "книга", "дорога", "природа"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--meta_file", type=Path, default=DEFAULT_META_FILE)
    p.add_argument("--style_path", type=Path, default=DEFAULT_STYLE_PATH)
    p.add_argument("--stable_dif_path", type=Path, default=DEFAULT_SD_PATH)
    p.add_argument("--trocr_model", type=str, default=DEFAULT_TROCR_MODEL)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--trocr_device", type=str, default=None)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--emb_dim", type=int, default=320)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--img_height", type=int, default=IMG_HEIGHT)
    p.add_argument("--img_width", type=int, default=IMG_WIDTH)
    p.add_argument("--text_max_len", type=int, default=40)
    p.add_argument("--iam_writer", type=str, default="e01")
    p.add_argument("--ukr_writer_override", type=str, default=None)
    p.add_argument("--iam_paths", nargs="+", default=DEFAULT_IAM_PATHS)
    p.add_argument("--iam_words", nargs="+", default=DEFAULT_IAM_WORDS)
    p.add_argument("--ua_words", nargs="+", default=DEFAULT_UA_WORDS)
    p.add_argument(
        "--run_root",
        type=Path,
        default=ROOT / "generated" / "cross_lingual_single_row_20260426",
    )
    p.add_argument(
        "--final_output",
        type=Path,
        default=ROOT / "thesis" / "Roadmap_Ahitoliev_Andrii" / "Figures" / "cross_lingual_transfer_single_row.png",
    )
    return p.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def derive_word_seed(base_seed: int, tag: str, word: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{tag}|{word}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def resize_gray_for_display(path: str, target_h: int) -> Image.Image:
    img = Image.open(path).convert("L")
    w, h = img.size
    new_w = max(1, int(round(w * target_h / max(h, 1))))
    return img.resize((new_w, target_h), Image.Resampling.LANCZOS)


def build_panel(
    ref_images: Sequence[Image.Image],
    ref_words: Sequence[str],
    gen_images: Sequence[Image.Image],
    gen_words: Sequence[str],
    output_path: Path,
) -> None:
    outer = 28
    section_gap = 26
    col_gap = 18
    label_gap = 10
    ref_h = 96
    gen_h = 88

    font_header = load_font(24)
    font_label = load_font(20)

    ref_resized = []
    for im in ref_images:
        w, h = im.size
        new_w = max(1, int(round(w * ref_h / max(h, 1))))
        ref_resized.append(im.resize((new_w, ref_h), Image.Resampling.LANCZOS))

    gen_resized = []
    for im in gen_images:
        w, h = im.size
        new_w = max(1, int(round(w * gen_h / max(h, 1))))
        gen_resized.append(im.resize((new_w, gen_h), Image.Resampling.LANCZOS))

    ref_col_widths = [img.width for img in ref_resized]
    gen_col_widths = [img.width for img in gen_resized]

    ref_total_w = sum(ref_col_widths) + col_gap * max(0, len(ref_col_widths) - 1)
    gen_total_w = sum(gen_col_widths) + col_gap * max(0, len(gen_col_widths) - 1)
    body_w = max(ref_total_w, gen_total_w)

    label_h = 24
    header_h = 38
    total_h = outer * 2 + header_h + ref_h + label_h + section_gap + header_h + gen_h + label_h
    total_w = outer * 2 + body_w

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    # EN section
    y = outer
    draw.text((outer, y), "English IAM reference words", fill=(25, 25, 25), font=font_header)
    y += header_h
    x = outer + (body_w - ref_total_w) // 2
    for word, img in zip(ref_words, ref_resized):
        canvas.paste(img.convert("RGB"), (x, y))
        tw, _ = text_size(draw, word, font_label)
        draw.text((x + (img.width - tw) // 2, y + ref_h + 4), word, fill=(45, 45, 45), font=font_label)
        x += img.width + col_gap

    # UA section
    y += ref_h + label_h + section_gap
    draw.text((outer, y), "Generated Ukrainian words", fill=(25, 25, 25), font=font_header)
    y += header_h
    x = outer + (body_w - gen_total_w) // 2
    for word, img in zip(gen_words, gen_resized):
        canvas.paste(img.convert("RGB"), (x, y))
        tw, _ = text_size(draw, word, font_label)
        draw.text((x + (img.width - tw) // 2, y + gen_h + 4), word, fill=(45, 45, 45), font=font_label)
        x += img.width + col_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    if len(args.iam_paths) != len(args.iam_words):
        raise ValueError("iam_paths and iam_words must have the same length")

    args.run_root.mkdir(parents=True, exist_ok=True)

    models = load_models(args)
    writer_id_map = build_ukr_writer_id_map(str(args.meta_file))
    cache_path = args.run_root / "ukr_style_index_final.pt"
    if cache_path.exists():
        ukr_index = torch.load(cache_path, map_location="cpu")
    else:
        ukr_index = build_ukr_embedding_index(
            str(args.dataset_root),
            str(args.meta_file),
            writer_id_map,
            models["style_extractor"],
            models["device"],
        )
        torch.save(ukr_index, cache_path)

    for payload in ukr_index.values():
        if isinstance(payload.get("embedding"), torch.Tensor):
            payload["embedding"] = payload["embedding"].to(models["device"])

    iam_tensor = load_images_from_paths(args.iam_paths)
    if iam_tensor is None or len(iam_tensor) < len(args.iam_paths):
        raise RuntimeError("Could not load IAM references")
    iam_tensor = iam_tensor[: len(args.iam_paths)].to(models["device"])

    emb = compute_style_embedding(args.iam_paths, models["style_extractor"], models["device"])
    if args.ukr_writer_override is not None:
        match_str = args.ukr_writer_override
        if match_str not in ukr_index:
            raise ValueError(f"Unknown Ukrainian writer override: {match_str}")
        match_idx = ukr_index[match_str]["idx"]
        sim = float(
            torch.nn.functional.cosine_similarity(
                emb.unsqueeze(0),
                ukr_index[match_str]["embedding"].unsqueeze(0),
            ).item()
        )
    else:
        match_str, match_idx, sim = find_nearest_ukr_writer(emb, ukr_index, top_k=1)[0]

    style_payload = precompute_style_payloads(
        style_refs={match_idx: iam_tensor},
        style_extractor=models["style_extractor"],
        device=models["device"],
        img_height=args.img_height,
        img_width=args.img_width,
    )[match_idx]

    gen_dir = args.run_root / "generated_words"
    gen_dir.mkdir(parents=True, exist_ok=True)

    ref_display = [resize_gray_for_display(path, 96) for path in args.iam_paths]
    gen_display: List[Image.Image] = []
    generated_paths: List[str] = []
    word_seeds = {}

    null_context = encode_text_context(
        unet=models["unet"],
        tokenizer=models["tokenizer"],
        texts=[""],
        device=models["device"],
        text_max_len=args.text_max_len,
    ) if args.cfg_scale > 1.0 else None

    for word in args.ua_words:
        word_seed = derive_word_seed(args.seed, args.iam_writer, word)
        word_seeds[word] = word_seed
        set_global_seed(word_seed)
        text_context = encode_text_context(
            unet=models["unet"],
            tokenizer=models["tokenizer"],
            texts=[word],
            device=models["device"],
            text_max_len=args.text_max_len,
        )
        out = generate_single_word(
            word=word,
            unet=models["unet"],
            vae=models["vae"],
            style_extractor=models["style_extractor"],
            tokenizer=models["tokenizer"],
            noise_scheduler=models["noise_scheduler"],
            style_ref=iam_tensor,
            writer_idx=match_idx,
            device=models["device"],
            cfg_scale=args.cfg_scale,
            img_height=args.img_height,
            img_width=args.img_width,
            text_max_len=args.text_max_len,
            style_features=style_payload["style_features"],
            label_tensor=style_payload["label_tensor"],
            text_context=text_context,
            null_context=null_context,
        ).convert("L")
        cropped = crop_whitespace_h(out)
        if isinstance(cropped, np.ndarray):
            out = Image.fromarray(cropped).convert("L")
        else:
            out = cropped.convert("L")
        out_path = gen_dir / f"{word}.png"
        out.save(out_path)
        generated_paths.append(str(out_path))
        gen_display.append(out)

    build_panel(ref_display, args.iam_words, gen_display, args.ua_words, args.final_output)

    manifest = {
        "iam_writer": args.iam_writer,
        "iam_paths": args.iam_paths,
        "iam_words": args.iam_words,
        "matched_ukr_writer": match_str,
        "matched_ukr_index": match_idx,
        "cosine_sim": sim,
        "seed": args.seed,
        "word_seeds": word_seeds,
        "ua_words": args.ua_words,
        "generated_paths": generated_paths,
        "figure_output": str(args.final_output),
    }
    (args.run_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.final_output)


if __name__ == "__main__":
    main()
