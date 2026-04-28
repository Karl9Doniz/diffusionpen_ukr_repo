#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineModel, CanineTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_extractor import ImageEncoder
from generate_sentence import (
    build_writer_id_map,
    crop_whitespace,
    detect_baseline_and_clean,
    generate_single_word,
    load_style_images,
    remove_underline,
    set_global_seed,
    strip_dp_prefix,
    detect_num_classes,
)
from unet import UNetModel
from utils.word_cleanup_nafnet import NAFNetWordCleaner


DEFAULT_WRITERS: Dict[str, Dict[str, List[str]]] = {
    "0220": {
        "display": [
            "a01-218-0220-05-w03.png",  # світло
            "a01-218-0220-01-w02.png",  # коли
            "a01-219-0220-08-w03.png",  # миті
            "a01-219-0220-10-w01.png",  # зимових
        ],
        "style_extra": ["a01-220-0220-10-w04.png"],  # кличуть
    },
    "0219": {
        "display": [
            "a01-205-0219-07-w01.png",  # світ
            "a01-210-0219-08-w01.png",  # лише
            "a01-209-0219-06-w00.png",  # фруктів
            "a01-203-0219-06-w03.png",  # залежить
        ],
        "style_extra": ["a01-210-0219-11-w02.png"],  # дістатися
    },
    "0543": {
        "display": [
            "a01-229-0543-09-w01.png",  # далекі
            "a01-231-0543-03-w02.png",  # кожен
            "a01-224-0543-10-w03.png",  # люди
            "a01-232-0543-01-w02.png",  # сімейному
        ],
        "style_extra": ["a01-232-0543-03-w02.png"],  # зібралися
    },
    "0597": {
        "display": [
            "a01-230-0597-08-w00.png",  # стоїть
            "a01-221-0597-05-w01.png",  # кафе
            "a01-231-0597-08-w01.png",  # відчути
            "a01-224-0597-05-w01.png",  # ліхтарів
        ],
        "style_extra": ["a01-234-0597-02-w02.png"],  # думки
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/oles/DiffusionPen/output/diffusionpen_ukr_v10_ulcleannaf_trocr_tf32bs128/models/ema_ckpt.pt"),
    )
    p.add_argument(
        "--stable_dif_path",
        type=Path,
        default=Path("/home/oles/DiffusionPen/stable-diffusion-v1-5"),
    )
    p.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/extra_space2/oles_new/UkrHandwritten_Words_CC_ULCleanNAF_v1"),
    )
    p.add_argument(
        "--meta_file",
        type=Path,
        default=Path("/extra_space2/oles_new/UkrHandwritten_Words_CC_ULCleanNAF_v1/METAFILE_extended_trocr_local3_balanced_20260421_181218.tsv"),
    )
    p.add_argument(
        "--style_path",
        type=Path,
        default=Path("/home/oles/DiffusionPen/style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth"),
    )
    p.add_argument(
        "--nafnet_ckpt",
        type=Path,
        default=Path("/home/oles/DiffusionPen/output/lines204_nafnet_v1/checkpoint_best.pt"),
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--img_height", type=int, default=64)
    p.add_argument("--img_width", type=int, default=256)
    p.add_argument("--text_max_len", type=int, default=40)
    p.add_argument("--emb_dim", type=int, default=320)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--nafnet_blend", type=float, default=0.85)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/home/oles/DiffusionPen/thesis/Roadmap_Ahitoliev_Andrii/Figures/style_conditioning_real_vs_generated_pairs.png"),
    )
    p.add_argument("--pairs_per_panel", type=int, default=2)
    p.add_argument("--group_gap", type=int, default=44)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/oles/DiffusionPen/generated/real_vs_generated_pairs_20260426/manifest.json"),
    )
    return p.parse_args()


def load_font(size: int):
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        p = Path(candidate)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def load_generation_stack(args: argparse.Namespace):
    device = torch.device(args.device)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    state_dict = strip_dp_prefix(state_dict)
    num_classes = detect_num_classes(state_dict)

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    canine_model = CanineModel.from_pretrained("google/canine-c")

    from utils.word_dataset import char_classes as WORD_CHAR_CLASSES

    fake_args = SimpleNamespace(interpolation=False, mix_rate=None)
    unet = UNetModel(
        image_size=(args.img_height, args.img_width),
        in_channels=args.channels,
        model_channels=args.emb_dim,
        out_channels=args.channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=args.num_heads,
        num_classes=num_classes,
        context_dim=args.emb_dim,
        vocab_size=WORD_CHAR_CLASSES,
        text_encoder=canine_model,
        args=fake_args,
    )
    unet.load_state_dict(state_dict)
    unet = unet.to(device).eval()

    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    noise_scheduler = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    style_extractor = ImageEncoder(model_name="mobilenetv2_100", num_classes=0, pretrained=False, trainable=False)
    style_sd = torch.load(args.style_path, map_location="cpu")
    model_dict = style_extractor.state_dict()
    style_sd = {k: v for k, v in style_sd.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(style_sd)
    style_extractor.load_state_dict(model_dict)
    style_extractor = style_extractor.to(device).eval()

    nafnet_cleaner = NAFNetWordCleaner(
        checkpoint_path=str(args.nafnet_ckpt),
        device=args.device,
        blend=args.nafnet_blend,
    )

    return device, tokenizer, unet, vae, noise_scheduler, style_extractor, nafnet_cleaner


def collect_transcriptions(meta_file: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    import csv
    with meta_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mapping[row["filename"]] = row["transcription"]
    return mapping


def build_panel(
    pairs_by_writer: Dict[str, List[Dict[str, object]]],
    output: Path,
    group_gap: int,
) -> None:
    writers = list(pairs_by_writer.keys())
    font_title = load_font(24)
    font_word = load_font(18)
    font_small = load_font(15)
    font_writer = load_font(20)

    header_h = 68
    subheader_h = 30
    row_gap = 26
    writer_label_w = 150
    cell_pad_x = 12
    cell_pad_y = 10
    inner_gap = 18
    outer_pad = 24

    n_pairs = max(len(v) for v in pairs_by_writer.values())
    pair_widths = [0] * n_pairs
    row_heights = [0] * len(writers)

    for r_idx, writer in enumerate(writers):
        max_h = 0
        for c_idx, pair in enumerate(pairs_by_writer[writer]):
            real = pair["real_img"]
            gen = pair["gen_img"]
            pair_w = real.width + gen.width + inner_gap + cell_pad_x * 4
            pair_widths[c_idx] = max(pair_widths[c_idx], pair_w)
            max_h = max(max_h, max(real.height, gen.height) + cell_pad_y * 2)
        row_heights[r_idx] = max(max_h, 84)

    total_w = outer_pad * 2 + writer_label_w + sum(pair_widths) + group_gap * (n_pairs - 1)
    total_h = outer_pad * 2 + header_h + subheader_h + sum(row_heights) + row_gap * (len(writers) - 1)

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Headers
    x = outer_pad + writer_label_w
    y_word = outer_pad
    y_sub = outer_pad + header_h - 8
    for c_idx in range(n_pairs):
        group_x0 = x
        group_x1 = x + pair_widths[c_idx]
        word_label = pairs_by_writer[writers[0]][c_idx]["word"]
        tw, th = text_size(draw, str(word_label), font_word)
        draw.text((group_x0 + (pair_widths[c_idx] - tw) / 2, y_word + 6), str(word_label), fill="black", font=font_word)

        half = (pair_widths[c_idx] - inner_gap) / 2
        rw, _ = text_size(draw, "Real", font_small)
        gw, _ = text_size(draw, "Generated", font_small)
        draw.text((group_x0 + (half - rw) / 2, y_sub), "Real", fill="#333333", font=font_small)
        draw.text((group_x0 + half + inner_gap + (half - gw) / 2, y_sub), "Generated", fill="#333333", font=font_small)
        x = group_x1 + group_gap

    # Rows
    y = outer_pad + header_h + subheader_h
    for r_idx, writer in enumerate(writers):
        rh = row_heights[r_idx]
        ww, wh = text_size(draw, f"writer {writer}", font_writer)
        draw.text((outer_pad, y + (rh - wh) / 2), f"writer {writer}", fill="black", font=font_writer)

        x = outer_pad + writer_label_w
        for c_idx, pair in enumerate(pairs_by_writer[writer]):
            group_x0 = x
            group_w = pair_widths[c_idx]
            real: Image.Image = pair["real_img"]
            gen: Image.Image = pair["gen_img"]

            mid_x = group_x0 + group_w // 2
            draw.line((mid_x, y + 6, mid_x, y + rh - 6), fill=(190, 190, 190), width=1)

            real_x = group_x0 + cell_pad_x + ((group_w // 2 - inner_gap // 2) - 2 * cell_pad_x - real.width) // 2
            gen_area_x0 = mid_x + inner_gap // 2
            gen_x = gen_area_x0 + cell_pad_x + ((group_w // 2 - inner_gap // 2) - 2 * cell_pad_x - gen.width) // 2
            real_y = y + (rh - real.height) // 2
            gen_y = y + (rh - gen.height) // 2

            canvas.paste(real.convert("RGB"), (real_x, real_y))
            canvas.paste(gen.convert("RGB"), (gen_x, gen_y))

            x += group_w + group_gap
        y += rh + row_gap

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def split_pairs(
    pairs_by_writer: Dict[str, List[Dict[str, object]]],
    pairs_per_panel: int,
) -> List[Dict[str, List[Dict[str, object]]]]:
    writers = list(pairs_by_writer.keys())
    total_pairs = max(len(v) for v in pairs_by_writer.values())
    chunks = []
    for start in range(0, total_pairs, pairs_per_panel):
        end = start + pairs_per_panel
        chunk: Dict[str, List[Dict[str, object]]] = {}
        for writer in writers:
            chunk[writer] = pairs_by_writer[writer][start:end]
        chunks.append(chunk)
    return chunks


def main() -> None:
    args = parse_args()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    words_dir = args.dataset_root / "words" / "words"
    transcription_by_file = collect_transcriptions(args.meta_file)
    writer_id_map = build_writer_id_map(str(args.meta_file))
    idx_to_writer = {v: k for k, v in writer_id_map.items()}

    style_ref_override = {}
    for writer, cfg in DEFAULT_WRITERS.items():
        style_ref_override[writer] = list(cfg["display"]) + list(cfg["style_extra"])

    device, tokenizer, unet, vae, noise_scheduler, style_extractor, nafnet_cleaner = load_generation_stack(args)

    writer_indices = [writer_id_map[w] for w in DEFAULT_WRITERS.keys()]
    style_refs, style_refs_used = load_style_images(
        str(args.dataset_root),
        str(args.meta_file),
        writer_indices,
        writer_id_map,
        img_height=args.img_height,
        img_width=args.img_width,
        style_ref_override=style_ref_override,
    )

    pairs_by_writer: Dict[str, List[Dict[str, object]]] = {}
    manifest: Dict[str, object] = {"writers": {}}

    for wid in writer_indices:
        writer = idx_to_writer[wid]
        manifest["writers"][writer] = {"style_refs_used": style_refs_used.get(wid, []), "pairs": []}

        style_features = None
        label_tensor = torch.tensor([wid], dtype=torch.long, device=device)
        pairs = []
        for i, filename in enumerate(DEFAULT_WRITERS[writer]["display"]):
            word = transcription_by_file[filename]
            real_img = Image.open(words_dir / filename).convert("L")

            # Deterministic but different per writer/word cell.
            cell_seed = int(args.seed) + wid * 100 + i
            set_global_seed(cell_seed)
            gen_pil = generate_single_word(
                word=word,
                unet=unet,
                vae=vae,
                style_extractor=style_extractor,
                tokenizer=tokenizer,
                noise_scheduler=noise_scheduler,
                style_ref=style_refs[wid],
                writer_idx=wid,
                device=device,
                cfg_scale=args.cfg_scale,
                img_height=args.img_height,
                img_width=args.img_width,
                text_max_len=args.text_max_len,
                style_features=style_features,
                label_tensor=label_tensor,
            )
            if style_features is None:
                style_batch = style_refs[wid].unsqueeze(0).to(device)
                style_flat = style_batch.reshape(-1, 3, args.img_height, args.img_width)
                style_features = style_extractor(style_flat).to(device)

            cropped = crop_whitespace(gen_pil)
            img_cleaned, ul_y = detect_baseline_and_clean(cropped)
            img_cleaned = remove_underline(img_cleaned, ul_y)
            img_cleaned = nafnet_cleaner.clean_gray(img_cleaned)
            gen_img = Image.fromarray(img_cleaned)

            pairs.append({"word": word, "real_img": real_img, "gen_img": gen_img})
            manifest["writers"][writer]["pairs"].append(
                {
                    "filename": filename,
                    "word": word,
                    "seed": cell_seed,
                }
            )
        pairs_by_writer[writer] = pairs

    panels = split_pairs(pairs_by_writer, args.pairs_per_panel)
    outputs = []
    stem = args.output.stem
    suffix = args.output.suffix
    for idx, panel_pairs in enumerate(panels, start=1):
        out_path = args.output.with_name(f"{stem}_{idx}{suffix}")
        build_panel(panel_pairs, out_path, group_gap=args.group_gap)
        outputs.append(str(out_path))
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in outputs:
        print(p)
    print(args.manifest)


if __name__ == "__main__":
    main()
