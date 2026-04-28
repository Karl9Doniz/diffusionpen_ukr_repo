"""
DiffusionPen Ukrainian Handwriting Demo

Usage:
    python demo/app.py --checkpoint output/diffusionpen_ukr_v9/models/ckpt.pt
    python demo/app.py --checkpoint pen_checkpoints/v8/ckpt_ep200.pt --num_res_blocks 2
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import gradio as gr
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineTokenizer, CanineModel

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from unet import UNetModel
from feature_extractor import ImageEncoder
from generate_sentence import (
    PUNCTUATION, strip_dp_prefix, detect_num_classes, build_writer_id_map,
    load_style_images, generate_single_word, crop_whitespace,
    erase_bottom_artifacts, detect_baseline_and_clean,
    erase_underline_surgical, measure_ink_top,
    align_to_baseline, normalize_ink_brightness,
    stitch_paragraph, sample_punctuation, split_word_for_generation,
)
from utils.word_dataset import char_classes as WORD_CHAR_CLASSES


_DEFAULTS = dict(
    stable_dif_path="/home/oles/DiffusionPen/stable-diffusion-v1-5",
    style_path="/home/oles/DiffusionPen/style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth",
    dataset_root="/extra_space2/oles_new/UkrHandwritten_Words_CC",
    meta_file="/extra_space2/oles_new/UkrHandwritten_Words_CC/METAFILE_extended_balanced.tsv",
    punct_bank="/home/oles/DiffusionPen/generated/punct_bank",
    num_res_blocks=2,
    img_height=64,
    img_width=256,
    text_max_len=40,
    canvas_height=104,
    min_free_gb=6.0,
)

MODEL: dict = {}
WRITER_LIST: list = []
WRITER_ID_MAP: dict = {}

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_DEMO_DIR, "demo_logs")
IMG_LOG_DIR = os.path.join(LOG_DIR, "images")
JSONL_LOG = os.path.join(LOG_DIR, "generations.jsonl")


def select_gpu(min_free_gb: float = 6.0):
    """Pick the GPU with most free VRAM; fall back to CPU if none suitable."""
    if not torch.cuda.is_available():
        return "cpu", 0.0
    best_idx, best_free = 0, 0.0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        free_gb = free / 1e9
        if free_gb > best_free:
            best_free = free_gb
            best_idx = i
    if best_free < min_free_gb:
        print(f"WARNING: best GPU only {best_free:.1f} GB free. Using CPU.")
        return "cpu", 0.0
    return f"cuda:{best_idx}", best_free


def load_models(args):
    global WRITER_LIST, WRITER_ID_MAP

    device_str, free_gb = select_gpu(args.min_free_gb)
    device = torch.device(device_str)
    print(f"[demo] Device: {device_str}  ({free_gb:.1f} GB free)")

    print(f"[demo] Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    state_dict = strip_dp_prefix(state_dict)
    num_classes = detect_num_classes(state_dict)
    print(f"[demo]   {num_classes} writer classes")

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    canine_model = CanineModel.from_pretrained("google/canine-c")

    unet = UNetModel(
        image_size=(args.img_height, args.img_width),
        in_channels=4, model_channels=320, out_channels=4,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=4,
        num_classes=num_classes, context_dim=320,
        vocab_size=WORD_CHAR_CLASSES, text_encoder=canine_model,
        args=SimpleNamespace(interpolation=False, mix_rate=None),
    )
    unet.load_state_dict(state_dict)
    unet = unet.to(device).eval()

    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
    vae = vae.to(device).requires_grad_(False)

    noise_scheduler = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    style_extractor = ImageEncoder(model_name="mobilenetv2_100", num_classes=0,
                                   pretrained=False, trainable=False)
    style_sd = torch.load(args.style_path, map_location="cpu")
    model_dict = style_extractor.state_dict()
    style_sd = {k: v for k, v in style_sd.items()
                if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(style_sd)
    style_extractor.load_state_dict(model_dict)
    style_extractor = style_extractor.to(device).eval()

    WRITER_ID_MAP = build_writer_id_map(args.meta_file)
    WRITER_LIST = sorted(WRITER_ID_MAP.keys())
    print(f"[demo]   {len(WRITER_LIST)} writers indexed")

    MODEL.update({
        "unet": unet, "vae": vae,
        "style_extractor": style_extractor,
        "tokenizer": tokenizer,
        "noise_scheduler": noise_scheduler,
        "device": device, "device_str": device_str,
        "num_classes": num_classes,
        "dataset_root": args.dataset_root,
        "meta_file": args.meta_file,
        "img_height": args.img_height, "img_width": args.img_width,
        "text_max_len": args.text_max_len,
        "canvas_height": args.canvas_height,
        "punct_bank": args.punct_bank,
    })
    print("[demo] Ready.")


def _log_generation(text, writer_str, writer_idx, cfg_scale, seed, duration_s, img_path, status):
    record = {
        "timestamp": datetime.now().isoformat(),
        "text": text, "writer_str": writer_str,
        "writer_idx": int(writer_idx), "cfg_scale": float(cfg_scale),
        "seed": int(seed), "duration_s": round(float(duration_s), 2),
        "output": img_path, "status": status,
    }
    with open(JSONL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_recent_table(n: int = 10):
    if not os.path.exists(JSONL_LOG):
        return []
    rows = []
    with open(JSONL_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    table = []
    for r in reversed(rows[-n:]):
        ts = r.get("timestamp", "")[:19].replace("T", " ")
        table.append([
            ts, r.get("text", "")[:40], r.get("writer_str", ""),
            r.get("cfg_scale", ""), r.get("seed", ""),
            f"{r.get('duration_s', 0):.1f}s", r.get("status", ""),
        ])
    return table


def _random_writer():
    return random.choice(WRITER_LIST) if WRITER_LIST else "Random"


def generate(text_raw: str, writer_str: str, cfg_scale: float, seed_val: int):
    if not MODEL:
        return None, "Models not loaded.", _load_recent_table()

    text = text_raw.strip()
    if not text:
        return None, "Please enter some text.", _load_recent_table()

    if writer_str == "Random" or writer_str not in WRITER_ID_MAP:
        writer_str = _random_writer()
    writer_idx = WRITER_ID_MAP[writer_str]

    # Seed controls the initial diffusion noise. same seed + same writer = identical output.
    seed = int(seed_val)
    if seed < 0:
        seed = random.randint(0, 2 ** 31 - 1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)

    device = MODEL["device"]
    img_height = MODEL["img_height"]
    img_width = MODEL["img_width"]

    style_refs = load_style_images(
        MODEL["dataset_root"], MODEL["meta_file"],
        [writer_idx], WRITER_ID_MAP,
        img_height=img_height, img_width=img_width,
    )
    if writer_idx not in style_refs:
        return None, f"No style images for writer '{writer_str}'.", _load_recent_table()

    style_ref = style_refs[writer_idx]
    t0 = time.time()
    word_images, expanded_words, punct_flags, word_underline_ys = [], [], [], []

    for word in text.split():
        punct_suffix = []
        w = word
        while w and w[-1] in PUNCTUATION:
            punct_suffix.insert(0, w[-1])
            w = w[:-1]

        if w:
            punct_bank = MODEL.get("punct_bank")
            for part, is_mark in split_word_for_generation(w, punct_bank, img_height):
                if is_mark:
                    ch_arr = sample_punctuation(part, img_height, punct_bank, writer_str)
                    if ch_arr is not None:
                        word_images.append(ch_arr)
                        expanded_words.append(part)
                        punct_flags.append(True)
                        word_underline_ys.append(None)
                else:
                    img_pil = generate_single_word(
                        word=part, unet=MODEL["unet"], vae=MODEL["vae"],
                        style_extractor=MODEL["style_extractor"],
                        tokenizer=MODEL["tokenizer"],
                        noise_scheduler=MODEL["noise_scheduler"],
                        style_ref=style_ref, writer_idx=writer_idx,
                        device=device, cfg_scale=cfg_scale,
                        img_height=img_height, img_width=img_width,
                        text_max_len=MODEL["text_max_len"],
                    )
                    img_cleaned, ul_y = detect_baseline_and_clean(crop_whitespace(img_pil))
                    word_images.append(img_cleaned)
                    expanded_words.append(part)
                    punct_flags.append(False)
                    word_underline_ys.append(ul_y)

        for ch in punct_suffix:
            ch_arr = sample_punctuation(ch, img_height, MODEL.get("punct_bank"), writer_str)
            if ch_arr is not None:
                word_images.append(ch_arr)
                expanded_words.append(ch)
                punct_flags.append(True)
                word_underline_ys.append(None)

    if not word_images:
        return None, "No words generated.", _load_recent_table()

    word_images, word_shifts, max_bottom = align_to_baseline(
        word_images, is_punct=punct_flags, underline_ys=word_underline_ys,
    )
    for i in range(len(word_shifts)):
        if punct_flags[i]:
            prev = i - 1
            while prev >= 0 and punct_flags[prev]:
                prev -= 1
            if prev >= 0:
                word_shifts[i] = word_shifts[prev]
    for i, (ch, is_p) in enumerate(zip(expanded_words, punct_flags)):
        if is_p:
            repositioned = sample_punctuation(
                ch, img_height, MODEL.get("punct_bank"), writer_str,
                baseline_y=max_bottom,
            )
            if repositioned is not None:
                word_images[i] = repositioned
    for i, (img, is_p, ul_y) in enumerate(zip(word_images, punct_flags, word_underline_ys)):
        if not is_p:
            if ul_y is not None:
                word_images[i] = erase_underline_surgical(img, ul_y)
            else:
                word_images[i] = erase_bottom_artifacts(img)

    word_images = normalize_ink_brightness(word_images)

    baselines_64 = [max_bottom - s for s in word_shifts]
    ink_tops = [
        measure_ink_top(img) if not is_p else None
        for img, is_p in zip(word_images, punct_flags)
    ]
    ink_heights = [
        baselines_64[i] - ink_tops[i]
        for i in range(len(word_images))
        if not punct_flags[i] and ink_tops[i] is not None
        and baselines_64[i] > ink_tops[i]
    ]
    if ink_heights:
        target_ink_h = float(np.median(ink_heights))
        h_scales = []
        for i in range(len(word_images)):
            if punct_flags[i] or ink_tops[i] is None:
                h_scales.append(1.0)
            else:
                ink_h = baselines_64[i] - ink_tops[i]
                if ink_h <= 0 or ink_h <= target_ink_h:
                    h_scales.append(1.0)
                else:
                    h_scales.append(max(0.90, target_ink_h / ink_h))
    else:
        h_scales = [1.0] * len(word_images)

    paragraph = stitch_paragraph(word_images, expanded_words,
                                 gen_height=img_height, canvas_height=MODEL["canvas_height"],
                                 shifts=word_shifts, h_scales=h_scales,
                                 ref_baseline=max_bottom)
    duration = time.time() - t0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_text = "".join(c if c.isalnum() or c in " _-" else "_" for c in text)[:40].strip().replace(" ", "_")
    img_fname = f"{ts}_{safe_text}_w{writer_str}_cfg{int(cfg_scale)}.png"
    img_path = os.path.join(IMG_LOG_DIR, img_fname)
    paragraph.save(img_path)
    _log_generation(text, writer_str, writer_idx, cfg_scale, seed, duration, img_path, "ok")

    info = (f"Writer: {writer_str}  |  CFG: {cfg_scale:.1f}  |  Variation seed: {seed}  |  "
            f"GPU: {MODEL['device_str']}  |  Time: {duration:.1f}s  |  "
            f"Saved: demo/demo_logs/images/{img_fname}")
    return paragraph, info, _load_recent_table()


def build_ui():
    device_str = MODEL.get("device_str", "unknown")
    num_classes = MODEL.get("num_classes", "?")

    with gr.Blocks(title="DiffusionPen Ukrainian Demo") as demo:
        gr.Markdown(
            f"## DiffusionPen. Ukrainian Handwriting Generation\n"
            f"Diffusion model conditioned on writer style + text content.  "
            f"**{len(WRITER_LIST)} writers** ({num_classes} classes) &nbsp;|&nbsp; "
            f"**Device:** `{device_str}`"
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text (word or sentence)",
                    placeholder="Реве та стогне Дніпр широкий",
                    lines=2,
                )
                with gr.Row():
                    writer_dd = gr.Dropdown(
                        choices=["Random"] + WRITER_LIST, value="Random",
                        label="Writer ID", scale=3,
                    )
                    random_btn = gr.Button("Random", variant="secondary", scale=1)
                cfg_slider = gr.Slider(
                    minimum=4.0, maximum=7.0, step=1.0, value=5.0,
                    label="CFG Scale",
                )
                seed_input = gr.Number(
                    value=-1,
                    label="Variation seed  (-1 = random; fix to reproduce exact output)",
                    precision=0,
                )
                gen_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=3):
                output_img = gr.Image(label="Generated Image", type="pil")
                info_box = gr.Textbox(label="Generation Info", interactive=False, lines=2)

        gr.Markdown(
            f"### Recent Generations\n"
            f"Log: `demo/demo_logs/generations.jsonl`  |  Images: `demo/demo_logs/images/`"
        )
        log_table = gr.Dataframe(
            headers=["Timestamp", "Text", "Writer", "CFG", "Seed", "Time", "Status"],
            datatype=["str", "str", "str", "number", "number", "str", "str"],
            value=_load_recent_table(),
            interactive=False, wrap=True,
        )

        random_btn.click(fn=_random_writer, inputs=[], outputs=[writer_dd])
        gen_btn.click(
            fn=generate,
            inputs=[text_input, writer_dd, cfg_slider, seed_input],
            outputs=[output_img, info_box, log_table],
        )
        text_input.submit(
            fn=generate,
            inputs=[text_input, writer_dd, cfg_slider, seed_input],
            outputs=[output_img, info_box, log_table],
        )

    return demo


def parse_args():
    p = argparse.ArgumentParser(description="DiffusionPen Ukrainian Handwriting Demo")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset_root", type=str, default=_DEFAULTS["dataset_root"])
    p.add_argument("--meta_file", type=str, default=_DEFAULTS["meta_file"])
    p.add_argument("--stable_dif_path", type=str, default=_DEFAULTS["stable_dif_path"])
    p.add_argument("--style_path", type=str, default=_DEFAULTS["style_path"])
    p.add_argument("--punct_bank", type=str, default=_DEFAULTS["punct_bank"])
    p.add_argument("--num_res_blocks", type=int, default=_DEFAULTS["num_res_blocks"])
    p.add_argument("--img_height", type=int, default=_DEFAULTS["img_height"])
    p.add_argument("--img_width", type=int, default=_DEFAULTS["img_width"])
    p.add_argument("--text_max_len", type=int, default=_DEFAULTS["text_max_len"])
    p.add_argument("--canvas_height", type=int, default=_DEFAULTS["canvas_height"])
    p.add_argument("--min_free_gb", type=float, default=_DEFAULTS["min_free_gb"])
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio share link (gradio.live tunnel)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(IMG_LOG_DIR, exist_ok=True)
    load_models(args)
    demo = build_ui()
    print(f"\n[demo] Launching at http://{args.host}:{args.port}")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                theme=gr.themes.Default())


if __name__ == "__main__":
    main()
