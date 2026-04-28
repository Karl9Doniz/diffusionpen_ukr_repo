#!/usr/bin/env python3
"""
Empirical legibility evaluation for generated Ukrainian word images.

Phase 1A:
- build a deterministic seen-writer IV/OOV manifest from the final v10 metafile
- generate raw word images with the final checkpoint
- run Cyrillic TrOCR on generated images
- compute normalized CER and aggregate breakdowns
- write results, tables, figures, and audit artifacts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderKL, DDIMScheduler
from tqdm import tqdm
from transformers import CanineModel, CanineTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

from feature_extractor import ImageEncoder
from generate_sentence import (
    build_writer_id_map,
    detect_num_classes,
    encode_text_context,
    generate_single_word,
    load_style_images,
    set_global_seed,
    strip_dp_prefix,
)
from unet import UNetModel
from utils.word_dataset import char_classes as WORD_CHAR_CLASSES

DEFAULT_CHECKPOINT = ROOT / "output" / "diffusionpen_ukr_v10_ulcleannaf_trocr_tf32bs128" / "models" / "ema_ckpt.pt"
DEFAULT_DATASET_ROOT = Path("/extra_space2/oles_new/UkrHandwritten_Words_CC_ULCleanNAF_v1")
DEFAULT_META_FILE = DEFAULT_DATASET_ROOT / "METAFILE_extended_trocr_local3_balanced_20260421_181218.tsv"
DEFAULT_STYLE_PATH = ROOT / "style_models" / "ukr_mixed_wt0p7" / "mixed_ukr_mobilenetv2_100.pth"
DEFAULT_SD_PATH = ROOT / "stable-diffusion-v1-5"
DEFAULT_OUTPUT_ROOT = ROOT / "generated" / "empirical_metrics"
DEFAULT_TROCR_MODEL = "cyrillic-trocr/trocr-handwritten-cyrillic"
DEFAULT_EVAL_VERSION = "cer_v1_seen_words"

LENGTH_BUCKETS = ("1-3", "4-6", "7-9", "10+")
RARE_LETTERS = tuple("фґєїщ")


@dataclass(frozen=True)
class Record:
    filename: str
    transcription: str
    writer_id: str
    normalized: str
    length_bucket: str
    rare_letter_set: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated-word TrOCR CER.")
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
    parser.add_argument("--eval_version", type=str, default=DEFAULT_EVAL_VERSION)
    parser.add_argument("--samples_per_writer", type=int, default=16)
    parser.add_argument("--iv_per_writer", type=int, default=8)
    parser.add_argument("--oov_per_writer", type=int, default=8)
    parser.add_argument("--trocr_batch_size", type=int, default=32)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--manifest_path", type=Path, default=None)
    parser.add_argument("--manifest_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--audit_count", type=int, default=24)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.strip().split())
    return text.lower()


def has_cyrillic_letter(text: str) -> bool:
    for ch in text:
        if ch.isalpha() and "CYRILLIC" in unicodedata.name(ch, ""):
            return True
    return False


def token_length(text: str) -> int:
    letters = sum(1 for ch in text if ch.isalpha())
    return letters if letters > 0 else len(text)


def classify_length_bucket(text: str) -> str:
    length = token_length(text)
    if length <= 3:
        return "1-3"
    if length <= 6:
        return "4-6"
    if length <= 9:
        return "7-9"
    return "10+"


def rare_letter_string(text: str) -> str:
    norm = normalize_text(text)
    letters = sorted({ch for ch in norm if ch in RARE_LETTERS})
    return "".join(letters)


def is_candidate_word(text: str) -> bool:
    norm = normalize_text(text)
    if not norm or " " in norm:
        return False
    return has_cyrillic_letter(norm)


def stable_seed(eval_version: str, writer_id: str, target_text: str) -> int:
    payload = f"{eval_version}|{writer_id}|{normalize_text(target_text)}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, replace))
        prev = cur
    return prev[-1]


def cer_components(target_text: str, pred_text: str) -> tuple[str, str, int, float]:
    target_norm = normalize_text(target_text)
    pred_norm = normalize_text(pred_text)
    errors = edit_distance(target_norm, pred_norm)
    denom = max(len(target_norm), 1)
    cer = errors / float(denom)
    return target_norm, pred_norm, errors, cer


def checkpoint_id(path: Path) -> str:
    parent = path.parent.parent.name if path.parent.parent.exists() else path.parent.name
    return f"{parent}/{path.name}"


def meta_id(path: Path) -> str:
    return path.name


def read_records(meta_file: Path) -> List[Record]:
    records: List[Record] = []
    with meta_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            transcription = row["transcription"].strip()
            if not is_candidate_word(transcription):
                continue
            filename = row["filename"].strip()
            parts = filename.replace(".png", "").split("-")
            writer_id = parts[2] if len(parts) >= 3 else parts[0]
            norm = normalize_text(transcription)
            records.append(
                Record(
                    filename=filename,
                    transcription=transcription,
                    writer_id=writer_id,
                    normalized=norm,
                    length_bucket=classify_length_bucket(norm),
                    rare_letter_set=rare_letter_string(norm),
                )
            )
    return records


def build_vocab_index(records: Sequence[Record]) -> tuple[Dict[str, Dict[str, List[str]]], Dict[str, str], Dict[str, List[str]]]:
    writer_vocab: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    writer_files: Dict[str, List[str]] = defaultdict(list)
    global_surface: Dict[str, str] = {}
    for record in records:
        writer_vocab[record.writer_id][record.normalized].append(record.filename)
        writer_files[record.writer_id].append(record.filename)
        global_surface.setdefault(record.normalized, record.transcription)
    return writer_vocab, global_surface, writer_files


def choose_words_with_bucket_balance(
    word_to_surface: Dict[str, str],
    total: int,
    rng_seed: int,
) -> List[str]:
    rng = np.random.default_rng(rng_seed)
    by_bucket: Dict[str, List[str]] = {bucket: [] for bucket in LENGTH_BUCKETS}
    for norm in sorted(word_to_surface):
        by_bucket[classify_length_bucket(norm)].append(norm)

    selected: List[str] = []
    taken = set()
    target_per_bucket = max(total // len(LENGTH_BUCKETS), 1)

    for bucket in LENGTH_BUCKETS:
        pool = by_bucket[bucket][:]
        rng.shuffle(pool)
        for norm in pool[:target_per_bucket]:
            if norm not in taken:
                selected.append(norm)
                taken.add(norm)

    if len(selected) < total:
        leftovers: List[str] = []
        for bucket in LENGTH_BUCKETS:
            pool = [norm for norm in by_bucket[bucket] if norm not in taken]
            rng.shuffle(pool)
            leftovers.extend(pool)
        for norm in leftovers:
            if len(selected) >= total:
                break
            selected.append(norm)
            taken.add(norm)

    return selected[:total]


def build_manifest_rows(
    records: Sequence[Record],
    eval_version: str,
    meta_path: Path,
    checkpoint_path: Path,
    iv_per_writer: int,
    oov_per_writer: int,
) -> List[Dict[str, str]]:
    writer_vocab, global_surface, writer_files = build_vocab_index(records)
    all_norms = set(global_surface)

    manifest_rows: List[Dict[str, str]] = []
    for writer_id in sorted(writer_vocab):
        iv_norms = set(writer_vocab[writer_id])
        oov_norms = all_norms - iv_norms

        if len(iv_norms) < iv_per_writer:
            raise ValueError(f"Writer {writer_id} has only {len(iv_norms)} unique IV words.")
        if len(oov_norms) < oov_per_writer:
            raise ValueError(f"Writer {writer_id} has only {len(oov_norms)} unique OOV words.")

        iv_surface = {norm: global_surface[norm] for norm in iv_norms}
        oov_surface = {norm: global_surface[norm] for norm in oov_norms}

        iv_selected = choose_words_with_bucket_balance(iv_surface, iv_per_writer, stable_seed(eval_version, writer_id, "__iv__"))
        oov_selected = choose_words_with_bucket_balance(oov_surface, oov_per_writer, stable_seed(eval_version, writer_id, "__oov__"))

        style_refs = writer_files[writer_id][:5]
        if not style_refs:
            raise ValueError(f"Writer {writer_id} has no style reference files.")
        while len(style_refs) < 5:
            style_refs.append(style_refs[0])

        for source_type, selected_norms in (("iv", iv_selected), ("oov", oov_selected)):
            for norm in selected_norms:
                target_text = global_surface[norm]
                rare_set = rare_letter_string(norm)
                manifest_rows.append(
                    {
                        "cohort": "seen_writer",
                        "writer_id": writer_id,
                        "target_text": target_text,
                        "target_norm": norm,
                        "source_type": source_type,
                        "length_bucket": classify_length_bucket(norm),
                        "rare_letter_flag": "1" if rare_set else "0",
                        "rare_letter_set": rare_set,
                        "seed": str(stable_seed(eval_version, writer_id, target_text)),
                        "style_ref_filenames": "|".join(style_refs),
                        "checkpoint_id": checkpoint_id(checkpoint_path),
                        "meta_file_id": meta_id(meta_path),
                    }
                )

    manifest_rows.sort(key=lambda row: (row["writer_id"], row["source_type"], row["length_bucket"], row["target_norm"]))
    return manifest_rows


def write_tsv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def spread_limit_rows(rows: Sequence[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    if limit >= len(rows):
        return list(rows)
    indices = np.linspace(0, len(rows) - 1, num=limit, dtype=int)
    seen = set()
    selected: List[Dict[str, str]] = []
    for idx in indices:
        idx = int(idx)
        if idx not in seen:
            selected.append(rows[idx])
            seen.add(idx)
    if len(selected) < limit:
        for idx, row in enumerate(rows):
            if idx in seen:
                continue
            selected.append(row)
            if len(selected) >= limit:
                break
    return selected


def balanced_limit_rows(rows: Sequence[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    if limit >= len(rows):
        return list(rows)
    iv_rows = [row for row in rows if row.get("source_type") == "iv"]
    oov_rows = [row for row in rows if row.get("source_type") == "oov"]
    iv_target = min(len(iv_rows), max(1, limit // 2))
    oov_target = min(len(oov_rows), max(1, limit - iv_target))

    selected = spread_limit_rows(iv_rows, iv_target) + spread_limit_rows(oov_rows, oov_target)

    if len(selected) < limit:
        selected_keys = {(row["writer_id"], row["source_type"], row["target_norm"]) for row in selected}
        for row in spread_limit_rows(rows, limit * 2):
            key = (row["writer_id"], row["source_type"], row["target_norm"])
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            if len(selected) >= limit:
                break

    selected.sort(key=lambda row: (row["writer_id"], row["source_type"], row["length_bucket"], row["target_norm"]))
    return selected[:limit]


def load_models(args: argparse.Namespace):
    device = torch.device(args.device)

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    state_dict = strip_dp_prefix(state_dict)
    num_classes = detect_num_classes(state_dict)

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    canine_model = CanineModel.from_pretrained("google/canine-c")

    from types import SimpleNamespace

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
    unet = unet.to(device)
    unet.eval()

    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)

    noise_scheduler = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    style_extractor = ImageEncoder(
        model_name="mobilenetv2_100",
        num_classes=0,
        pretrained=False,
        trainable=False,
    )
    style_sd = torch.load(args.style_path, map_location="cpu")
    model_dict = style_extractor.state_dict()
    style_sd = {k: v for k, v in style_sd.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(style_sd)
    style_extractor.load_state_dict(model_dict)
    style_extractor = style_extractor.to(device)
    style_extractor.eval()

    trocr_device = torch.device(args.trocr_device or args.device)
    trocr_processor = TrOCRProcessor.from_pretrained(args.trocr_model)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(args.trocr_model).to(trocr_device)
    trocr_model.eval()

    return {
        "device": device,
        "trocr_device": trocr_device,
        "tokenizer": tokenizer,
        "unet": unet,
        "vae": vae,
        "noise_scheduler": noise_scheduler,
        "style_extractor": style_extractor,
        "trocr_processor": trocr_processor,
        "trocr_model": trocr_model,
    }


def run_trocr_batch(
    images: Sequence[Image.Image],
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    device: torch.device,
) -> List[str]:
    if not images:
        return []
    pixel_values = processor(images=list(images), return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=25, num_beams=1)
    return [text.strip() for text in processor.batch_decode(generated_ids, skip_special_tokens=True)]


def precompute_style_payloads(
    style_refs: Dict[int, torch.Tensor],
    style_extractor: ImageEncoder,
    device: torch.device,
    img_height: int,
    img_width: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    payloads: Dict[int, Dict[str, torch.Tensor]] = {}
    for writer_idx, style_ref in style_refs.items():
        style_batch = style_ref.unsqueeze(0).to(device)
        style_flat = style_batch.reshape(-1, 3, img_height, img_width)
        style_features = style_extractor(style_flat).detach()
        label_tensor = torch.tensor([writer_idx], dtype=torch.long, device=device)
        payloads[writer_idx] = {
            "style_features": style_features,
            "label_tensor": label_tensor,
        }
    return payloads


def select_audit_indices(rows: Sequence[Dict[str, str]], audit_count: int) -> set[int]:
    groups: Dict[tuple[str, str], List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[(row["source_type"], row["length_bucket"])].append(idx)
    selected: List[int] = []
    for key in sorted(groups):
        selected.extend(groups[key][: max(1, audit_count // max(len(groups), 1))])
    if len(selected) < audit_count:
        seen = set(selected)
        for idx in range(len(rows)):
            if len(selected) >= audit_count:
                break
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
    return set(selected[:audit_count])


def save_audit_image(path: Path, image: Image.Image, label_text: str, pred_text: str, cer: float) -> None:
    font = None
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, size=14)
            break
    if font is None:
        font = ImageFont.load_default()

    word_img = image.convert("L")
    pad = 12
    info_h = 56
    canvas = Image.new("RGB", (word_img.width + pad * 2, word_img.height + info_h + pad * 2), "white")
    canvas.paste(word_img.convert("RGB"), (pad, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, word_img.height + pad + 6), f"target: {label_text}", fill=(20, 20, 20), font=font)
    draw.text((pad, word_img.height + pad + 24), f"pred: {pred_text}", fill=(20, 20, 20), font=font)
    draw.text((pad, word_img.height + pad + 42), f"CER: {cer:.4f}", fill=(20, 20, 20), font=font)
    canvas.save(path)


def image_sha256(image: Image.Image) -> str:
    data = image.convert("L").tobytes()
    return hashlib.sha256(data).hexdigest()


def aggregate_metrics(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    total = len(rows)
    overall_char_errors = sum(int(row["char_errors"]) for row in rows)
    overall_target_chars = sum(int(row["target_length"]) for row in rows)
    writer_to_errors: Dict[str, int] = defaultdict(int)
    writer_to_chars: Dict[str, int] = defaultdict(int)
    by_length: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    by_rare: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    by_source: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in rows:
        writer_to_errors[row["writer_id"]] += int(row["char_errors"])
        writer_to_chars[row["writer_id"]] += int(row["target_length"])
        by_length[row["length_bucket"]].append(row)
        by_rare["rare" if row["rare_letter_flag"] == "1" else "common"].append(row)
        by_source[row["source_type"]].append(row)

    def summarize(subset: Sequence[Dict[str, str]]) -> Dict[str, float | int]:
        char_errors = sum(int(row["char_errors"]) for row in subset)
        target_chars = sum(int(row["target_length"]) for row in subset)
        return {
            "count": len(subset),
            "char_errors": char_errors,
            "target_chars": target_chars,
            "cer": (char_errors / float(max(target_chars, 1))) if subset else math.nan,
        }

    summary = {
        "total_samples": total,
        "overall_char_errors": overall_char_errors,
        "overall_target_chars": overall_target_chars,
        "overall_cer": overall_char_errors / float(max(overall_target_chars, 1)),
        "writer_micro_cer": overall_char_errors / float(max(overall_target_chars, 1)),
        "writer_macro_cer": float(
            np.mean(
                [
                    writer_to_errors[writer_id] / float(max(writer_to_chars[writer_id], 1))
                    for writer_id in writer_to_errors
                ]
            )
        ),
        "cer_by_length_bucket": {bucket: summarize(by_length.get(bucket, [])) for bucket in LENGTH_BUCKETS},
        "cer_by_rare_letter": {key: summarize(by_rare.get(key, [])) for key in ("rare", "common")},
        "cer_by_source_type": {key: summarize(by_source.get(key, [])) for key in ("iv", "oov")},
        "unique_writers": len(writer_to_errors),
    }
    return summary


def write_table_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_markdown_table(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(header) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(cell) for cell in row) + " |\n")


def save_bar_chart(path_base: Path, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def render_audit_contact_sheet(audit_dir: Path, output_path: Path) -> None:
    images = sorted(audit_dir.glob("*.png"))
    if not images:
        return
    loaded = [Image.open(path).convert("RGB") for path in images]
    cols = 3
    cell_w = max(img.width for img in loaded)
    cell_h = max(img.height for img in loaded)
    rows = math.ceil(len(loaded) / cols)
    pad = 12
    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad), "white")
    for idx, img in enumerate(loaded):
        row = idx // cols
        col = idx % cols
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + pad)
        canvas.paste(img, (x, y))
    canvas.save(output_path)


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    artifacts = output_dir / "artifacts"
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    audit = output_dir / "audit"
    for path in (output_dir, artifacts, figures, tables, audit):
        path.mkdir(parents=True, exist_ok=True)
    return {"root": output_dir, "artifacts": artifacts, "figures": figures, "tables": tables, "audit": audit}


def main() -> None:
    args = parse_args()
    if args.samples_per_writer != args.iv_per_writer + args.oov_per_writer:
        raise ValueError("--samples_per_writer must equal --iv_per_writer + --oov_per_writer")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.eval_version}_{timestamp}"
    output_dir = args.output_root / run_name
    paths = ensure_dirs(output_dir)

    manifest_path = args.manifest_path or (output_dir / "manifest.tsv")
    config_path = paths["artifacts"] / "config.json"

    config = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "meta_file": str(args.meta_file),
        "style_path": str(args.style_path),
        "stable_dif_path": str(args.stable_dif_path),
        "trocr_model": args.trocr_model,
        "device": args.device,
        "trocr_device": args.trocr_device or args.device,
        "cfg_scale": args.cfg_scale,
        "eval_version": args.eval_version,
        "iv_per_writer": args.iv_per_writer,
        "oov_per_writer": args.oov_per_writer,
        "samples_per_writer": args.samples_per_writer,
        "audit_count": args.audit_count,
        "num_res_blocks": args.num_res_blocks,
        "timestamp": timestamp,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.eval_only:
        records = read_records(args.meta_file)
        manifest_rows = build_manifest_rows(
            records=records,
            eval_version=args.eval_version,
            meta_path=args.meta_file,
            checkpoint_path=args.checkpoint,
            iv_per_writer=args.iv_per_writer,
            oov_per_writer=args.oov_per_writer,
        )
        write_tsv(
            manifest_path,
            manifest_rows,
            fieldnames=[
                "cohort",
                "writer_id",
                "target_text",
                "target_norm",
                "source_type",
                "length_bucket",
                "rare_letter_flag",
                "rare_letter_set",
                "seed",
                "style_ref_filenames",
                "checkpoint_id",
                "meta_file_id",
            ],
        )
        print(f"Manifest written to {manifest_path}")

    if args.manifest_only:
        return

    manifest_rows = read_tsv(manifest_path)
    if args.limit is not None:
        manifest_rows = balanced_limit_rows(manifest_rows, args.limit)

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
    null_context = encode_text_context(
        unet=models["unet"],
        tokenizer=models["tokenizer"],
        texts=[""],
        device=models["device"],
        text_max_len=args.text_max_len,
    ) if args.cfg_scale > 1.0 else None

    audit_indices = select_audit_indices(manifest_rows, args.audit_count)
    sample_metrics: List[Dict[str, str]] = []
    trocr_images: List[Image.Image] = []
    trocr_meta: List[tuple[int, Image.Image]] = []
    start_time = time.time()

    def flush_trocr_batch() -> None:
        if not trocr_images:
            return
        preds = run_trocr_batch(
            images=trocr_images,
            processor=models["trocr_processor"],
            model=models["trocr_model"],
            device=models["trocr_device"],
        )
        for pred_text, (row_idx, image) in zip(preds, trocr_meta):
            row = manifest_rows[row_idx]
            target_norm, pred_norm, char_errors, cer = cer_components(row["target_text"], pred_text)
            metrics = {
                "row_index": str(row_idx),
                "cohort": row["cohort"],
                "writer_id": row["writer_id"],
                "target_text": row["target_text"],
                "target_norm": target_norm,
                "trocr_pred": pred_text,
                "trocr_pred_norm": pred_norm,
                "cer": f"{cer:.8f}",
                "char_errors": str(char_errors),
                "target_length": str(max(len(target_norm), 1)),
                "source_type": row["source_type"],
                "length_bucket": row["length_bucket"],
                "rare_letter_flag": row["rare_letter_flag"],
                "rare_letter_set": row["rare_letter_set"],
                "seed": row["seed"],
                "style_ref_filenames": row["style_ref_filenames"],
                "checkpoint_id": row["checkpoint_id"],
                "meta_file_id": row["meta_file_id"],
                "image_sha256": image_sha256(image),
            }
            sample_metrics.append(metrics)
            if row_idx in audit_indices:
                audit_name = f"{row_idx:04d}_{row['writer_id']}_{row['source_type']}_{target_norm}.png"
                save_audit_image(paths["audit"] / audit_name, image, row["target_text"], pred_text, cer)
        trocr_images.clear()
        trocr_meta.clear()

    for row_idx, row in enumerate(tqdm(manifest_rows, desc="Generating + OCR")):
        writer_idx = writer_id_map[row["writer_id"]]
        style_payload = style_payloads.get(writer_idx)
        if style_payload is None:
            raise KeyError(f"Missing precomputed style payload for writer {row['writer_id']} (idx={writer_idx})")

        set_global_seed(int(row["seed"]))
        image = generate_single_word(
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
        )

        trocr_images.append(image.convert("RGB"))
        trocr_meta.append((row_idx, image))
        if len(trocr_images) >= args.trocr_batch_size:
            flush_trocr_batch()

    flush_trocr_batch()
    runtime_sec = time.time() - start_time

    sample_metrics.sort(key=lambda row: int(row["row_index"]))
    metrics_path = output_dir / "sample_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(sample_metrics)

    summary = aggregate_metrics(sample_metrics)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_id": checkpoint_id(args.checkpoint),
            "meta_file": str(args.meta_file),
            "meta_file_id": meta_id(args.meta_file),
            "trocr_model": args.trocr_model,
            "dataset_root": str(args.dataset_root),
            "cfg_scale": args.cfg_scale,
            "eval_version": args.eval_version,
            "runtime_sec": runtime_sec,
            "timestamp": timestamp,
        }
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    main_rows = [
        ["overall", summary["total_samples"], summary["overall_char_errors"], summary["overall_target_chars"], f"{summary['overall_cer']:.6f}"],
        ["iv", summary["cer_by_source_type"]["iv"]["count"], summary["cer_by_source_type"]["iv"]["char_errors"], summary["cer_by_source_type"]["iv"]["target_chars"], f"{summary['cer_by_source_type']['iv']['cer']:.6f}"],
        ["oov", summary["cer_by_source_type"]["oov"]["count"], summary["cer_by_source_type"]["oov"]["char_errors"], summary["cer_by_source_type"]["oov"]["target_chars"], f"{summary['cer_by_source_type']['oov']['cer']:.6f}"],
        ["rare", summary["cer_by_rare_letter"]["rare"]["count"], summary["cer_by_rare_letter"]["rare"]["char_errors"], summary["cer_by_rare_letter"]["rare"]["target_chars"], f"{summary['cer_by_rare_letter']['rare']['cer']:.6f}"],
        ["common", summary["cer_by_rare_letter"]["common"]["count"], summary["cer_by_rare_letter"]["common"]["char_errors"], summary["cer_by_rare_letter"]["common"]["target_chars"], f"{summary['cer_by_rare_letter']['common']['cer']:.6f}"],
    ]
    length_rows = [
        [bucket, summary["cer_by_length_bucket"][bucket]["count"], summary["cer_by_length_bucket"][bucket]["char_errors"], summary["cer_by_length_bucket"][bucket]["target_chars"], f"{summary['cer_by_length_bucket'][bucket]['cer']:.6f}"]
        for bucket in LENGTH_BUCKETS
    ]
    write_table_csv(paths["tables"] / "cer_main_table.csv", ["group", "count", "char_errors", "target_chars", "cer"], main_rows)
    write_markdown_table(paths["tables"] / "cer_main_table.md", ["group", "count", "char_errors", "target_chars", "cer"], main_rows)
    write_table_csv(paths["tables"] / "cer_by_length_bucket.csv", ["length_bucket", "count", "char_errors", "target_chars", "cer"], length_rows)
    write_markdown_table(paths["tables"] / "cer_by_length_bucket.md", ["length_bucket", "count", "char_errors", "target_chars", "cer"], length_rows)

    save_bar_chart(
        paths["figures"] / "cer_overview",
        labels=[row[0] for row in main_rows],
        values=[float(row[4]) for row in main_rows],
        title="Generated-word TrOCR CER overview",
        ylabel="CER",
    )
    save_bar_chart(
        paths["figures"] / "cer_by_length_bucket",
        labels=list(LENGTH_BUCKETS),
        values=[summary["cer_by_length_bucket"][bucket]["cer"] for bucket in LENGTH_BUCKETS],
        title="Generated-word TrOCR CER by word length",
        ylabel="CER",
    )
    render_audit_contact_sheet(paths["audit"], paths["audit"] / "audit_contact_sheet.png")

    print(f"Results written to {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Sample metrics: {metrics_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
