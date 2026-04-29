import argparse
import csv
import json
import os
import random
import re

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineTokenizer, CanineModel
from tqdm import tqdm

from unet import UNetModel
from feature_extractor import ImageEncoder
from utils.word_cleanup_nafnet import NAFNetWordCleaner


def set_global_seed(seed: int) -> None:
    """Best-effort deterministic seeding for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# Characters that get special (smaller) rendering
PUNCTUATION = set(".,;:!?…()\"'«»-–—")


def strip_dp_prefix(state_dict):
    """Strip DataParallel 'module.' prefix from state dict keys."""
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "", 1)
        if new_key.startswith("text_encoder.module."):
            new_key = new_key.replace("text_encoder.module.", "text_encoder.", 1)
        new_sd[new_key] = v
    return new_sd


def detect_num_classes(state_dict):
    """Auto-detect num_classes from label_emb.weight shape."""
    for key in ["label_emb.weight", "module.label_emb.weight"]:
        if key in state_dict:
            return state_dict[key].shape[0]
    raise KeyError("Cannot find label_emb.weight in checkpoint")


def build_writer_id_map(meta_file):
    """Build writer_str -> writer_idx mapping matching UkrWordDataset."""
    writers = set()
    with open(meta_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            parts = row[0].replace(".png", "").split("-")
            if len(parts) >= 3:
                writers.add(parts[2])
    sorted_writers = sorted(writers)
    return {w: i for i, w in enumerate(sorted_writers)}


def load_style_images(dataset_root, meta_file, writer_indices, writer_id_map,
                      img_height=64, img_width=256, style_ref_override=None):
    """Load 5 reference style images per requested writer.
    Returns:
        style_refs: dict writer_idx -> tensor [5, 3, H, W]
        refs_used: dict writer_idx -> list[str] (filenames used as references)
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    words_dir = os.path.join(dataset_root, "words", "words")
    if not os.path.isdir(words_dir):
        raise FileNotFoundError(f"Words directory not found: {words_dir}")

    # Group filenames by writer
    writer_images = {}
    with open(meta_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            fname = row[0]
            parts = fname.replace(".png", "").split("-")
            if len(parts) >= 3:
                wstr = parts[2]
                if wstr not in writer_images:
                    writer_images[wstr] = []
                writer_images[wstr].append(fname)

    idx_to_str = {v: k for k, v in writer_id_map.items()}
    style_refs = {}
    refs_used = {}

    for wid in writer_indices:
        wstr = idx_to_str.get(wid)
        if wstr is None or wstr not in writer_images:
            continue
        fnames = []

        # Optional explicit override per writer (for controlled style conditioning).
        if style_ref_override and wstr in style_ref_override:
            override_list = style_ref_override.get(wstr) or []
            for ref_name in override_list:
                base = os.path.basename(ref_name)
                if base.endswith(".png"):
                    fnames.append(base)
                else:
                    fnames.append(base + ".png")

        if not fnames:
            fnames = writer_images[wstr][:5]

        while len(fnames) < 5:
            fnames.append(fnames[0])

        imgs = []
        used = []
        for fn in fnames:
            path = os.path.join(words_dir, fn)
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            w, h = img.size
            scale = min(img_height / float(h), img_width / float(w))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (img_width, img_height), (255, 255, 255))
            # Preserve aspect ratio and bottom-align so baseline/slant cues survive.
            canvas.paste(img, (0, img_height - new_h))
            img = canvas
            imgs.append(transform(img))
            used.append(fn)

        if len(imgs) == 5:
            style_refs[wid] = torch.stack(imgs)
            refs_used[wid] = used

    return style_refs, refs_used


def crop_whitespace(img_pil):
    """Crop only horizontal (left/right) whitespace using Otsu binarisation."""
    img_gray = np.array(img_pil)
    _, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    if coords is None:
        return img_gray
    x, y, w, h = cv2.boundingRect(coords)
    return img_gray[:, x:x+w]


def erase_bottom_artifacts(img_gray):
    """Erase thin horizontal underline artifact rows at the bottom of a word image.

    Scans rows from the bottom upward. A row is considered an artifact if its ink
    spans >40% of the image width but has <18% pixel density (thin line, not a letter).
    Very sparse rows (density < 10%) are skipped so a few stray edge pixels don't
    stop the scan before the underline is reached.
    Real letter content (density >= 10% and span <= 40%) stops the scan.
    """
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = thresh.shape
    out = img_gray.copy()
    for row in range(h - 1, -1, -1):
        ink_cols = np.where(thresh[row] > 0)[0]
        if len(ink_cols) == 0:
            continue
        density = len(ink_cols) / w
        span = (ink_cols[-1] - ink_cols[0]) / w
        if span > 0.4 and density < 0.18:
            out[row, :] = 255          # underline row — erase
        elif density < 0.10:
            continue                   # too sparse to signal real letter content — skip
        else:
            break                      # real ink starts here
    return out


def remove_underline(img_gray, underline_y=None):
    """Remove underline artifact using dual-threshold erasure with descender preservation.

    Uses thresh_underline=249 to detect pixels belonging to the line (catches faint
    VAE artifacts at values 145–249 that Otsu and Sauvola both miss). Uses
    thresh_letter=200 to identify real letter strokes above the line band. Only erases
    columns where the line exists AND no letter ink is present directly above
    (i.e. no descender crossing through the line band).
    """
    y = underline_y if underline_y is not None else detect_underline(img_gray)
    if y is None:
        return img_gray

    h, w = img_gray.shape
    out = img_gray.copy()

    y0 = max(0, y - 2)
    y1 = min(h, y + 3)          # 5-row erase band centred on the underline

    # thresh_letter: real pen strokes are typically < 200; background and faint line are ≥ 200
    _, thresh_letter = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    # thresh_underline: catches dark (< 180) and very faint (180–252) line pixels;
    # matches detect_underline pass-2 threshold so everything detected is also erased
    _, thresh_underline = cv2.threshold(img_gray, 252, 255, cv2.THRESH_BINARY_INV)

    # Check only the 2 rows directly above the band: true descenders reach row y0-1;
    # letter baseline strokes typically end 3+ rows above y0, leaving a gap.
    check_start = max(0, y0 - 2)
    above_has_ink = thresh_letter[check_start:y0, :].any(axis=0)   # shape [W], bool
    in_band       = thresh_underline[y0:y1, :].any(axis=0)         # underline pixels in band

    erase_cols = in_band & ~above_has_ink
    for row in range(y0, y1):
        out[row, erase_cols] = 255
    return out


def erase_underline_rows(img_gray, underline_y=None, n_rows=4):
    """Blank from the detected underline row to the bottom.

    When underline_y is provided (detected by detect_underline), erases from
    underline_y-1 downward — targeted, 2-5 rows only. Falls back to bottom
    n_rows when no underline was detected.
    """
    out = img_gray.copy()
    h = out.shape[0]
    if underline_y is not None:
        erase_from = max(0, underline_y - 1)
    else:
        erase_from = max(0, h - n_rows)
    out[erase_from:, :] = 255
    return out


def measure_ink_top(img_gray):
    """First row from top where ink spans >=30% of image width (ascender/cap top).
    Used to measure letter body height for size normalization across words."""
    _, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
    h, w = thresh.shape
    min_span = max(int(w * 0.30), 3)
    for row in range(h):
        ink_cols = np.where(thresh[row] > 0)[0]
        if len(ink_cols) >= 2 and int(ink_cols[-1]) - int(ink_cols[0]) >= min_span:
            return row
    return 0


def word_alpha_length(word):
    """Count alphabetic characters used for short-word scale priors."""
    return sum(1 for ch in word if ch.isalpha())


def _safe_token_name(token: str) -> str:
    token = re.sub(r"\s+", "_", token.strip())
    token = re.sub(r"[^0-9A-Za-zА-Яа-яІіЇїЄєҐґ'_-]+", "", token)
    return token or "token"


def _save_debug_tokens(stage_dir, words, images):
    os.makedirs(stage_dir, exist_ok=True)
    for i, (word, img) in enumerate(zip(words, images)):
        Image.fromarray(img).save(
            os.path.join(stage_dir, f"{i:02d}_{_safe_token_name(word)}.png")
        )


def save_sentence_debug_bundle(debug_dir, writer_str, expanded_words,
                               raw_word_images, cleaned_word_images,
                               punct_flags, word_shifts, max_bottom,
                               h_scales, final_paragraph,
                               gen_height, canvas_height, max_line_width):
    os.makedirs(debug_dir, exist_ok=True)

    _save_debug_tokens(os.path.join(debug_dir, "raw_words"), expanded_words, raw_word_images)
    _save_debug_tokens(os.path.join(debug_dir, "cleaned_words"), expanded_words, cleaned_word_images)

    aligned_strip = stitch_paragraph(
        raw_word_images, expanded_words,
        max_line_width=max_line_width,
        gen_height=gen_height,
        canvas_height=canvas_height,
        shifts=word_shifts,
        h_scales=[1.0] * len(expanded_words),
        ref_baseline=max_bottom,
    )
    aligned_strip.save(os.path.join(debug_dir, "aligned_strip.png"))

    final_paragraph.save(os.path.join(debug_dir, "final_strip.png"))

    token_debug = []
    for i, word in enumerate(expanded_words):
        token_debug.append({
            "index": i,
            "token": word,
            "is_punctuation": bool(punct_flags[i]),
            "alpha_length": int(word_alpha_length(word)) if not punct_flags[i] else 0,
            "shift_px": int(word_shifts[i]),
            "height_scale": float(h_scales[i]),
        })

    payload = {
        "writer_id": writer_str,
        "baseline_y": int(max_bottom),
        "tokens": token_debug,
    }
    with open(os.path.join(debug_dir, "debug_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_sentence_geometry_debug_bundle(debug_dir, writer_str, expanded_words,
                                        raw_word_images, cleaned_word_images,
                                        final_paragraph, geometry_payload):
    os.makedirs(debug_dir, exist_ok=True)
    _save_debug_tokens(os.path.join(debug_dir, "raw_words"), expanded_words, raw_word_images)
    _save_debug_tokens(os.path.join(debug_dir, "cleaned_words"), expanded_words, cleaned_word_images)
    final_paragraph.save(os.path.join(debug_dir, "aligned_strip.png"))
    final_paragraph.save(os.path.join(debug_dir, "final_strip.png"))

    payload = {
        "writer_id": writer_str,
        **geometry_payload,
    }
    with open(os.path.join(debug_dir, "debug_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def detect_underline(img_gray):
    """Return Y of the underline row using 3-pass approach.

    Pass 1: threshold 249, span >= 35% — dark underlines
    Pass 2: threshold 252, span >= 35% — mid-faint underlines
    Pass 3: densest row in bottom 5 rows at threshold 254 — very faint (252-253)
    """
    h, w = img_gray.shape
    zone_start = int(h * 0.94)   # bottom 6% only — avoids letter strokes at rows 55-60

    for thresh in (249, 252):
        for row in range(h - 1, zone_start - 1, -1):
            ink_cols = np.where(img_gray[row] < thresh)[0]
            if len(ink_cols) < 2:
                continue
            span = (int(ink_cols[-1]) - int(ink_cols[0])) / w
            if span >= 0.35:
                return row

    best_row, best_count = -1, 0
    for row in range(max(0, h - 5), h):
        count = int(np.sum(img_gray[row] < 254))
        if count > best_count:
            best_count = count
            best_row = row
    if best_count >= max(int(w * 0.10), 3):
        return best_row

    return None


def detect_baseline_and_clean(img_gray):
    """Detect and erase underline artifacts before any geometry measurement."""
    y = detect_underline(img_gray)
    cleaned = remove_underline(img_gray, y)
    return cleaned, y


def _threshold_ink_mask(img_gray):
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def _bounding_box_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def measure_word_geometry(img_gray):
    """Measure robust baseline/body geometry on a cleaned grayscale word image."""
    ink = _threshold_ink_mask(img_gray)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 8 or h < 2:
            continue
        comps.append((i, x, y, w, h, area))

    h, w = img_gray.shape
    if not comps:
        crop = img_gray.copy()
        baseline = max(0, h - 8)
        body_top = max(0, baseline - max(8, h // 2))
        body_h = max(4, baseline - body_top)
        return {
            "crop": crop,
            "crop_x0": 0,
            "crop_y0": 0,
            "baseline": baseline,
            "body_top": body_top,
            "body_h": body_h,
            "crop_h": int(crop.shape[0]),
            "crop_w": int(crop.shape[1]),
        }

    largest_area = max(area for _, _, _, _, _, area in comps)
    body_ids = {
        i for i, x, y, cw, ch, area in comps
        if area >= max(12, int(0.12 * largest_area)) or ch >= 8
    }

    clean = np.zeros_like(ink)
    body = np.zeros_like(ink)
    for i, x, y, cw, ch, area in comps:
        mask = labels == i
        clean[mask] = 255
        if i in body_ids:
            body[mask] = 255

    bbox = _bounding_box_from_mask(clean)
    if bbox is None:
        return {
            "crop": img_gray.copy(),
            "crop_x0": 0,
            "crop_y0": 0,
            "baseline": max(0, h - 8),
            "body_top": max(0, h // 3),
            "body_h": max(4, h // 2),
            "crop_h": h,
            "crop_w": w,
        }
    x0, y0, x1, y1 = bbox
    pad = 2
    x0 = max(0, x0 - pad)
    x1 = min(w, x1 + pad)
    y0 = max(0, y0 - pad)
    y1 = min(h, y1 + pad)

    bottoms = []
    for x in range(x0, x1):
        col = np.where(clean[:, x] > 0)[0]
        if len(col) > 0:
            bottoms.append(int(col.max()))
    baseline = int(np.percentile(bottoms, 65)) if bottoms else (y1 - 1)

    body_ys = np.where(body > 0)[0]
    if len(body_ys) > 0:
        body_top = int(np.percentile(body_ys, 5))
    else:
        body_top = y0
    body_h = max(4, baseline - body_top)
    crop = img_gray[y0:y1, x0:x1]

    return {
        "crop": crop,
        "crop_x0": int(x0),
        "crop_y0": int(y0),
        "baseline": int(baseline),
        "body_top": int(body_top),
        "body_h": int(body_h),
        "crop_h": int(crop.shape[0]),
        "crop_w": int(crop.shape[1]),
    }


def place_punctuation_on_canvas(ch, canvas_height, canvas_baseline,
                                punct_bank=None, writer_str=None,
                                standalone=False, target_body_h=None):
    """Render punctuation directly into the sentence canvas geometry."""
    punct = sample_punctuation(
        ch,
        img_height=canvas_height,
        punct_bank=punct_bank,
        writer_str=writer_str,
        baseline_y=canvas_baseline,
        standalone=standalone,
        target_body_h=target_body_h,
    )
    if punct is None:
        return np.full((canvas_height, 6), 255, dtype=np.uint8), {
            "baseline": canvas_baseline,
            "body_h": 0,
            "scale": 1.0,
            "top": 0,
            "crop_h": 0,
            "crop_w": 0,
        }
    crop = crop_whitespace(Image.fromarray(punct))
    if crop.ndim == 1:
        crop = crop.reshape(canvas_height, -1)
    bbox = _bounding_box_from_mask(_threshold_ink_mask(crop))
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        crop = crop[y0:y1, x0:x1]
    tile = np.full((canvas_height, crop.shape[1] + 4), 255, dtype=np.uint8)
    h = crop.shape[0]
    if ch == '-':
        top = max(0, min(canvas_height - h, canvas_baseline - int(h * 0.55)))
    elif ch in (',', '.'):
        if ch == ',':
            top = max(0, min(canvas_height - h, canvas_baseline - int(h * 0.28)))
        else:  # period
            top = max(0, min(canvas_height - h, canvas_baseline - max(2, int(h * 0.08))))
    else:
        top = max(0, min(canvas_height - h, canvas_baseline - h))
    tile[top:top + h, 2:2 + crop.shape[1]] = crop
    return tile, {
        "baseline": int(canvas_baseline),
        "body_h": int(h),
        "scale": 1.0,
        "top": int(top),
        "crop_h": int(h),
        "crop_w": int(crop.shape[1]),
    }


def compose_sentence_geometry(word_images, expanded_words, punct_flags,
                              punct_standalone_flags=None, punct_bank=None,
                              writer_str=None, canvas_height=104, gap_width=16):
    """Compose a sentence from cleaned glyph crops around a shared baseline."""
    if punct_standalone_flags is None:
        punct_standalone_flags = [False] * len(expanded_words)

    geoms = [
        None if is_punct else measure_word_geometry(img)
        for img, is_punct in zip(word_images, punct_flags)
    ]

    target_pool = [
        g["body_h"]
        for g, word, is_punct in zip(geoms, expanded_words, punct_flags)
        if g is not None and not is_punct and word_alpha_length(word) >= 4
    ]
    if not target_pool:
        target_pool = [g["body_h"] for g in geoms if g is not None]
    if not target_pool:
        target_pool = [int(round(canvas_height * 0.45))]

    target_body_h = float(np.percentile(target_pool, 30))
    canvas_baseline = int(np.clip(round(target_body_h * 1.25), 18, canvas_height - 8))

    parts = [np.full((canvas_height, gap_width), 255, dtype=np.uint8)]
    token_debug = []

    for img, word, is_punct, is_standalone, geom in zip(
        word_images, expanded_words, punct_flags, punct_standalone_flags, geoms
    ):
        if is_punct:
            tile, punct_debug = place_punctuation_on_canvas(
                word,
                canvas_height=canvas_height,
                canvas_baseline=canvas_baseline,
                punct_bank=punct_bank,
                writer_str=writer_str,
                standalone=is_standalone,
                target_body_h=target_body_h,
            )
            token_debug.append({
                "token": word,
                "is_punctuation": True,
                **punct_debug,
            })
            parts.extend([tile, np.full((canvas_height, gap_width), 255, dtype=np.uint8)])
            continue

        alpha_len = word_alpha_length(word)
        _ALPHA_CAP = {1: 0.78, 2: 0.82, 3: 0.90}
        cap = _ALPHA_CAP.get(alpha_len, 1.0)
        scale = float(np.clip(cap * target_body_h / float(max(geom["body_h"], 1)), 0.40, 1.15))
        crop = geom["crop"]
        new_w = max(4, int(round(crop.shape[1] * scale)))
        new_h = max(4, int(round(crop.shape[0] * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

        baseline_in_crop = geom["baseline"] - geom["crop_y0"]
        baseline_scaled = int(round(baseline_in_crop * scale))
        top = int(np.clip(canvas_baseline - baseline_scaled, 0, canvas_height - new_h))

        tile = np.full((canvas_height, new_w + 4), 255, dtype=np.uint8)
        tile[top:top + new_h, 2:2 + new_w] = resized
        parts.extend([tile, np.full((canvas_height, gap_width), 255, dtype=np.uint8)])
        token_debug.append({
            "token": word,
            "is_punctuation": False,
            "baseline": int(canvas_baseline),
            "body_h": int(geom["body_h"]),
            "body_top": int(geom["body_top"]),
            "crop_h": int(geom["crop_h"]),
            "crop_w": int(geom["crop_w"]),
            "scale": float(scale),
            "top": int(top),
        })

    paragraph = Image.fromarray(np.concatenate(parts, axis=1))
    payload = {
        "baseline_y": int(canvas_baseline),
        "target_body_h": float(target_body_h),
        "tokens": token_debug,
    }
    return paragraph, payload


def align_to_baseline(word_images, threshold_ratio=0.25, is_punct=None, underline_ys=None):
    """Shift word images vertically so their text baselines align.

    Baseline per word is taken from underline_ys[i] if provided (Hough-detected
    underline Y), otherwise falls back to span-based text_bottom() detection.
    Words with ink sitting higher are shifted down (white rows added at top,
    equivalent rows removed from the already-clean bottom). All images keep the
    same height.

    Args:
        is_punct: optional list of bools — bank punctuation images are passed
            through unchanged and excluded from max_bottom computation.
        underline_ys: optional list of int|None — Hough-detected underline Y per
            word; None entries fall back to text_bottom().

    Returns:
        (aligned_images, max_bottom) — aligned list and the baseline Y used.
    """
    def text_bottom(img_gray):
        _, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
        h, w = thresh.shape
        min_span = max(int(w * 0.35), 4)
        for row in range(h - 1, -1, -1):
            ink_cols = np.where(thresh[row] > 0)[0]
            if len(ink_cols) < 2:
                continue
            span_px = int(ink_cols[-1]) - int(ink_cols[0])
            if span_px >= min_span:
                density = len(ink_cols) / float(w)
                # Confidence: broad dense rows are more likely true body-bottom rows.
                conf = (span_px / float(w)) * min(1.0, density / 0.20)
                return row, float(max(conf, 0.05))
        for row in range(h - 1, -1, -1):
            if thresh[row].any():
                return row, 0.05
        return 0, 0.05

    def weighted_median(values, weights):
        if not values:
            return 0
        pairs = sorted(zip(values, weights), key=lambda x: x[0])
        total = float(sum(w for _, w in pairs))
        if total <= 0:
            return int(np.median(values))
        acc = 0.0
        half = total * 0.5
        for v, w in pairs:
            acc += float(w)
            if acc >= half:
                return int(v)
        return int(pairs[-1][0])

    bottoms = []
    confs = []
    for i, img in enumerate(word_images):
        b_text, c_text = text_bottom(img)
        if underline_ys and underline_ys[i] is not None:
            # Keep underline as primary anchor and blend in text-bottom only
            # when the two estimates are already close.
            b_ul = int(underline_ys[i])
            delta = abs(b_ul - b_text)
            if delta <= 3:
                b = int(round(0.7 * b_ul + 0.3 * b_text))
            elif delta <= 6:
                b = int(round(0.85 * b_ul + 0.15 * b_text))
            else:
                b = b_ul
            c = max(c_text, 1.0)
            bottoms.append(b)
            confs.append(c)
        else:
            bottoms.append(b_text)
            confs.append(c_text)

    if is_punct:
        word_bottoms = [b for b, p in zip(bottoms, is_punct) if not p]
        word_confs = [c for c, p in zip(confs, is_punct) if not p]
    else:
        word_bottoms = bottoms
        word_confs = confs
    max_bottom = weighted_median(word_bottoms, word_confs) if word_bottoms else max(bottoms)

    # Initial per-word downward shifts (baseline target is robust median, not max).
    shifts = []
    for i, bot in enumerate(bottoms):
        if is_punct and is_punct[i]:
            shifts.append(None)
        else:
            shifts.append(max(0, max_bottom - bot))

    # Clamp extreme outlier shifts so one bad baseline estimate does not cause
    # visible sentence jumps in neighboring words.
    word_shifts = [s for s in shifts if s is not None]
    if word_shifts:
        p90 = int(np.percentile(word_shifts, 90))
        max_allowed = min(8, max(4, p90 + 1))
        for i, s in enumerate(shifts):
            if s is not None:
                shifts[i] = min(int(s), max_allowed)

    # Local smoothing: enforce gentle baseline drift instead of abrupt jumps.
    smoothed = shifts[:]
    for i, s in enumerate(shifts):
        if s is None:
            continue
        neighbors = []
        for j in range(max(0, i - 2), min(len(shifts), i + 3)):
            if shifts[j] is not None:
                neighbors.append(shifts[j])
        if neighbors:
            neigh_med = float(np.median(neighbors))
            cand = 0.65 * float(s) + 0.35 * neigh_med
            smoothed[i] = int(round(np.clip(cand, float(s) - 1.0, float(s) + 1.0)))

    # Final continuity pass: neighboring words should not differ by more than
    # one pixel unless punctuation sits between them.
    prev_i = None
    for i, s in enumerate(smoothed):
        if s is None:
            continue
        if prev_i is not None:
            prev_s = smoothed[prev_i]
            if prev_s is not None and abs(int(s) - int(prev_s)) > 1:
                smoothed[i] = int(prev_s + np.sign(int(s) - int(prev_s)))
        prev_i = i

    # Keep punctuation untouched here; punctuation follows previous word shift later.
    final_shifts = [0 if s is None else int(s) for s in smoothed]
    return word_images, final_shifts, max_bottom


def normalize_ink_brightness(word_images):
    """Normalize ink brightness across word images so stroke density is consistent.

    Uses the first word as reference. Scales each word's ink pixels so their
    mean distance-from-white matches the reference (darker reference → darken others).
    Background pixels are left at 255.
    """
    if len(word_images) <= 1:
        return word_images

    def ink_distance_mean(img):
        """Mean of (255 - pixel) for ink pixels — higher = darker ink."""
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_pixels = img[thresh > 0]
        return float((255 - ink_pixels).mean()) if len(ink_pixels) > 0 else 127.0

    ref_dist = ink_distance_mean(word_images[0])
    normalized = [word_images[0]]

    for img in word_images[1:]:
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_mask = thresh > 0
        if ink_mask.sum() == 0 or ref_dist <= 0:
            normalized.append(img)
            continue
        cur_dist = float((255 - img[ink_mask]).mean())
        if cur_dist <= 0:
            normalized.append(img)
            continue
        # Scale distance-from-white so cur matches ref
        scale = ref_dist / cur_dist
        out = img.astype(np.float32)
        out[ink_mask] = 255 - (255 - out[ink_mask]) * scale
        out = np.clip(out, 0, 255).astype(np.uint8)
        normalized.append(out)

    return normalized


_PUNCT_SUBDIR = {
    ',': 'comma',
    '.': 'period',
    '?': 'question',
    '!': 'exclaim',
    ':': 'colon',
    ';': 'semicolon',
    '-': 'hyphen',
}

_STANDALONE_PUNCT_SUBDIR = {
    '-': 'dash',
}

# (target_height_fraction, top_offset_fraction)
# top_offset_fraction=None → bottom-align at baseline
_PUNCT_LAYOUT = {
    ',': (0.35, None),
    '.': (0.20, None),
    '?': (0.70, 0.05),
    '!': (0.70, 0.05),
    ':': (0.45, 0.20),
    ';': (0.45, 0.20),
    '-': (0.12, 0.40),   # short horizontal mark at mid-height (x-height)
}

# Characters that appear INLINE inside words (split word, insert mark from bank)
INLINE_SPLIT_CHARS = {'-'}

_PUNCT_BANK_CACHE: dict = {}  # subdir_path -> {'__all__': [...], writer_id: [...], ...}


def _load_punct_subdir(subdir):
    """Build per-writer file index for a bank subdirectory.

    Filenames are expected as '{char}_{writer_id}_{index}.png'.
    Falls back gracefully to old format (no writer_id segment).

    Also pre-computes ink_px per file and median ink_px per writer so that
    sample_punctuation can do size-aware style matching for writers that have
    no exact marks in the bank.
    """
    by_writer: dict[str, list] = {}
    all_files: list = []
    ink_by_file: dict[str, int] = {}
    if not os.path.isdir(subdir):
        return {'__all__': [], '__ink__': {}, '__writer_ink__': {}}
    for f in sorted(os.listdir(subdir)):
        if not f.endswith('.png') or f.startswith('_'):
            continue
        path = os.path.join(subdir, f)
        all_files.append(path)
        # measure ink_px for size matching
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            ink_by_file[path] = int((bw > 0).sum())
        parts = f[:-4].split('_')   # strip .png
        if len(parts) >= 3:         # {char}_{writer_id}_{index}
            wid = parts[1]
            by_writer.setdefault(wid, []).append(path)
    # per-writer median ink_px
    writer_ink = {
        wid: float(np.median([ink_by_file.get(p, 0) for p in paths]))
        for wid, paths in by_writer.items()
    }
    return {'__all__': all_files, '__ink__': ink_by_file,
            '__writer_ink__': writer_ink, **by_writer}


def _synthesize_dash(img_height, standalone=True):
    """Fallback handwritten-like dash when no hyphen bank is available.

    This is intentionally longer than the inline hyphen used inside compound
    words, which makes standalone clause dashes read correctly in thesis
    figures even without a full punctuation bank.
    """
    width_frac = 0.62 if standalone else 0.34
    target_w = max(int(img_height * width_frac), 18)
    target_h = max(int(img_height * 0.12), 4)
    canvas = np.full((img_height, target_w + 6), 255, dtype=np.uint8)
    y_center = int(img_height * 0.46)
    thickness = max(2, target_h // 2)
    # Slightly slanted, anti-aliased line looks less mechanical than a flat bar.
    cv2.line(
        canvas,
        (3, y_center + 1),
        (target_w + 1, max(0, y_center - 1)),
        color=0,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return canvas


def _punct_available(ch, punct_bank):
    """Whether a punctuation mark can be rendered either from bank or fallback."""
    if ch == '-':
        return True
    if not punct_bank or ch not in _PUNCT_SUBDIR:
        return False
    subdir = os.path.join(punct_bank, _PUNCT_SUBDIR.get(ch, ''))
    return os.path.isdir(subdir) and any(f.endswith('.png') for f in os.listdir(subdir))


def _size_compatible_pool(files: list, cache: dict, target_body_h: float) -> list:
    """Filter a file list to marks whose ink size is compatible with target_body_h.

    Reference: median comma ink_px ≈ 112 at median body_h ≈ 28.
    Expected ink scales roughly as body_h^1.5 (area grows faster than linear).
    Tolerance is ±55% to keep enough candidates even for extreme writer sizes.
    Returns the filtered list, or the original list if too few pass the filter.
    """
    ref_ink = 112.0 * (target_body_h / 28.0) ** 1.5
    lo, hi = ref_ink * 0.45, ref_ink * 1.55
    ink_map = cache.get('__ink__', {})
    filtered = [f for f in files if lo <= ink_map.get(f, ref_ink) <= hi]
    return filtered if len(filtered) >= 4 else files


def _nearest_writer_pool(cache: dict, target_body_h: float, k: int = 4) -> list:
    """Return a pooled file list from the K writers whose median ink size is
    closest to what we'd expect for target_body_h.  Used as an intermediate
    fallback between writer-exact and fully-global sampling.
    """
    writer_ink: dict = cache.get('__writer_ink__', {})
    if not writer_ink:
        return []
    ref_ink = 112.0 * (target_body_h / 28.0) ** 1.5
    ranked = sorted(writer_ink.items(), key=lambda kv: abs(kv[1] - ref_ink))
    pool: list = []
    for wid, _ in ranked[:k]:
        pool.extend(cache.get(wid, []))
    return pool


def sample_punctuation(ch, img_height, punct_bank=None, writer_str=None,
                       baseline_y=None, standalone=False, target_body_h=None):
    """Return a grayscale numpy array (img_height tall) with a handwritten punctuation mark.

    Sampling priority (first non-empty pool wins):
      1. Writer-exact: marks from the same writer ID
      2. Nearest-writer: marks from the K writers with most similar stroke weight
         (estimated via target_body_h → expected ink area)
      3. Size-filtered global: global pool filtered to size-compatible marks
      4. Full global pool (random)

    baseline_y: if provided, trailing marks are bottom-anchored at this row.
    target_body_h: sentence body height in px — used for size-aware style matching.
    """
    if not punct_bank or ch not in _PUNCT_SUBDIR:
        if ch == '-':
            return _synthesize_dash(img_height, standalone=standalone)
        return None

    subdir_name = _PUNCT_SUBDIR[ch]
    if standalone and ch in _STANDALONE_PUNCT_SUBDIR:
        standalone_subdir = os.path.join(punct_bank, _STANDALONE_PUNCT_SUBDIR[ch])
        if os.path.isdir(standalone_subdir) and any(f.endswith('.png') for f in os.listdir(standalone_subdir)):
            subdir_name = _STANDALONE_PUNCT_SUBDIR[ch]

    subdir = os.path.join(punct_bank, subdir_name)
    if subdir not in _PUNCT_BANK_CACHE:
        _PUNCT_BANK_CACHE[subdir] = _load_punct_subdir(subdir)

    cache = _PUNCT_BANK_CACHE[subdir]

    # 1. writer-exact
    files = (cache.get(writer_str) or []) if writer_str else []
    # 2. nearest-writer by stroke weight
    if not files and target_body_h is not None:
        files = _nearest_writer_pool(cache, target_body_h)
    # 3. size-filtered global
    if not files and target_body_h is not None:
        files = _size_compatible_pool(cache.get('__all__', []), cache, target_body_h)
    # 4. full global
    if not files:
        files = cache.get('__all__', [])
    if not files:
        if ch == '-':
            return _synthesize_dash(img_height, standalone=standalone)
        return None

    crop = cv2.imread(random.choice(files), cv2.IMREAD_GRAYSCALE)
    if crop is None:
        return None

    height_frac, top_frac = _PUNCT_LAYOUT.get(ch, (0.40, None))
    h, w = crop.shape
    target_h = max(int(img_height * height_frac), 4)
    target_w = max(int(w * target_h / h), 3)
    crop_resized = cv2.resize(crop, (target_w, target_h))

    canvas = np.full((img_height, target_w + 4), 255, dtype=np.uint8)
    if ch == '-':
        # inline mark: keep relative mid-height regardless of baseline
        y_off = int(img_height * top_frac)
    elif baseline_y is not None:
        if ch in (',', '.'):
            # comma/period: body straddles baseline — 60% above, tail hangs 40% below
            y_off = baseline_y - int(target_h * 0.6)
        else:
            y_off = baseline_y - target_h
    elif top_frac is None:
        y_off = img_height - target_h - int(img_height * 0.08)
    else:
        y_off = int(img_height * top_frac)
    y_off = max(0, min(y_off, img_height - target_h))
    canvas[y_off:y_off + target_h, 2:2 + target_w] = crop_resized
    return canvas


def split_word_for_generation(word, punct_bank, img_height):
    """Split a word on inline characters that have bank images (e.g. hyphens in compound words).

    Returns a list of (text_or_char, is_punct_mark) tuples.
    Parts where is_punct_mark=True should be rendered from the bank;
    parts where is_punct_mark=False should be generated via diffusion.

    If no bank exists for an inline char, the word is NOT split on that char
    (it is passed whole to the diffusion model).
    """
    # Determine which inline chars actually have a bank
    splittable = set()
    for ch in INLINE_SPLIT_CHARS:
        if _punct_available(ch, punct_bank):
            splittable.add(ch)

    if not splittable:
        return [(word, False)]

    parts = []
    current = ''
    for ch in word:
        if ch in splittable:
            if current:
                parts.append((current, False))
                current = ''
            parts.append((ch, True))
        else:
            current += ch
    if current:
        parts.append((current, False))
    return parts


@torch.inference_mode()
def encode_text_context(unet, tokenizer, texts, device, text_max_len=40):
    """Encode text once for reuse across diffusion steps."""
    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=text_max_len,
    ).to(device)
    context = unet.text_encoder(**tokens).last_hidden_state
    if unet.cont_dim == 320:
        context = unet.text_lin(context)
    return context


@torch.inference_mode()
def generate_single_word(word, unet, vae, style_extractor, tokenizer,
                         noise_scheduler, style_ref, writer_idx, device,
                         cfg_scale=5.0, img_height=64, img_width=256,
                         text_max_len=40, style_features=None,
                         label_tensor=None, text_context=None,
                         null_context=None):
    """Generate a single word image. Returns a grayscale PIL Image."""
    unet.eval()

    if text_context is None:
        text_context = encode_text_context(
            unet=unet,
            tokenizer=tokenizer,
            texts=[word],
            device=device,
            text_max_len=text_max_len,
        )

    if cfg_scale > 1.0 and null_context is None:
        null_context = encode_text_context(
            unet=unet,
            tokenizer=tokenizer,
            texts=[""],
            device=device,
            text_max_len=text_max_len,
        )

    n = 1
    if style_features is None:
        style_batch = style_ref.unsqueeze(0).to(device)  # [1, 5, 3, H, W]
        style_flat = style_batch.reshape(-1, 3, img_height, img_width)
        style_features = style_extractor(style_flat).to(device)
    else:
        style_features = style_features.to(device)

    if label_tensor is None:
        label_tensor = torch.tensor([writer_idx], dtype=torch.long, device=device)
    else:
        label_tensor = label_tensor.to(device)

    x = torch.randn(n, 4, img_height // 8, img_width // 8).to(device)

    noise_scheduler.set_timesteps(50)
    for t_step in noise_scheduler.timesteps:
        t = (torch.ones(n, device=device) * t_step.item()).long()
        noise_pred = unet(
            x, t, text_context, label_tensor,
            mix_rate=None,
            style_extractor=style_features,
        )
        if cfg_scale > 1.0 and null_context is not None:
            noise_pred_uncond = unet(
                x, t, null_context, label_tensor,
                mix_rate=None,
                style_extractor=style_features,
            )
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)
        x = noise_scheduler.step(noise_pred, t_step, x).prev_sample

    latents = x / 0.18215
    images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)

    img_pil = torchvision.transforms.ToPILImage()(images[0])
    return img_pil.convert("L")


def stitch_paragraph(word_images, words, max_line_width=900, gen_height=64,
                     canvas_height=88, gap_width=16, shifts=None,
                     h_scales=None, ref_baseline=None):
    """Stitch word images into a paragraph with line-wrapping.

    Words are scaled proportionally to character count (longer words wider).
    Each image is padded from gen_height to canvas_height (extra space below for descenders).

    Args:
        shifts: per-word downward shift from baseline alignment (top padding).
        h_scales: per-word height scale in (0, 1] — words with oversized glyphs are
            reduced; others kept at 1.0. Width scales proportionally so aspect ratio
            is preserved. Baseline alignment is maintained via pad_top recalculation.
        ref_baseline: the common canvas baseline row (= max_bottom from alignment).
            Required when h_scales are provided.

    Returns:
        PIL Image of the full paragraph
    """
    # Calibrate average character width from the longest word
    longest_idx = max(range(len(words)), key=lambda i: len(words[i]))
    longest_word_len = len(words[longest_idx])
    longest_img = Image.fromarray(word_images[longest_idx])
    avg_char_width = longest_img.width / longest_word_len

    pad_buffer = canvas_height - gen_height  # total extra rows available

    scaled_words = []
    for idx, (word, img_arr) in enumerate(zip(words, word_images)):
        img_pil = Image.fromarray(img_arr)
        shift = (shifts[idx] if shifts else 0)
        shift = min(shift, pad_buffer)
        h_scale = (h_scales[idx] if h_scales else 1.0)

        if word in PUNCTUATION:
            pad_top = shift
            pad_bot = pad_buffer - shift
            padded = np.pad(img_arr, ((pad_top, pad_bot), (0, 0)),
                            mode='constant', constant_values=255)
        else:
            n_alpha = word_alpha_length(word)
            if n_alpha <= 1:
                width_chars = 1.05
            elif n_alpha == 2:
                width_chars = 1.65
            elif n_alpha == 3:
                width_chars = 2.45
            elif n_alpha == 4:
                width_chars = 3.55
            else:
                width_chars = float(len(word))
            scaled_w = max(int(avg_char_width * width_chars), 4)
            target_w = max(int(scaled_w * h_scale), 4)
            target_h = max(int(gen_height * h_scale), 4)
            resized = np.array(img_pil.resize((target_w, target_h)))

            if ref_baseline is not None:
                # Recompute padding so the word baseline stays at ref_baseline
                # in the stitched canvas for both down-scaling and up-scaling.
                baseline_64 = ref_baseline - shift
                baseline_in_resized = int(round(baseline_64 * h_scale))
                pad_top = max(0, ref_baseline - baseline_in_resized)
            else:
                pad_top = shift
            pad_bot = max(0, canvas_height - target_h - pad_top)
            padded = np.pad(resized, ((pad_top, pad_bot), (0, 0)),
                            mode='constant', constant_values=255)

        scaled_words.append(padded)

    height = canvas_height
    gap = np.ones((height, gap_width), dtype=np.uint8) * 255

    # Single line — concatenate all words with gaps, no wrapping
    parts = [gap]
    for img in scaled_words:
        parts.append(img)
        parts.append(gap)

    line_img = np.concatenate(parts, axis=1)
    return Image.fromarray(line_img)


def main():
    parser = argparse.ArgumentParser(description="Generate sentence/paragraph in a writer style")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--text", type=str, required=True, help="Sentence to generate (words separated by spaces)")
    parser.add_argument("--writer", type=str, nargs="+", default=None,
                        help="Writer string ID(s) from the dataset (e.g. 001 023)")
    parser.add_argument("--writer_idx", type=int, nargs="+", default=None,
                        help="Writer numeric index(es) (e.g. 12 25)")
    parser.add_argument("--random_writer", action="store_true", help="Pick a random writer")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional deterministic seed for reproducible inference")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--stable_dif_path", type=str, default="./stable-diffusion-v1-5")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--meta_file", type=str, default=None)
    parser.add_argument("--style_ref_map_json", type=str, default=None,
                        help="Optional JSON map {writer_id: [filename1.png, ...]} to override style refs per writer")
    parser.add_argument("--style_path", type=str,
                        default="style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--img_height", type=int, default=64)
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--text_max_len", type=int, default=40)
    parser.add_argument("--max_line_width", type=int, default=900)
    parser.add_argument("--canvas_height", type=int, default=104,
                        help="Canvas height per line (>img_height adds space for descenders)")
    parser.add_argument("--punct_bank", type=str,
                        default="/home/oles/DiffusionPen/generated/punct_bank",
                        help="Directory with punct_bank/{comma,period,question,exclaim}/ subdirs")
    parser.add_argument("--emb_dim", type=int, default=320)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--enable_nafnet_cleanup", action="store_true",
                        help="Apply optional NAFNet cleanup to each generated word (thesis visual workaround)")
    parser.add_argument("--nafnet_ckpt", type=str,
                        default="output/lines204_nafnet_v1/checkpoint_best.pt",
                        help="Path to NAFNet checkpoint for optional word cleanup")
    parser.add_argument("--nafnet_device", type=str, default=None,
                        help="Device for NAFNet cleanup (default: same as --device)")
    parser.add_argument("--nafnet_blend", type=float, default=0.85,
                        help="Blend strength [0..1] for NAFNet cleanup (1.0 = full model output)")
    parser.add_argument("--debug_sentence_dir", type=str, default=None,
                        help="Optional directory to save intermediate sentence-assembly artifacts")
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    state_dict = strip_dp_prefix(state_dict)
    num_classes = detect_num_classes(state_dict)
    print(f"Detected {num_classes} writer classes")

    print("Loading CANINE-C...")
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    canine_model = CanineModel.from_pretrained("google/canine-c")

    from utils.word_dataset import char_classes as WORD_CHAR_CLASSES
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

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)

    noise_scheduler = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    print(f"Loading style encoder: {args.style_path}")
    style_extractor = ImageEncoder(model_name="mobilenetv2_100", num_classes=0,
                                   pretrained=False, trainable=False)
    style_sd = torch.load(args.style_path, map_location="cpu")
    model_dict = style_extractor.state_dict()
    style_sd = {k: v for k, v in style_sd.items()
                if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(style_sd)
    style_extractor.load_state_dict(model_dict)
    style_extractor = style_extractor.to(device)
    style_extractor.eval()

    nafnet_cleaner = None
    if args.enable_nafnet_cleanup:
        cleanup_device = args.nafnet_device or args.device
        print(
            f"Loading optional NAFNet word cleaner: ckpt={args.nafnet_ckpt}, "
            f"device={cleanup_device}, blend={args.nafnet_blend}"
        )
        nafnet_cleaner = NAFNetWordCleaner(
            checkpoint_path=args.nafnet_ckpt,
            device=cleanup_device,
            blend=args.nafnet_blend,
        )

    dataset_root = args.dataset_root
    if dataset_root is None:
        for candidate in ["datasets/UkrHandwritten_Words_CC",
                          "datasets/UkrHandwritten_Words_Clean"]:
            if os.path.isdir(candidate):
                dataset_root = candidate
                break
    if dataset_root is None:
        raise FileNotFoundError("Cannot find dataset. Use --dataset_root.")

    meta_file = args.meta_file or os.path.join(dataset_root, "METAFILE.tsv")
    writer_id_map = build_writer_id_map(meta_file)
    idx_to_str = {v: k for k, v in writer_id_map.items()}

    style_ref_override = None
    if args.style_ref_map_json:
        with open(args.style_ref_map_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("--style_ref_map_json must contain a JSON object {writer_id: [filenames...]}")
        # Normalize values to lists of strings.
        style_ref_override = {}
        for k, v in loaded.items():
            if isinstance(v, str):
                style_ref_override[str(k)] = [v]
            elif isinstance(v, list):
                style_ref_override[str(k)] = [str(x) for x in v]
            else:
                raise ValueError(f"Invalid refs for writer '{k}': expected string or list, got {type(v).__name__}")

    writer_indices = []
    if args.writer:
        for wstr in args.writer:
            if wstr in writer_id_map:
                writer_indices.append(writer_id_map[wstr])
            else:
                print(f"WARNING: writer '{wstr}' not found in dataset. "
                      f"Available: {sorted(writer_id_map.keys())[:10]}...")
    elif args.writer_idx is not None:
        writer_indices = args.writer_idx
    elif args.random_writer:
        writer_indices = [random.choice(list(idx_to_str.keys()))]
        print(f"Random writer: index={writer_indices[0]}, id={idx_to_str[writer_indices[0]]}")
    else:
        parser.error("Specify --writer, --writer_idx, or --random_writer")

    print(f"Loading style references from: {dataset_root}")
    style_refs, style_refs_used = load_style_images(
        dataset_root, meta_file, writer_indices, writer_id_map,
        img_height=args.img_height, img_width=args.img_width,
        style_ref_override=style_ref_override,
    )

    words = args.text.strip().split()
    output_dir = args.output_dir or f"generated/sentence_cfg{args.cfg_scale}"
    os.makedirs(output_dir, exist_ok=True)

    for wid in writer_indices:
        if args.seed is not None:
            # Re-seed per writer so reruns stay stable even when multiple writers are
            # generated in the same invocation.
            writer_seed = int(args.seed) + int(wid)
            print(f"Using deterministic seed for writer {wid}: {writer_seed}")
            set_global_seed(writer_seed)
        if wid not in style_refs:
            print(f"WARNING: No style images for writer index {wid}, skipping.")
            continue

        writer_str = idx_to_str.get(wid, f"w{wid:04d}")
        print(f"\n--- Writer {writer_str} (idx={wid}) ---")
        if wid in style_refs_used:
            print(f"Style refs used: {style_refs_used[wid]}")

        word_images = []
        expanded_words = []  # mirrors word_images, used by stitch_paragraph
        punct_flags = []     # True for bank images, False for diffusion-generated
        punct_standalone_flags = []  # True for standalone clause dashes, False for inline marks
        word_underline_ys = []  # Hough-detected underline Y per image, or None
        for word in tqdm(words, desc=f"Generating words (writer {writer_str})"):
            # Strip trailing punctuation (e.g. "Реве," -> "Реве" + [","])
            punct_suffix = []
            w = word
            while w and w[-1] in PUNCTUATION:
                punct_suffix.insert(0, w[-1])
                w = w[:-1]

            # Split on inline chars with bank entries (e.g. "будь-яка" -> ["будь", "-", "яка"])
            if w:
                for part, is_mark in split_word_for_generation(w, args.punct_bank, args.img_height):
                    if is_mark:
                        ch_arr = sample_punctuation(
                            part,
                            args.img_height,
                            args.punct_bank,
                            writer_str,
                            standalone=False,
                        )
                        if ch_arr is not None:
                            word_images.append(ch_arr)
                            expanded_words.append(part)
                            punct_flags.append(True)
                            punct_standalone_flags.append(False)
                            word_underline_ys.append(None)
                    else:
                        img_pil = generate_single_word(
                            word=part, unet=unet, vae=vae,
                            style_extractor=style_extractor, tokenizer=tokenizer,
                            noise_scheduler=noise_scheduler,
                            style_ref=style_refs[wid], writer_idx=wid,
                            device=device, cfg_scale=args.cfg_scale,
                            img_height=args.img_height, img_width=args.img_width,
                            text_max_len=args.text_max_len,
                        )
                        cropped = crop_whitespace(img_pil)
                        img_cleaned, ul_y = detect_baseline_and_clean(cropped)
                        word_images.append(img_cleaned)
                        expanded_words.append(part)
                        punct_flags.append(False)
                        punct_standalone_flags.append(False)
                        word_underline_ys.append(ul_y)
                        print(f"  [{part}] crop_w={cropped.shape[1]}  ul_y={ul_y}")

            for ch in punct_suffix:
                is_standalone_dash = (ch == '-' and not w)
                ch_arr = sample_punctuation(
                    ch,
                    args.img_height,
                    args.punct_bank,
                    writer_str,
                    standalone=is_standalone_dash,
                )
                if ch_arr is not None:
                    word_images.append(ch_arr)
                    expanded_words.append(ch)
                    punct_flags.append(True)
                    punct_standalone_flags.append(is_standalone_dash)
                    word_underline_ys.append(None)

        raw_word_images = [img.copy() for img in word_images]

        if nafnet_cleaner is not None:
            for i, is_p in enumerate(punct_flags):
                if not is_p:
                    word_images[i] = nafnet_cleaner.clean_gray(word_images[i])

        word_images = normalize_ink_brightness(word_images)
        cleaned_word_images = [img.copy() for img in word_images]
        paragraph, geometry_payload = compose_sentence_geometry(
            word_images=word_images,
            expanded_words=expanded_words,
            punct_flags=punct_flags,
            punct_standalone_flags=punct_standalone_flags,
            punct_bank=args.punct_bank,
            writer_str=writer_str,
            canvas_height=args.canvas_height,
        )
        print(
            f"  geometry baseline={geometry_payload['baseline_y']}  "
            f"target_body_h={geometry_payload['target_body_h']:.2f}"
        )

        out_path = os.path.join(output_dir, f"sentence_writer_{writer_str}.png")
        paragraph.save(out_path)
        print(f"Saved: {out_path}")

        if args.debug_sentence_dir:
            debug_dir = os.path.join(args.debug_sentence_dir, f"writer_{writer_str}")
            save_sentence_geometry_debug_bundle(
                debug_dir=debug_dir,
                writer_str=writer_str,
                expanded_words=expanded_words,
                raw_word_images=raw_word_images,
                cleaned_word_images=cleaned_word_images,
                final_paragraph=paragraph,
                geometry_payload=geometry_payload,
            )
            print(f"  Debug saved: {debug_dir}")

    print(f"\nAll done. Output in: {output_dir}")


if __name__ == "__main__":
    main()
