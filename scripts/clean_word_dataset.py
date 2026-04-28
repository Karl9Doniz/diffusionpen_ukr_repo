import argparse
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def stage1_text_filter(df, keep_short=False):
    """Remove entries with garbage, multi-word, or punctuation-only labels.

    If keep_short=True, 1-3 char words are kept (extended dataset mode).
    """
    n0 = len(df)
    reasons = Counter()

    mask = pd.Series(True, index=df.index)

    is_filename = df['transcription'].str.contains(r'\.png$|\.jpg$|\.jpeg$', case=False, na=False)
    reasons['filename_as_text'] = is_filename.sum()
    mask &= ~is_filename

    # Multi-word (contains spaces) is bad segmentation
    has_space = df['transcription'].str.contains(' ', na=False)
    reasons['multi_word'] = has_space.sum()
    mask &= ~has_space

    # Punctuation-only
    punct_only = df['transcription'].str.match(r'^[\-\–\—\.\,\;\:\!\?\…\(\)\"\'«»]+$', na=False)
    reasons['punctuation_only'] = punct_only.sum()
    mask &= ~punct_only

    has_latin = df['transcription'].str.contains(r'[a-zA-Z]', na=False)
    reasons['has_latin'] = has_latin.sum()
    mask &= ~has_latin

    # Empty or whitespace
    empty = df['transcription'].str.strip().eq('') | df['transcription'].isna()
    reasons['empty'] = empty.sum()
    mask &= ~empty

    if not keep_short:
        # Single character (too little to learn)
        single_char = df['transcription'].str.len() <= 1
        reasons['single_char'] = single_char.sum()
        mask &= ~single_char

        short = df['transcription'].str.len().between(2, 3)
        reasons['short_2_3_char'] = short.sum()
        mask &= ~short

    df_clean = df[mask].copy()

    print(f"  Input:  {n0:,}")
    print(f"  Output: {len(df_clean):,}  (removed {n0 - len(df_clean):,})")
    for reason, count in reasons.most_common():
        print(f"    - {reason}: {count:,}")

    return df_clean


_TRAILING_PUNCT_RE = re.compile(r'[\.\,\;\:\!\?\…\)\]»\"\']+$')
_LEADING_PUNCT_RE  = re.compile(r'^[\(\[«\"\']+')


def stage2_strip_punctuation(df, keep_short=False, reject_trailing_punct=False):
    """Strip trailing/leading punctuation from transcriptions so labels are pure words.

    If keep_short=True, words that become 1+ chars after stripping are kept.
    If reject_trailing_punct=True, entries whose original label had trailing
    punctuation are REJECTED outright (image likely contains a visible comma/period).
    This eliminates the label-image mismatch where the model learns to generate
    trailing commas.
    """
    df = df.copy()

    # Identify rows with trailing punctuation BEFORE stripping
    has_trailing = df['transcription'].str.contains(_TRAILING_PUNCT_RE, na=False)

    if reject_trailing_punct:
        n_rejected = has_trailing.sum()
        df = df[~has_trailing].copy()
        print(f"  Rejected (trailing punct in image): {n_rejected:,}")

    n_modified = 0

    def _strip(text):
        nonlocal n_modified
        original = text
        text = _TRAILING_PUNCT_RE.sub('', text)
        text = _LEADING_PUNCT_RE.sub('', text)
        if text != original:
            n_modified += 1
        return text if text else original

    df['transcription'] = df['transcription'].apply(_strip)

    min_len = 1 if keep_short else 4
    too_short = df['transcription'].str.len() < min_len
    n_removed = too_short.sum()
    df = df[~too_short]

    print(f"  Modified labels: {n_modified:,}")
    print(f"  Removed (became <{min_len} chars): {n_removed:,}")
    print(f"  Remaining: {len(df):,}")

    return df


def stage3_dimension_filter(df, img_root, max_width=800, min_width=15,
                            min_height=10, max_aspect=12.0):
    n0 = len(df)
    reasons = Counter()
    widths = []
    heights = []
    keep_mask = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stage 3: checking dims"):
        img_path = os.path.join(img_root, 'words', 'words', row['filename'])
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            reasons['unreadable'] += 1
            keep_mask.append(False)
            continue

        widths.append(w)
        heights.append(h)

        if w > max_width:
            reasons['too_wide'] += 1
            keep_mask.append(False)
        elif w < min_width:
            reasons['too_narrow'] += 1
            keep_mask.append(False)
        elif h < min_height:
            reasons['too_short'] += 1
            keep_mask.append(False)
        elif w / max(h, 1) > max_aspect:
            reasons['extreme_aspect'] += 1
            keep_mask.append(False)
        else:
            char_count = len(row['transcription'])
            px_per_char = w / max(char_count, 1)
            # Short words (1-2 chars) can have wider per-char px; relax to 200
            max_px_per_char = 200 if char_count <= 2 else 120
            if px_per_char > max_px_per_char:
                reasons['width_vs_chars'] += 1
                keep_mask.append(False)
            elif px_per_char < 10:  # <10px per character is too short
                reasons['too_cramped'] += 1
                keep_mask.append(False)
            else:
                keep_mask.append(True)

    df_clean = df[keep_mask].copy()

    print(f"  Input:  {n0:,}")
    print(f"  Output: {len(df_clean):,}  (removed {n0 - len(df_clean):,})")
    for reason, count in reasons.most_common():
        print(f"    - {reason}: {count:,}")
    if widths:
        print(f"  Width  — median: {np.median(widths):.0f}, p95: {np.percentile(widths, 95):.0f}")
        print(f"  Height — median: {np.median(heights):.0f}, p95: {np.percentile(heights, 95):.0f}")

    return df_clean


# ---------------------------------------------------------------------------
# Stage 4: TrOCR label validation
# ---------------------------------------------------------------------------

def _trocr_infer_chunk(device, filenames, transcriptions, img_root, model_name,
                       batch_size, progress_desc="TrOCR"):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from difflib import SequenceMatcher

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    model.eval()

    similarities = []
    ocr_texts = []

    for batch_start in tqdm(range(0, len(filenames), batch_size), desc=progress_desc):
        batch_end = min(batch_start + batch_size, len(filenames))
        pil_images = []
        batch_valid = []

        for i in range(batch_start, batch_end):
            img_path = os.path.join(img_root, 'words', 'words', filenames[i])
            try:
                img = Image.open(img_path).convert('RGB')
                pil_images.append(img)
                batch_valid.append(True)
            except Exception:
                pil_images.append(Image.new('RGB', (64, 64), 'white'))
                batch_valid.append(False)

        pixel_values = processor(
            images=pil_images, return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values, max_new_tokens=25, num_beams=1
            )

        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for j in range(len(decoded)):
            i = batch_start + j
            if not batch_valid[j]:
                similarities.append(0.0)
                ocr_texts.append('')
                continue
            ocr_text = decoded[j].strip()
            gt_text = transcriptions[i]
            sim = SequenceMatcher(None, ocr_text.lower(), gt_text.lower()).ratio()
            similarities.append(sim)
            ocr_texts.append(ocr_text)

    return similarities, ocr_texts


def _trocr_worker(worker_id, device, filenames, transcriptions, img_root, model_name,
                  batch_size, result_dict):
    try:
        similarities, ocr_texts = _trocr_infer_chunk(
            device=device,
            filenames=filenames,
            transcriptions=transcriptions,
            img_root=img_root,
            model_name=model_name,
            batch_size=batch_size,
            progress_desc=f"{device}",
        )
        result_dict[worker_id] = {
            "ok": True,
            "similarities": similarities,
            "ocr_texts": ocr_texts,
            "device": device,
        }
    except Exception:
        result_dict[worker_id] = {
            "ok": False,
            "device": device,
            "error": traceback.format_exc(),
        }


def stage4_trocr_validation(df, img_root, device='cuda:0',
                            model_name='cyrillic-trocr/trocr-handwritten-cyrillic',
                            min_similarity=0.4, batch_size=32):
    """Use Cyrillic TrOCR to verify labels across multiple GPUs."""
    import multiprocessing as mp

    def _resolve_devices(requested_device: str):
        if requested_device.startswith("cpu"):
            return ["cpu"]
        if not torch.cuda.is_available():
            return ["cpu"]
        # Explicit single GPU target (default path)
        if requested_device.startswith("cuda:"):
            try:
                gpu_expr = requested_device.split(":", 1)[1]
                gpu_ids = [int(x) for x in gpu_expr.split(",")]
            except ValueError as exc:
                raise ValueError(f"Invalid --device value: {requested_device}") from exc
            n = torch.cuda.device_count()
            for gpu_id in gpu_ids:
                if gpu_id < 0 or gpu_id >= n:
                    raise ValueError(
                        f"Requested GPU {gpu_id} but only {n} GPU(s) are visible."
                    )
            return [f"cuda:{gpu_id}" for gpu_id in gpu_ids]
        # Plain 'cuda' means use all visible GPUs
        if requested_device == "cuda":
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        # Fallback to explicit single-device behavior for unusual strings
        return [requested_device]

    devices = _resolve_devices(device)
    print(f"  Using device(s): {devices} | batch_size={batch_size}, greedy decoding")

    filenames = df['filename'].values
    transcriptions = df['transcription'].values
    n = len(filenames)

    if len(devices) > 1:
        # CUDA + multiprocessing requires spawn (fork re-init fails on PyTorch).
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        result_dict = manager.dict()
        chunk_size = (n + len(devices) - 1) // len(devices)
        processes = []
        chunks = []

        for worker_id, worker_device in enumerate(devices):
            start = worker_id * chunk_size
            end = min(start + chunk_size, n)
            if start >= n:
                break
            p = ctx.Process(
                target=_trocr_worker,
                args=(worker_id, worker_device, filenames[start:end], transcriptions[start:end],
                      img_root, model_name, batch_size, result_dict)
            )
            p.start()
            processes.append(p)
            chunks.append((worker_id, worker_device, start, end))

        for p in processes:
            p.join()

        all_sims = []
        all_texts = []
        for worker_id, worker_device, start, end in chunks:
            payload = result_dict.get(worker_id)
            if not payload:
                raise RuntimeError(
                    f"TrOCR worker {worker_id} on {worker_device} exited without result. "
                    "This usually indicates an OOM or process crash."
                )
            if not payload.get("ok", False):
                raise RuntimeError(
                    f"TrOCR worker {worker_id} on {worker_device} failed.\n"
                    f"{payload.get('error', 'No traceback captured')}"
                )
            sims = payload["similarities"]
            texts = payload["ocr_texts"]
            all_sims.extend(sims)
            all_texts.extend(texts)
    else:
        all_sims, all_texts = _trocr_infer_chunk(
            device=devices[0],
            filenames=filenames,
            transcriptions=transcriptions,
            img_root=img_root,
            model_name=model_name,
            batch_size=batch_size,
            progress_desc=devices[0],
        )

    # Length-stratified thresholds:
    #   <= 3 chars: skip TrOCR (always pass) — single-char OCR is unreliable
    #   4-5 chars:  use min(min_similarity, 0.2)
    #   >= 6 chars: use min_similarity as specified
    def _threshold(gt_text):
        n = len(gt_text)
        if n <= 3:
            return None   # always pass
        elif n <= 5:
            return min(min_similarity, 0.2)
        else:
            return min_similarity

    valid_mask = []
    for s, gt in zip(all_sims, transcriptions):
        t = _threshold(gt)
        valid_mask.append(t is None or s >= t)

    df = df.copy()
    df['trocr_text'] = all_texts
    df['trocr_similarity'] = all_sims

    n_rejected = sum(1 for v in valid_mask if not v)
    df_clean = df[valid_mask].copy()

    long_sims = [s for s, gt in zip(all_sims, transcriptions) if len(gt) >= 6]
    sim_arr = np.array(long_sims) if long_sims else np.array(all_sims)
    print(f"  Input:  {n:,}")
    print(f"  Output: {len(df_clean):,}  (rejected {n_rejected:,})")
    print(f"  Similarity (>=6 char words) — mean: {sim_arr.mean():.3f}, "
          f"median: {np.median(sim_arr):.3f}, p10: {np.percentile(sim_arr, 10):.3f}")
    skipped = sum(1 for gt in transcriptions if len(gt) <= 3)
    print(f"  Skipped TrOCR (<=3 chars): {skipped:,}")

    rejected_idx = [i for i, v in enumerate(valid_mask) if not v]
    if rejected_idx:
        print(f"\n  Sample rejections (GT vs OCR):")
        for i in rejected_idx[:15]:
            print(f"    GT=\"{transcriptions[i]}\"  OCR=\"{all_texts[i]}\"  sim={all_sims[i]:.2f}")

    return df_clean


def stage5_writer_balance(df, img_root, min_samples=50):
    """Remove writers with too few samples after cleaning."""
    # Extract writer ID from filename
    # Writer is the 4-digit field
    def get_writer(fn):
        parts = fn.split('-')
        if len(parts) >= 3:
            return parts[2]
        return 'unknown'

    df = df.copy()
    df['writer'] = df['filename'].apply(get_writer)

    writer_counts = df['writer'].value_counts()
    small_writers = writer_counts[writer_counts < min_samples].index.tolist()

    n_removed_writers = len(small_writers)
    removed_samples = df[df['writer'].isin(small_writers)]

    df_clean = df[~df['writer'].isin(small_writers)].copy()

    print(f"  Total writers before: {len(writer_counts)}")
    print(f"  Writers removed (<{min_samples} samples): {n_removed_writers}")
    if small_writers:
        for w in small_writers[:10]:
            print(f"    writer {w}: {writer_counts[w]} samples")
    print(f"  Samples removed: {len(removed_samples):,}")
    print(f"  Remaining: {len(df_clean):,}  ({df_clean['writer'].nunique()} writers)")

    final_counts = df_clean['writer'].value_counts()
    print(f"  Samples/writer — min: {final_counts.min()}, max: {final_counts.max()}, "
          f"median: {final_counts.median():.0f}, mean: {final_counts.mean():.0f}")

    return df_clean


def main():
    parser = argparse.ArgumentParser(description='Clean Ukrainian word dataset')
    parser.add_argument('--input', type=str, required=True,
                        help='Root of segmented word dataset (contains words/words/ and METAFILE.tsv)')
    parser.add_argument('--output-tsv', type=str, default=None,
                        help='Output METAFILE path (default: <input>/METAFILE_clean.tsv)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--min-similarity', type=float, default=0.4,
                        help='Minimum TrOCR similarity to keep (0-1)')
    parser.add_argument('--min-writer-samples', type=int, default=50,
                        help='Minimum samples per writer to keep')
    parser.add_argument('--skip-trocr', action='store_true',
                        help='Skip TrOCR validation (for testing)')
    parser.add_argument('--keep-short', action='store_true',
                        help='Keep 1-3 char words (extended dataset mode). '
                             'Relaxes stage1/stage2 filters and skips TrOCR for short words.')
    parser.add_argument('--reject-trailing-punct', action='store_true',
                        help='Reject entries whose original label had trailing punctuation. '
                             'Eliminates label/image mismatch (image shows comma, label does not).')
    parser.add_argument('--trocr-batch-size', type=int, default=32)

    args = parser.parse_args()

    input_root = args.input
    metafile = os.path.join(input_root, 'METAFILE.tsv')
    output_tsv = args.output_tsv or os.path.join(input_root, 'METAFILE_clean.tsv')

    print(f"Loading {metafile}")
    df = pd.read_csv(metafile, sep='\t')
    print(f"Loaded {len(df):,} entries")

    # Use the unfiltered dataset (all 155K) as input to get maximum coverage
    # The old filter only checked word count match — we do much more now

    df = stage1_text_filter(df, keep_short=args.keep_short)
    df = stage2_strip_punctuation(df, keep_short=args.keep_short,
                                  reject_trailing_punct=args.reject_trailing_punct)
    df = stage3_dimension_filter(df, input_root)

    if not args.skip_trocr:
        df = stage4_trocr_validation(df, input_root, device=args.device,
                                     min_similarity=args.min_similarity,
                                     batch_size=args.trocr_batch_size)

    df = stage5_writer_balance(df, input_root,
                               min_samples=args.min_writer_samples)

    # Drop helper columns before saving
    for col in ['writer', 'trocr_text', 'trocr_similarity']:
        if col in df.columns:
            df = df.drop(columns=[col])

    df.to_csv(output_tsv, sep='\t', index=False)

    print(f"\n{'='*60}")
    print(f"FINAL: {len(df):,} clean entries saved to {output_tsv}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
