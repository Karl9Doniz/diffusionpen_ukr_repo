"""
Build punct_bank from raw word images in the Ukrainian handwriting dataset.

Three extraction strategies:
  - Trailing mark: column-projection gap between word body and trailing punct
  - Standalone: label is exactly the character (good source for `-`)
  - Copy: reuse an existing directory (existing comma_bank for `,`)

Output structure:
    generated/punct_bank/
        comma/       ← ','
        period/      ← '.'
        question/    ← '?'
        exclaim/     ← '!'
        colon/       ← ':'
        semicolon/   ← ';'
        hyphen/      ← '-'  (from standalone '-' labels)

Usage:
    # Dry run — print quality stats without writing anything
    python scripts/build_punct_bank.py --metafile ... --words_dir ... --output_dir ... --dry_run

    # Full extraction
    python scripts/build_punct_bank.py --metafile ... --words_dir ... --output_dir ... \\
        --existing_comma_bank /home/oles/DiffusionPen/generated/comma_bank
"""

import argparse
import csv
import os
import shutil

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Character → subdir mapping
# ---------------------------------------------------------------------------

TRAILING_CHARS = {
    ',': 'comma',
    '.': 'period',
    '?': 'question',
    '!': 'exclaim',
    ':': 'colon',
    ';': 'semicolon',
}

STANDALONE_CHARS = {
    '-': 'hyphen',
}

# ---------------------------------------------------------------------------
# Per-character quality thresholds
#
# All applied to the RAW (pre-scaled) tight-cropped mark:
#   min_h / max_h  : raw crop height in pixels
#   max_w          : raw crop width in pixels
#   min_r / max_r  : width/height aspect ratio
#   max_cc         : max connected components after morphological closing
#                    (merges nearby ink gaps — clean marks have few CCs)
#   min_ink        : minimum ink coverage % (dark pixels / total pixels)
# ---------------------------------------------------------------------------

_Q = {
    ',': dict(min_h=8,  max_h=60,  max_w=80,  min_r=0.3,  max_r=4.0,  max_cc=2, min_ink=5.0),
    '.': dict(min_h=3,  max_h=22,  max_w=35,  min_r=0.4,  max_r=3.5,  max_cc=1, min_ink=8.0),
    '?': dict(min_h=18, max_h=105, max_w=75,  min_r=0.3,  max_r=1.9,  max_cc=5, min_ink=5.0),
    '!': dict(min_h=15, max_h=105, max_w=55,  min_r=0.1,  max_r=1.3,  max_cc=5, min_ink=5.0),
    ':': dict(min_h=12, max_h=80,  max_w=45,  min_r=0.15, max_r=2.0,  max_cc=4, min_ink=6.0),
    ';': dict(min_h=18, max_h=90,  max_w=50,  min_r=0.15, max_r=2.0,  max_cc=4, min_ink=5.0),
    '-': dict(min_h=3,  max_h=42,  max_w=200, min_r=1.5,  max_r=14.0, max_cc=2, min_ink=10.0),
}

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _binarize(img_gray):
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def check_quality(crop, ch):
    """Return (passed: bool, reason: str) for a tight-cropped mark image."""
    q = _Q.get(ch)
    if q is None:
        return True, 'ok'

    h, w = crop.shape
    ratio = w / h if h > 0 else 0

    if h < q['min_h']:
        return False, f'h={h} < min_h={q["min_h"]}'
    if h > q['max_h']:
        return False, f'h={h} > max_h={q["max_h"]}'
    if w > q['max_w']:
        return False, f'w={w} > max_w={q["max_w"]}'
    if ratio < q['min_r']:
        return False, f'ratio={ratio:.1f} < min_r={q["min_r"]}'
    if ratio > q['max_r']:
        return False, f'ratio={ratio:.1f} > max_r={q["max_r"]}'

    thresh = _binarize(crop)
    ink_pct = thresh.sum() / (255 * h * w) * 100
    if ink_pct < q['min_ink']:
        return False, f'ink={ink_pct:.1f}% < min_ink={q["min_ink"]}%'

    # CC count after morphological closing to merge nearby ink gaps
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    n_cc = cv2.connectedComponents(closed)[0] - 1  # subtract background
    if n_cc > q['max_cc']:
        return False, f'cc={n_cc} > max_cc={q["max_cc"]}'

    return True, 'ok'


# ---------------------------------------------------------------------------
# Crop extraction helpers
# ---------------------------------------------------------------------------

def tight_crop(img_gray, min_ink_sum=8):
    """Remove blank rows and columns from all four sides."""
    thresh = _binarize(img_gray)
    row_sum = thresh.sum(axis=1)
    col_sum = thresh.sum(axis=0)
    ink_rows = np.where(row_sum >= min_ink_sum)[0]
    ink_cols = np.where(col_sum >= min_ink_sum)[0]
    if len(ink_rows) == 0 or len(ink_cols) == 0:
        return None
    return img_gray[ink_rows[0]:ink_rows[-1] + 1, ink_cols[0]:ink_cols[-1] + 1]


def extract_trailing_mark(img_gray, min_gap_cols=3, min_ink_sum=8, max_width_frac=0.25):
    """Isolate the trailing mark via column ink projection gap.

    Returns a tight-cropped mark, or None if no clear gap or mark too wide.
    """
    w_orig = img_gray.shape[1]
    thresh = _binarize(img_gray)
    col_sum = thresh.sum(axis=0).astype(np.float32)

    ink_cols = np.where(col_sum >= min_ink_sum)[0]
    if len(ink_cols) < 2:
        return None
    right_edge = int(ink_cols[-1])

    gap_end = gap_start = None
    i = right_edge
    while i >= 0:
        if col_sum[i] < min_ink_sum:
            if gap_end is None:
                gap_end = i
        else:
            if gap_end is not None:
                gap_start = i + 1
                if gap_end - gap_start + 1 >= min_gap_cols:
                    break
                else:
                    gap_end = None
        i -= 1

    if gap_start is None or gap_end is None:
        return None

    mark = img_gray[:, gap_end + 1:right_edge + 1]
    if mark.shape[1] < 2:
        return None
    if mark.shape[1] > w_orig * max_width_frac:
        return None

    return tight_crop(mark, min_ink_sum)


# ---------------------------------------------------------------------------
# Stats helpers for dry run / post-run verification
# ---------------------------------------------------------------------------

def print_crop_stats(crops_by_reason):
    """Print pass/fail breakdown and distribution of key metrics for a batch."""
    passed = [(c, r) for c, r in crops_by_reason if r == 'ok']
    failed = [(c, r) for c, r in crops_by_reason if r != 'ok']

    print(f"    passed: {len(passed)} / {len(crops_by_reason)}")

    if failed:
        reasons = {}
        for _, r in failed:
            key = r.split('=')[0]  # 'h', 'w', 'ratio', 'ink', 'cc'
            reasons[key] = reasons.get(key, 0) + 1
        print(f"    rejected by: {dict(sorted(reasons.items(), key=lambda x: -x[1]))}")

    if passed:
        heights = sorted(c.shape[0] for c, _ in passed)
        widths  = sorted(c.shape[1] for c, _ in passed)
        n = len(heights)
        print(f"    height: {heights[0]} – {heights[n//4]} – {heights[n//2]} – {heights[3*n//4]} – {heights[-1]}  (min/p25/median/p75/max)")
        print(f"    width:  {widths[0]} – {widths[n//4]} – {widths[n//2]} – {widths[3*n//4]} – {widths[-1]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_writer_id(fname):
    """Extract writer ID string from source filename (e.g. 'a01-001-0023-02-w01.png' → '0023')."""
    parts = fname.replace(".png", "").split("-")
    return parts[2] if len(parts) >= 3 else "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metafile", required=True)
    p.add_argument("--words_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_per_char", type=int, default=500)
    p.add_argument("--dry_run", action="store_true",
                   help="Print quality stats for each char without writing any files")
    args = p.parse_args()

    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Collect source entries (fname, writer_id) ---
    trailing_entries = {ch: [] for ch in TRAILING_CHARS}
    standalone_entries = {ch: [] for ch in STANDALONE_CHARS}

    with open(args.metafile, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            fname, label = row[0], row[1].strip()
            if not label:
                continue
            wid = get_writer_id(fname)
            if label in STANDALONE_CHARS:
                standalone_entries[label].append((fname, wid))
            elif label[-1] in TRAILING_CHARS:
                trailing_entries[label[-1]].append((fname, wid))

    print("Source counts:")
    for ch, subdir in TRAILING_CHARS.items():
        n_writers = len(set(wid for _, wid in trailing_entries[ch]))
        print(f"  {repr(ch)} trailing ({subdir}): {len(trailing_entries[ch])} from {n_writers} writers")
    for ch, subdir in STANDALONE_CHARS.items():
        n_writers = len(set(wid for _, wid in standalone_entries[ch]))
        print(f"  {repr(ch)} standalone ({subdir}): {len(standalone_entries[ch])} from {n_writers} writers")

    # --- Trailing extraction ---
    print("\nTrailing mark extraction:")
    for ch, subdir_name in TRAILING_CHARS.items():
        subdir = os.path.join(args.output_dir, subdir_name)
        if not args.dry_run:
            os.makedirs(subdir, exist_ok=True)
            existing_count = len([f for f in os.listdir(subdir) if f.endswith(".png")])
        else:
            existing_count = 0

        print(f"  {repr(ch)} ({subdir_name}):")
        results = []  # (mark, reason, writer_id)
        for fname, wid in trailing_entries[ch]:
            if len([r for r in results if r[1] == 'ok']) >= args.max_per_char - existing_count:
                break
            img = cv2.imread(os.path.join(args.words_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            mark = extract_trailing_mark(img)
            if mark is None:
                results.append((None, 'no_gap', wid))
                continue
            ok, reason = check_quality(mark, ch)
            results.append((mark, reason, wid))

        print_crop_stats([(c, r) for c, r, _ in results if r != 'no_gap'])
        no_gap = sum(1 for _, r, _ in results if r == 'no_gap')
        saved_writers = set(wid for _, r, wid in results if r == 'ok')
        print(f"    no_gap: {no_gap}  |  writers in output: {len(saved_writers)}")

        if not args.dry_run:
            saved = 0
            for mark, reason, wid in results:
                if reason == 'ok' and mark is not None:
                    out_name = f"{subdir_name}_{wid}_{existing_count + saved:05d}.png"
                    cv2.imwrite(os.path.join(subdir, out_name), mark)
                    saved += 1
            print(f"    → saved {saved}")

    # --- Standalone extraction ---
    print("\nStandalone mark extraction:")
    for ch, subdir_name in STANDALONE_CHARS.items():
        subdir = os.path.join(args.output_dir, subdir_name)
        if not args.dry_run:
            os.makedirs(subdir, exist_ok=True)
            existing_count = len([f for f in os.listdir(subdir) if f.endswith(".png")])
        else:
            existing_count = 0

        print(f"  {repr(ch)} ({subdir_name}):")
        results = []
        for fname, wid in standalone_entries[ch]:
            if len([r for r in results if r[1] == 'ok']) >= args.max_per_char - existing_count:
                break
            img = cv2.imread(os.path.join(args.words_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            mark = tight_crop(img)
            if mark is None or mark.shape[1] < 2:
                results.append((None, 'empty', wid))
                continue
            ok, reason = check_quality(mark, ch)
            results.append((mark, reason, wid))

        print_crop_stats([(c, r) for c, r, _ in results if r not in ('empty', 'no_gap')])
        saved_writers = set(wid for _, r, wid in results if r == 'ok')
        print(f"    writers in output: {len(saved_writers)}")

        if not args.dry_run:
            saved = 0
            for mark, reason, wid in results:
                if reason == 'ok' and mark is not None:
                    out_name = f"{subdir_name}_{wid}_{existing_count + saved:05d}.png"
                    cv2.imwrite(os.path.join(subdir, out_name), mark)
                    saved += 1
            print(f"    → saved {saved}")

    # --- Summary ---
    if not args.dry_run:
        print("\nFinal bank:")
        all_subdirs = list(TRAILING_CHARS.values()) + list(STANDALONE_CHARS.values())
        for subdir_name in sorted(set(all_subdirs)):
            subdir = os.path.join(args.output_dir, subdir_name)
            if os.path.isdir(subdir):
                files = [f for f in os.listdir(subdir) if f.endswith(".png")]
                writers = set(f.split('_')[1] for f in files if len(f.split('_')) >= 3)
                print(f"  {subdir_name}/: {len(files)} images, {len(writers)} writers")


if __name__ == "__main__":
    main()
