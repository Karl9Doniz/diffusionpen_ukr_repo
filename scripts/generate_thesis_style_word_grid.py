#!/usr/bin/env python3
"""
Build a full-size (no-resize) thesis panel:
rows    -> writers
columns -> reference word crops + generated target words
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", type=Path, required=True, help="Root directory with generated outputs.")
    p.add_argument("--word_dirs", nargs="+", required=True, help="Subdirectories in run_root, one per target word.")
    p.add_argument("--word_labels", nargs="+", required=True, help="Display labels for the generated-word columns.")
    p.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/extra_space2/oles_new/UkrHandwritten_Words_CC_ULCleanNAF_v1"),
    )
    p.add_argument(
        "--meta_file",
        type=Path,
        default=Path("/extra_space2/oles_new/UkrHandwritten_Words_CC_ULCleanNAF_v1/METAFILE_extended_balanced.tsv"),
    )
    p.add_argument("--writers", nargs="+", default=["0023", "0595", "0600", "0642"])
    p.add_argument(
        "--top_writers",
        type=int,
        default=0,
        help="If >0, override --writers and select top-K writers by sample count from --meta_file.",
    )
    p.add_argument("--refs_per_writer", type=int, default=3)
    p.add_argument(
        "--separator_after_refs",
        action="store_true",
        help="Draw a visual separator after the reference columns.",
    )
    p.add_argument(
        "--separator_gap",
        type=int,
        default=28,
        help="Extra horizontal gap inserted between reference and generated columns.",
    )
    p.add_argument(
        "--separator_width",
        type=int,
        default=2,
        help="Separator line width in pixels.",
    )
    p.add_argument(
        "--min_ref_word_len",
        type=int,
        default=1,
        help="Prefer reference transcriptions with length >= this value.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--match_generate_sentence_refs",
        action="store_true",
        help="Use the same ref selection rule as generate_sentence.py (first N in metafile order).",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--refs_override_json",
        type=Path,
        default=None,
        help="Optional JSON mapping writer IDs to an explicit list of reference filenames.",
    )
    return p.parse_args()


def extract_writer_id(filename: str) -> str:
    parts = filename.split("-")
    return parts[2] if len(parts) >= 3 else ""


def load_meta_by_writer(meta_file: Path) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    with meta_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            fn = row.get("filename", "")
            tr = row.get("transcription", "")
            if not fn or not tr:
                continue
            wid = extract_writer_id(fn)
            if wid:
                out[wid].append((fn, tr))
    return out


def build_filename_lookup(per_writer: Dict[str, List[Tuple[str, str]]]) -> Dict[str, Dict[str, Tuple[str, str]]]:
    lookup: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for wid, rows in per_writer.items():
        lookup[wid] = {fn: (fn, tr) for fn, tr in rows}
    return lookup


def pick_top_writers(per_writer: Dict[str, List[Tuple[str, str]]], k: int) -> List[str]:
    ranked = sorted(per_writer.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return [wid for wid, _ in ranked[:k]]


def choose_reference_samples(
    rows: List[Tuple[str, str]],
    refs_per_writer: int,
    rng: random.Random,
    min_ref_word_len: int = 1,
) -> List[Tuple[str, str]]:
    filtered = [(fn, tr) for fn, tr in rows if any(ch.isalpha() for ch in tr)]
    preferred = [(fn, tr) for fn, tr in filtered if len(tr) >= min_ref_word_len]
    if preferred:
        filtered = preferred
    if not filtered:
        filtered = rows[:]
    short_pool = [(fn, tr) for fn, tr in filtered if len(tr) <= 3]
    mid_pool = [(fn, tr) for fn, tr in filtered if 4 <= len(tr) <= 6]
    long_pool = [(fn, tr) for fn, tr in filtered if len(tr) >= 7]

    picks: List[Tuple[str, str]] = []
    for pool in (short_pool, mid_pool, long_pool):
        if pool:
            picks.append(rng.choice(pool))

    remaining = [x for x in filtered if x not in picks]
    rng.shuffle(remaining)
    while len(picks) < refs_per_writer and remaining:
        picks.append(remaining.pop())
    return picks[:refs_per_writer]


def choose_reference_samples_like_generate_sentence(
    rows: List[Tuple[str, str]],
    refs_per_writer: int,
) -> List[Tuple[str, str]]:
    # Mirror generate_sentence.py load_style_images():
    #   fnames = writer_images[wstr][:5]
    # We display first refs_per_writer of the same ordered pool.
    if not rows:
        return []
    out = list(rows[:refs_per_writer])
    while len(out) < refs_per_writer and out:
        out.append(out[0])
    return out[:refs_per_writer]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def main() -> None:
    args = parse_args()
    if len(args.word_dirs) != len(args.word_labels):
        raise ValueError("--word_dirs and --word_labels must have the same length")

    words_dir = args.dataset_root / "words" / "words"
    if not words_dir.exists():
        raise FileNotFoundError(f"Missing words/words directory: {words_dir}")

    per_writer = load_meta_by_writer(args.meta_file)
    filename_lookup = build_filename_lookup(per_writer)
    refs_override = {}
    if args.refs_override_json is not None and args.refs_override_json.exists():
        refs_override = json.loads(args.refs_override_json.read_text(encoding="utf-8"))
    if args.top_writers and args.top_writers > 0:
        args.writers = pick_top_writers(per_writer, args.top_writers)
        print(f"Selected top writers by sample count: {args.writers}")

    # Collect images per row/column (no resizing).
    # columns = refs + generated words
    col_labels = [f"Ref {i+1}" for i in range(args.refs_per_writer)] + list(args.word_labels)
    n_cols = len(col_labels)

    row_images: List[List[Image.Image | None]] = []
    refs_manifest: Dict[str, List[Dict[str, str]]] = {}
    for writer in args.writers:
        writer_rows = per_writer.get(writer, [])
        if writer in refs_override:
            refs = []
            for fn in refs_override[writer]:
                row = filename_lookup.get(writer, {}).get(fn)
                if row is not None:
                    refs.append(row)
        elif args.match_generate_sentence_refs:
            refs = choose_reference_samples_like_generate_sentence(writer_rows, args.refs_per_writer)
        else:
            rng = random.Random(args.seed + int(writer))
            refs = choose_reference_samples(
                writer_rows,
                args.refs_per_writer,
                rng,
                min_ref_word_len=args.min_ref_word_len,
            )

        row: List[Image.Image | None] = []
        refs_manifest[writer] = []
        for i in range(args.refs_per_writer):
            if i < len(refs):
                fn, _ = refs[i]
                refs_manifest[writer].append({"filename": fn, "transcription": refs[i][1]})
                p = words_dir / fn
                if p.exists():
                    row.append(Image.open(p).convert("L"))
                else:
                    row.append(None)
            else:
                refs_manifest[writer].append({"filename": "", "transcription": ""})
                row.append(None)

        for wd in args.word_dirs:
            gp = args.run_root / wd / f"sentence_writer_{writer}.png"
            if gp.exists():
                row.append(Image.open(gp).convert("L"))
            else:
                row.append(None)

        row_images.append(row)

    # Layout dimensions.
    cell_pad_x = 16
    cell_pad_y = 12
    col_gap = 14
    row_gap = 18
    left_label_w = 120
    header_h = 52
    top_pad = 18
    bottom_pad = 18

    col_widths = [0] * n_cols
    row_heights = [0] * len(args.writers)

    for r_idx, row in enumerate(row_images):
        max_h = 0
        for c_idx, im in enumerate(row):
            if im is None:
                continue
            w, h = im.size
            col_widths[c_idx] = max(col_widths[c_idx], w + 2 * cell_pad_x)
            max_h = max(max_h, h + 2 * cell_pad_y)
        row_heights[r_idx] = max(max_h, 80)

    for i in range(n_cols):
        if col_widths[i] == 0:
            col_widths[i] = 180 if i >= args.refs_per_writer else 120

    extra_gap = args.separator_gap if args.separator_after_refs else 0
    total_w = left_label_w + sum(col_widths) + (n_cols - 1) * col_gap + extra_gap + 24
    total_h = top_pad + header_h + sum(row_heights) + (len(args.writers) - 1) * row_gap + bottom_pad

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_hdr = load_font(24)
    font_writer = load_font(24)

    # Column headers.
    x = left_label_w
    y_hdr = top_pad
    for c_idx, label in enumerate(col_labels):
        cw = col_widths[c_idx]
        tw, th = text_size(draw, label, font_hdr)
        tx = x + (cw - tw) // 2
        ty = y_hdr + (header_h - th) // 2
        draw.text((tx, ty), label, fill=(30, 30, 30), font=font_hdr)
        x += cw + col_gap
        if args.separator_after_refs and c_idx == args.refs_per_writer - 1:
            x += extra_gap

    # Rows.
    y = top_pad + header_h
    for r_idx, writer in enumerate(args.writers):
        rh = row_heights[r_idx]
        tw, th = text_size(draw, writer, font_writer)
        draw.text((left_label_w - tw - 18, y + (rh - th) // 2), writer, fill=(20, 20, 20), font=font_writer)

        x = left_label_w
        for c_idx, im in enumerate(row_images[r_idx]):
            cw = col_widths[c_idx]
            # light border
            draw.rectangle([x, y, x + cw, y + rh], outline=(220, 220, 220), width=1)
            if im is not None:
                iw, ih = im.size
                px = x + (cw - iw) // 2
                py = y + (rh - ih) // 2
                canvas.paste(im.convert("RGB"), (px, py))
            x += cw + col_gap
            if args.separator_after_refs and c_idx == args.refs_per_writer - 1:
                x += extra_gap
        y += rh + row_gap

    if args.separator_after_refs:
        sep_x = left_label_w + sum(col_widths[:args.refs_per_writer]) + col_gap * args.refs_per_writer + extra_gap // 2
        draw.line(
            [(sep_x, top_pad + 8), (sep_x, total_h - bottom_pad)],
            fill=(165, 165, 165),
            width=max(1, args.separator_width),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved figure: {args.output}")
    print(f"Canvas size: {canvas.size}")

    manifest_path = args.output.with_suffix(".refs.json")
    manifest = {
        "run_root": str(args.run_root),
        "writers": args.writers,
        "refs_per_writer": args.refs_per_writer,
        "match_generate_sentence_refs": args.match_generate_sentence_refs,
        "word_dirs": args.word_dirs,
        "word_labels": args.word_labels,
        "refs": refs_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved refs manifest: {manifest_path}")


if __name__ == "__main__":
    main()
