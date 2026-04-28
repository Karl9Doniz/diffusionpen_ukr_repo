#!/usr/bin/env python3
"""
Oversample words containing rare Ukrainian letters to balance letter frequency.

Reads a METAFILE TSV, duplicates rows whose transcriptions contain underrepresented
letters, and writes a balanced METAFILE. No new images are needed — existing word
images are reused with their existing labels.

Usage:
    python scripts/balance_rare_letters.py \
        --input  /path/to/METAFILE_extended_v2.tsv \
        --output /path/to/METAFILE_extended_balanced.tsv \
        --max-factor 5

The oversample factor per letter is: min(target_count / current_count, max_factor),
rounded to the nearest integer. A word containing multiple rare letters is
oversampled by the maximum factor across all rare letters it contains, to avoid
exponential duplication.
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Target letter configuration
# Letters and their desired minimum occurrence counts in the final dataset.
# Letters not in this dict are left as-is.
# ---------------------------------------------------------------------------
TARGET_COUNTS = {
    'ї': 8_000,
    'щ': 6_000,
    'ф': 6_000,
    'Щ': 1_000,
    'Є': 500,
    'Ц': 300,
    'ґ': 500,
    # є and ц are already well-represented (5800+ and 4900+), leave them
}


def letter_freq(transcriptions):
    counter = Counter()
    for t in transcriptions:
        counter.update(t)
    return counter


def main():
    parser = argparse.ArgumentParser(description='Balance rare letter frequency by oversampling')
    parser.add_argument('--input',  type=str, required=True, help='Input METAFILE TSV')
    parser.add_argument('--output', type=str, required=True, help='Output balanced METAFILE TSV')
    parser.add_argument('--max-factor', type=int, default=5,
                        help='Maximum oversample multiplier per word (default: 5)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading {args.input}")
    df = pd.read_csv(args.input, sep='\t')
    print(f"Input: {len(df):,} entries")

    # --- Baseline letter counts ---
    base_freq = letter_freq(df['transcription'].dropna())
    print(f"\nBaseline counts for target letters:")
    for ch, target in TARGET_COUNTS.items():
        current = base_freq.get(ch, 0)
        factor = min(target / max(current, 1), args.max_factor)
        print(f"  {ch!r:4s}: current={current:>6,}  target={target:>6,}  "
              f"factor={factor:.2f}x  {'(skip)' if factor < 1.5 else ''}")

    # --- Compute per-row oversample factor ---
    # For each row, find the max factor across all rare letters it contains.
    # Factor of 1 means no duplication (row appears once).
    def row_factor(transcription):
        if not isinstance(transcription, str):
            return 1
        max_f = 1
        for ch, target in TARGET_COUNTS.items():
            if ch in transcription:
                current = base_freq.get(ch, 1)
                f = min(target / max(current, 1), args.max_factor)
                f = max(round(f), 1)
                max_f = max(max_f, f)
        return max_f

    df['_factor'] = df['transcription'].apply(row_factor)

    n_boosted = (df['_factor'] > 1).sum()
    print(f"\nWords to oversample: {n_boosted:,} "
          f"({100*n_boosted/len(df):.1f}%)")

    # --- Build balanced dataframe ---
    parts = []
    for factor, group in df.groupby('_factor'):
        if factor == 1:
            parts.append(group)
        else:
            # Original + (factor-1) duplicates
            parts.append(group)
            for _ in range(factor - 1):
                parts.append(group)

    df_balanced = pd.concat(parts, ignore_index=True)
    df_balanced = df_balanced.drop(columns=['_factor'])

    # Shuffle so duplicates aren't adjacent in the dataloader
    df_balanced = df_balanced.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # --- Report new letter counts ---
    new_freq = letter_freq(df_balanced['transcription'].dropna())
    print(f"\nAfter balancing ({len(df_balanced):,} total entries):")
    print(f"  {'Letter':<6} {'Before':>8}  {'After':>8}  {'Factor':>8}")
    print(f"  {'-'*38}")
    for ch, target in TARGET_COUNTS.items():
        before = base_freq.get(ch, 0)
        after  = new_freq.get(ch, 0)
        factor = after / max(before, 1)
        print(f"  {ch!r:<6} {before:>8,}  {after:>8,}  {factor:>7.2f}x")

    # Also show top common letters to confirm they're not badly skewed
    print(f"\nTop-10 letters after balancing:")
    for ch, cnt in new_freq.most_common(10):
        print(f"  {ch!r}: {cnt:,}")

    # --- Save ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(out_path, sep='\t', index=False)
    print(f"\nSaved {len(df_balanced):,} entries to {out_path}")


if __name__ == '__main__':
    main()
