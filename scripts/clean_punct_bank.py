"""
Automatic punctuation bank cleaner.

Moves suspicious images to <subdir>/_rejected/ for manual review.
Does NOT delete — always verify _rejected/ before removing.

Run:
    python scripts/clean_punct_bank.py --bank generated/punct_bank [--dry-run]
"""

import argparse, os, shutil
import cv2
import numpy as np
from PIL import Image


def image_metrics(path: str) -> dict:
    img = np.array(Image.open(path).convert("L"))
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ink = (bw > 0).astype(np.uint8)
    h, w = img.shape
    ink_pixels = int(ink.sum())
    ir = ink_pixels / (h * w) if h * w > 0 else 0.0
    ar = h / w if w > 0 else 1.0

    pts = np.column_stack(np.where(ink > 0))
    if len(pts) >= 3:
        hull = cv2.convexHull(pts[:, ::-1].astype(np.float32))
        hull_area = cv2.contourArea(hull)
        solidity = ink_pixels / hull_area if hull_area > 0 else 1.0
    else:
        solidity = 1.0

    n_cc, cc_labels = cv2.connectedComponents(ink)
    n_cc = n_cc - 1  # subtract background

    # vertical centroid (0 = top, 1 = bottom)
    rows = ink.mean(axis=1)
    denom = rows.sum()
    cm_y = float((rows * np.arange(h)).sum() / denom) / h if denom > 0 else 0.5

    return dict(
        h=h, w=w, ar=ar, ir=ir,
        sol=solidity, ink_px=ink_pixels,
        n_cc=n_cc, cm_y=cm_y,
    )


# ── per-sign reject predicates ────────────────────────────────────────────────

def _bad_comma(m: dict) -> str | None:
    """A comma is a thin curved stroke with a downward tail.
    Reject if: letter-like enclosed shape (low sol + large), or absurdly huge."""
    if m["sol"] < 0.33 and m["ink_px"] > 170:
        return f"letter-like shape (sol={m['sol']:.3f}, ink_px={m['ink_px']})"
    if m["ink_px"] > 700:
        return f"too large (ink_px={m['ink_px']})"
    if m["ar"] < 0.30:
        return f"too horizontal for a comma (ar={m['ar']:.2f})"
    return None


def _bad_period(m: dict) -> str | None:
    """A period is a small filled dot — roughly square/circular."""
    if m["ar"] < 0.25:
        return f"too horizontal for a period (ar={m['ar']:.2f})"
    if m["ar"] > 4.5:
        return f"too vertical for a period (ar={m['ar']:.2f})"
    if m["ink_px"] > 800:
        return f"too large (ink_px={m['ink_px']})"
    if m["ir"] < 0.06:
        return f"too sparse for a period (ir={m['ir']:.3f})"
    return None


def _bad_colon(m: dict) -> str | None:
    """A colon is two vertically stacked dots."""
    if m["ir"] < 0.04:
        return f"nearly blank (ir={m['ir']:.3f}, ink_px={m['ink_px']})"
    if m["ar"] > 6.0:
        return f"extremely tall/narrow — likely not a colon (ar={m['ar']:.2f})"
    if m["ar"] < 0.40:
        return f"too horizontal for a colon (ar={m['ar']:.2f})"
    if m["ink_px"] > 900:
        return f"too large (ink_px={m['ink_px']})"
    return None


def _bad_hyphen(m: dict) -> str | None:
    """A hyphen is a short horizontal stroke — width >> height."""
    if m["ar"] > 0.75:
        return f"not horizontal enough for a hyphen (ar={m['ar']:.2f})"
    if m["ink_px"] > 600:
        return f"too large (ink_px={m['ink_px']})"
    if m["ir"] < 0.05:
        return f"too sparse (ir={m['ir']:.3f})"
    return None


def _bad_question(m: dict) -> str | None:
    """Question marks are complex — only remove clear outliers."""
    if m["ink_px"] > 1200:
        return f"too large (ink_px={m['ink_px']})"
    if m["ar"] < 0.30:
        return f"too horizontal for a question mark (ar={m['ar']:.2f})"
    return None


def _bad_exclaim(m: dict) -> str | None:
    """Exclamation: tall vertical stroke + dot below (2 components max)."""
    if m["ar"] < 0.50:
        return f"too horizontal for an exclamation mark (ar={m['ar']:.2f})"
    if m["ink_px"] > 1200:
        return f"too large (ink_px={m['ink_px']})"
    if m["n_cc"] >= 3:
        return f"too many disconnected components (n_cc={m['n_cc']})"
    return None


def _bad_semicolon(m: dict) -> str | None:
    """Semicolon: comma-tail + dot above."""
    if m["ink_px"] > 900:
        return f"too large (ink_px={m['ink_px']})"
    if m["ar"] < 0.35:
        return f"too horizontal for a semicolon (ar={m['ar']:.2f})"
    if m["ir"] < 0.05:
        return f"nearly blank (ir={m['ir']:.3f}, ink_px={m['ink_px']})"
    if m["ar"] > 6.0 and m["n_cc"] == 1:
        return f"single tall stroke — not a semicolon (ar={m['ar']:.2f}, n_cc=1)"
    return None


SIGN_PREDICATES = {
    "comma":     _bad_comma,
    "period":    _bad_period,
    "colon":     _bad_colon,
    "hyphen":    _bad_hyphen,
    "question":  _bad_question,
    "exclaim":   _bad_exclaim,
    "semicolon": _bad_semicolon,
    "dash":      None,  # only 1 file, skip
}


# ── main ──────────────────────────────────────────────────────────────────────

def clean_subdir(subdir: str, pred, dry_run: bool) -> tuple[int, int]:
    if pred is None:
        return 0, 0

    files = sorted(f for f in os.listdir(subdir)
                   if f.lower().endswith(".png") and not f.startswith("_"))
    rejected, kept = [], []
    for fname in files:
        path = os.path.join(subdir, fname)
        try:
            m = image_metrics(path)
        except Exception as e:
            print(f"  WARN: could not read {fname}: {e}")
            continue
        reason = pred(m)
        if reason:
            rejected.append((fname, reason))
        else:
            kept.append(fname)

    reject_dir = os.path.join(subdir, "_rejected")
    if rejected:
        if not dry_run:
            os.makedirs(reject_dir, exist_ok=True)
        for fname, reason in rejected:
            src = os.path.join(subdir, fname)
            dst = os.path.join(reject_dir, fname)
            print(f"  REJECT {fname}: {reason}")
            if not dry_run:
                shutil.move(src, dst)

    return len(rejected), len(kept)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default="generated/punct_bank")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be rejected without moving files")
    args = parser.parse_args()

    bank = os.path.abspath(args.bank)
    if not os.path.isdir(bank):
        raise SystemExit(f"Bank directory not found: {bank}")

    total_rej = total_kept = 0
    for sign, pred in sorted(SIGN_PREDICATES.items()):
        subdir = os.path.join(bank, sign)
        if not os.path.isdir(subdir):
            continue
        n_files = sum(1 for f in os.listdir(subdir)
                      if f.lower().endswith(".png") and not f.startswith("_"))
        print(f"\n── {sign} ({n_files} files) ──")
        if pred is None:
            print("  skipped (no predicate)")
            continue
        rej, kept = clean_subdir(subdir, pred, args.dry_run)
        print(f"  → rejected {rej}, kept {kept}")
        total_rej += rej
        total_kept += kept

    mode = "DRY RUN" if args.dry_run else "DONE"
    print(f"\n{mode}: {total_rej} rejected, {total_kept} kept")
    if not args.dry_run and total_rej > 0:
        print("Rejected files moved to <sign>/_rejected/ — review before deleting.")


if __name__ == "__main__":
    main()
