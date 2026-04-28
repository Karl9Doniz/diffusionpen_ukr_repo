import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class HKRRecord:
    """Single HKR annotation entry."""

    sample_id: str
    filename: str
    text: str
    writer_id: str


SPLIT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "train": ("train",),
    "val": ("val", "validation", "dev"),
    "validation": ("val", "validation", "dev"),
    "test": ("test",),
    "train_val": ("train_val",),
}

DEFAULT_SPLIT_VARIANTS: Tuple[str, ...] = ("mapped", "fixed", "official", "v1", "")


def _parse_uttlist_line(line: str) -> Optional[Tuple[str, str, str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # First try CSV parsing with commas.
    csv_row = next(csv.reader([line], delimiter=",", quotechar='"'))
    if len(csv_row) >= 3:
        writer = csv_row[0].strip()
        sample = csv_row[1].strip()
        text = ",".join(csv_row[2:]).strip()
        return writer, sample, text

    if len(csv_row) == 2:
        writer = csv_row[0].strip()
        remainder = csv_row[1].strip()
        if " " in remainder:
            sample, text = remainder.split(" ", 1)
        else:
            sample, text = remainder, ""
        return writer, sample.strip(), text.strip()

    # Fallback to tab-separated values.
    tab_parts = line.split("\t")
    if len(tab_parts) >= 3:
        writer = tab_parts[0].strip()
        sample = tab_parts[1].strip()
        text = "\t".join(tab_parts[2:]).strip()
        return writer, sample, text

    if len(tab_parts) == 2:
        writer = tab_parts[0].strip()
        sample = tab_parts[1].strip()
        return writer, sample, ""

    # Final fallback: split on the first comma.
    if "," in line:
        writer, rest = line.split(",", 1)
        rest = rest.strip()
        if " " in rest:
            sample, text = rest.split(" ", 1)
        else:
            sample, text = rest, ""
        return writer.strip(), sample.strip(), text.strip()

    parts = line.split()
    if not parts:
        return None
    sample = parts[0]
    text = " ".join(parts[1:])
    return "", sample, text


def _resolve_candidates(
    basefolder: Optional[str],
    relative: Iterable[str],
) -> Sequence[Path]:
    candidates = []
    repo_root = Path(__file__).resolve().parent.parent
    for rel in relative:
        candidates.append(repo_root.joinpath(rel))

    if basefolder:
        base = Path(basefolder)
        for rel in relative:
            candidates.append(base.joinpath(rel))
        candidates.append(base.parent.joinpath(relative[0]))

    deduped: Dict[Path, None] = {}
    for candidate in candidates:
        cand = candidate.resolve()
        deduped[cand] = None
    return tuple(deduped.keys())


def resolve_default_mapping_path(basefolder: Optional[str] = None) -> Path:
    candidates = _resolve_candidates(basefolder, ("all_mapped.txt",))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "HKR mapping file 'all_mapped.txt' was not found in the repository root or dataset folder."
    )


def resolve_default_split_dir(basefolder: Optional[str] = None) -> Optional[Path]:
    candidates = _resolve_candidates(basefolder, ("utils/hkr_split", "hkr_split", "splits"))
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


@lru_cache(maxsize=None)
def load_hkr_records(mapping_path: str) -> Dict[str, HKRRecord]:
    path = Path(mapping_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"HKR mapping file not found at {path}")

    records: Dict[str, HKRRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_uttlist_line(line)
            if not parsed:
                continue
            writer_id, sample_id, text = parsed
            sample_id = sample_id.strip()
            writer_id = writer_id.strip()
            if not sample_id:
                continue
            filename = f"{sample_id}.jpg"
            records[sample_id] = HKRRecord(
                sample_id=sample_id,
                filename=filename,
                text=text.strip(),
                writer_id=writer_id,
            )
    return records


@lru_cache(maxsize=None)
def _load_split_ids_cached(split_path: str) -> Set[str]:
    ids: Set[str] = set()
    path = Path(split_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_uttlist_line(line)
            if not parsed:
                continue
            _, sample_id, _ = parsed
            sample_id = sample_id.strip()
            if sample_id:
                ids.add(sample_id)
    return ids


def find_split_file(
    subset: str,
    split_dir: Optional[Path],
    variants: Sequence[str] = DEFAULT_SPLIT_VARIANTS,
) -> Optional[Path]:
    if split_dir is None:
        return None

    subset = subset.lower()
    names = [subset]
    names.extend(SPLIT_ALIASES.get(subset, ()))

    candidates: Sequence[Path] = ()
    for name in names:
        for variant in variants:
            possible: Tuple[Path, ...] = ()
            if variant:
                possible = (
                    split_dir / f"{name}_{variant}.uttlist",
                    split_dir / f"{name}-{variant}.uttlist",
                )
            else:
                possible = (split_dir / f"{name}.uttlist",)
            candidates += possible

    seen: Set[Path] = set()
    ordered: Sequence[Path] = ()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered += (resolved,)

    for path in ordered:
        if path.exists():
            return path
    return None


def load_split_ids(
    subset: str,
    split_dir: Optional[Path],
    variants: Sequence[str] = DEFAULT_SPLIT_VARIANTS,
) -> Optional[Set[str]]:
    split_file = find_split_file(subset, split_dir, variants)
    if split_file is None:
        return None
    return _load_split_ids_cached(str(split_file))

