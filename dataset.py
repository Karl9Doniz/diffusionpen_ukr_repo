import os
from pathlib import Path
import json
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image
from tqdm import tqdm

from utils.auxilary_functions import centered_PIL
from utils.auxilary_functions import image_resize_PIL
from utils.hkr_utils import DEFAULT_SPLIT_VARIANTS
from utils.hkr_utils import HKRRecord
from utils.hkr_utils import load_hkr_records
from utils.hkr_utils import load_split_ids
from utils.hkr_utils import resolve_default_mapping_path
from utils.hkr_utils import resolve_default_split_dir
from utils.word_dataset import WordLineDataset


class HKRStyle(WordLineDataset):
    """Dataset helper for the HKR (Handwritten Kazakh/Russian) word images.

    The original IAM-style pipeline assumes Latin characters.  This wrapper
    mirrors that behaviour but reads annotations from ``output.csv`` inside
    ``hkr-dataset`` and exposes writer-aware splits so the rest of the training
    code can remain untouched.
    """

    def __init__(
        self,
        basefolder: str,
        subset: str,
        segmentation_level: str,
        fixed_size: Tuple[int, Optional[int]],
        tokenizer=None,
        text_encoder=None,
        feat_extractor=None,
        transforms=None,
        character_classes: Optional[Sequence[str]] = None,
        writer_splits: Optional[Dict[str, Iterable[str]]] = None,
        mapping_file: Optional[str] = None,
        split_dir: Optional[str] = None,
        split_variants: Optional[Sequence[str]] = None,
        sample_splits: Optional[Dict[str, Iterable[str]]] = None,
        cluster_map_file: Optional[str] = None,
    ):
        super().__init__(
            basefolder=basefolder,
            subset=subset,
            segmentation_level=segmentation_level,
            fixed_size=fixed_size,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            feat_extractor=feat_extractor,
            transforms=transforms,
            character_classes=character_classes,
        )
        self.setname = "HKR_v2"
        self.annotation_file = os.path.join(self.basefolder, "output.csv")
        self.image_dir = os.path.join(self.basefolder, "img")
        self.mapping_path = self._resolve_mapping_path(mapping_file)
        self.split_dir = self._resolve_split_dir(split_dir)
        self.split_variants = tuple(split_variants or DEFAULT_SPLIT_VARIANTS)
        self.sample_cluster_map = self._load_cluster_map(cluster_map_file)
        self._records_by_sample = self._load_records()
        self._records = [
            self._records_by_sample[key] for key in sorted(self._records_by_sample)
        ]
        self._sample_split_overrides = self._prepare_sample_overrides(sample_splits)
        self._sample_split_cache: Dict[str, Set[str]] = {}
        self.writer_splits = self._build_writer_splits(writer_splits)
        super().__finalize__()

    def _resolve_mapping_path(self, mapping_file: Optional[str]) -> str:
        if mapping_file:
            if not os.path.exists(mapping_file):
                raise FileNotFoundError(
                    f"HKR mapping file not found at '{mapping_file}'."
                )
            return os.path.abspath(mapping_file)
        return str(resolve_default_mapping_path(self.basefolder))

    def _resolve_split_dir(self, split_dir: Optional[str]) -> Optional[str]:
        if split_dir:
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(
                    f"HKR split directory not found at '{split_dir}'."
                )
            return str(Path(split_dir).resolve())
        resolved = resolve_default_split_dir(self.basefolder)
        return str(resolved) if resolved is not None else None

    def _load_cluster_map(self, cluster_map_file: Optional[str]) -> Dict[str, str]:
        candidates = []
        if cluster_map_file:
            candidates.append(cluster_map_file)
        candidates.append(os.path.join(self.basefolder, "writer_cluster_map_embedded.json"))
        candidates.append("writer_cluster_map_embedded.json")

        for path in candidates:
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                sample_map = data.get("sample_cluster", data)
                return {k: str(v) for k, v in sample_map.items()}
        return {}

    def _prepare_sample_overrides(
        self, overrides: Optional[Dict[str, Iterable[str]]]
    ) -> Dict[str, Set[str]]:
        if not overrides:
            return {}
        prepared: Dict[str, Set[str]] = {}
        for key, values in overrides.items():
            subset = key.lower()
            sample_ids = {self._to_sample_id(value) for value in values}
            prepared[subset] = {sid for sid in sample_ids if sid}
        return prepared

    @staticmethod
    def _to_sample_id(value: str) -> str:
        value = value.strip()
        if value.endswith(".jpg"):
            value = value[:-4]
        return value

    def _load_records(self) -> Dict[str, HKRRecord]:
        raw_records = load_hkr_records(self.mapping_path)
        cleaned: Dict[str, HKRRecord] = {}
        for sample_id, record in raw_records.items():
            text = self._normalise_text(record.text)
            writer_id = record.writer_id or sample_id.split("_", 1)[0]
            # remap writer id if we have a clustered mapping
            writer_id = self.sample_cluster_map.get(record.filename, writer_id)
            cleaned[sample_id] = HKRRecord(
                sample_id=sample_id,
                filename=record.filename,
                text=text,
                writer_id=writer_id,
            )
        return cleaned

    @staticmethod
    def _normalise_text(text: str) -> str:
        return " ".join(text.replace("\u00a0", " ").split())

    @staticmethod
    def _prepare_image(img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")

        target_height, target_width = 64, 256
        # Preserve aspect ratio while fitting the height first.
        img = image_resize_PIL(img, height=target_height)
        if img.width > target_width:
            img = image_resize_PIL(img, width=target_width)
        img = centered_PIL(img, (target_height, target_width), border_value=255.0)
        return img

    def _load_sample_split(self, subset: str) -> Set[str]:
        subset = subset.lower()
        if subset in self._sample_split_overrides:
            return self._sample_split_overrides[subset]

        if subset in self._sample_split_cache:
            return self._sample_split_cache[subset]

        split_path = Path(self.split_dir) if self.split_dir is not None else None
        split_ids = load_split_ids(subset, split_path, self.split_variants)
        if split_ids is None:
            result: Set[str] = set()
        else:
            result = {self._to_sample_id(sample) for sample in split_ids}
        self._sample_split_cache[subset] = result
        return result

    def _build_writer_splits(
        self, overrides: Optional[Dict[str, Iterable[str]]]
    ) -> Dict[str, List[str]]:
        if overrides is not None:
            return {k: list(v) for k, v in overrides.items()}

        splits: Dict[str, List[str]] = {}
        for subset in ("train", "val", "test"):
            sample_ids = self._load_sample_split(subset)
            if not sample_ids:
                continue
            writers = {
                self._records_by_sample[sample_id].writer_id
                for sample_id in sample_ids
                if sample_id in self._records_by_sample
            }
            splits[subset] = sorted(writers)

        if splits:
            return splits

        writers = sorted({record.writer_id for record in self._records})
        if not writers:
            return {"train": [], "val": [], "test": []}

        if len(writers) == 1:
            train = writers.copy()
            return {"train": train, "val": train.copy(), "test": train.copy()}

        if len(writers) == 2:
            return {
                "train": [writers[0]],
                "val": [writers[0]],
                "test": [writers[1]],
            }

        return {
            "train": writers[:-2],
            "val": [writers[-2]],
            "test": [writers[-1]],
        }

    def _filtered_annotations(
        self, subset: str
    ) -> List[HKRRecord]:
        subset = subset.lower()
        if subset in ("all", "full"):
            return list(self._records)

        sample_ids = self._load_sample_split(subset)
        if sample_ids:
            return [
                record
                for record in self._records
                if record.sample_id in sample_ids
            ]

        allowed = set(self.writer_splits.get(subset, []))
        if not allowed:
            return list(self._records)

        return [
            record
            for record in self._records
            if record.writer_id in allowed
        ]

    def main_loader(self, subset, segmentation_level) -> List[Tuple[Image.Image, str, str, str]]:
        if segmentation_level.lower() != "word":
            raise ValueError(
                f"HKR dataset currently supports only 'word' level, got '{segmentation_level}'."
            )

        records = self._filtered_annotations(subset)
        data: List[Tuple[Image.Image, str, str, str]] = []
        missing_images = 0
        skipped = 0

        iterator = tqdm(
            records,
            desc=f"Loading HKR ({subset})",
            leave=False,
        )
        for record in iterator:
            filename = record.filename
            image_path = os.path.join(self.image_dir, filename)
            if not os.path.exists(image_path):
                missing_images += 1
                continue

            try:
                img = Image.open(image_path)
            except Exception:
                skipped += 1
                continue

            try:
                img = self._prepare_image(img)
            except Exception:
                skipped += 1
                continue

            text = record.text
            if not text:
                skipped += 1
                continue

            writer_id = record.writer_id or record.sample_id.split("_", 1)[0]

            data.append((img, text, writer_id, image_path))

        if missing_images:
            print(f"HKRStyle: skipped {missing_images} items (missing files).")
        if skipped:
            print(f"HKRStyle: skipped {skipped} items due to processing issues.")

        return data
