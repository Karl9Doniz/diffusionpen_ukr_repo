import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import torch

from utils.word_dataset import WordLineDataset


class UkrDataset(WordLineDataset):
    """
    Dataset wrapper for the UkrTextRec line-level dataset.
    Uses METAFILE.tsv (filename, transcription) and extracts writer ID from
    the third chunk of the filename: a01-001-0023-01 -> writer '0023'.
    """

    def __init__(
        self,
        basefolder,
        subset,
        segmentation_level,
        fixed_size,
        tokenizer,
        text_encoder,
        feat_extractor,
        transforms,
        meta_file=None,
        character_classes=None,
        lazy_images=True,
    ):
        super().__init__(
            basefolder,
            subset,
            segmentation_level,
            fixed_size,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            feat_extractor=feat_extractor,
            transforms=transforms,
            character_classes=character_classes,
        )
        self.lazy_images = bool(lazy_images)
        self.setname = "UKR_lazy" if self.lazy_images else "UKR"
        self.image_dir = os.path.join(self.basefolder, "lines", "lines")
        self.meta_file = meta_file or os.path.join(self.basefolder, "METAFILE.tsv")
        super().__finalize__()

    @staticmethod
    def _iter_metafile_rows(meta_file):
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"UkrTextRec metafile not found at '{meta_file}'.")
        with open(meta_file, "r", encoding="utf-8") as handle:
            first = handle.readline()
            if not first:
                return
            has_header = first.lower().startswith("filename")

            def _parse_line(line):
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    return None
                return parts[0].strip(), parts[1].strip()

            if has_header:
                for line in handle:
                    parsed = _parse_line(line)
                    if parsed is not None:
                        yield parsed
            else:
                parsed = _parse_line(first)
                if parsed is not None:
                    yield parsed
                for line in handle:
                    parsed = _parse_line(line)
                    if parsed is not None:
                        yield parsed

    @staticmethod
    def _normalise_text(text):
        return " ".join(text.replace("\u00a0", " ").split())

    def _load_image(self, image_path):
        return Image.open(image_path).convert("RGB")

    def _prepare_image(self, img):
        fheight, fwidth = self.fixed_size
        if fheight is None or fwidth is None:
            return img
        if img.width < fwidth or img.height < fheight:
            return ImageOps.pad(img, size=(fwidth, fheight), color="white")
        return img.resize((fwidth, fheight), Image.Resampling.LANCZOS)

    def __getitem__(self, index):
        img_ref, transcr, wid, img_path = self.data[index]
        if self.lazy_images:
            img = self._load_image(img_ref)
        else:
            img = img_ref
        img = self._prepare_image(img)

        if self.transforms is not None:
            img = self.transforms(img)

        writer_samples = self._samples_by_writer.get(wid, [])
        positive_samples = [p for p in writer_samples if len(p[1]) > 3]
        if len(positive_samples) == 0:
            positive_samples = writer_samples
        if len(positive_samples) == 0:
            # Defensive fallback for rare malformed records:
            # keep batch shape stable by using the current sample.
            positive_samples = [(img_ref, transcr, wid, img_path)]

        # Always provide exactly 5 style references so DataLoader collation
        # remains shape-stable across mixed writers.
        if len(positive_samples) >= 5:
            random_samples = random.sample(positive_samples, k=5)
        else:
            random_samples = random.choices(positive_samples, k=5)

        style_images = []
        for sample in random_samples:
            s_img_ref = sample[0]
            s_img = self._load_image(s_img_ref) if self.lazy_images else s_img_ref
            s_img = self._prepare_image(s_img)
            if self.transforms is not None:
                s_img_tensor = self.transforms(s_img)
            else:
                s_img_tensor = s_img
            style_images.append(s_img_tensor)

        s_imgs = torch.stack(style_images)

        cor_image_ref = random.choice(positive_samples)[0]
        cor_im = self._load_image(cor_image_ref) if self.lazy_images else cor_image_ref
        cor_im = self._prepare_image(cor_im)
        if self.transforms is not None:
            cor_im = self.transforms(cor_im)

        return img, transcr, wid, s_imgs, img_path, cor_im

    def main_loader(self, subset, segmentation_level):
        if segmentation_level.lower() not in ("word", "line"):
            raise ValueError(
                f"UKR dataset supports 'word'/'line' levels, got '{segmentation_level}'."
            )

        raw_records = []
        writer_raw_ids = []
        for filename, text in self._iter_metafile_rows(self.meta_file):
            sample_id = Path(filename).stem
            parts = sample_id.split("-")
            writer_raw = parts[2] if len(parts) >= 3 else parts[0]
            raw_records.append((filename, text, writer_raw))
            writer_raw_ids.append(writer_raw)

        writer_to_idx = {
            writer: idx for idx, writer in enumerate(sorted(set(writer_raw_ids)))
        }

        data = []
        missing_images = 0
        skipped = 0
        for filename, text, writer_raw in raw_records:
            image_path = os.path.join(self.image_dir, filename)
            if not os.path.exists(image_path):
                missing_images += 1
                continue

            text = self._normalise_text(text)
            if not text:
                skipped += 1
                continue

            writer_idx = writer_to_idx[writer_raw]
            if self.lazy_images:
                data.append((image_path, text, writer_idx, image_path))
            else:
                try:
                    img = Image.open(image_path).convert("RGB")
                except Exception:
                    skipped += 1
                    continue
                data.append((img, text, writer_idx, image_path))

        if missing_images:
            print(f"UkrDataset: skipped {missing_images} items (missing files).")
        if skipped:
            print(f"UkrDataset: skipped {skipped} items due to processing issues.")

        return data


class UkrWordDataset(WordLineDataset):
    """
    Dataset wrapper for word-level Ukrainian dataset.
    Uses word-segmented data with METAFILE.tsv format:
    filename, transcription, line_source, word_index, bbox

    Writer ID extraction remains unchanged: parts[2] from filename
    (e.g., a01-001-0023-01-w00.png -> writer '0023')
    """

    def __init__(
        self,
        basefolder,
        subset,
        segmentation_level,
        fixed_size,
        tokenizer,
        text_encoder,
        feat_extractor,
        transforms,
        meta_file=None,
        character_classes=None,
        lazy_images=True,
    ):
        super().__init__(
            basefolder,
            subset,
            segmentation_level,
            fixed_size,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            feat_extractor=feat_extractor,
            transforms=transforms,
            character_classes=character_classes,
        )
        self.lazy_images = bool(lazy_images)
        self.setname = "UKR_WORD_lazy" if self.lazy_images else "UKR_WORD"
        self.image_dir = os.path.join(self.basefolder, "words", "words")
        self.meta_file = meta_file or os.path.join(self.basefolder, "METAFILE.tsv")
        super().__finalize__()

    @staticmethod
    def _iter_metafile_rows(meta_file):
        """
        Parse word-level METAFILE.tsv with format:
        filename, transcription, line_source, word_index, bbox
        """
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"UkrWord metafile not found at '{meta_file}'.")

        with open(meta_file, "r", encoding="utf-8") as handle:
            first = handle.readline()
            if not first:
                return

            has_header = first.lower().startswith("filename")

            def _parse_line(line):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    return None
                # Return filename and transcription (first 2 columns)
                return parts[0].strip(), parts[1].strip()

            if has_header:
                for line in handle:
                    parsed = _parse_line(line)
                    if parsed is not None:
                        yield parsed
            else:
                parsed = _parse_line(first)
                if parsed is not None:
                    yield parsed
                for line in handle:
                    parsed = _parse_line(line)
                    if parsed is not None:
                        yield parsed

    @staticmethod
    def _normalise_text(text):
        return " ".join(text.replace("\u00a0", " ").split())

    def _load_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        arr_gray = np.array(img.convert("L"))
        h, w = arr_gray.shape

        # Two raw thresholds (no Sauvola — adaptive thresholds miss faint ruling dots):
        #   thresh_line:   detects ruling line pixels (value < 249, printed paper lines 170–248)
        #   thresh_letter: detects genuine pen strokes (value < 200, darker ink)
        thresh_line   = (arr_gray < 249).astype(np.uint8) * 255
        thresh_letter = (arr_gray < 200).astype(np.uint8) * 255

        arr = np.array(img)
        zone = int(h * 0.80)
        for row in range(h - 1, zone - 1, -1):
            ink_cols = np.where(thresh_line[row] > 0)[0]
            if len(ink_cols) < 2:
                continue
            density = len(ink_cols) / w
            span = (int(ink_cols[-1]) - int(ink_cols[0])) / w
            if span > 0.40 and density < 0.18:
                # Descender check: only look at the 3 rows directly above the
                # ruling row. All-rows check protects every column (word covers
                # full width); tight window correctly preserves only true descenders.
                check_start = max(0, row - 3)
                ink_above = thresh_letter[check_start:row, :].any(axis=0)
                erase_mask = (thresh_line[row] > 0) & ~ink_above
                arr[row, erase_mask] = 255
        return Image.fromarray(arr)

    def _prepare_image(self, img):
        fheight, fwidth = self.fixed_size
        if fheight is None or fwidth is None:
            return img
        if img.width < fwidth or img.height < fheight:
            return ImageOps.pad(img, size=(fwidth, fheight), color="white")
        return img.resize((fwidth, fheight), Image.Resampling.LANCZOS)

    def __getitem__(self, index):
        img_ref, transcr, wid, img_path = self.data[index]
        if self.lazy_images:
            img = self._load_image(img_ref)
        else:
            img = img_ref
        img = self._prepare_image(img)

        if self.transforms is not None:
            img = self.transforms(img)

        # Select style samples (5 random samples from same writer)
        writer_samples = self._samples_by_writer.get(wid, [])
        positive_samples = [p for p in writer_samples if len(p[1]) > 3]
        if len(positive_samples) == 0:
            positive_samples = writer_samples
        if len(positive_samples) == 0:
            # Defensive fallback for rare malformed records:
            # keep batch shape stable by using the current sample.
            positive_samples = [(img_ref, transcr, wid, img_path)]

        # Always provide exactly 5 style references so DataLoader collation
        # remains shape-stable across mixed writers.
        if len(positive_samples) >= 5:
            random_samples = random.sample(positive_samples, k=5)
        else:
            random_samples = random.choices(positive_samples, k=5)

        style_images = []
        for sample in random_samples:
            s_img_ref = sample[0]
            s_img = self._load_image(s_img_ref) if self.lazy_images else s_img_ref
            s_img = self._prepare_image(s_img)
            if self.transforms is not None:
                s_img_tensor = self.transforms(s_img)
            else:
                s_img_tensor = s_img
            style_images.append(s_img_tensor)

        s_imgs = torch.stack(style_images)

        cor_image_ref = random.choice(positive_samples)[0]
        cor_im = self._load_image(cor_image_ref) if self.lazy_images else cor_image_ref
        cor_im = self._prepare_image(cor_im)
        if self.transforms is not None:
            cor_im = self.transforms(cor_im)

        return img, transcr, wid, s_imgs, img_path, cor_im

    def main_loader(self, subset, segmentation_level):
        """
        Load word-level data.
        Writer ID extraction: parts[2] from filename (unchanged from line-level)
        Example: a01-001-0023-01-w00.png -> writer '0023'
        """
        if segmentation_level.lower() not in ("word", "line"):
            raise ValueError(
                f"UKRWord dataset supports 'word'/'line' levels, got '{segmentation_level}'."
            )

        raw_records = []
        writer_raw_ids = []
        for filename, text in self._iter_metafile_rows(self.meta_file):
            sample_id = Path(filename).stem
            parts = sample_id.split("-")
            # Extract writer ID from 3rd segment (same as line-level)
            writer_raw = parts[2] if len(parts) >= 3 else parts[0]
            raw_records.append((filename, text, writer_raw))
            writer_raw_ids.append(writer_raw)

        writer_to_idx = {
            writer: idx for idx, writer in enumerate(sorted(set(writer_raw_ids)))
        }

        data = []
        missing_images = 0
        skipped = 0
        for filename, text, writer_raw in raw_records:
            image_path = os.path.join(self.image_dir, filename)
            if not os.path.exists(image_path):
                missing_images += 1
                continue

            text = self._normalise_text(text)
            if not text:
                skipped += 1
                continue

            writer_idx = writer_to_idx[writer_raw]
            if self.lazy_images:
                data.append((image_path, text, writer_idx, image_path))
            else:
                try:
                    img = Image.open(image_path).convert("RGB")
                except Exception:
                    skipped += 1
                    continue
                data.append((img, text, writer_idx, image_path))

        if missing_images:
            print(f"UkrWordDataset: skipped {missing_images} items (missing files).")
        if skipped:
            print(f"UkrWordDataset: skipped {skipped} items due to processing issues.")

        return data
