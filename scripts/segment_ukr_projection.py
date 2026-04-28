import argparse
import cv2
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from difflib import SequenceMatcher

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


class CCSegmenter:
    """
    Segments handwritten lines into words using connected components.
    Merges nearby components (broken strokes, diacritics), then picks
    the top N-1 widest gaps as word boundaries (N = ground truth word count).
    Optionally validates with TrOCR.
    """

    def __init__(self, trocr_model='cyrillic-trocr/trocr-handwritten-cyrillic',
                 device='cuda:0', use_trocr=True, merge_dist=8,
                 min_component_area=10):
        self.device = device
        self.use_trocr = use_trocr
        self.merge_dist = merge_dist
        self.min_component_area = min_component_area

        if use_trocr:
            print(f"Loading TrOCR model: {trocr_model}")
            self.trocr_processor = TrOCRProcessor.from_pretrained(trocr_model)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model)
            self.trocr_model = self.trocr_model.to(device)
            self.trocr_model.eval()
            print("TrOCR loaded.")
        else:
            self.trocr_processor = None
            self.trocr_model = None
            print("Running without TrOCR.")

    def _merge_nearby_components(self, components):
        """Merge components whose x-ranges are within merge_dist pixels."""
        if not components:
            return []
        comps = sorted(components, key=lambda c: c['x'])
        merged = [comps[0].copy()]
        for c in comps[1:]:
            last = merged[-1]
            if c['x'] <= last['x_end'] + self.merge_dist:
                last['x'] = min(last['x'], c['x'])
                last['y'] = min(last['y'], c['y'])
                last['x_end'] = max(last['x_end'], c['x_end'])
                y_end = max(last['y'] + last['h'], c['y'] + c['h'])
                last['w'] = last['x_end'] - last['x']
                last['h'] = y_end - last['y']
                last['area'] += c['area']
            else:
                merged.append(c.copy())
        return merged

    def _find_cc_gaps(self, gray):
        """Find connected components, merge nearby ones, return gaps between groups."""
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        components = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < self.min_component_area:
                continue
            components.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                              'area': int(area), 'x_end': int(x + w)})

        if not components:
            return [], None, None

        groups = self._merge_nearby_components(components)

        x_min = min(g['x'] for g in groups)
        x_max = max(g['x_end'] for g in groups)

        gaps = []
        for i in range(len(groups) - 1):
            gap_start = groups[i]['x_end']
            gap_end = groups[i + 1]['x']
            gap_width = gap_end - gap_start
            if gap_width > self.merge_dist:
                gaps.append({'start': gap_start, 'end': gap_end,
                            'width': gap_width,
                            'center': (gap_start + gap_end) // 2})

        # Sort by width descending
        gaps.sort(key=lambda g: g['width'], reverse=True)
        return gaps, x_min, x_max

    def recognize_word(self, word_img):
        rgb = cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pixel_values = self.trocr_processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.trocr_model.generate(pixel_values, max_length=64)
        text = self.trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return text.strip()

    def text_similarity(self, a, b):
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def segment_line(self, img_path, ground_truth):
        img = cv2.imread(img_path)
        if img is None:
            return [], 'failed', {}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        gt_words = ground_truth.split()
        target_count = len(gt_words)

        gaps, x_min, x_max = self._find_cc_gaps(gray)

        if x_min is None:
            return [], 'failed', {}

        # Pick top N-1 widest gaps as word boundaries
        n_splits = min(target_count - 1, len(gaps))
        selected = sorted(gaps[:n_splits], key=lambda g: g['center'])

        # Build x-regions from selected gaps
        regions = []
        prev_end = x_min
        for gap in selected:
            if gap['start'] > prev_end:
                regions.append((prev_end, gap['start']))
            prev_end = gap['end']
        if x_max > prev_end:
            regions.append((prev_end, x_max))

        regions = [(s, e) for s, e in regions if e - s >= 5]

        if not regions:
            return [], 'failed', {}

        # Crop each region vertically to tight ink bbox
        word_data = []
        for start_x, end_x in regions:
            col_slice = gray[:, start_x:end_x]
            _, col_binary = cv2.threshold(col_slice, 0, 255,
                                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            row_proj = np.sum(col_binary > 0, axis=1)
            non_zero_rows = np.where(row_proj > 0)[0]

            if len(non_zero_rows) == 0:
                continue

            y_min = max(0, non_zero_rows[0] - 2)
            y_max = min(h, non_zero_rows[-1] + 3)
            x_min_c = max(0, start_x - 2)
            x_max_c = min(w, end_x + 2)

            bbox = (x_min_c, y_min, x_max_c - x_min_c, y_max - y_min)
            word_img = img[y_min:y_max, x_min_c:x_max_c]

            if word_img.size == 0:
                continue

            word_data.append((bbox, word_img))

        if not word_data:
            return [], 'failed', {}

        detected_count = len(word_data)

        if self.use_trocr:
            recognized = []
            for bbox, word_img in word_data:
                text = self.recognize_word(word_img)
                recognized.append((bbox, text))
            aligned = self._align_with_ground_truth(recognized, gt_words)
        else:
            aligned = self._align_simple(word_data, gt_words)

        metadata = {
            'method': 'cc_topk',
            'detected_count': detected_count,
            'target_count': target_count,
            'count_match': detected_count == target_count,
            'total_gaps': len(gaps),
            'merge_dist': self.merge_dist,
        }

        return aligned, 'cc_topk', metadata

    def _align_with_ground_truth(self, recognized, gt_words):
        if not recognized:
            return []

        aligned = []

        if len(recognized) == len(gt_words):
            for (bbox, trocr_text), gt_word in zip(recognized, gt_words):
                conf = self.text_similarity(trocr_text, gt_word)
                aligned.append((bbox, gt_word, trocr_text, conf))

        elif len(recognized) < len(gt_words):
            total_width = sum(bbox[2] for bbox, _ in recognized)
            gt_idx = 0
            for bbox, trocr_text in recognized:
                region_frac = bbox[2] / total_width
                n_words = max(1, round(region_frac * len(gt_words)))
                n_words = min(n_words, len(gt_words) - gt_idx)

                assigned = ' '.join(gt_words[gt_idx:gt_idx + n_words])
                conf = self.text_similarity(trocr_text, assigned)
                aligned.append((bbox, assigned, trocr_text, conf))
                gt_idx += n_words

            if gt_idx < len(gt_words) and aligned:
                last = aligned[-1]
                extra = ' '.join(gt_words[gt_idx:])
                new_assigned = last[1] + ' ' + extra
                aligned[-1] = (last[0], new_assigned, last[2],
                               self.text_similarity(last[2], new_assigned))

        else:
            boxes_per_word = len(recognized) / len(gt_words)
            for i, gt_word in enumerate(gt_words):
                start_idx = int(i * boxes_per_word)
                end_idx = int((i + 1) * boxes_per_word)
                end_idx = min(end_idx, len(recognized))

                if start_idx >= len(recognized):
                    break

                boxes = [recognized[j][0] for j in range(start_idx, end_idx)]
                trocr_texts = [recognized[j][1] for j in range(start_idx, end_idx)]

                if not boxes:
                    continue

                x_min = min(b[0] for b in boxes)
                y_min = min(b[1] for b in boxes)
                x_max = max(b[0] + b[2] for b in boxes)
                y_max = max(b[1] + b[3] for b in boxes)
                merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

                merged_trocr = ' '.join(trocr_texts)
                conf = self.text_similarity(merged_trocr, gt_word)
                aligned.append((merged_bbox, gt_word, merged_trocr, conf))

        return aligned

    def _align_simple(self, word_data, gt_words):
        aligned = []
        n_det = len(word_data)
        n_gt = len(gt_words)

        if n_det == n_gt:
            for (bbox, _), gt_word in zip(word_data, gt_words):
                aligned.append((bbox, gt_word, '', 0.0))
        elif n_det < n_gt:
            total_w = sum(b[2] for b, _ in word_data)
            gt_idx = 0
            for bbox, _ in word_data:
                frac = bbox[2] / max(total_w, 1)
                nw = max(1, round(frac * n_gt))
                nw = min(nw, n_gt - gt_idx)
                assigned = ' '.join(gt_words[gt_idx:gt_idx + nw])
                aligned.append((bbox, assigned, '', 0.0))
                gt_idx += nw
            if gt_idx < n_gt and aligned:
                last = aligned[-1]
                extra = ' '.join(gt_words[gt_idx:])
                aligned[-1] = (last[0], last[1] + ' ' + extra, '', 0.0)
        else:
            bpw = n_det / n_gt
            for i, gt_word in enumerate(gt_words):
                si = int(i * bpw)
                ei = min(int((i + 1) * bpw), n_det)
                if si >= n_det:
                    break
                boxes = [word_data[j][0] for j in range(si, ei)]
                if boxes:
                    x_min = min(b[0] for b in boxes)
                    y_min = min(b[1] for b in boxes)
                    x_max = max(b[0] + b[2] for b in boxes)
                    y_max = max(b[1] + b[3] for b in boxes)
                    merged = (x_min, y_min, x_max - x_min, y_max - y_min)
                    aligned.append((merged, gt_word, '', 0.0))
        return aligned


def process_dataset(segmenter, input_path, output_path, max_lines=None):
    metafile_path = os.path.join(input_path, 'METAFILE.tsv')
    df = pd.read_csv(metafile_path, sep='\t', quoting=3)

    if max_lines:
        df = df.head(max_lines)

    # Filter corrupted entries (multi-line transcriptions from quoted fields)
    df = df[df['transcription'].str.split().str.len() <= 15].copy()

    os.makedirs(os.path.join(output_path, 'words', 'words'), exist_ok=True)

    results = []
    metadata_lines = {}
    failed_count = 0
    match_counts = {0: 0, 1: 0, 2: 0, '3+': 0}

    print(f"\nProcessing {len(df)} lines with CC segmentation...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        line_filename = row['filename']
        ground_truth = row['transcription']

        img_path = os.path.join(input_path, 'lines', 'lines', line_filename)

        if not os.path.exists(img_path):
            failed_count += 1
            continue

        words, method, line_meta = segmenter.segment_line(img_path, ground_truth)

        if method == 'failed' or not words:
            failed_count += 1
            metadata_lines[line_filename] = {
                'method': 'failed',
                'detected_count': 0,
                'target_count': len(ground_truth.split())
            }
            continue

        diff = abs(line_meta['detected_count'] - line_meta['target_count'])
        if diff == 0:
            match_counts[0] += 1
        elif diff == 1:
            match_counts[1] += 1
        elif diff == 2:
            match_counts[2] += 1
        else:
            match_counts['3+'] += 1

        img = cv2.imread(img_path)
        stem = Path(line_filename).stem

        for word_idx, (bbox, gt_word, trocr_text, conf) in enumerate(words):
            x, y, w, h = bbox

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img.shape[1], x + w)
            y2 = min(img.shape[0], y + h)

            word_img = img[y1:y2, x1:x2]
            if word_img.size == 0:
                continue

            word_filename = f"{stem}-w{word_idx:02d}.png"
            word_path = os.path.join(output_path, 'words', 'words', word_filename)
            cv2.imwrite(word_path, word_img)

            results.append({
                'filename': word_filename,
                'transcription': gt_word,
                'trocr_text': trocr_text,
                'confidence': round(conf, 3),
                'line_source': line_filename,
                'word_index': word_idx,
                'bbox': f"{x},{y},{w},{h}"
            })

        metadata_lines[line_filename] = line_meta

    print("\nSaving METAFILE")
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(output_path, 'METAFILE.tsv'), sep='\t', index=False
    )

    metadata = {
        'total_lines': len(df),
        'total_words': len(results),
        'failed_lines': failed_count,
        'match_counts': {str(k): v for k, v in match_counts.items()},
        'lines': {k: {'detected_count': v.get('detected_count', 0),
                       'target_count': v.get('target_count', 0)}
                  for k, v in metadata_lines.items()},
    }

    with open(os.path.join(output_path, 'segmentation_metadata.json'), 'w',
              encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    total_processed = len(df) - failed_count
    print(f"Total lines: {len(df):,}")
    print(f"Total words: {len(results):,}")
    print(f"Failed lines: {failed_count} ({failed_count/len(df)*100:.1f}%)")
    print(f"Avg words/line: {len(results)/max(total_processed, 1):.2f}")
    print(f"\nWord count accuracy:")
    print(f"  Perfect match: {match_counts[0]:4d} ({match_counts[0]/max(total_processed,1)*100:.1f}%)")
    print(f"  Off by 1:      {match_counts[1]:4d} ({match_counts[1]/max(total_processed,1)*100:.1f}%)")
    print(f"  Off by 2:      {match_counts[2]:4d} ({match_counts[2]/max(total_processed,1)*100:.1f}%)")
    print(f"  Off by 3+:     {match_counts['3+']:4d} ({match_counts['3+']/max(total_processed,1)*100:.1f}%)")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description='CC-based word segmentation for Ukrainian handwriting'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input line-level dataset path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output word-level dataset path')
    parser.add_argument('--max-lines', type=int, default=None,
                        help='Maximum lines to process (for testing)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (cuda:0, cpu)')
    parser.add_argument('--no-trocr', action='store_true',
                        help='Skip TrOCR validation (much faster)')
    parser.add_argument('--merge-dist', type=int, default=8,
                        help='Max pixel distance to merge nearby components (default: 8)')

    args = parser.parse_args()

    segmenter = CCSegmenter(device=args.device, use_trocr=not args.no_trocr,
                            merge_dist=args.merge_dist)
    process_dataset(segmenter, args.input, args.output, max_lines=args.max_lines)


if __name__ == '__main__':
    main()
