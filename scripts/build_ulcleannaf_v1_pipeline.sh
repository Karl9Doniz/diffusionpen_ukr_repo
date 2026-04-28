#!/usr/bin/env bash
set -euo pipefail

# Full rebuild pipeline:
# 1) NAFNet line cleaning (pre-segmentation snapshot)
# 2) CC segmentation
# 3) Extended word cleaning
# 4) Rare-letter balancing (final v9 metafile)

INPUT_LINES_ROOT="/extra_space2/oles_new/UkrHandwritten"
CLEAN_LINES_ROOT="/extra_space2/oles_new/UkrHandwritten_ULCleanNAF_v1"
CLEAN_WORDS_ROOT="/extra_space2/oles_new/UkrHandwritten_Words_CC_ULCleanNAF_v1"
PYTHON_BIN="${PYTHON_BIN:-/home/oles/DiffusionPen/.venv/bin/python}"

CHECKPOINT="/home/oles/DiffusionPen/output/lines204_nafnet_v1/checkpoint_best.pt"
DEVICE="cuda:0"
MERGE_DIST="8"
MIN_SIMILARITY="0.4"
MIN_WRITER_SAMPLES="50"
MAX_FACTOR="5"
LIMIT_LINES=""
SAVE_DEBUG="0"
DEBUG_SAMPLES="100"
SKIP_LINE_CLEAN="0"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --input-lines-root PATH       (default: ${INPUT_LINES_ROOT})
  --clean-lines-root PATH       (default: ${CLEAN_LINES_ROOT})
  --clean-words-root PATH       (default: ${CLEAN_WORDS_ROOT})
  --python-bin PATH             (default: ${PYTHON_BIN})
  --checkpoint PATH             (default: ${CHECKPOINT})
  --device DEV                  (default: ${DEVICE})
  --merge-dist N                (default: ${MERGE_DIST})
  --min-similarity F            (default: ${MIN_SIMILARITY})
  --min-writer-samples N        (default: ${MIN_WRITER_SAMPLES})
  --max-factor N                (default: ${MAX_FACTOR})
  --limit-lines N               Optional limit for smoke runs
  --save-debug                  Save debug panels in line cleaner
  --debug-samples N             (default: ${DEBUG_SAMPLES})
  --skip-line-clean             Reuse existing clean-lines root
  -h, --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-lines-root) INPUT_LINES_ROOT="$2"; shift 2 ;;
    --clean-lines-root) CLEAN_LINES_ROOT="$2"; shift 2 ;;
    --clean-words-root) CLEAN_WORDS_ROOT="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --merge-dist) MERGE_DIST="$2"; shift 2 ;;
    --min-similarity) MIN_SIMILARITY="$2"; shift 2 ;;
    --min-writer-samples) MIN_WRITER_SAMPLES="$2"; shift 2 ;;
    --max-factor) MAX_FACTOR="$2"; shift 2 ;;
    --limit-lines) LIMIT_LINES="$2"; shift 2 ;;
    --save-debug) SAVE_DEBUG="1"; shift 1 ;;
    --debug-samples) DEBUG_SAMPLES="$2"; shift 2 ;;
    --skip-line-clean) SKIP_LINE_CLEAN="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

echo "======================================================================"
echo "NAFNet pre-segmentation rebuild pipeline"
echo "Input lines root:      ${INPUT_LINES_ROOT}"
echo "Clean lines root:      ${CLEAN_LINES_ROOT}"
echo "Clean words root:      ${CLEAN_WORDS_ROOT}"
echo "Python bin:            ${PYTHON_BIN}"
echo "Checkpoint:            ${CHECKPOINT}"
echo "Device:                ${DEVICE}"
echo "Limit lines:           ${LIMIT_LINES:-<none>}"
echo "======================================================================"

if [[ "${SKIP_LINE_CLEAN}" != "1" ]]; then
  CLEAN_CMD=(
    "${PYTHON_BIN}" scripts/clean_lines_nafnet.py
    --input_root "${INPUT_LINES_ROOT}"
    --output_root "${CLEAN_LINES_ROOT}"
    --checkpoint "${CHECKPOINT}"
    --device "${DEVICE}"
  )
  if [[ -n "${LIMIT_LINES}" ]]; then
    CLEAN_CMD+=(--limit "${LIMIT_LINES}")
  fi
  if [[ "${SAVE_DEBUG}" == "1" ]]; then
    CLEAN_CMD+=(--save-debug --debug-samples "${DEBUG_SAMPLES}")
  fi
  echo "[1/4] Cleaning line-level images with NAFNet"
  "${CLEAN_CMD[@]}"
else
  echo "[1/4] Skipped line cleaning (using existing ${CLEAN_LINES_ROOT})"
fi

SEG_CMD=(
  "${PYTHON_BIN}" scripts/segment_ukr_projection.py
  --input "${CLEAN_LINES_ROOT}"
  --output "${CLEAN_WORDS_ROOT}"
  --no-trocr
  --merge-dist "${MERGE_DIST}"
)
if [[ -n "${LIMIT_LINES}" ]]; then
  SEG_CMD+=(--max-lines "${LIMIT_LINES}")
fi
echo "[2/4] Segmenting cleaned lines into words (CC)"
"${SEG_CMD[@]}"

echo "[3/4] Building extended cleaned word metafile"
"${PYTHON_BIN}" scripts/clean_word_dataset.py \
  --input "${CLEAN_WORDS_ROOT}" \
  --output-tsv "${CLEAN_WORDS_ROOT}/METAFILE_extended.tsv" \
  --min-similarity "${MIN_SIMILARITY}" \
  --keep-short \
  --reject-trailing-punct \
  --min-writer-samples "${MIN_WRITER_SAMPLES}" \
  --device "${DEVICE}"

echo "[4/4] Balancing rare letters (final v9 metafile)"
"${PYTHON_BIN}" scripts/balance_rare_letters.py \
  --input "${CLEAN_WORDS_ROOT}/METAFILE_extended.tsv" \
  --output "${CLEAN_WORDS_ROOT}/METAFILE_extended_balanced.tsv" \
  --max-factor "${MAX_FACTOR}"

echo "======================================================================"
echo "Pipeline complete."
echo "Final metafiles:"
echo "  ${CLEAN_WORDS_ROOT}/METAFILE_extended.tsv"
echo "  ${CLEAN_WORDS_ROOT}/METAFILE_extended_balanced.tsv"
echo "======================================================================"
