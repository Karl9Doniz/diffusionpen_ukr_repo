# DiffusionPen — Ukrainian Handwritten Text Generation

Adaptation of [DiffusionPen](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11492_ECCV_2024_paper.php) (Nikolaidou et al., ECCV 2024) for **Ukrainian handwritten text generation** at the word level.

The original model was trained on the English IAM dataset. This repository replaces the dataset with a Ukrainian handwriting corpus (~37K handwritten lines, 308 writers), retrains the style encoder from scratch for Cyrillic, and fine-tunes the full diffusion pipeline.

---

## Table of Contents

1. [What this model does](#what-this-model-does)
2. [Repository structure](#repository-structure)
3. [Setup](#setup)
4. [Download weights](#download-weights)
5. [Inference — word generation](#inference--word-generation)
6. [Inference — sentence generation](#inference--sentence-generation)
7. [Training from scratch](#training-from-scratch)
8. [Dataset pipeline](#dataset-pipeline)
9. [Evaluation](#evaluation)
10. [Citation](#citation)


## What this model does

Given a few reference images of a writer's handwriting (5 samples), the model generates any Ukrainian word in that writer's style. The diffusion process is conditioned on:

- **Text content** — via a character-level CANINE encoder
- **Writer style** — via a MobileNetV2-based style encoder trained with metric learning on Cyrillic handwriting

Words can be assembled into sentence strips with baseline alignment, geometric normalization, and real handwritten punctuation sampled from a punctuation bank.


## Setup

### 1. Clone and create environment

```bash
git clone <repo-url> DiffusionPen
cd DiffusionPen
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `diffusers`, `transformers`, `accelerate`, `opencv-python`, `Pillow`, `numpy`, `scipy`, `pandas`, `tqdm`, `matplotlib`, `einops`, `wandb`.

Tested on Python 3.10+, PyTorch 2.x, CUDA 11.8/12.1.

### 2. Download Stable Diffusion v1-5 VAE and scheduler

DiffusionPen uses **only the VAE encoder/decoder and DDIM scheduler** from SD v1-5. No text encoder or UNet from SD is used.

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'stable-diffusion-v1-5/stable-diffusion-v1-5',
    allow_patterns=['vae/*', 'scheduler/*'],
    local_dir='stable-diffusion-v1-5'
)
"
```

Expected result:
```
stable-diffusion-v1-5/
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors   (~850 MB)
└── scheduler/
    └── scheduler_config.json
```

---

## Download weights

Download `diffusionpen_ukr_weights.zip` and extract it into the repo root:

```bash
unzip diffusionpen_ukr_weights.zip
```

This will create/populate:
```
checkpoint/ema_ckpt.pt                              # 709 MB — main diffusion model
style_models/ukr_mixed_wt0p7/
    mixed_ukr_mobilenetv2_100.pth                   # 11 MB  — Cyrillic style encoder
stable-diffusion-v1-5/vae/ + scheduler/             # 958 MB — SD VAE
generated/punct_bank/                               # 7 MB   — punctuation bank
```

Move the checkpoint to the expected path if needed:
```bash
mkdir -p output/diffusionpen_ukr_v10/models
mv checkpoint/ema_ckpt.pt output/diffusionpen_ukr_v10/models/ema_ckpt.pt
```

---

## Inference — word generation

### Generate a single word in all writer styles

```bash
python generate_all_styles.py \
  --checkpoint output/diffusionpen_ukr_v10/models/ema_ckpt.pt \
  --style_path style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth \
  --stable_dif_path stable-diffusion-v1-5 \
  --dataset_root /path/to/UkrHandwritten_Words \
  --meta_file /path/to/METAFILE.tsv \
  --word "Україна" \
  --cfg_scale 5.0 \
  --output_dir generated/all_styles_Україна
```

This produces one PNG per writer plus a combined contact sheet.

### Generate a word for specific writers

```bash
python generate_sentence.py \
  --checkpoint output/diffusionpen_ukr_v10/models/ema_ckpt.pt \
  --style_path style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth \
  --stable_dif_path stable-diffusion-v1-5 \
  --dataset_root /path/to/UkrHandwritten_Words \
  --meta_file /path/to/METAFILE.tsv \
  --text "слово" \
  --writer 0093 0104 0197 \
  --cfg_scale 5.0 \
  --output_dir generated/word_test
```

### Key inference parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--cfg_scale` | `5.0` | Classifier-free guidance scale. 3.0 = softer, 7.0 = sharper but may over-sharpen |
| `--num_res_blocks` | `2` | Must match training. Default is 2 for v10 checkpoint |
| `--seed` | any int | Set for reproducible outputs |
| `--canvas_height` | `104` | Sentence canvas height in px. Word generation uses 64 internally |

---

## Inference — sentence generation

`generate_sentence.py` generates each word individually via diffusion, then assembles them into a sentence strip with:

- Ink geometry measurement and scaling to a common body height
- Baseline alignment across all words
- Real handwritten punctuation sampled from the punct bank (style-matched by writer)
- Optional NAFNet denoising pass

### Single sentence, multiple writers

```bash
python generate_sentence.py \
  --checkpoint output/diffusionpen_ukr_v10/models/ema_ckpt.pt \
  --style_path style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth \
  --stable_dif_path stable-diffusion-v1-5 \
  --dataset_root /path/to/UkrHandwritten_Words \
  --meta_file /path/to/METAFILE.tsv \
  --text "Слухай голос розуму, а не гніву" \
  --writer 0093 0104 0197 0440 0508 \
  --cfg_scale 5.0 \
  --seed 42 \
  --canvas_height 104 \
  --punct_bank generated/punct_bank \
  --enable_nafnet_cleanup \
  --nafnet_blend 0.85 \
  --output_dir generated/sentence_test
```

Output: one `sentence_writer_{id}.png` per writer in `--output_dir`.

### Batch sweep across multiple sentences (recommended)

Edit the config at the top of `scripts/run_sentence_sweep.py` (writers, texts, paths), then:

```bash
python scripts/run_sentence_sweep.py
```

This runs all sentences × all writers and produces a `contact_sheet.png` per sentence (all writers stacked vertically for easy comparison).

### Punctuation handling

- Commas, periods, colons, semicolons, question and exclamation marks are sampled from the `generated/punct_bank/` image bank
- Dashes (standalone ` - `) are drawn from the `dash/` subdir (longer marks)
- Hyphens inside compound words use the `hyphen/` subdir (shorter marks)
- If no bank is configured, a synthetic dash is drawn as a fallback
- The bank uses **style-matched sampling**: writer-exact marks are preferred; if unavailable, marks from the K nearest writers by stroke weight are used; otherwise a size-filtered global sample

---

## Training from scratch

### Step 1 — Prepare the dataset

Start from raw handwritten line images. See `scripts/build_ulcleannaf_v1_pipeline.sh` for the full pipeline. Summary:

```bash
# 1. Segment lines into words (CC-based, ~5 min for 37K lines)
python scripts/segment_ukr_projection.py \
  --input /path/to/UkrHandwritten \
  --output /path/to/UkrHandwritten_Words \
  --no-trocr --merge-dist 8

# 2. Filter and balance (TrOCR quality filter + writer balance)
python scripts/clean_word_dataset.py \
  --input /path/to/UkrHandwritten_Words \
  --min-similarity 0.4 \
  --keep-short \
  --reject-trailing-punct

# 3. Oversample rare Cyrillic letters
python scripts/balance_rare_letters.py \
  --meta /path/to/METAFILE.tsv \
  --output /path/to/METAFILE_balanced.tsv
```

### Step 2 — Train the style encoder

```bash
python style_encoder_train_cyrillic.py \
  --dataset_root /path/to/UkrHandwritten_Words \
  --meta_file /path/to/METAFILE_balanced.tsv \
  --save_path style_models/ukr_new \
  --epochs 100
```

### Step 3 — Train the diffusion model

```bash
python train.py \
  --dataset ukr \
  --dataset_root /path/to/UkrHandwritten_Words \
  --ukr_meta_file /path/to/METAFILE_balanced.tsv \
  --style_path style_models/ukr_new/model.pth \
  --stable_dif_path stable-diffusion-v1-5 \
  --save_path output/diffusionpen_ukr_new \
  --device cuda:0 \
  --epochs 200 \
  --batch_size 8 \
  --img_height 64 \
  --img_width 256 \
  --text_max_len 40 \
  --num_res_blocks 2 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --lr_schedule cosine \
  --text_drop_prob 0.2 \
  --cfg_scale 3.0 \
  --val_size 512 \
  --sample_every 10 \
  --save_every 20 \
  --save_last True \
  --wandb_log True
```

For multi-GPU training (DataParallel):

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python train.py \
  ... \
  --dataparallel True \
  --batch_size 24
```

### Key training parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--img_height` | `64` | Word-level height. Do not use 1360 (line-level, OOM) |
| `--num_res_blocks` | `2` | Matches original DiffusionPen paper and v10 checkpoint |
| `--text_drop_prob` | `0.2` | Enables classifier-free guidance. Without this, CFG has no effect |
| `--cfg_scale` | `3.0` at train, `5.0` at inference | 5.0 is the empirically best inference scale |
| `--batch_size` | `8` on 11 GB VRAM, `24` on 3× 11 GB | Adjust to GPU memory |
| `--save_every` | `20` | Use ≤20 to avoid losing epochs on crash |

---

## Dataset pipeline

The training dataset is derived from a Ukrainian handwriting corpus:

| Stage | Count |
|-------|-------|
| Source line images | 37,111 |
| Segmented word crops | 155,001 |
| After TrOCR + writer-balance filter | 116,707 |
| After rare-letter oversampling | 126,177 |
| Writers in final training set | 308 |

Segmentation uses connected-component gap detection (OpenCV `connectedComponentsWithStats`) — chosen because CRAFT, Surya OCR, and EasyOCR all fail on cursive Ukrainian handwriting.

---

## Evaluation

### CER (Character Error Rate)

```bash
python scripts/evaluate_generated_word_cer.py \
  --checkpoint output/diffusionpen_ukr_v10/models/ema_ckpt.pt \
  --style_path style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth \
  --stable_dif_path stable-diffusion-v1-5 \
  --dataset_root /path/to/UkrHandwritten_Words \
  --meta_file /path/to/METAFILE.tsv \
  --output_dir generated/cer_eval \
  --n_samples 5000
```

Reported results (v10 checkpoint, 5000 seen-word samples):

| Metric | Value |
|--------|-------|
| Overall CER | 0.1601 |
| Writer-macro CER | 0.1596 |
| CER, length 1–3 | 0.4271 |
| CER, length 4–6 | 0.1084 |
| CER, length 7–9 | 0.1212 |
| CER, length 10+ | 0.1519 |
| Rare-letter CER | 0.1721 |
| OOV CER | 0.1556 |

Note: CER is measured with TrOCR, which is imperfect on Ukrainian handwriting — treat as a relative internal metric rather than an absolute quality measure.

### FID (Fréchet Inception Distance)

```bash
python scripts/evaluate_generated_word_fid.py \
  --checkpoint output/diffusionpen_ukr_v10/models/ema_ckpt.pt \
  --style_path style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth \
  --stable_dif_path stable-diffusion-v1-5 \
  --dataset_root /path/to/UkrHandwritten_Words \
  --meta_file /path/to/METAFILE.tsv \
  --output_dir generated/fid_eval \
  --n_samples 5000
```

Reported result: **FID = 23.09** (5000 samples, 308 writers).

---

## Citation

If you use this work, please cite both the original DiffusionPen paper and this adaptation:

```bibtex
@inproceedings{nikolaidou2024diffusionpen,
  title     = {DiffusionPen: Towards Controlling the Style of Handwritten Text Generation},
  author    = {Nikolaidou, Konstantina and Retsinas, George and Sfikas, Giorgos and Liwicki, Marcus},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```
