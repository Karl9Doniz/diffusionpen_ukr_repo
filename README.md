# Ukrainian Handwritten Text Generation

A latent diffusion model that generates Ukrainian handwritten text conditioned on writer style and text content. Given a few reference images from a writer, it synthesizes any Ukrainian word in that person's handwriting — then assembles words into full sentence strips.

Trained on a Ukrainian handwriting corpus of ~37K handwritten lines from 308 writers. The style encoder is trained from scratch on Cyrillic handwriting using metric learning.


Model weights can be found by this links: https://drive.google.com/file/d/1Tt54-LCsak7sp-qInckbXj9hTtQjT_XC/view?usp=share_link.

For the cleaned version of dataset used in the most recent v10 training: https://drive.google.com/file/d/1Tt54-LCsak7sp-qInckbXj9hTtQjT_XC/view?usp=share_link

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Repository structure](#repository-structure)
3. [Setup](#setup)
4. [Download weights](#download-weights)
5. [Inference — word generation](#inference--word-generation)
6. [Inference — sentence generation](#inference--sentence-generation)
7. [Training from scratch](#training-from-scratch)
8. [Dataset pipeline](#dataset-pipeline)
9. [Evaluation](#evaluation)
10. [Citation](#citation)

---

## What it does

Given 5 reference handwriting samples from a writer, the model generates any Ukrainian word in that writer's style. Generation is conditioned on:

- **Text content** — via a character-level CANINE encoder
- **Writer style** — via a MobileNetV2 style encoder trained with metric learning on Cyrillic handwriting

Words can be assembled into sentence strips with baseline alignment, geometric normalization, and real handwritten punctuation sampled from a curated punctuation bank.

---

## Repository structure

```
├── train.py                        # Main training script
├── generate_all_styles.py          # Generate a word in every writer's style
├── generate_sentence.py            # Assemble words into sentence strips
├── unet.py                         # UNet diffusion backbone
├── feature_extractor.py            # Style encoder wrapper
├── style_encoder_train_cyrillic.py # Style encoder training for Cyrillic
├── utils/
│   ├── ukr_dataset.py              # Ukrainian word-level dataset loader
│   ├── word_dataset.py             # Cached dataset wrapper
│   └── word_cleanup_nafnet.py      # NAFNet-based word image denoising
├── scripts/
│   ├── segment_ukr_projection.py   # CC-based word segmentation from lines
│   ├── clean_word_dataset.py       # TrOCR filtering + writer balancing
│   ├── balance_rare_letters.py     # Rare Cyrillic letter oversampling
│   ├── build_punct_bank.py         # Extract punctuation crops from dataset
│   ├── clean_punct_bank.py         # Automatic punctuation bank QC
│   ├── evaluate_generated_word_cer.py  # CER metric on generated words
│   ├── evaluate_generated_word_fid.py  # FID metric on generated words
│   └── run_sentence_sweep.py       # Batch sentence generation script
├── generated/
│   └── punct_bank/                 # Handwritten punctuation image bank
│       ├── comma/ period/ colon/
│       ├── hyphen/ dash/ semicolon/
│       └── question/ exclaim/
├── style_models/
│   └── ukr_mixed_wt0p7/
│       └── mixed_ukr_mobilenetv2_100.pth   # Cyrillic style encoder weights
├── stable-diffusion-v1-5/          # SD v1-5 VAE + scheduler (see Setup)
│   ├── vae/
│   └── scheduler/
└── output/
    └── <run_name>/
        └── models/
            └── ema_ckpt.pt         # Main model checkpoint
```

---

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url> ukr-htg
cd ukr-htg
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `diffusers`, `transformers`, `accelerate`, `opencv-python`, `Pillow`, `numpy`, `scipy`, `pandas`, `tqdm`, `matplotlib`, `einops`, `wandb`.

Tested on Python 3.10+, PyTorch 2.x, CUDA 11.8/12.1.

### 2. Download the VAE and scheduler

The model uses **only the VAE encoder/decoder and DDIM scheduler** from Stable Diffusion v1-5 — no text encoder or SD UNet.

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

Download `ukr_htg_weights.zip` and extract into the repo root:

```bash
unzip ukr_htg_weights.zip
```

This creates:
```
checkpoint/ema_ckpt.pt                              # 709 MB — main diffusion model
style_models/ukr_mixed_wt0p7/
    mixed_ukr_mobilenetv2_100.pth                   # 11 MB  — Cyrillic style encoder
stable-diffusion-v1-5/vae/ + scheduler/             # 958 MB — VAE
generated/punct_bank/                               # 7 MB   — punctuation bank
```

Move the checkpoint to the expected path:
```bash
mkdir -p output/ukr_htg_v10/models
mv checkpoint/ema_ckpt.pt output/ukr_htg_v10/models/ema_ckpt.pt
```

---

## Inference — word generation

### Generate a word in all writer styles

```bash
python generate_all_styles.py \
  --checkpoint output/ukr_htg_v10/models/ema_ckpt.pt \
  --style_path style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth \
  --stable_dif_path stable-diffusion-v1-5 \
  --dataset_root /path/to/UkrHandwritten_Words \
  --meta_file /path/to/METAFILE.tsv \
  --word "Україна" \
  --cfg_scale 5.0 \
  --output_dir generated/all_styles_Україна
```

Produces one PNG per writer plus a combined contact sheet.

### Generate a word for specific writers

```bash
python generate_sentence.py \
  --checkpoint output/ukr_htg_v10/models/ema_ckpt.pt \
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
| `--cfg_scale` | `5.0` | Classifier-free guidance scale. 3–5 is the practical range |
| `--num_res_blocks` | `2` | Must match training; v10 checkpoint uses 2 |
| `--seed` | any int | Set for reproducible outputs |
| `--canvas_height` | `104` | Sentence canvas height in px |

---

## Inference — sentence generation

`generate_sentence.py` generates each word individually, then assembles them into a sentence strip with:

- Ink geometry measurement and scaling to a common body height
- Baseline alignment across all words
- Real handwritten punctuation from the punct bank, style-matched per writer
- Optional NAFNet denoising pass

### Single sentence, multiple writers

```bash
python generate_sentence.py \
  --checkpoint output/ukr_htg_v10/models/ema_ckpt.pt \
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

### Batch sweep across multiple sentences

Edit the config at the top of `scripts/run_sentence_sweep.py` (writers, texts, paths), then:

```bash
python scripts/run_sentence_sweep.py
```

Runs all sentences × all writers and saves a `contact_sheet.png` per sentence (all writers stacked vertically).

### Punctuation handling

- Commas, periods, colons, semicolons, question and exclamation marks are sampled from `generated/punct_bank/`
- Dashes (standalone ` - `) come from `dash/` (longer marks); hyphens inside compound words from `hyphen/`
- **Style-matched sampling:** writer-exact marks are preferred; if unavailable, marks from the K nearest writers by stroke weight; otherwise a size-filtered global sample
- Synthetic fallback if no bank is configured

---

## Training from scratch

### Step 1 — Prepare the dataset

Start from raw handwritten line images:

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
  --save_path output/ukr_htg_new \
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
| `--img_height` | `64` | Word-level height |
| `--num_res_blocks` | `2` | Match this at inference |
| `--text_drop_prob` | `0.2` | Enables classifier-free guidance; without it CFG has no effect |
| `--cfg_scale` | `3.0` at train, `5.0` at inference | 5.0 is the empirically best inference scale |
| `--batch_size` | `8` on 11 GB VRAM, `24` on 3× 11 GB | Adjust to GPU memory |
| `--save_every` | `20` | Keep ≤20 to avoid losing work on crash |

## Citation

The diffusion architecture builds on DiffusionPen (Nikolaidou et al., ECCV 2024):

```bibtex
@inproceedings{nikolaidou2024diffusionpen,
  title     = {DiffusionPen: Towards Controlling the Style of Handwritten Text Generation},
  author    = {Nikolaidou, Konstantina and Retsinas, George and Sfikas, Giorgos and Liwicki, Marcus},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```
