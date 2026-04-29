import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineTokenizer, CanineModel
from tqdm import tqdm

from unet import UNetModel
from feature_extractor import ImageEncoder
from utils.word_dataset import char_classes as WORD_CHAR_CLASSES

IMG_HEIGHT = 64
IMG_WIDTH = 256
TEXT_MAX_LEN = 40

TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def strip_dp_prefix(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "", 1)
        if new_key.startswith("text_encoder.module."):
            new_key = new_key.replace("text_encoder.module.", "text_encoder.", 1)
        new_sd[new_key] = v
    return new_sd


def detect_num_classes(state_dict):
    for key in ["label_emb.weight", "module.label_emb.weight"]:
        if key in state_dict:
            return state_dict[key].shape[0]
    raise KeyError("Cannot find label_emb.weight in checkpoint")



def resize_to_canvas(img_pil, height=IMG_HEIGHT, width=IMG_WIDTH):
    """Resize PIL image to (height, width) with white padding if needed."""
    w, h = img_pil.size
    new_w = int(w * height / h)
    img_pil = img_pil.resize((new_w, height), Image.BILINEAR)
    w, h = img_pil.size
    if w < width:
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        canvas.paste(img_pil, (0, 0))
        return canvas
    elif w > width:
        return img_pil.resize((width, height), Image.BILINEAR)
    return img_pil


def load_images_from_paths(paths, height=IMG_HEIGHT, width=IMG_WIDTH):
    """Load a list of image paths → stacked tensor [N, 3, H, W]."""
    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            img = resize_to_canvas(img, height, width)
            imgs.append(TRANSFORM(img))
        except Exception as e:
            print(f"  Warning: could not load {p}: {e}")
    return torch.stack(imgs) if imgs else None


def get_iam_writer_images(iam_root, writer_id, n=5):
    """Find up to n word image paths for the given IAM writer prefix (e.g. 'a01').

    IAM structure: iam_root/<writer_prefix>/<form_id>/<word>.png
    """
    writer_dir = os.path.join(iam_root, writer_id)
    if not os.path.isdir(writer_dir):
        raise FileNotFoundError(f"IAM writer directory not found: {writer_dir}")

    paths = []
    for root, _, files in os.walk(writer_dir):
        for f in files:
            if f.endswith(".png"):
                paths.append(os.path.join(root, f))

    if not paths:
        raise FileNotFoundError(f"No PNG images found for IAM writer: {writer_id}")

    random.shuffle(paths)
    selected = paths[:n]
    while len(selected) < n:
        selected.append(selected[0])
    return selected


def build_ukr_writer_id_map(meta_file):
    """Build writer_str -> writer_idx mapping (matching UkrWordDataset)."""
    writers = set()
    with open(meta_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 1:
                continue
            parts = row[0].replace(".png", "").split("-")
            if len(parts) >= 3:
                writers.add(parts[2])
    sorted_writers = sorted(writers)
    return {w: i for i, w in enumerate(sorted_writers)}


def get_ukr_writer_image_paths(dataset_root, meta_file, writer_str, n=5):
    """Get up to n image paths for a Ukrainian writer string ID."""
    words_dir = os.path.join(dataset_root, "words", "words")
    paths = []
    with open(meta_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 1:
                continue
            fname = row[0]
            parts = fname.replace(".png", "").split("-")
            if len(parts) >= 3 and parts[2] == writer_str:
                full = os.path.join(words_dir, fname)
                if os.path.exists(full):
                    paths.append(full)
            if len(paths) >= n:
                break
    while len(paths) < n and paths:
        paths.append(paths[0])
    return paths


@torch.no_grad()
def compute_style_embedding(image_paths, style_extractor, device,
                            height=IMG_HEIGHT, width=IMG_WIDTH):
    """Load images, pass through style encoder, return mean embedding [D]."""
    imgs = load_images_from_paths(image_paths, height, width)
    if imgs is None:
        return None
    imgs = imgs.to(device)                          # [N, 3, H, W]
    features = style_extractor(imgs)                # [N, D]
    return features.mean(dim=0)                     # [D]


@torch.no_grad()
def build_ukr_embedding_index(dataset_root, meta_file, writer_id_map,
                               style_extractor, device,
                               height=IMG_HEIGHT, width=IMG_WIDTH):
    """Compute mean style embedding for every Ukrainian writer.

    Returns:
        dict: writer_str -> {"idx": int, "embedding": tensor [D]}
    """
    print("Building Ukrainian style embedding index...")
    index = {}
    idx_to_str = {v: k for k, v in writer_id_map.items()}

    for writer_str, writer_idx in tqdm(writer_id_map.items(), desc="Encoding UKR writers"):
        paths = get_ukr_writer_image_paths(dataset_root, meta_file, writer_str, n=5)
        if not paths:
            continue
        emb = compute_style_embedding(paths, style_extractor, device, height, width)
        if emb is not None:
            index[writer_str] = {"idx": writer_idx, "embedding": emb}

    print(f"Index built: {len(index)} writers.")
    return index


def find_nearest_ukr_writer(iam_embedding, ukr_index, top_k=3):
    """Find top-k Ukrainian writers by cosine similarity to the IAM embedding.

    Returns list of (writer_str, writer_idx, cosine_sim) sorted descending.
    """
    writer_strs = list(ukr_index.keys())
    embeddings = torch.stack([ukr_index[w]["embedding"] for w in writer_strs])  # [N, D]

    iam_norm = F.normalize(iam_embedding.unsqueeze(0), dim=1)  # [1, D]
    ukr_norm = F.normalize(embeddings, dim=1)                  # [N, D]
    sims = (iam_norm @ ukr_norm.T).squeeze(0)                  # [N]

    top_values, top_indices = torch.topk(sims, k=min(top_k, len(writer_strs)))

    results = []
    for sim_val, i in zip(top_values.tolist(), top_indices.tolist()):
        wstr = writer_strs[i]
        results.append((wstr, ukr_index[wstr]["idx"], sim_val))
    return results


@torch.no_grad()
def generate_word(word, unet, vae, style_extractor, tokenizer,
                  noise_scheduler, style_ref_tensor, writer_idx, device,
                  cfg_scale=5.0, height=IMG_HEIGHT, width=IMG_WIDTH,
                  text_max_len=TEXT_MAX_LEN):
    """Generate one word. style_ref_tensor: [5, 3, H, W]. Returns grayscale PIL."""
    unet.eval()

    text_tokens = tokenizer(
        [word], padding="max_length", truncation=True,
        return_tensors="pt", max_length=text_max_len,
    ).to(device)

    null_tokens = tokenizer(
        [""], padding="max_length", truncation=True,
        return_tensors="pt", max_length=text_max_len,
    ).to(device) if cfg_scale > 1.0 else None

    style_batch = style_ref_tensor.unsqueeze(0).to(device)   # [1, 5, 3, H, W]
    style_flat  = style_batch.reshape(-1, 3, height, width)  # [5, 3, H, W]
    style_feats = style_extractor(style_flat).to(device)     # [5, D]
    labels = torch.tensor([writer_idx], dtype=torch.long).to(device)

    x = torch.randn(1, 4, height // 8, width // 8, device=device)

    noise_scheduler.set_timesteps(50)
    for t_step in noise_scheduler.timesteps:
        t = (torch.ones(1, device=device) * t_step.item()).long()
        noise_pred = unet(x, t, text_tokens, labels,
                          original_images=style_batch,
                          mix_rate=None,
                          style_extractor=style_feats)
        if null_tokens is not None:
            noise_uncond = unet(x, t, null_tokens, labels,
                                original_images=style_batch,
                                mix_rate=None,
                                style_extractor=style_feats)
            noise_pred = noise_uncond + cfg_scale * (noise_pred - noise_uncond)
        x = noise_scheduler.step(noise_pred, t_step, x).prev_sample

    latents = x / 0.18215
    decoded = vae.decode(latents).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    return torchvision.transforms.ToPILImage()(decoded[0]).convert("L")


def crop_whitespace_h(img_pil):
    """Crop horizontal whitespace only."""
    arr = np.array(img_pil)
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return arr
    x, y, w, h = cv2.boundingRect(coords)
    return arr[:, x:x+w]


def stitch_sentence(word_images, words, gap=16, canvas_height=88, gen_height=64):
    """Stitch word images into a single horizontal strip."""
    if not word_images:
        return Image.new("L", (256, canvas_height), 255)

    longest_idx = max(range(len(words)), key=lambda i: len(words[i]))
    avg_char_w = Image.fromarray(word_images[longest_idx]).width / max(len(words[longest_idx]), 1)
    pad = canvas_height - gen_height

    parts = [np.ones((canvas_height, gap), dtype=np.uint8) * 255]
    for word, arr in zip(words, word_images):
        scaled_w = max(int(avg_char_w * len(word)), int(avg_char_w * 2))
        resized = np.array(Image.fromarray(arr).resize((scaled_w, gen_height)))
        padded = np.pad(resized, ((0, pad), (0, 0)), constant_values=255)
        parts.append(padded)
        parts.append(np.ones((canvas_height, gap), dtype=np.uint8) * 255)

    return Image.fromarray(np.concatenate(parts, axis=1))


def make_strip(image_paths, strip_h=64, max_w=600):
    """Make a horizontal strip from a list of image paths (for reference panel)."""
    imgs = []
    for p in image_paths[:5]:
        try:
            img = Image.open(p).convert("L")
            w, h = img.size
            new_w = int(w * strip_h / h)
            img = img.resize((new_w, strip_h), Image.BILINEAR)
            imgs.append(img)
        except Exception:
            pass

    if not imgs:
        return Image.new("L", (max_w, strip_h), 255)

    # Fit within max_w
    total_w = sum(im.width for im in imgs) + 8 * len(imgs)
    if total_w > max_w:
        scale = max_w / total_w
        imgs = [im.resize((max(1, int(im.width * scale)), strip_h), Image.BILINEAR)
                for im in imgs]

    canvas = Image.new("L", (max_w, strip_h), 255)
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + 8
    return canvas


def make_comparison_figure(iam_writer, iam_paths,
                           ukr_writer_str, ukr_paths, cosine_sim,
                           generated_sentence,
                           panel_w=900, ref_h=64):
    """Build a labelled 3-row figure:
        Row 1: IAM reference images (English)
        Row 2: Generated Ukrainian sentence
        Row 3: Matched Ukrainian reference images
    """
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 14)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    label_h = 20
    margin = 8

    iam_strip  = make_strip(iam_paths,  ref_h, panel_w)
    ukr_strip  = make_strip(ukr_paths,  ref_h, panel_w)

    gen_w, gen_h = generated_sentence.size
    scale = panel_w / gen_w
    gen_resized = generated_sentence.resize(
        (panel_w, max(1, int(gen_h * scale))), Image.BILINEAR)
    gen_h_scaled = gen_resized.height

    total_h = (label_h + ref_h + margin) * 2 + label_h + gen_h_scaled + margin * 2
    fig = Image.new("L", (panel_w, total_h), 255)
    draw = ImageDraw.Draw(fig)

    y = margin
    draw.text((4, y), f"IAM writer: {iam_writer}  (English reference images)", fill=0, font=font)
    y += label_h
    fig.paste(iam_strip, (0, y))
    y += ref_h + margin

    draw.text((4, y), f"Generated Ukrainian text  (cfg={args_cfg_scale:.1f})", fill=0, font=font)
    y += label_h
    fig.paste(gen_resized, (0, y))
    y += gen_h_scaled + margin

    draw.text((4, y),
              f"Nearest Ukrainian writer: {ukr_writer_str}  "
              f"(cosine sim = {cosine_sim:.4f})",
              fill=0, font=font)
    y += label_h
    fig.paste(ukr_strip, (0, y))

    return fig


# Shared cfg_scale for figure label (set in main)
args_cfg_scale = 5.0


def plot_tsne(ukr_index, iam_embeddings, output_path, perplexity=30, seed=42):
    """Plot t-SNE of Ukrainian writer embeddings + IAM writer embeddings.

    Ukrainian writers: small grey dots.
    IAM writers: larger colored markers, labeled, connected to their nearest match.

    Args:
        ukr_index: dict writer_str -> {"idx": int, "embedding": tensor [D]}
        iam_embeddings: dict iam_writer_id -> {"embedding": tensor [D],
                                                "match_str": str,
                                                "match_sim": float}
        output_path: where to save the PNG
        perplexity: t-SNE perplexity
        seed: random seed for reproducibility
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    ukr_strs  = list(ukr_index.keys())
    ukr_embs  = torch.stack([ukr_index[w]["embedding"] for w in ukr_strs]).cpu().numpy()

    iam_ids   = list(iam_embeddings.keys())
    iam_embs  = np.stack([iam_embeddings[k]["embedding"].cpu().numpy() for k in iam_ids])

    all_embs  = np.concatenate([ukr_embs, iam_embs], axis=0)
    n_ukr     = len(ukr_strs)

    print(f"Running t-SNE on {len(all_embs)} points (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                max_iter=1000, init="pca")
    coords = tsne.fit_transform(all_embs)

    ukr_coords = coords[:n_ukr]
    iam_coords = coords[n_ukr:]

    # Build lookup: ukr_str -> 2D position
    ukr_pos = {s: ukr_coords[i] for i, s in enumerate(ukr_strs)}

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(ukr_coords[:, 0], ukr_coords[:, 1],
               c="#aaaaaa", s=18, alpha=0.5, zorder=1, label="Ukrainian writers (n=323)")

    colors = plt.cm.Set1(np.linspace(0, 0.8, len(iam_ids)))
    for i, (iam_id, color) in enumerate(zip(iam_ids, colors)):
        ix, iy = iam_coords[i]
        ax.scatter(ix, iy, color=color, s=120, zorder=4, edgecolors="black", linewidths=0.8)
        ax.annotate(f"IAM {iam_id}", (ix, iy),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color=color, fontweight="bold", zorder=5)

        # Draw line to matched Ukrainian writer
        match_str = iam_embeddings[iam_id].get("match_str")
        if match_str and match_str in ukr_pos:
            mx, my = ukr_pos[match_str]
            ax.plot([ix, mx], [iy, my], color=color, linewidth=1.2,
                    linestyle="--", alpha=0.7, zorder=2)
            ax.scatter(mx, my, color=color, s=60, zorder=3,
                       marker="^", edgecolors="black", linewidths=0.6)
            sim = iam_embeddings[iam_id].get("match_sim", 0)
            ax.annotate(f"{match_str}\n(sim={sim:.2f})", (mx, my),
                        textcoords="offset points", xytext=(4, -14),
                        fontsize=7, color=color, zorder=5)

    ax.set_title("t-SNE: Style Embedding Space\n"
                 "Grey = Ukrainian writers · Coloured = IAM (English) writers · "
                 "Triangle = nearest Ukrainian match",
                 fontsize=11)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"t-SNE saved: {output_path}")


def main():
    global args_cfg_scale

    parser = argparse.ArgumentParser(description="Cross-lingual style transfer experiment")
    parser.add_argument("--checkpoint", type=str,
                        default="output/diffusionpen_ukr_v9/models/ckpt.pt")
    parser.add_argument("--text", type=str,
                        default="Реве та стогне Дніпр широкий")
    parser.add_argument("--iam_root", type=str,
                        default="/extra_space2/oles_new/iam_data/words")
    parser.add_argument("--iam_writers", type=str, nargs="+",
                        default=["a01", "b04", "c01", "d01", "e01",
                                 "a02", "b01", "c04", "d04", "g01"])
    parser.add_argument("--ukr_dataset_root", type=str,
                        default="/extra_space2/oles_new/UkrHandwritten_Words_CC")
    parser.add_argument("--ukr_meta_file", type=str,
                        default="/extra_space2/oles_new/UkrHandwritten_Words_CC/METAFILE_extended_balanced.tsv")
    parser.add_argument("--style_path", type=str,
                        default="style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth")
    parser.add_argument("--stable_dif_path", type=str,
                        default="stable-diffusion-v1-5")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=1,
                        help="Number of nearest Ukrainian writers to show per IAM writer")
    parser.add_argument("--output_dir", type=str, default="generated/cross_lingual")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne_only", action="store_true",
                        help="Only run t-SNE (skip generation, no UNet/VAE needed)")
    parser.add_argument("--tsne_perplexity", type=int, default=30)
    args = parser.parse_args()

    args_cfg_scale = args.cfg_scale
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load style encoder ----
    print("Loading style encoder...")
    style_extractor = ImageEncoder(model_name="mobilenetv2_100", num_classes=0,
                                   pretrained=False, trainable=False)
    style_sd = torch.load(args.style_path, map_location="cpu")
    model_dict = style_extractor.state_dict()
    style_sd = {k: v for k, v in style_sd.items()
                if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(style_sd)
    style_extractor.load_state_dict(model_dict)
    style_extractor = style_extractor.to(device).eval()

    # Build Ukrainian embedding index ----
    writer_id_map = build_ukr_writer_id_map(args.ukr_meta_file)
    ukr_index = build_ukr_embedding_index(
        args.ukr_dataset_root, args.ukr_meta_file,
        writer_id_map, style_extractor, device,
    )

    # Compute IAM embeddings (needed for both t-SNE and generation) ----
    iam_data = {}  # iam_writer_id -> {embedding, match_str, match_sim, paths}
    for iam_writer in args.iam_writers:
        try:
            paths = get_iam_writer_images(args.iam_root, iam_writer, n=5)
        except FileNotFoundError as e:
            print(f"  SKIP {iam_writer}: {e}")
            continue
        emb = compute_style_embedding(paths, style_extractor, device)
        if emb is None:
            continue
        matches = find_nearest_ukr_writer(emb, ukr_index, top_k=args.top_k)
        ukr_str, ukr_idx, sim = matches[0]
        print(f"  {iam_writer} -> UKR {ukr_str} (idx={ukr_idx}, sim={sim:.4f})")
        iam_data[iam_writer] = {
            "embedding": emb,
            "match_str": ukr_str,
            "match_idx": ukr_idx,
            "match_sim": sim,
            "all_matches": matches,
            "paths": paths,
        }

    tsne_path = os.path.join(args.output_dir, "tsne_style_space.png")
    plot_tsne(ukr_index, iam_data, tsne_path, perplexity=args.tsne_perplexity, seed=args.seed)

    if args.tsne_only:
        print("--tsne_only: stopping after t-SNE.")
        return

    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    state_dict = strip_dp_prefix(state_dict)
    num_classes = detect_num_classes(state_dict)
    print(f"  num_classes = {num_classes}")

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    canine    = CanineModel.from_pretrained("google/canine-c")

    from types import SimpleNamespace
    fake_args = SimpleNamespace(interpolation=False, mix_rate=None)

    unet = UNetModel(
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        in_channels=4,
        model_channels=320,
        out_channels=4,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=4,
        num_classes=num_classes,
        context_dim=320,
        vocab_size=WORD_CHAR_CLASSES,
        text_encoder=canine,
        args=fake_args,
    )
    unet.load_state_dict(state_dict)
    unet = unet.to(device).eval()

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(device)
    vae.requires_grad_(False)

    noise_scheduler = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    words = args.text.strip().split()

    summary_rows = []

    for iam_writer, data in iam_data.items():
        print(f"\n{'='*60}")
        print(f"IAM writer: {iam_writer}")

        iam_paths = data["paths"]
        ukr_str   = data["match_str"]
        ukr_idx   = data["match_idx"]
        sim       = data["match_sim"]
        for rank, (ws, wi, s) in enumerate(data["all_matches"], 1):
            print(f"  Match #{rank}: UKR {ws} (idx={wi}), cosine sim={s:.4f}")
        summary_rows.append((iam_writer, ukr_str, ukr_idx, sim))

        # Load IAM images as style_ref tensor for generation
        iam_tensor = load_images_from_paths(iam_paths)
        if iam_tensor is None or len(iam_tensor) < 5:
            print(f"  SKIP: failed to load enough IAM images")
            continue
        iam_tensor = iam_tensor[:5]  # exactly 5

        # Generate each word
        word_images = []
        for word in tqdm(words, desc=f"Generating ({iam_writer})"):
            img_pil = generate_word(
                word, unet, vae, style_extractor, tokenizer,
                noise_scheduler, iam_tensor, ukr_idx, device,
                cfg_scale=args.cfg_scale,
            )
            img_arr = crop_whitespace_h(img_pil)
            word_images.append(img_arr)

        sentence_img = stitch_sentence(word_images, words)

        # Load Ukrainian reference images for figure
        ukr_paths = get_ukr_writer_image_paths(
            args.ukr_dataset_root, args.ukr_meta_file, ukr_str, n=5)

        # Build comparison figure
        fig = make_comparison_figure(
            iam_writer, iam_paths,
            ukr_str, ukr_paths, sim,
            sentence_img,
        )
        out_path = os.path.join(args.output_dir, f"iam_{iam_writer}_ukr_{ukr_str}.png")
        fig.save(out_path)
        print(f"  Saved: {out_path}")

        # Also save the raw generated sentence separately
        sentence_path = os.path.join(args.output_dir, f"sentence_iam_{iam_writer}.png")
        sentence_img.save(sentence_path)

    report_path = os.path.join(args.output_dir, "similarity_report.txt")
    with open(report_path, "w") as f:
        f.write("IAM writer  -> UKR writer  (idx)   cosine sim\n")
        f.write("-" * 50 + "\n")
        for iam_w, ukr_w, ukr_i, sim in summary_rows:
            f.write(f"{iam_w:<12} -> {ukr_w:<12} ({ukr_i:>3})   {sim:.4f}\n")
    print(f"\nSimilarity report saved: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
