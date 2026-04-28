import os
import argparse
import zipfile
import torch
import torchvision
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineTokenizer
from tqdm import tqdm

from unet import UNetModel
from feature_extractor import ImageEncoder
from utils.word_cleanup_nafnet import NAFNetWordCleaner


# ---------------------------------------------------------------------------
# Predefined model configs
# ---------------------------------------------------------------------------
MODEL_PRESETS = {
    "v2": {
        "checkpoint": "output/diffusionpen_ukr_words_v2/models/ema_ckpt.pt",
        "style_path": "style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth",
    },
    "v3": {
        "checkpoint": "pen_checkpoints/diffusionpen_ukr_v3/models/ema_ckpt.pt",
        "style_path": "style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth",
    },
    "v4": {
        "checkpoint": "pen_checkpoints/diffusionpen_ukr_v4/models/ema_ckpt.pt",
        "style_path": "style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth",
    },
    "v5": {
        "checkpoint": "pen_checkpoints/diffusionpen_ukr_v5/models/ema_ckpt.pt",
        "style_path": "style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth",
    },
    "v6": {
        "checkpoint": "pen_checkpoints/diffusionpen_ukr_v6/models/ema_ckpt.pt",
        "style_path": "style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth",
    },
    "local": {
        "checkpoint": "diffusionpen_ukr_model/models/ema_ckpt.pt",
        "style_path": "style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth",
    },
}

RUN_ALL_PRESETS = [
    {"model": "v3", "word": "Олесь", "cfg_scale": 5.0},
    {"model": "v3", "word": "привіт", "cfg_scale": 5.0},
    {"model": "v3", "word": "Київ", "cfg_scale": 5.0},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_dp_prefix(state_dict):
    """Strip DataParallel 'module.' prefix from state dict keys."""
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "", 1)
        # Also handle nested module. in text_encoder
        if new_key.startswith("text_encoder.module."):
            new_key = new_key.replace("text_encoder.module.", "text_encoder.", 1)
        new_sd[new_key] = v
    return new_sd


def detect_num_classes(state_dict):
    """Auto-detect num_classes from label_emb.weight shape."""
    key = "label_emb.weight"
    if key not in state_dict:
        # Try with module. prefix
        key = "module.label_emb.weight"
    if key in state_dict:
        return state_dict[key].shape[0]
    raise KeyError("Cannot find label_emb.weight in checkpoint to detect num_classes")


def load_style_images(dataset_root, meta_file, writer_id_map, img_height=64, img_width=256):
    """Load one representative style image per writer from the dataset.

    Returns dict: writer_idx -> tensor [5, 3, H, W] (5 style reference images)
    """
    import csv
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    words_dir = os.path.join(dataset_root, "words", "words")
    if not os.path.isdir(words_dir):
        raise FileNotFoundError(f"Words directory not found: {words_dir}")

    # Group images by writer
    writer_images = {}  # writer_str -> list of filenames
    with open(meta_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            fname = row[0]
            # Extract writer ID from filename: aXX-YYY-ZZZZ-... -> ZZZZ
            parts = fname.replace(".png", "").split("-")
            if len(parts) >= 3:
                writer_str = parts[2]
            else:
                continue
            if writer_str not in writer_images:
                writer_images[writer_str] = []
            writer_images[writer_str].append(fname)

    style_refs = {}
    for writer_str, writer_idx in writer_id_map.items():
        if writer_str not in writer_images:
            continue
        fnames = writer_images[writer_str]
        # Pick up to 5 images
        selected = fnames[:5]
        while len(selected) < 5:
            selected.append(selected[0])

        imgs = []
        for fn in selected:
            path = os.path.join(words_dir, fn)
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            w, h = img.size
            img = img.resize((int(w * img_height / h), img_height))
            w, h = img.size
            if w < img_width:
                new_img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
                new_img.paste(img, (0, 0))
                img = new_img
            elif w > img_width:
                img = img.resize((img_width, img_height))
            imgs.append(transform(img))

        if len(imgs) == 5:
            style_refs[writer_idx] = torch.stack(imgs)

    return style_refs


def build_writer_id_map(dataset_root, meta_file):
    """Build writer_str -> writer_idx mapping matching how UkrWordDataset does it."""
    import csv
    writers = set()
    with open(meta_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            fname = row[0]
            parts = fname.replace(".png", "").split("-")
            if len(parts) >= 3:
                writers.add(parts[2])
    sorted_writers = sorted(writers)
    return {w: i for i, w in enumerate(sorted_writers)}


@torch.no_grad()
def generate_word(
    word,
    unet,
    vae,
    style_extractor,
    tokenizer,
    noise_scheduler,
    style_refs,
    writer_indices,
    device,
    cfg_scale=5.0,
    img_height=64,
    img_width=256,
    text_max_len=40,
    batch_size=32,
):
    """Generate the given word for each writer index. Returns list of (writer_idx, PIL image)."""
    unet.eval()
    results = []

    # Tokenize text
    text_tokens = tokenizer(
        [word],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=text_max_len,
    ).to(device)

    # Null text for CFG
    null_tokens = None
    if cfg_scale > 1.0:
        null_tokens = tokenizer(
            [""],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=text_max_len,
        ).to(device)

    # Process in batches
    for batch_start in tqdm(range(0, len(writer_indices), batch_size), desc=f"Generating '{word}'"):
        batch_wids = writer_indices[batch_start : batch_start + batch_size]
        n = len(batch_wids)

        # Expand text features
        text_feat = {k: v.expand(n, -1) for k, v in text_tokens.items()}

        # Gather style images and compute features
        style_imgs = []
        for wid in batch_wids:
            if wid in style_refs:
                style_imgs.append(style_refs[wid])
            else:
                # Fallback: white images
                style_imgs.append(torch.ones(5, 3, img_height, img_width) * -1.0)
        style_batch = torch.stack(style_imgs).to(device)  # [n, 5, 3, H, W]
        style_flat = style_batch.reshape(-1, 3, img_height, img_width)
        style_features = style_extractor(style_flat).to(device)

        labels = torch.tensor(batch_wids, dtype=torch.long).to(device)

        # Start from noise in latent space
        x = torch.randn(n, 4, img_height // 8, img_width // 8).to(device)

        # DDIM sampling
        noise_scheduler.set_timesteps(50)
        for t_step in noise_scheduler.timesteps:
            t = (torch.ones(n, device=device) * t_step.item()).long()

            noise_pred = unet(
                x, t, text_feat, labels,
                original_images=style_batch,
                mix_rate=None,
                style_extractor=style_features,
            )

            if cfg_scale > 1.0 and null_tokens is not None:
                null_feat = {k: v.expand(n, -1) for k, v in null_tokens.items()}
                noise_pred_uncond = unet(
                    x, t, null_feat, labels,
                    original_images=style_batch,
                    mix_rate=None,
                    style_extractor=style_features,
                )
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)

            x = noise_scheduler.step(noise_pred, t_step, x).prev_sample

        # Decode latents
        latents = x / 0.18215
        images = vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        for i, wid in enumerate(batch_wids):
            img_pil = torchvision.transforms.ToPILImage()(images[i])
            img_pil = img_pil.convert("L")
            results.append((wid, img_pil))

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate word in all writer styles")
    parser.add_argument("--model", type=str, default=None, help="Model preset name (v2, v3, v4, v5, v6, local)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (overrides --model)")
    parser.add_argument("--style_path", type=str, default=None, help="Path to style encoder checkpoint")
    parser.add_argument("--word", type=str, default="Олесь", help="Word to generate")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale (best at 5.0)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--stable_dif_path", type=str, default="./stable-diffusion-v1-5")
    parser.add_argument("--dataset_root", type=str, default=None, help="Dataset root for style images")
    parser.add_argument("--meta_file", type=str, default=None, help="METAFILE.tsv path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--img_height", type=int, default=64)
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--text_max_len", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=320)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--run_all", action="store_true", help="Run all presets in RUN_ALL_PRESETS")
    parser.add_argument("--zip", action="store_true", default=True, help="Create zip archive of results")
    parser.add_argument("--enable_nafnet_cleanup", action="store_true",
                        help="Apply optional NAFNet cleanup to each generated word (thesis visual workaround)")
    parser.add_argument("--nafnet_ckpt", type=str,
                        default="output/lines204_nafnet_v1/checkpoint_best.pt",
                        help="Path to NAFNet checkpoint for optional word cleanup")
    parser.add_argument("--nafnet_device", type=str, default=None,
                        help="Device for NAFNet cleanup (default: same as --device)")
    parser.add_argument("--nafnet_blend", type=float, default=0.85,
                        help="Blend strength [0..1] for NAFNet cleanup (1.0 = full model output)")
    args = parser.parse_args()

    if args.run_all:
        for preset in RUN_ALL_PRESETS:
            cmd_args = [
                "--model", preset["model"],
                "--word", preset["word"],
                "--cfg_scale", str(preset["cfg_scale"]),
                "--device", args.device,
                "--stable_dif_path", args.stable_dif_path,
            ]
            if args.dataset_root:
                cmd_args += ["--dataset_root", args.dataset_root]
            if args.meta_file:
                cmd_args += ["--meta_file", args.meta_file]
            sub_args = parser.parse_args(cmd_args)
            run_single(sub_args)
        return

    run_single(args)


def run_single(args):
    # Resolve checkpoint path
    if args.checkpoint:
        ckpt_path = args.checkpoint
        style_path = args.style_path or "style_models/ukr_mixed_wt0p7/mixed_ukr_mobilenetv2_100.pth"
    elif args.model and args.model in MODEL_PRESETS:
        preset = MODEL_PRESETS[args.model]
        ckpt_path = preset["checkpoint"]
        style_path = args.style_path or preset["style_path"]
    else:
        raise ValueError(f"Specify --model (one of {list(MODEL_PRESETS.keys())}) or --checkpoint")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device)

    # Load checkpoint and detect num_classes
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = strip_dp_prefix(state_dict)
    num_classes = detect_num_classes(state_dict)
    print(f"Detected {num_classes} writer classes")

    # Load CANINE-C tokenizer + model
    print("Loading CANINE-C...")
    from transformers import CanineModel
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    canine_model = CanineModel.from_pretrained("google/canine-c")

    # Build UNet
    from utils.word_dataset import char_classes as WORD_CHAR_CLASSES
    from types import SimpleNamespace
    vocab_size = WORD_CHAR_CLASSES

    fake_args = SimpleNamespace(interpolation=False, mix_rate=None)
    unet = UNetModel(
        image_size=(args.img_height, args.img_width),
        in_channels=args.channels,
        model_channels=args.emb_dim,
        out_channels=args.channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=args.num_heads,
        num_classes=num_classes,
        context_dim=args.emb_dim,
        vocab_size=vocab_size,
        text_encoder=canine_model,
        args=fake_args,
    )
    unet.load_state_dict(state_dict)
    unet = unet.to(device)
    unet.eval()

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)

    # Load DDIM scheduler
    noise_scheduler = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    # Load style encoder
    print(f"Loading style encoder: {style_path}")
    style_extractor = ImageEncoder(model_name="mobilenetv2_100", num_classes=0, pretrained=False, trainable=False)
    style_sd = torch.load(style_path, map_location="cpu")
    model_dict = style_extractor.state_dict()
    style_sd = {k: v for k, v in style_sd.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(style_sd)
    style_extractor.load_state_dict(model_dict)
    style_extractor = style_extractor.to(device)
    style_extractor.eval()

    nafnet_cleaner = None
    if args.enable_nafnet_cleanup:
        cleanup_device = args.nafnet_device or args.device
        print(
            f"Loading optional NAFNet word cleaner: ckpt={args.nafnet_ckpt}, "
            f"device={cleanup_device}, blend={args.nafnet_blend}"
        )
        nafnet_cleaner = NAFNetWordCleaner(
            checkpoint_path=args.nafnet_ckpt,
            device=cleanup_device,
            blend=args.nafnet_blend,
        )

    # Find dataset for style references
    dataset_root = args.dataset_root
    meta_file = args.meta_file
    if dataset_root is None:
        # Try common locations
        for candidate in [
            "datasets/UkrHandwritten_Words_Clean",
            "UkrHandwritten_Words_Clean",
            "../datasets/UkrHandwritten_Words_Clean",
        ]:
            if os.path.isdir(candidate):
                dataset_root = candidate
                break
    if dataset_root is None:
        raise FileNotFoundError(
            "Cannot find dataset. Use --dataset_root to specify path to UkrHandwritten_Words_Clean"
        )
    if meta_file is None:
        meta_file = os.path.join(dataset_root, "METAFILE.tsv")

    print(f"Loading style references from: {dataset_root}")
    writer_id_map = build_writer_id_map(dataset_root, meta_file)
    style_refs = load_style_images(
        dataset_root, meta_file, writer_id_map,
        img_height=args.img_height, img_width=args.img_width,
    )
    writer_indices = sorted(style_refs.keys())
    print(f"Found {len(writer_indices)} writers with style references")

    if not writer_indices:
        raise RuntimeError("No style references loaded. Check dataset path.")

    # Generate
    results = generate_word(
        word=args.word,
        unet=unet,
        vae=vae,
        style_extractor=style_extractor,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        style_refs=style_refs,
        writer_indices=writer_indices,
        device=device,
        cfg_scale=args.cfg_scale,
        img_height=args.img_height,
        img_width=args.img_width,
        text_max_len=args.text_max_len,
        batch_size=args.batch_size,
    )

    # Save results
    model_name = args.model or "custom"
    output_dir = args.output_dir or f"generated/{model_name}_{args.word}_cfg{args.cfg_scale}"
    os.makedirs(output_dir, exist_ok=True)

    # Reverse map: idx -> writer_str
    idx_to_writer = {v: k for k, v in writer_id_map.items()}

    for wid, img in results:
        writer_str = idx_to_writer.get(wid, f"w{wid:04d}")
        if nafnet_cleaner is not None:
            gray = np.array(img.convert("L"))
            gray = nafnet_cleaner.clean_gray(gray)
            img = Image.fromarray(gray)
        img.save(os.path.join(output_dir, f"writer_{writer_str}.png"))

    print(f"Saved {len(results)} images to {output_dir}")

    # Create zip
    if args.zip:
        zip_path = output_dir.rstrip("/") + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(output_dir):
                if fname.endswith(".png"):
                    zf.write(os.path.join(output_dir, fname), fname)
        print(f"Zip archive: {zip_path}")


if __name__ == "__main__":
    main()
