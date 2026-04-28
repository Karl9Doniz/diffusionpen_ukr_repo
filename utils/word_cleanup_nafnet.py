import os
import sys

import numpy as np
import torch


def _import_nafnet_components():
    """Import NAFNetLite and forward_padded from training script."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scripts_dir = os.path.join(repo_root, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from train_lines204_nafnet import NAFNetLite, forward_padded  # type: ignore

    return NAFNetLite, forward_padded


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


class NAFNetWordCleaner:
    """Optional word-level post-cleaner using the trained line-level NAFNet checkpoint."""

    def __init__(self, checkpoint_path: str, device: str = "cuda:0", blend: float = 0.85):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"NAFNet checkpoint not found: {checkpoint_path}")

        self.device = _resolve_device(device)
        self.blend = float(np.clip(blend, 0.0, 1.0))
        NAFNetLite, self._forward_padded = _import_nafnet_components()

        payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict) and "model_state" in payload:
            model_state = payload["model_state"]
            args = payload.get("args", {})
            model = NAFNetLite(
                in_ch=1,
                width=int(args.get("width", 32)),
                enc_blocks=int(args.get("enc_blocks", 2)),
                middle_blocks=int(args.get("middle_blocks", 4)),
                dec_blocks=int(args.get("dec_blocks", 2)),
            )
        elif isinstance(payload, dict):
            model_state = payload
            model = NAFNetLite(in_ch=1, width=32, enc_blocks=2, middle_blocks=4, dec_blocks=2)
        else:
            raise RuntimeError(f"Unsupported checkpoint payload at {checkpoint_path}")

        model.load_state_dict(model_state)
        model.to(self.device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def clean_gray(self, img_gray: np.ndarray) -> np.ndarray:
        """Run NAFNet cleanup on a single grayscale image (uint8)."""
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)

        inp = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        inp = inp.to(self.device)
        pred = self._forward_padded(self.model, inp, factor=4)
        if self.blend < 1.0:
            pred = inp * (1.0 - self.blend) + pred * self.blend

        out = torch.clamp(pred[0, 0] * 255.0, 0, 255).round().byte().cpu().numpy()
        return out
