"""
da360_predictor.py
------------------
Loads DA360 and runs it on a full equirectangular panorama,
returning a raw disparity map for use in the stitcher's depth alignment.

DA360 is panorama-aware - it understands the spherical geometry of
equirectangular images and produces globally consistent depth across
the full 360° view. This is the key advantage over Depth Anything V2
which treats each slice independently.

The output is raw disparity (larger = closer). The stitcher samples
this panoramic disparity map at each slice's projected pixel locations
to align all slices to the same global metric scale.

Usage:
    from da360_predictor import DA360Predictor

    predictor = DA360Predictor(
        model_path="./DA360/checkpoints/DA360_small.pth",
        da360_root="./DA360",
    )
    disparity = predictor.predict(panorama_rgb)  # (H, W) float32

Dependencies:
    DA360 repo must be cloned at da360_root.
    pip install torch torchvision timm einops opencv-python
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


class DA360Predictor:
    """
    Wraps the DA360 model for single-image inference.

    Loads the model once at construction time. Call predict() for
    each panorama - it handles resizing to the model's expected
    input dimensions and returns a full-resolution disparity map.

    Args:
        model_path: path to DA360 .pth checkpoint file
        da360_root: path to the cloned DA360 repository root
                    (needed so we can import its networks module)
        device:     torch device - defaults to CUDA if available
    """

    # DA360's fixed input resolution (from test.py)
    MODEL_HEIGHT = 518
    MODEL_WIDTH  = 1036

    def __init__(
        self,
        model_path: str,
        da360_root: str,
        device: torch.device = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Add DA360 repo to path so we can import its modules
        da360_root = str(Path(da360_root).resolve())
        if da360_root not in sys.path:
            sys.path.insert(0, da360_root)

        import networks #can't import this before the path is set up

        # Load checkpoint - same logic as test.py
        print(f"Loading DA360 from: {model_path}")
        model_dict = torch.load(model_path, map_location=self.device)

        # Fill defaults for older checkpoints (from test.py)
        model_dict.setdefault("net",             "DA360")
        model_dict.setdefault("dinov2_encoder",  "vits")
        model_dict.setdefault("height",          self.MODEL_HEIGHT)
        model_dict.setdefault("width",           self.MODEL_WIDTH)

        Net   = getattr(networks, model_dict["net"])
        model = Net(
            model_dict["height"],
            model_dict["width"],
            dinov2_encoder=model_dict["dinov2_encoder"],
        )
        model.to(self.device)

        # Load weights - strict=False matches test.py behaviour
        model_state = model.state_dict()
        model.load_state_dict(
            {k: v for k, v in model_dict.items() if k in model_state},
            strict=False,
        )
        model.eval()
        self.model = model

        print(f"DA360 loaded on {self.device}")

    def predict(self, panorama_rgb: np.ndarray) -> np.ndarray:
        """
        Run DA360 on an equirectangular panorama.

        Args:
            panorama_rgb: (H, W, 3) uint8 RGB equirectangular panorama
                          Any aspect ratio accepted - resized internally.

        Returns:
            disparity: (H, W) float32 disparity map in panorama resolution
                       Larger values = closer to camera.
                       This is raw pred_disp - NOT normalised, NOT inverted.
                       Pass directly to the stitcher's align_splat_depths.
        """
        orig_h, orig_w = panorama_rgb.shape[:2]

        # Normalise to [0, 1] float and convert to tensor
        # DA360's Real dataset does: (rgb / 255 - mean) / std
        # We replicate that preprocessing here
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = panorama_rgb.astype(np.float32) / 255.0
        img = (img - mean) / std                          # (H, W, 3)
        img = torch.from_numpy(img).permute(2, 0, 1)     # (3, H, W)
        img = img.unsqueeze(0).to(self.device)            # (1, 3, H, W)

        # Resize to model's expected input resolution
        img_resized = F.interpolate(
            img,
            size=(self.MODEL_HEIGHT, self.MODEL_WIDTH),
            mode="bilinear",
            align_corners=True,
        )

        with torch.no_grad():
            outputs  = self.model(img_resized)
            pred_disp = outputs["pred_disp"]  # (1, 1, H_model, W_model)

        # Remove batch and channel dims → (H_model, W_model)
        pred_disp = pred_disp.squeeze().cpu().float()

        # Upsample back to original panorama resolution
        pred_disp = F.interpolate(
            pred_disp.unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=True,
        ).squeeze().numpy()  # (H, W) float32

        return pred_disp.astype(np.float32)