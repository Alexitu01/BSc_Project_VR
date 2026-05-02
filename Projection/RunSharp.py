import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Add ml-sharp src to path
import sharp
from sharp.cli.predict import predict_image
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import save_ply
import torch

# ── Config ──────────────────────────────────────────────────────────────────
SLICES_DIR  = Path("slices") #Check these paths are still correct on the container
OUTPUT_DIR  = Path("ml-sharp") / "gaussians"
CHECKPOINT  = Path("ml-sharp") / "sharp.pt"
META_FILE   = SLICES_DIR / "slices_meta.txt"
# ────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
ML-Sharp needs to know the focal length the image is 'taken' with to know how wide it needs to make the geomatry, it can get the angle of a given point with:
angle = atan(pixel_offset / focal_length), without this it assumes 30mm focal, and none of the sides will actually be the widt we need them to be to stitch the whole thing back together
"""
# Read focal length from metadata
focal_x = focal_y = None
for line in META_FILE.read_text().splitlines():
    if line.startswith("# focal_x"):
        focal_x = float(line.split("=")[1])
    elif line.startswith("# focal_y"):
        focal_y = float(line.split("=")[1])

if focal_x is None:
    raise ValueError("Could not read focal length from slices_meta.txt")

print(f"Using focal_x={focal_x:.2f}, focal_y={focal_y:.2f}")
f_px_tuple = (focal_x, focal_y)

# Load ml-sharp model once
print("Loading SHARP model...")
state_dict = torch.load(str(CHECKPOINT), map_location=device)
predictor  = create_predictor(PredictorParams())
predictor.load_state_dict(state_dict)
predictor.eval()
predictor.to(device)
print("Model loaded.")

# Process each slice
for img_path in sorted(SLICES_DIR.glob("slice_*.png")):
    name     = img_path.stem
    out_path = OUTPUT_DIR / f"{name}.ply"

    print(f"Processing {name} (focal={focal_x:.1f}px)...")
    face_img = np.array(Image.open(img_path).convert("RGB"))

    # Call ml-sharp Python API directly with correct focal length
    gaussians = predict_image(predictor, face_img, focal_x, device)

    print(f"  {gaussians.mean_vectors.shape[1]:,} Gaussians → {out_path.name}")
    save_ply(gaussians, f_px=focal_x,
             image_shape=(face_img.shape[0], face_img.shape[1]),
             path=out_path)

print("\nDone.")