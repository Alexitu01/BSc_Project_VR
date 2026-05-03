from pathlib import Path
from PIL import Image
import numpy as np
from SplatStitcher import stitch_slices, load_da360
from SplatExporter import export_ply

# ---------------------------------------------------------------------------
# Paths — adjust these to your setup
# ---------------------------------------------------------------------------
PANORAMA_PATH = r'.\test.png'
META_PATH     = Path(r'.\slices\slices_meta.txt')
PLY_FOLDER    = Path(r'.\ml-sharp\gaussians')
DA360_ROOT    = r'.\DA360' #TODO: ensure this points to the root of the da360 clone on the container as well
DA360_MODEL   = r'.\DA360\checkpoints\DA360_small.pth' # Using the small one for testing, but we can download the big one from DA360 google drive

# ---------------------------------------------------------------------------
# 1. Load slice metadata
# ---------------------------------------------------------------------------
focal_x = None
focal_y = None
slices  = []

for line in META_PATH.read_text().splitlines():
    if line.startswith("# focal_x"):
        focal_x = float(line.split("=")[1])
    elif line.startswith("# focal_y"):
        focal_y = float(line.split("=")[1])
    elif not line.startswith("#") and line.strip():
        name, azimuth, img_name = [p.strip() for p in line.split(",")]
        slices.append({
            'ply':         PLY_FOLDER / f"{name}.ply",
            'image':       META_PATH.parent / img_name,
            'azimuth_deg': float(azimuth),
        })

if focal_x is None or focal_y is None:
    raise ValueError("Could not read focal_x / focal_y from metadata file")

print(f"Loaded {len(slices)} slices  focal_x={focal_x:.2f}  focal_y={focal_y:.2f}")

# ---------------------------------------------------------------------------
# 2. Run DA360 on the full panorama once
# ---------------------------------------------------------------------------
pano = np.array(Image.open(PANORAMA_PATH).convert("RGB"))
pano_h, pano_w = pano.shape[:2]
print(f"Panorama: {pano_w}x{pano_h}")

predictor       = load_da360(DA360_MODEL, DA360_ROOT)
da360_disparity = predictor.predict(pano)

print(f"DA360 raw: min={da360_disparity.min():.4f} max={da360_disparity.max():.4f} mean={da360_disparity.mean():.4f}")

print(f"DA360 disparity: shape={da360_disparity.shape}  "
      f"min={da360_disparity.min():.4f}  max={da360_disparity.max():.4f}")

# ---------------------------------------------------------------------------
# 3. Stitch
# ---------------------------------------------------------------------------
cloud = stitch_slices(
    slices          = slices,
    focal_x         = focal_x,
    focal_y         = focal_y,
    pano_width      = pano_w,
    pano_height     = pano_h,
    da360_disparity = da360_disparity,
)

export_ply(cloud, 'stitched_output.ply')