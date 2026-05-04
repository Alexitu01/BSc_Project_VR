"""
face_extractor.py
-----------------
Extracts N overlapping perspective slice images from a 360° equirectangular
panorama arranged in a horizontal ring around the horizon.

Each slice is a tall perspective image facing outward at a given azimuth.
Adjacent slices overlap by overlap_degrees on each side, giving the depth
model better edge context and providing genuine redundant splat coverage
for smooth seam blending in the stitcher.

The extractor returns:
  - The slice images (numpy arrays)
  - The azimuth angle for each slice (passed to the stitcher)
  - The focal length used (must be passed to the stitcher and ML sharp)

Usage:
    python face_extractor.py panorama.jpg ./slices/ --slices 6 --overlap 8 

Dependencies:
    pip install numpy pillow scipy
"""

import argparse
import math
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import map_coordinates



TWO_PI = 2.0 * math.pi
PI     = math.pi


# ---------------------------------------------------------------------------
# Core slice extraction
# ---------------------------------------------------------------------------

def extract_slice(
    pano:         np.ndarray,
    azimuth_deg:  float,
    image_width:  int,
    image_height: int,
    focal_x:      float,
    focal_y:      float,
) -> np.ndarray:
    """
    Extract one perspective slice from an equirectangular panorama. (we're using 2.33:1 which is close enough to 2:1)
    Essentially what we're doing here, is working backwards, we feed it how big a slice should be, (which we figure out later) along with the angle and focal 
    (Focal lenght is some camera magic, but important bit is we can figure out pixel_offset = focal_length * tan(angle)) 
    to sample the correct pixels from the panorama image to the slice.

    Full perspective projection on both axes 
    consistent with DA360 depth alignment which assumes perspective geometry.

    focal_y = focal_x (square pixels). The tall image_height relative to
    focal_x naturally produces a wide vertical FOV without fisheye distortion.

    Args:
        pano:         (H, W, 3) uint8 equirectangular panorama
        azimuth_deg:  centre azimuth in degrees (0=+Z, 90=+X, 180=-Z)
        image_width:  output slice width in pixels
        image_height: output slice height (= panorama height)
        focal_x:      horizontal focal length in pixels
        focal_y:      vertical focal length in pixels (= focal_x)

    Returns:
        (image_height, image_width, 3) uint8 perspective slice image
    """
    #Get dimensions from pano
    erp_h, erp_w = pano.shape[:2]
    az = math.radians(azimuth_deg) #Converting to radians cuz python math uses radians 

    # Camera axes - (it's images so Y+ is down)
    forward = np.array([ math.sin(az), 0.0,  math.cos(az)], dtype=np.float64)
    right   = np.array([ math.cos(az), 0.0, -math.sin(az)], dtype=np.float64)
    down    = np.array([ 0.0,          1.0,  0.0          ], dtype=np.float64)
    R       = np.column_stack([right, down, forward]) #Rotation matrix

    # Pixel grid - centre of each pixel
    u  = np.arange(image_width,  dtype=np.float64) + 0.5
    v  = np.arange(image_height, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v, indexing="xy")

    # Local ray - pinhole camera
    cx      = image_width  / 2.0
    cy      = image_height / 2.0
    local_x = (uu - cx) / focal_x
    local_y = (vv - cy) / focal_y
    local_z = np.ones_like(uu)

    # Rotate to world space
    wx = R[0,0]*local_x + R[0,1]*local_y + R[0,2]*local_z
    wy = R[1,0]*local_x + R[1,1]*local_y + R[1,2]*local_z
    wz = R[2,0]*local_x + R[2,1]*local_y + R[2,2]*local_z

    # Normalise to unit vectors (easier to wrok with)
    norm = np.maximum(np.sqrt(wx**2 + wy**2 + wz**2), 1e-12)
    wx /= norm; wy /= norm; wz /= norm

    # World direction -> spherical coordinates, basically vectors to long, lat
    longitude = np.arctan2(wx, wz)
    latitude  = np.arcsin(np.clip(wy, -1.0, 1.0))

    # long, lat -> panorama pixel coordinates
    sample_x = (longitude / TWO_PI + 0.5) * erp_w - 0.5
    sample_y = (latitude  / PI     + 0.5) * erp_h - 0.5

    #Sample the panorama image at the given coords, luckily scipy handles bilinear interpolation and seam wrapping, praise be scipy
    slc = np.stack([
        map_coordinates(pano[:,:,c], [sample_y, sample_x], order=1, mode='wrap')
        for c in range(3)
    ], axis=-1)

    return slc.astype(np.uint8)


def build_slice_layout(
    panorama_width:  int,
    panorama_height: int,
    n_slices:        int   = 6,
    overlap_degrees: float = 8.0, #8 seems to give good results
    face_size:       int   = None,
) -> tuple[list[dict], float, float]:
    """
    Compute extraction parameters for N evenly-spaced horizon slices.

    Matches SPAG4D exactly:
      - focal_y = focal_x  (square pixels)
      - image_height = panorama_height  (full vertical, no cropping)
      - image_width widened for overlap FOV
      - focal_x from horizontal FOV

    Returns slices list, focal_x, focal_y.
    """
    #If size is not given, it evenly divides the panorama, just let it do this, don't give it a size
    if face_size is None:
        face_size = panorama_width // n_slices

    span_deg      = 360.0 / n_slices
    total_fov_deg = min(170.0, span_deg + 2.0 * overlap_degrees)
    image_width   = int(round(face_size * (total_fov_deg / span_deg)))

    # f = (w/2) / tan(fov/2)
    # This is just the pinhole formula rearanged: https://csundergrad.science.uoit.ca/courses/cv-notes/notebooks/01-image-formation.html
    focal_x = (image_width / 2.0) / math.tan(math.radians(total_fov_deg / 2.0))
    focal_y = focal_x          # square pixels 
    image_height = panorama_height  # full panorama height always, if we don't do this the top and bottom of the image clip, idk couldn't figure out a smarter fix

    #Just for prints
    vfov = 2.0 * math.degrees(math.atan((image_height / 2.0) / focal_y))

    print(f"  Slice dimensions: {image_width}x{image_height} px")
    print(f"  Horizontal FOV:   {total_fov_deg:.1f}deg  "
          f"(span={span_deg:.0f}deg + {overlap_degrees:.0f}deg overlap each side)")
    print(f"  Vertical FOV:     {vfov:.1f}deg  "
          f"(focal={focal_x:.1f}px, height={image_height}px)")
    if vfov < 90.0:
        print(f"  WARNING: vertical FOV {vfov:.1f}deg may clip - "
              f"consider fewer slices or a taller panorama")

    #build slice meta objects
    slices = []
    for i in range(n_slices):
        slices.append({
            "name":         f"slice_{i:03d}",
            "azimuth_deg":  i * span_deg,
            "image_width":  image_width,
            "image_height": image_height,
            "focal_x":      focal_x,
            "focal_y":      focal_y,
        })

    return slices, focal_x, focal_y


def extract_all_slices(
    pano:            np.ndarray,
    n_slices:        int   = 6,
    overlap_degrees: float = 8.0,
    face_size:       int   = None,
) -> tuple[list[dict], float]:
    """
    Extract all perspective slices from an equirectangular panorama. -> just calls the two previous methods and does a lot of debug printing

    Args:
        pano:            (H, W, 3) uint8 equirectangular panorama
        n_slices:        number of horizontal slices (default 6)
        overlap_degrees: overlap per side in degrees (default 8)
        face_size:       base width per slice (default panorama_width//n_slices*2)

    Returns:
        slices:   list of dicts, each containing:
                      image       - (H, W, 3) uint8 slice image
                      azimuth_deg - centre azimuth (pass to stitcher)
                      focal_x     - focal length (pass to stitcher)
                      focal_y     - focal length (pass to stitcher)
                      name        - slice name
        focal_px: focal length in pixels (same for all slices)
    """
    h, w = pano.shape[:2]

    layout, focal_px, focal_y = build_slice_layout(w, h, n_slices, overlap_degrees, face_size)

    span_deg    = 360.0 / n_slices #If I wasn't lazy then this should just be a global
    total_fov   = min(170.0, span_deg + 2.0 * overlap_degrees)

    print(f"Extracting {n_slices} horizontal slices:")
    print(f"  Panorama:       {w}×{h}")
    print(f"  Slice size:     {layout[0]['image_width']}×{layout[0]['image_height']} px")
    print(f"  Span per slice: {span_deg:.1f}°")
    print(f"  FOV per slice:  {total_fov:.1f}° ({overlap_degrees}° overlap each side)")
    print(f"  Focal X:        {focal_px:.2f} px (horizontal)")
    print(f"  Focal Y:        {layout[0]['focal_y']:.2f} px (vertical)")
    print(f"  Vertical FOV:   {2*math.degrees(math.atan(layout[0]['image_height']/2/layout[0]['focal_y'])):.1f}°")
    print(f"  ← pass focal_x={focal_px:.2f}, focal_y={layout[0]['focal_y']:.2f} to stitch_faces()")
    print()

    results = []
    for s in layout:
        print(f"  Extracting: {s['name']} (azimuth={s['azimuth_deg']:.1f}°)...")
        img = extract_slice(
            pano,
            s["azimuth_deg"],
            s["image_width"],
            s["image_height"],
            s["focal_x"],
            s["focal_y"],
        )
        results.append({
            "name":         s["name"],
            "image":        img,
            "azimuth_deg":  s["azimuth_deg"],
            "focal_x":      s["focal_x"],
            "focal_y":      s["focal_y"],
            "image_width":  s["image_width"],
            "image_height": s["image_height"],
        })

    print("\nDone.")
    return results, focal_px, focal_y


# ---------------------------------------------------------------------------
# Entry point 
# ---------------------------------------------------------------------------

def Extract():
    parser = argparse.ArgumentParser(
        description="Extract overlapping perspective slices from a 360° panorama"
    )
    parser.add_argument("panorama",      help="Input equirectangular panorama (.jpg/.png)")
    parser.add_argument("output_folder", help="Folder to save slice images")
    parser.add_argument("--slices",  type=int,   default=6,    help="Number of slices (default 6)")
    parser.add_argument("--overlap", type=float, default=8.0, help="Overlap degrees per side (default 8)")
    parser.add_argument("--size",    type=int,   default=None, help="Base slice width in pixels")
    args = parser.parse_args()

    pano = np.array(Image.open(args.panorama).convert("RGB"))
    print(f"Loaded: {args.panorama}  ({pano.shape[1]}×{pano.shape[0]})\n")

    slices, focal_px, focal_y = extract_all_slices(
        pano,
        n_slices=args.slices,
        overlap_degrees=args.overlap,
        face_size=args.size,
    )

    out = Path(args.output_folder)
    out.mkdir(parents=True, exist_ok=True)

    # Save a metadata file alongside the images so the stitcher knows
    # the azimuth and focal length for each slice
    slices, focal_px, focal_y = slices, focal_px, focal_y
    meta_lines = [
        f"# Slice metadata - pass these values to stitch_faces()",
        f"# focal_x = {focal_px:.4f}",
        f"# focal_y = {focal_y:.4f}",
        f"# Format: name, azimuth_deg, image_path",
    ]

    for s in slices:
        path = out / f"{s['name']}.png"
        Image.fromarray(s["image"]).save(path)
        print(f"  Saved: {path}  (azimuth={s['azimuth_deg']:.1f}°)")
        meta_lines.append(f"{s['name']}, {s['azimuth_deg']:.4f}, {path.name}")

    meta_path = out / "slices_meta.txt"
    meta_path.write_text("\n".join(meta_lines))
    print(f"\n  Metadata saved: {meta_path}")
    print(f"\nfocal_x={focal_px:.2f}, focal_y={focal_y:.2f}  <- pass both to stitch_faces()")


if __name__ == "__main__":
    Extract()