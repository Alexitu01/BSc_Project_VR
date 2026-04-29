"""
splat_stitcher.py
-----------------
Takes 6 ml-sharp .ply files (one per cube face) and merges them into
a single world-space SplatCloud ready for optimisation and export.

Pipeline per face:
  1. Parse .ply -> SplatCloud
  2. Run Depth Anything V2 on the face image -> depth map
  3. Align splat depths to the depth map (robust median scale per grid cell)
  4. Voronoi border clipping (remove splats outside this face's angular zone)
  5. Rotate positions and orientations into world space
  6. Merge all faces + Y-flip to match viewer convention

Usage:
    from splat_stitcher import stitch_faces
    cloud = stitch_faces(
        face_plys={
            'front':  'face_front.ply',
            'back':   'face_back.ply',
            'right':  'face_right.ply',
            'left':   'face_left.ply',
            'top':    'face_top.ply',
            'bottom': 'face_bottom.ply',
        },
        face_images={
            'front':  'face_front.jpg',
            ...
        },
        focal_x=740.0,   # from your cubemap extraction
        focal_y=740.0,
    )

Dependencies:
    pip install numpy pillow transformers torch
"""

import math
import numpy as np
from pathlib import Path
from PIL import Image
from SplatParser import parse_ply, SplatCloud


PI = math.pi


# ---------------------------------------------------------------------------
# Rotation matrix builders
# ---------------------------------------------------------------------------

def _rot_x(degrees: float) -> np.ndarray:
    r = np.radians(degrees)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

def _rot_y(degrees: float) -> np.ndarray:
    r = np.radians(degrees)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

def _identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


# ---------------------------------------------------------------------------
# Face definitions
# ---------------------------------------------------------------------------
#
# Each face is defined by its rotation matrix R (3x3).
# R maps from view-local coordinates to world coordinates:
#   world_point = R @ local_point
#
# The cubemap convention (matching your existing cubemap extractor):
#   front  = looking along +Z
#   back   = looking along -Z  -> rotate 180° around Y
#   right  = looking along +X  -> rotate +90° around Y  (left-handed: positive = right)
#   left   = looking along -X  -> rotate -90° around Y
#   top    = looking along +Y  -> rotate +90° around X  (left-handed: positive = up)
#   bottom = looking along -Y  -> rotate -90° around X
#
# These signs were determined empirically from your data.

FACE_ROTATIONS: dict[str, np.ndarray] = {
    'front':  _identity(),
    'back':   _rot_y(180),
    'right':  _rot_y(90),
    'left':   _rot_y(-90),
    #'top':    _rot_x(90),
    #'bottom': _rot_x(-90),
}

# Horizontal angular span each face covers in world space (degrees)
# For a cubemap, each face covers 90° of horizontal FOV
FACE_SPAN_DEGREES = 90.0


# ---------------------------------------------------------------------------
# Depth Anything V2 loader (lazy - loaded once on first use)
# ---------------------------------------------------------------------------

_depth_pipe = None

def _get_depth_pipe():
    """
    Load Depth Anything V2 Small via HuggingFace transformers.
    Loaded lazily so import doesn't trigger a download.

    Uses the Small model for speed - swap to 'Base' or 'Large' for
    better quality at the cost of memory and speed.
    """
    global _depth_pipe
    if _depth_pipe is None:
        from transformers import pipeline
        print("  Loading Depth Anything V2 (downloads on first run)...")
        _depth_pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
        )
        print("  Depth model loaded.")
    return _depth_pipe


def predict_depth(face_image: np.ndarray) -> np.ndarray:
    """
    Run Depth Anything V2 on a face image.

    Args:
        face_image: (H, W, 3) uint8 RGB face image

    Returns:
        (H, W) float32 depth map - larger values = farther away.

    Note: Depth Anything outputs INVERSE depth (disparity-like) by default
    from the HuggingFace pipeline. We invert it here to get depth where
    larger = farther. This matches the convention of ml-sharp's z coordinates.
    """
    pipe = _get_depth_pipe()
    pil_image = Image.fromarray(face_image)

    # Pipeline returns a PIL depth image - convert to float numpy
    result = pipe(pil_image)
    depth_pil = result["depth"]
    depth = np.array(depth_pil).astype(np.float32)

    # HuggingFace depth-anything pipeline returns values where
    # LARGER = CLOSER (disparity convention).
    # We need LARGER = FARTHER to match ml-sharp's z convention.
    # Invert: depth = max - raw  (keeps values positive)
    depth = depth.max() - depth

    # No per-face normalisation - raw inverted values flow through unchanged.
    # Per-face scaling (even percentile-based) creates inconsistent metric
    # spaces across faces, which the reference_scale correction in stitch_faces
    # then has to fight against. Keeping values raw means reference_scale
    # is working with comparable numbers across all faces.
    depth = depth.astype(np.float32)
    depth += 1e-6  # avoid zeros

    return depth


# ---------------------------------------------------------------------------
# Depth alignment
# ---------------------------------------------------------------------------

def align_splat_depths(
    cloud: SplatCloud,
    depth_map: np.ndarray,
    focal_x: float,
    focal_y: float,
    image_width: int,
    image_height: int,
    grid_size: int = 8,
) -> tuple[SplatCloud, float]:
    """
    Scale each splat's depth to agree with the Depth Anything V2 depth map.

    The problem: ml-sharp outputs splats at arbitrary scale. Each face may
    have a completely different scale. We need all faces at the same scale
    before rotating them into world space, or the seams won't meet.

    The approach (simplified from SPAG4D's DA360 alignment):
    https://github.com/cedarconnor/SPAG4d/blob/main/spag4d/sharp360.py
    1. Project each splat's 3D position back to 2D pixel coordinates
    2. Sample the reference depth map at those pixel coordinates
    3. Compute a scale factor: reference_depth / splat_depth
    4. Apply that scale factor to positions and splat sizes

    We do this on a coarse grid (default 8x8) rather than per-splat,
    which gives a smooth spatially-varying scale field. This handles
    scenes where depth varies significantly across the face (e.g. a
    corridor with a close wall on one side and open space on the other).

    Args:
        cloud:        SplatCloud in view-local coordinates
        depth_map:    (H, W) float32 depth map from Depth Anything V2
        focal_x:      horizontal focal length in pixels
        focal_y:      vertical focal length in pixels
        image_width:  face image width
        image_height: face image height
        grid_size:    NxN grid for the scale field

    Returns:
        Tuple of (depth-aligned SplatCloud, global_scale float).
        All return paths return this tuple - the scale is used by the
        caller to anchor subsequent faces to the same metric space.
    """
    positions = cloud.positions.astype(np.float64)
    depth_z = positions[:, 2]

    # Only process splats in front of the camera (positive z)
    valid = depth_z > 1e-6

    if valid.sum() < 32:
        # Too few valid splats to align - return unchanged with neutral scale
        print("    Warning: too few valid splats for depth alignment, skipping")
        return cloud, 1.0

    # Project splat centres to pixel coordinates
    # Pinhole projection: px = (x/z) * focal + cx
    cx = image_width  / 2.0
    cy = image_height / 2.0

    # Use safe_z to avoid division by zero for behind-camera splats.
    # Those positions are masked out by the `valid` array anyway, so
    # the 1.0 fallback value never reaches the final result.
    safe_z = np.where(valid, depth_z, 1.0)
    px = np.where(valid, (positions[:, 0] / safe_z) * focal_x + cx, 0.0)
    py = np.where(valid, (positions[:, 1] / safe_z) * focal_y + cy, 0.0)

    # The face was extracted with an overlap FOV wider than 90°, so splats
    # near the edges project to pixel coordinates outside [0, face_size].
    # fov_scale = tan(half_fov) = half_width / focal_x is the ratio by which
    # the face extends beyond the standard 90° boundary. We allow projections
    # up to margin_px outside the image bounds before discarding them.
    fov_scale = cx / focal_x
    margin_px = cx * (fov_scale - 1.0)

    in_bounds = (
        valid
        & (px >= -margin_px) & (px < image_width  + margin_px)
        & (py >= -margin_px) & (py < image_height + margin_px)
    )

    if in_bounds.sum() < 32:
        print("    Warning: too few in-bounds splats for depth alignment, skipping")
        return cloud, 1.0

    # Use radial distance - not camera Z depth.
    # Monocular depth models like Depth Anything behave more like radial
    # distance than camera-space Z. At cube face edges, Z becomes compressed
    # while radial remains geometrically correct. Using Z causes edge expansion
    # and seam gaps.
    splat_radial_samples = np.linalg.norm(positions[in_bounds], axis=1)

    # Sample reference depth at each splat's projected pixel.
    # Clamp to valid pixel range for the lookup even though in_bounds already
    # restricts us - the overlap margin means px/py can be slightly outside
    # [0, image_size-1] and we want a valid array index.
    px_int = np.clip(px[in_bounds].astype(np.int32), 0, image_width  - 1)
    py_int = np.clip(py[in_bounds].astype(np.int32), 0, image_height - 1)

    # Resize depth map to face image dimensions if needed.
    # Cast to float32 explicitly: PIL's fromarray requires float32 for mode-F
    # images and will raise TypeError on float64 input.
    dh, dw = depth_map.shape
    if dh != image_height or dw != image_width:
        from PIL import Image as PILImage
        depth_pil = PILImage.fromarray(depth_map.astype(np.float32))
        depth_pil = depth_pil.resize((image_width, image_height), PILImage.BILINEAR)
        depth_map_resized = np.array(depth_pil).astype(np.float32)
    else:
        depth_map_resized = depth_map.astype(np.float32)

    ref_depth_samples = depth_map_resized[py_int, px_int]

    # Scale ratio: reference_depth / splat_radial
    valid_ratio = (ref_depth_samples > 1e-4) & (splat_radial_samples > 1e-6)
    if valid_ratio.sum() < 32:
        print("    Warning: insufficient valid depth samples, using median normalisation")
        median_z = float(np.median(depth_z[depth_z > 1e-6]))
        sf = 1.0 / median_z if median_z > 1e-6 else 1.0
        return SplatCloud(
            count      = cloud.count,
            positions  = (positions * sf).astype(np.float32),
            normals    = cloud.normals,
            colors_rgb = cloud.colors_rgb,
            f_dc       = cloud.f_dc,
            opacities  = cloud.opacities,
            scales     = (cloud.scales * sf).astype(np.float32),
            rotations  = cloud.rotations,
        ), sf

    # raw_scales, px_valid and py_valid are all indexed over the same subset:
    # positions[in_bounds][valid_ratio]. Assert this so any future refactor
    # that breaks the alignment fails loudly rather than silently corrupting
    # the scale field.
    raw_scales = ref_depth_samples[valid_ratio] / splat_radial_samples[valid_ratio]
    px_valid   = px[in_bounds][valid_ratio]
    py_valid   = py[in_bounds][valid_ratio]
    assert len(raw_scales) == len(px_valid) == len(py_valid)

    # Global robust median scale (trim top/bottom 5% to remove outliers)
    lo, hi = np.quantile(raw_scales, [0.05, 0.95])
    trimmed = raw_scales[(raw_scales >= lo) & (raw_scales <= hi)]
    global_scale = float(np.median(trimmed)) if trimmed.size > 0 else float(np.median(raw_scales))

    # ------------------------------------------------------------------
    # Build a coarse NxN grid of local scale factors
    #
    # Rather than one global scale for the whole face, we compute a
    # per-cell median. This handles scenes where depth varies across
    # the face. The grid values are then bilinearly interpolated to
    # give a smooth per-splat scale field.
    # ------------------------------------------------------------------
    cell_w = image_width  / grid_size
    cell_h = image_height / grid_size
    grid = np.full((grid_size, grid_size), global_scale, dtype=np.float64)

    for gy in range(grid_size):
        for gx in range(grid_size):
            in_cell = (
                (px_valid >= gx * cell_w) & (px_valid < (gx + 1) * cell_w) &
                (py_valid >= gy * cell_h) & (py_valid < (gy + 1) * cell_h)
            )
            if in_cell.sum() >= 4:
                cs = raw_scales[in_cell]
                cl, ch = np.quantile(cs, [0.1, 0.9])
                ct = cs[(cs >= cl) & (cs <= ch)]
                if ct.size > 0:
                    grid[gy, gx] = float(np.median(ct))

    # Clamp grid values to a reasonable range around the global scale.
    # ±20% allows meaningful spatial correction for scenes with genuine
    # depth variation across the face (e.g. a near wall on one side,
    # open space on the other), while still preventing extreme local
    # distortion that would shear seams apart. Cross-face scale drift
    # is handled separately by the reference_scale correction in
    # stitch_faces, so we don't need to be overly conservative here.
    grid = np.clip(grid, global_scale * 0.80, global_scale * 1.20)

    # Bilinear interpolate grid to every splat's position
    all_px = np.clip(px, 0, image_width  - 1)
    all_py = np.clip(py, 0, image_height - 1)
    gxc = all_px / cell_w - 0.5
    gyc = all_py / cell_h - 0.5
    gx0 = np.clip(np.floor(gxc).astype(np.int32), 0, grid_size - 1)
    gy0 = np.clip(np.floor(gyc).astype(np.int32), 0, grid_size - 1)
    gx1 = np.clip(gx0 + 1, 0, grid_size - 1)
    gy1 = np.clip(gy0 + 1, 0, grid_size - 1)
    wx = np.clip(gxc - gx0, 0.0, 1.0)
    wy = np.clip(gyc - gy0, 0.0, 1.0)

    per_splat_scale = (
        grid[gy0, gx0] * (1 - wx) * (1 - wy) +
        grid[gy0, gx1] * wx       * (1 - wy) +
        grid[gy1, gx0] * (1 - wx) * wy       +
        grid[gy1, gx1] * wx       * wy
    ).astype(np.float32)

    # Splats behind the camera (valid=False) had their px/py forced to 0.0,
    # so they would otherwise receive whatever scale the grid happens to have
    # at the image centre. Apply global_scale instead so they are consistently
    # normalised without being distorted by a depth sample that has nothing
    # to do with their actual position.
    per_splat_scale[~valid] = global_scale

    # Apply per-splat scale to positions and splat sizes
    new_positions = (positions * per_splat_scale[:, None]).astype(np.float32)
    new_scales    = (cloud.scales * per_splat_scale[:, None]).astype(np.float32)

    print(f"    Depth aligned: global_scale={global_scale:.4f}  "
          f"grid_range=[{grid.min():.4f}, {grid.max():.4f}]  "
          f"splat_range=[{per_splat_scale.min():.4f}, {per_splat_scale.max():.4f}]")

    return SplatCloud(
        count      = cloud.count,
        positions  = new_positions,
        normals    = cloud.normals,
        colors_rgb = cloud.colors_rgb,
        f_dc       = cloud.f_dc,
        opacities  = cloud.opacities,
        scales     = new_scales,
        rotations  = cloud.rotations,
    ), global_scale


# ---------------------------------------------------------------------------
# Edge fade 
# ---------------------------------------------------------------------------

def apply_edge_fade(
    cloud: SplatCloud,
    focal_x: float,
    focal_y: float,
    image_width: int,
    image_height: int,
    fade_start: float = 0.75,
) -> SplatCloud:
    """
    Attenuate opacity of splats near the face edges rather than hard-clipping.

    Hard Voronoi clipping creates cracks and sparse regions at seams because
    the boundary is binary - a splat is either kept or discarded. This creates
    visible face boundaries especially at corners.

    Soft edge fading instead reduces the opacity of splats near the edge of
    the face's FOV zone. Splats near the centre stay fully opaque. Splats
    near the edge fade out. When two overlapping faces both have faded edges,
    the overlap region blends smoothly rather than cutting sharply.

    How it works:
    1. Project each splat to normalised image coordinates [-1, 1]
    2. Compute how far the splat is from the face centre (Chebyshev distance)
       - this is max(|nx|, |ny|) which forms a square boundary matching the
       face shape, rather than a circle
    3. Apply a linear fade from fade_start to 1.0
       - splats within fade_start of centre are fully opaque
       - splats at the edge (r=1.0) have zero weight
       - splats beyond the edge (overlap region) fade to zero

    Args:
        cloud:        SplatCloud in view-local coordinates
        focal_x:      horizontal focal length in pixels
        focal_y:      vertical focal length in pixels
        image_width:  face image width in pixels
        image_height: face image height in pixels
        fade_start:   normalised distance from centre where fade begins
                      0.75 means inner 75% is fully opaque, outer 25% fades

    Returns:
        SplatCloud with attenuated opacities near edges.
        No splats are removed - only opacities are modified.
    """

    positions = cloud.positions.astype(np.float64)
    depth_z   = positions[:, 2]
    cx = image_width  / 2.0
    cy = image_height / 2.0

    # Project splat centres to pixel coordinates
    safe_z = np.where(depth_z > 1e-6, depth_z, 1.0)
    px = (positions[:, 0] / safe_z) * focal_x + cx
    py = (positions[:, 1] / safe_z) * focal_y + cy
    
    # Recover the fov_scale from the focal length and face size
    # tan(half_fov) = (face_size/2) / focal
    fov_scale_x = (image_width  / 2.0) / focal_x
    fov_scale_y = (image_height / 2.0) / focal_y

    # Normalise to [-1, 1] at the 90° boundary (not the full face extent)
    nx = (px / image_width  * 2.0 - 1.0) / fov_scale_x
    ny = (py / image_height * 2.0 - 1.0) / fov_scale_y

    # Chebyshev distance: max(|nx|, |ny|) forms a square boundary
    # matching the face shape. r=0 at centre, r=1 at edge, r>1 in overlap.
    r = np.maximum(np.abs(nx), np.abs(ny))

    # Linear fade: 1.0 inside fade_start, 0.0 at r=1.0, clipped to [0,1]
    # weight = (1 - r) / (1 - fade_start)  then clipped
    weight = np.clip((1.0 - r) / (1.0 - fade_start), 0.0, 1.0).astype(np.float32)

    # Splats behind camera get zero weight
    weight[depth_z <= 1e-6] = 0.0

    new_opacities = cloud.opacities * weight

    n_faded = int((weight < 1.0).sum())
    print(f"    Edge fade: {n_faded:,} splats attenuated "
          f"({100*n_faded/cloud.count:.1f}%)")

    return SplatCloud(
        count      = cloud.count,
        positions  = cloud.positions,
        normals    = cloud.normals,
        colors_rgb = cloud.colors_rgb,
        f_dc       = cloud.f_dc,
        opacities  = new_opacities,
        scales     = cloud.scales,
        rotations  = cloud.rotations,
    )


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion (w, x, y, z)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Compose face rotation q1 (4,) with N splat rotations q2 (N, 4).
    Returns (N, 4).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
    result = np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=1)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return (result / np.where(norms > 0, norms, 1.0)).astype(np.float32)


def transform_face(cloud: SplatCloud, face_name: str) -> SplatCloud:
    """
    Rotate a face's SplatCloud from view-local space into world space.
    Rotates both positions and orientation quaternions.
    """
    R = FACE_ROTATIONS[face_name]
    new_positions = (R @ cloud.positions.T).T.astype(np.float32)
    q_face        = matrix_to_quaternion(R)
    new_rotations = quaternion_multiply(q_face, cloud.rotations)
    return SplatCloud(
        count      = cloud.count,
        positions  = new_positions,
        normals    = cloud.normals,
        colors_rgb = cloud.colors_rgb,
        f_dc       = cloud.f_dc,
        opacities  = cloud.opacities,
        scales     = cloud.scales,
        rotations  = new_rotations,
    )


def flip_y(cloud: SplatCloud) -> SplatCloud:
    """
    Negate the Y axis of all splat positions.

    ml-sharp uses Y-down convention (down = +Y in world space).
    Standard 3DGS viewers and Unity use Y-up convention.
    Negating Y converts between them.
    """
    new_positions = cloud.positions.copy()
    new_positions[:, 1] *= -1.0
    return SplatCloud(
        count      = cloud.count,
        positions  = new_positions,
        normals    = cloud.normals,
        colors_rgb = cloud.colors_rgb,
        f_dc       = cloud.f_dc,
        opacities  = cloud.opacities,
        scales     = cloud.scales,
        rotations  = cloud.rotations,
    )


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_clouds(clouds: list[SplatCloud]) -> SplatCloud:
    """Concatenate a list of SplatClouds into one."""
    total = sum(c.count for c in clouds)
    print(f"  Merging {len(clouds)} faces -> {total:,} total splats")
    return SplatCloud(
        count      = total,
        positions  = np.concatenate([c.positions  for c in clouds]),
        normals    = np.concatenate([c.normals    for c in clouds]),
        colors_rgb = np.concatenate([c.colors_rgb for c in clouds]),
        f_dc       = np.concatenate([c.f_dc       for c in clouds]),
        opacities  = np.concatenate([c.opacities  for c in clouds]),
        scales     = np.concatenate([c.scales     for c in clouds]),
        rotations  = np.concatenate([c.rotations  for c in clouds]),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def stitch_faces(
    face_plys:   dict[str, str | Path],
    face_images: dict[str, str | Path],
    focal_x:     float,
    focal_y:     float,
    use_depth_alignment: bool = True,
    grid_size:           int  = 8,
) -> SplatCloud:
    """
    Stitch 6 ml-sharp face .ply files into a single world-space SplatCloud.

    Args:
        face_plys:           dict face_name -> path to ml-sharp output .ply
        face_images:         dict face_name -> path to face image (for depth)
        focal_x:             horizontal focal length used during face extraction
        focal_y:             vertical focal length used during face extraction
        use_depth_alignment: whether to run Depth Anything V2 alignment
                             (disable for faster testing)
        grid_size:           NxN grid for depth alignment (default 8)

    Returns:
        Merged SplatCloud in world space, Y-up convention, ready for optimiser.
    """
    #quick checker
    print(f"focal_x={focal_x}, focal_y={focal_y}")

    expected = set(FACE_ROTATIONS.keys())
    #if set(face_plys.keys()) != expected:
    #    raise ValueError(f"Expected faces {expected}, got {set(face_plys.keys())}")
    #if set(face_images.keys()) != expected:
    #    raise ValueError(f"face_images must have same keys as face_plys")

    transformed = []
    reference_scale = None  # shared metric anchor across all faces

    for face_name in FACE_ROTATIONS:
        print(f"\n── Face: {face_name} ──")

        # 1. Parse .ply
        cloud = parse_ply(face_plys[face_name])

        # 2. Load face image
        face_img = np.array(Image.open(face_images[face_name]).convert("RGB"))
        ih, iw   = face_img.shape[:2]
        print(f"  Image: {iw}×{ih}")

        # 3. Depth alignment
        if use_depth_alignment:
            print("  Running Depth Anything V2...")
            depth_map = predict_depth(face_img)
            cloud, face_scale = align_splat_depths(
                cloud, depth_map, focal_x, focal_y, iw, ih, grid_size
            )

            # Anchor all faces to the same metric scale as the first face.
            # Depth Anything is only locally consistent - each face gets a
            # different arbitrary scale. Without this, faces sit at different
            # radii from the origin (the "cube exploded" effect).
            if reference_scale is None:
                reference_scale = face_scale
                print(f"  Reference scale set: {reference_scale:.4f}")
            else:
                match_scale = reference_scale / face_scale
                print(f"  Cross-face scale correction: {match_scale:.4f} "
                      f"(face={face_scale:.4f}, ref={reference_scale:.4f})")
                cloud = SplatCloud(
                    count      = cloud.count,
                    positions  = (cloud.positions * match_scale).astype(np.float32),
                    normals    = cloud.normals,
                    colors_rgb = cloud.colors_rgb,
                    f_dc       = cloud.f_dc,
                    opacities  = cloud.opacities,
                    scales     = (cloud.scales * match_scale).astype(np.float32),
                    rotations  = cloud.rotations,
                )
        else:
            # Simple median normalisation fallback
            median_z = float(np.median(cloud.positions[:, 2][cloud.positions[:, 2] > 1e-6]))
            sf = 1.0 / median_z if median_z > 1e-6 else 1.0
            cloud = SplatCloud(
                count      = cloud.count,
                positions  = (cloud.positions * sf).astype(np.float32),
                normals    = cloud.normals,
                colors_rgb = cloud.colors_rgb,
                f_dc       = cloud.f_dc,
                opacities  = cloud.opacities,
                scales     = (cloud.scales * sf).astype(np.float32),
                rotations  = cloud.rotations,
            )
            print(f"  Median normalised (scale={sf:.4f})")
            face_scale = sf

        # 4. Edge fade - attenuate splat opacity near face edges.
        # Soft fading preserves overlap continuity and blends seams smoothly.
        # Hard clipping (old Voronoi approach) created cracks and sparse edges.
        cloud = apply_edge_fade(cloud, focal_x, focal_y, iw, ih, fade_start=0.75)

        # 5. Rotate into world space
        cloud = transform_face(cloud, face_name)
        print(f"  Rotated into world space")

        transformed.append(cloud)

    # 6. Merge all faces
    print(f"\n── Merging ──")
    merged = merge_clouds(transformed)

    # 7. Flip Y: ml-sharp Y-down -> viewer Y-up
    merged = flip_y(merged)
    print("  Y-axis flipped (Y-down -> Y-up)")

    print(f"\n{'='*52}")
    print(f"Stitched cloud summary")
    print(f"{'='*52}")
    print(f"  Total splats : {merged.count:,}")
    print(f"  World-space bounds:")
    for i, label in enumerate(["x", "y", "z"]):
        lo = merged.positions[:, i].min()
        hi = merged.positions[:, i].max()
        print(f"    {label}  [{lo:.3f}, {hi:.3f}]")
    print(f"{'='*52}")

    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 14:
        print(
            "Usage: python splat_stitcher.py focal_x focal_y \\\n"
            "  front.ply back.ply right.ply left.ply top.ply bottom.ply \\\n"
            "  front.jpg back.jpg right.jpg left.jpg top.jpg bottom.jpg"
        )
        sys.exit(1)

    focal_x = float(sys.argv[1])
    focal_y = float(sys.argv[2])
    face_names = ['front', 'back', 'right', 'left', 'top', 'bottom']
    plys   = {n: sys.argv[3  + i] for i, n in enumerate(face_names)}
    images = {n: sys.argv[3+6+i] for i, n in enumerate(face_names)}

    cloud = stitch_faces(plys, images, focal_x, focal_y)
    print(f"\nDone. {cloud.count:,} splats ready for optimisation.")