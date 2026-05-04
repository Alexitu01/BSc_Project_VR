"""
splat_stitcher.py
-----------------
Stitches N ml-sharp .ply files (one per panorama slice) into a single
world-space SplatCloud ready for optimisation and export.

Unlike the cubemap approach, slices all face outward along the horizon
at different azimuths. Each slice's rotation matrix is computed
analytically from its azimuth angle - no hardcoded face table needed.
This means any number of slices works automatically.

Pipeline per slice:
  1. Parse .ply -> SplatCloud
  2. Run Depth Anything V2 on the slice image -> depth map
  3. Align splat depths to the depth map (radial distance, per-grid scale)
  4. Cross-face scale correction (anchor all slices to first slice's scale)
  5. Edge fade (attenuate opacity near slice edges for smooth blending)
  6. Rotate positions and orientations into world space (from azimuth angle)
  7. Merge all slices + Y-flip to match Unity/viewer Y-up convention

Usage:
    from splat_stitcher import stitch_slices
    cloud = stitch_slices(
        slices=[
            {"ply": "slice_000.ply", "image": "slice_000.png", "azimuth_deg": 0.0},
            {"ply": "slice_001.ply", "image": "slice_001.png", "azimuth_deg": 60.0},
            ...
        ],
        focal_x=412.0,
        focal_y=412.0,
    )

Dependencies:
    pip install numpy pillow transformers torch
"""

import math
import numpy as np
from pathlib import Path
from PIL import Image
from SplatParser import parse_ply, SplatCloud


PI     = math.pi
TWO_PI = 2.0 * PI


# ---------------------------------------------------------------------------
# Rotation matrix from azimuth angle
# ---------------------------------------------------------------------------

def rotation_matrix_from_azimuth(azimuth_deg: float) -> np.ndarray:
    """
    Build a 3x3 rotation matrix for a horizon-facing slice at the given azimuth.

    The slice camera uses ml-sharp convention:
        forward = [sin(az), 0, cos(az)]   - outward along horizon
        right   = [cos(az), 0, -sin(az)]  - rightward perpendicular
        down    = [0, 1, 0]               - Y-down (cuz images)

    The rotation matrix R has these as columns:
        R = [right | down | forward]

    So world_point = R @ local_point maps from the slice's local camera
    space into world space. This is the same convention used in the
    face extractor, so the two are guaranteed to be consistent.

    Args:
        azimuth_deg: horizontal angle in degrees
                     0° = looking along +Z (forward)
                     90° = looking along +X (right)
                     180° = looking along -Z (back)
                     270° = looking along -X (left)

    Returns:
        (3, 3) float64 rotation matrix
    """
    az = math.radians(azimuth_deg)

    forward = np.array([ math.sin(az), 0.0,  math.cos(az)], dtype=np.float64)
    right   = np.array([ math.cos(az), 0.0, -math.sin(az)], dtype=np.float64)
    down    = np.array([ 0.0,          1.0,  0.0          ], dtype=np.float64)

    # Columns = [right, down, forward]
    return np.column_stack([right, down, forward])


# ---------------------------------------------------------------------------
# DA360 depth predictor
# ---------------------------------------------------------------------------
#
# DA360 runs ONCE on the full panorama and produces a globally consistent
# disparity map. The stitcher samples this map at each slice's projected
# pixel locations - all slices reference the same depth map so cross-slice
# scale is consistent without needing the reference_scale hack.
#
# Usage: call load_da360() once before stitching, then pass the returned
# disparity map to stitch_slices() via the da360_disparity argument.

def load_da360(
    model_path: str,
    da360_root: str,
    device = None,
):
    """
    Load the DA360 predictor. Call once before stitching.

    Args:
        model_path: path to DA360 .pth checkpoint
        da360_root: path to cloned DA360 repository root
        device:     torch device (defaults to CUDA if available)

    Returns:
        DA360Predictor instance - call .predict(panorama_rgb) on it
    """
    from Da360Predictor import DA360Predictor
    return DA360Predictor(model_path, da360_root, device)


def sample_da360_for_slice(
    da360_disparity: np.ndarray,
    azimuth_deg:     float,
    focal_x:         float,
    focal_y:         float,
    image_width:     int,
    image_height:    int,
    pano_width:      int,
    pano_height:     int,
) -> np.ndarray:
    """
    Extract the DA360 disparity values(inverse depth valeus) for one perspective slice.

    DA360 produces a full panoramic disparity map. For each pixel in
    the slice output image, we compute which panorama pixel it came
    from (using the same perspective projection as the face extractor)
    and sample the DA360 map there.

    This gives a per-slice disparity map in slice-image coordinates
    that can be passed directly to align_splat_depths.

    essentially it's extract_slice but for depth values instead of colours.

    Args:
        da360_disparity: (H_pano, W_pano) float32 panoramic disparity
        azimuth_deg:     slice centre azimuth in degrees
        focal_x:         horizontal focal length in pixels
        focal_y:         vertical focal length in pixels
        image_width:     slice image width
        image_height:    slice image height
        pano_width:      full panorama width
        pano_height:     full panorama height

    Returns:
        (image_height, image_width) float32 disparity map for this slice
    """
    from scipy.ndimage import map_coordinates #Radians, just like in Face extractor
    az = math.radians(azimuth_deg)

    # Camera axes - basically identical same face extractor
    forward = np.array([ math.sin(az), 0.0,  math.cos(az)], dtype=np.float64)
    right   = np.array([ math.cos(az), 0.0, -math.sin(az)], dtype=np.float64)
    down    = np.array([ 0.0,          1.0,  0.0          ], dtype=np.float64)
    R       = np.column_stack([right, down, forward])

    # Pixel grid
    u  = np.arange(image_width,  dtype=np.float64) + 0.5
    v  = np.arange(image_height, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v, indexing="xy")

    cx      = image_width  / 2.0
    cy      = image_height / 2.0
    local_x = (uu - cx) / focal_x
    local_y = (vv - cy) / focal_y
    local_z = np.ones_like(uu)

    wx = R[0,0]*local_x + R[0,1]*local_y + R[0,2]*local_z
    wy = R[1,0]*local_x + R[1,1]*local_y + R[1,2]*local_z
    wz = R[2,0]*local_x + R[2,1]*local_y + R[2,2]*local_z

    norm = np.maximum(np.sqrt(wx**2 + wy**2 + wz**2), 1e-12)
    wx /= norm; wy /= norm; wz /= norm

    longitude = np.arctan2(wx, wz)
    latitude  = np.arcsin(np.clip(wy, -1.0, 1.0))

    sample_x = (longitude / TWO_PI + 0.5) * pano_width  - 0.5 
    sample_y = (latitude  / PI     + 0.5) * pano_height - 0.5

    # This is where it differs, we don't sample RGB values but disparity values from the DA360 map as flaots in a 2D array matching the slice size
    slice_disp = map_coordinates(
        da360_disparity,
        [sample_y, sample_x],
        order=1,
        mode='wrap',
    ).astype(np.float32)

    return slice_disp


# ---------------------------------------------------------------------------
# Depth alignment
# ---------------------------------------------------------------------------

def align_splat_depths_da360(
    cloud,
    da360_disp,
    focal_x,
    focal_y,
    image_width,
    image_height,
    grid_size=8,
):
    """
    Align splat depths using DA360 raw disparity.

    DA360 outputs disparity where larger = closer.
    ml-sharp outputs positions where larger z = farther.

    Correct scale: scale = 1 / (disparity * radial)
    This maps ml-sharp radial distances to DA360 metric depth space,
    making all slices share the same consistent scale automatically.
    """
    positions = cloud.positions.astype(float)
    depth_z   = positions[:, 2]
    N         = len(positions)

    valid = depth_z > 1e-6
    if valid.sum() < 32:
        return cloud, 1.0

    cx     = image_width  / 2.0
    cy     = image_height / 2.0
    safe_z = np.where(valid, depth_z, 1.0)
    px     = (positions[:, 0] / safe_z) * focal_x + cx
    py     = (positions[:, 1] / safe_z) * focal_y + cy

    in_bounds = (
        valid &
        (px >= 0) & (px < image_width) &
        (py >= 0) & (py < image_height)
    )

    if in_bounds.sum() < 32:
        return cloud, 1.0

    splat_radial = np.linalg.norm(positions[in_bounds], axis=1)

    from scipy.ndimage import map_coordinates

    dh, dw = da360_disp.shape
    if dh != image_height or dw != image_width:
        disp_pil = Image.fromarray(da360_disp).resize(
            (image_width, image_height), Image.BILINEAR
        )
        disp_r = np.array(disp_pil).astype(np.float32)
    else:
        disp_r = da360_disp

    # Bilinear sampling - critical at Voronoi boundaries where adjacent
    # slices must agree on disparity values. Nearest-neighbour snapping
    # creates inconsistent scale at exactly the seam location.
    ref_disp = map_coordinates(
        disp_r,
        [py[in_bounds], px[in_bounds]],
        order=1,
        mode='nearest',
    ).astype(np.float32)

    valid_ratio = (ref_disp > 1e-4) & (splat_radial > 1e-6)
    if valid_ratio.sum() < 32:
        print("    Warning: insufficient DA360 samples, skipping alignment")
        return cloud, 1.0

    # DA360 gives disparity not depth, meaning larger is closer (depth = 1 / disparity)
    raw_scales = 1.0 / (ref_disp[valid_ratio] * splat_radial[valid_ratio])

    lo, hi  = np.quantile(raw_scales, [0.05, 0.95])
    trimmed = raw_scales[(raw_scales >= lo) & (raw_scales <= hi)]
    global_scale = float(np.median(trimmed)) if trimmed.size > 0 \
                   else float(np.median(raw_scales))

    px_valid = px[in_bounds][valid_ratio]
    py_valid = py[in_bounds][valid_ratio]

    # Aspect-ratio-aware grid
    # For a 396x672 slice: grid_cols=8, grid_rows=round(8*672/396)=14
    # This gives ~50px cells vertically instead of ~84px, improving vertical accuracy
    grid_cols = grid_size
    grid_rows = max(1, round(grid_size * image_height / image_width))
    cell_w    = image_width  / grid_cols
    cell_h    = image_height / grid_rows
    grid      = np.full((grid_rows, grid_cols), global_scale, dtype=float)

    for gy in range(grid_rows):
        for gx in range(grid_cols):
            in_cell = (
                (px_valid >= gx * cell_w) & (px_valid < (gx+1) * cell_w) &
                (py_valid >= gy * cell_h) & (py_valid < (gy+1) * cell_h)
            )
            if in_cell.sum() >= 4:
                cs = raw_scales[in_cell]
                cl, ch = np.quantile(cs, [0.1, 0.9])
                ct = cs[(cs >= cl) & (cs <= ch)]
                if ct.size > 0:
                    grid[gy, gx] = float(np.median(ct))

    # 0.1x to 10x range, less usually leads to flat looking geometry, (if seam issues pop up try lowering this)
    grid = np.clip(grid, global_scale * 0.1, global_scale * 10.0)

    all_px = np.clip(px, 0, image_width  - 1)
    all_py = np.clip(py, 0, image_height - 1)
    gxc    = all_px / cell_w - 0.5
    gyc    = all_py / cell_h - 0.5
    gx0    = np.clip(np.floor(gxc).astype(np.int32), 0, grid_cols - 1)
    gy0    = np.clip(np.floor(gyc).astype(np.int32), 0, grid_rows - 1)
    gx1    = np.clip(gx0 + 1, 0, grid_cols - 1)
    gy1    = np.clip(gy0 + 1, 0, grid_rows - 1)
    wx     = np.clip(gxc - gx0, 0.0, 1.0)
    wy     = np.clip(gyc - gy0, 0.0, 1.0)

    per_splat_scale = (
        grid[gy0, gx0] * (1-wx) * (1-wy) +
        grid[gy0, gx1] * wx     * (1-wy) +
        grid[gy1, gx0] * (1-wx) * wy     +
        grid[gy1, gx1] * wx     * wy
    ).astype(np.float32)

    print(f"    DA360 aligned: global_scale={global_scale:.6f}  "
          f"range=[{per_splat_scale.min():.6f}, {per_splat_scale.max():.6f}]")

    return SplatCloud(
        count      = cloud.count,
        positions  = (positions * per_splat_scale[:, None]).astype(np.float32),
        normals    = cloud.normals,
        colors_rgb = cloud.colors_rgb,
        f_dc       = cloud.f_dc,
        opacities  = cloud.opacities,
        scales     = (cloud.scales * per_splat_scale[:, None]).astype(np.float32),
        rotations  = cloud.rotations,
    ), global_scale



def voronoi_clip(cloud: SplatCloud, span_degrees: float) -> SplatCloud:
    """
    Hard Voronoi clipping, keep only splats within the natural span, reduces duplicate geometry.

    Clips to exactly span_degrees horizontal FOV (e.g. 60° for 6 slices).
    Done BEFORE rotation so we work in view-local space where depth is +Z
    and horizontal angle is simply atan2(x, z).
    Because DA360 makes adjacent slices geometrically consistent, hard
    clipping at the natural span boundary *shouldn't* create depth discontinuities.

    Args:
        cloud:        SplatCloud in view-local coordinates
        span_degrees: natural horizontal span (360 / n_slices)

    Returns:
        Clipped SplatCloud with overlap region removed
    """
    positions = cloud.positions
    depth_z   = positions[:, 2]
    in_front  = depth_z > 1e-6

    # Keep splats within ±half_span of the forward direction
    half = span_degrees / 2.0
    h_limit = math.tan(math.radians(half))
    h_ratio = np.abs(positions[:, 0]) / np.where(in_front, depth_z, 1.0)
    mask = in_front & (h_ratio <= h_limit)

    #Debug info printed to terminal
    n_before = cloud.count
    n_after  = int(mask.sum())
    print(f"    Voronoi clip ({span_degrees:.0f}°): {n_before:,} -> {n_after:,} "
          f"({100*(n_before-n_after)/n_before:.1f}% removed)")

    return SplatCloud(
        count      = n_after,
        positions  = cloud.positions [mask],
        normals    = cloud.normals   [mask],
        colors_rgb = cloud.colors_rgb[mask],
        f_dc       = cloud.f_dc      [mask],
        opacities  = cloud.opacities [mask],
        scales     = cloud.scales    [mask],
        rotations  = cloud.rotations [mask],
    )


# ---------------------------------------------------------------------
# Rotation helpers
# --------------------------------------------------------------------
# this but dumber https://www.johndcook.com/blog/2025/05/07/quaternions-and-rotation-matrices/ 

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


def transform_slice(cloud: SplatCloud, azimuth_deg: float) -> SplatCloud:
    """
    Rotate a slice's SplatCloud from local camera space into world space.

    The rotation matrix is computed analytically from the azimuth angle -
    no hardcoded face table. This works for any number of slices.

    Rotates both positions and orientation quaternions so splat ellipsoids
    point correctly in world space.
    """
    R = rotation_matrix_from_azimuth(azimuth_deg)

    # Rotate positions: R @ each position vector
    new_positions = (R @ cloud.positions.T).T.astype(np.float32)

    # Compose slice rotation with each splat's own orientation
    q_slice       = matrix_to_quaternion(R)
    new_rotations = quaternion_multiply(q_slice, cloud.rotations)

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
    Negate Y axis: ml-sharp Y-down -> viewer/Unity Y-up convention.
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


# ------------------------------------------------------------------------
# Merge
# ------------------------------------------------------------------------

def merge_clouds(clouds: list[SplatCloud]) -> SplatCloud:
    """Concatenate a list of SplatClouds into one."""
    total = sum(c.count for c in clouds)
    print(f"  Merging {len(clouds)} slices -> {total:,} total splats")
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

def stitch_slices(
    slices:              list[dict],
    focal_x:             float,
    focal_y:             float,
    pano_width:          int   = None,
    pano_height:         int   = None,
    da360_disparity:     np.ndarray = None,
    use_depth_alignment: bool  = True,
    grid_size:           int   = 8,
    fade_start:          float = 0.7,
) -> SplatCloud:
    """
    Stitch N ml-sharp slice .ply files into a single world-space SplatCloud.

    Args:
        slices: list of dicts, one per slice, each containing:
                    ply         - path to ml-sharp output .ply
                    image       - path to slice image
                    azimuth_deg - centre azimuth in degrees (from face_extractor)
        focal_x:             horizontal focal length (from face_extractor)
        focal_y:             vertical focal length (from face_extractor)
        pano_width:          original panorama width (needed for DA360 sampling)
        pano_height:         original panorama height (needed for DA360 sampling)
        da360_disparity:     (H, W) float32 panoramic disparity from DA360.
                             If provided, used for globally consistent depth
                             alignment instead of per-slice Depth Anything V2.
                             Larger values = closer to camera.
        use_depth_alignment: enable depth alignment (disable for quick testing)
        grid_size:           NxN grid for depth alignment (default 8)
        fade_start:          edge fade start distance (default 0.7)

    Returns:
        Merged SplatCloud in world space, Y-up, ready for optimiser.
    """
    transformed         = []
    original_median_radii = []  # for global scale restore

    first_img = Image.open(slices[0]["image"]) 
    iw, ih = first_img.size #all images should have same size

    for i, s in enumerate(slices):
        az  = s["azimuth_deg"]
        print(f"\n── Slice {i} (azimuth={az:.1f}°) ──")

        # 1. Parse .ply
        cloud = parse_ply(s["ply"])

        # 2. Depth alignment
        # Since ML-sharp is unaware of the other slices each slice share no consitent size or depth, we use DA360 to mitigate this
        # Store pre-alignment radius BEFORE any depth scaling.
        # DA360 shrinks everything to (~0.006x unit) so we record ML-sharps metric space,
        # and we restore after merging.
        pre_align_radii = np.linalg.norm(cloud.positions, axis=1)
        original_median_radii.append(float(np.median(pre_align_radii)))
        print(f"  Pre-alignment median radius: {original_median_radii[-1]:.4f}")

        if use_depth_alignment:
            if da360_disparity is not None:
                # DA360 path: globally consistent disparity from panorama.
                # Sample the panoramic disparity map at this slice's pixel
                # locations. All slices reference the same map so cross-slice
                print("  Sampling DA360 disparity for this slice...")
                depth_map = sample_da360_for_slice(
                    da360_disparity,
                    az, focal_x, focal_y,
                    iw, ih, pano_width, pano_height,
                )
                # DA360 returns raw disparity (larger=closer).
                # We use it directly - align_splat_depths computes:
                #   scale = ref_depth / splat_radial
                # For DA360 disparity we instead compute:
                #   scale = splat_radial / (1/ref_disparity)
                #         = splat_radial * ref_disparity
                # which gives the correct scale to map ml-sharp's arbitrary
                # units to DA360's consistent metric space.
                # Implement this by passing disparity and flipping the ratio.
                cloud, _ = align_splat_depths_da360(
                    cloud, depth_map, focal_x, focal_y, iw, ih, grid_size
                )

                # Per-slice scale restore, bring each slice back to its own
                # pre-alignment scale immediately after DA360 alignment.
                # This makes adjacent slices geometrically consistent because
                # each slice's depth structure matches its pre-alignment shape
                # before rotation into world space.
                # Without this, cross-slice DA360 scale variation (~48% range)
                # causes duplicate geometry at seams even after global restore.
                post_align_median = float(np.median(
                    np.linalg.norm(cloud.positions, axis=1)
                ))
                if post_align_median > 1e-8:
                    per_slice_restore = original_median_radii[-1] / post_align_median
                    print(f"  Per-slice restore: x{per_slice_restore:.4f} "
                          f"({post_align_median:.4f} -> {original_median_radii[-1]:.4f})")
                    cloud = SplatCloud(
                        count      = cloud.count,
                        positions  = (cloud.positions * per_slice_restore).astype(np.float32),
                        normals    = cloud.normals,
                        colors_rgb = cloud.colors_rgb,
                        f_dc       = cloud.f_dc,
                        opacities  = cloud.opacities,
                        scales     = (cloud.scales * per_slice_restore).astype(np.float32),
                        rotations  = cloud.rotations,
                    )
        else:
            # No depth alignment - simple median normalisation fall back (This is purely for testing and not a viable second option)
            print(f" DA360 failed to load, basic median normalisation fallback, if this is not testing: check paths and retry")
            median_z = float(np.median(
                cloud.positions[:, 2][cloud.positions[:, 2] > 1e-6]
            ))
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

        # 3. Hard Voronoi clipping - remove overlap region before rotation.
        # DA360 alignment makes adjacent slices consistent so hard clipping
        # doesn't* create depth discontinuities at seams.
        span_deg = 360.0 / len(slices)
        cloud = voronoi_clip(cloud, span_deg)

        # 4. Rotate into world space using azimuth angle
        cloud = transform_slice(cloud, az)
        print(f"  Rotated into world space")

        transformed.append(cloud)

    # 5. Merge all slices
    print(f"\n── Merging ──")
    merged = merge_clouds(transformed)

    # 6. Global scale restore.
    # DA360 alignment shrinks the scene to metric disparity scale (~0.006).
    # Restore to the original pre-alignment median radius so the scene
    # is the same size as ml-sharp originally produced. 
    # (different to step 3's restore which primary function was uniformity for easier merging, could you do it one step? prolly idk. This works)
    if original_median_radii:
        original_scene_median = float(np.median(original_median_radii))
        current_median = float(np.median(np.linalg.norm(merged.positions, axis=1)))
        if current_median > 1e-8:
            global_restore = original_scene_median / current_median
            print(f"  Global scale restore: x{global_restore:.4f} "
                  f"({current_median:.4f} -> {original_scene_median:.4f} median radius)")
            merged = SplatCloud(
                count      = merged.count,
                positions  = (merged.positions * global_restore).astype(np.float32),
                normals    = merged.normals,
                colors_rgb = merged.colors_rgb,
                f_dc       = merged.f_dc,
                opacities  = merged.opacities,
                scales     = (merged.scales * global_restore).astype(np.float32),
                rotations  = merged.rotations,
            )

    # 7. Y-flip: ml-sharp Y-down -> viewer Y-up if we don't do this the scene will appear upside down in most renderers and require manual flipping,
    merged = flip_y(merged)
    print("  Y-axis flipped (Y-down -> Y-up)")

    # Final merged cloud info 
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
# This is mainly for testing you should use RunStitcher.py instead

if __name__ == "__main__":
    import sys

    # Read metadata file written by face_extractor.py
    # Usage: python splat_stitcher.py slices_meta.txt focal_length ply_folder/
    if len(sys.argv) < 5:
        print(
            "Usage: python splat_stitcher.py slices_meta.txt focal_x focal_y ply_folder/\n"
            "  slices_meta.txt - written by face_extractor.py\n"
            "  focal_x         - horizontal focal length (printed by face_extractor.py)\n"
            "  focal_y         - vertical focal length (printed by face_extractor.py)\n"
            "  ply_folder/     - folder containing ml-sharp .ply outputs"
        )
        sys.exit(1)

    meta_path  = Path(sys.argv[1])
    focal_x    = float(sys.argv[2])
    focal_y    = float(sys.argv[3])
    ply_folder = Path(sys.argv[4])

    # Parse metadata file
    slices = []
    for line in meta_path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        name, azimuth_deg, img_name = parts[0], float(parts[1]), parts[2]
        slices.append({
            "ply":         ply_folder / f"{name}.ply",
            "image":       meta_path.parent / img_name,
            "azimuth_deg": azimuth_deg,
        })

    from splat_exporter import export_ply
    cloud = stitch_slices(slices, focal_x, focal_y)
    export_ply(cloud, "stitched_output.ply")
    print(f"\nDone. {cloud.count:,} splats -> stitched_output.ply")