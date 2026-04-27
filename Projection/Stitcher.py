"""
splat_stitcher.py
-----------------
Takes 6 ml-sharp .ply files (one per cube face) and merges them into
a single world-space SplatCloud ready for optimisation and export.

Key findings from measured data:
- ml-sharp always outputs splats looking down +z regardless of face
- Each face has a different depth scale (front z=151, side faces z=6)
- ml-sharp uses a left-handed coordinate system (x points left, y points down)
- Depth normalisation is required before rotation to get seams to meet

Usage:
    python splat_stitcher.py front.ply back.ply right.ply left.ply top.ply bottom.ply

Dependencies:
    pip install numpy
"""

import numpy as np
from pathlib import Path
from SplatParser import parse_ply, SplatCloud


# ---------------------------------------------------------------------------
# Rotation matrix builders
# ---------------------------------------------------------------------------

def _rot_x(degrees: float) -> np.ndarray:
    """
    3×3 rotation matrix around the X axis (pitch).
    Positive degrees tilts the scene upward.
    """
    r = np.radians(degrees)
    c, s = np.cos(r), np.sin(r)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c],
    ], dtype=np.float32)


def _rot_y(degrees: float) -> np.ndarray:
    """
    3×3 rotation matrix around the Y axis (yaw).
    Positive degrees turns the scene to the left.
    """
    r = np.radians(degrees)
    c, s = np.cos(r), np.sin(r)
    return np.array([
        [ c,  0,  s],
        [ 0,  1,  0],
        [-s,  0,  c],
    ], dtype=np.float32)


def _identity() -> np.ndarray:
    return np.eye(3, dtype=np.float32)


# ---------------------------------------------------------------------------
# Face rotation definitions
# ---------------------------------------------------------------------------
#
# Corrected for ml-sharp's left-handed coordinate system.
# Signs are flipped vs a standard right-handed system on left/right/top/bottom.
# Derived from measured position data — not assumed from convention.

FACE_ROTATIONS: dict[str, np.ndarray] = {
    'front':  _identity(),
    'back':   _rot_y(180),
    'right':  _rot_y(90),    # left-handed: positive turns right
    'left':   _rot_y(-90),   # left-handed: negative turns left
    'top':    _rot_x(90),    # left-handed: positive tilts up
    'bottom': _rot_x(-90),   # left-handed: negative tilts down
}


# ---------------------------------------------------------------------------
# Depth normalisation
# ---------------------------------------------------------------------------

def normalise_depth(cloud: SplatCloud) -> tuple[SplatCloud, float]:
    """
    Scale all positions so the median scene depth (z) equals 1.0.

    Why this is necessary:
    ml-sharp produces each face at a different depth scale. The front face
    may extend to z=151 while side faces only reach z=6. Without normalisation,
    rotating a side face 90° puts it at x=6 while the front face extends to
    z=151 — the seams don't meet and you get floating islands.

    By normalising each face to median z=1.0 before rotation, all faces
    end up in the same depth ballpark and seams connect properly.

    We also scale the splat sizes (scales array) proportionally — a splat
    that was 0.01 units wide in a z=100 scene should remain visually the
    same size after normalisation, which means scaling it down by the same
    factor as the positions.

    Returns:
        normalised cloud  — positions and scales rescaled
        scale_factor      — what we multiplied by (1 / median_z)
                            useful for debugging, not needed downstream
    """
    # Use median rather than mean — robust to outlier splats that ended up
    # very far away during reconstruction
    median_z = np.median(cloud.positions[:, 2])

    if median_z <= 0:
        raise ValueError(
            f"Median z depth is {median_z:.3f} — scene appears to be "
            f"behind the camera. Check ml-sharp output for this face."
        )

    scale_factor = 1.0 / median_z

    return SplatCloud(
        count      = cloud.count,
        positions  = (cloud.positions * scale_factor).astype(np.float32),
        normals    = cloud.normals,
        colors_rgb = cloud.colors_rgb,
        f_dc       = cloud.f_dc,
        opacities  = cloud.opacities,
        scales     = (cloud.scales * scale_factor).astype(np.float32),
        rotations  = cloud.rotations,
    ), scale_factor


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3×3 rotation matrix to a unit quaternion (w, x, y, z).

    Uses Shepperd's method — picks the numerically stable formula based
    on which diagonal element of R is largest, avoiding division by
    near-zero values.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Compose a single quaternion q1 with N quaternions q2.

    q1: shape (4,)   — face rotation quaternion
    q2: shape (N, 4) — all splat orientation quaternions for one face

    Returns shape (N, 4) — each splat's orientation after applying
    the face rotation on top of its own local orientation.

    Order: q1 × q2 means "apply q2 first, then q1" — which is what
    we want: apply the splat's own tilt, then rotate into world space.
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]

    result = np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
    ], axis=1)

    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return (result / np.where(norms > 0, norms, 1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Single face transform
# ---------------------------------------------------------------------------

def transform_face(cloud: SplatCloud, face_name: str) -> SplatCloud:
    """
    Rotate a single face's SplatCloud from local +z space into world space.

    Two things are rotated:
    1. Positions  — moves each splat to the right place in world space
    2. Rotations  — keeps each splat's gaussian ellipsoid pointing correctly

    Positions are multiplied by the 3×3 rotation matrix directly.
    Rotations are composed via quaternion multiplication.

    Colors, opacity, and scale magnitudes are unchanged.
    """
    R = FACE_ROTATIONS[face_name]

    # Rotate positions: (R @ positions.T).T applies R to every row
    new_positions = (R @ cloud.positions.T).T.astype(np.float32)

    # Convert face rotation to quaternion for composing with splat rotations
    q_face = matrix_to_quaternion(R)

    # Compose face rotation with each splat's own orientation
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


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_clouds(clouds: list[SplatCloud]) -> SplatCloud:
    """
    Concatenate a list of SplatClouds into one along the splat axis.

    No blending or deduplication — just stacking all arrays.
    The optimiser handles reduction after this.
    """
    total = sum(c.count for c in clouds)
    print(f"  Merging {len(clouds)} faces → {total:,} total splats")

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

def stitch_faces(face_paths: dict[str, str | Path]) -> SplatCloud:
    """
    Load 6 face .ply files, normalise depth, rotate into world space, merge.

    Arguments:
        face_paths: dict mapping face name → path to .ply file
                    Keys must be: 'front', 'back', 'right', 'left',
                                  'top', 'bottom'

    Returns:
        A single merged SplatCloud in world space, ready for optimisation.
    """
    expected = set(FACE_ROTATIONS.keys())
    provided = set(face_paths.keys())
    if provided != expected:
        raise ValueError(
            f"Face mismatch.\n"
            f"  Missing:    {expected - provided}\n"
            f"  Unexpected: {provided - expected}"
        )

    transformed = []

    for face_name, path in face_paths.items():
        print(f"\n── Face: {face_name} ──")

        # Parse .ply → SplatCloud (local +z space)
        cloud = parse_ply(path)

        # Normalise depth so all faces sit at comparable scale
        cloud, sf = normalise_depth(cloud)
        print(f"  Depth normalised  (scale factor: {sf:.4f})")

        # Rotate into world space
        cloud = transform_face(cloud, face_name)
        print(f"  Rotated into world space")

        transformed.append(cloud)

    # Merge all faces
    print(f"\n── Merging ──")
    merged = merge_clouds(transformed)

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
    face_names = ['front', 'back', 'right', 'left', 'top', 'bottom']

    if len(sys.argv) < 7:
        print(
            "Usage: python splat_stitcher.py "
            "front.ply back.ply right.ply left.ply top.ply bottom.ply"
        )
        sys.exit(1)

    paths = {name: sys.argv[i+1] for i, name in enumerate(face_names)}
    cloud = stitch_faces(paths)
    print(f"\nStitching complete. {cloud.count:,} splats ready for optimisation.")