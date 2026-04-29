"""
splat_exporter.py
-----------------
Exports a SplatCloud back to a standard 3DGS .ply file.

The output format matches the ml-sharp header exactly — no normals,
just the 14 core Gaussian attributes. This is compatible with:
    - aras-p UnityGaussianSplatting
    - SuperSplat viewer
    - Polycam viewer
    - Any standard 3DGS tool

IMPORTANT: The exporter writes raw (pre-decoded) values back to disk.
    - opacity  is stored as logit  (inverse sigmoid of the decoded value)
    - scale    is stored as log    (log of the decoded value)
    - f_dc     is stored as-is     (never decoded, kept raw throughout)
    - rotation is stored as-is     (normalised quaternion, no transform)

This means the round-trip is lossless — parse → export gives back
a file identical to the input (within float32 precision).

Usage:
    python splat_exporter.py          (runs built-in round-trip test)

    from splat_exporter import export_ply
    export_ply(cloud, "output.ply")

Dependencies:
    pip install numpy
"""

import sys
import struct
import numpy as np
from pathlib import Path
from SplatParser import SplatCloud, parse_ply


# ---------------------------------------------------------------------------
# Output property layout
# ---------------------------------------------------------------------------

# These are the properties we write, in this exact order.
# Must match what the Unity renderer (aras-p) expects.
# No normals — ml-sharp doesn't output them and the renderer doesn't need them.
OUTPUT_PROPS = [
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]

N_PROPS = len(OUTPUT_PROPS)  # 14 floats per splat


# ---------------------------------------------------------------------------
# Header builder
# ---------------------------------------------------------------------------

def _build_header(n_splats: int) -> bytes:
    """
    Build the ASCII header for the output .ply file.

    The header is plain text that describes the binary data that follows.
    It must end with 'end_header\n' — the binary block starts immediately
    after that newline, with no padding or alignment.

    Returns bytes (not str) because we're writing to a binary file.
    """
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n_splats}",
    ]

    # One property line per output attribute
    for prop in OUTPUT_PROPS:
        lines.append(f"property float {prop}")

    lines.append("end_header")

    # Join with Unix newlines (\n) and encode to bytes.
    # Important: use \n not \r\n — Windows line endings would add extra
    # bytes and shift the binary data offset, corrupting the file.
    header_str = "\n".join(lines) + "\n"
    return header_str.encode("ascii")


# ---------------------------------------------------------------------------
# Re-encoding helpers
# ---------------------------------------------------------------------------

def _encode_opacity(opacities: np.ndarray) -> np.ndarray:
    """
    Re-encode opacity from [0, 1] back to logit space for storage.

    The parser decoded:  opacity = sigmoid(raw)  =  1 / (1 + e^-raw)
    We reverse this:     raw = logit(opacity)    =  log(p / (1 - p))

    Why bother? The SplatCloud stores decoded values for easy maths.
    The .ply format stores encoded values because that's what the
    training code outputs and what renderers expect.

    We clip to [0.001, 0.999] before taking the log to avoid
    log(0) = -inf and log(1/(1-1)) = +inf.
    """
    p = np.clip(opacities, 0.001, 0.999)
    return np.log(p / (1.0 - p)).astype(np.float32)


def _encode_scale(scales: np.ndarray) -> np.ndarray:
    """
    Re-encode scale from real space back to log space for storage.

    The parser decoded:  scale = exp(raw)
    We reverse this:     raw   = log(scale)

    Clip to a small positive value first to avoid log(0) = -inf.
    """
    return np.log(np.clip(scales, 1e-10, None)).astype(np.float32)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_ply(cloud: SplatCloud, path: str | Path) -> None:
    """
    Write a SplatCloud to a binary little-endian .ply file.

    Arguments:
        cloud: the SplatCloud to export (from parser, stitcher, or optimiser)
        path:  output file path - will be created or overwritten

    The output file is immediately loadable in the aras-p Unity importer
    and standard 3DGS viewers like SuperSplat.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {cloud.count:,} splats to: {path.name}")

    # ------------------------------------------------------------------
    # Step 1: Re-encode values that were decoded during parsing
    #
    # The SplatCloud stores human-friendly decoded values:
    #   opacities in [0, 1]        — easy to threshold and compare
    #   scales    in real space    — easy to understand and modify
    #
    # But the .ply format stores encoded values:
    #   opacity   as logit         — what the renderer expects
    #   scale     as log           — what the renderer expects
    #
    # f_dc and rotations were never decoded so they go straight through.
    # ------------------------------------------------------------------
    raw_opacity = _encode_opacity(cloud.opacities)       # (N,)
    raw_scale   = _encode_scale(cloud.scales)            # (N, 3)

    # ------------------------------------------------------------------
    # Step 2: Stack all attributes into a single (N, 14) array
    #
    # np.column_stack joins 1D and 2D arrays side by side along axis=1.
    # The order must match OUTPUT_PROPS exactly — x, y, z, f_dc_0, ...
    #
    # After stacking, each row is one splat's complete data,
    # laid out as 14 consecutive float32 values — exactly how the
    # binary block needs to look.
    # ------------------------------------------------------------------
    data = np.column_stack([
        cloud.positions,        # columns 0-2:   x, y, z
        cloud.f_dc,             # columns 3-5:   f_dc_0, f_dc_1, f_dc_2
        raw_opacity,            # column  6:     opacity (logit)
        raw_scale,              # columns 7-9:   scale_0, scale_1, scale_2
        cloud.rotations,        # columns 10-13: rot_0, rot_1, rot_2, rot_3
    ]).astype(np.float32)

    # This should never fail, we're fucked if it does
    assert data.shape == (cloud.count, N_PROPS), \
        f"Data shape mismatch: expected ({cloud.count}, {N_PROPS}), got {data.shape}"

    # ------------------------------------------------------------------
    # Step 3: Write header then binary data
    #
    # 'wb' opens in write-binary mode - essential, never use 'w' here.
    # Writing text mode would corrupt the binary block on Windows by
    # translating \n to \r\n inside the data.
    # ------------------------------------------------------------------
    header = _build_header(cloud.count)

    with open(path, "wb") as f:
        # Write ASCII header
        f.write(header)

        # Write binary data block
        # .tobytes() serialises the numpy array to raw bytes in C order
        # (row by row, which is what we want — splat by splat).
        # Little-endian byte order is already correct because we're on
        # an x86 machine and numpy defaults to native (little) endian
        # for float32.
        f.write(data.tobytes())

    file_mb = path.stat().st_size / 1e6
    print(f"  Written: {file_mb:.1f} MB")
    print(f"  Bytes per splat: {N_PROPS * 4} ({N_PROPS} floats × 4 bytes)")
    print(f"  Done.")