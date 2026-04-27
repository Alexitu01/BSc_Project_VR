"""
splat_parser.py
---------------
Parses a 3D Gaussian Splatting .ply file.

Handles two formats:
  - Original format (with normals nx/ny/nz)
  - ml-sharp format (no normals, has extra metadata elements)

Both produce the same SplatCloud dataclass for downstream processing.

Usage:
    python splat_parser.py path/to/scene.ply

Dependencies:
    pip install numpy
"""

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# SH degree-0 constant
# ---------------------------------------------------------------------------

# Spherical harmonics degree-0 basis function value.
# This is the mathematical constant that converts a raw f_dc coefficient
# into a visible RGB colour: colour = 0.5 + SH_C0 * f_dc
SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class SplatCloud:
    """
    All decoded attributes for every Gaussian splat in the scene.

    Every array has N rows where N = number of splats.

    positions   (N, 3)  float32  — world-space x, y, z
    colors_rgb  (N, 3)  float32  — RGB in [0, 1] decoded from f_dc
    f_dc        (N, 3)  float32  — raw SH degree-0 coefficients (kept for export)
    opacities   (N,)    float32  — [0, 1] after sigmoid decode
    scales      (N, 3)  float32  — real scale after exp() decode
    rotations   (N, 4)  float32  — normalised quaternion (w, x, y, z)
    normals     (N, 3)  float32  — optional, zeros if not present in file
    """
    count:      int
    positions:  np.ndarray
    colors_rgb: np.ndarray
    f_dc:       np.ndarray
    opacities:  np.ndarray
    scales:     np.ndarray
    rotations:  np.ndarray
    # Normals are optional — ml-sharp doesn't output them
    # We store zeros so downstream code can always expect this field to exist
    normals:    np.ndarray = field(default=None)

    def __post_init__(self):
        if self.normals is None:
            self.normals = np.zeros((self.count, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# Core vertex properties — present in both formats
# ---------------------------------------------------------------------------

# These are the properties we always expect to find, in this exact order,
# in the binary data. The order matters because we read raw bytes and
# interpret them positionally.
CORE_PROPS = [
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]

# The original WorldGen format also has normals before f_dc
PROPS_WITH_NORMALS = [
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]


# ---------------------------------------------------------------------------
# Header parser
# ---------------------------------------------------------------------------

def _parse_header(f) -> tuple[int, int, list[str]]:
    """
    Read and interpret the ASCII header from an open binary file.

    Returns:
        n_splats      — how many splats are in the file
        header_bytes  — how many bytes the header occupies
                        (so we know where binary data starts)
        vertex_props  — list of property names found in the vertex element
                        (order matches the binary layout)

    The header is plain ASCII text that ends with 'end_header'.
    Everything after that is raw binary data.
    """
    header_lines = []
    while True:
        line = f.readline().decode("ascii").strip()
        header_lines.append(line)
        if line == "end_header":
            break

    # Basic format checks
    if header_lines[0] != "ply":
        raise ValueError("Not a valid .ply file — first line must be 'ply'")
    if "binary_little_endian" not in header_lines[1]:
        raise ValueError(
            f"Expected binary_little_endian encoding, got: {header_lines[1]}\n"
            f"ASCII .ply files are not supported."
        )

    # Walk through header lines collecting what we need.
    # We track which 'element' we're currently inside because a .ply file
    # can have multiple elements (vertex, face, extrinsic, etc.) each with
    # their own property lists. We only want the vertex properties.
    n_splats = None
    vertex_props = []
    current_element = None

    for line in header_lines:
        if line.startswith("element vertex"):
            # "element vertex 1179648" → split on spaces, last part is count
            n_splats = int(line.split()[-1])
            current_element = "vertex"
        elif line.startswith("element"):
            # Any other element (extrinsic, intrinsic, etc.) — stop collecting
            current_element = line.split()[1]
        elif line.startswith("property") and current_element == "vertex":
            # "property float x" → last word is the name
            vertex_props.append(line.split()[-1])

    if n_splats is None:
        raise ValueError("No 'element vertex' found in header")

    # Header byte length: each ASCII line + 1 byte for the newline character.
    # This is the offset we pass to numpy.fromfile to skip past the header
    # and start reading binary data.
    header_bytes = sum(len(line.encode("ascii")) + 1 for line in header_lines)

    return n_splats, header_bytes, vertex_props


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_ply(path: str | Path) -> SplatCloud:
    """
    Load and decode a 3DGS .ply file into a SplatCloud.

    Strategy: read the entire binary vertex block in one numpy call,
    then slice out columns by name. This is much faster than row-by-row
    parsing — for an 85 MB file the difference is seconds vs minutes.

    Works with both the original format (has normals) and ml-sharp format
    (no normals, has extra metadata elements we silently skip).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    file_mb = path.stat().st_size / 1e6
    print(f"Loading: {path.name}  ({file_mb:.1f} MB)")

    # Read just the header first (small, ASCII)
    with open(path, "rb") as f:
        n_splats, header_size, vertex_props = _parse_header(f)

    # Decide which format we're dealing with
    has_normals = "nx" in vertex_props
    n_props_per_splat = len(vertex_props)

    print(f"  Splats:           {n_splats:,}")
    print(f"  Properties/splat: {n_props_per_splat}")
    print(f"  Has normals:      {has_normals}")
    print(f"  Header size:      {header_size} bytes")
    print(f"  Binary data size: {n_splats * n_props_per_splat * 4 / 1e6:.1f} MB\n")

    # -----------------------------------------------------------------------
    # Read all binary vertex data in one shot.
    #
    # numpy.fromfile reads raw bytes and interprets them as float32 values.
    # count = total number of float32 values = splats × properties
    # offset = skip past the ASCII header
    #
    # The result is a 1D array which we reshape to (N_splats, N_props).
    # Each row is one splat, each column is one property, in the order
    # they appear in the header.
    # -----------------------------------------------------------------------
    raw = np.fromfile(
        path,
        dtype=np.float32,
        count=n_splats * n_props_per_splat,
        offset=header_size,
    ).reshape(n_splats, n_props_per_splat)

    # Build a name→column_index lookup so we can slice by name not number.
    # e.g. col["opacity"] = 6  means opacity is in column 6 of raw
    col = {name: idx for idx, name in enumerate(vertex_props)}

    # -----------------------------------------------------------------------
    # Unpack each attribute from its column(s)
    # -----------------------------------------------------------------------

    # --- Positions: already in world space, no transform needed ---
    positions = raw[:, [col["x"], col["y"], col["z"]]].copy()

    # --- Normals: only in the original format ---
    if has_normals:
        normals = raw[:, [col["nx"], col["ny"], col["nz"]]].copy()
    else:
        normals = np.zeros((n_splats, 3), dtype=np.float32)

    # --- SH degree-0 (colour) coefficients ---
    # These are stored raw (not decoded) because we need them raw for export.
    # The decoded RGB below is just for visualisation/debugging.
    f_dc = raw[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]].copy()

    # --- Opacity: stored as logit, must decode via sigmoid ---
    #
    # Why logit? Neural networks output unbounded values. To constrain opacity
    # to [0, 1], the training code applies sigmoid(x) = 1 / (1 + e^-x).
    # The file stores the pre-sigmoid value (the logit). We decode here.
    #
    # sigmoid(0)   = 0.5   (neutral)
    # sigmoid(5)   ≈ 0.99  (nearly opaque)
    # sigmoid(-5)  ≈ 0.01  (nearly transparent)
    raw_opacity = raw[:, col["opacity"]]
    opacities = (1.0 / (1.0 + np.exp(-raw_opacity))).astype(np.float32)

    # --- Scale: stored in log space, must decode via exp ---
    #
    # Why log? Scale must always be positive. Training stores log(scale)
    # which can be any real number, then exp() gives us a positive scale.
    # We decode here.
    scales = np.exp(
        raw[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]
    ).astype(np.float32)

    # --- Rotation: stored as raw quaternion, normalise to unit length ---
    #
    # A quaternion must have length = 1 to represent a pure rotation.
    # Small floating point errors during training can make them slightly
    # non-unit. We renormalise here so downstream math stays correct.
    #
    # np.linalg.norm with axis=1 computes the length of each row (each quaternion).
    # keepdims=True keeps the shape (N, 1) so we can divide (N, 4) by (N, 1)
    # and numpy broadcasts correctly — dividing each row by its own length.
    rotations = raw[
        :, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]
    ].copy()
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    # np.where avoids division by zero for any degenerate zero quaternions
    rotations /= np.where(norms > 0, norms, 1.0)

    # --- Decoded RGB for diagnostics (not used in export) ---
    # colour = 0.5 + SH_C0 * f_dc  — this is the standard 3DGS colour decode
    colors_rgb = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0).astype(np.float32)

    cloud = SplatCloud(
        count      = n_splats,
        positions  = positions,
        normals    = normals,
        colors_rgb = colors_rgb,
        f_dc       = f_dc,
        opacities  = opacities,
        scales     = scales,
        rotations  = rotations,
    )

    _print_summary(cloud)
    return cloud


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _print_summary(cloud: SplatCloud) -> None:
    print("=" * 52)
    print("Splat cloud summary")
    print("=" * 52)
    print(f"  Total splats : {cloud.count:,}")
    print()
    print("  Positions (world space):")
    for i, label in enumerate(["x", "y", "z"]):
        lo = cloud.positions[:, i].min()
        hi = cloud.positions[:, i].max()
        print(f"    {label}  [{lo:.3f}, {hi:.3f}]")
    print()
    print("  Opacity (after sigmoid):")
    op = cloud.opacities
    print(f"    min {op.min():.4f}  max {op.max():.4f}  mean {op.mean():.4f}")
    for threshold in [0.05, 0.10, 0.20]:
        n = (op < threshold).sum()
        print(f"    below {threshold:.2f}: {n:,}  ({100*n/cloud.count:.1f}%)")
    print()
    print("  Scale (after exp):")
    ms = cloud.scales.mean(axis=0)
    print(f"    mean  x={ms[0]:.4f}  y={ms[1]:.4f}  z={ms[2]:.4f}")
    print()
    print("  Base RGB (from f_dc):")
    mr = cloud.colors_rgb.mean(axis=0)
    print(f"    mean  R={mr[0]:.3f}  G={mr[1]:.3f}  B={mr[2]:.3f}")
    print("=" * 52)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python splat_parser.py path/to/scene.ply")
        sys.exit(1)
    cloud = parse_ply(sys.argv[1])
    print("\nParsing complete. SplatCloud ready for stitching / optimisation.")