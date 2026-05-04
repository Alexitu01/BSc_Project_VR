# Panorama to Gaussian Splat Pipeline

Converts a 360° equirectangular panorama into a stitched 3D Gaussian Splat `.ply` file ready for VR viewing. Built as part of a feasibility study into AI-generated VR environments for psychiatric exposure therapy.

## What it does

Takes a single equirectangular panorama image and produces a merged world-space `.ply` Gaussian splat by:

1. Extracting 6 overlapping perspective slice images from the panorama
2. Running Apple's ml-sharp on each slice to generate per-slice Gaussian splats
3. Running DA360 on the full panorama to produce a globally consistent depth map
4. Stitching all slices into a single world-space splat cloud using DA360 for depth alignment
5. Exporting the result as a standard `.ply` compatible with the aras-p Unity Gaussian splat renderer

---

## Repository structure

```
├── FaceExtractor.py       - Extracts perspective slice images from panorama
├── RunSharp.py            - Runs ml-sharp on slices via Python API (correct focal length)
├── RunStitcher.py         - Orchestrates DA360 + stitching + export
├── SplatStitcher.py       - Core stitching logic (DA360 alignment, Voronoi clipping, world rotation)
├── SplatParser.py         - Parses 3DGS .ply files into SplatCloud dataclass
├── SplatExporter.py       - Exports SplatCloud back to .ply
├── Da360Predictor.py      - Wraps DA360 model for panorama depth inference
├── DA360/                 - DA360 submodule (panorama-aware depth estimation)
├── ml-sharp/              - ml-sharp submodule (monocular Gaussian splat prediction)
└── Dockerfile             - Docker setup for RunPod deployment
```

---

## Requirements

### Conda environments

This pipeline requires **two separate conda environments** because ml-sharp and DA360 have conflicting dependencies.

**Environment 1 - `sharp` (for RunSharp.py)**
```bash
conda create -n sharp python=3.11
conda activate sharp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd ml-sharp && pip install -e . && cd ..
pip install -r requirements-sharp.txt
```

**Environment 2 - `da360` (for RunStitcher.py)**
```bash
conda create -n da360 python=3.10
conda activate da360
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-da360.txt
```

### Submodules

Clone the repo with submodules:
```bash
git clone --recurse-submodules <repo-url>
```

Or if already cloned:
```bash
git submodule update --init --recursive
```

### Model weights

Model weights are not included in the repository. Download them manually:

**ml-sharp checkpoint**
Download `sharp.pt` from the [ml-sharp releases page](https://github.com/apple/ml-sharp) and place at:
```
ml-sharp/sharp.pt
```

**DA360 checkpoint**
Download `DA360_small.pth` from the [DA360 repository](https://github.com/Insta360-Research-Team/DA360) and place at:
```
DA360/checkpoints/DA360_small.pth
```
This can be changed to a larger weight just correct the file name in RunStitcher.py

---

## How to run

### Step 1 - Extract slice images from panorama

Run from any environment with numpy, pillow, scipy:
```bash
python FaceExtractor.py panorama.jpg ./slices/ --slices 6 --overlap 8
```

This saves 6 perspective slice images to `./slices/` and a `slices_meta.txt` metadata file containing the focal length and azimuth for each slice. The metadata file is the source of truth for all downstream steps - re-extracting automatically updates it.

**Arguments:**
- `--slices` - number of horizontal slices (default 6)
- `--overlap` - overlap in degrees per side beyond the natural span (default 15, recommended 8)
- `--size` - base slice width in pixels (default: panorama_width / n_slices)

---

### Step 2 - Run ml-sharp on each slice

Run from the `sharp` environment:
```bash
conda activate sharp
python RunSharp.py
```

Edit the paths at the top of `RunSharp.py` before running:
```python
SLICES_DIR  = Path("slices")
OUTPUT_DIR  = Path("ml-sharp") / "gaussians"
CHECKPOINT  = Path("ml-sharp") / "sharp.pt"
META_FILE   = SLICES_DIR / "slices_meta.txt"
```

This reads the focal length from `slices_meta.txt` and passes it directly to ml-sharp's Python API, ensuring each slice is reconstructed with the correct field of view. Output `.ply` files are saved to `ml-sharp/gaussians/`.

---

### Step 3 - Stitch slices into world-space splat

Run from the `da360` environment:
```bash
conda activate da360
python RunStitcher.py
```

Edit the paths at the top of `RunStitcher.py` before running:
```python
PANORAMA_PATH = 'panorama.jpg'
META_PATH     = Path('slices/slices_meta.txt')
PLY_FOLDER    = Path('ml-sharp/gaussians')
DA360_ROOT    = 'DA360'
DA360_MODEL   = 'DA360/checkpoints/DA360_small.pth'
```

This will:
- Load slice metadata and focal lengths
- Run DA360 on the full panorama (once)
- For each slice: align depths to DA360 reference, Voronoi clip overlap, rotate into world space
- Merge all slices, restore scene scale, flip Y axis to match viewer convention
- Export to `stitched_output.ply`

---

## Panorama requirements

- Format: equirectangular (360° × 180°)
- Recommended aspect ratio: 2:1 - 21:9 (wider ratios reduce vertical coverage)
- Minimum recommended resolution: 1024px wide
- The pipeline is designed for AI-generated panoramas and handles non-standard aspect ratios gracefully

---

## Output

`stitched_output.ply` is a standard 3DGS `.ply` file 

**Note for Unity:** ml-sharp uses OpenCV coordinate convention (Y-down). The stitcher applies a Y-flip before export. When importing into the aras-p Unity renderer, set the GaussianSplatRenderer transform rotation to `(0, 0, 180)` as a starting point to match Unity's left-handed coordinate system.

---

## Known limitations

- Pole distortion at the top and bottom of the scene is inherent to equirectangular panoramas and cannot be fully corrected at the projection stage
- Seam quality depends on DA360's depth consistency - scenes with large homogeneous regions (blank walls, sky) may show more visible seams
- ml-sharp is a monocular model trained on perspective images - reconstructed geometry is plausible but not metrically accurate
- The pipeline is designed for offline processing, not real-time generation