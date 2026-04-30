import base64
import io
import math
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import map_coordinates


"""
Loads panoramic image from base64 to numpy array
we'll be doing a lot of matrix math, so that's a better format
"""
def load_panorama_from_base64(base64_str):
    # Decode base64 → bytes
    image_bytes = base64.b64decode(base64_str)
    
    # Load with PIL
    image = Image.open(io.BytesIO(image_bytes))
    
    # Ensure consistent format (important!)
    image = image.convert("RGB")
    
    # Convert to NumPy
    pano = np.array(image)
    
    return pano

'''
Validates that panoramic image has proper aspec ratio (2 to 1, width - height) 
and gets the size of each of the four faces for a matching cubemap projection, 
adding a scaler k (2 is default) to help avoid empty spots in final splat.

Returns pano, face_size, and focal_px.
focal_px MUST be passed to stitch_faces() - it encodes the FOV used during
extraction so the depth alignment projection is correct.
'''
def prepare_panorama(base64_str, k = 2, overlap_degrees=10.0):
    pano = load_panorama_from_base64(base64_str)
    
    # Get widht and height
    h, w, _ = pano.shape
    
    # Validate
    aspect = w / h
    if not (1.9 < aspect < 2.1):
        raise ValueError("Expected 2:1 equirectangular panorama")
    
    # Compute face resolution, we add a K for better gaussian splat resolution
    face_size = w // 4 * k

    # Compute focal length from the actual FOV used (90° + overlap on both sides)
    # This must match what extract_face uses internally
    total_fov_deg = 90.0 + 2.0 * overlap_degrees
    focal_px = (face_size / 2.0) / math.tan(math.radians(total_fov_deg / 2.0))
    
    return pano, face_size, focal_px

'''
Define the 6 face direction formulas as lambdas.
Each takes (u, v) grids and returns (x, y, z) - all same shape

Face	Fixed axis
front	z = +1
back	z = -1
right	x = +1
left	x = -1
top	    y = +1
bottom	y = -1
leaving this here so I don't mess it up (think cube)
'''
FACE_DIRECTIONS = {
    # using np.ones_like since u and v are arrays
    'front':  lambda u, v: (u,    -v,  np.ones_like(u)),
    'back':   lambda u, v: (-u,   -v, -np.ones_like(u)),
    'right':  lambda u, v: (np.ones_like(u),  -v, -u),
    'left':   lambda u, v: (-np.ones_like(u), -v,  u),
    'top':    lambda u, v: (u,  np.ones_like(u),  v),
    'bottom': lambda u, v: (u, -np.ones_like(u), -v),
}

# fov_deg for introducing overlap
def extract_face(pano, face_name, face_size, fov_deg=90.0):
    h, w = pano.shape[:2]
    
    # Build a grid(2d array) of pixel coordinates for the output face
    # i = row (0 to face_size-1), j = col (0 to face_size-1)
    j, i = np.meshgrid(np.arange(face_size), np.arange(face_size))
    
    # Normalise pixel coords to [-1, 1] at the edge of a 90° FOV face.
    # For FOV > 90°, we scale u and v beyond [-1, 1] so the rays fan out
    # wider than a standard cubemap face, capturing overlap with neighbours.
    #
    # The scale factor converts from pixel space to the tangent of the
    # actual half-angle. For 90° FOV, tan(45°) = 1.0 so no scaling needed.
    # For 110° FOV, tan(55°) ≈ 1.428 so we scale by 1.428 to get the
    # wider ray directions.
    u = (j + 0.5) / face_size * 2 - 1
    v = (i + 0.5) / face_size * 2 - 1

    # Apply FOV scaling
    # fov_scale > 1.0 widens the rays beyond the standard 90° cubemap
    fov_scale = math.tan(math.radians(fov_deg / 2.0))
    u = u * fov_scale
    v = v * fov_scale
    
    # Get 3D direction for this face
    x, y, z = FACE_DIRECTIONS[face_name](u, v)
    
    # Normalise to unit vector (norm should never be able to be 0)
    #Stole this math from stack overflow: https://stackoverflow.com/questions/19299155/normalize-a-vector-of-3d-coordinates-to-be-in-between-0-and-1
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/norm, y/norm, z/norm
    
    # Convert our unit vector to longitude and latitude
    #https://www.mathworks.com/help/matlab/ref/cart2sph.html OBS. their x -> z, y -> x, z -> -y (y is flipped cuz image coords are weird)
    lon = np.arctan2(x, z)          # -π to π
    lat = np.arctan2(-y, np.sqrt(x**2 + z**2))  # -π/2 to π/2
    
    # Convert to panorama pixel coordinates
    #We first normalize it from -π, π to -1, 1, then shift it to be between 0, and 2, since pixels can't be negative, 
    #and then scale it to be between 0 and 1, so it's essentially percent. and then multiply it by widht or height of image to get right pixel... I hope
    px = (lon / np.pi + 1) / 2 * (w - 1)
    py = (lat / (np.pi / 2) + 1) / 2 * (h - 1)
    
    # Sample the panorama at these coordinates (one channel at a time) using some scipy voodoo magic
    face = np.stack([
        map_coordinates(pano[:,:,c], [py, px], order=1, mode='wrap')
        for c in range(3)
    ], axis=-1)
    
    return face.astype(np.uint8)

#Just runs through all the faces
def extract_all_faces(pano, face_size, overlap_degrees=10.0):
    #needs to parse this
    total_fov_deg = 90.0 + 2.0 * overlap_degrees

    return {
        name: extract_face(pano, name, face_size, fov_deg=total_fov_deg)
        for name in FACE_DIRECTIONS
    }

def load_panorama_from_file(path):
    image = Image.open(path).convert("RGB")
    return np.array(image)

#Test function
if __name__ == "__main__":
    import sys
 
    path     = sys.argv[1] if len(sys.argv) > 1 else "test.png"
    overlap  = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
 
    pano = load_panorama_from_file(path)
    
    h, w, _ = pano.shape
    face_size = w // 4
 
    total_fov = 90.0 + 2.0 * overlap
    focal_px  = (face_size / 2.0) / math.tan(math.radians(total_fov / 2.0))
 
    print(f"Panorama: {w}×{h}")
    print(f"Face size: {face_size}px,  FOV: {total_fov:.0f}°,  focal_px: {focal_px:.2f}")
    print(f"Pass focal_x={focal_px:.2f}, focal_y={focal_px:.2f} to stitch_faces()\n")
 
    faces = extract_all_faces(pano, face_size, overlap_degrees=overlap)
 
    output_dir = Path("faces")
    output_dir.mkdir(exist_ok=True)
 
    for name, face in faces.items():
        Image.fromarray(face).save(output_dir / f"{name}.png")
        print(f"  Saved: faces/{name}.png")
 
    print("\nDone. Saved 6 cube faces.")


def makeCubeMap(image_path):
    import sys
 
    path     = image_path
    overlap  = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
 
    pano = load_panorama_from_file(path)
    
    h, w, _ = pano.shape
    face_size = w // 4
 
    total_fov = 90.0 + 2.0 * overlap
    focal_px  = (face_size / 2.0) / math.tan(math.radians(total_fov / 2.0))
 
    print(f"Panorama: {w}×{h}")
    print(f"Face size: {face_size}px,  FOV: {total_fov:.0f}°,  focal_px: {focal_px:.2f}")
    print(f"Pass focal_x={focal_px:.2f}, focal_y={focal_px:.2f} to stitch_faces()\n")
 
    faces = extract_all_faces(pano, face_size, overlap_degrees=overlap)
    
    output_dir = Path("faces")
    output_dir.mkdir(exist_ok=True)
    facePngs={}
    for name, face in faces.items():
        if name == "top" or name == "bottom":
            Image.fromarray(face).save(output_dir / f"{name}.png")
            facePngs[name] = str(f"faces/{name}.png")
            print(f"  Saved: faces/{name}.png")
    
    
    return facePngs

