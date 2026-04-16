import base64
import io
import numpy as np
from PIL import Image
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
'''
def prepare_panorama(base64_str, k = 2):
    pano = load_panorama_from_base64(base64_str)
    
    # Get widht and height
    h, w, _ = pano.shape
    
    # Validate
    aspect = w / h
    if not (1.9 < aspect < 2.1):
        raise ValueError("Expected 2:1 equirectangular panorama")
    
    # Compute face resolution, we add a K for better gaussian splat resolution
    face_size = w // 4 * k
    
    return pano, face_size

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
leaving this here so I don't mess it up
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

def extract_face(pano, face_name, face_size):
    h, w = pano.shape[:2]
    
    # Build a grid(2d array) of pixel coordinates for the output face
    # i = row (0 to face_size-1), j = col (0 to face_size-1)
    j, i = np.meshgrid(np.arange(face_size), np.arange(face_size))
    
    # Normalise our pixel coords to fit between [-1, 1]
    u = (j + 0.5) / face_size * 2 - 1
    v = (i + 0.5) / face_size * 2 - 1
    
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
def extract_all_faces(pano, face_size):
    return {
        name: extract_face(pano, name, face_size)
        for name in FACE_DIRECTIONS
    }


