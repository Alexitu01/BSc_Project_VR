import base64
import io
import numpy as np
from PIL import Image

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

'''
def prepare_panorama(base64_str, k = 2):
    pano = load_panorama_from_base64(base64_str)
    
    # Get widht and height
    h, w, _ = pano.shape
    
    # Validate
    aspect = w / h
    if not (1.9 < aspect < 2.1):
        raise ValueError("Expected 2:1 equirectangular panorama")
    
    # Compute face resolution
    face_size = w // 4 * k
    
    return pano, face_size


