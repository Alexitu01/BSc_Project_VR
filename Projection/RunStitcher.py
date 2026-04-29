from PIL import Image

from SplatStitcher import stitch_faces
from SplatExporter import export_ply

# Read one image to get dimensions
img = Image.open(r'.\faces\front.png')
width, height = img.size

# Cube-map faces usually use 90° FOV
focal_x = width / 2
focal_y = height / 2

cloud = stitch_faces(
    face_plys={
        'front':  r'.\ml-sharp\gaussians\front.ply',
        'back':   r'.\ml-sharp\gaussians\back.ply',
        'right':  r'.\ml-sharp\gaussians\right.ply',
        'left':   r'.\ml-sharp\gaussians\left.ply',
        #'top':    r'.\ml-sharp\gaussians\top.ply',
        #'bottom': r'.\ml-sharp\gaussians\bottom.ply',
    },
    face_images={
        'front':  r'.\faces\front.png',
        'back':   r'.\faces\back.png',
        'right':  r'.\faces\right.png',
        'left':   r'.\faces\left.png',
        #'top':    r'.\faces\top.png',
        #'bottom': r'.\faces\bottom.png',
    },
    focal_x=focal_x,
    focal_y=focal_y,
)

export_ply(cloud, 'stitched_output.ply')