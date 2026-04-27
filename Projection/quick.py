from Stitcher import stitch_faces
from SplatExporter import export_ply

cloud = stitch_faces({
    'front':  './ml-sharp/gaussians/front.ply',
    'back':   './ml-sharp/gaussians/back.ply',
    'right':  './ml-sharp/gaussians/right.ply',
    'left':   './ml-sharp/gaussians/left.ply',
    'top':    './ml-sharp/gaussians/top.ply',
    'bottom': './ml-sharp/gaussians/bottom.ply',
})

export_ply(cloud, 'stitched_output.ply')