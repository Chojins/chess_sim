"""Rasterize the explicit desired-board SVG as a separate RGB policy input."""
import io
from pathlib import Path
import numpy as np


def render_goal(svg):
    import cairosvg
    from PIL import Image
    pixels=cairosvg.svg2png(bytestring=Path(svg).read_bytes(),output_width=480,output_height=480)
    board=np.asarray(Image.open(io.BytesIO(pixels)).convert('RGB'),dtype=np.uint8)
    result=np.zeros((480,640,3),np.uint8);result[:,80:560]=board
    return result
