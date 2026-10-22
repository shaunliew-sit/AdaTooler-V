"""Crop rendering for SFT trace generation — reuses the EXACT live RL crop
(`pixel_reasoner.crop` with coordinate_scale=1000) so SFT crops == RL crops.
"""
import base64
import io

from PIL import Image

from verl_tool.servers.tools.pixel_reasoner import crop as _rl_crop

COORDINATE_SCALE = 1000.0  # grid1000 — matches PIXEL_REASONER_BBOX_MODE in the train script


def render_zoom(image, bbox_2d):
    """image: PIL.Image or path. bbox_2d: 0-1000 [x1,y1,x2,y2]. Returns a PIL.Image crop
    identical to what the live zoom_in tool would return."""
    return _rl_crop(image, list(bbox_2d), coordinate_scale=COORDINATE_SCALE)


def pil_to_data_url(img, fmt="JPEG", quality=90):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


def load_image(path):
    return Image.open(path).convert("RGB")
