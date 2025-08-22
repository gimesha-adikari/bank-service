# --- helpers for this test ---
from io import BytesIO
from PIL import Image, ImageDraw
import base64

def _png_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _face_like(size=(128, 128)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    d.ellipse((16, 16, size[0]-16, size[1]-16), outline="black", width=3)
    d.ellipse((44, 48, 60, 64), fill="black")
    d.ellipse((68, 48, 84, 64), fill="black")
    d.arc((44, 64, 84, 100), start=200, end=340, fill="black", width=3)
    return img
