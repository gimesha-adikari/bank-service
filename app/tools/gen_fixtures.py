from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw
import base64, json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

def b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def circle_face(size=(128,128)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    d.ellipse((16, 16, size[0]-16, size[1]-16), outline="black", width=3)
    d.ellipse((44, 48, 60, 64), fill="black")
    d.ellipse((68, 48, 84, 64), fill="black")
    d.arc((44, 64, 84, 100), start=200, end=340, fill="black", width=3)
    return img

def square_face(size=(128,128)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    d.rectangle((20,20,size[0]-20,size[1]-20), outline="black", width=3)
    d.rectangle((48,48,60,60), fill="black")
    d.rectangle((68,48,80,60), fill="black")
    d.line((48,90,80,90), fill="black", width=3)
    return img

def checkerboard(size=(128,128), step=8) -> Image.Image:
    img = Image.new("L", size, 0)
    d = ImageDraw.Draw(img)
    for y in range(0, size[1], step):
        for x in range(0, size[0], step):
            if ((x//step + y//step) % 2) == 0:
                d.rectangle((x,y,x+step-1,y+step-1), fill=255)
    return img.convert("RGB")

def doc_text() -> Image.Image:
    img = Image.new("RGB", (700, 260), "white")
    d = ImageDraw.Draw(img)
    y=24
    for ln in [
        "REPUBLIC OF SRI LANKA",
        "NATIONAL IDENTITY CARD",
        "Department of Registration of Persons",
        "Name: Test User",
        "DOB: 1999-01-01",
        "ID: ABC1234567",
    ]:
        d.text((20,y), ln, fill="black")
        y += 34
    return img

def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj), encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}")

def main():
    selfie_circle = circle_face()
    portrait_circle = circle_face()
    selfie_square  = square_face()
    selfie_flat    = Image.new("RGB", (128,128), (128,128,128))
    doc            = doc_text()

    approve = {
        "selfie": b64(selfie_circle),
        "docPortraitImage": b64(portrait_circle),
        "docFrontImage": b64(doc),
        "meta": {"countryHint":"LK", "docClassHint":"NIC"}
    }
    under_review = {
        "selfie": b64(selfie_square),
        "docPortraitImage": b64(portrait_circle),
        "docFrontImage": b64(doc),
        "meta": {"countryHint":"LK", "docClassHint":"NIC"}
    }
    reject = {
        "selfie": b64(selfie_flat),
        "docPortraitImage": b64(portrait_circle),
        "meta": {"countryHint":"LK", "docClassHint":"NIC"}
    }

    write_json(OUT / "approve.json", approve)
    write_json(OUT / "under_review.json", under_review)
    write_json(OUT / "reject.json", reject)

if __name__ == "__main__":
    main()
