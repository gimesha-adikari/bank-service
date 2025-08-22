export APP_DOC_PORTRAIT_MODE=provided
export APP_FACE_THRESHOLD=0.70   # dev tweak for quick demo
# build payload inline
python - <<'PY' > payload_provided.json
from PIL import Image, ImageDraw
from io import BytesIO
import base64, json
def png_b64(img):
    buf = BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
img = Image.new("RGB",(128,128),"white")
d = ImageDraw.Draw(img)
d.ellipse((16,16,112,112), outline="black", width=3)
d.ellipse((44,48,60,64), fill="black")
d.ellipse((68,48,84,64), fill="black")
d.arc((44,64,84,100), start=200, end=340, fill="black", width=3)
payload = {"selfie": png_b64(img), "docPortraitImage": png_b64(img)}
print(json.dumps(payload))
PY

curl -s -X POST http://127.0.0.1:8000/api/v1/kyc/aggregate \
  -H "Content-Type: application/json" \
  --data-binary @payload_provided.json | jq .
