def test_aggregate_face_fallback_when_no_doc_images(client, monkeypatch):
    monkeypatch.setenv("APP_DOC_PORTRAIT_MODE", "auto")
    # only selfie provided -> face check cannot run properly
    from io import BytesIO
    from PIL import Image
    import base64

    img = Image.new("RGB", (64,64), "white")
    buf = BytesIO(); img.save(buf, format="PNG")
    s_b64 = base64.b64encode(buf.getvalue()).decode()

    r = client.post("/api/v1/kyc/aggregate", json={"selfie": s_b64})
    assert r.status_code == 200
    data = r.json()
    face = next(c for c in data["checks"] if c["type"] == "FACE_MATCH")
    assert face["details"]["source"] in ("none", "front_fallback")
