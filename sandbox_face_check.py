# sandbox_face_check.py
import cv2 as cv, sys, numpy as np, pathlib

root = pathlib.Path(__file__).resolve().parent
yunet = str(root/"models/face/face_detection_yunet_2023mar.onnx")
sface = str(root/"models/face/face_recognition_sface_2021dec.onnx")

det = cv.FaceDetectorYN_create(yunet, "", (320, 320), score_threshold=0.8, nms_threshold=0.3, top_k=5000)
rec = cv.FaceRecognizerSF_create(sface, "")

def feat(img_bgr):
    h, w = img_bgr.shape[:2]
    det.setInputSize((w, h))
    faces = det.detect(img_bgr)[1]
    if faces is None or len(faces)==0:
        raise RuntimeError("No face found")
    # take best face
    f = max(faces, key=lambda r: r[-1])
    # f: [x,y,w,h, l0x,l0y, ..., l4x,l4y, score]
    aligned = rec.alignCrop(img_bgr, f)
    return rec.feature(aligned)

def sim(a, b):  # cosine similarity in OpenCV’s matcher style
    return rec.match(a, b, cv.FaceRecognizerSF_FR_COSINE)

img1 = cv.imread(sys.argv[1])
img2 = cv.imread(sys.argv[2])
f1, f2 = feat(img1), feat(img2)
print("similarity (cosine):", sim(f1, f2))
