# AI model inventory

The GitHub baseline intentionally contains only the small tracked OCR
dictionary. The original local checkout contained additional ONNX files and
configuration paths. Those files were preserved in the external snapshot at
`/home/gimesha/My_Projects/multi-lng/revival-snapshots/20260919-001405` and are
not silently committed here.

Use the following relative runtime paths when the preserved files are made
available locally:

| Feature | Expected path |
| --- | --- |
| Document classifier | `models/doc/layoutlmv3_small.onnx` |
| Face detector | `models/face/scrfd_10g_bnkps.onnx` |
| Face recognizer | `models/face/arcface_r100_glint360k.onnx` |
| Liveness | `models/liveness/fas_rgb.onnx` |
| Address NER | `models/nlp/address_ner.onnx` |
| OCR detector | `models/ocr/en_PP-OCRv3_det_infer.onnx` |
| OCR recognizer | `models/ocr/en_PP-OCRv3_rec_infer.onnx` |
| OCR dictionary | `models/ocr/latin_dict.txt` |

The preserved inventory and SHA-256 manifest are the provenance evidence. A
future release decision must identify licenses, source URLs/version, file
ownership, and whether the models are downloaded at setup time or supplied as
protected runtime assets. Do not put secrets or private download URLs in this
repository.

The preserved local inventory currently contains two non-empty ONNX files
(`face_detection_yunet.onnx` and `face_recognition_sface.onnx`); several other
files are zero-byte placeholders. Zero-byte files are not valid model assets and
must not be selected as an advanced backend. The external snapshot retains the
exact files and SHA-256 manifest for later provenance work.
