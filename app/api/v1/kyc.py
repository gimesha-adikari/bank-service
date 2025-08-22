from __future__ import annotations

import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Response

from .schemas import KycPayload, KycResult, AggregateResponse, CheckResult
from app.core.config import get_settings, segmented_threshold
from app.features.factory import get_factory
from app.core.policy import decide
from app.core.calib_log import write_row

router = APIRouter(prefix="/kyc", tags=["kyc"])


@router.get("/ping")
def ping():
    return {"ok": True, "feature": "kyc"}


@router.post("/face/match", response_model=KycResult)
def face_match(p: KycPayload):
    settings = get_settings()
    doc_front = p.docFront or p.docFrontImage
    if p.selfie is None or doc_front is None:
        raise HTTPException(
            status_code=400,
            detail="selfie and docFront/docFrontImage are required",
        )

    score, extra = get_factory().face_matcher().score(
        bytes(p.selfie), bytes(doc_front)
    )
    passed = score >= settings.face_threshold
    extra["threshold"] = settings.face_threshold
    return KycResult(
        type="FACE_MATCH",
        score=score,
        passed=passed,
        detailsJson=json.dumps(extra),
    )


@router.post("/liveness", response_model=KycResult)
def liveness(p: KycPayload):
    settings = get_settings()
    if p.selfie is None:
        raise HTTPException(status_code=400, detail="selfie is required")

    score, extra = get_factory().liveness().score(bytes(p.selfie))
    passed = score >= settings.live_threshold
    extra["threshold"] = settings.live_threshold
    return KycResult(
        type="LIVENESS",
        score=score,
        passed=passed,
        detailsJson=json.dumps(extra),
    )


@router.post("/ocr/id", response_model=KycResult)
def ocr_id(p: KycPayload):
    settings = get_settings()
    front_b = p.docFrontImage or p.docFront
    back_b = p.docBackImage
    if front_b is None and back_b is None:
        raise HTTPException(
            status_code=400,
            detail="docFrontImage/docFront or docBackImage is required",
        )

    score, fields = get_factory().ocr().extract(
        bytes(front_b) if front_b else None,
        bytes(back_b) if back_b else None,
    )
    passed = score >= settings.ocr_threshold
    fields["threshold"] = settings.ocr_threshold
    return KycResult(
        type="OCR_ID",
        score=score,
        passed=passed,
        detailsJson=json.dumps(fields),
    )


@router.post("/doc/class", response_model=KycResult)
def doc_class(p: KycPayload):
    settings = get_settings()
    front_b = p.docFrontImage or p.docFront
    back_b = p.docBackImage
    if front_b is None and back_b is None:
        raise HTTPException(
            status_code=400,
            detail="docFrontImage/docFront or docBackImage is required",
        )

    score, extra = get_factory().doc_classifier().classify(
        bytes(front_b) if front_b else None,
        bytes(back_b) if back_b else None,
    )
    passed = score >= settings.doc_threshold
    extra["threshold"] = settings.doc_threshold
    return KycResult(
        type="DOC_CLASS",
        score=score,
        passed=passed,
        detailsJson=json.dumps(extra),
    )


@router.post("/aggregate", response_model=AggregateResponse)
def aggregate(p: KycPayload, response: Response):
    settings = get_settings()
    factory = get_factory()

    request_id = str(uuid4())
    response.headers["X-Request-ID"] = request_id

    results: dict[str, dict] = {}

    face_score = None; face_details: dict = {}
    if p.selfie is not None:
        portrait = p.docPortraitImage
        doc_front = p.docFront or p.docFrontImage
        used_source = None; portrait_info: dict = {}; portrait_bytes = None
        mode = settings.doc_portrait_mode
        factory_port = factory.portrait()
        if portrait is not None and mode in ("auto", "provided"):
            used_source = "provided"; portrait_bytes = bytes(portrait)
        elif doc_front is not None and mode == "auto" and factory_port is not None:
            crop, info = factory_port.extract(bytes(doc_front))
            if crop is not None:
                used_source = "auto"; portrait_bytes = crop; portrait_info = info or {}
            else:
                used_source = "front_fallback"; portrait_bytes = bytes(doc_front)
                if info: face_details["portrait_extractor_reason"] = info.get("reason")
        else:
            if doc_front is not None:
                used_source = "front_fallback"; portrait_bytes = bytes(doc_front)
            else:
                used_source = "none"; portrait_bytes = None
        if portrait_bytes is not None:
            s, extra = factory.face_matcher().score(bytes(p.selfie), portrait_bytes)
            face_score = s
            face_details = {**extra, "portrait_used": used_source in ("auto","provided"), "source": used_source}
            if portrait_info: face_details["portrait_bbox"] = portrait_info.get("bbox")
        else:
            face_details = {"source": used_source, "reason": "no_doc_image"}

    results["FACE_MATCH"] = {"score": face_score, "details": face_details}

    live_score = None; live_details: dict = {}
    if p.selfie is not None:
        s, extra = factory.liveness().score(bytes(p.selfie))
        live_score = s; live_details = extra
    results["LIVENESS"] = {"score": live_score, "details": live_details}

    ocr_score = None; ocr_details: dict = {}
    if (p.docFront or p.docFrontImage) or p.docBackImage:
        s, fields = factory.ocr().extract(
            bytes(p.docFrontImage or p.docFront) if (p.docFrontImage or p.docFront) else None,
            bytes(p.docBackImage) if p.docBackImage else None,
        )
        ocr_score = s; ocr_details = fields
    results["OCR_ID"] = {"score": ocr_score, "details": ocr_details}

    cls_score = None; cls_details: dict = {}
    if (p.docFront or p.docFrontImage) or p.docBackImage:
        s, extra = factory.doc_classifier().classify(
            bytes(p.docFrontImage or p.docFront) if (p.docFrontImage or p.docFront) else None,
            bytes(p.docBackImage) if p.docBackImage else None,
        )
        cls_score = s; cls_details = extra
    results["DOC_CLASS"] = {"score": cls_score, "details": cls_details}

    country = None; doc_cls = None
    try:
        country = (results.get("DOC_CLASS",{}).get("details") or {}).get("country") or None
        doc_cls = (results.get("DOC_CLASS",{}).get("details") or {}).get("class") or None
        if not country:
            text_preview = (results.get("OCR_ID",{}).get("details") or {}).get("textPreview") or ""
            if "sri lanka" in text_preview.lower(): country = "LK"
    except Exception:
        pass

    face_thr = segmented_threshold("APP_FACE_THRESHOLD", country, doc_cls, settings.face_threshold)
    live_thr = segmented_threshold("APP_LIVE_THRESHOLD", country, doc_cls, settings.live_threshold)
    ocr_thr  = segmented_threshold("APP_OCR_THRESHOLD",  country, doc_cls, settings.ocr_threshold)
    doc_thr  = segmented_threshold("APP_DOC_THRESHOLD",  country, doc_cls, settings.doc_threshold)

    def pass_at(score, thr):
        return (score is not None) and (score >= thr)

    results["FACE_MATCH"].update({"passed": pass_at(results["FACE_MATCH"]["score"], face_thr)})
    results["LIVENESS"].update({"passed": pass_at(results["LIVENESS"]["score"], live_thr)})
    results["OCR_ID"].update({"passed": pass_at(results["OCR_ID"]["score"], ocr_thr)})
    results["DOC_CLASS"].update({"passed": pass_at(results["DOC_CLASS"]["score"], doc_thr)})

    (results["FACE_MATCH"]["details"] or {}).update({"threshold": face_thr})
    (results["LIVENESS"]["details"] or {}).update({"threshold": live_thr})
    (results["OCR_ID"]["details"] or {}).update({"threshold": ocr_thr})
    (results["DOC_CLASS"]["details"] or {}).update({"threshold": doc_thr})

    decision, reasons = decide(settings, results)

    country = None
    doc_cls = None
    try:
        doc_details = results.get("DOC_CLASS", {}).get("details") or {}
        country = doc_details.get("country") or None
        doc_cls = doc_details.get("class") or None
        if not country:
            ocr_details = results.get("OCR_ID", {}).get("details") or {}
            text_preview = (ocr_details.get("textPreview") or "").lower()
            if "sri lanka" in text_preview:
                country = "LK"
    except Exception:
        pass

    if settings.calibration_log:
        try:
            write_row(
                dirpath=settings.calibration_dir,
                request_id=request_id,
                instance_id=settings.instance_id,
                decision=decision,
                reasons=reasons,
                checks=results,
                country=country,
                doc_class=doc_cls,
            )
        except Exception:
            pass

    checks = [
        CheckResult(type=k, score=v.get("score"), passed=v.get("passed"), details=v.get("details"))
        for k, v in results.items()
    ]
    return AggregateResponse(decision=decision, reasons=reasons, checks=checks)
