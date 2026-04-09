import hashlib
import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

import cv2
from paddleocr import PaddleOCR

EARLY_OK = float(os.getenv("OCR_EARLY_OK", "0.93"))
ORTHOGONAL_OK = float(os.getenv("OCR_ORTHOGONAL_OK", "0.83"))
DIAGONAL_START_BELOW = float(os.getenv("OCR_DIAGONAL_START_BELOW", "0.70"))
DIAG_SCALE = float(os.getenv("OCR_DIAG_SCALE", "0.72"))
CACHE_SIZE = int(os.getenv("OCR_CACHE_SIZE", "256"))

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = (BASE_DIR / ".." / "models" / "paddleocr").resolve()
DET_MODEL_DIR = Path(os.getenv("PADDLE_DET_MODEL_DIR", MODELS_DIR / "PP-OCRv5_server_det"))
REC_MODEL_DIR = Path(os.getenv("PADDLE_REC_MODEL_DIR", MODELS_DIR / "en_PP-OCRv5_mobile_rec"))
TEXTLINE_ORI_MODEL_DIR = Path(os.getenv("PADDLE_TEXTLINE_ORI_MODEL_DIR", MODELS_DIR / "PP-LCNet_x1_0_textline_ori"))


def _build_ocr() -> PaddleOCR:
    cpu_threads = max(1, int(os.getenv("PADDLE_CPU_THREADS", "2")))
    kwargs: dict[str, Any] = {
        "lang": "en",
        "use_textline_orientation": True,
        "device": os.getenv("PADDLE_DEVICE", "cpu"),
        "enable_hpi": False,
        "enable_mkldnn": False,
        "cpu_threads": cpu_threads,
    }

    if DET_MODEL_DIR.exists() and REC_MODEL_DIR.exists():
        kwargs["text_detection_model_dir"] = str(DET_MODEL_DIR)
        kwargs["text_recognition_model_dir"] = str(REC_MODEL_DIR)
    if TEXTLINE_ORI_MODEL_DIR.exists():
        kwargs["textline_orientation_model_dir"] = str(TEXTLINE_ORI_MODEL_DIR)

    return PaddleOCR(**kwargs)


ocr = _build_ocr()

_CLEAN_RE = re.compile(r"[^A-Z0-9]+")
_CACHE: "OrderedDict[str, tuple[str, int | None, float]]" = OrderedDict()


def clean(s: str) -> str:
    return _CLEAN_RE.sub("", s.upper())


def _seal_text_quality(text: str) -> float:
    if not text:
        return 0.0

    digits = sum(ch.isdigit() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    length = len(text)

    if length < 5:
        return 0.0

    len_score = min(1.0, length / 10.0)
    digit_score = 1.0 if digits > 0 else 0.0
    mix_score = 1.0 if (digits > 0 and letters > 0) else 0.6

    return 0.45 * len_score + 0.35 * digit_score + 0.20 * mix_score


def _preprocess_fast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _preprocess_hard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 40, 40)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        3,
    )
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)


def rotate_any(img, angle_deg: float):
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(c, angle_deg, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    matrix[0, 2] += (nw / 2.0) - c[0]
    matrix[1, 2] += (nh / 2.0) - c[1]

    return cv2.warpAffine(
        img,
        matrix,
        (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _rotate_90(img, a: int):
    if a == 0:
        return img
    if a == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if a == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if a == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _extract_result_json(pred_item) -> dict:
    payload = getattr(pred_item, "json", pred_item)
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _run_ocr(img):
    if hasattr(ocr, "predict"):
        try:
            return ocr.predict(img)
        except Exception:
            # Fallback for environments where PaddleOCR v3 predict path fails on CPU runtime.
            if hasattr(ocr, "ocr"):
                return ocr.ocr(img)
            raise
    return ocr.ocr(img)


def _best_text_from_any_result(result) -> tuple[str, float]:
    if not result:
        return "", -1.0

    # Newer PaddleOCR result objects
    if isinstance(result, list) and result and hasattr(result[0], "json"):
        j = _extract_result_json(result[0])
        res = j.get("res", {})
        texts = res.get("rec_texts", []) or []
        scores = res.get("rec_scores", []) or []
        best = ""
        best_final = -1.0
        for raw_text, raw_score in zip(texts, scores):
            text = clean(str(raw_text))
            if not text:
                continue
            ocr_score = float(raw_score)
            quality = _seal_text_quality(text)
            final = 0.8 * ocr_score + 0.2 * quality
            if final > best_final:
                best_final = final
                best = text
        return best, best_final

    # Classic PaddleOCR list format
    best = ""
    best_final = -1.0
    lines = result[0] if isinstance(result, list) and result else []
    for line in lines or []:
        if not line or len(line) < 2:
            continue
        text = clean(str(line[1][0]))
        if not text:
            continue
        ocr_score = float(line[1][1])
        quality = _seal_text_quality(text)
        final = 0.8 * ocr_score + 0.2 * quality
        if final > best_final:
            best_final = final
            best = text
    return best, best_final


def _predict_at_angle(img, angle: int) -> tuple[str, float]:
    rotated = _rotate_90(img, angle) if angle in (0, 90, 180, 270) else rotate_any(img, angle)
    result = _run_ocr(rotated)
    return _best_text_from_any_result(result)


def _cache_key(img) -> str:
    h, w = img.shape[:2]
    digest = hashlib.blake2b(img.tobytes(), digest_size=16).hexdigest()
    return f"{h}x{w}:{digest}"


def _cache_get(key: str):
    value = _CACHE.get(key)
    if value is None:
        return None
    _CACHE.move_to_end(key)
    return value


def _cache_set(key: str, value):
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    while len(_CACHE) > CACHE_SIZE:
        _CACHE.popitem(last=False)


def ocr_best(img):
    if img is None:
        return "", None, -1.0

    key = _cache_key(img)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    img_fast = _preprocess_fast(img)

    best_text = ""
    best_score = -1.0
    best_angle = None

    for angle in (0, 180):
        text, score = _predict_at_angle(img_fast, angle)
        if score > best_score:
            best_text, best_score, best_angle = text, score, angle
        if best_score >= EARLY_OK:
            result = (best_text, best_angle, best_score)
            _cache_set(key, result)
            return result

    if best_score < ORTHOGONAL_OK:
        for angle in (90, 270):
            text, score = _predict_at_angle(img_fast, angle)
            if score > best_score:
                best_text, best_score, best_angle = text, score, angle
            if best_score >= EARLY_OK:
                result = (best_text, best_angle, best_score)
                _cache_set(key, result)
                return result

    if best_score < DIAGONAL_START_BELOW:
        img_hard = _preprocess_hard(img)

        for angle in (0, 180):
            text, score = _predict_at_angle(img_hard, angle)
            if score > best_score:
                best_text, best_score, best_angle = text, score, angle
            if best_score >= EARLY_OK:
                result = (best_text, best_angle, best_score)
                _cache_set(key, result)
                return result

        h, w = img_hard.shape[:2]
        diag_img = cv2.resize(
            img_hard,
            (max(1, int(w * DIAG_SCALE)), max(1, int(h * DIAG_SCALE))),
            interpolation=cv2.INTER_AREA,
        )
        for angle in (45, 135, 225, 315):
            text, score = _predict_at_angle(diag_img, angle)
            if score > best_score:
                best_text, best_score, best_angle = text, score, angle
            if best_score >= EARLY_OK:
                break

    result = (best_text, best_angle, best_score)
    _cache_set(key, result)
    return result
