import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

_DLL_DIR_HANDLES: list[Any] = []


def _configure_windows_gpu_dlls() -> None:
    if os.name != "nt":
        return

    paddle_device = os.getenv("PADDLE_DEVICE", "cpu").strip().lower()
    if not paddle_device.startswith("gpu"):
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    candidate_dirs = [
        site_packages / "nvidia" / "cu13" / "bin" / "x86_64",
        site_packages / "nvidia" / "cu13" / "bin",
        site_packages / "nvidia" / "cu13" / "lib" / "x64",
        site_packages / "nvidia" / "cudnn" / "bin",
    ]

    existing_dirs = [path for path in candidate_dirs if path.exists()]
    if not existing_dirs:
        return

    os.environ["PATH"] = ";".join([*(str(path) for path in existing_dirs), os.environ.get("PATH", "")])
    for path in existing_dirs:
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(path)))
        except (AttributeError, FileNotFoundError, OSError):
            continue


_configure_windows_gpu_dlls()

import cv2
import numpy as np
from paddleocr import PaddleOCR

from .container_layout import classify_container_layout, extract_split_door_rois, extract_two_line_rois, foreground_mask, merge_bands

logger = logging.getLogger("container_ocr")

EARLY_OK = float(os.getenv("OCR_EARLY_OK", "0.93"))
CACHE_SIZE = int(os.getenv("OCR_CACHE_SIZE", "256"))
HORIZONTAL_RATIO_THRESHOLD = float(os.getenv("OCR_HORIZONTAL_RATIO_THRESHOLD", "3.2"))
VERTICAL_RATIO_THRESHOLD = float(os.getenv("OCR_VERTICAL_RATIO_THRESHOLD", "0.22"))

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = (BASE_DIR / ".." / "models" / "paddleocr").resolve()
DET_MODEL_DIR = Path(os.getenv("PADDLE_DET_MODEL_DIR", MODELS_DIR / "PP-OCRv5_server_det"))
REC_MODEL_DIR = Path(os.getenv("PADDLE_REC_MODEL_DIR", MODELS_DIR / "en_PP-OCRv5_mobile_rec"))
TEXTLINE_ORI_MODEL_DIR = Path(os.getenv("PADDLE_TEXTLINE_ORI_MODEL_DIR", MODELS_DIR / "PP-LCNet_x1_0_textline_ori"))
DOC_ORI_MODEL_DIR = Path(
    os.getenv("PADDLE_DOC_ORI_MODEL_DIR", MODELS_DIR / "PP-LCNet_x1_0_doc_ori")
)
DOC_UNWARP_MODEL_DIR = Path(os.getenv("PADDLE_DOC_UNWARP_MODEL_DIR", MODELS_DIR / "UVDoc"))
_TESSERACT_EXE_CACHE: str | None = None


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


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
    if DOC_ORI_MODEL_DIR.exists():
        kwargs["doc_orientation_classify_model_dir"] = str(DOC_ORI_MODEL_DIR)
    if DOC_UNWARP_MODEL_DIR.exists():
        kwargs["doc_unwarping_model_dir"] = str(DOC_UNWARP_MODEL_DIR)

    return PaddleOCR(**kwargs)


def _resolve_tesseract_executable() -> str:
    global _TESSERACT_EXE_CACHE
    if _TESSERACT_EXE_CACHE is not None:
        return _TESSERACT_EXE_CACHE

    candidates = [
        os.getenv("TESSERACT_EXE", "").strip(),
        str((BASE_DIR / ".." / ".." / "third_party" / "tesseract" / "tesseract.exe").resolve()),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        shutil.which("tesseract") or "",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            _TESSERACT_EXE_CACHE = candidate
            return candidate

    _TESSERACT_EXE_CACHE = ""
    return _TESSERACT_EXE_CACHE


ocr = _build_ocr()

_CLEAN_RE = re.compile(r"[^A-Z0-9]+")
_CONTAINER_RE = re.compile(r"[A-Z]{4}\d{7}")
_CONTAINER10_RE = re.compile(r"[A-Z]{4}\d{6}")
_DIGIT_CHUNK_RE = re.compile(r"\d+")
_CACHE: "OrderedDict[str, tuple[str, float, bool, dict[str, Any]]]" = OrderedDict()

_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "B",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "Z",
    "8": "B",
}

_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "U": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}


def clean(s: str) -> str:
    return _CLEAN_RE.sub("", s.upper())


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


def _get_text_rec_model():
    try:
        return ocr.paddlex_pipeline._pipeline.text_rec_model
    except Exception:
        return None


def _run_text_recognition_batch(images: list[Any]) -> list[tuple[str, float]]:
    samples = [_ensure_bgr(img) for img in images if img is not None and getattr(img, "size", 0)]
    if not samples:
        return []

    rec_model = _get_text_rec_model()
    if rec_model is None:
        out: list[tuple[str, float]] = []
        for img in samples:
            pairs = _extract_text_pairs(_run_ocr(img))
            if pairs:
                out.append(max(pairs, key=lambda item: item[1]))
            else:
                out.append(("", 0.0))
        return out

    results: list[tuple[str, float]] = []
    try:
        predicted = list(rec_model.predict(samples, batch_size=len(samples)))
    except Exception:
        predicted = []

    if not predicted:
        for img in samples:
            pairs = _extract_text_pairs(_run_ocr(img))
            if pairs:
                results.append(max(pairs, key=lambda item: item[1]))
            else:
                results.append(("", 0.0))
        return results

    for item in predicted:
        payload = getattr(item, "json", {})
        if isinstance(payload, dict):
            res = payload.get("res", {})
        else:
            res = {}
        text = clean(str(res.get("rec_text") or ""))
        try:
            score = float(res.get("rec_score") or 0.0)
        except Exception:
            score = 0.0
        results.append((text, score))
    return results


def _letter_value(ch: str) -> int:
    value = 10
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if value % 11 == 0:
            value += 1
        if letter == ch:
            return value
        value += 1
    raise ValueError(f"Unsupported letter: {ch}")


def _iso6346_check_digit(code10: str) -> int:
    total = 0
    for pos, ch in enumerate(code10):
        if ch.isdigit():
            value = int(ch)
        else:
            value = _letter_value(ch)
        total += value * (2 ** pos)

    remainder = total % 11
    return 0 if remainder == 10 else remainder


def _is_valid_iso6346(code11: str) -> bool:
    code11 = clean(code11)
    if not _CONTAINER_RE.fullmatch(code11):
        return False
    expected = _iso6346_check_digit(code11[:10])
    actual = int(code11[10])
    return expected == actual


def _force_owner_category_u(prefix: str) -> str:
    prefix = "".join(_TO_LETTER.get(ch, ch) for ch in clean(prefix))
    if len(prefix) < 4:
        return prefix
    return f"{prefix[:3]}U"


def _normalize_window_to_container(part: str) -> Optional[str]:
    part = clean(part)
    if len(part) != 11:
        return None

    prefix = _force_owner_category_u(part[:4])
    suffix = "".join(_TO_DIGIT.get(ch, ch) for ch in part[4:])
    candidate = prefix + suffix
    if _CONTAINER_RE.fullmatch(candidate):
        return candidate
    return None


def _normalized_candidates(raw: str) -> list[str]:
    s = clean(raw)
    if len(s) < 11:
        return []

    out = []
    seen = set()
    for i in range(len(s) - 10):
        part = s[i:i + 11]
        candidate = _normalize_window_to_container(part)
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _normalize_window_to_container10(part: str) -> Optional[str]:
    part = clean(part)
    if len(part) != 10:
        return None

    prefix = _force_owner_category_u(part[:4])
    suffix = "".join(_TO_DIGIT.get(ch, ch) for ch in part[4:])
    candidate = prefix + suffix
    if _CONTAINER10_RE.fullmatch(candidate):
        return candidate
    return None


def _normalized_base_candidates(raw: str) -> list[str]:
    s = clean(raw)
    if len(s) < 10:
        return []

    out = []
    seen = set()
    for i in range(len(s) - 9):
        part = s[i:i + 10]
        candidate = _normalize_window_to_container10(part)
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)

    # Reordered blocks are common on crops: e.g. "438607TCKU" should become "TCKU438607".
    for m in re.finditer(r"\d{6}[A-Z]{4}", s):
        part = m.group(0)
        candidate = _normalize_window_to_container10(part[6:] + part[:6])
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)

    # Heuristic pairing for slightly noisy OCR streams with separated alpha/digit groups.
    alpha_blocks = [m.group(0) for m in re.finditer(r"[A-Z]{4,}", s)]
    digit_blocks = [m.group(0) for m in re.finditer(r"\d{6,}", s)]
    for a in alpha_blocks:
        for d in digit_blocks:
            candidate = _normalize_window_to_container10(a[:4] + d[:6])
            if candidate and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out


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


def _extract_text_pairs_from_predict(j: dict) -> list[tuple[str, float]]:
    res = j.get("res", {})
    texts = res.get("rec_texts", []) or []
    scores = res.get("rec_scores", []) or []

    pairs = []
    for raw_text, raw_score in zip(texts, scores):
        text = clean(str(raw_text))
        if not text:
            continue
        try:
            score = float(raw_score)
        except Exception:
            score = 0.0
        pairs.append((text, score))
    return pairs


def _extract_text_pairs(result) -> list[tuple[str, float]]:
    if not result:
        return []

    if isinstance(result, list) and result and hasattr(result[0], "json"):
        return _extract_text_pairs_from_predict(_extract_result_json(result[0]))

    pairs = []
    lines = result[0] if isinstance(result, list) and result else []
    for line in lines or []:
        if not line or len(line) < 2:
            continue
        text = clean(str(line[1][0]))
        if not text:
            continue
        pairs.append((text, float(line[1][1])))
    return pairs


def _text_pair_preview(text_pairs: list[tuple[str, float]], *, limit: int = 8) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for text, score in text_pairs[:limit]:
        preview.append(
            {
                "text": str(text),
                "score": round(float(score), 4),
            }
        )
    return preview


def _candidate_groups(text_pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not text_pairs:
        return []

    groups: list[tuple[str, float]] = []
    seen = set()
    texts = [t for t, _ in text_pairs]
    scores = [s for _, s in text_pairs]
    n = len(texts)

    def add_candidate(raw: str, score: float):
        key = (raw, round(score, 6))
        if raw and key not in seen:
            seen.add(key)
            groups.append((raw, score))

    for text, score in text_pairs:
        add_candidate(text, score)

    add_candidate("".join(texts), sum(scores) / max(1, len(scores)))
    add_candidate(" ".join(texts), sum(scores) / max(1, len(scores)))

    for size in range(2, min(7, n + 1)):
        for i in range(n - size + 1):
            joined = "".join(texts[i:i + size])
            avg_score = sum(scores[i:i + size]) / size
            add_candidate(joined, avg_score)

    return groups


def _container_text_quality(code: str) -> float:
    if not code:
        return 0.0

    code = clean(code)
    if not _CONTAINER_RE.fullmatch(code):
        return 0.0

    score = 0.85
    if code[:4].isalpha():
        score += 0.05
    if code[4:].isdigit():
        score += 0.05
    if _is_valid_iso6346(code):
        score += 0.05
    return min(1.0, score)


def _container_base_quality(code10: str) -> float:
    code10 = clean(code10)
    if not _CONTAINER10_RE.fullmatch(code10):
        return 0.0
    score = 0.8
    if code10[:4].isalpha():
        score += 0.1
    if code10[4:].isdigit():
        score += 0.1
    return min(1.0, score)


def _is_horizontal_crop(img) -> bool:
    if img is None or img.size == 0:
        return False
    h, w = img.shape[:2]
    if h <= 0:
        return False
    return (w / float(h)) >= HORIZONTAL_RATIO_THRESHOLD


def _is_vertical_crop(img) -> bool:
    if img is None or img.size == 0:
        return False
    h, w = img.shape[:2]
    if h <= 0:
        return False
    return (w / float(h)) <= VERTICAL_RATIO_THRESHOLD


def _vertical_foreground_mask(img):
    gray = _ensure_gray(img)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(mask) > (mask.size * 0.5):
        mask = 255 - mask
    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _merge_vertical_bands(bands: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not bands:
        return []
    merged = [bands[0]]
    for start, end in bands[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _split_vertical_band_by_projection(
    row_strength: np.ndarray,
    start: int,
    end: int,
    parts: int,
) -> list[tuple[int, int]]:
    length = end - start
    if parts <= 1 or length <= 0:
        return [(start, end)]

    segment = row_strength[start:end].astype(np.float32)
    if segment.size == 0:
        return [(start, end)]

    smoothed = cv2.GaussianBlur(segment.reshape(-1, 1), (1, 7), 0).reshape(-1)
    min_part_h = max(10, int(length / max(parts * 2, 2)))
    boundaries = [0]

    for part_idx in range(1, parts):
        target = int(round(length * part_idx / parts))
        window = max(6, int(length / max(parts * 3, 3)))
        lo = max(boundaries[-1] + min_part_h, target - window)
        hi = min(length - ((parts - part_idx) * min_part_h), target + window)
        if hi <= lo:
            cut = target
        else:
            cut = lo + int(np.argmin(smoothed[lo:hi]))
        boundaries.append(cut)

    boundaries.append(length)
    out: list[tuple[int, int]] = []
    for local_start, local_end in zip(boundaries, boundaries[1:]):
        if (local_end - local_start) >= 8:
            out.append((start + local_start, start + local_end))
    return out if len(out) > 1 else [(start, end)]


def _refine_vertical_bands(
    row_strength: np.ndarray,
    bands: list[tuple[int, int]],
    img_h: int,
) -> list[tuple[int, int]]:
    if not bands:
        return []

    min_band_h = max(8, int(img_h * 0.015))
    band_heights = [end - start for start, end in bands if (end - start) >= min_band_h]
    if not band_heights:
        return bands

    short_heights = [height for height in band_heights if height <= np.percentile(band_heights, 70)]
    typical_h = int(np.median(short_heights or band_heights))
    typical_h = max(min_band_h, typical_h)

    refined: list[tuple[int, int]] = []
    for start, end in bands:
        height = end - start
        parts = 1
        if height >= int(typical_h * 1.6):
            parts = max(2, min(6, int(round(height / float(typical_h)))))
        if parts > 1:
            refined.extend(_split_vertical_band_by_projection(row_strength, start, end, parts))
        else:
            refined.append((start, end))
    return refined


def _extract_vertical_symbol_rois(img) -> list[Any]:
    if img is None or img.size == 0:
        return []

    mask = _vertical_foreground_mask(img)
    h, w = mask.shape[:2]
    row_strength = np.count_nonzero(mask, axis=1)
    row_threshold = max(2, int(w * 0.08))

    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for idx, value in enumerate(row_strength):
        if value >= row_threshold and not in_band:
            start = idx
            in_band = True
        elif value < row_threshold and in_band:
            bands.append((start, idx))
            in_band = False
    if in_band:
        bands.append((start, h))

    raw_bands = list(bands)
    refined_bands = _refine_vertical_bands(row_strength, raw_bands, h)
    merged_bands = _merge_vertical_bands(raw_bands, max_gap=max(0, int(h * 0.002)))

    if 10 <= len(refined_bands) <= 12:
        bands = refined_bands
    elif 10 <= len(raw_bands) <= 12:
        bands = raw_bands
    elif 10 <= len(merged_bands) <= 12:
        bands = merged_bands
    elif len(refined_bands) > len(raw_bands) and len(refined_bands) >= 8:
        bands = refined_bands
    elif len(raw_bands) > len(merged_bands) and len(raw_bands) >= 8:
        bands = raw_bands
    else:
        bands = merged_bands

    rois: list[Any] = []
    min_band_h = max(8, int(h * 0.015))
    for y1, y2 in bands:
        if (y2 - y1) < min_band_h:
            continue
        band_mask = mask[y1:y2, :]
        cols = np.where(np.count_nonzero(band_mask, axis=0) > 0)[0]
        if cols.size == 0:
            continue
        x1 = max(0, int(cols[0]) - 4)
        x2 = min(w, int(cols[-1]) + 5)
        roi = img[max(0, y1 - 2):min(h, y2 + 2), x1:x2]
        if roi.size == 0:
            continue
        rois.append(roi)
    return rois


def _foreground_bbox_from_mask(mask, pad: int = 2) -> Optional[tuple[int, int, int, int]]:
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    h, w = mask.shape[:2]
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w, int(xs.max()) + pad + 1)
    y2 = min(h, int(ys.max()) + pad + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _tight_foreground_crop(roi, *, square_inner: bool = False):
    if roi is None or roi.size == 0:
        return roi

    source = roi
    if square_inner:
        inner = _safe_inner_crop(roi, margin_ratio=0.14)
        if inner is not None and inner.size:
            source = inner

    mask = _vertical_foreground_mask(source)
    bbox = _foreground_bbox_from_mask(mask, pad=2)
    if bbox is None:
        return source
    x1, y1, x2, y2 = bbox
    cropped = source[y1:y2, x1:x2]
    return cropped if cropped.size else source


def _compose_symbol_strip(
    rois: list[Any],
    *,
    target_h: int = 92,
    gap: int = 12,
    outer_pad: int = 12,
    square_last: bool = False,
):
    if not rois:
        return None

    prepared: list[Any] = []
    for idx, roi in enumerate(rois):
        cropped = _tight_foreground_crop(roi, square_inner=square_last and idx == len(rois) - 1)
        if cropped is None or cropped.size == 0:
            continue
        h, w = cropped.shape[:2]
        if h <= 0 or w <= 0:
            continue
        scale = target_h / float(h)
        resized = cv2.resize(
            cropped,
            (max(1, int(round(w * scale))), target_h),
            interpolation=cv2.INTER_CUBIC,
        )
        prepared.append(resized)

    if not prepared:
        return None

    canvas_h = target_h + (outer_pad * 2)
    total_w = (outer_pad * 2) + sum(img.shape[1] for img in prepared) + (gap * max(0, len(prepared) - 1))
    canvas = np.zeros((canvas_h, total_w, 3), dtype=np.uint8)

    x = outer_pad
    for img in prepared:
        y = (canvas_h - img.shape[0]) // 2
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        x += img.shape[1] + gap
    return canvas


def _vertical_grid_rois(img, parts: int) -> list[Any]:
    if img is None or img.size == 0 or parts < 10:
        return []

    mask = _vertical_foreground_mask(img)
    bbox = _foreground_bbox_from_mask(mask, pad=4)
    if bbox is None:
        return []

    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    crop_h = crop.shape[0]
    min_part_h = max(8, int(crop_h * 0.04))
    rois: list[Any] = []
    for idx in range(parts):
        local_y1 = int(round((idx * crop_h) / float(parts)))
        local_y2 = int(round(((idx + 1) * crop_h) / float(parts)))
        if (local_y2 - local_y1) < min_part_h:
            continue
        roi = crop[local_y1:local_y2, :]
        tight = _tight_foreground_crop(roi, square_inner=(idx == parts - 1 and parts >= 11))
        if tight is None or tight.size == 0:
            continue
        rois.append(tight)
    return rois


def _collect_strip_pairs(strip, *, phase_bonus: float = 0.0, profile: str = "full") -> list[tuple[str, float]]:
    if strip is None or strip.size == 0:
        return []

    if profile == "minimal":
        passes = [
            (cv2.resize(strip, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC), phase_bonus + 0.03),
        ]
    elif profile == "light":
        passes = [
            (cv2.resize(strip, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC), phase_bonus + 0.03),
            (cv2.resize(_preprocess_fast(strip), None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC), phase_bonus + 0.04),
        ]
    else:
        passes = [
            (cv2.resize(strip, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC), phase_bonus + 0.03),
            (cv2.resize(_preprocess_fast(strip), None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC), phase_bonus + 0.04),
            (cv2.resize(_preprocess_hard(strip), None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC), phase_bonus + 0.02),
        ]

    merged: list[tuple[str, float]] = []
    for prepared, bonus in passes:
        for text, score in _extract_text_pairs(_run_ocr(prepared)):
            merged.append((text, min(1.0, float(score) + bonus)))
    return merged


def _collect_vertical_strip_pairs(strip, *, phase_bonus: float = 0.0, profile: str = "full") -> list[tuple[str, float]]:
    if strip is None or strip.size == 0:
        return []

    if profile == "minimal":
        passes = [
            (strip, phase_bonus + 0.03),
        ]
    elif profile == "light":
        passes = [
            (strip, phase_bonus + 0.03),
            (_preprocess_fast(strip), phase_bonus + 0.04),
        ]
    else:
        passes = [
            (strip, phase_bonus + 0.03),
            (_preprocess_fast(strip), phase_bonus + 0.04),
            (_preprocess_hard(strip), phase_bonus + 0.02),
        ]

    merged: list[tuple[str, float]] = []
    recognized = _run_text_recognition_batch([prepared for prepared, _ in passes])
    for (_prepared, bonus), (text, score) in zip(passes, recognized):
        if text:
            merged.append((text, min(1.0, float(score) + bonus)))
    return merged


def _collect_compact_strip_pairs(strip, *, phase_bonus: float = 0.0, profile: str = "full") -> list[tuple[str, float]]:
    return _collect_vertical_strip_pairs(strip, phase_bonus=phase_bonus, profile=profile)


def _best_digit_from_strip(strip, *, profile: str = "full") -> tuple[str, float]:
    best_digit = ""
    best_score = -1.0
    for text, score in _collect_strip_pairs(strip, phase_bonus=0.0, profile=profile):
        digit, digit_score = _best_digit_from_pairs([(text, score)])
        if digit and digit_score > best_score:
            best_digit = digit
            best_score = digit_score
    return best_digit, best_score


def _best_vertical_numeric_read(text_pairs: list[tuple[str, float]]) -> tuple[str, str, float, str]:
    best_base6 = ""
    best_check_digit = ""
    best_score = -1.0
    best_raw = ""

    for raw_text, raw_score in text_pairs:
        mapped = "".join(_TO_DIGIT.get(ch, ch) for ch in clean(raw_text))
        for chunk in _DIGIT_CHUNK_RE.findall(mapped):
            if len(chunk) < 6:
                continue
            if len(chunk) >= 7:
                base6 = chunk[:6]
                check_digit = chunk[6]
                score = float(raw_score) + 0.18
            else:
                base6 = chunk[:6]
                check_digit = ""
                score = float(raw_score) + 0.08
            if score > best_score:
                best_base6 = base6
                best_check_digit = check_digit
                best_score = score
                best_raw = chunk

    return best_base6, best_check_digit, best_score, best_raw


def _extract_horizontal_symbol_rois(img) -> list[Any]:
    if img is None or img.size == 0:
        return []

    mask = foreground_mask(img)
    h, w = mask.shape[:2]
    col_strength = np.count_nonzero(mask, axis=0)
    col_threshold = max(2, int(h * 0.14))

    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for idx, value in enumerate(col_strength):
        if value >= col_threshold and not in_band:
            start = idx
            in_band = True
        elif value < col_threshold and in_band:
            bands.append((start, idx))
            in_band = False
    if in_band:
        bands.append((start, w))

    bands = merge_bands(bands, max_gap=max(2, int(w * 0.01)))
    min_band_w = max(5, int(w * 0.018))
    rois: list[Any] = []
    for x1, x2 in bands:
        if (x2 - x1) < min_band_w:
            continue
        band_mask = mask[:, x1:x2]
        rows = np.where(np.count_nonzero(band_mask, axis=1) > 0)[0]
        if rows.size == 0:
            continue
        y1 = max(0, int(rows[0]) - 3)
        y2 = min(h, int(rows[-1]) + 4)
        roi = img[y1:y2, max(0, x1 - 2):min(w, x2 + 3)]
        if roi.size == 0:
            continue
        rois.append(roi)
    return rois


def _light_symbol_mask(img, *, v_min: int = 140, s_max: int = 60):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, v_min], dtype=np.uint8),
        np.array([180, s_max, 255], dtype=np.uint8),
    )
    return mask


def _split_wide_component(mask, rect: tuple[int, int, int, int], *, min_part_w: int) -> list[tuple[int, int, int, int]]:
    x, y, w, h = rect
    if w < max(20, min_part_w * 2):
        return [rect]

    local = mask[y:y + h, x:x + w]
    if local.size == 0:
        return [rect]

    col_strength = np.count_nonzero(local, axis=0).astype(np.float32)
    if col_strength.size < 12:
        return [rect]

    search_from = max(2, int(w * 0.28))
    search_to = min(w - 2, int(w * 0.72))
    if search_to - search_from < 4:
        return [rect]

    window = col_strength[search_from:search_to]
    split_rel = int(np.argmin(window))
    split_x = search_from + split_rel
    valley = float(window[split_rel])
    peak = float(col_strength.max()) if col_strength.size else 0.0
    if peak <= 0.0 or valley > peak * 0.62:
        return [rect]

    left_w = split_x
    right_w = w - split_x
    if left_w < min_part_w or right_w < min_part_w:
        return [rect]

    return [
        (x, y, left_w, h),
        (x + split_x, y, right_w, h),
    ]


def _extract_light_symbol_rois(
    img,
    *,
    expected: int = 0,
    min_h_ratio: float = 0.32,
    min_w_ratio: float = 0.02,
    max_w_ratio: float = 0.42,
    v_min: int = 140,
    s_max: int = 60,
    pad: int = 4,
) -> list[Any]:
    if img is None or img.size == 0:
        return []

    mask = _light_symbol_mask(img, v_min=v_min, s_max=s_max)
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects: list[tuple[int, int, int, int]] = []
    min_h = max(12, int(h * min_h_ratio))
    min_w = max(8, int(w * min_w_ratio))
    max_w = max(min_w + 1, int(w * max_w_ratio))

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh < min_h or bw < min_w or bw > max_w:
            continue
        rects.append((x, y, bw, bh))

    rects.sort(key=lambda item: item[0])
    if expected and 1 <= len(rects) < expected:
        widths = [rect[2] for rect in rects]
        median_w = float(np.median(widths)) if widths else 0.0
        expanded: list[tuple[int, int, int, int]] = []
        for rect in rects:
            if median_w > 0 and rect[2] > median_w * 1.35 and len(expanded) + (len(rects) - len(expanded)) < expected:
                expanded.extend(_split_wide_component(mask, rect, min_part_w=min_w))
            else:
                expanded.append(rect)
        rects = sorted(expanded, key=lambda item: item[0])

    rois: list[Any] = []
    for x, y, bw, bh in rects:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        rois.append(roi)
    return rois


def _read_digit_sequence(rois: list[Any], *, limit: int, square_last: bool = False) -> tuple[str, float]:
    digits: list[str] = []
    total_score = 0.0
    selected = rois[:limit]
    for idx, roi in enumerate(selected):
        target_roi = roi
        if square_last and idx == len(selected) - 1:
            inner = _safe_inner_crop(roi, margin_ratio=0.12)
            if inner is not None and inner.size:
                target_roi = inner
        digit, score = _read_digit_from_roi(target_roi, square_bias=square_last and idx == len(selected) - 1)
        if not digit:
            break
        digits.append(digit)
        total_score += max(score, 0.0)
    if len(digits) != len(selected):
        return "", -1.0
    return "".join(digits), total_score / float(max(1, len(digits)))


def _digit_chunks_from_pairs(
    text_pairs: list[tuple[str, float]],
    *,
    min_len: int,
    max_len: int | None = None,
    allow_short_final: bool = False,
) -> list[tuple[str, float, str]]:
    chunks: list[tuple[str, float, str]] = []
    seen: set[tuple[str, int]] = set()
    for raw_text, raw_score in _candidate_groups(text_pairs):
        mapped = "".join(_TO_DIGIT.get(ch, ch) for ch in clean(raw_text))
        for chunk in _DIGIT_CHUNK_RE.findall(mapped):
            if len(chunk) < min_len:
                continue
            trimmed = chunk if max_len is None else chunk[:max_len]
            if max_len is not None and len(trimmed) < min_len:
                continue
            if not allow_short_final and max_len is not None and len(trimmed) != max_len and len(chunk) >= max_len:
                continue
            key = (trimmed, int(round(float(raw_score) * 1000.0)))
            if key in seen:
                continue
            seen.add(key)
            chunks.append((trimmed, float(raw_score), raw_text))
    return chunks


def _split_door_owner_read_hard(top_left) -> tuple[str, float, str]:
    if top_left is None or top_left.size == 0:
        return "", -1.0, ""

    best_owner = ""
    best_score = -1.0
    best_raw = ""

    roi_groups: list[list[Any]] = []
    roi_groups.append(_extract_horizontal_symbol_rois(top_left))
    for params in (
        {"expected": 4, "min_h_ratio": 0.22, "min_w_ratio": 0.012, "max_w_ratio": 0.48, "v_min": 92, "s_max": 155},
        {"expected": 4, "min_h_ratio": 0.26, "min_w_ratio": 0.015, "max_w_ratio": 0.44, "v_min": 104, "s_max": 135},
        {"expected": 4, "min_h_ratio": 0.30, "min_w_ratio": 0.018, "max_w_ratio": 0.42, "v_min": 118, "s_max": 120},
    ):
        roi_groups.append(_extract_light_symbol_rois(top_left, **params))

    seen_shapes: set[tuple[int, int, int]] = set()
    for rois in roi_groups:
        if len(rois) < 3:
            continue
        subset = rois[:4]
        strip = _compose_symbol_strip(subset, target_h=108, gap=18, outer_pad=14)
        if strip is None or strip.size == 0:
            continue
        shape_key = (strip.shape[0], strip.shape[1], len(subset))
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)
        for profile, bonus in (("minimal", 0.04), ("light", 0.07), ("full", 0.09)):
            pairs = _collect_strip_pairs(strip, phase_bonus=bonus, profile=profile)
            owner, score, raw = _best_owner_code_from_pairs(pairs)
            if owner and score > best_score:
                best_owner = owner
                best_score = score
                best_raw = raw

    for profile, bonus in (("minimal", 0.01), ("light", 0.03), ("full", 0.04)):
        pairs = _collect_strip_pairs(top_left, phase_bonus=bonus, profile=profile)
        owner, score, raw = _best_owner_code_from_pairs(pairs)
        if owner and score > best_score:
            best_owner = owner
            best_score = score
            best_raw = raw

    return best_owner, best_score, best_raw


def _split_door_numeric_read_hard(bottom_left, bottom_right) -> tuple[str, str, float, str]:
    if bottom_left is None or bottom_left.size == 0 or bottom_right is None or bottom_right.size == 0:
        return "", "", -1.0, ""

    best_base6 = ""
    best_check_digit = ""
    best_score = -1.0
    best_raw = ""

    left_roi_groups: list[list[Any]] = [_extract_horizontal_symbol_rois(bottom_left)]
    right_roi_groups: list[list[Any]] = [_extract_horizontal_symbol_rois(bottom_right)]
    for params in (
        {"expected": 4, "min_h_ratio": 0.22, "min_w_ratio": 0.010, "max_w_ratio": 0.38, "v_min": 92, "s_max": 155},
        {"expected": 4, "min_h_ratio": 0.28, "min_w_ratio": 0.012, "max_w_ratio": 0.42, "v_min": 108, "s_max": 135},
        {"expected": 4, "min_h_ratio": 0.34, "min_w_ratio": 0.014, "max_w_ratio": 0.46, "v_min": 122, "s_max": 120},
    ):
        left_roi_groups.append(_extract_light_symbol_rois(bottom_left, **params))
    for params in (
        {"expected": 3, "min_h_ratio": 0.22, "min_w_ratio": 0.010, "max_w_ratio": 0.44, "v_min": 92, "s_max": 155},
        {"expected": 3, "min_h_ratio": 0.28, "min_w_ratio": 0.012, "max_w_ratio": 0.46, "v_min": 108, "s_max": 135},
        {"expected": 3, "min_h_ratio": 0.34, "min_w_ratio": 0.014, "max_w_ratio": 0.48, "v_min": 122, "s_max": 120},
    ):
        right_roi_groups.append(_extract_light_symbol_rois(bottom_right, **params))

    left_candidates: list[tuple[str, float, str]] = []
    seen_left: set[str] = set()
    for rois in left_roi_groups:
        if len(rois) >= 4:
            seq, seq_score = _read_digit_sequence(rois, limit=4)
            if len(seq) == 4 and seq not in seen_left:
                seen_left.add(seq)
                left_candidates.append((seq, seq_score + 0.10, seq))
            strip = _compose_symbol_strip(rois[:4], target_h=106, gap=14, outer_pad=12)
            if strip is not None and strip.size != 0:
                for profile, bonus in (("minimal", 0.05), ("light", 0.08), ("full", 0.09)):
                    chunks = _digit_chunks_from_pairs(
                        _collect_strip_pairs(strip, phase_bonus=bonus, profile=profile),
                        min_len=4,
                        max_len=4,
                    )
                    for chunk, score, raw in chunks:
                        if chunk in seen_left:
                            continue
                        seen_left.add(chunk)
                        left_candidates.append((chunk, score + 0.05, raw))

    for profile, bonus in (("minimal", 0.01), ("light", 0.03), ("full", 0.04)):
        chunks = _digit_chunks_from_pairs(
            _collect_strip_pairs(bottom_left, phase_bonus=bonus, profile=profile),
            min_len=4,
            max_len=4,
        )
        for chunk, score, raw in chunks:
            if chunk in seen_left:
                continue
            seen_left.add(chunk)
            left_candidates.append((chunk, score, raw))

    tail_candidates: list[tuple[str, str, float, str]] = []
    seen_tail: set[tuple[str, str]] = set()
    for rois in right_roi_groups:
        if len(rois) >= 3:
            seq, seq_score = _read_digit_sequence(rois, limit=3, square_last=True)
            if len(seq) == 3:
                key = (seq[:2], seq[2])
                if key not in seen_tail:
                    seen_tail.add(key)
                    tail_candidates.append((seq[:2], seq[2], seq_score + 0.11, seq))
        if len(rois) >= 2:
            seq2, seq2_score = _read_digit_sequence(rois, limit=2)
            if len(seq2) == 2:
                key = (seq2[:2], "")
                if key not in seen_tail:
                    seen_tail.add(key)
                    tail_candidates.append((seq2[:2], "", seq2_score + 0.05, seq2))
        if len(rois) >= 2:
            strip = _compose_symbol_strip(
                rois[:3] if len(rois) >= 3 else rois[:2],
                target_h=106,
                gap=14,
                outer_pad=12,
                square_last=(len(rois) >= 3),
            )
            if strip is not None and strip.size != 0:
                for profile, bonus in (("minimal", 0.05), ("light", 0.08), ("full", 0.09)):
                    chunks = _digit_chunks_from_pairs(
                        _collect_strip_pairs(strip, phase_bonus=bonus, profile=profile),
                        min_len=2,
                        max_len=3,
                        allow_short_final=True,
                    )
                    for chunk, score, raw in chunks:
                        right2 = chunk[:2]
                        check = chunk[2] if len(chunk) >= 3 else ""
                        if len(right2) < 2:
                            continue
                        key = (right2, check)
                        if key in seen_tail:
                            continue
                        seen_tail.add(key)
                        tail_candidates.append((right2, check, score + 0.05, raw))

    for profile, bonus in (("minimal", 0.01), ("light", 0.03), ("full", 0.04)):
        chunks = _digit_chunks_from_pairs(
            _collect_strip_pairs(bottom_right, phase_bonus=bonus, profile=profile),
            min_len=2,
            max_len=3,
            allow_short_final=True,
        )
        for chunk, score, raw in chunks:
            right2 = chunk[:2]
            check = chunk[2] if len(chunk) >= 3 else ""
            if len(right2) < 2:
                continue
            key = (right2, check)
            if key in seen_tail:
                continue
            seen_tail.add(key)
            tail_candidates.append((right2, check, score, raw))

    square_digit, square_score = _read_right_check_digit(bottom_right)
    if square_digit:
        key = ("", square_digit)
        if key not in seen_tail:
            seen_tail.add(key)
            tail_candidates.append(("", square_digit, square_score + 0.04, square_digit))

    for left_digits, left_score, left_raw in left_candidates:
        for right_digits, check_digit, tail_score, tail_raw in tail_candidates:
            if len(left_digits) != 4 or len(right_digits) < 2:
                continue
            score = (left_score * 0.58) + (tail_score * 0.42)
            if check_digit:
                score += 0.04
            base6 = f"{left_digits[:4]}{right_digits[:2]}"
            raw_text = " ".join(part for part in (left_raw, tail_raw) if part).strip()
            if score > best_score:
                best_base6 = base6
                best_check_digit = check_digit
                best_score = score
                best_raw = raw_text

    if best_base6:
        return best_base6, best_check_digit, best_score, best_raw

    if square_digit:
        return "", square_digit, square_score, square_digit
    return "", "", -1.0, ""


def _ocr_split_door_container_regions_hard(img, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    split = extract_split_door_rois(img, config=config)
    if not split:
        return {
            "raw_text": "",
            "base10": "",
            "check_digit": "",
            "full_code": "",
            "score": -1.0,
            "is_valid_iso": False,
            "status": "NOT_FOUND",
        }

    owner, owner_score, owner_raw = _split_door_owner_read_hard(split["top_left"])
    base6, check_digit, numeric_score, numeric_raw = _split_door_numeric_read_hard(split["bottom_left"], split["bottom_right"])
    raw_text = " ".join(part for part in (owner_raw, numeric_raw) if part).strip()

    if len(owner) != 4 or len(base6) != 6:
        return {
            "raw_text": raw_text,
            "base10": "",
            "check_digit": check_digit if len(check_digit) == 1 else "",
            "full_code": "",
            "score": -1.0,
            "is_valid_iso": False,
            "status": "RAW_ONLY" if raw_text else "NOT_FOUND",
        }

    base10 = f"{owner}{base6}"
    full_code = f"{base10}{check_digit}" if check_digit else base10
    return {
        "raw_text": raw_text,
        "base10": base10,
        "check_digit": check_digit,
        "full_code": full_code,
        "score": float(max(owner_score, 0.0) + max(numeric_score, 0.0)),
        "is_valid_iso": bool(len(full_code) == 11 and _is_valid_iso6346(full_code)),
        "status": "FULL_11" if len(full_code) == 11 else "PARTIAL_10",
    }


def _split_door_owner_read(top_left) -> tuple[str, float, str]:
    if top_left is None or top_left.size == 0:
        return "", -1.0, ""

    best_owner = ""
    best_score = -1.0
    best_raw = ""

    for owner_input, bonus, profile in (
        (top_left, 0.03, "light"),
        (top_left, 0.01, "minimal"),
    ):
        pairs = _collect_compact_strip_pairs(owner_input, phase_bonus=bonus, profile=profile)
        if not pairs:
            continue
        owner, score, raw = _best_owner_code_from_pairs(pairs)
        if owner and score > best_score:
            best_owner = owner
            best_score = score
            best_raw = raw

    top_symbols = _extract_light_symbol_rois(top_left, expected=4, min_h_ratio=0.45, min_w_ratio=0.025, max_w_ratio=0.38, v_min=118, s_max=105)
    if len(top_symbols) >= 4:
        strip = _compose_symbol_strip(top_symbols[:4], target_h=64, gap=10, outer_pad=8)
        pairs = _collect_compact_strip_pairs(strip, phase_bonus=0.05, profile="minimal")
        owner, score, raw = _best_owner_code_from_pairs(pairs)
        if owner and score > best_score:
            best_owner = owner
            best_score = score
            best_raw = raw

    return best_owner, best_score, best_raw


def _split_door_numeric_read(bottom_left, bottom_right) -> tuple[str, str, float, str]:
    left_digits = ""
    right_digits = ""
    check_digit = ""
    best_score = -1.0
    raw_parts: list[str] = []

    left_symbols = _extract_light_symbol_rois(bottom_left, expected=4, min_h_ratio=0.45, min_w_ratio=0.025, max_w_ratio=0.42, v_min=140, s_max=60)
    if len(left_symbols) >= 4:
        left_digits, left_score = _read_digit_sequence(left_symbols, limit=4)
        if left_digits:
            best_score = max(best_score, left_score)
            raw_parts.append(left_digits)

    if len(left_digits) < 4:
        pairs = _collect_compact_strip_pairs(bottom_left, phase_bonus=0.03, profile="minimal")
        raw = " ".join(text for text, _ in pairs).strip()
        if raw:
            raw_parts.append(raw)
        mapped = "".join(_TO_DIGIT.get(ch, ch) for ch in clean(raw))
        chunks = [chunk for chunk in _DIGIT_CHUNK_RE.findall(mapped) if len(chunk) >= 4]
        if chunks:
            left_digits = chunks[0][:4]

    right_symbols = _extract_light_symbol_rois(bottom_right, expected=3, min_h_ratio=0.40, min_w_ratio=0.02, max_w_ratio=0.44, v_min=140, s_max=75)
    if len(right_symbols) >= 3:
        tail_digits, tail_score = _read_digit_sequence(right_symbols, limit=3, square_last=True)
        if tail_digits:
            right_digits = tail_digits[:2]
            check_digit = tail_digits[2]
            best_score = max(best_score, tail_score)
            raw_parts.append(tail_digits)
    elif len(right_symbols) >= 2:
        tail_digits, tail_score = _read_digit_sequence(right_symbols, limit=2)
        if tail_digits:
            right_digits = tail_digits[:2]
            best_score = max(best_score, tail_score)
            raw_parts.append(tail_digits)

    if len(right_digits) < 2:
        pairs = _collect_compact_strip_pairs(bottom_right, phase_bonus=0.03, profile="minimal")
        raw = " ".join(text for text, _ in pairs).strip()
        if raw:
            raw_parts.append(raw)
        mapped = "".join(_TO_DIGIT.get(ch, ch) for ch in clean(raw))
        chunks = [chunk for chunk in _DIGIT_CHUNK_RE.findall(mapped) if len(chunk) >= 2]
        if chunks:
            picked = max(chunks, key=len)
            if len(picked) >= 3 and not check_digit:
                right_digits = picked[:2]
                check_digit = picked[2]
            else:
                right_digits = picked[:2]

    if not check_digit:
        digit, score = _read_right_check_digit(bottom_right)
        if digit:
            check_digit = digit
            best_score = max(best_score, score)

    base6 = f"{left_digits[:4]}{right_digits[:2]}" if len(left_digits) >= 4 and len(right_digits) >= 2 else ""
    raw_text = " ".join(part for part in raw_parts if part).strip()
    return base6, check_digit, best_score, raw_text


def _ocr_split_door_container_regions(img, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    split = extract_split_door_rois(img, config=config)
    if not split:
        return {
            "raw_text": "",
            "base10": "",
            "check_digit": "",
            "full_code": "",
            "score": -1.0,
            "is_valid_iso": False,
            "status": "NOT_FOUND",
        }

    owner, owner_score, owner_raw = _split_door_owner_read(split["top_left"])
    base6, check_digit, numeric_score, numeric_raw = _split_door_numeric_read(split["bottom_left"], split["bottom_right"])
    raw_text = " ".join(part for part in (owner_raw, numeric_raw) if part).strip()

    if raw_text and (len(owner) != 4 or len(base6) != 6):
        fallback_owner, fallback_owner_score, fallback_owner_raw = _best_owner_code_from_pairs([(raw_text, 0.35)])
        if fallback_owner and fallback_owner_score > owner_score:
            owner = fallback_owner
            owner_score = fallback_owner_score
            owner_raw = fallback_owner_raw

        numeric_tokens = [clean(token) for token in re.split(r"\s+", numeric_raw) if clean(token)]
        digit_chunks: list[str] = []
        for token in numeric_tokens:
            mapped_token = "".join(_TO_DIGIT.get(ch, ch) for ch in token)
            digit_part = "".join(ch for ch in mapped_token if ch.isdigit())
            if digit_part:
                digit_chunks.append(digit_part)
        joined_digits = "".join(digit_chunks)
        if len(base6) != 6 and len(joined_digits) >= 6:
            base6 = joined_digits[:6]
        if not check_digit and len(joined_digits) >= 7:
            check_digit = joined_digits[6]

    if len(owner) != 4 or len(base6) != 6:
        result = {
            "raw_text": raw_text,
            "base10": "",
            "check_digit": check_digit if len(check_digit) == 1 else "",
            "full_code": "",
            "score": -1.0,
            "is_valid_iso": False,
            "status": "RAW_ONLY" if raw_text else "NOT_FOUND",
        }
        enable_hard_fallback = bool((config or {}).get("enable_two_lines_hard_fallback", True))
        if enable_hard_fallback:
            hard = _ocr_split_door_container_regions_hard(img, config=config)
            if hard.get("base10"):
                return hard
            if hard.get("raw_text") and not result.get("raw_text"):
                return hard
        return result

    base10 = f"{owner}{base6}"
    full_code = f"{base10}{check_digit}" if check_digit else base10
    return {
        "raw_text": raw_text,
        "base10": base10,
        "check_digit": check_digit,
        "full_code": full_code,
        "score": float(max(owner_score, 0.0) + max(numeric_score, 0.0)),
        "is_valid_iso": bool(len(full_code) == 11 and _is_valid_iso6346(full_code)),
        "status": "FULL_11" if len(full_code) == 11 else "PARTIAL_10",
    }


def _best_letter_from_pairs(text_pairs: list[tuple[str, float]]) -> tuple[str, float]:
    best_letter = ""
    best_score = -1.0
    for text, score in text_pairs:
        mapped = "".join(_TO_LETTER.get(ch, ch) for ch in clean(text))
        letters = [(idx, ch) for idx, ch in enumerate(mapped) if ch.isalpha()]
        if not letters:
            continue
        for idx, letter in letters:
            candidate_score = float(score)
            if len(letters) == 1:
                candidate_score += 0.12
            if len(mapped) <= 2:
                candidate_score += 0.05
            if idx == 0 or idx == len(mapped) - 1:
                candidate_score += 0.02
            if len(letters) > 1:
                candidate_score -= 0.03 * float(len(letters) - 1)
            if candidate_score > best_score:
                best_letter = letter
                best_score = candidate_score
    return best_letter, best_score


def _paddle_letter_from_roi(roi) -> tuple[str, float]:
    if roi is None or roi.size == 0:
        return "", -1.0

    best_letter = ""
    best_score = -1.0
    variants = _digit_preprocess_variants(roi)
    for variant_idx in (1, 0, 3):
        if variant_idx >= len(variants):
            continue
        prepared = variants[variant_idx]
        up = cv2.resize(prepared, None, fx=2.8, fy=2.8, interpolation=cv2.INTER_CUBIC)
        pairs = _extract_text_pairs(_run_ocr(up))
        letter, score = _best_letter_from_pairs(pairs)
        if letter and score > best_score:
            best_letter = letter
            best_score = score
    return best_letter, best_score


def _vertical_symbol_sequence(rois: list[Any]) -> tuple[list[tuple[str, float]], str]:
    symbols: list[tuple[str, float]] = []
    raw_parts: list[str] = []
    for idx, roi in enumerate(rois):
        is_last = idx == len(rois) - 1
        h, w = roi.shape[:2]
        ratio = w / float(max(1, h))
        if idx < 4:
            symbol, score = _paddle_letter_from_roi(roi)
            if symbol:
                raw_parts.append(symbol)
                symbols.append((symbol, score))
            continue

        target_roi = roi
        if is_last and 0.55 <= ratio <= 1.35:
            inner = _safe_inner_crop(roi, margin_ratio=0.12)
            if inner is not None and inner.size:
                target_roi = inner
        symbol, score = _paddle_digit_from_roi(target_roi)
        if symbol:
                raw_parts.append(symbol)
                symbols.append((symbol, score))
    return symbols, "".join(raw_parts)


def _paddle_letter_from_roi_vertical(roi) -> tuple[str, float]:
    if roi is None or roi.size == 0:
        return "", -1.0

    best_letter = ""
    best_score = -1.0
    variants = _digit_preprocess_variants(roi)
    prepared_batch: list[Any] = []
    variant_order: list[int] = []
    for variant_idx in (1, 0, 3):
        if variant_idx < len(variants):
            prepared_batch.append(variants[variant_idx])
            variant_order.append(variant_idx)

    for _variant_idx, (text, score) in zip(variant_order, _run_text_recognition_batch(prepared_batch)):
        pairs = [(text, score)] if text else []
        letter, local_score = _best_letter_from_pairs(pairs)
        if letter and local_score > best_score:
            best_letter = letter
            best_score = local_score
    return best_letter, best_score


def _decode_vertical_rois(rois: list[Any], *, effort: str = "full") -> dict[str, Any]:
    if len(rois) < 10:
        return {
            "raw_text": "",
            "base10": "",
            "check_digit": "",
            "full_code": "",
            "score": -1.0,
            "is_valid_iso": False,
            "status": "NOT_FOUND",
        }

    rois = rois[:11]
    letter_rois = rois[:4]
    base_rois = rois[4:10]
    check_roi = rois[10] if len(rois) >= 11 else None

    symbol_h = 64 if effort == "full" else 44
    letter_gap = 10 if effort == "full" else 5
    digit_gap = 8 if effort == "full" else 4
    outer_pad = 8 if effort == "full" else 5

    letters_strip = _compose_symbol_strip(letter_rois, target_h=symbol_h, gap=letter_gap, outer_pad=outer_pad)
    digits6_strip = _compose_symbol_strip(base_rois, target_h=symbol_h, gap=digit_gap, outer_pad=outer_pad)
    digits7_strip = _compose_symbol_strip(
        base_rois + ([check_roi] if check_roi is not None else []),
        target_h=symbol_h,
        gap=digit_gap,
        outer_pad=outer_pad,
        square_last=check_roi is not None,
    )
    check_strip = (
        _compose_symbol_strip([check_roi], target_h=symbol_h, gap=0, outer_pad=outer_pad + 2, square_last=True)
        if check_roi is not None
        else None
    )

    owner = ""
    owner_score = -1.0
    owner_raw = ""
    letter_pairs = _collect_vertical_strip_pairs(letters_strip, phase_bonus=0.02, profile="light")
    if letter_pairs:
        owner, owner_score, owner_raw = _best_owner_code_from_pairs(letter_pairs)
    if not owner and effort == "full":
        letter_pairs = _collect_vertical_strip_pairs(letters_strip, phase_bonus=0.01, profile="full")
        if letter_pairs:
            owner, owner_score, owner_raw = _best_owner_code_from_pairs(letter_pairs)

    numeric_raw = ""
    base6 = ""
    check_digit = ""
    numeric_score = -1.0

    digits7_pairs = _collect_vertical_strip_pairs(digits7_strip, phase_bonus=0.08, profile="light")
    local_base6, local_check, local_score, local_raw = _best_vertical_numeric_read(digits7_pairs)
    if local_score > numeric_score:
        base6 = local_base6
        check_digit = local_check
        numeric_score = local_score
        numeric_raw = local_raw

    if not base6 and check_roi is None:
        digits6_pairs = _collect_vertical_strip_pairs(digits6_strip, phase_bonus=0.02, profile="light")
        local_base6, local_check, local_score, local_raw = _best_vertical_numeric_read(digits6_pairs)
        if local_score > numeric_score:
            base6 = local_base6
            check_digit = local_check
            numeric_score = local_score
            numeric_raw = local_raw

    if not base6 and effort == "full":
        digits6_pairs = _collect_vertical_strip_pairs(digits6_strip, phase_bonus=0.0, profile="full")
        local_base6, local_check, local_score, local_raw = _best_vertical_numeric_read(digits6_pairs)
        if local_score > numeric_score:
            base6 = local_base6
            check_digit = local_check
            numeric_score = local_score
            numeric_raw = local_raw

    if base6 and not check_digit and check_strip is not None:
        square_digit, square_score = _best_digit_from_strip(check_strip, profile="minimal")
        if square_digit:
            check_digit = square_digit
            numeric_score = max(numeric_score, square_score)

    raw_text = " ".join(part for part in (owner_raw, numeric_raw) if part).strip()
    if len(owner) != 4 or len(base6) != 6:
        return {
            "raw_text": raw_text,
            "base10": "",
            "check_digit": "",
            "full_code": "",
            "score": -1.0,
            "is_valid_iso": False,
            "status": "RAW_ONLY" if raw_text else "NOT_FOUND",
        }

    base10 = f"{owner}{base6}"
    full_code = f"{base10}{check_digit}" if check_digit else base10
    return {
        "raw_text": raw_text,
        "base10": base10,
        "check_digit": check_digit,
        "full_code": full_code,
        "score": float(max(owner_score, 0.0) + max(numeric_score, 0.0)),
        "is_valid_iso": bool(len(full_code) == 11 and _is_valid_iso6346(full_code)),
        "status": "FULL_11" if len(full_code) == 11 else "PARTIAL_10",
    }


def _vertical_result_rank(result: dict[str, Any]) -> float:
    status = str(result.get("status") or "")
    rank = float(result.get("score", -1.0))
    if result.get("raw_text"):
        rank += 1.0
    if result.get("base10"):
        rank += 10.0
    if status == "PARTIAL_10":
        rank += 2.0
    elif status == "FULL_11":
        rank += 4.0
    if result.get("is_valid_iso"):
        rank += 3.0
    return rank


def _should_try_vertical_grid10(result: dict[str, Any]) -> bool:
    if result.get("base10"):
        return False
    raw = clean(str(result.get("raw_text") or ""))
    if not raw:
        return False
    digit_count = sum(ch.isdigit() for ch in raw)
    letter_count = sum(ch.isalpha() for ch in raw)
    return letter_count == 0 and digit_count == 6


def _ocr_vertical_container_regions(img) -> dict[str, Any]:
    if img is None or img.size == 0:
        return {
            "raw_text": "",
            "base10": "",
            "check_digit": "",
            "full_code": "",
            "score": -1.0,
            "is_valid_iso": False,
            "status": "NOT_FOUND",
        }

    best_result = {
        "raw_text": "",
        "base10": "",
        "check_digit": "",
        "full_code": "",
        "score": -1.0,
        "is_valid_iso": False,
        "status": "NOT_FOUND",
    }
    best_rank = _vertical_result_rank(best_result)
    seen_signatures: set[tuple[tuple[int, int], ...]] = set()

    def _try_rois(rois: list[Any], effort: str) -> dict[str, Any]:
        nonlocal best_result, best_rank
        signature = tuple((int(roi.shape[0]), int(roi.shape[1])) for roi in rois)
        if signature in seen_signatures:
            return best_result
        seen_signatures.add(signature)
        result = _decode_vertical_rois(rois, effort=effort)
        rank = _vertical_result_rank(result)
        if rank > best_rank:
            best_result = result
            best_rank = rank
        return result

    segmented = _extract_vertical_symbol_rois(img)
    segmented_count = len(segmented)

    if 10 <= segmented_count <= 12:
        result = _try_rois(segmented, "full")
        if result.get("base10") and result.get("is_valid_iso"):
            return result
    elif 8 <= segmented_count <= 14:
        result = _try_rois(segmented, "light")
        if result.get("base10") and result.get("is_valid_iso"):
            return result
        grid11 = _vertical_grid_rois(img, 11)
        if grid11:
            result = _try_rois(grid11, "light")
            if result.get("base10") and result.get("is_valid_iso"):
                return result
    else:
        grid11 = _vertical_grid_rois(img, 11)
        if grid11:
            result = _try_rois(grid11, "light")
            if result.get("base10") and result.get("is_valid_iso"):
                return result
            if _should_try_vertical_grid10(result):
                grid10 = _vertical_grid_rois(img, 10)
                if grid10:
                    result = _try_rois(grid10, "light")
                    if result.get("base10") and result.get("is_valid_iso"):
                        return result

    return best_result


def _ocr_twoline_container_regions(img, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if img is None or img.size == 0:
        return {
            "raw_text": "",
            "base10": "",
            "check_digit": "",
            "full_code": "",
            "score": -1.0,
            "is_valid_iso": False,
            "status": "NOT_FOUND",
        }

    top_roi, bottom_roi = extract_two_line_rois(img)
    if top_roi is None or bottom_roi is None:
        return _ocr_split_door_container_regions(img, config=config)

    top_symbols = _extract_horizontal_symbol_rois(top_roi)
    bottom_symbols = _extract_horizontal_symbol_rois(bottom_roi)

    owner_input = (
        _compose_symbol_strip(top_symbols[:4], target_h=64, gap=10, outer_pad=8)
        if len(top_symbols) >= 4
        else top_roi
    )

    base_symbols = bottom_symbols[:6] if len(bottom_symbols) >= 6 else []
    check_symbol = bottom_symbols[6] if len(bottom_symbols) >= 7 else None
    use_symbol_bottom = len(base_symbols) == 6
    digits6_input = (
        _compose_symbol_strip(base_symbols, target_h=64, gap=8, outer_pad=8)
        if use_symbol_bottom
        else bottom_roi
    )
    digits7_input = (
        _compose_symbol_strip(
            base_symbols + ([check_symbol] if check_symbol is not None else []),
            target_h=64,
            gap=8,
            outer_pad=8,
            square_last=(check_symbol is not None),
        )
        if use_symbol_bottom
        else bottom_roi
    )
    check_input = (
        _compose_symbol_strip([check_symbol], target_h=64, gap=0, outer_pad=10, square_last=True)
        if check_symbol is not None
        else None
    )

    owner = ""
    owner_score = -1.0
    owner_raw = ""
    letter_pairs = _collect_compact_strip_pairs(owner_input, phase_bonus=0.03, profile="light")
    if letter_pairs:
        owner, owner_score, owner_raw = _best_owner_code_from_pairs(letter_pairs)

    numeric_candidates: list[dict[str, Any]] = []
    numeric_raw = ""
    for strip, bonus, profile in (
        (digits7_input, 0.06, "light"),
        (digits6_input, 0.04, "light"),
        (bottom_roi, 0.02, "minimal"),
    ):
        pairs = _collect_compact_strip_pairs(strip, phase_bonus=bonus, profile=profile)
        if pairs and len(" ".join(text for text, _ in pairs)) > len(numeric_raw):
            numeric_raw = " ".join(text for text, _ in pairs).strip()
        numeric_candidates.extend(_numeric_region_candidates(pairs, phase_bonus=bonus))

    base6, check_digit, numeric_score, numeric_best_raw, _meta = _pick_best_numeric_candidate(numeric_candidates)
    if numeric_best_raw and len(numeric_best_raw) > len(numeric_raw):
        numeric_raw = numeric_best_raw

    if base6 and not check_digit:
        right_digit = ""
        right_score = -1.0
        if check_input is not None:
            right_digit, right_score = _best_digit_from_strip(check_input)
        if not right_digit:
            right_digit, right_score = _read_right_check_digit(bottom_roi)
        if right_digit:
            check_digit = right_digit
            numeric_score = max(numeric_score, right_score)

    raw_text = " ".join(part for part in (owner_raw, numeric_raw) if part).strip()
    if len(owner) != 4 or len(base6) != 6:
        split_door = _ocr_split_door_container_regions(img, config=config)
        if split_door.get("base10") or split_door.get("raw_text"):
            return split_door
        return {
            "raw_text": raw_text,
            "base10": "",
            "check_digit": "",
            "full_code": "",
            "score": -1.0,
            "is_valid_iso": False,
            "status": "RAW_ONLY" if raw_text else "NOT_FOUND",
        }

    base10 = f"{owner}{base6}"
    full_code = f"{base10}{check_digit}" if check_digit else base10
    return {
        "raw_text": raw_text,
        "base10": base10,
        "check_digit": check_digit,
        "full_code": full_code,
        "score": float(max(owner_score, 0.0) + max(numeric_score, 0.0)),
        "is_valid_iso": bool(len(full_code) == 11 and _is_valid_iso6346(full_code)),
        "status": "FULL_11" if len(full_code) == 11 else "PARTIAL_10",
    }


def _owner_code_candidates(raw: str) -> list[tuple[str, float]]:
    part = clean(raw)
    if not part:
        return []

    mapped = "".join(_TO_LETTER.get(ch, ch) for ch in part)
    letters_only = "".join(ch for ch in mapped if ch.isalpha())
    if len(letters_only) < 4:
        return []

    out: list[tuple[str, float]] = []
    seen: set[str] = set()

    def add(candidate: str, bonus: float) -> None:
        candidate = _force_owner_category_u(candidate)
        if len(candidate) != 4 or not candidate.isalpha() or candidate in seen:
            return
        seen.add(candidate)
        out.append((candidate, bonus))

    if len(letters_only) == 4:
        add(letters_only, 0.12)
    if len(letters_only) == 5:
        for idx in range(5):
            add(letters_only[:idx] + letters_only[idx + 1 :], 0.07)
    for idx in range(len(letters_only) - 3):
        add(letters_only[idx : idx + 4], 0.03)

    return out


def _best_owner_code_from_pairs(text_pairs: list[tuple[str, float]]) -> tuple[str, float, str]:
    best_code = ""
    best_score = -1.0
    best_raw = ""

    for raw_text, raw_score in _candidate_groups(text_pairs):
        for candidate, bonus in _owner_code_candidates(raw_text):
            final = float(raw_score) + bonus
            if len(clean(raw_text)) == 4:
                final += 0.03
            if final > best_score:
                best_code = candidate
                best_score = final
                best_raw = raw_text

    return best_code, best_score, best_raw


def _numeric_region_candidates(
    text_pairs: list[tuple[str, float]],
    *,
    phase_bonus: float = 0.0,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    six_chunks: list[tuple[int, str, float]] = []
    check_chunks: list[tuple[int, str, float]] = []

    def add_candidate(
        base6: str,
        check_digit: str,
        score: float,
        raw: str,
        *,
        direct: bool = False,
        source: str = "unknown",
    ) -> None:
        if not base6 or len(base6) != 6 or not base6.isdigit():
            return
        if check_digit and (len(check_digit) != 1 or not check_digit.isdigit()):
            return
        candidates.append(
            {
                "base6": base6,
                "check_digit": check_digit,
                "score": float(score),
                "raw": raw,
                "direct": direct,
                "source": source,
            }
        )

    for idx, (raw_text, raw_score) in enumerate(text_pairs):
        mapped = "".join(_TO_DIGIT.get(ch, ch) for ch in clean(raw_text))
        chunks = _DIGIT_CHUNK_RE.findall(mapped)
        for chunk in chunks:
            score = float(raw_score) + phase_bonus
            if len(chunk) == 7:
                add_candidate(chunk[:6], chunk[6], score + 0.14, chunk, direct=True, source="direct7")
            elif len(chunk) == 6:
                six_chunks.append((idx, chunk, score))
                add_candidate(chunk, "", score + 0.04, chunk, direct=False, source="base6")
            elif len(chunk) == 1:
                check_chunks.append((idx, chunk, score + 0.03))
            elif len(chunk) == 2:
                check_chunks.append((idx, chunk[0], score - 0.05))
                check_chunks.append((idx, chunk[-1], score - 0.03))
            elif len(chunk) > 7:
                add_candidate(chunk[:6], chunk[-1], score - 0.02, chunk, direct=False, source="long")

    for six_idx, base6, six_score in six_chunks:
        for check_idx, check_digit, check_score in check_chunks:
            combo = (0.82 * six_score) + (0.18 * max(check_score, 0.0)) + 0.05
            if check_idx > six_idx:
                combo += 0.02
            add_candidate(base6, check_digit, combo, f"{base6}{check_digit}", direct=False, source="combo")

    return candidates


def _pick_best_numeric_candidate(candidates: list[dict[str, Any]]) -> tuple[str, str, float, str, dict[str, Any]]:
    if not candidates:
        return "", "", -1.0, "", {
            "has_direct_full": False,
            "check_vote_count": 0,
            "check_margin": 0.0,
        }

    base_votes: dict[str, float] = {}
    base_best: dict[str, tuple[float, str]] = {}
    check_votes: dict[tuple[str, str], float] = {}
    has_direct_full = False

    for item in candidates:
        base6 = item["base6"]
        score = float(item["score"])
        raw = str(item["raw"])
        base_votes[base6] = base_votes.get(base6, 0.0) + score
        best_score, _best_raw = base_best.get(base6, (-1.0, ""))
        if score > best_score:
            base_best[base6] = (score, raw)
        if item["check_digit"]:
            check_key = (base6, item["check_digit"])
            vote_bonus = 0.10 if item.get("direct") else 0.0
            check_votes[check_key] = check_votes.get(check_key, 0.0) + score + vote_bonus
            if item.get("direct"):
                has_direct_full = True

    best_base6 = max(base_votes.items(), key=lambda kv: kv[1])[0]
    best_raw = base_best.get(best_base6, (-1.0, ""))[1]

    per_base_digit_votes = [
        (digit, vote)
        for (base6, digit), vote in check_votes.items()
        if base6 == best_base6 and digit
    ]
    per_base_digit_votes.sort(key=lambda item: item[1], reverse=True)

    best_digit = ""
    best_score = base_votes[best_base6]
    if per_base_digit_votes:
        best_digit = per_base_digit_votes[0][0]
        best_score = per_base_digit_votes[0][1]

    margin = 0.0
    if len(per_base_digit_votes) >= 2:
        margin = per_base_digit_votes[0][1] - per_base_digit_votes[1][1]
    elif per_base_digit_votes:
        margin = per_base_digit_votes[0][1]

    meta = {
        "has_direct_full": has_direct_full,
        "check_vote_count": len(per_base_digit_votes),
        "check_margin": float(margin),
    }
    return best_base6, best_digit, float(best_score), best_raw, meta


def _collect_right_check_digit_votes(img, *, include_zone: bool = False) -> list[dict[str, Any]]:
    if img is None or img.size == 0:
        return []

    votes: list[dict[str, Any]] = []
    zones = [
        ("primary", _extract_zone(img, 0.72, 1.00, 0.03, 0.97)),
        ("mid", _extract_zone(img, 0.76, 1.00, 0.05, 0.95)),
        ("tight", _extract_zone(img, 0.84, 1.00, 0.02, 0.98)),
    ]

    for zone_name, zone in zones:
        if zone is None or not zone.size:
            continue

        for square_idx, (_rect, square_roi) in enumerate(_square_like_roi_entries(zone)[:2]):
            roi_candidates = [("square", square_roi)]
            inner = _safe_inner_crop(square_roi, margin_ratio=0.06)
            if inner is not None and inner.size:
                roi_candidates.append(("square_inner", inner))
            for roi_kind, roi in roi_candidates:
                digit, score = _read_digit_from_roi(roi, square_bias=True)
                if digit:
                    votes.append(
                        {
                            "digit": digit,
                            "score": float(score),
                            "source": roi_kind,
                            "zone": zone_name,
                            "order": square_idx,
                        }
                    )

        if include_zone:
            digit, score = _paddle_digit_from_roi(zone)
            if digit:
                votes.append(
                    {
                        "digit": digit,
                        "score": float(score),
                        "source": "zone",
                        "zone": zone_name,
                        "order": 99,
                    }
                )

    return votes


def _resolve_check_digit_for_base(
    base6: str,
    numeric_candidates: list[dict[str, Any]],
    right_side_votes: list[dict[str, Any]],
) -> tuple[str, float, dict[str, Any]]:
    if not base6:
        return "", -1.0, {"vote_count": 0, "margin": 0.0, "direct_count": 0, "square_count": 0}

    digit_votes: dict[str, float] = {}
    direct_count = 0
    square_count = 0
    numeric_digits_seen: set[str] = set()

    for item in numeric_candidates:
        if item.get("base6") != base6 or not item.get("check_digit"):
            continue
        digit = str(item["check_digit"])
        numeric_digits_seen.add(digit)
        score = float(item["score"])
        source = str(item.get("source") or "")
        if source == "direct7":
            weight = score + 0.34
            direct_count += 1
        elif source == "long":
            weight = score + 0.10
        elif source == "combo":
            weight = max(0.0, score * 0.74)
        else:
            weight = max(0.0, score * 0.66)
        digit_votes[digit] = digit_votes.get(digit, 0.0) + weight

    if right_side_votes:
        direct_digits = {
            str(item["check_digit"])
            for item in numeric_candidates
            if item.get("base6") == base6 and item.get("source") == "direct7" and item.get("check_digit")
        }
        for vote in right_side_votes:
            digit = str(vote["digit"])
            if numeric_digits_seen and digit not in numeric_digits_seen:
                continue
            score = float(vote["score"])
            source = str(vote.get("source") or "")
            if source.startswith("square"):
                weight = score + 0.18
                square_count += 1
            else:
                weight = max(0.0, score - 0.12)
            if direct_digits and digit in direct_digits:
                weight += 0.08
            digit_votes[digit] = digit_votes.get(digit, 0.0) + weight

    if not digit_votes:
        return "", -1.0, {"vote_count": 0, "margin": 0.0, "direct_count": direct_count, "square_count": square_count}

    ranked = sorted(digit_votes.items(), key=lambda item: item[1], reverse=True)
    best_digit, best_score = ranked[0]
    margin = ranked[0][1] - ranked[1][1] if len(ranked) >= 2 else ranked[0][1]
    meta = {
        "vote_count": len(ranked),
        "margin": float(margin),
        "direct_count": direct_count,
        "square_count": square_count,
    }
    return best_digit, float(best_score), meta


def _ocr_horizontal_container_regions(expanded, *, enable_resize_digit_pass: bool = True) -> dict[str, Any]:
    best_raw = ""

    letters_zone = _extract_zone(expanded, 0.02, 0.40, 0.02, 0.98)
    owner = ""
    owner_score = -1.0
    owner_raw = ""
    if letters_zone is not None and letters_zone.size:
        letter_passes = [(_preprocess_fast(letters_zone), 0.0)]
        for prepared, phase_bonus in letter_passes:
            pairs = _extract_text_pairs(_run_ocr(prepared))
            if pairs and len(" ".join(text for text, _ in pairs)) > len(best_raw):
                best_raw = " ".join(text for text, _ in pairs).strip()
            local_owner, local_score, local_raw = _best_owner_code_from_pairs(pairs)
            if local_owner:
                final = local_score + phase_bonus
                if final > owner_score:
                    owner = local_owner
                    owner_score = final
                    owner_raw = local_raw
        needs_hard_letters = (
            len(owner) != 4
            or owner_score < 1.145
            or len(clean(owner_raw)) != 4
        )
        if needs_hard_letters:
            pairs = _extract_text_pairs(_run_ocr(_preprocess_hard(letters_zone)))
            if pairs and len(" ".join(text for text, _ in pairs)) > len(best_raw):
                best_raw = " ".join(text for text, _ in pairs).strip()
            local_owner, local_score, local_raw = _best_owner_code_from_pairs(pairs)
            if local_owner and local_score > owner_score:
                owner = local_owner
                owner_score = local_score
                owner_raw = local_raw

    numeric_candidates: list[dict[str, Any]] = []
    fast_digit_zones: list[tuple[Any, float, float]] = []
    digit_specs_fast = [
        (0.30, 1.00, _preprocess_fast, 0.03),
        (0.34, 1.00, _preprocess_fast, 0.02),
        (0.40, 1.00, _preprocess_fast, 0.00),
    ]
    for x1, x2, preprocessor, phase_bonus in digit_specs_fast:
        zone = _extract_zone(expanded, x1, x2, 0.02, 0.98)
        if zone is None or not zone.size:
            continue
        pairs = _extract_text_pairs(_run_ocr(preprocessor(zone)))
        if pairs and len(" ".join(text for text, _ in pairs)) > len(best_raw):
            best_raw = " ".join(text for text, _ in pairs).strip()
        numeric_candidates.extend(_numeric_region_candidates(pairs, phase_bonus=phase_bonus))
        fast_digit_zones.append((zone, phase_bonus, x1))

    base6, check_digit, numeric_score, numeric_raw, numeric_meta = _pick_best_numeric_candidate(numeric_candidates)

    has_zero_check_votes = numeric_meta.get("check_vote_count", 0) <= 0
    needs_resize_digits = (
        not base6
        or (
            not check_digit
            and not has_zero_check_votes
        )
        or (
            check_digit
            and not numeric_meta["has_direct_full"]
            and numeric_meta.get("check_margin", 0.0) < 1.02
        )
    )
    if enable_resize_digit_pass and needs_resize_digits:
        resize_digit_zones = fast_digit_zones
        if check_digit and numeric_meta.get("check_vote_count", 0) <= 1:
            resize_digit_zones = [item for item in fast_digit_zones if item[2] <= 0.30]
        for zone, phase_bonus, _x1 in resize_digit_zones:
            pairs = _collect_strip_pairs(zone, phase_bonus=phase_bonus, profile="light")
            if pairs and len(" ".join(text for text, _ in pairs)) > len(best_raw):
                best_raw = " ".join(text for text, _ in pairs).strip()
            numeric_candidates.extend(_numeric_region_candidates(pairs, phase_bonus=phase_bonus))
        base6, check_digit, numeric_score, numeric_raw, numeric_meta = _pick_best_numeric_candidate(numeric_candidates)

    needs_hard_digits = (
        not base6
        or not check_digit
        or (
            not numeric_meta["has_direct_full"]
            and numeric_meta["check_vote_count"] > 1
            and numeric_meta.get("check_margin", 0.0) < 0.90
        )
    )
    if needs_hard_digits:
        digit_specs_hard = [
            (0.30, 1.00, _preprocess_hard, 0.01),
            (0.34, 1.00, _preprocess_hard, 0.02),
            (0.40, 1.00, _preprocess_hard, 0.00),
        ]
        for x1, x2, preprocessor, phase_bonus in digit_specs_hard:
            zone = _extract_zone(expanded, x1, x2, 0.02, 0.98)
            if zone is None or not zone.size:
                continue
            pairs = _extract_text_pairs(_run_ocr(preprocessor(zone)))
            if pairs and len(" ".join(text for text, _ in pairs)) > len(best_raw):
                best_raw = " ".join(text for text, _ in pairs).strip()
            numeric_candidates.extend(_numeric_region_candidates(pairs, phase_bonus=phase_bonus))
        base6, check_digit, numeric_score, numeric_raw, numeric_meta = _pick_best_numeric_candidate(numeric_candidates)

    should_consult_square = bool(
        base6
        and (
            not check_digit
            or (
                not numeric_meta.get("has_direct_full")
                and (
                    numeric_meta.get("check_vote_count", 0) <= 0
                    or numeric_meta.get("check_margin", 0.0) < 1.00
                )
            )
        )
    )
    if should_consult_square:
        right_side_votes = _collect_right_check_digit_votes(expanded, include_zone=False)
        resolved_digit, resolved_score, resolved_meta = _resolve_check_digit_for_base(base6, numeric_candidates, right_side_votes)
        if resolved_digit:
            check_digit = resolved_digit
            numeric_score = max(numeric_score, resolved_score)
            numeric_meta["resolved_vote_count"] = resolved_meta["vote_count"]
            numeric_meta["resolved_margin"] = resolved_meta["margin"]
            numeric_meta["resolved_direct_count"] = resolved_meta["direct_count"]
            numeric_meta["resolved_square_count"] = resolved_meta["square_count"]
    else:
        numeric_meta["resolved_vote_count"] = 0
        numeric_meta["resolved_margin"] = 0.0
        numeric_meta["resolved_direct_count"] = 0
        numeric_meta["resolved_square_count"] = 0

    full_code = ""
    if owner and base6:
        full_code = f"{owner}{base6}{check_digit}" if check_digit else f"{owner}{base6}"

    if owner_raw and numeric_raw:
        best_raw = f"{owner_raw} {numeric_raw}".strip()
    elif owner_raw:
        best_raw = owner_raw
    elif numeric_raw:
        best_raw = numeric_raw

    combined_score = max(owner_score, 0.0) + max(numeric_score, 0.0)
    return {
        "raw_text": best_raw,
        "base10": f"{owner}{base6}" if owner and base6 else "",
        "check_digit": check_digit if full_code and len(full_code) == 11 else "",
        "full_code": full_code,
        "score": float(combined_score),
        "is_valid_iso": bool(len(full_code) == 11 and _is_valid_iso6346(full_code)),
        "status": "FULL_11" if len(full_code) == 11 else ("PARTIAL_10" if len(full_code) == 10 else ("RAW_ONLY" if best_raw else "NOT_FOUND")),
    }


def _expand_crop(img, left_ratio: float = 0.10, right_ratio: float = 0.12, top_ratio: float = 0.07, bottom_ratio: float = 0.07):
    h, w = img.shape[:2]
    left = max(1, int(w * left_ratio))
    right = max(1, int(w * right_ratio))
    top = max(1, int(h * top_ratio))
    bottom = max(1, int(h * bottom_ratio))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)


def _rotate_image(img, angle_deg: float):
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _best_digit_from_pairs(text_pairs: list[tuple[str, float]]) -> tuple[str, float]:
    best_digit = ""
    best_score = -1.0
    for text, score in text_pairs:
        s = clean(text)
        mapped = "".join(_TO_DIGIT.get(ch, ch) for ch in s)
        digits = [(idx, ch) for idx, ch in enumerate(mapped) if ch.isdigit()]
        if not digits:
            continue

        for idx, digit in digits:
            candidate_score = float(score)
            if len(digits) == 1:
                candidate_score += 0.12
            if idx == len(mapped) - 1:
                candidate_score += 0.04
            if len(mapped) <= 2:
                candidate_score += 0.04
            if len(digits) > 1:
                candidate_score -= 0.03 * float(len(digits) - 1)

            if candidate_score > best_score:
                best_digit = digit
                best_score = candidate_score
    return best_digit, best_score


def _gray_to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _digit_preprocess_variants(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 40, 40)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        3,
    )
    return [img, _gray_to_bgr(gray), _gray_to_bgr(otsu_inv), _gray_to_bgr(adaptive)]


def _extract_zone(img, x_from: float, x_to: float, y_from: float, y_to: float):
    h, w = img.shape[:2]
    x1 = max(0, int(w * x_from))
    x2 = min(w, int(w * x_to))
    y1 = max(0, int(h * y_from))
    y2 = min(h, int(h * y_to))
    if x2 <= x1 or y2 <= y1:
        return None
    zone = img[y1:y2, x1:x2]
    if zone.size == 0:
        return None
    return zone


def _safe_inner_crop(img, margin_ratio: float = 0.16):
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return None
    dx = max(1, int(w * margin_ratio))
    dy = max(1, int(h * margin_ratio))
    if dx * 2 >= w or dy * 2 >= h:
        return None
    cropped = img[dy:h - dy, dx:w - dx]
    if cropped.size == 0:
        return None
    return cropped


def _rect_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter / max(1.0, area_a + area_b - inter)


def _square_like_roi_entries(zone) -> list[tuple[tuple[int, int, int, int], Any]]:
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    zh, zw = zone.shape[:2]
    zone_area = float(max(1, zh * zw))
    raw_entries: list[tuple[float, tuple[int, int, int, int], Any]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w <= 2 or h <= 2:
            continue
        area = float(w * h)
        ratio = w / float(h)
        if area < zone_area * 0.01 or area > zone_area * 0.45:
            continue
        if ratio < 0.62 or ratio > 1.45:
            continue
        if x < int(zw * 0.08):
            continue

        pad = int(max(w, h) * 0.14)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(zw, x + w + pad)
        y2 = min(zh, y + h + pad)
        roi = zone[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        raw_entries.append((area, (x1, y1, x2, y2), roi))

    raw_entries.sort(key=lambda it: it[0], reverse=True)
    deduped: list[tuple[tuple[int, int, int, int], Any]] = []
    for _area, rect, roi in raw_entries:
        if any(_rect_iou(rect, kept_rect) >= 0.72 for kept_rect, _ in deduped):
            continue
        deduped.append((rect, roi))
        if len(deduped) >= 2:
            break
    return deduped


def _square_like_rois(zone):
    return [roi for _, roi in _square_like_roi_entries(zone)]


def _tesseract_digit_from_image(img, *, psm: int = 10) -> str:
    exe = _resolve_tesseract_executable()
    if not exe or img is None or img.size == 0:
        return ""

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
        cv2.imwrite(temp_path, img)
        completed = subprocess.run(
            [
                exe,
                temp_path,
                "stdout",
                "--psm",
                str(psm),
                "--oem",
                "1",
                "-c",
                "tessedit_char_whitelist=0123456789",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return ""
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    text = "".join(ch for ch in completed.stdout if ch.isdigit())
    return text if len(text) == 1 else ""


def _tesseract_vote_digit(
    roi,
    *,
    base_weight: float,
    preprocess_indices: tuple[int, ...],
    scales: tuple[float, ...],
) -> tuple[str, float]:
    votes: dict[str, float] = {}
    variants = _digit_preprocess_variants(roi)
    for variant_idx in preprocess_indices:
        if variant_idx >= len(variants):
            continue
        prepared = variants[variant_idx]
        for scale in scales:
            up = cv2.resize(prepared, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            digit = _tesseract_digit_from_image(up, psm=10)
            if not digit:
                continue
            weight = base_weight
            if variant_idx in (1, 2):
                weight += 0.45
            elif variant_idx == 3:
                weight += 0.15
            votes[digit] = votes.get(digit, 0.0) + weight

    if not votes:
        return "", -1.0
    return max(votes.items(), key=lambda kv: kv[1])


def _paddle_digit_from_roi(roi) -> tuple[str, float]:
    if roi is None or roi.size == 0:
        return "", -1.0

    best_digit = ""
    best_score = -1.0
    variants = _digit_preprocess_variants(roi)
    # Limit digit OCR to the most useful variants to keep latency reasonable.
    for variant_idx in (1, 3, 0):
        if variant_idx >= len(variants):
            continue
        prepared = variants[variant_idx]
        up = cv2.resize(prepared, None, fx=2.8, fy=2.8, interpolation=cv2.INTER_CUBIC)
        pairs = _extract_text_pairs(_run_ocr(up))
        digit, score = _best_digit_from_pairs(pairs)
        if digit and score > best_score:
            best_digit = digit
            best_score = score
    return best_digit, best_score


def _paddle_digit_from_square_roi(roi) -> tuple[str, float]:
    if roi is None or roi.size == 0:
        return "", -1.0

    best_digit = ""
    best_score = -1.0
    variants = _digit_preprocess_variants(roi)
    # For boxed check-digits adaptive/inverted variants often deform "4" into "7".
    for variant_idx in (0, 1):
        if variant_idx >= len(variants):
            continue
        prepared = variants[variant_idx]
        for scale, bonus in ((3.0, 0.03), (3.6, 0.05)):
            up = cv2.resize(prepared, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            pairs = _extract_text_pairs(_run_ocr(up))
            digit, score = _best_digit_from_pairs(pairs)
            if digit and (score + bonus) > best_score:
                best_digit = digit
                best_score = score + bonus
    return best_digit, best_score


def _read_digit_from_roi(roi, *, square_bias: bool = False) -> tuple[str, float]:
    if square_bias:
        digit, score = _paddle_digit_from_square_roi(roi)
    else:
        digit, score = _paddle_digit_from_roi(roi)
    if not digit:
        return "", -1.0
    adjusted = score + (0.03 if square_bias else 0.0)
    return digit, adjusted


def _read_right_check_digit(img) -> tuple[str, float]:
    votes = _collect_right_check_digit_votes(img, include_zone=True)
    if not votes:
        return "", -1.0
    digit_totals: dict[str, float] = {}
    for vote in votes:
        digit = str(vote["digit"])
        score = float(vote["score"])
        source = str(vote.get("source") or "")
        weight = score + 0.08 if source.startswith("square") else max(0.0, score - 0.12)
        digit_totals[digit] = digit_totals.get(digit, 0.0) + weight
    return max(digit_totals.items(), key=lambda item: item[1])


def _best_container_from_result(result) -> tuple[str, float, bool, dict[str, Any]]:
    text_pairs = _extract_text_pairs(result)
    debug: dict[str, Any] = {
        "text_pair_count": len(text_pairs),
        "text_pairs": _text_pair_preview(text_pairs),
    }
    if not text_pairs:
        debug["reason"] = "ocr_no_text_pairs"
        return "", -1.0, False, debug

    best_code = ""
    best_final = -1.0
    best_valid = False
    candidate_groups = _candidate_groups(text_pairs)
    debug["candidate_group_count"] = len(candidate_groups)
    debug["candidate_groups"] = [raw for raw, _score in candidate_groups[:8]]
    normalized_hits: list[str] = []

    for raw_candidate, raw_score in candidate_groups:
        for normalized in _normalized_candidates(raw_candidate):
            if normalized not in normalized_hits and len(normalized_hits) < 8:
                normalized_hits.append(normalized)
            quality = _container_text_quality(normalized)
            final = 0.75 * float(raw_score) + 0.25 * quality
            is_valid = _is_valid_iso6346(normalized)
            if is_valid:
                final += 0.10
            if final > best_final:
                best_code = normalized
                best_final = final
                best_valid = is_valid

    debug["normalized_candidate_count"] = len(normalized_hits)
    debug["normalized_candidates"] = normalized_hits
    if best_code:
        debug["reason"] = "ok"
        debug["best_code"] = best_code
        debug["best_score"] = round(float(best_final), 4)
        debug["best_valid"] = bool(best_valid)
    else:
        debug["reason"] = "ocr_text_pairs_no_container_candidate"
    return best_code, best_final, best_valid, debug


def _predict_image(img) -> tuple[str, float, bool, dict[str, Any]]:
    result = _run_ocr(img)
    return _best_container_from_result(result)


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


def ocr_container_best_details(img) -> dict[str, Any]:
    if img is None:
        return {
            "text": "",
            "score": -1.0,
            "is_valid": False,
            "ocr_debug": {"reason": "image_none"},
        }

    key = _cache_key(img)
    cached = _cache_get(key)
    if cached is not None:
        text, score, is_valid, ocr_debug = cached
        return {
            "text": text,
            "score": score,
            "is_valid": is_valid,
            "ocr_debug": dict(ocr_debug),
        }

    best_text = ""
    best_score = -1.0
    best_valid = False
    best_debug: dict[str, Any] = {"reason": "uninitialized"}

    img_fast = _preprocess_fast(img)
    text, score, is_valid, ocr_debug = _predict_image(img_fast)
    ocr_debug = dict(ocr_debug)
    ocr_debug["variant"] = "fast"
    if score > best_score:
        best_text, best_score, best_valid, best_debug = text, score, is_valid, ocr_debug

    if best_valid and best_score >= EARLY_OK:
        result = (best_text, best_score, best_valid, best_debug)
        _cache_set(key, result)
        return {
            "text": best_text,
            "score": best_score,
            "is_valid": best_valid,
            "ocr_debug": dict(best_debug),
        }

    img_hard = _preprocess_hard(img)
    text, score, is_valid, ocr_debug = _predict_image(img_hard)
    ocr_debug = dict(ocr_debug)
    ocr_debug["variant"] = "hard"
    if score > best_score:
        best_text, best_score, best_valid, best_debug = text, score, is_valid, ocr_debug

    result = (best_text, best_score, best_valid, best_debug)
    _cache_set(key, result)
    return {
        "text": best_text,
        "score": best_score,
        "is_valid": best_valid,
        "ocr_debug": dict(best_debug),
    }


def ocr_container_best(img):
    details = ocr_container_best_details(img)
    return (
        str(details.get("text") or ""),
        float(details.get("score") or -1.0),
        bool(details.get("is_valid")),
    )


def _ocr_container_from_crop_generic(img) -> dict[str, Any]:
    if img is None:
        return {
            "raw_text": "",
            "base10": "",
            "check_digit": "",
            "full_code": "",
            "score": -1.0,
            "is_valid_iso": False,
            "status": "NOT_FOUND",
        }

    best_raw = ""
    best_norm = ""
    best_base10 = ""
    best_pairs: list[tuple[str, float]] = []
    best_score = -1.0
    best_valid = False

    def _evaluate_variant(base_img):
        nonlocal best_raw, best_norm, best_base10, best_pairs, best_score, best_valid
        for prepared in (_preprocess_fast(base_img), _preprocess_hard(base_img)):
            result = _run_ocr(prepared)
            text_pairs = _extract_text_pairs(result)
            raw_text = " ".join(text for text, _ in text_pairs).strip()

            local_best_norm = ""
            local_best_base10 = ""
            local_score = -1.0
            local_valid = False

            for raw_candidate, raw_score in _candidate_groups(text_pairs):
                for normalized in _normalized_candidates(raw_candidate):
                    quality = _container_text_quality(normalized)
                    final = 0.8 * float(raw_score) + 0.2 * quality
                    is_valid = _is_valid_iso6346(normalized)
                    if final > local_score:
                        local_best_norm = normalized
                        local_best_base10 = normalized[:10]
                        local_score = final
                        local_valid = is_valid

                for base10 in _normalized_base_candidates(raw_candidate):
                    quality10 = _container_base_quality(base10)
                    final10 = 0.8 * float(raw_score) + 0.2 * quality10 - 0.02
                    if final10 > local_score:
                        local_best_norm = ""
                        local_best_base10 = base10
                        local_score = final10
                        local_valid = False

            rank = local_score
            if rank <= -1.0 and text_pairs:
                # Keep strongest raw pass even without normalized candidate.
                rank = max(float(s) for _, s in text_pairs) * 0.5

            if rank > best_score:
                best_raw = raw_text
                best_norm = local_best_norm
                best_base10 = local_best_base10
                best_pairs = list(text_pairs)
                best_score = rank
                best_valid = local_valid

    expanded = _expand_crop(img)
    variants = [img, expanded]
    if img.shape[1] > 0:
        variants.append(cv2.resize(expanded, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC))
    for base_img in variants:
        _evaluate_variant(base_img)

    # Hard fallback: tiny skew often kills first letters on already-cropped images.
    if not best_base10 and not best_norm:
        for angle in (-3.0, 3.0):
            _evaluate_variant(_rotate_image(expanded, angle))
            if best_base10 or best_norm:
                break

    if best_base10:
        # Always prefer the specialized right-side digit reader over a full-code guess
        # produced by the general OCR stream.
        digit, digit_score = _read_right_check_digit(expanded)
        if digit:
            best_norm = f"{best_base10}{digit}"
            best_score = max(best_score, digit_score)

    result = best_norm or best_base10
    base10 = ""
    check_digit = ""
    if result and len(result) >= 10 and result[:4].isalpha() and result[4:10].isdigit():
        base10 = result[:10]
    if result and len(result) == 11 and result[-1].isdigit():
        check_digit = result[-1]

    if len(result) == 11:
        status = "FULL_11"
    elif len(result) == 10:
        status = "PARTIAL_10"
    elif best_raw:
        status = "RAW_ONLY"
    else:
        status = "NOT_FOUND"

    return {
        "raw_text": best_raw,
        "base10": base10,
        "check_digit": check_digit,
        "full_code": result if result else "",
        "score": float(best_score),
        "is_valid_iso": bool(best_valid),
        "status": status,
    }


def ocr_container_from_crop_details(img, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    OCR for already-cropped container region.
    Returns:
      raw_text, base10, check_digit, full_code, score, is_valid_iso, status
    """
    if img is None:
        return {
            "raw_text": "",
            "base10": "",
            "check_digit": "",
            "full_code": "",
            "score": -1.0,
            "is_valid_iso": False,
            "status": "NOT_FOUND",
        }

    expanded = _expand_crop(img)
    runtime_config = config or {}
    enable_resize_digit_pass = bool(runtime_config.get("enable_resize_digit_pass", True))
    layout_type = classify_container_layout(img, runtime_config)
    if layout_type == "oneline" or _is_horizontal_crop(img):
        horizontal = _ocr_horizontal_container_regions(
            expanded,
            enable_resize_digit_pass=enable_resize_digit_pass,
        )
        if horizontal.get("base10"):
            return horizontal
    if layout_type == "twolines":
        twolines = _ocr_twoline_container_regions(img, config=runtime_config)
        if twolines.get("base10"):
            return twolines
        if twolines.get("raw_text"):
            return twolines
    if layout_type == "vertical" or _is_vertical_crop(img):
        vertical = _ocr_vertical_container_regions(img)
        if vertical.get("base10"):
            return vertical
        if vertical.get("raw_text") and layout_type != "oneline":
            return vertical

    return _ocr_container_from_crop_generic(img)


def ocr_container_from_crop(img) -> tuple[str, str, float, bool]:
    details = ocr_container_from_crop_details(img)
    return (
        details.get("raw_text", ""),
        details.get("full_code", ""),
        float(details.get("score", -1.0)),
        bool(details.get("is_valid_iso", False)),
    )
