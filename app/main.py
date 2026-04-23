import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import OCR
from . import container_crop_service
from . import container_ocr
from . import crop_service

app = FastAPI(title="FastService OCR-like API")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dashboard_data"
HISTORY_DB_PATH = DATA_DIR / "recognitions.db"
DEBUG_IMAGE_PATH = DATA_DIR / "last_debug.jpg"
DEBUG_CROP_IMAGE_PATH = DATA_DIR / "last_debug_crop.jpg"
DEBUG_SEGMENTATION_IMAGE_PATH = DATA_DIR / "last_debug_segmentation.jpg"
DEBUG_META_PATH = DATA_DIR / "last_debug.json"
CONTAINER_CONFIG_PATH = Path(os.getenv("CONTAINER_CONFIG_PATH", PROJECT_ROOT / "app" / "container_recognition_config.json"))

_DEFAULT_LOG_PATH = PROJECT_ROOT / "app.log"
LOG_PATH = Path(os.getenv("LOG_PATH", str(_DEFAULT_LOG_PATH)))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
if not any(
    isinstance(h, logging.FileHandler) and Path(h.baseFilename) == LOG_PATH
    for h in logger.handlers
):
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    logger.addHandler(file_handler)
logger.propagate = False

MAX_FILES = int(os.getenv("MAX_FILES", "20"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
SIMULATED_DELAY_SEC = float(os.getenv("SIMULATED_DELAY_SEC", "0"))
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

_DEFAULT_CONTAINER_CONFIG: dict[str, Any] = {
    "use_crop_service": True,
    "max_crops_per_image": 8,
    "stop_on_first_valid": True,
    "accept_non_iso_result": False,
    "save_debug_on_not_found": True,
    "enable_resize_digit_pass": True,
    "enable_two_lines_hard_fallback": False,
    "layout_twolines_max_ratio": 3.0,
    "layout_oneline_min_ratio": 3.0,
    "layout_split_door_min_height": 420,
}

_DEBUG_LOCK = threading.Lock()
_DEBUG_VISUALIZATION_ENABLED = os.getenv("DEBUG_VISUALIZATION_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class DebugConfigPayload(BaseModel):
    enabled: bool


@app.on_event("startup")
def on_startup() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_container_config()
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recognized_numbers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                filename TEXT,
                raw_text TEXT,
                normalized_text TEXT,
                status TEXT NOT NULL,
                elapsed_ms REAL,
                debug_enabled INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def _ensure_container_config() -> None:
    CONTAINER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONTAINER_CONFIG_PATH.exists():
        CONTAINER_CONFIG_PATH.write_text(
            json.dumps(_DEFAULT_CONTAINER_CONFIG, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _load_container_config() -> dict[str, Any]:
    _ensure_container_config()
    try:
        data = json.loads(CONTAINER_CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config root must be object")
    except Exception:
        data = {}
    merged = dict(_DEFAULT_CONTAINER_CONFIG)
    merged.update(data)
    return merged


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_recognition(
    *,
    endpoint: str,
    filename: str | None,
    raw_text: str | None,
    normalized_text: str | None,
    status: str,
    elapsed_ms: float,
) -> None:
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO recognized_numbers (
                created_at,
                endpoint,
                filename,
                raw_text,
                normalized_text,
                status,
                elapsed_ms,
                debug_enabled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now_iso(),
                endpoint,
                filename,
                raw_text,
                normalized_text,
                status,
                elapsed_ms,
                int(_DEBUG_VISUALIZATION_ENABLED),
            ),
        )
        conn.commit()


def _save_debug_snapshot(
    *,
    image_bgr,
    crop_bgr=None,
    segmentation_bgr=None,
    box_xyxy: tuple[int, int, int, int] | None,
    text: str,
    endpoint: str,
    filename: str | None,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    if not _DEBUG_VISUALIZATION_ENABLED:
        return

    try:
        canvas = image_bgr.copy()
        if box_xyxy is not None:
            x1, y1, x2, y2 = box_xyxy
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = text or "NOT_FOUND"
        cv2.putText(
            canvas,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        meta = {
            "updated_at": _utc_now_iso(),
            "endpoint": endpoint,
            "filename": filename,
            "text": label,
            "bbox": list(box_xyxy) if box_xyxy is not None else None,
        }
        if extra_meta:
            meta.update(extra_meta)

        with _DEBUG_LOCK:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(DEBUG_IMAGE_PATH), canvas)
            if crop_bgr is not None and getattr(crop_bgr, "size", 0):
                cv2.imwrite(str(DEBUG_CROP_IMAGE_PATH), crop_bgr)
            elif DEBUG_CROP_IMAGE_PATH.exists():
                DEBUG_CROP_IMAGE_PATH.unlink()
            if segmentation_bgr is not None and getattr(segmentation_bgr, "size", 0):
                cv2.imwrite(str(DEBUG_SEGMENTATION_IMAGE_PATH), segmentation_bgr)
            elif DEBUG_SEGMENTATION_IMAGE_PATH.exists():
                DEBUG_SEGMENTATION_IMAGE_PATH.unlink()
            meta["crop_available"] = DEBUG_CROP_IMAGE_PATH.exists()
            meta["segmentation_available"] = DEBUG_SEGMENTATION_IMAGE_PATH.exists()
            DEBUG_META_PATH.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except Exception as exc:
        logger.warning("debug snapshot save failed: %s", exc)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def decode_image(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _safe_detect_and_crop_seal_with_box(img):
    fn = getattr(crop_service, "detect_and_crop_seal_with_box", None)
    if callable(fn):
        return fn(img)

    crops = crop_service.detect_and_crop(img)
    if not crops:
        return None, None
    return crops[0], None


def _safe_detect_and_crop_container_with_box(img, settings: dict[str, Any]):
    if not settings.get("use_crop_service", True):
        return img, None

    fn = getattr(container_crop_service, "detect_and_crop_container_with_box", None)
    if callable(fn):
        return fn(img)

    fallback = getattr(container_crop_service, "detect_and_crop", None)
    if callable(fallback):
        crops = fallback(img)
        if crops:
            return crops[0], None

    return img, None


def _recognize_seal(img) -> str:
    text, _angle, _score = OCR.ocr_best(img)
    return text or ""


def _recognize_container(
    img, settings: dict[str, Any] | None = None
) -> tuple[str, str, dict[str, float], dict[str, Any]]:
    ocr_ms = 0.0
    validation_ms = 0.0
    postprocess_ms = 0.0

    t_ocr = time.perf_counter()
    details = container_ocr.ocr_container_from_crop_details(img, config=settings or {})
    ocr_ms += (time.perf_counter() - t_ocr) * 1000.0

    t_valid = time.perf_counter()
    raw_text = str(details.get("raw_text") or "")
    full_code = str(details.get("full_code") or "")
    base10 = str(details.get("base10") or "")
    check_digit = str(details.get("check_digit") or "")
    status = str(details.get("status") or "NOT_FOUND")
    is_valid_iso = bool(details.get("is_valid_iso"))
    score = float(details.get("score") or -1.0)
    normalized = full_code if full_code else "NOT_FOUND"
    ocr_debug: dict[str, Any] = {
        "reason": "ok" if normalized != "NOT_FOUND" else status.lower(),
        "status": status,
        "base10": base10,
        "check_digit": check_digit,
        "score": round(score, 4),
        "is_valid_iso": is_valid_iso,
    }
    validation_ms += (time.perf_counter() - t_valid) * 1000.0

    return raw_text, normalized, {
        "ocr_ms": round(ocr_ms, 3),
        "validation_ms": round(validation_ms, 3),
        "postprocess_ms": round(postprocess_ms, 3),
    }, ocr_debug


def _shape_list(img) -> list[int] | None:
    if img is None or not hasattr(img, "shape"):
        return None
    try:
        return [int(v) for v in img.shape]
    except Exception:
        return None


def _ensure_bgr_local(img):
    if img is None or not getattr(img, "size", 0):
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _resize_debug_preview(img, *, max_w: int = 520, max_h: int = 220):
    if img is None or not getattr(img, "size", 0):
        return None
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale >= 0.999:
        return img.copy()
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _debug_tile(label: str, img):
    preview = _resize_debug_preview(_ensure_bgr_local(img))
    if preview is None:
        return None
    header_h = 28
    canvas = np.full((header_h + preview.shape[0], preview.shape[1], 3), 255, dtype=np.uint8)
    canvas[header_h:, :] = preview
    cv2.putText(
        canvas,
        label[:48],
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (40, 60, 90),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _pad_debug_tile(img, width: int, height: int):
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y = max(0, (height - img.shape[0]) // 2)
    x = max(0, (width - img.shape[1]) // 2)
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def _stack_debug_tiles(tiles: list[np.ndarray], *, columns: int = 2):
    valid = [tile for tile in tiles if tile is not None and getattr(tile, "size", 0)]
    if not valid:
        return None
    rows: list[np.ndarray] = []
    for idx in range(0, len(valid), columns):
        row_tiles = valid[idx:idx + columns]
        max_h = max(tile.shape[0] for tile in row_tiles)
        max_w = max(tile.shape[1] for tile in row_tiles)
        padded = [_pad_debug_tile(tile, max_w, max_h) for tile in row_tiles]
        if len(padded) < columns:
            padded.extend([np.full((max_h, max_w, 3), 255, dtype=np.uint8) for _ in range(columns - len(padded))])
        rows.append(cv2.hconcat(padded))
    total_w = max(row.shape[1] for row in rows)
    padded_rows = [_pad_debug_tile(row, total_w, row.shape[0]) for row in rows]
    return cv2.vconcat(padded_rows)


def _compose_symbol_strip_safe(rois, **kwargs):
    if not rois:
        return None
    try:
        return container_ocr._compose_symbol_strip(rois, **kwargs)
    except Exception:
        return None


def _build_container_segmentation_debug(crop_bgr, settings: dict[str, Any] | None = None):
    if crop_bgr is None or not getattr(crop_bgr, "size", 0):
        return None, {}

    runtime_config = settings or {}
    layout_type = str(container_ocr.classify_container_layout(crop_bgr, runtime_config) or "unknown")
    seg_meta: dict[str, Any] = {"layout_type": layout_type}
    tiles: list[np.ndarray] = []

    try:
        if layout_type == "twolines":
            top_roi, bottom_roi = container_ocr.extract_two_line_rois(crop_bgr)
            if top_roi is not None and bottom_roi is not None:
                top_symbols = container_ocr._extract_horizontal_symbol_rois(top_roi)
                bottom_symbols = container_ocr._extract_horizontal_symbol_rois(bottom_roi)
                seg_meta["top_symbol_count"] = len(top_symbols)
                seg_meta["bottom_symbol_count"] = len(bottom_symbols)
                tiles.extend(
                    [
                        _debug_tile("Top ROI", top_roi),
                        _debug_tile("Bottom ROI", bottom_roi),
                        _debug_tile(
                            f"Top strip ({len(top_symbols)})",
                            _compose_symbol_strip_safe(top_symbols[:4], target_h=64, gap=10, outer_pad=8),
                        ),
                        _debug_tile(
                            f"Bottom strip ({len(bottom_symbols)})",
                            _compose_symbol_strip_safe(bottom_symbols[:7], target_h=64, gap=8, outer_pad=8, square_last=len(bottom_symbols) >= 7),
                        ),
                    ]
                )
            else:
                split = container_ocr.extract_split_door_rois(crop_bgr, config=runtime_config)
                if split:
                    owner_symbols = container_ocr._extract_light_symbol_rois(
                        split["top_left"],
                        expected=4,
                        min_h_ratio=0.45,
                        min_w_ratio=0.025,
                        max_w_ratio=0.38,
                        v_min=118,
                        s_max=105,
                    )
                    left_symbols = container_ocr._extract_light_symbol_rois(
                        split["bottom_left"],
                        expected=4,
                        min_h_ratio=0.45,
                        min_w_ratio=0.025,
                        max_w_ratio=0.42,
                        v_min=140,
                        s_max=60,
                    )
                    right_symbols = container_ocr._extract_light_symbol_rois(
                        split["bottom_right"],
                        expected=3,
                        min_h_ratio=0.40,
                        min_w_ratio=0.02,
                        max_w_ratio=0.44,
                        v_min=140,
                        s_max=75,
                    )
                    seg_meta["segmentation_kind"] = "split_door"
                    seg_meta["owner_symbol_count"] = len(owner_symbols)
                    seg_meta["left_symbol_count"] = len(left_symbols)
                    seg_meta["right_symbol_count"] = len(right_symbols)
                    tiles.extend(
                        [
                            _debug_tile("Top left", split["top_left"]),
                            _debug_tile("Bottom left", split["bottom_left"]),
                            _debug_tile("Bottom right", split["bottom_right"]),
                            _debug_tile(
                                f"Owner strip ({len(owner_symbols)})",
                                _compose_symbol_strip_safe(owner_symbols[:4], target_h=64, gap=10, outer_pad=8),
                            ),
                            _debug_tile(
                                f"Left digits ({len(left_symbols)})",
                                _compose_symbol_strip_safe(left_symbols[:4], target_h=64, gap=8, outer_pad=8),
                            ),
                            _debug_tile(
                                f"Right digits ({len(right_symbols)})",
                                _compose_symbol_strip_safe(
                                    right_symbols[:3],
                                    target_h=64,
                                    gap=8,
                                    outer_pad=8,
                                    square_last=len(right_symbols) >= 3,
                                ),
                            ),
                        ]
                    )
        elif layout_type == "vertical":
            segmented = container_ocr._extract_vertical_symbol_rois(crop_bgr)
            grid11 = container_ocr._vertical_grid_rois(crop_bgr, 11)
            seg_meta["segmented_symbol_count"] = len(segmented)
            seg_meta["grid11_symbol_count"] = len(grid11)
            tiles.extend(
                [
                    _debug_tile(
                        f"Vertical segmented ({len(segmented)})",
                        _compose_symbol_strip_safe(segmented[:11], target_h=92, gap=8, outer_pad=10, square_last=len(segmented) >= 11),
                    ),
                    _debug_tile(
                        f"Vertical grid11 ({len(grid11)})",
                        _compose_symbol_strip_safe(grid11[:11], target_h=92, gap=8, outer_pad=10, square_last=len(grid11) >= 11),
                    ),
                ]
            )
        else:
            expanded = container_ocr._expand_crop(crop_bgr)
            letters_zone = container_ocr._extract_zone(expanded, 0.02, 0.40, 0.02, 0.98)
            digits_zone = container_ocr._extract_zone(expanded, 0.30, 1.00, 0.02, 0.98)
            right_zone = container_ocr._extract_zone(expanded, 0.72, 1.00, 0.03, 0.97)
            tight_zone = container_ocr._extract_zone(expanded, 0.84, 1.00, 0.02, 0.98)
            square_rois = []
            if right_zone is not None and getattr(right_zone, "size", 0):
                square_rois = container_ocr._square_like_rois(right_zone)
            seg_meta["square_like_count"] = len(square_rois)
            tiles.extend(
                [
                    _debug_tile("Expanded crop", expanded),
                    _debug_tile("Letters zone", letters_zone),
                    _debug_tile("Digits zone", digits_zone),
                    _debug_tile("Right zone", right_zone),
                    _debug_tile("Tight right zone", tight_zone),
                    _debug_tile("Square check ROI", square_rois[0] if square_rois else None),
                ]
            )
    except Exception as exc:
        seg_meta["segmentation_error"] = str(exc)

    segmentation_bgr = _stack_debug_tiles(tiles, columns=2)
    return segmentation_bgr, seg_meta


def _save_container_debug_snapshot(
    *,
    image_bgr,
    crop_bgr,
    box_xyxy: tuple[int, int, int, int] | None,
    text: str,
    endpoint: str,
    filename: str | None,
    settings: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    segmentation_bgr = None
    seg_meta: dict[str, Any] = {}
    if _DEBUG_VISUALIZATION_ENABLED and crop_bgr is not None and getattr(crop_bgr, "size", 0):
        segmentation_bgr, seg_meta = _build_container_segmentation_debug(crop_bgr, settings)
    merged_meta = dict(extra_meta or {})
    merged_meta.update(seg_meta)
    _save_debug_snapshot(
        image_bgr=image_bgr,
        crop_bgr=crop_bgr,
        segmentation_bgr=segmentation_bgr,
        box_xyxy=box_xyxy,
        text=text,
        endpoint=endpoint,
        filename=filename,
        extra_meta=merged_meta,
    )


def _print_crop_ratio_debug(label: str, filename: str | None, source_img, crop_img, bbox: tuple[int, int, int, int] | None = None) -> None:
    def _ratio_parts(img) -> tuple[int | None, int | None, float | None]:
        if img is None or not hasattr(img, "shape"):
            return None, None, None
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return int(h), int(w), None
        return int(h), int(w), float(w) / float(h)

    src_h, src_w, src_ratio = _ratio_parts(source_img)
    crop_h, crop_w, crop_ratio = _ratio_parts(crop_img)
    bbox_w = None
    bbox_h = None
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        bbox_w = max(0, int(x2) - int(x1))
        bbox_h = max(0, int(y2) - int(y1))
    print(
        "[crop-ratio] "
        f"label={label} "
        f"file={filename or ''} "
        f"src_h={src_h} src_w={src_w} src_ratio={src_ratio if src_ratio is None else round(src_ratio, 3)} "
        f"crop_h={crop_h} crop_w={crop_w} crop_ratio={crop_ratio if crop_ratio is None else round(crop_ratio, 3)} "
        f"container_h={crop_h} container_w={crop_w} bbox_h={bbox_h} bbox_w={bbox_w}"
    )


def _recognize_container_crop(img, settings: dict[str, Any] | None = None) -> tuple[str, str, str, str, str, str, dict[str, float]]:
    ocr_ms = 0.0
    validation_ms = 0.0
    postprocess_ms = 0.0

    t_ocr = time.perf_counter()
    details = container_ocr.ocr_container_from_crop_details(img, config=settings or {})
    ocr_ms += (time.perf_counter() - t_ocr) * 1000.0

    t_valid = time.perf_counter()
    raw_text = str(details.get("raw_text") or "")
    candidate = str(details.get("full_code") or "")
    base10 = str(details.get("base10") or "")
    check_digit = str(details.get("check_digit") or "")
    status = str(details.get("status") or "NOT_FOUND")
    normalized = candidate if candidate else "NOT_FOUND"
    validation_ms += (time.perf_counter() - t_valid) * 1000.0

    return raw_text, candidate, base10, check_digit, status, normalized, {
        "ocr_ms": round(ocr_ms, 3),
        "validation_ms": round(validation_ms, 3),
        "postprocess_ms": round(postprocess_ms, 3),
    }


@app.post("/RecognizeSealNumber")
async def RecognizeSealNumber(files: list[UploadFile] = File(default_factory=list)) -> dict[str, Any]:
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files. Max is {MAX_FILES}")

    processed_count = 0
    last_text = ""
    last_result = "NOT_FOUND"
    last_timings: dict[str, float] = {}
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    for f in files:
        t0 = time.perf_counter()
        stage_timings: dict[str, float] = {
            "read_ms": 0.0,
            "decode_ms": 0.0,
            "bbox_ms": 0.0,
            "ocr_ms": 0.0,
            "validation_ms": 0.0,
            "postprocess_ms": 0.0,
            "total_ms": 0.0,
        }

        if f.content_type and f.content_type.lower() not in ALLOWED_CONTENT_TYPES:
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            continue

        t_read = time.perf_counter()
        file_bytes = await f.read()
        stage_timings["read_ms"] = round((time.perf_counter() - t_read) * 1000.0, 3)
        if len(file_bytes) > max_size_bytes:
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            continue

        t_decode = time.perf_counter()
        img = decode_image(file_bytes)
        stage_timings["decode_ms"] = round((time.perf_counter() - t_decode) * 1000.0, 3)
        if img is None:
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            continue

        t_bbox = time.perf_counter()
        img_crop, bbox = _safe_detect_and_crop_seal_with_box(img)
        stage_timings["bbox_ms"] = round((time.perf_counter() - t_bbox) * 1000.0, 3)
        if img_crop is None:
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            continue

        t_ocr = time.perf_counter()
        text = _recognize_seal(img_crop)
        stage_timings["ocr_ms"] = round((time.perf_counter() - t_ocr) * 1000.0, 3)
        if not text:
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            continue

        t_validation = time.perf_counter()
        normalized = "".join(ch for ch in text.upper() if ch.isalnum())
        stage_timings["validation_ms"] = round((time.perf_counter() - t_validation) * 1000.0, 3)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stage_timings["total_ms"] = round(elapsed_ms, 3)
        last_timings = stage_timings
        processed_count += 1
        last_text = text
        last_result = normalized or "NOT_FOUND"

        _record_recognition(
            endpoint="RecognizeSealNumber",
            filename=f.filename,
            raw_text=text,
            normalized_text=normalized,
            status="DONE",
            elapsed_ms=elapsed_ms,
        )
        _save_debug_snapshot(
            image_bgr=img,
            box_xyxy=bbox,
            text=normalized or text,
            endpoint="RecognizeSealNumber",
            filename=f.filename,
        )

    if SIMULATED_DELAY_SEC > 0:
        await asyncio.sleep(SIMULATED_DELAY_SEC)

    return {
        "message": f"processed: {processed_count}, RESULT: {last_result}",
        "result": last_result,
        "raw_text": last_text,
        "timings_ms": last_timings,
    }


@app.post("/RecognizeContainerNumber")
async def RecognizeContainerNumber(files: list[UploadFile] = File(default_factory=list)) -> dict[str, Any]:
    logger.info("RecognizeContainerNumber start: files=%d", len(files))
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files. Max is {MAX_FILES}")

    settings = _load_container_config()
    processed_count = 0
    last_text = ""
    last_failure_reason = ""
    last_ocr_debug: dict[str, Any] = {}
    found_container = None
    last_timings: dict[str, float] = {}
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    for f in files:
        t0 = time.perf_counter()
        stage_timings: dict[str, float] = {
            "read_ms": 0.0,
            "decode_ms": 0.0,
            "bbox_ms": 0.0,
            "ocr_ms": 0.0,
            "validation_ms": 0.0,
            "postprocess_ms": 0.0,
            "total_ms": 0.0,
        }

        if f.content_type and f.content_type.lower() not in ALLOWED_CONTENT_TYPES:
            logger.info("skip file=%s reason=content_type content_type=%s", f.filename, f.content_type)
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            continue

        t_read = time.perf_counter()
        file_bytes = await f.read()
        stage_timings["read_ms"] = round((time.perf_counter() - t_read) * 1000.0, 3)
        if len(file_bytes) > max_size_bytes:
            logger.info("skip file=%s reason=size size=%d", f.filename, len(file_bytes))
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            continue

        t_decode = time.perf_counter()
        img = decode_image(file_bytes)
        stage_timings["decode_ms"] = round((time.perf_counter() - t_decode) * 1000.0, 3)
        if img is None:
            logger.info("skip file=%s reason=decode_failed", f.filename)
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            continue

        t_bbox = time.perf_counter()
        img_crop, bbox = _safe_detect_and_crop_container_with_box(img, settings)
        stage_timings["bbox_ms"] = round((time.perf_counter() - t_bbox) * 1000.0, 3)
        logger.info(
            "container bbox file=%s img_shape=%s crop_shape=%s bbox=%s bbox_ms=%.3f",
            f.filename,
            _shape_list(img),
            _shape_list(img_crop),
            list(bbox) if bbox is not None else None,
            stage_timings["bbox_ms"],
        )
        _print_crop_ratio_debug("RecognizeContainerNumber", f.filename, img, img_crop, bbox)
        if img_crop is None:
            logger.info("skip file=%s reason=crop_not_found", f.filename)
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            last_failure_reason = "crop_not_found"
            last_ocr_debug = {}
            continue

        raw_text, normalized, rc_timings, ocr_debug = _recognize_container(img_crop, settings)
        stage_timings["ocr_ms"] = rc_timings.get("ocr_ms", 0.0)
        stage_timings["validation_ms"] = rc_timings.get("validation_ms", 0.0)
        stage_timings["postprocess_ms"] = rc_timings.get("postprocess_ms", 0.0)
        last_ocr_debug = dict(ocr_debug)
        if not raw_text and normalized == "NOT_FOUND":
            ocr_reason = str(ocr_debug.get("reason") or "ocr_empty")
            logger.info(
                "skip file=%s reason=ocr_empty ocr_reason=%s img_shape=%s crop_shape=%s bbox=%s ocr_ms=%.3f text_pairs=%s candidates=%s",
                f.filename,
                ocr_reason,
                _shape_list(img),
                _shape_list(img_crop),
                list(bbox) if bbox is not None else None,
                stage_timings["ocr_ms"],
                ocr_debug.get("text_pairs"),
                ocr_debug.get("normalized_candidates"),
            )
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            last_failure_reason = ocr_reason
            if settings.get("save_debug_on_not_found", True):
                _save_container_debug_snapshot(
                    image_bgr=img,
                    crop_bgr=img_crop,
                    box_xyxy=bbox,
                    text=ocr_reason,
                    endpoint="RecognizeContainerNumber",
                    filename=f.filename,
                    settings=settings,
                    extra_meta={
                        "reason": "ocr_empty",
                        "ocr_reason": ocr_reason,
                        "image_shape": _shape_list(img),
                        "crop_shape": _shape_list(img_crop),
                        "timings_ms": stage_timings,
                        "ocr_debug": ocr_debug,
                    },
                )
            continue

        if normalized == "NOT_FOUND" and settings.get("accept_non_iso_result", False):
            normalized = raw_text

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stage_timings["total_ms"] = round(elapsed_ms, 3)
        last_timings = stage_timings
        processed_count += 1
        last_text = raw_text
        last_failure_reason = ""

        status = "DONE" if normalized != "NOT_FOUND" else "NOT_FOUND"
        _record_recognition(
            endpoint="RecognizeContainerNumber",
            filename=f.filename,
            raw_text=raw_text,
            normalized_text=normalized,
            status=status,
            elapsed_ms=elapsed_ms,
        )

        save_on_not_found = settings.get("save_debug_on_not_found", True)
        if normalized != "NOT_FOUND" or save_on_not_found:
            _save_container_debug_snapshot(
                image_bgr=img,
                crop_bgr=img_crop,
                box_xyxy=bbox,
                text=normalized if normalized != "NOT_FOUND" else raw_text,
                endpoint="RecognizeContainerNumber",
                filename=f.filename,
                settings=settings,
                extra_meta={
                    "image_shape": _shape_list(img),
                    "crop_shape": _shape_list(img_crop),
                    "timings_ms": stage_timings,
                    "ocr_debug": ocr_debug,
                },
            )

        if normalized != "NOT_FOUND":
            found_container = normalized
            logger.info("valid container found=%s file=%s", normalized, f.filename)
            if settings.get("stop_on_first_valid", True):
                break

    if SIMULATED_DELAY_SEC > 0:
        await asyncio.sleep(SIMULATED_DELAY_SEC)

    result_text = found_container or "NOT_FOUND"
    logger.info(
        "RecognizeContainerNumber end: processed=%d result=%s",
        processed_count,
        result_text,
    )
    return {
        "message": f"processed: {processed_count}, RESULT: {result_text}",
        "result": result_text,
        "raw_text": last_text,
        "failure_reason": last_failure_reason,
        "ocr_debug": last_ocr_debug,
        "timings_ms": last_timings,
    }


@app.post("/RecognizeContainerCropNumber")
async def RecognizeContainerCropNumber(files: list[UploadFile] = File(default_factory=list)) -> dict[str, Any]:
    logger.info("RecognizeContainerCropNumber start: files=%d", len(files))
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files. Max is {MAX_FILES}")

    settings = _load_container_config()
    processed_count = 0
    last_raw_text = ""
    last_candidate = ""
    last_base10 = ""
    last_check_digit = ""
    last_status = "NOT_FOUND"
    last_result = "NOT_FOUND"
    found_container = None
    last_timings: dict[str, float] = {}
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    for f in files:
        t0 = time.perf_counter()
        stage_timings: dict[str, float] = {
            "read_ms": 0.0,
            "decode_ms": 0.0,
            "bbox_ms": 0.0,
            "ocr_ms": 0.0,
            "validation_ms": 0.0,
            "postprocess_ms": 0.0,
            "total_ms": 0.0,
        }

        if f.content_type and f.content_type.lower() not in ALLOWED_CONTENT_TYPES:
            logger.info("skip crop file=%s reason=content_type content_type=%s", f.filename, f.content_type)
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            continue

        t_read = time.perf_counter()
        file_bytes = await f.read()
        stage_timings["read_ms"] = round((time.perf_counter() - t_read) * 1000.0, 3)
        if len(file_bytes) > max_size_bytes:
            logger.info("skip crop file=%s reason=size size=%d", f.filename, len(file_bytes))
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            continue

        t_decode = time.perf_counter()
        img = decode_image(file_bytes)
        stage_timings["decode_ms"] = round((time.perf_counter() - t_decode) * 1000.0, 3)
        if img is None:
            logger.info("skip crop file=%s reason=decode_failed", f.filename)
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            continue

        _print_crop_ratio_debug("RecognizeContainerCropNumber", f.filename, img, img, None)
        raw_text, candidate, base10, check_digit, status, normalized, rc_timings = _recognize_container_crop(img, settings)
        stage_timings["ocr_ms"] = rc_timings.get("ocr_ms", 0.0)
        stage_timings["validation_ms"] = rc_timings.get("validation_ms", 0.0)
        stage_timings["postprocess_ms"] = rc_timings.get("postprocess_ms", 0.0)

        if normalized == "NOT_FOUND" and settings.get("accept_non_iso_result", False):
            fallback_value = candidate or raw_text
            if fallback_value:
                normalized = fallback_value

        if not raw_text and normalized == "NOT_FOUND":
            logger.info("skip crop file=%s reason=ocr_empty", f.filename)
            stage_timings["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            last_timings = stage_timings
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stage_timings["total_ms"] = round(elapsed_ms, 3)
        last_timings = stage_timings
        processed_count += 1
        last_raw_text = raw_text
        last_candidate = candidate
        last_base10 = base10
        last_check_digit = check_digit
        last_status = status
        last_result = normalized
        _record_recognition(
            endpoint="RecognizeContainerCropNumber",
            filename=f.filename,
            raw_text=raw_text,
            normalized_text=normalized if normalized != "NOT_FOUND" else candidate,
            status=status,
            elapsed_ms=elapsed_ms,
        )

        save_on_not_found = settings.get("save_debug_on_not_found", True)
        if normalized != "NOT_FOUND" or save_on_not_found:
            _save_container_debug_snapshot(
                image_bgr=img,
                crop_bgr=img,
                box_xyxy=None,
                text=normalized if normalized != "NOT_FOUND" else (candidate or raw_text),
                endpoint="RecognizeContainerCropNumber",
                filename=f.filename,
                settings=settings,
                extra_meta={
                    "image_shape": _shape_list(img),
                    "crop_shape": _shape_list(img),
                    "timings_ms": stage_timings,
                    "status": status,
                },
            )

        if status == "FULL_11":
            found_container = normalized
            logger.info("valid crop container found=%s file=%s", normalized, f.filename)
            if settings.get("stop_on_first_valid", True):
                break

    if SIMULATED_DELAY_SEC > 0:
        await asyncio.sleep(SIMULATED_DELAY_SEC)

    result_text = found_container or last_result or "NOT_FOUND"
    logger.info(
        "RecognizeContainerCropNumber end: processed=%d result=%s",
        processed_count,
        result_text,
    )
    return {
        "message": f"processed: {processed_count}, RESULT: {result_text}",
        "result": result_text,
        "candidate": last_candidate,
        "base10": last_base10,
        "check_digit": last_check_digit,
        "status": last_status,
        "raw_text": last_raw_text,
        "timings_ms": last_timings,
    }


@app.post("/admin/models/reload")
def admin_reload_models() -> dict[str, Any]:
    details: dict[str, Any] = {}
    try:
        reload_fn = getattr(crop_service, "reload_models", None)
        if callable(reload_fn):
            details["seal"] = reload_fn()
        else:
            if hasattr(crop_service, "_MODEL"):
                crop_service._MODEL = None
            details["seal"] = {"reloaded": True, "mode": "fallback"}

        container_reload = getattr(container_crop_service, "reload_models", None)
        if callable(container_reload):
            details["container"] = container_reload()
        else:
            if hasattr(container_crop_service, "_MODEL"):
                container_crop_service._MODEL = None
            details["container"] = {"reloaded": True, "mode": "fallback"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {exc}") from exc
    return {"message": "models reloaded", "details": details}


@app.get("/admin/debug/config")
def admin_get_debug_config() -> dict[str, Any]:
    return {"enabled": _DEBUG_VISUALIZATION_ENABLED}


@app.post("/admin/debug/config")
def admin_set_debug_config(payload: DebugConfigPayload) -> dict[str, Any]:
    global _DEBUG_VISUALIZATION_ENABLED
    _DEBUG_VISUALIZATION_ENABLED = bool(payload.enabled)
    return {"enabled": _DEBUG_VISUALIZATION_ENABLED}


@app.get("/admin/debug/last-meta")
def admin_debug_last_meta() -> dict[str, Any]:
    if not DEBUG_META_PATH.exists():
        return {"available": False}
    try:
        data = json.loads(DEBUG_META_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"available": False}
    return {"available": True, "meta": data}


@app.get("/admin/debug/last-image")
def admin_debug_last_image():
    if not DEBUG_IMAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="No debug image yet")
    return FileResponse(str(DEBUG_IMAGE_PATH), media_type="image/jpeg")


@app.get("/admin/debug/last-crop-image")
def admin_debug_last_crop_image():
    if not DEBUG_CROP_IMAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="No debug crop image yet")
    return FileResponse(str(DEBUG_CROP_IMAGE_PATH), media_type="image/jpeg")


@app.get("/admin/debug/last-segmentation-image")
def admin_debug_last_segmentation_image():
    if not DEBUG_SEGMENTATION_IMAGE_PATH.exists():
        raise HTTPException(status_code=404, detail="No debug segmentation image yet")
    return FileResponse(str(DEBUG_SEGMENTATION_IMAGE_PATH), media_type="image/jpeg")


@app.get("/admin/recognized")
def admin_recognized(
    limit: int = Query(default=500, ge=1, le=5000),
    endpoint: str | None = Query(default=None),
) -> dict[str, Any]:
    sql = """
        SELECT
            id,
            created_at,
            endpoint,
            filename,
            raw_text,
            normalized_text,
            status,
            elapsed_ms,
            debug_enabled
        FROM recognized_numbers
    """
    params: list[Any] = []
    if endpoint:
        sql += " WHERE endpoint = ?"
        params.append(endpoint)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()

    return {"items": [dict(row) for row in rows]}


if __name__ == "__main__":
    import uvicorn

    workers = int(os.getenv("WORKERS", "1"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=workers)
