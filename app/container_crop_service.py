#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Any

import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

_MODEL = None
_MODEL_PATH: str | None = None


def _resolve_model_path() -> Path:
    configured = os.getenv("CONTAINER_MODEL_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path("models/YOLO/container.pt")


def _get_model():
    global _MODEL, _MODEL_PATH
    if YOLO is None:
        return None

    model_path = _resolve_model_path()
    model_key = str(model_path.resolve()) if model_path.exists() else str(model_path)
    if _MODEL is not None and _MODEL_PATH == model_key:
        return _MODEL
    if not model_path.exists():
        return None

    _MODEL = YOLO(str(model_path))
    _MODEL_PATH = model_key
    return _MODEL


def reload_models() -> dict[str, Any]:
    global _MODEL, _MODEL_PATH
    _MODEL = None
    _MODEL_PATH = None
    model_path = _resolve_model_path()
    exists = model_path.exists()
    if exists:
        _get_model()
    return {
        "reloaded": True,
        "model_path": str(model_path),
        "exists": bool(exists),
        "mode": "yolo",
    }


def crop_image(image, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _detect_boxes(img):
    if img is None:
        return []

    model = _get_model()
    if model is None:
        return []

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, verbose=False)
    if not results:
        return []

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(xyxy)
        for box, conf in zip(xyxy, confs):
            detections.append((float(conf), box.tolist()))

    detections.sort(key=lambda item: item[0], reverse=True)
    return detections


def detect_and_crop(img):
    crops = []
    for _conf, box in _detect_boxes(img):
        crop = crop_image(img, box)
        if crop is not None:
            crops.append(crop)
    return crops


def detect_and_crop_container_with_box(img):
    detections = _detect_boxes(img)
    if not detections:
        return None, None

    _conf, box = detections[0]
    crop = crop_image(img, box)
    if crop is None:
        return None, None
    return crop, [int(v) for v in box]
