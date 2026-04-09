from __future__ import annotations

import os
from typing import Any, Optional

import cv2
import numpy as np

HORIZONTAL_RATIO_THRESHOLD = float(os.getenv("OCR_HORIZONTAL_RATIO_THRESHOLD", "3.2"))
VERTICAL_RATIO_THRESHOLD = float(os.getenv("OCR_VERTICAL_RATIO_THRESHOLD", "0.22"))
DEFAULT_TWOLINES_MAX_RATIO = float(os.getenv("OCR_TWOLINES_MAX_RATIO", "3.0"))
DEFAULT_ONELINE_MIN_RATIO = float(os.getenv("OCR_ONELINE_MIN_RATIO", "3.0"))
DEFAULT_SPLIT_DOOR_MIN_HEIGHT = int(os.getenv("OCR_SPLIT_DOOR_MIN_HEIGHT", "420"))


def _layout_setting(config: dict[str, Any] | None, key: str, default: float | int) -> float | int:
    if not config:
        return default
    value = config.get(key, default)
    if isinstance(default, int):
        try:
            return int(value)
        except Exception:
            return default
    try:
        return float(value)
    except Exception:
        return default


def foreground_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(mask) > (mask.size * 0.5):
        mask = 255 - mask
    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def merge_bands(bands: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
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


def extract_text_row_bands(img) -> list[tuple[int, int, int]]:
    if img is None or img.size == 0:
        return []

    mask = foreground_mask(img)
    h, w = mask.shape[:2]
    row_strength = np.count_nonzero(mask, axis=1)
    row_threshold = max(2, int(w * 0.08))

    raw_bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for idx, value in enumerate(row_strength):
        if value >= row_threshold and not in_band:
            start = idx
            in_band = True
        elif value < row_threshold and in_band:
            raw_bands.append((start, idx))
            in_band = False
    if in_band:
        raw_bands.append((start, h))

    merged = merge_bands(raw_bands, max_gap=max(3, int(h * 0.018)))
    min_band_h = max(8, int(h * 0.06))
    out: list[tuple[int, int, int]] = []
    for y1, y2 in merged:
        if (y2 - y1) < min_band_h:
            continue
        strength = int(row_strength[y1:y2].sum())
        out.append((y1, y2, strength))
    return out


def _band_roi(img, y1: int, y2: int, *, pad_x: int = 6, pad_y: int = 4):
    h, w = img.shape[:2]
    band = img[max(0, y1 - pad_y):min(h, y2 + pad_y), :]
    if band.size == 0:
        return None
    mask = foreground_mask(band)
    cols = np.where(np.count_nonzero(mask, axis=0) > 0)[0]
    if cols.size == 0:
        return band
    x1 = max(0, int(cols[0]) - pad_x)
    x2 = min(w, int(cols[-1]) + pad_x + 1)
    roi = band[:, x1:x2]
    return roi if roi.size else band


def extract_two_line_rois(img) -> tuple[Any | None, Any | None]:
    if img is None or img.size == 0:
        return None, None

    h, w = img.shape[:2]
    bands = extract_text_row_bands(img)
    if len(bands) < 2:
        return None, None

    ranked = sorted(bands, key=lambda item: (item[2], item[1] - item[0]), reverse=True)[:2]
    ranked.sort(key=lambda item: item[0])
    top, bottom = ranked
    gap = bottom[0] - top[1]
    if gap < max(6, int(h * 0.04)):
        return None, None

    top_roi = _band_roi(img, top[0], top[1], pad_x=max(6, int(w * 0.01)), pad_y=max(3, int(h * 0.015)))
    bottom_roi = _band_roi(img, bottom[0], bottom[1], pad_x=max(6, int(w * 0.01)), pad_y=max(3, int(h * 0.015)))
    return top_roi, bottom_roi


def _estimate_vertical_seam_x(img) -> int | None:
    if img is None or img.size == 0:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w < 200 or h < 120:
        return None

    x1 = int(w * 0.28)
    x2 = int(w * 0.72)
    if x2 - x1 < 24:
        return None

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_strength = cv2.GaussianBlur(np.abs(gx), (9, 9), 0).mean(axis=0)
    text_mask = foreground_mask(img)
    text_density = np.count_nonzero(text_mask, axis=0) / float(max(1, h))
    score = edge_strength - (text_density * float(edge_strength.max(initial=1.0)) * 0.35)
    local = score[x1:x2]
    if local.size == 0:
        return None

    if np.max(local) <= 0:
        return None

    candidate_count = min(10, max(3, local.size // 40))
    candidate_indices = np.argpartition(local, -candidate_count)[-candidate_count:]
    candidate_indices = sorted({int(idx) for idx in candidate_indices}, key=lambda idx: float(local[idx]), reverse=True)

    seam_pad = max(14, int(w * 0.014))
    best_x: int | None = None
    best_score = -1.0

    for rel_idx in candidate_indices:
        seam_x = x1 + rel_idx
        left_x2 = max(1, seam_x - seam_pad)
        right_x1 = min(w, seam_x + seam_pad)
        if left_x2 < int(w * 0.22) or (w - right_x1) < int(w * 0.12):
            continue

        left = img[:, :left_x2]
        bands = extract_text_row_bands(left)
        if len(bands) < 2:
            continue

        ranked = sorted(bands, key=lambda item: (item[2], item[1] - item[0]), reverse=True)[:2]
        ranked.sort(key=lambda item: item[0])
        top, bottom = ranked
        gap = bottom[0] - top[1]
        if gap < max(5, int(h * 0.02)):
            continue

        left_ratio = left_x2 / float(max(1, w))
        balance_penalty = abs(left_ratio - 0.42)
        score = float(local[rel_idx]) + (0.0008 * (top[2] + bottom[2])) - (balance_penalty * 20.0) + (gap * 0.4)
        if score > best_score:
            best_score = score
            best_x = seam_x

    if best_x is not None:
        return best_x

    best_idx = int(np.argmax(local))
    return x1 + best_idx


def extract_split_door_rois(img, config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]
    split_door_min_height = int(_layout_setting(config, "layout_split_door_min_height", DEFAULT_SPLIT_DOOR_MIN_HEIGHT))
    twolines_max_ratio = float(_layout_setting(config, "layout_twolines_max_ratio", DEFAULT_TWOLINES_MAX_RATIO))
    if h < split_door_min_height:
        return None
    if w / float(max(1, h)) > twolines_max_ratio:
        return None
    if w / float(max(1, h)) < 1.35:
        return None

    seam_pad = max(14, int(w * 0.014))
    pad_x = max(6, int(w * 0.008))
    pad_y = max(4, int(h * 0.015))

    def build_candidate(seam_x: int) -> tuple[float, dict[str, Any]] | None:
        left_x2 = max(1, seam_x - seam_pad)
        right_x1 = min(w, seam_x + seam_pad)
        if left_x2 < int(w * 0.22) or (w - right_x1) < int(w * 0.12):
            return None

        left = img[:, :left_x2]
        right = img[:, right_x1:]
        left_bands = extract_text_row_bands(left)
        if len(left_bands) < 2:
            return None

        ranked = sorted(left_bands, key=lambda item: (item[2], item[1] - item[0]), reverse=True)[:2]
        ranked.sort(key=lambda item: item[0])
        top, bottom = ranked
        gap = bottom[0] - top[1]
        if gap < max(5, int(h * 0.015)):
            return None

        top_left = _band_roi(left, top[0], top[1], pad_x=pad_x, pad_y=pad_y)
        bottom_left = _band_roi(left, bottom[0], bottom[1], pad_x=pad_x, pad_y=pad_y)
        y1 = max(0, int(bottom[0]) - pad_y)
        y2 = min(h, int(bottom[1]) + pad_y)
        bottom_right = right[y1:y2, :]
        bottom_right = _band_roi(bottom_right, 0, bottom_right.shape[0], pad_x=pad_x, pad_y=max(2, pad_y // 2))

        if top_left is None or bottom_left is None or bottom_right is None:
            return None
        if top_left.size == 0 or bottom_left.size == 0 or bottom_right.size == 0:
            return None

        left_ratio = left_x2 / float(max(1, w))
        score = (top[2] + bottom[2]) * 0.001 + gap - abs(left_ratio - 0.40) * 25.0
        return score, {
            "seam_x": int(seam_x),
            "top_left": top_left,
            "bottom_left": bottom_left,
            "bottom_right": bottom_right,
        }

    candidates: list[int] = []
    estimated = _estimate_vertical_seam_x(img)
    if estimated is not None:
        candidates.append(int(estimated))

    step = max(18, int(w * 0.018))
    for seam_x in range(int(w * 0.30), int(w * 0.70), step):
        candidates.append(int(seam_x))

    seen: set[int] = set()
    best_score = -1.0
    best_payload: dict[str, Any] | None = None
    for seam_x in candidates:
        if seam_x in seen:
            continue
        seen.add(seam_x)
        built = build_candidate(seam_x)
        if built is None:
            continue
        score, payload = built
        if score > best_score:
            best_score = score
            best_payload = payload

    return best_payload


def classify_container_layout(img, config: dict[str, Any] | None = None) -> str:
    if img is None or img.size == 0:
        return "unknown"

    h, w = img.shape[:2]
    if h <= 0:
        return "unknown"

    ratio = w / float(h)
    if ratio <= VERTICAL_RATIO_THRESHOLD:
        return "vertical"

    twolines_max_ratio = float(_layout_setting(config, "layout_twolines_max_ratio", DEFAULT_TWOLINES_MAX_RATIO))
    oneline_min_ratio = float(_layout_setting(config, "layout_oneline_min_ratio", DEFAULT_ONELINE_MIN_RATIO))

    if ratio <= twolines_max_ratio:
        return "twolines"

    top_roi, bottom_roi = extract_two_line_rois(img)
    if top_roi is not None and bottom_roi is not None:
        return "twolines"

    if ratio >= oneline_min_ratio:
        return "oneline"

    return "unknown"
