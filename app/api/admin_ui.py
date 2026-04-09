from __future__ import annotations

import html
import importlib.util
import json
import logging
import mimetypes
import os
import re
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from ..config.admin_ui_settings import get_bool, get_int, get_visible_tabs
from ..container_layout import classify_container_layout

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "dashboard_data"
ADMIN_CFG_PATH = DATA_DIR / "admin_service_config.json"
ADMIN_DB_PATH = DATA_DIR / "admin_ui.db"
SERVICE_STDOUT_LOG_PATH = DATA_DIR / "service_stdout.log"
APP_LOG_PATH = PROJECT_ROOT / "app.log"
UI_HTML_PATH = PROJECT_ROOT / "app" / "ui" / "admin_ui_page.html"
TEST_REPORTS_DIR = DATA_DIR / "test_reports"
UI_TABS = ("service", "logs", "debug", "testing", "about")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONTAINER_RE = re.compile(r"\b[A-Z]{4}\d{7}\b")
RESULT_RE = re.compile(r"RESULT\s*:\s*(.+)$", re.IGNORECASE)
REPORT_FILE_RE = re.compile(r"^run_(\d+)\.html$")
DEFAULT_TEST_TIMEOUT_SEC = 120.0
MAX_TEST_TIMEOUT_SEC = 1800.0


class ServiceConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = Field(default=1, ge=1, le=16)
    debug_visualization: bool = False


class DebugConfigPayload(BaseModel):
    enabled: bool


class TestRunPayload(BaseModel):
    dataset_dir: str
    mode: str = Field(default="container", pattern="^(container|seal)$")
    timeout_sec: float = Field(default=DEFAULT_TEST_TIMEOUT_SEC, ge=1.0, le=MAX_TEST_TIMEOUT_SEC)


class WorkersPayload(BaseModel):
    workers: int = Field(ge=1, le=16)


class CleanupReportsPayload(BaseModel):
    mode: str = Field(default="policy", pattern="^(policy|all)$")
    dry_run: bool = False


app = FastAPI(title="FastService Admin UI")
logger = logging.getLogger("admin_ui")

_managed_lock = threading.Lock()
_managed_process: subprocess.Popen | None = None


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # для pydantic 2
    return model.dict()  # для pydantic 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config() -> ServiceConfig:
    if not ADMIN_CFG_PATH.exists():
        return ServiceConfig()
    try:
        data = json.loads(ADMIN_CFG_PATH.read_text(encoding="utf-8"))
        return ServiceConfig(**data)
    except Exception:
        return ServiceConfig()


def _save_config(cfg: ServiceConfig) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ADMIN_CFG_PATH.write_text(json.dumps(_model_dump(cfg), ensure_ascii=False, indent=2), encoding="utf-8")


def _visible_ui_tabs() -> list[str]:
    return get_visible_tabs(UI_TABS)


def _reports_cleanup_settings() -> dict[str, int | bool]:
    enabled = get_bool(
        "testing.reports_cleanup.enabled",
        True,
        env_var="ADMIN_UI_REPORTS_CLEANUP_ENABLED",
    )
    max_age_days = max(
        1,
        get_int(
            "testing.reports_cleanup.max_age_days",
            30,
            env_var="ADMIN_UI_REPORTS_CLEANUP_MAX_AGE_DAYS",
        ),
    )
    keep_last_runs = max(
        0,
        get_int(
            "testing.reports_cleanup.keep_last_runs",
            100,
            env_var="ADMIN_UI_REPORTS_CLEANUP_KEEP_LAST_RUNS",
        ),
    )
    return {
        "enabled": enabled,
        "max_age_days": max_age_days,
        "keep_last_runs": keep_last_runs,
    }


def _load_recent_run_ids(limit: int) -> set[int]:
    if limit <= 0 or not ADMIN_DB_PATH.exists():
        return set()
    try:
        with sqlite3.connect(ADMIN_DB_PATH) as conn:
            rows = conn.execute(
                """
                SELECT id
                FROM test_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except sqlite3.Error:
        return set()
    return {int(row[0]) for row in rows}


def _cleanup_test_reports(*, mode: str = "policy", dry_run: bool = False, reason: str = "manual") -> dict[str, Any]:
    settings = _reports_cleanup_settings()
    TEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in TEST_REPORTS_DIR.glob("run_*.html") if p.is_file()], key=lambda p: p.name)

    if mode == "policy" and not bool(settings["enabled"]) and reason != "manual":
        return {
            "mode": mode,
            "reason": reason,
            "dry_run": dry_run,
            "cleanup_enabled": bool(settings["enabled"]),
            "max_age_days": int(settings["max_age_days"]),
            "keep_last_runs": int(settings["keep_last_runs"]),
            "total_files": len(files),
            "removed_count": 0,
            "removed_files": [],
            "skipped": "auto cleanup disabled",
        }

    keep_ids = _load_recent_run_ids(int(settings["keep_last_runs"])) if mode == "policy" else set()
    cutoff_ts = time.time() - int(settings["max_age_days"]) * 86400
    removable: list[Path] = []

    if mode == "all":
        removable = files
    else:
        for path in files:
            match = REPORT_FILE_RE.match(path.name)
            run_id = int(match.group(1)) if match else None
            if run_id is not None and run_id in keep_ids:
                continue
            try:
                is_old = path.stat().st_mtime < cutoff_ts
            except OSError:
                is_old = False
            if is_old:
                removable.append(path)

    removed_names: list[str] = []
    removed_count = 0
    for path in removable:
        removed_names.append(path.name)
        if dry_run:
            continue
        try:
            path.unlink()
            removed_count += 1
        except OSError:
            continue

    return {
        "mode": mode,
        "reason": reason,
        "dry_run": dry_run,
        "cleanup_enabled": bool(settings["enabled"]),
        "max_age_days": int(settings["max_age_days"]),
        "keep_last_runs": int(settings["keep_last_runs"]),
        "total_files": len(files),
        "candidate_count": len(removable),
        "removed_count": len(removable) if dry_run else removed_count,
        "removed_files": removed_names,
        "kept_recent_run_ids": sorted(keep_ids),
    }


def _testing_report_path(run_id: int) -> Path:
    return TEST_REPORTS_DIR / f"run_{run_id}.html"


def _service_base_url(cfg: ServiceConfig) -> str:
    return f"http://{cfg.host}:{cfg.port}"


def _is_running(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def _can_bind_address(host: str, port: int) -> tuple[bool, str | None]:
    try:
        addr_infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        return False, str(exc)

    for family, socktype, proto, _canonname, sockaddr in addr_infos:
        s = socket.socket(family, socktype, proto)
        try:
            if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            s.bind(sockaddr)
            return True, None
        except OSError as exc:
            return False, f"[Errno {exc.errno}] {exc.strerror}"
        finally:
            s.close()
    return True, None


def _effective_workers(requested_workers: int) -> int:
    if os.name == "nt" and requested_workers > 1:
        return 1
    return max(1, requested_workers)


def _start_managed_service() -> dict[str, Any]:
    global _managed_process
    cfg = _load_config()
    base_url = _service_base_url(cfg)
    workers_used = _effective_workers(cfg.workers)
    with _managed_lock:
        if _is_running(_managed_process):
            return {"running": True, "pid": _managed_process.pid, "message": "already running"}

        health = _probe_health(base_url)
        if health.get("reachable"):
            return {
                "running": False,
                "message": "service already reachable on configured host/port (likely started outside admin)",
                "base_url": base_url,
                "health": health,
            }

        can_bind, bind_error = _can_bind_address(cfg.host, cfg.port)
        if not can_bind:
            return {
                "running": False,
                "message": "cannot bind configured host/port",
                "base_url": base_url,
                "error": bind_error,
            }

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log_file = open(SERVICE_STDOUT_LOG_PATH, "a", encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api.main:app",
            "--host",
            cfg.host,
            "--port",
            str(cfg.port),
            "--workers",
            str(workers_used),
        ]
        env = os.environ.copy()
        env["DEBUG_VISUALIZATION_ENABLED"] = "1" if cfg.debug_visualization else "0"
        env.setdefault("LOG_PATH", str(APP_LOG_PATH))
        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW

        _managed_process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
        )
        time.sleep(0.7)
        if not _is_running(_managed_process):
            _managed_process = None
            return {
                "running": False,
                "message": "managed service exited immediately after start",
                "base_url": base_url,
                "log_tail": _tail_lines(SERVICE_STDOUT_LOG_PATH, 40),
                "workers_requested": cfg.workers,
                "workers_used": workers_used,
            }

    response: dict[str, Any] = {
        "running": True,
        "pid": _managed_process.pid,
        "message": "started",
        "workers_requested": cfg.workers,
        "workers_used": workers_used,
    }
    if workers_used != cfg.workers:
        response["note"] = "workers>1 is unstable on Windows for managed mode; started with workers=1"
    return response


def _stop_managed_service() -> dict[str, Any]:
    global _managed_process
    with _managed_lock:
        if not _is_running(_managed_process):
            _managed_process = None
            return {"running": False, "message": "not running"}
        assert _managed_process is not None
        proc = _managed_process
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        _managed_process = None
    return {"running": False, "message": "stopped"}


def _restart_managed_service() -> dict[str, Any]:
    _stop_managed_service()
    return _start_managed_service()


def _probe_health(base_url: str, timeout: float = 2.0) -> dict[str, Any]:
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        return {"reachable": resp.status_code == 200, "status_code": resp.status_code}
    except requests.RequestException as exc:
        return {"reachable": False, "error": str(exc)}


def _tail_lines(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "\n".join(text.splitlines()[-lines:])


def _proxy_json(method: str, path: str, payload: dict | None = None) -> dict[str, Any]:
    cfg = _load_config()
    url = f"{_service_base_url(cfg)}{path}"
    try:
        resp = requests.request(method=method, url=url, json=payload, timeout=30)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Service request failed: {exc}") from exc
    try:
        data = resp.json()
    except ValueError:
        data = {"raw": resp.text}
    if not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=data)
    return data


def _init_admin_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(ADMIN_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                dataset_dir TEXT NOT NULL,
                mode TEXT NOT NULL,
                timeout_sec REAL NOT NULL,
                labels_available INTEGER NOT NULL,
                text_available INTEGER NOT NULL,
                summary_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS test_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                number_type TEXT,
                predicted_text TEXT,
                raw_text TEXT,
                normalized_predicted TEXT,
                expected_label TEXT,
                expected_text TEXT,
                status_code INTEGER,
                elapsed_ms REAL,
                stage_timings_json TEXT,
                error TEXT,
                FOREIGN KEY(run_id) REFERENCES test_runs(id)
            )
            """
        )
        cols = [row[1] for row in conn.execute("PRAGMA table_info(test_items)").fetchall()]
        if "number_type" not in cols:
            conn.execute("ALTER TABLE test_items ADD COLUMN number_type TEXT")
        if "raw_text" not in cols:
            conn.execute("ALTER TABLE test_items ADD COLUMN raw_text TEXT")
        if "stage_timings_json" not in cols:
            conn.execute("ALTER TABLE test_items ADD COLUMN stage_timings_json TEXT")
        conn.commit()


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch for ch in value.upper() if ch.isalnum())


def _extract_result_text(response_obj: Any, fallback_text: str) -> str:
    if isinstance(response_obj, dict):
        result = response_obj.get("result")
        if isinstance(result, str) and result.strip():
            return result.strip()
        m = RESULT_RE.search(str(response_obj.get("message", "")))
        if m:
            return m.group(1).strip()
    m = RESULT_RE.search(fallback_text or "")
    return m.group(1).strip() if m else ""


def _extract_raw_text(response_obj: Any, fallback_text: str) -> str:
    if isinstance(response_obj, dict):
        raw_text = response_obj.get("raw_text")
        if isinstance(raw_text, str) and raw_text.strip():
            return raw_text.strip()
    return _extract_result_text(response_obj, fallback_text)


def _extract_stage_timings(response_obj: Any) -> dict[str, float]:
    if not isinstance(response_obj, dict):
        return {}
    raw = response_obj.get("timings_ms")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key in ("read_ms", "decode_ms", "bbox_ms", "ocr_ms", "validation_ms", "postprocess_ms", "total_ms"):
        value = raw.get(key)
        try:
            if value is None:
                continue
            out[key] = round(float(value), 3)
        except (TypeError, ValueError):
            continue
    return out


def _mime_for_filename(name: str) -> str:
    mime, _ = mimetypes.guess_type(name)
    return mime or "application/octet-stream"


def _format_request_error(exc: requests.RequestException, timeout_sec: float) -> str:
    if isinstance(exc, requests.ReadTimeout):
        return f"Read timeout after {timeout_sec:.1f}s. Increase timeout and retry."
    return str(exc)


def _find_match_file(base_dir: Path, image_path: Path) -> Path | None:
    if not base_dir.exists():
        return None
    txt = base_dir / f"{image_path.stem}.txt"
    if txt.exists():
        return txt
    same = base_dir / image_path.name
    return same if same.exists() else None


def _read_expected_label(labels_dir: Path | None, image_path: Path) -> str | None:
    if labels_dir is None:
        return None
    path = _find_match_file(labels_dir, image_path)
    if path is None:
        return None
    raw = path.read_text(encoding="utf-8", errors="ignore")
    found = CONTAINER_RE.findall(raw.upper())
    if found:
        return found[0]
    normalized = _normalize_text(raw)
    return normalized or None


def _read_expected_text(text_dir: Path | None, image_path: Path) -> str | None:
    if text_dir is None:
        return None
    path = _find_match_file(text_dir, image_path)
    if path is None:
        return None
    normalized = _normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    return normalized or None


def _detect_number_type_from_image_bytes(image_bytes: bytes, mode: str) -> str:
    if str(mode).lower() != "container":
        return "unknown"
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return "unknown"
    return str(classify_container_layout(img) or "unknown")


def _detect_number_type_from_image_path(image_path: Path, mode: str) -> str:
    if str(mode).lower() != "container":
        return "unknown"
    img = cv2.imread(str(image_path))
    if img is None:
        return "unknown"
    return str(classify_container_layout(img) or "unknown")


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * p
    low = int(rank)
    high = min(low + 1, len(data) - 1)
    frac = rank - low
    return data[low] + (data[high] - data[low]) * frac


def _finalize_and_store_test_run(
    *,
    dataset_id: str,
    mode: str,
    endpoint: str,
    timeout_sec: float,
    items: list[dict[str, Any]],
    labels_available: bool,
    text_available: bool,
) -> dict[str, Any]:
    latencies = [float(it["elapsed_ms"]) for it in items if it.get("elapsed_ms") is not None]
    ok_items = [it for it in items if it["status_code"] == 200 and not it["error"]]
    not_found_count = sum(1 for it in ok_items if it["normalized_predicted"] in {"", "NOTFOUND"})
    labels_items = [it for it in items if it["expected_label"]]
    text_items = [it for it in items if it["expected_text"]]
    label_matches = sum(
        1
        for it in labels_items
        if it["normalized_predicted"] == _normalize_text(it["expected_label"]) and it["normalized_predicted"]
    )
    text_matches = sum(
        1
        for it in text_items
        if it["normalized_predicted"] == _normalize_text(it["expected_text"]) and it["normalized_predicted"]
    )

    summary = {
        "created_at": _utc_now_iso(),
        "dataset_dir": dataset_id,
        "mode": mode,
        "endpoint": endpoint,
        "total_images": len(items),
        "ok_requests": len(ok_items),
        "failed_requests": len(items) - len(ok_items),
        "not_found_count": not_found_count,
        "latency_mean_ms": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "latency_p95_ms": round(_percentile(latencies, 0.95), 3) if latencies else None,
        "labels_available": labels_available,
        "text_available": text_available,
        "label_total": len(labels_items),
        "label_matches": label_matches,
        "label_accuracy": round(label_matches / len(labels_items), 4) if labels_items else None,
        "text_total": len(text_items),
        "text_matches": text_matches,
        "text_accuracy": round(text_matches / len(text_items), 4) if text_items else None,
    }

    with sqlite3.connect(ADMIN_DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO test_runs (
                created_at, dataset_dir, mode, timeout_sec, labels_available, text_available, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary["created_at"],
                summary["dataset_dir"],
                summary["mode"],
                timeout_sec,
                int(summary["labels_available"]),
                int(summary["text_available"]),
                json.dumps(summary, ensure_ascii=False),
            ),
        )
        run_id = int(cur.lastrowid)
        for it in items:
            conn.execute(
                """
                INSERT INTO test_items (
                    run_id, file_name, number_type, predicted_text, raw_text, normalized_predicted, expected_label, expected_text, status_code, elapsed_ms, stage_timings_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    it["file_name"],
                    it.get("number_type", "unknown"),
                    it["predicted_text"],
                    it.get("raw_text", ""),
                    it["normalized_predicted"],
                    it["expected_label"],
                    it["expected_text"],
                    it["status_code"],
                    it["elapsed_ms"],
                    json.dumps(it.get("stage_timings") or {}, ensure_ascii=False),
                    it["error"],
                ),
            )
        conn.commit()
    return {"run_id": run_id, "summary": summary, "items": items}


def _load_test_run_data(run_id: int) -> dict[str, Any]:
    with sqlite3.connect(ADMIN_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, created_at, dataset_dir, mode, timeout_sec, labels_available, text_available, summary_json
            FROM test_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        items = conn.execute(
            """
            SELECT file_name, number_type, predicted_text, raw_text, normalized_predicted, expected_label, expected_text, status_code, elapsed_ms, stage_timings_json, error
            FROM test_items
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
    parsed_items: list[dict[str, Any]] = []
    for x in items:
        item = dict(x)
        raw_stage = item.pop("stage_timings_json", None)
        try:
            item["stage_timings"] = json.loads(raw_stage) if raw_stage else {}
        except json.JSONDecodeError:
            item["stage_timings"] = {}
        parsed_items.append(item)
    return {
        "run_id": int(row["id"]),
        "created_at": row["created_at"],
        "dataset_dir": row["dataset_dir"],
        "mode": row["mode"],
        "timeout_sec": float(row["timeout_sec"]),
        "labels_available": bool(row["labels_available"]),
        "text_available": bool(row["text_available"]),
        "summary": json.loads(row["summary_json"]),
        "items": parsed_items,
    }


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _build_svg_bar_chart(
    *,
    labels: list[str],
    values: list[float],
    title: str,
    y_title: str,
    x_title: str = "Stage number",
    multicolor: bool = False,
) -> str:
    if not labels or not values:
        return '<div class="plot-empty">No data</div>'

    palette = [
        "#2f6db5",
        "#1f9d8b",
        "#d9822b",
        "#ad1457",
        "#6d4c41",
        "#5e35b1",
        "#00897b",
        "#ef6c00",
        "#455a64",
        "#7cb342",
    ]

    w = max(920, min(3600, 220 + len(values) * 24))
    h = 420
    pad_l, pad_r, pad_t, pad_b = 70, 20, 50, 95
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    vmax = max(max(values), 1.0)
    y_max = vmax * 1.1
    ticks = 5
    bw = plot_w / max(len(values), 1)
    bar_w = max(5.0, bw * 0.68)
    label_step = max(1, len(values) // 25)

    out: list[str] = []
    out.append(f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{html.escape(title)}">')
    out.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    out.append(
        f'<text x="{w/2:.1f}" y="28" text-anchor="middle" fill="#1f2937" '
        'font-size="16" font-family="Arial, sans-serif" font-weight="700">'
        f"{html.escape(title)}</text>"
    )

    for i in range(ticks + 1):
        y = pad_t + plot_h - (plot_h * i / ticks)
        tick_val = y_max * i / ticks
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" stroke="#e5eaf2" stroke-width="1"/>')
        out.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#607086" '
            f'font-size="11" font-family="Arial, sans-serif">{tick_val:.0f}</text>'
        )

    out.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{w - pad_r}" y2="{pad_t + plot_h}" stroke="#9aa8bc" stroke-width="1.2"/>'
    )
    out.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#9aa8bc" stroke-width="1.2"/>')

    for idx, (label, raw_value) in enumerate(zip(labels, values)):
        value = max(0.0, float(raw_value))
        bh = 0.0 if y_max <= 0 else plot_h * value / y_max
        x = pad_l + bw * idx + (bw - bar_w) / 2
        y = pad_t + plot_h - bh
        color = palette[idx % len(palette)] if multicolor else "#2f6db5"
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}" rx="2"/>')

        if len(values) <= 40 or idx % label_step == 0:
            out.append(
                f'<text x="{x + bar_w/2:.1f}" y="{pad_t + plot_h + 16:.1f}" text-anchor="middle" '
                'fill="#607086" font-size="10" font-family="Arial, sans-serif">'
                f"{html.escape(str(label))}</text>"
            )

    out.append(
        f'<text x="{pad_l - 50:.1f}" y="{pad_t - 10:.1f}" text-anchor="start" '
        'fill="#4b5b71" font-size="11" font-family="Arial, sans-serif">'
        f"{html.escape(y_title)}</text>"
    )
    out.append(
        f'<text x="{w/2:.1f}" y="{h - 12:.1f}" text-anchor="middle" fill="#4b5b71" '
        f'font-size="11" font-family="Arial, sans-serif">{html.escape(x_title)}</text>'
    )
    out.append("</svg>")
    return "".join(out)


_STATS_MODULE: Any | None = None


def _load_stats_from_logs_module() -> Any:
    global _STATS_MODULE
    if _STATS_MODULE is not None:
        return _STATS_MODULE
    module_path = PROJECT_ROOT.parent / "tools" / "stats_from_logs.py"
    spec = importlib.util.spec_from_file_location("stats_from_logs_runtime", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tools/stats_from_logs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _STATS_MODULE = module
    return module


def _as_container_candidate(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_text(value)
    m = re.search(r"[A-Z]{4}\d{7}", normalized)
    if m:
        return m.group(0)
    return None


def _build_log_rows_from_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(run["items"], start=1):
        # Для точности берем expected_text; label может содержать bbox.
        expected = _as_container_candidate(item.get("expected_text")) or _as_container_candidate(item.get("expected_label"))
        predicted = _as_container_candidate(item.get("normalized_predicted") or item.get("predicted_text"))
        elapsed_ms = float(item.get("elapsed_ms") or 0.0)
        stage_timings = item.get("stage_timings") or {}
        recognition_ms = (
            float(stage_timings.get("bbox_ms") or 0.0)
            + float(stage_timings.get("ocr_ms") or 0.0)
            + float(stage_timings.get("validation_ms") or 0.0)
            + float(stage_timings.get("postprocess_ms") or 0.0)
        )
        elapsed_sec = max(0.0, elapsed_ms / 1000.0)
        recognition_sec = max(0.0, recognition_ms / 1000.0)
        if recognition_sec <= 0.0:
            recognition_sec = elapsed_sec
        overhead_sec = max(0.0, elapsed_sec - recognition_sec)
        rows.append(
            {
                "request_id": idx,
                "elapsed_sec": elapsed_sec,
                "recognition_elapsed_sec": recognition_sec,
                "queue_overhead_sec": overhead_sec,
                "photos": [
                    {
                        "file": item.get("file_name"),
                        "type": item.get("number_type") or "unknown",
                        "expected": [expected] if expected else [],
                        "predicted": [predicted] if predicted else [],
                    }
                ],
            }
        )
    return rows


def _build_stage_chart_png(
    stats_module: Any, run: dict[str, Any]
) -> tuple[str | None, list[tuple[int, str, str]], list[tuple[int, dict[str, float]]]]:
    items = run["items"]
    stage_defs = [
        ("read_ms", "read file", "#2f6db5"),
        ("decode_ms", "decode image", "#1f9d8b"),
        ("bbox_ms", "bbox detection", "#d9822b"),
        ("ocr_ms", "ocr", "#8e24aa"),
        ("validation_ms", "validation", "#e65100"),
        ("postprocess_ms", "postprocess", "#455a64"),
    ]
    legend_rows = [(idx, title, color) for idx, (_k, title, color) in enumerate(stage_defs, start=1)]

    per_request: list[tuple[int, dict[str, float]]] = []
    for req_idx, item in enumerate(items, start=1):
        raw = item.get("stage_timings") or {}
        stage_values: dict[str, float] = {}
        for key, _title, _color in stage_defs:
            try:
                stage_values[key] = max(0.0, float(raw.get(key) or 0.0))
            except (TypeError, ValueError):
                stage_values[key] = 0.0
        # Совместимость со старыми прогонами без таймингов этапов.
        if sum(stage_values.values()) <= 0.0:
            try:
                fallback_total = max(0.0, float(item.get("elapsed_ms") or 0.0))
            except (TypeError, ValueError):
                fallback_total = 0.0
            stage_values["postprocess_ms"] = fallback_total
        per_request.append((req_idx, stage_values))

    if not per_request:
        return None, legend_rows, per_request

    xs = [idx for idx, _ in per_request]

    def _plot(ax):
        bottoms = [0.0] * len(xs)
        for key, title, color in stage_defs:
            vals = [stages.get(key, 0.0) for _, stages in per_request]
            ax.bar(xs, vals, bottom=bottoms, color=color, label=title)
            bottoms = [bottoms[i] + vals[i] for i in range(len(vals))]
        ax.set_xticks(xs, [str(i) for i in xs])
        ax.set_title("Stage Breakdown by request (stacked)")
        ax.set_xlabel("Request number (1..n)")
        ax.set_ylabel("Time (ms)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)

    img = stats_module.make_plot_png(_plot)
    return img, legend_rows, per_request


def _build_stage_chart_section(
    img_stage: str | None,
    legend_rows: list[tuple[int, str, str]],
    per_request: list[tuple[int, dict[str, float]]],
) -> str:
    if not img_stage:
        return ""
    legend_html = []
    for idx, name, color in legend_rows:
        legend_html.append(
            '<div style="display:flex;align-items:center;gap:6px;padding:2px 8px;border:1px solid #d8dfea;border-radius:8px;background:#fff;">'
            f'<span style="display:inline-block;width:11px;height:11px;background:{color};border-radius:2px;"></span>'
            f'<span style="font-size:12px;color:#36485f;">{idx}. {html.escape(name)}</span>'
            "</div>"
        )

    req_rows = []
    for req_idx, stages in per_request:
        total = sum(stages.values())
        req_rows.append(
            "<tr>"
            f"<td>{req_idx}</td>"
            f"<td>{stages.get('read_ms', 0.0):.3f}</td>"
            f"<td>{stages.get('decode_ms', 0.0):.3f}</td>"
            f"<td>{stages.get('bbox_ms', 0.0):.3f}</td>"
            f"<td>{stages.get('ocr_ms', 0.0):.3f}</td>"
            f"<td>{stages.get('validation_ms', 0.0):.3f}</td>"
            f"<td>{stages.get('postprocess_ms', 0.0):.3f}</td>"
            f"<td>{total:.3f}</td>"
            "</tr>"
        )

    return (
        '<h2>Stage Breakdown</h2>'
        '<div class="subtitle" style="color:#5a6472;margin-bottom:8px;">Units: time = milliseconds (ms)</div>'
        '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px;">'
        f"{''.join(legend_html)}"
        "</div>"
        '<div class="plots">'
        f'<div class="plot"><img alt="Stage breakdown" src="data:image/png;base64,{img_stage}" /></div>'
        "</div>"
        "<table><thead><tr>"
        "<th>Request #</th><th>read (ms)</th><th>decode (ms)</th><th>bbox (ms)</th><th>ocr (ms)</th><th>validation (ms)</th><th>postprocess (ms)</th><th>total (ms)</th>"
        "</tr></thead>"
        f"<tbody>{''.join(req_rows)}</tbody></table>"
    )


def _build_testing_html_report(run: dict[str, Any]) -> str:
    if str(run.get("mode", "")).lower() == "seal":
        # Для режима seal без аналитики по типам.
        return _build_testing_html_report_fallback(run)

    try:
        stats_module = _load_stats_from_logs_module()
        log_rows = _build_log_rows_from_run(run)
        prefix_map = stats_module.load_type_map(None)
        report = stats_module.build_stats(
            log_rows,
            prefix_map,
            count_unrecognized_in_error=True,
        )
        tmp_path = TEST_REPORTS_DIR / f"run_{run['run_id']}_stats_base.html"
        stats_module.write_html_report(report, tmp_path)
        base_html = tmp_path.read_text(encoding="utf-8", errors="ignore")
        request_table_block = _build_requests_table_section(run["items"])
        img_stage, legend_rows, per_request = _build_stage_chart_png(stats_module, run)
        stage_block = _build_stage_chart_section(img_stage, legend_rows, per_request)
        injected_block = "".join(block for block in (stage_block, request_table_block) if block)
        if injected_block:
            if "</div>\n</body>" in base_html:
                base_html = base_html.replace("</div>\n</body>", f"{injected_block}\n  </div>\n</body>")
            elif "</body>" in base_html:
                base_html = base_html.replace("</body>", f"{injected_block}\n</body>")
        return base_html
    except Exception as exc:
        logger.warning("stats_from_logs report build failed, fallback report used: %s", exc)
        return _build_testing_html_report_fallback(run)


def _build_request_rows_html(items: list[dict[str, Any]]) -> str:
    row_html: list[str] = []
    for it in items:
        row_html.append(
            "<tr>"
            f"<td>{html.escape(str(it.get('file_name') or ''))}</td>"
            f"<td>{html.escape(str(it.get('number_type') or 'unknown'))}</td>"
            f"<td>{html.escape(str(it.get('predicted_text') or ''))}</td>"
            f"<td>{html.escape(str(it.get('raw_text') or ''))}</td>"
            f"<td>{html.escape(str(it.get('expected_label') or ''))}</td>"
            f"<td>{html.escape(str(it.get('expected_text') or ''))}</td>"
            f"<td>{html.escape(str(it.get('status_code') or ''))}</td>"
            f"<td>{_fmt_num(it.get('elapsed_ms'))}</td>"
            f"<td>{html.escape(str(it.get('error') or ''))}</td>"
            "</tr>"
        )
    return "".join(row_html)


def _build_requests_table_section(items: list[dict[str, Any]]) -> str:
    row_html = _build_request_rows_html(items)
    return f"""
    <h2>Requests Table</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>File</th><th>Type</th><th>Predicted</th><th>Raw</th><th>Expected Label</th><th>Expected Text</th><th>Status</th><th>Latency ms</th><th>Error</th>
          </tr>
        </thead>
        <tbody>
          {row_html if row_html else '<tr><td colspan="9">No data</td></tr>'}
        </tbody>
      </table>
    </div>
"""


def _build_testing_html_report_fallback(run: dict[str, Any]) -> str:
    summary = run["summary"]
    items = run["items"]
    success = [it for it in items if it.get("status_code") == 200 and not it.get("error")]
    failed = [it for it in items if it.get("error") or it.get("status_code") != 200]
    latencies = [float(it["elapsed_ms"]) for it in items if it.get("elapsed_ms") is not None]
    median_ms = _percentile(latencies, 0.5) if latencies else None
    p95_ms = _percentile(latencies, 0.95) if latencies else None
    not_found_count = sum(1 for it in success if (it.get("normalized_predicted") or "") in {"", "NOTFOUND"})
    done_count = max(0, len(success) - not_found_count)

    stage_labels = [str(i) for i in range(1, len(items) + 1)]
    stage_values = [float(it.get("elapsed_ms") or 0.0) for it in items]
    stage_chart = _build_svg_bar_chart(
        labels=stage_labels,
        values=stage_values,
        title="Time by stage (stage = processing order 1..n)",
        y_title="Time, ms",
        x_title="Stage number",
        multicolor=True,
    )

    status_chart = _build_svg_bar_chart(
        labels=["DONE", "NOT_FOUND", "FAILED"],
        values=[float(done_count), float(not_found_count), float(len(failed))],
        title="Status distribution",
        y_title="Count",
        x_title="Status",
        multicolor=True,
    )

    accuracy_labels: list[str] = []
    accuracy_values: list[float] = []
    if summary.get("label_accuracy") is not None:
        accuracy_labels.append("Label accuracy")
        accuracy_values.append(float(summary["label_accuracy"]) * 100.0)
    if summary.get("text_accuracy") is not None:
        accuracy_labels.append("Text accuracy")
        accuracy_values.append(float(summary["text_accuracy"]) * 100.0)
    accuracy_chart = _build_svg_bar_chart(
        labels=accuracy_labels,
        values=accuracy_values,
        title="Accuracy metrics (%)",
        y_title="Percent",
        x_title="Metric",
        multicolor=True,
    )

    requests_table_block = _build_requests_table_section(items)

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Testing Report #{run['run_id']}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; color: #111; background: #f4f6fa; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
    h1, h2 {{ margin: 12px 0 8px; }}
    .meta {{ color: #5a6472; margin-bottom: 12px; }}
    .grid {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:10px; margin-bottom: 16px; }}
    .card {{ border:1px solid #d8dfea; border-radius:10px; padding:12px; background:#fff; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
    .label {{ font-size: 12px; color: #555; }}
    .value {{ font-size: 18px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0 16px; background: #fff; border: 1px solid #d8dfea; }}
    th, td {{ border: 1px solid #e6ebf3; padding: 8px; font-size: 12px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3fb; position: sticky; top: 0; z-index: 1; }}
    tbody tr:nth-child(odd) {{ background: #fbfcff; }}
    tbody tr:hover {{ background: #f1f6ff; }}
    .table-wrap {{ max-height: 620px; overflow:auto; border:1px solid #d8dfea; border-radius:10px; }}
    .plots {{ display:grid; grid-template-columns: 1fr; gap:12px; }}
    .plot {{ border:1px solid #d8dfea; border-radius:10px; background:#fff; padding:8px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); overflow:auto; }}
    .plot-empty {{ font-size: 13px; color: #5a6472; padding: 18px 12px; }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Testing Report</h1>
    <div class="meta">
      Run ID: {run['run_id']} |
      Created: {html.escape(str(run['created_at']))} |
      Dataset: {html.escape(str(run['dataset_dir']))} |
      Mode: {html.escape(str(run['mode']))} |
      Timeout: {run['timeout_sec']}s
    </div>

    <h2>Summary</h2>
    <div class="grid">
      <div class="card"><div class="label">Total images</div><div class="value">{summary.get('total_images', 0)}</div></div>
      <div class="card"><div class="label">OK requests</div><div class="value">{summary.get('ok_requests', 0)}</div></div>
      <div class="card"><div class="label">Failed requests</div><div class="value">{summary.get('failed_requests', 0)}</div></div>
      <div class="card"><div class="label">Not found</div><div class="value">{summary.get('not_found_count', 0)}</div></div>
      <div class="card"><div class="label">Latency mean (ms)</div><div class="value">{_fmt_num(summary.get('latency_mean_ms'))}</div></div>
      <div class="card"><div class="label">Latency median (ms)</div><div class="value">{_fmt_num(median_ms)}</div></div>
      <div class="card"><div class="label">Latency p95 (ms)</div><div class="value">{_fmt_num(p95_ms)}</div></div>
      <div class="card"><div class="label">Label accuracy</div><div class="value">{_fmt_pct(summary.get('label_accuracy'))}</div></div>
      <div class="card"><div class="label">Text accuracy</div><div class="value">{_fmt_pct(summary.get('text_accuracy'))}</div></div>
      <div class="card"><div class="label">Labels available</div><div class="value">{'yes' if run['labels_available'] else 'no'}</div></div>
      <div class="card"><div class="label">Text available</div><div class="value">{'yes' if run['text_available'] else 'no'}</div></div>
      <div class="card"><div class="label">Errors count</div><div class="value">{len(failed)}</div></div>
    </div>

    <h2>Charts</h2>
    <div class="plots">
      <div class="plot">{stage_chart}</div>
      <div class="plot">{status_chart}</div>
      <div class="plot">{accuracy_chart}</div>
    </div>

    {requests_table_block}

    <h2>Counts</h2>
    <div class="grid">
      <div class="card"><div class="label">Successful rows</div><div class="value">{len(success)}</div></div>
      <div class="card"><div class="label">Failed rows</div><div class="value">{len(failed)}</div></div>
    </div>
  </div>
</body>
</html>
"""
    return html_content


def _generate_testing_report_html_file(run_id: int) -> Path:
    run = _load_test_run_data(run_id)
    TEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = _testing_report_path(run_id)
    html_path.write_text(_build_testing_html_report(run), encoding="utf-8")
    _cleanup_test_reports(mode="policy", dry_run=False, reason="report_generated")
    return html_path


def _ensure_testing_report_html_file(run_id: int) -> Path:
    html_path = _testing_report_path(run_id)
    if html_path.exists():
        return html_path
    return _generate_testing_report_html_file(run_id)


def _run_dataset_test(payload: TestRunPayload) -> dict[str, Any]:
    cfg = _load_config()
    dataset_dir = Path(payload.dataset_dir).expanduser()
    if not dataset_dir.exists():
        raise HTTPException(status_code=400, detail=f"dataset_dir not found: {dataset_dir}")

    images_dir = dataset_dir / "images"
    source_images_dir = images_dir if images_dir.exists() and images_dir.is_dir() else dataset_dir
    labels_dir = dataset_dir / "labels"
    labels_dir = labels_dir if labels_dir.exists() and labels_dir.is_dir() else None
    text_dir = dataset_dir / "text"
    text_dir = text_dir if text_dir.exists() and text_dir.is_dir() else None

    image_files = sorted(
        [p for p in source_images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )
    if not image_files:
        raise HTTPException(status_code=400, detail="No images found in dataset")

    endpoint = "/RecognizeContainerNumber" if payload.mode == "container" else "/RecognizeSealNumber"
    url = f"{_service_base_url(cfg)}{endpoint}"
    items: list[dict[str, Any]] = []
    latencies: list[float] = []

    for img_path in image_files:
        expected_label = _read_expected_label(labels_dir, img_path)
        expected_text = _read_expected_text(text_dir, img_path)
        number_type = _detect_number_type_from_image_path(img_path, payload.mode)
        start = time.perf_counter()
        status_code, predicted, raw_text, normalized_predicted, error = None, "", "", "", None
        stage_timings: dict[str, float] = {}
        try:
            with img_path.open("rb") as fh:
                files = {"files": (img_path.name, fh.read(), _mime_for_filename(img_path.name))}
                resp = requests.post(url, files=files, timeout=payload.timeout_sec)
            status_code = resp.status_code
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
            try:
                obj = resp.json()
                fallback = json.dumps(obj, ensure_ascii=False)
            except ValueError:
                obj = None
                fallback = resp.text
            predicted = _extract_result_text(obj, fallback)
            raw_text = _extract_raw_text(obj, fallback)
            stage_timings = _extract_stage_timings(obj)
            normalized_predicted = _normalize_text(predicted)
            if not resp.ok:
                error = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            error = _format_request_error(exc, payload.timeout_sec)

        items.append(
            {
                "file_name": img_path.name,
                "number_type": number_type,
                "predicted_text": predicted,
                "raw_text": raw_text,
                "normalized_predicted": normalized_predicted,
                "expected_label": expected_label,
                "expected_text": expected_text,
                "status_code": status_code,
                "elapsed_ms": round(elapsed_ms, 3),
                "stage_timings": stage_timings,
                "error": error,
            }
        )

    return _finalize_and_store_test_run(
        dataset_id=str(dataset_dir),
        mode=payload.mode,
        endpoint=endpoint,
        timeout_sec=payload.timeout_sec,
        items=items,
        labels_available=(labels_dir is not None),
        text_available=(text_dir is not None),
    )


@app.on_event("startup")
def on_startup() -> None:
    _init_admin_db()
    if not ADMIN_CFG_PATH.exists():
        _save_config(ServiceConfig())
    _cleanup_test_reports(mode="policy", dry_run=False, reason="startup")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    if not UI_HTML_PATH.exists():
        return "<html><body><h3>admin_ui_page.html missing</h3></body></html>"
    html_text = UI_HTML_PATH.read_text(encoding="utf-8", errors="ignore")
    return html_text.replace("__ADMIN_UI_VISIBLE_TABS_JSON__", json.dumps(_visible_ui_tabs(), ensure_ascii=False))


@app.get("/api/service/config")
def get_service_config() -> dict[str, Any]:
    return {"config": _model_dump(_load_config())}


@app.post("/api/service/config")
def set_service_config(payload: ServiceConfig) -> dict[str, Any]:
    _save_config(payload)
    return {"config": _model_dump(payload)}


@app.get("/api/service/status")
def service_status() -> dict[str, Any]:
    cfg = _load_config()
    workers_used = _effective_workers(cfg.workers)
    with _managed_lock:
        proc = _managed_process
        running = _is_running(proc)
        pid = proc.pid if running and proc is not None else None
    return {
        "config": _model_dump(cfg),
        "effective_workers": workers_used,
        "managed_running": running,
        "managed_pid": pid,
        "health": _probe_health(_service_base_url(cfg)),
        "base_url": _service_base_url(cfg),
    }


@app.post("/api/service/start")
def start_service() -> dict[str, Any]:
    return _start_managed_service()


@app.post("/api/service/stop")
def stop_service() -> dict[str, Any]:
    return _stop_managed_service()


@app.post("/api/service/restart")
def restart_service() -> dict[str, Any]:
    return _restart_managed_service()


@app.post("/api/service/workers")
def set_workers(payload: WorkersPayload) -> dict[str, Any]:
    cfg = _load_config()
    cfg.workers = payload.workers
    _save_config(cfg)
    with _managed_lock:
        running = _is_running(_managed_process)
    if running:
        _restart_managed_service()
    return {"workers": cfg.workers, "restarted": running}


@app.get("/api/logs")
def get_logs(
    source: str = Query(default="app", pattern="^(app|stdout|all)$"),
    lines: int = Query(default=300, ge=20, le=5000),
) -> dict[str, str]:
    app_chunk = _tail_lines(APP_LOG_PATH, lines)
    stdout_chunk = _tail_lines(SERVICE_STDOUT_LOG_PATH, lines)
    if source == "app":
        return {"text": app_chunk}
    if source == "stdout":
        return {"text": stdout_chunk}
    text = ""
    if app_chunk:
        text += "===== app.log =====\n" + app_chunk + "\n"
    if stdout_chunk:
        text += "===== service_stdout.log =====\n" + stdout_chunk
    return {"text": text}


@app.get("/api/debug/config")
def get_debug_config() -> dict[str, Any]:
    return _proxy_json("GET", "/admin/debug/config")


@app.post("/api/debug/config")
def set_debug_config(payload: DebugConfigPayload) -> dict[str, Any]:
    cfg = _load_config()
    cfg.debug_visualization = payload.enabled
    _save_config(cfg)
    return _proxy_json("POST", "/admin/debug/config", _model_dump(payload))


@app.post("/api/models/reload")
def reload_models() -> dict[str, Any]:
    return _proxy_json("POST", "/admin/models/reload")


@app.get("/api/debug/meta")
def get_debug_meta() -> dict[str, Any]:
    return _proxy_json("GET", "/admin/debug/last-meta")


@app.get("/api/debug/image")
def get_debug_image() -> Response:
    cfg = _load_config()
    try:
        resp = requests.get(f"{_service_base_url(cfg)}/admin/debug/last-image", timeout=15)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Service request failed: {exc}") from exc
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="No debug image yet")
    if not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return Response(content=resp.content, media_type="image/jpeg")


@app.get("/api/recognized")
def get_recognized(limit: int = Query(default=500, ge=1, le=5000)) -> dict[str, Any]:
    return _proxy_json("GET", f"/admin/recognized?limit={limit}")


@app.post("/api/testing/run")
def run_testing(payload: TestRunPayload) -> dict[str, Any]:
    return _run_dataset_test(payload)


@app.post("/api/testing/run-files")
async def run_testing_files(
    mode: str = Form(default="container"),
    timeout_sec: float = Form(default=DEFAULT_TEST_TIMEOUT_SEC),
    files: list[UploadFile] = File(default_factory=list),
) -> dict[str, Any]:
    mode = (mode or "").strip().lower()
    if mode not in {"container", "seal"}:
        raise HTTPException(status_code=400, detail="mode must be 'container' or 'seal'")
    if timeout_sec < 1.0 or timeout_sec > MAX_TEST_TIMEOUT_SEC:
        raise HTTPException(status_code=400, detail=f"timeout_sec must be in range [1, {int(MAX_TEST_TIMEOUT_SEC)}]")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    cfg = _load_config()
    endpoint = "/RecognizeContainerNumber" if mode == "container" else "/RecognizeSealNumber"
    url = f"{_service_base_url(cfg)}{endpoint}"

    images: list[tuple[str, str, bytes]] = []
    labels_by_stem: dict[str, str] = {}
    text_by_stem: dict[str, str] = {}
    roots: set[str] = set()

    for upl in files:
        rel = (upl.filename or "").replace("\\", "/").strip("/")
        if not rel:
            continue
        parts = [p for p in rel.split("/") if p]
        if not parts:
            continue
        roots.add(parts[0])

        low_rel = rel.lower()
        parent = parts[-2].lower() if len(parts) >= 2 else ""
        name = parts[-1]
        stem = Path(name).stem
        suffix = Path(name).suffix.lower()
        data = await upl.read()

        if suffix in IMAGE_EXTS and ("/images/" in f"/{low_rel}/" or parent == "images" or len(parts) == 1):
            images.append((rel, name, data))
            continue

        if suffix != ".txt":
            continue

        txt = data.decode("utf-8", errors="ignore")
        if "/labels/" in f"/{low_rel}/" or parent == "labels":
            found = CONTAINER_RE.findall(txt.upper())
            label = found[0] if found else (_normalize_text(txt) or None)
            if label:
                labels_by_stem[stem] = label
        elif "/text/" in f"/{low_rel}/" or parent == "text":
            norm = _normalize_text(txt)
            if norm:
                text_by_stem[stem] = norm

    if not images:
        raise HTTPException(status_code=400, detail="No image files found in selected folder")

    items: list[dict[str, Any]] = []
    for rel, name, data in images:
        stem = Path(name).stem
        expected_label = labels_by_stem.get(stem)
        expected_text = text_by_stem.get(stem)
        number_type = _detect_number_type_from_image_bytes(data, mode)

        start = time.perf_counter()
        status_code, predicted, raw_text, normalized_predicted, error = None, "", "", "", None
        stage_timings: dict[str, float] = {}
        try:
            resp = requests.post(
                url,
                files={"files": (name, data, _mime_for_filename(name))},
                timeout=timeout_sec,
            )
            status_code = resp.status_code
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            try:
                obj = resp.json()
                fallback = json.dumps(obj, ensure_ascii=False)
            except ValueError:
                obj = None
                fallback = resp.text
            predicted = _extract_result_text(obj, fallback)
            raw_text = _extract_raw_text(obj, fallback)
            stage_timings = _extract_stage_timings(obj)
            normalized_predicted = _normalize_text(predicted)
            if not resp.ok:
                error = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            error = _format_request_error(exc, timeout_sec)

        items.append(
            {
                "file_name": rel,
                "number_type": number_type,
                "predicted_text": predicted,
                "raw_text": raw_text,
                "normalized_predicted": normalized_predicted,
                "expected_label": expected_label,
                "expected_text": expected_text,
                "status_code": status_code,
                "elapsed_ms": round(elapsed_ms, 3),
                "stage_timings": stage_timings,
                "error": error,
            }
        )

    dataset_id = f"uploaded:{next(iter(roots))}" if len(roots) == 1 else "uploaded:folder"
    return _finalize_and_store_test_run(
        dataset_id=dataset_id,
        mode=mode,
        endpoint=endpoint,
        timeout_sec=timeout_sec,
        items=items,
        labels_available=bool(labels_by_stem),
        text_available=bool(text_by_stem),
    )


@app.get("/api/testing/runs")
def list_testing_runs(limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    with sqlite3.connect(ADMIN_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, dataset_dir, mode, labels_available, text_available, summary_json
            FROM test_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    runs = []
    for row in rows:
        run_id = int(row["id"])
        runs.append(
            {
                "id": run_id,
                "created_at": row["created_at"],
                "dataset_dir": row["dataset_dir"],
                "mode": row["mode"],
                "labels_available": bool(row["labels_available"]),
                "text_available": bool(row["text_available"]),
                "summary": json.loads(row["summary_json"]),
                "has_report": _testing_report_path(run_id).exists(),
            }
        )
    return {"runs": runs}


@app.get("/api/testing/runs/{run_id}")
def get_testing_run(run_id: int) -> dict[str, Any]:
    run = _load_test_run_data(run_id)
    return {"run_id": run_id, "summary": run["summary"], "items": run["items"]}


@app.get("/api/testing/runs/{run_id}/report/download")
def download_testing_report(run_id: int):
    html_path = _ensure_testing_report_html_file(run_id)
    return FileResponse(
        str(html_path),
        media_type="text/html; charset=utf-8",
        filename=f"testing_report_run_{run_id}.html",
    )


@app.get("/api/testing/runs/{run_id}/report/view", response_class=HTMLResponse)
def view_testing_report(run_id: int) -> str:
    html_path = _ensure_testing_report_html_file(run_id)
    return html_path.read_text(encoding="utf-8", errors="ignore")


@app.get("/api/testing/reports/cleanup/config")
def get_testing_reports_cleanup_config() -> dict[str, Any]:
    return {"config": _reports_cleanup_settings()}


@app.post("/api/testing/reports/cleanup")
def cleanup_testing_reports(payload: CleanupReportsPayload) -> dict[str, Any]:
    return {"result": _cleanup_test_reports(mode=payload.mode, dry_run=payload.dry_run, reason="manual")}
