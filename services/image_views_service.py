from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock

from services.config import DATA_DIR

IMAGE_VIEWS_FILE = DATA_DIR / "image_views.json"
_VIEWS_LOCK = Lock()


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_relative_path(path: str) -> str | None:
    value = str(path or "").strip().replace("\\", "/").lstrip("/")
    if not value:
        return None
    parts = Path(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return Path(*parts).as_posix()


def _load_views_unlocked() -> dict[str, str]:
    if not IMAGE_VIEWS_FILE.exists():
        return {}
    try:
        data = json.loads(IMAGE_VIEWS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in data.items():
        rel = _safe_relative_path(str(key))
        viewed_at = str(value or "").strip()
        if rel and viewed_at:
            result[rel] = viewed_at
    return result


def _save_views_unlocked(data: dict[str, str]) -> None:
    IMAGE_VIEWS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = IMAGE_VIEWS_FILE.with_suffix(IMAGE_VIEWS_FILE.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(IMAGE_VIEWS_FILE)


def load_views() -> dict[str, str]:
    with _VIEWS_LOCK:
        return _load_views_unlocked()


def mark_viewed(paths: list[str]) -> dict[str, object]:
    viewed_at = _now_text()
    with _VIEWS_LOCK:
        data = _load_views_unlocked()
        updated = 0
        for path in paths:
            rel = _safe_relative_path(path)
            if not rel or rel in data:
                continue
            data[rel] = viewed_at
            updated += 1
        if updated > 0:
            _save_views_unlocked(data)
    return {"ok": True, "updated": updated, "viewed_at": viewed_at}


def remove_views(paths: list[str]) -> int:
    with _VIEWS_LOCK:
        data = _load_views_unlocked()
        removed = 0
        for path in paths:
            rel = _safe_relative_path(path)
            if rel and data.pop(rel, None) is not None:
                removed += 1
        if removed > 0:
            _save_views_unlocked(data)
        return removed
