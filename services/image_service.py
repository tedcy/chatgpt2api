from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps

from services.config import config
from services.image_storage_service import image_storage_service
from services.image_tags_service import load_tags, remove_tags
from services.image_views_service import load_views, remove_views

THUMBNAIL_SIZE = (320, 320)


def _cleanup_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def _safe_relative_path(path: str) -> str:
    value = str(path or "").strip().replace("\\", "/").lstrip("/")
    if not value:
        raise HTTPException(status_code=404, detail="image not found")
    parts = Path(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise HTTPException(status_code=404, detail="image not found")
    return Path(*parts).as_posix()


def _safe_image_path(relative_path: str) -> Path:
    rel = _safe_relative_path(relative_path)
    root = config.images_dir.resolve()
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="image not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return path


def get_image_response(relative_path: str) -> FileResponse | Response:
    if image_storage_service.has_local(relative_path):
        return FileResponse(_safe_image_path(relative_path))
    return Response(content=image_storage_service.get_bytes(relative_path), media_type="image/png")


def _thumbnail_path(relative_path: str) -> Path:
    rel = _safe_relative_path(relative_path)
    return config.image_thumbnails_dir / f"{rel}.png"


def thumbnail_url(base_url: str, relative_path: str) -> str:
    return f"{base_url.rstrip('/')}/image-thumbnails/{_safe_relative_path(relative_path)}"


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def ensure_thumbnail(relative_path: str) -> Path:
    target = _thumbnail_path(relative_path)
    source_mtime = 0.0
    source: Path | None = None
    if image_storage_service.has_local(relative_path):
        source = _safe_image_path(relative_path)
        source_mtime = source.stat().st_mtime
    if target.exists() and (not source_mtime or target.stat().st_mtime >= source_mtime):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        image_source = source if source is not None else io.BytesIO(image_storage_service.get_bytes(relative_path))
        with Image.open(image_source) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            image.save(target, format="PNG", optimize=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="failed to create thumbnail") from exc
    return target


def get_thumbnail_response(relative_path: str) -> FileResponse:
    return FileResponse(ensure_thumbnail(relative_path))


def get_image_download_response(relative_path: str) -> FileResponse:
    if image_storage_service.has_local(relative_path):
        path = _safe_image_path(relative_path)
        return FileResponse(path, filename=path.name)
    rel = _safe_relative_path(relative_path)
    return Response(
        content=image_storage_service.get_bytes(rel),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{Path(rel).name}"'},
    )


def cleanup_image_thumbnails() -> int:
    thumbnails_root = config.image_thumbnails_dir
    known_paths = image_storage_service.known_paths(refresh_local=True)
    removed = 0
    for path in thumbnails_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(thumbnails_root).as_posix()
        if not rel.endswith(".png") or rel[:-4] not in known_paths:
            path.unlink()
            removed += 1
    _cleanup_empty_dirs(thumbnails_root)
    return removed

def _normalize_page(value: int | str, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _normalize_page_size(value: int | str, default: int = 12) -> int:
    try:
        return min(100, max(1, int(value)))
    except (TypeError, ValueError):
        return default


def _normalize_view_status(value: str) -> str:
    status = str(value or "").strip().lower()
    return status if status in {"viewed", "unviewed"} else "all"


def list_images(
    base_url: str,
    start_date: str = "",
    end_date: str = "",
    *,
    page: int | str = 1,
    page_size: int | str = 12,
    tags: list[str] | None = None,
    view_status: str = "all",
    refresh: bool = False,
) -> dict[str, object]:
    if refresh:
        config.cleanup_old_images()
        cleanup_image_thumbnails()
    all_tags = load_tags()
    all_views = load_views()
    selected_tags = [tag for tag in (tags or []) if tag]
    items = [
        {
            **item,
            "url": str(item.get("url") or f"{base_url.rstrip('/')}/images/{item['path']}"),
            "thumbnail_url": thumbnail_url(base_url, str(item["path"])),
            "tags": all_tags.get(str(item["path"]), []),
            "viewed": bool(all_views.get(str(item["path"]))),
            "viewed_at": all_views.get(str(item["path"])),
        }
        for item in image_storage_service.list_items(base_url, start_date, end_date, refresh=refresh)
    ]
    if selected_tags:
        items = [item for item in items if all(tag in item.get("tags", []) for tag in selected_tags)]
    view_counts = {
        "all": len(items),
        "viewed": sum(1 for item in items if item.get("viewed")),
        "unviewed": sum(1 for item in items if not item.get("viewed")),
    }
    normalized_view_status = _normalize_view_status(view_status)
    if normalized_view_status == "viewed":
        items = [item for item in items if item.get("viewed")]
    elif normalized_view_status == "unviewed":
        items = [item for item in items if not item.get("viewed")]
    total = len(items)
    normalized_page_size = _normalize_page_size(page_size)
    page_count = max(1, (total + normalized_page_size - 1) // normalized_page_size)
    normalized_page = min(_normalize_page(page), page_count)
    start = (normalized_page - 1) * normalized_page_size
    paged_items = items[start:start + normalized_page_size]
    groups: dict[str, list[dict[str, object]]] = {}
    for item in paged_items:
        groups.setdefault(str(item["date"]), []).append(item)
    return {
        "items": paged_items,
        "groups": [{"date": key, "items": value} for key, value in groups.items()],
        "total": total,
        "page": normalized_page,
        "page_size": normalized_page_size,
        "view_counts": view_counts,
    }


def delete_images(
    paths: list[str] | None = None,
    start_date: str = "",
    end_date: str = "",
    all_matching: bool = False,
    tags: list[str] | None = None,
    view_status: str = "all",
) -> dict[str, int]:
    root = config.images_dir.resolve()
    if all_matching:
        all_tags = load_tags()
        all_views = load_views()
        selected_tags = [tag for tag in (tags or []) if tag]
        normalized_view_status = _normalize_view_status(view_status)
        targets = []
        for item in image_storage_service.list_items("", start_date=start_date, end_date=end_date):
            rel = str(item["path"])
            if selected_tags and not all(tag in all_tags.get(rel, []) for tag in selected_tags):
                continue
            viewed = bool(all_views.get(rel))
            if normalized_view_status == "viewed" and not viewed:
                continue
            if normalized_view_status == "unviewed" and viewed:
                continue
            targets.append(rel)
    else:
        targets = paths or []
    removed = 0
    for item in targets:
        path = (root / item).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if image_storage_service.delete(item):
            removed += 1
        for thumbnail in (_thumbnail_path(item), config.image_thumbnails_dir / _safe_relative_path(item)):
            if thumbnail.is_file():
                thumbnail.unlink()
        remove_tags(item)
        remove_views([item])
    _cleanup_empty_dirs(root)
    _cleanup_empty_dirs(config.image_thumbnails_dir)
    return {"removed": removed}


def download_images_zip(paths: list[str]) -> io.BytesIO:
    root = config.images_dir.resolve()
    buf = io.BytesIO()
    added = 0
    used_names: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in paths:
            rel = _safe_relative_path(item)
            path = (root / rel).resolve()
            payload: bytes | None = None
            try:
                path.relative_to(root)
            except ValueError:
                continue
            if path.is_file():
                payload = path.read_bytes()
            else:
                try:
                    payload = image_storage_service.get_bytes(rel)
                except Exception:
                    continue
            name = path.name
            if name in used_names:
                stem = path.stem
                suffix = path.suffix
                counter = 2
                while f"{stem}_{counter}{suffix}" in used_names:
                    counter += 1
                name = f"{stem}_{counter}{suffix}"
            used_names.add(name)
            zf.writestr(name, payload)
            added += 1
    if added == 0:
        raise HTTPException(status_code=404, detail="no images found")
    buf.seek(0)
    return buf
