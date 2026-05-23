from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from services.config import DATA_DIR, config
from services.content_filter import request_text
from services.account_service import account_service
from services.log_service import LOG_TYPE_CALL, log_service
from services.protocol import openai_v1_image_edit, openai_v1_image_generations

TASK_STATUS_QUEUED = "queued"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCESS = "success"
TASK_STATUS_ERROR = "error"
TERMINAL_STATUSES = {TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}
UNFINISHED_STATUSES = {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING}


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _timestamp(value: object) -> float:
    if not isinstance(value, str) or not value.strip():
        return 0.0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:26], fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _clean(value: object, default: str = "") -> str:
    return str(value or default).strip()


def _excerpt(value: object, limit: int = 1000) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _owner_id(identity: dict[str, object]) -> str:
    return _clean(identity.get("id")) or "anonymous"


def _task_key(owner_id: str, task_id: str) -> str:
    return f"{owner_id}:{task_id}"


def _collect_image_urls(data: list[Any]) -> list[str]:
    urls: list[str] = []
    for item in data:
        if isinstance(item, dict):
            url = item.get("url")
            if isinstance(url, str) and url:
                urls.append(url)
    return urls


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    item = {
        "id": task.get("id"),
        "status": task.get("status"),
        "mode": task.get("mode"),
        "model": task.get("model"),
        "size": task.get("size"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
    }
    if task.get("started_at"):
        item["started_at"] = task.get("started_at")
    if task.get("finished_at"):
        item["finished_at"] = task.get("finished_at")
    if task.get("duration_ms") is not None:
        item["duration_ms"] = task.get("duration_ms")
    if task.get("data") is not None:
        item["data"] = task.get("data")
    if task.get("error"):
        item["error"] = task.get("error")
    return item


class ImageTaskService:
    def __init__(
        self,
        path: Path,
        *,
        generation_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_generations.handle,
        edit_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_edit.handle,
        retention_days_getter: Callable[[], int] | None = None,
    ):
        self.path = path
        self.generation_handler = generation_handler
        self.edit_handler = edit_handler
        self.retention_days_getter = retention_days_getter or (lambda: config.image_retention_days)
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._pending_payloads: dict[str, tuple[str, dict[str, Any], dict[str, object], str]] = {}
        self._task_start_cooldowns: list[float] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._tasks = self._load_locked()
            changed = self._recover_unfinished_locked()
            changed = self._cleanup_locked() or changed
            if changed:
                self._save_locked()

    def submit_generation(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        base_url: str,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "model": model,
            "n": 1,
            "size": size,
            "response_format": "url",
            "base_url": base_url,
        }
        return self._submit(identity, client_task_id=client_task_id, mode="generate", payload=payload)

    def submit_edit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        base_url: str,
        images: list[tuple[bytes, str, str]],
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "images": images,
            "model": model,
            "n": 1,
            "size": size,
            "response_format": "url",
            "base_url": base_url,
        }
        return self._submit(identity, client_task_id=client_task_id, mode="edit", payload=payload)

    def list_tasks(self, identity: dict[str, object], task_ids: list[str]) -> dict[str, Any]:
        owner = _owner_id(identity)
        requested_ids = [_clean(task_id) for task_id in task_ids if _clean(task_id)]
        with self._lock:
            if self._cleanup_locked():
                self._save_locked()
            items = []
            missing_ids = []
            for task_id in requested_ids:
                task = self._tasks.get(_task_key(owner, task_id))
                if task is None:
                    missing_ids.append(task_id)
                else:
                    items.append(_public_task(task))
            if not requested_ids:
                items = [
                    _public_task(task)
                    for task in self._tasks.values()
                    if task.get("owner_id") == owner
                ]
                items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
                missing_ids = []
            return {"items": items, "missing_ids": missing_ids, "summary": self._summary_locked(owner)}

    def stop_processing(self, identity: dict[str, object], task_ids: list[str] | None = None) -> dict[str, Any]:
        owner = _owner_id(identity)
        requested_keys = {
            _task_key(owner, task_id)
            for task_id in [_clean(task_id) for task_id in (task_ids or [])]
            if task_id
        }
        stopped_count = 0
        with self._lock:
            for key, task in self._tasks.items():
                if task.get("owner_id") != owner or task.get("status") != TASK_STATUS_QUEUED:
                    continue
                if requested_keys and key not in requested_keys:
                    continue
                task["status"] = TASK_STATUS_ERROR
                task["error"] = "已停止处理，未开始的图片任务已跳过"
                task["updated_at"] = _now_iso()
                self._pending_payloads.pop(key, None)
                stopped_count += 1
            if stopped_count:
                self._save_locked()
            summary = self._summary_locked(owner)
        return {"stopped_count": stopped_count, "summary": summary}

    def _submit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        mode: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = _clean(client_task_id)
        if not task_id:
            raise ValueError("client_task_id is required")
        owner = _owner_id(identity)
        key = _task_key(owner, task_id)
        now = _now_iso()
        starts: list[tuple[str, str, dict[str, Any], dict[str, object], str]] = []
        with self._lock:
            cleaned = self._cleanup_locked()
            task = self._tasks.get(key)
            if task is not None:
                if cleaned:
                    self._save_locked()
                return _public_task(task)
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": TASK_STATUS_QUEUED,
                "mode": mode,
                "model": _clean(payload.get("model"), "gpt-image-2"),
                "size": _clean(payload.get("size")),
                "created_at": now,
                "updated_at": now,
            }
            self._tasks[key] = task
            self._pending_payloads[key] = (mode, payload, dict(identity), _clean(payload.get("model"), "gpt-image-2"))
            self._save_locked()
            starts = self._start_queued_tasks_locked()

        self._start_task_threads(starts)
        return _public_task(task)

    def _task_worker_limit(self) -> int:
        try:
            account_count = len(account_service.list_tokens())
        except Exception:
            account_count = 1
        try:
            per_account = max(1, int(config.image_account_concurrency or 1))
        except Exception:
            per_account = 1
        return max(1, max(1, account_count) * per_account)

    def _prune_task_start_cooldowns_locked(self) -> None:
        now = time.monotonic()
        self._task_start_cooldowns = [until for until in self._task_start_cooldowns if until > now]

    @staticmethod
    def _task_next_interval_delay() -> tuple[float, float, float]:
        base_secs = float(config.image_task_next_interval_secs)
        if base_secs <= 0:
            return 0.0, 0.0, 0.0
        jitter_min = config.image_poll_jitter_min_secs
        jitter_max = config.image_poll_jitter_max_secs
        jitter_secs = random.uniform(jitter_min, jitter_max) if jitter_max > 0 or jitter_min > 0 else 0.0
        wait_secs = base_secs + jitter_secs
        return base_secs, jitter_secs, wait_secs

    def _start_queued_tasks_locked(self) -> list[tuple[str, str, dict[str, Any], dict[str, object], str]]:
        self._prune_task_start_cooldowns_locked()
        running = sum(1 for task in self._tasks.values() if task.get("status") == TASK_STATUS_RUNNING)
        cooling_down = len(self._task_start_cooldowns)
        slots = max(0, self._task_worker_limit() - running - cooling_down)
        if slots <= 0:
            return []

        starts: list[tuple[str, str, dict[str, Any], dict[str, object], str]] = []
        changed = False
        queued = [
            (key, task)
            for key, task in self._tasks.items()
            if task.get("status") == TASK_STATUS_QUEUED
        ]
        queued.sort(key=lambda item: (_timestamp(item[1].get("created_at")), str(item[1].get("id") or "")))
        for key, task in queued:
            if len(starts) >= slots:
                break
            pending = self._pending_payloads.pop(key, None)
            if pending is None:
                task["status"] = TASK_STATUS_ERROR
                task["error"] = "图片任务排队数据丢失，请重新提交"
                task["updated_at"] = _now_iso()
                changed = True
                continue
            now = _now_iso()
            task["status"] = TASK_STATUS_RUNNING
            task["error"] = ""
            task["started_at"] = now
            task["updated_at"] = now
            starts.append((key, *pending))
            changed = True

        if changed:
            self._save_locked()
        return starts

    def _start_queued_tasks(self) -> None:
        with self._lock:
            starts = self._start_queued_tasks_locked()
        self._start_task_threads(starts)

    def _start_queued_tasks_after_delay(self, delay_secs: float) -> None:
        if delay_secs <= 0:
            self._start_queued_tasks()
            return

        def delayed_start() -> None:
            time.sleep(delay_secs)
            self._start_queued_tasks()

        thread = threading.Thread(
            target=delayed_start,
            name="image-task-next-delay",
            daemon=True,
        )
        thread.start()

    def _start_task_threads(self, starts: list[tuple[str, str, dict[str, Any], dict[str, object], str]]) -> None:
        for key, mode, payload, identity, model in starts:
            task_id = key.rsplit(":", 1)[-1]
            thread = threading.Thread(
                target=self._run_task,
                args=(key, mode, payload, identity, model),
                name=f"image-task-{task_id[:16]}",
                daemon=True,
            )
            thread.start()

    def _run_task(
        self,
        key: str,
        mode: str,
        payload: dict[str, Any],
        identity: dict[str, object],
        model: str,
    ) -> None:
        started = time.time()
        started_at = datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S")
        self._update_task(key, status=TASK_STATUS_RUNNING, error="", started_at=started_at)
        next_delay_secs = 0.0
        try:
            handler = self.edit_handler if mode == "edit" else self.generation_handler
            result = handler(payload)
            if not isinstance(result, dict):
                raise RuntimeError("image task returned streaming result unexpectedly")
            data = result.get("data")
            if not isinstance(data, list) or not data:
                upstream = _clean(result.get("message"))
                if upstream:
                    message = upstream
                else:
                    message = "号池中没有可用账号或所有账号均被限流，请检查号池状态（账号额度、是否被封禁、是否到达生图上限）"
                raise RuntimeError(message)
            next_delay_secs = self._complete_task(
                key,
                status=TASK_STATUS_SUCCESS,
                data=data,
                error="",
                finished_at=_now_iso(),
                duration_ms=int((time.time() - started) * 1000),
            )
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用完成",
                request_preview=request_text(payload.get("prompt")),
                urls=_collect_image_urls(data),
            )
        except Exception as exc:
            error_message = str(exc) or "image task failed"
            next_delay_secs = self._complete_task(
                key,
                status=TASK_STATUS_ERROR,
                error=error_message,
                data=[],
                finished_at=_now_iso(),
                duration_ms=int((time.time() - started) * 1000),
            )
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用失败",
                request_preview=request_text(payload.get("prompt")),
                status="failed",
                error=error_message,
            )
        finally:
            self._start_queued_tasks_after_delay(next_delay_secs)

    def _log_call(
        self,
        identity: dict[str, object],
        mode: str,
        model: str,
        started: float,
        suffix: str,
        *,
        request_preview: str = "",
        status: str = "success",
        error: str = "",
        urls: list[str] | None = None,
    ) -> None:
        endpoint = "/v1/images/edits" if mode == "edit" else "/v1/images/generations"
        summary_prefix = "图生图" if mode == "edit" else "文生图"
        detail = {
            "key_id": identity.get("id"),
            "key_name": identity.get("name"),
            "role": identity.get("role"),
            "endpoint": endpoint,
            "model": model,
            "started_at": datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": _now_iso(),
            "duration_ms": int((time.time() - started) * 1000),
            "status": status,
        }
        if request_preview:
            detail["request_text"] = request_preview
        if error:
            detail["error"] = _excerpt(error)
        if urls:
            detail["urls"] = list(dict.fromkeys(urls))
        try:
            log_service.add(LOG_TYPE_CALL, f"{summary_prefix}{suffix}", detail)
        except Exception:
            pass

    def _summary_locked(self, owner: str) -> dict[str, object]:
        owner_tasks = [task for task in self._tasks.values() if task.get("owner_id") == owner]
        queued_count = sum(1 for task in owner_tasks if task.get("status") == TASK_STATUS_QUEUED)
        running_count = sum(1 for task in owner_tasks if task.get("status") == TASK_STATUS_RUNNING)
        runtime = account_service.image_runtime_summary()
        worker_capacity = int(runtime.get("worker_capacity") or 0)
        average_duration_ms = runtime.get("recent_average_duration_ms")
        unfinished_count = queued_count + running_count
        estimated_processing_ms_per_account = None
        if isinstance(average_duration_ms, int) and average_duration_ms > 0 and worker_capacity > 0:
            estimated_processing_ms_per_account = int((unfinished_count / worker_capacity) * average_duration_ms)
        return {
            "queued_count": queued_count,
            "running_count": running_count,
            "unfinished_count": unfinished_count,
            "estimated_processing_ms_per_account": estimated_processing_ms_per_account,
            **runtime,
        }

    def _update_task(self, key: str, **updates: Any) -> None:
        with self._lock:
            task = self._tasks.get(key)
            if task is None:
                return
            task.update(updates)
            task["updated_at"] = _now_iso()
            self._save_locked()

    def _complete_task(self, key: str, **updates: Any) -> float:
        _, _, wait_secs = self._task_next_interval_delay()
        with self._lock:
            task = self._tasks.get(key)
            if task is not None:
                task.update(updates)
                task["updated_at"] = _now_iso()
                if wait_secs > 0:
                    self._task_start_cooldowns.append(time.monotonic() + wait_secs)
                self._save_locked()
        return wait_secs

    def _load_locked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw_items = raw.get("tasks") if isinstance(raw, dict) else raw
        if not isinstance(raw_items, list):
            return {}
        tasks: dict[str, dict[str, Any]] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            task_id = _clean(item.get("id"))
            owner = _clean(item.get("owner_id"))
            if not task_id or not owner:
                continue
            status = _clean(item.get("status"))
            if status not in {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING, TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}:
                status = TASK_STATUS_ERROR
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": status,
                "mode": "edit" if item.get("mode") == "edit" else "generate",
                "model": _clean(item.get("model"), "gpt-image-2"),
                "size": _clean(item.get("size")),
                "created_at": _clean(item.get("created_at"), _now_iso()),
                "updated_at": _clean(item.get("updated_at"), _clean(item.get("created_at"), _now_iso())),
            }
            started_at = _clean(item.get("started_at"))
            if started_at:
                task["started_at"] = started_at
            finished_at = _clean(item.get("finished_at"))
            if finished_at:
                task["finished_at"] = finished_at
            if item.get("duration_ms") is not None:
                try:
                    task["duration_ms"] = max(0, int(item.get("duration_ms") or 0))
                except (TypeError, ValueError):
                    pass
            data = item.get("data")
            if isinstance(data, list):
                task["data"] = data
            error = _clean(item.get("error"))
            if error:
                task["error"] = error
            tasks[_task_key(owner, task_id)] = task
        return tasks

    def _save_locked(self) -> None:
        items = sorted(self._tasks.values(), key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps({"tasks": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.path)

    def _recover_unfinished_locked(self) -> bool:
        changed = False
        for task in self._tasks.values():
            if task.get("status") in UNFINISHED_STATUSES:
                task["status"] = TASK_STATUS_ERROR
                task["error"] = "服务已重启，未完成的图片任务已中断"
                task["updated_at"] = _now_iso()
                changed = True
        return changed

    def _cleanup_locked(self) -> bool:
        try:
            retention_days = max(1, int(self.retention_days_getter()))
        except Exception:
            retention_days = 30
        cutoff = time.time() - retention_days * 86400
        removed_keys = [
            key
            for key, task in self._tasks.items()
            if task.get("status") in TERMINAL_STATUSES and _timestamp(task.get("updated_at")) < cutoff
        ]
        for key in removed_keys:
            self._tasks.pop(key, None)
        return bool(removed_keys)


image_task_service = ImageTaskService(DATA_DIR / "image_tasks.json")
