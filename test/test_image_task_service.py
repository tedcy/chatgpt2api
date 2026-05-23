from __future__ import annotations

import json
import threading
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from services.config import config
from services.image_task_service import ImageTaskService


OWNER = {"id": "owner-1", "name": "Owner", "role": "admin"}
OTHER_OWNER = {"id": "owner-2", "name": "Other", "role": "user"}


def wait_for_task(service: ImageTaskService, identity: dict[str, object], task_id: str, status: str, timeout: float = 2.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        result = service.list_tasks(identity, [task_id])
        last = (result.get("items") or [None])[0]
        if last and last.get("status") == status:
            return last
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} did not reach {status}, last={last}")


class ImageTaskServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.log_patcher = patch("services.image_task_service.log_service.add")
        self.log_patcher.start()
        self.addCleanup(self.log_patcher.stop)

    def make_service(self, path: Path, handler=None) -> ImageTaskService:
        return ImageTaskService(
            path,
            generation_handler=handler or (lambda _payload: {"data": [{"url": "http://example.test/image.png"}]}),
            edit_handler=handler or (lambda _payload: {"data": [{"url": "http://example.test/edit.png"}]}),
            retention_days_getter=lambda: 30,
        )

    def patch_config(self, updates: dict[str, object]) -> None:
        original_config = dict(config.data)
        config.data.update(updates)
        self.addCleanup(lambda: setattr(config, "data", original_config))

    def test_task_next_interval_delay_uses_configured_base_plus_jitter(self):
        self.patch_config({
            "image_task_next_interval_secs": 11,
            "image_poll_jitter_min_secs": 2,
            "image_poll_jitter_max_secs": 4,
        })

        with patch("services.image_task_service.random.uniform", return_value=3.0) as uniform_mock:
            base_secs, jitter_secs, wait_secs = ImageTaskService._task_next_interval_delay()

        uniform_mock.assert_called_once_with(2, 4)
        self.assertEqual(base_secs, 11.0)
        self.assertEqual(jitter_secs, 3.0)
        self.assertEqual(wait_secs, 14.0)

    def test_completed_worker_waits_before_starting_next_queued_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            events: list[tuple[str, str, float]] = []
            lock = threading.Lock()

            def handler(payload):
                with lock:
                    events.append((payload["prompt"], "start", time.monotonic()))
                time.sleep(0.03)
                with lock:
                    events.append((payload["prompt"], "end", time.monotonic()))
                return {"data": [{"url": f"http://example.test/{payload['prompt']}.png"}]}

            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
            service._task_worker_limit = lambda: 1  # type: ignore[method-assign]
            service._task_next_interval_delay = lambda: (0.02, 0.03, 0.05)  # type: ignore[method-assign]

            service.submit_generation(
                OWNER,
                client_task_id="task-1",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )
            time.sleep(0.005)
            service.submit_generation(
                OWNER,
                client_task_id="task-2",
                prompt="dog",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            wait_for_task(service, OWNER, "task-1", "success")
            wait_for_task(service, OWNER, "task-2", "success")

            event_map = {(prompt, event): timestamp for prompt, event, timestamp in events}
            self.assertGreaterEqual(event_map[("dog", "start")] - event_map[("cat", "end")], 0.045)

    def test_duplicate_submit_uses_existing_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            calls = 0

            def handler(_payload):
                nonlocal calls
                calls += 1
                time.sleep(0.05)
                return {"data": [{"url": "http://example.test/image.png"}]}

            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
            first = service.submit_generation(
                OWNER,
                client_task_id="task-1",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )
            second = service.submit_generation(
                OWNER,
                client_task_id="task-1",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            self.assertEqual(first["id"], "task-1")
            self.assertEqual(second["id"], "task-1")
            task = wait_for_task(service, OWNER, "task-1", "success")
            self.assertEqual(task["data"][0]["url"], "http://example.test/image.png")
            self.assertIn("started_at", task)
            self.assertIn("finished_at", task)
            self.assertIsInstance(task.get("duration_ms"), int)
            self.assertEqual(calls, 1)

    def test_list_tasks_includes_queue_runtime_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            def handler(payload):
                time.sleep(0.1)
                return {"data": [{"url": f"http://example.test/{payload['prompt']}.png"}]}

            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
            service._task_worker_limit = lambda: 1  # type: ignore[method-assign]
            service._task_next_interval_delay = lambda: (0.0, 0.0, 0.0)  # type: ignore[method-assign]

            with patch(
                "services.image_task_service.account_service.image_runtime_summary",
                return_value={
                    "account_count": 3,
                    "available_account_count": 2,
                    "image_account_concurrency": 1,
                    "worker_capacity": 2,
                    "recent_sample_count": 2,
                    "recent_average_duration_ms": 10000,
                },
            ):
                service.submit_generation(
                    OWNER,
                    client_task_id="task-1",
                    prompt="cat",
                    model="gpt-image-2",
                    size=None,
                    base_url="http://local.test",
                )
                service.submit_generation(
                    OWNER,
                    client_task_id="task-2",
                    prompt="dog",
                    model="gpt-image-2",
                    size=None,
                    base_url="http://local.test",
                )

                result = service.list_tasks(OWNER, [])
                wait_for_task(service, OWNER, "task-1", "success")
                wait_for_task(service, OWNER, "task-2", "success")

            summary = result["summary"]
            self.assertEqual(summary["queued_count"], 1)
            self.assertEqual(summary["running_count"], 1)
            self.assertEqual(summary["unfinished_count"], 2)
            self.assertEqual(summary["recent_average_duration_ms"], 10000)
            self.assertEqual(summary["estimated_processing_ms_per_account"], 10000)

    def test_stop_processing_cancels_queued_tasks_without_blocking_future_tasks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            calls: list[str] = []
            started = threading.Event()
            release = threading.Event()

            def handler(payload):
                calls.append(payload["prompt"])
                started.set()
                self.assertTrue(release.wait(1.0))
                return {"data": [{"url": f"http://example.test/{payload['prompt']}.png"}]}

            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
            service._task_worker_limit = lambda: 1  # type: ignore[method-assign]
            service._task_next_interval_delay = lambda: (0.0, 0.0, 0.0)  # type: ignore[method-assign]

            service.submit_generation(
                OWNER,
                client_task_id="task-1",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )
            self.assertTrue(started.wait(1.0))
            service.submit_generation(
                OWNER,
                client_task_id="task-2",
                prompt="dog",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            result = service.stop_processing(OWNER)
            self.assertEqual(result["stopped_count"], 1)

            release.set()
            wait_for_task(service, OWNER, "task-1", "success")
            time.sleep(0.05)

            task_2 = service.list_tasks(OWNER, ["task-2"])["items"][0]
            self.assertEqual(task_2["status"], "error")
            self.assertIn("已停止", task_2["error"])
            self.assertEqual(calls, ["cat"])

            started.clear()
            release.clear()
            service.submit_generation(
                OWNER,
                client_task_id="task-3",
                prompt="fox",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )
            self.assertTrue(started.wait(1.0))
            release.set()
            wait_for_task(service, OWNER, "task-3", "success")
            self.assertEqual(calls, ["cat", "fox"])

    def test_queued_tasks_obey_worker_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            active = 0
            max_active = 0
            lock = threading.Lock()

            def handler(payload):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.12)
                with lock:
                    active -= 1
                return {"data": [{"url": f"http://example.test/{payload['prompt']}.png"}]}

            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
            service._task_worker_limit = lambda: 1  # type: ignore[method-assign]
            service._task_next_interval_delay = lambda: (0.0, 0.0, 0.0)  # type: ignore[method-assign]

            service.submit_generation(
                OWNER,
                client_task_id="task-1",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )
            second = service.submit_generation(
                OWNER,
                client_task_id="task-2",
                prompt="dog",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            self.assertEqual(second["status"], "queued")
            wait_for_task(service, OWNER, "task-1", "success")
            task = wait_for_task(service, OWNER, "task-2", "success")
            self.assertEqual(task["data"][0]["url"], "http://example.test/dog.png")
            self.assertEqual(max_active, 1)

    def test_different_owner_cannot_query_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self.make_service(Path(tmp_dir) / "image_tasks.json")
            service.submit_generation(
                OWNER,
                client_task_id="private-task",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            wait_for_task(service, OWNER, "private-task", "success")
            result = service.list_tasks(OTHER_OWNER, ["private-task"])

            self.assertEqual(result["items"], [])
            self.assertEqual(result["missing_ids"], ["private-task"])

    def test_success_task_persists_to_new_service_instance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            service = self.make_service(path)
            service.submit_generation(
                OWNER,
                client_task_id="persisted-task",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )
            wait_for_task(service, OWNER, "persisted-task", "success")

            reloaded = self.make_service(path)
            result = reloaded.list_tasks(OWNER, ["persisted-task"])

            self.assertEqual(result["missing_ids"], [])
            self.assertEqual(result["items"][0]["status"], "success")
            self.assertEqual(result["items"][0]["data"][0]["url"], "http://example.test/image.png")

    def test_startup_marks_unfinished_tasks_as_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "id": "queued-task",
                                "owner_id": "owner-1",
                                "status": "queued",
                                "mode": "generate",
                                "model": "gpt-image-2",
                                "created_at": "2099-01-01 00:00:00",
                                "updated_at": "2099-01-01 00:00:00",
                            },
                            {
                                "id": "running-task",
                                "owner_id": "owner-1",
                                "status": "running",
                                "mode": "generate",
                                "model": "gpt-image-2",
                                "created_at": "2099-01-01 00:00:00",
                                "updated_at": "2099-01-01 00:00:00",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            service = self.make_service(path)
            result = service.list_tasks(OWNER, ["queued-task", "running-task"])

            self.assertEqual([item["status"] for item in result["items"]], ["error", "error"])
            self.assertTrue(all("已中断" in item.get("error", "") for item in result["items"]))


if __name__ == "__main__":
    unittest.main()
