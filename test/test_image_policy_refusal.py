from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from services.config import config
from services.openai_backend_api import OpenAIBackendAPI
from services.protocol.conversation import iter_conversation_payloads


def _conversation_with_image(file_id: str = "file-123") -> dict:
    return {
        "mapping": {
            "tool-message": {
                "message": {
                    "author": {"role": "tool"},
                    "create_time": 1,
                    "metadata": {"async_task_type": "image_gen"},
                    "content": {
                        "content_type": "multimodal_text",
                        "parts": [{"asset_pointer": f"file-service://{file_id}"}],
                    },
                },
            },
        },
    }


class ImagePolicyRefusalTests(unittest.TestCase):
    def patch_config(self, updates: dict[str, object]) -> None:
        original_config = dict(config.data)
        config.data.update(updates)
        self.addCleanup(lambda: setattr(config, "data", original_config))

    def test_extracts_policy_refusal_from_conversation_mapping(self):
        backend = OpenAIBackendAPI(access_token="")
        result = backend._extract_image_poll_result({
            "mapping": {
                "assistant-message": {
                    "message": {
                        "author": {"role": "assistant"},
                        "create_time": 1,
                        "content": {
                            "content_type": "text",
                            "parts": [
                                "非常抱歉，生成的图片可能违反了关于裸露、色情或情色内容的防护限制。",
                            ],
                        },
                    },
                },
            },
        })

        self.assertTrue(result.blocked)
        self.assertIn("非常抱歉", result.message)
        self.assertEqual(result.file_ids, [])
        self.assertEqual(result.sediment_ids, [])

    def test_sse_policy_refusal_text_marks_stream_blocked(self):
        events = list(iter_conversation_payloads(iter([
            json.dumps({
                "type": "unknown",
                "v": {
                    "notice": "非常抱歉，生成的图片可能违反了关于裸露、色情或情色内容的防护限制。",
                },
            }, ensure_ascii=False),
            "[DONE]",
        ])))

        self.assertEqual(events[0]["type"], "conversation.delta")
        self.assertTrue(events[0]["blocked"])
        self.assertIn("非常抱歉", events[0]["text"])

    def test_poll_wait_uses_configured_interval_plus_jitter(self):
        self.patch_config({
            "image_poll_interval_secs": 7,
            "image_poll_jitter_min_secs": 2,
            "image_poll_jitter_max_secs": 4,
        })
        backend = OpenAIBackendAPI(access_token="")
        conversations = iter([{"mapping": {}}, _conversation_with_image()])
        sleeps: list[float] = []
        backend._get_conversation = lambda _conversation_id: next(conversations)  # type: ignore[method-assign]

        with patch("services.openai_backend_api.random.uniform", return_value=3.0) as uniform_mock, \
                patch("services.openai_backend_api.time.sleep", side_effect=sleeps.append):
            result = backend._poll_image_results("conversation-1", timeout_secs=30)

        self.assertEqual(result.file_ids, ["file-123"])
        uniform_mock.assert_called_once_with(2, 4)
        self.assertEqual(sleeps, [10.0])

    def test_poll_429_wait_uses_configured_retry_interval_plus_jitter(self):
        self.patch_config({
            "image_poll_rate_limit_retry_secs": 11,
            "image_poll_jitter_min_secs": 1,
            "image_poll_jitter_max_secs": 5,
        })
        backend = OpenAIBackendAPI(access_token="")
        calls = iter([
            RuntimeError("/backend-api/conversation/conversation-1 failed: status=429"),
            _conversation_with_image(),
        ])
        sleeps: list[float] = []

        def get_conversation(_conversation_id):
            value = next(calls)
            if isinstance(value, Exception):
                raise value
            return value

        backend._get_conversation = get_conversation  # type: ignore[method-assign]

        with patch("services.openai_backend_api.random.uniform", return_value=4.0) as uniform_mock, \
                patch("services.openai_backend_api.time.sleep", side_effect=sleeps.append):
            result = backend._poll_image_results("conversation-1", timeout_secs=30)

        self.assertEqual(result.file_ids, ["file-123"])
        uniform_mock.assert_called_once_with(1, 5)
        self.assertEqual(sleeps, [15.0])


if __name__ == "__main__":
    unittest.main()
