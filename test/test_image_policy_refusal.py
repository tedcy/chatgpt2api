from __future__ import annotations

import json
import unittest

from services.openai_backend_api import OpenAIBackendAPI
from services.protocol.conversation import iter_conversation_payloads


class ImagePolicyRefusalTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
