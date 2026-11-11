from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from workflow import llm


class _Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class LlmLogTests(unittest.TestCase):
    def test_semantic_factory_declares_vision_and_forces_xhigh_only_for_images(self):
        from workflow.loop import make_ask

        ask = make_ask("gpt-6.1-sol", "https://api.openai.com/v1", "k")
        self.assertTrue(ask.supports_images)
        with mock.patch.object(llm, "ask", return_value=("{}", {})) as transport:
            ask("review", "room", images=["png-data-url"])
            self.assertEqual(transport.call_args.kwargs["reasoning_effort"], "xhigh")
            self.assertEqual(transport.call_args.kwargs["images"], ["png-data-url"])
            ask("review", "room")
            self.assertNotIn("reasoning_effort", transport.call_args.kwargs)

    def test_every_call_is_logged_without_the_key(self):
        answer = {"status": "completed", "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                  "output": [{"type": "message", "content": [{"type": "output_text", "text": "{\"ok\": 1}"}]}]}
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.dict(os.environ, {"WORKFLOW_LLM_LOG_DIR": folder, "WORKFLOW_LLM_TAG": "room_007"}), \
                mock.patch.object(llm.urllib.request, "urlopen", lambda *a, **k: _Answer(json.dumps(answer).encode())):
            text, call = llm.ask("do X", "the room", model="gpt-5.6-sol", base_url="https://api.openai.com/v1",
                                 api_key="sk-secret-test-key")
            raw = open(os.path.join(folder, "room_007.jsonl"), encoding="utf-8").read()
        self.assertEqual(text, "{\"ok\": 1}")
        record = json.loads(raw)
        self.assertEqual((record["tag"], record["instructions"], record["prompt"], record["response"], record["ok"]),
                         ("room_007", "do X", "the room", "{\"ok\": 1}", True))
        self.assertEqual(record["usage"]["prompt_tokens"], 10)
        self.assertNotIn("sk-secret-test-key", raw)
        self.assertNotIn("Authorization", raw)

    def test_nothing_is_written_without_the_setting(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WORKFLOW_LLM_LOG_DIR", None)
            llm._log({"prompt": "x"})
            self.assertEqual(os.listdir(folder), [])


class RelayAndRateTests(unittest.TestCase):
    def test_relay_settings_are_read_in_place_and_hosts_checked(self):
        from workflow import credentials
        with tempfile.TemporaryDirectory() as folder:
            env = os.path.join(folder, "relay.env")
            with open(env, "w") as f:
                f.write("OPENAI_API_KEY=sk-test-relay\nOPENAI_MODEL=gpt-5.6-sol\nOPENAI_BASE_URL=https://old.example/codex/v1\n")
            with mock.patch.object(credentials, "RELAY_ENV_FILE", credentials.Path(env)):
                s = credentials.api_settings("relay")
                self.assertEqual((s["base_url"], s["api_key"], s["model"]),
                                 ("https://api.kbsfdxc.net/codex/v1", "sk-test-relay", "gpt-5.6-sol"))
                self.assertEqual(credentials.api_settings("relay", "https://api.kbsfdxc.com/codex/v1/")["base_url"],
                                 "https://api.kbsfdxc.com/codex/v1")
                with self.assertRaises(credentials.FatalAPIError):
                    credentials.api_settings("relay", "https://evil.example.com/v1")

    def test_the_rate_limit_waits_when_the_minute_is_full(self):
        waits = []
        with tempfile.TemporaryDirectory() as folder:
            state = os.path.join(folder, "rpm.json")
            with mock.patch.dict(os.environ, {"WORKFLOW_RPM": "2", "WORKFLOW_RPM_FILE": state}):
                llm._throttle()
                llm._throttle()
                self.assertEqual(len(json.load(open(state))), 2)

                def fake_sleep(seconds):  # the third request waits; pretend the minute passed
                    waits.append(seconds)
                    with open(state, "w") as f:
                        f.write("[]")
                with mock.patch.object(llm.time, "sleep", fake_sleep):
                    llm._throttle()
        self.assertEqual(len(waits), 1)
        self.assertGreater(waits[0], 50)

    def test_the_relay_gets_chat_completions_with_a_system_message(self):
        responses = {"status": "completed", "usage": {},
                     "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]}
        chat = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 2}}
        sent = []

        def fake_urlopen(request, timeout=None):
            sent.append((request.full_url, json.loads(request.data)))
            return _Answer(json.dumps(chat if request.full_url.endswith("/chat/completions") else responses).encode())
        with mock.patch.object(llm.urllib.request, "urlopen", fake_urlopen), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WORKFLOW_RPM", None)
            llm.ask("do X", "room", model="gpt-5.6-sol", base_url="https://api.kbsfdxc.net/codex/v1", api_key="k")
            llm.ask("do X", "room", model="gpt-5.6-sol", base_url="https://api.openai.com/v1", api_key="k")
        (relay_url, relay), (official_url, official) = sent
        self.assertTrue(relay_url.endswith("/codex/v1/chat/completions"))
        self.assertEqual(relay["messages"], [{"role": "system", "content": "do X"}, {"role": "user", "content": "room"}])
        self.assertTrue(official_url.endswith("/v1/responses"))
        self.assertEqual(official["instructions"], "do X")
        self.assertEqual(official["input"], [{"role": "user", "content": "room"}])


class VisionTests(unittest.TestCase):
    PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aL1sAAAAASUVORK5CYII=")
    IMAGE = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")

    def test_both_api_shapes_send_images_and_log_only_fingerprints(self):
        sent = []
        responses = {"status": "completed", "usage": {},
                     "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]}
        chat = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {}}

        def fake_urlopen(request, timeout=None):
            sent.append(json.loads(request.data))
            answer = chat if request.full_url.endswith("/chat/completions") else responses
            return _Answer(json.dumps(answer).encode())

        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.dict(os.environ, {"WORKFLOW_LLM_LOG_DIR": folder, "WORKFLOW_LLM_TAG": "vision"}), \
                mock.patch.object(llm.urllib.request, "urlopen", fake_urlopen), \
                mock.patch.object(llm, "_throttle"):
            for base_url in ("https://api.kbsfdxc.net/codex/v1", "https://api.openai.com/v1"):
                _, call = llm.ask("review", "room", model="gpt-6.1-sol", base_url=base_url, api_key="test-key",
                                  reasoning_effort="xhigh", images=[self.IMAGE])
                self.assertEqual(call["image_count"], 1)
                self.assertEqual(call["image_sha256"], [hashlib.sha256(self.PNG).hexdigest()])
            raw = (Path(folder) / "vision.jsonl").read_text(encoding="utf-8")
        self.assertEqual(sent[0]["messages"][1]["content"], [
            {"type": "text", "text": "room"}, {"type": "image_url", "image_url": {"url": self.IMAGE}}])
        self.assertEqual(sent[1]["input"][0]["content"], [
            {"type": "input_text", "text": "room"}, {"type": "input_image", "image_url": self.IMAGE}])
        self.assertEqual(sent[0]["reasoning_effort"], "xhigh")
        self.assertEqual(sent[1]["reasoning"], {"effort": "xhigh"})
        self.assertNotIn(self.IMAGE, raw)
        self.assertNotIn(self.IMAGE.split(",", 1)[1], raw)
        self.assertEqual([json.loads(line)["image_count"] for line in raw.splitlines()], [1, 1])

    def test_invalid_images_fail_before_network(self):
        invalid = ("https://example.org/image.png", ["https://example.org/image.png"], [None],
                   ["data:image/jpeg;base64,YQ=="], ["data:image/png;base64,%%%"], ["data:image/png;base64,"],
                   ["data:image/png;base64,YQ=="])
        with mock.patch.object(llm.urllib.request, "urlopen") as request:
            for images in invalid:
                with self.subTest(images=images), self.assertRaises(ValueError):
                    llm.ask("review", "room", model="gpt-6.1-sol", base_url="https://api.openai.com/v1",
                            api_key="test-key", images=images)
        request.assert_not_called()

    def test_failed_vision_call_keeps_image_fingerprints(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.dict(os.environ, {"WORKFLOW_LLM_LOG_DIR": folder, "WORKFLOW_LLM_TAG": "failed"}), \
                mock.patch.object(llm.urllib.request, "urlopen", side_effect=llm.urllib.error.HTTPError(
                    "https://api.openai.com/v1/responses", 403, "forbidden", {}, io.BytesIO(b"denied"))), \
                mock.patch.object(llm, "_throttle"):
            with self.assertRaises(llm.FatalAPIError):
                llm.ask("review", "room", model="gpt-6.1-sol", base_url="https://api.openai.com/v1",
                        api_key="test-key", images=[self.IMAGE])
            raw = (Path(folder) / "failed.jsonl").read_text(encoding="utf-8")
        record = json.loads(raw)
        self.assertFalse(record["ok"])
        self.assertEqual(record["image_count"], 1)
        self.assertEqual(record["image_sha256"], [hashlib.sha256(self.PNG).hexdigest()])
        self.assertNotIn(self.IMAGE.split(",", 1)[1], raw)
