"""Offline tests for llm_client.py. No API key and no network needed.

Run:  python -m unittest test_llm_client -v
"""

import io
import json
import os
import unittest
import urllib.error
from unittest import mock

import logging

import llm_client as lc
from constants import LINE_NEW

from test_support import Isolation

_ISO = Isolation()


def setUpModule():
    _ISO.start()


def tearDownModule():
    _ISO.stop()

logging.getLogger("llm_client").addHandler(logging.NullHandler())  # keep test output quiet

PRODUCT = {
    "product_name": "Apple iPhone 17 Pro Max",
    "brand": "Apple",
    "variant": "1 TB",
    "offers": [{"title": "t", "description": "d", "full_details": {"modal_raw_text": "x"}}],
}


def fake_response(body: dict):
    """A context-manager object that mimics urllib's response."""
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(body).encode("utf-8")
    return _Resp()


def gemini_ok(text='{"a": 1}'):
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 5},
    }


def http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(b'{"error":"boom"}'))


class ExtractJsonTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(lc.extract_json('{"a": 1}'), {"a": 1})

    def test_code_fence(self):
        self.assertEqual(lc.extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(lc.extract_json('```\n{"a": 1}\n```'), {"a": 1})

    def test_stray_text_around(self):
        self.assertEqual(lc.extract_json('Sure! Here it is:\n{"a": {"b": 2}}\nHope that helps.'), {"a": {"b": 2}})

    def test_braces_inside_strings(self):
        text = 'note {"q": "uses } and { inside", "n": 1} trailing'
        self.assertEqual(lc.extract_json(text), {"q": "uses } and { inside", "n": 1})

    def test_escaped_quote_inside_string(self):
        self.assertEqual(lc.extract_json('{"q": "say \\"hi\\" }"}'), {"q": 'say "hi" }'})

    def test_failures(self):
        for bad in ["", "   ", None, "no json here", '{"a": 1', "[1, 2, 3]", 123]:
            with self.assertRaises(lc.LLMParseError, msg=repr(bad)):
                lc.extract_json(bad)


class GeminiClientTests(unittest.TestCase):
    def make(self, **kw):
        self.sleeps = []
        return lc.GeminiClient(api_key="TEST_KEY", model="m-test", sleep=self.sleeps.append, **kw)

    def test_missing_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(lc.LLMConfigError):
                lc.GeminiClient()

    def test_success_request_shape_and_usage(self):
        client = self.make()
        with mock.patch("urllib.request.urlopen", return_value=fake_response(gemini_ok())) as m:
            text = client.complete("SYS", "USER")
        self.assertEqual(text, '{"a": 1}')
        req = m.call_args[0][0]
        self.assertIn("models/m-test:generateContent", req.full_url)
        self.assertNotIn("TEST_KEY", req.full_url)  # key travels in a header, never the URL
        self.assertEqual(req.headers["X-goog-api-key"], "TEST_KEY")
        body = json.loads(req.data.decode())
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], "SYS")
        self.assertEqual(body["contents"][0]["parts"][0]["text"], "USER")
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
        self.assertNotIn("maxOutputTokens", body["generationConfig"])  # no output cap set
        self.assertEqual((client.usage.calls, client.usage.prompt_tokens,
                          client.usage.output_tokens, client.usage.thinking_tokens), (1, 100, 20, 5))

    def test_retries_on_429_then_succeeds(self):
        client = self.make()
        side = [http_error(429), http_error(503), fake_response(gemini_ok())]
        with mock.patch("urllib.request.urlopen", side_effect=side) as m:
            self.assertEqual(client.complete("s", "u"), '{"a": 1}')
        self.assertEqual(m.call_count, 3)
        self.assertEqual(len(self.sleeps), 2)
        self.assertLess(self.sleeps[0], self.sleeps[1])  # exponential backoff

    def test_no_retry_on_permanent_error(self):
        client = self.make()
        for code in (400, 401, 403, 404):
            with mock.patch("urllib.request.urlopen", side_effect=http_error(code)) as m:
                with self.assertRaises(lc.LLMPermanentError):
                    client.complete("s", "u")
            self.assertEqual(m.call_count, 1, f"HTTP {code} must not be retried")

    def test_timeout_gives_up_after_max_retries(self):
        client = self.make(max_retries=2)
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("slow")) as m:
            with self.assertRaises(lc.LLMTransientError) as ctx:
                client.complete("s", "u")
        self.assertEqual(m.call_count, 3)
        self.assertIn("Gave up", str(ctx.exception))

    def test_blocked_prompt_is_permanent(self):
        client = self.make()
        with mock.patch("urllib.request.urlopen",
                        return_value=fake_response({"promptFeedback": {"blockReason": "SAFETY"}})) as m:
            with self.assertRaises(lc.LLMPermanentError):
                client.complete("s", "u")
        self.assertEqual(m.call_count, 1)

    def test_empty_candidates_is_retried(self):
        client = self.make(max_retries=1)
        with mock.patch("urllib.request.urlopen", return_value=fake_response({"candidates": []})) as m:
            with self.assertRaises(lc.LLMTransientError):
                client.complete("s", "u")
        self.assertEqual(m.call_count, 2)

    def test_thought_parts_are_ignored(self):
        payload = {"candidates": [{"content": {"parts": [
            {"text": "internal reasoning", "thought": True}, {"text": '{"ok": true}'}]}}]}
        client = self.make()
        with mock.patch("urllib.request.urlopen", return_value=fake_response(payload)):
            self.assertEqual(client.complete("s", "u"), '{"ok": true}')

    def test_error_text_never_contains_key(self):
        client = self.make(max_retries=0)
        with mock.patch("urllib.request.urlopen", side_effect=http_error(500)):
            with self.assertRaises(lc.LLMTransientError) as ctx:
                client.complete("s", "u")
        self.assertNotIn("TEST_KEY", str(ctx.exception))


class ExtractForRowTests(unittest.TestCase):
    def test_success(self):
        client = lc.MockClient([{"switcher": {"amount": "$800"}}])
        result = lc.extract_for_row(client, PRODUCT, LINE_NEW)
        self.assertTrue(result.ok)
        self.assertEqual(result.data["switcher"]["amount"], "$800")
        self.assertEqual(result.attempts, 1)
        system, user = client.requests[0]
        self.assertIn("1 TB", user)
        self.assertIn('"simulated_line_type": "New Line"', user)

    def test_fenced_output_is_accepted(self):
        client = lc.MockClient(['```json\n{"x": 1}\n```'])
        self.assertEqual(lc.extract_for_row(client, PRODUCT, LINE_NEW).data, {"x": 1})

    def test_malformed_then_good_is_retried(self):
        client = lc.MockClient(["not json at all", '{"x": 2}'])
        result = lc.extract_for_row(client, PRODUCT, LINE_NEW)
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)

    def test_always_malformed_returns_error_not_exception(self):
        client = lc.MockClient(["bad", "worse", "[1]"])
        result = lc.extract_for_row(client, PRODUCT, LINE_NEW)
        self.assertFalse(result.ok)
        self.assertIn("Malformed JSON", result.error)
        self.assertEqual(result.attempts, 3)

    def test_transport_error_returns_error_not_exception(self):
        client = lc.MockClient([lc.LLMTransientError("Gave up after 5 attempts")])
        result = lc.extract_for_row(client, PRODUCT, LINE_NEW)
        self.assertFalse(result.ok)
        self.assertIn("Gave up", result.error)
        self.assertEqual(result.attempts, 1)  # not re-asked: client already retried

    def test_bad_line_type_returns_error(self):
        result = lc.extract_for_row(lc.MockClient([{}]), PRODUCT, "Bogus")
        self.assertFalse(result.ok)
        self.assertIn("Prompt build failed", result.error)

    def test_weird_product_does_not_crash(self):
        client = lc.MockClient([{}])
        self.assertIsInstance(lc.extract_for_row(client, {}, LINE_NEW), lc.ExtractionResult)


class FactoryTests(unittest.TestCase):
    def test_mock_and_unknown(self):
        self.assertIsInstance(lc.get_client("mock"), lc.MockClient)
        with self.assertRaises(lc.LLMConfigError):
            lc.get_client("nope")

    def test_default_is_gemini_and_needs_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(lc.LLMConfigError):
                lc.get_client()


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            client = lc.GeminiClient(api_key="k")
        self.assertEqual((client.timeout, client.max_retries), (90.0, 3))

    def test_env_overrides(self):
        env = {"GEMINI_TIMEOUT": "45", "GEMINI_MAX_RETRIES": "1"}
        with mock.patch.dict(os.environ, env, clear=True):
            client = lc.GeminiClient(api_key="k")
        self.assertEqual((client.timeout, client.max_retries), (45.0, 1))

    def test_explicit_args_beat_env(self):
        with mock.patch.dict(os.environ, {"GEMINI_TIMEOUT": "45"}, clear=True):
            client = lc.GeminiClient(api_key="k", timeout=10, max_retries=0)
        self.assertEqual((client.timeout, client.max_retries), (10, 0))

    def test_bad_number_is_a_config_error(self):
        with mock.patch.dict(os.environ, {"GEMINI_TIMEOUT": "fast"}, clear=True):
            with self.assertRaises(lc.LLMConfigError):
                lc.GeminiClient(api_key="k")

    def test_retry_is_logged(self):
        client = lc.GeminiClient(api_key="k", model="m", sleep=lambda s: None, max_retries=1)
        with mock.patch("urllib.request.urlopen", side_effect=[TimeoutError("slow"), fake_response(gemini_ok())]):
            with self.assertLogs("llm_client", level="WARNING") as logs:
                client.complete("s", "u")
        self.assertIn("Retrying", logs.output[0])


class FallbackAndTemperatureTests(unittest.TestCase):
    def make(self, **kw):
        return lc.GeminiClient(api_key="k", model="main-m", sleep=lambda s: None, max_retries=1, **kw)

    def urls(self, mocked):
        return [call.args[0].full_url for call in mocked.call_args_list]

    def test_fallback_used_when_main_model_stays_unavailable(self):
        client = self.make(fallback_model="backup-m")
        side = [http_error(503), http_error(503), fake_response(gemini_ok('{"from": "backup"}'))]
        with mock.patch("urllib.request.urlopen", side_effect=side) as m:
            with self.assertLogs("llm_client", level="WARNING") as logs:
                text = client.complete("s", "u")
        self.assertEqual(text, '{"from": "backup"}')
        urls = self.urls(m)
        self.assertIn("models/main-m:", urls[0]); self.assertIn("models/main-m:", urls[1])
        self.assertIn("models/backup-m:", urls[2])
        self.assertEqual(client.last_model_used, "backup-m")
        self.assertTrue(any("switching to fallback" in line for line in logs.output))

    def test_main_model_preferred_when_healthy(self):
        client = self.make(fallback_model="backup-m")
        with mock.patch("urllib.request.urlopen", return_value=fake_response(gemini_ok())) as m:
            client.complete("s", "u")
        self.assertEqual(m.call_count, 1)
        self.assertEqual(client.last_model_used, "main-m")

    def test_permanent_error_never_triggers_fallback(self):
        client = self.make(fallback_model="backup-m")
        with mock.patch("urllib.request.urlopen", side_effect=http_error(404)) as m:
            with self.assertRaises(lc.LLMPermanentError):
                client.complete("s", "u")
        self.assertEqual(m.call_count, 1)

    def test_both_models_down_reports_both(self):
        client = self.make(fallback_model="backup-m")
        with mock.patch("urllib.request.urlopen", side_effect=http_error(503)) as m:
            with self.assertRaises(lc.LLMTransientError) as ctx:
                client.complete("s", "u")
        self.assertEqual(m.call_count, 4)  # 2 attempts x 2 models
        self.assertIn("main-m", str(ctx.exception)); self.assertIn("backup-m", str(ctx.exception))

    def test_fallback_same_as_main_is_ignored(self):
        self.assertIsNone(self.make(fallback_model="main-m").fallback_model)

    def test_fallback_from_env(self):
        with mock.patch.dict(os.environ, {"GEMINI_FALLBACK_MODEL": "env-backup"}, clear=True):
            self.assertEqual(self.make().fallback_model, "env-backup")

    def test_temperature_default_zero_and_sent(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            body = self.make()._build_body("s", "u")
        self.assertEqual(body["generationConfig"]["temperature"], 0.0)

    def test_temperature_none_is_omitted(self):
        for raw in ("none", "NONE", "off", "omit"):
            with mock.patch.dict(os.environ, {"GEMINI_TEMPERATURE": raw}, clear=True):
                body = self.make()._build_body("s", "u")
            self.assertNotIn("temperature", body["generationConfig"], raw)
            self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")

    def test_temperature_bad_value(self):
        with mock.patch.dict(os.environ, {"GEMINI_TEMPERATURE": "hot"}, clear=True):
            with self.assertRaises(lc.LLMConfigError):
                self.make()

    def test_temperature_400_gives_hint(self):
        err = urllib.error.HTTPError("http://x", 400, "bad", {}, io.BytesIO(b'{"error":"temperature is not supported"}'))
        client = self.make()
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(lc.LLMPermanentError) as ctx:
                client.complete("s", "u")
        self.assertIn("GEMINI_TEMPERATURE=none", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
    