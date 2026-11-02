"""Offline tests for the Groq / OpenAI-compatible client (no key, no network).

Run:  python -m unittest test_groq_client -v
"""

import io
import json
import logging
import os
import shutil
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from unittest import mock

import llm_client as lc
import main as cli
from constants import LINE_NEW
from pipeline import Checkpoint, run_pipeline

from test_support import Isolation

_ISO = Isolation()


def setUpModule():
    _ISO.start()


def tearDownModule():
    _ISO.stop()

logging.getLogger("llm_client").addHandler(logging.NullHandler())  # keep test output quiet

_PATHS = ["Input.json", "/mnt/user-data/uploads/Input.json"]
INPUT_PATH = next(p for p in _PATHS if os.path.exists(p))
with open(INPUT_PATH, encoding="utf-8") as _fh:
    DATA = json.load(_fh)

KEY = "gsk_TEST_SECRET_KEY"


class FakeTime:
    """Deterministic clock: sleeping just moves time forward."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def ok_response(text='{"a": 1}', prompt=100, completion=20, total=None):
    body = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": total if total is not None else prompt + completion}}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(body).encode("utf-8")
    return _Resp()


def http_error(code, body="", headers=None):
    return urllib.error.HTTPError("http://x", code, "err", headers or {}, io.BytesIO(body.encode("utf-8")))


def make(**kw):
    ft = kw.pop("ft", None) or FakeTime()
    kw.setdefault("max_retries", 3)
    client = lc.GroqClient(api_key=KEY, sleep=ft.sleep, clock=ft.clock, **kw)
    return client, ft


class ParseRetryDelayTests(unittest.TestCase):
    def test_header(self):
        self.assertEqual(lc.parse_retry_delay("7", ""), 7.0)
        self.assertEqual(lc.parse_retry_delay("2.5", "try again in 1h"), 2.5)  # header wins

    def test_text_formats(self):
        cases = {
            "Please try again in 1h2m3.5s. Visit": 3723.5,
            "Please try again in 7m12.96s.": 432.96,
            "Please try again in 45s": 45.0,
            "Please try again in 850ms.": 0.85,
            "Please try again in 2h": 7200.0,
            "TRY AGAIN IN 3m": 180.0,
        }
        for text, expected in cases.items():
            self.assertAlmostEqual(lc.parse_retry_delay(None, text), expected, places=2, msg=text)

    def test_no_hint(self):
        for body in ("", "rate limited", None, "try again later"):
            self.assertIsNone(lc.parse_retry_delay(None, body))
        self.assertIsNone(lc.parse_retry_delay("Wed, 21 Oct 2026 07:28:00 GMT", ""))


class ConfigTests(unittest.TestCase):
    def test_missing_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(lc.LLMConfigError):
                lc.GroqClient()

    def test_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            c = lc.GroqClient(api_key="k")
        self.assertEqual((c.model, c.timeout, c.max_retries, c.tpm_limit, c.temperature, c.json_mode),
                         ("llama-3.3-70b-versatile", 90.0, 3, 12000, 0.0, True))
        self.assertEqual(c.base_url, "https://api.groq.com/openai/v1")

    def test_env_overrides(self):
        env = {"GROQ_API_KEY": "k", "GROQ_MODEL": "other-model", "GROQ_TIMEOUT": "30", "GROQ_MAX_RETRIES": "1",
               "GROQ_TPM_LIMIT": "6000", "GROQ_TEMPERATURE": "0.3", "GROQ_JSON_MODE": "off"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = lc.GroqClient()
        self.assertEqual((c.model, c.timeout, c.max_retries, c.tpm_limit, c.temperature, c.json_mode),
                         ("other-model", 30.0, 1, 6000, 0.3, False))

    def test_tpm_zero_disables_pacing_and_temperature_none(self):
        with mock.patch.dict(os.environ, {"GROQ_TPM_LIMIT": "0", "GROQ_TEMPERATURE": "none"}, clear=True):
            c = lc.GroqClient(api_key="k")
        self.assertIsNone(c.tpm_limit)
        self.assertIsNone(c.temperature)

    def test_bad_numbers(self):
        for name in ("GROQ_TIMEOUT", "GROQ_MAX_RETRIES", "GROQ_TPM_LIMIT", "GROQ_TEMPERATURE"):
            with mock.patch.dict(os.environ, {name: "abc"}, clear=True):
                with self.assertRaises(lc.LLMConfigError, msg=name):
                    lc.GroqClient(api_key="k")

    def test_factory(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "k"}, clear=True):
            self.assertIsInstance(lc.get_client("groq"), lc.GroqClient)
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(lc.LLMConfigError):
                lc.get_client("groq")
            with self.assertRaises(lc.LLMConfigError):
                lc.get_client("openai-compatible")
        env = {"LLM_BASE_URL": "https://x.test/v1/", "LLM_API_KEY": "k", "LLM_MODEL": "m"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = lc.get_client("openai-compatible")
        self.assertEqual((type(c), c.base_url, c.model), (lc.OpenAICompatibleClient, "https://x.test/v1", "m"))


class RequestTests(unittest.TestCase):
    def test_request_shape(self):
        client, _ = make()
        with mock.patch("urllib.request.urlopen", return_value=ok_response()) as m:
            text = client.complete("SYSTEM", "USER")
        self.assertEqual(text, '{"a": 1}')
        req = m.call_args[0][0]
        self.assertEqual(req.full_url, "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(req.get_header("Authorization"), f"Bearer {KEY}")
        self.assertTrue(req.get_header("User-agent"))
        body = json.loads(req.data.decode())
        self.assertEqual(body["model"], "llama-3.3-70b-versatile")
        self.assertEqual(body["messages"], [{"role": "system", "content": "SYSTEM"},
                                            {"role": "user", "content": "USER"}])
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        for forbidden in ("max_tokens", "max_completion_tokens"):
            self.assertNotIn(forbidden, body)  # no output cap

    def test_key_never_in_url_or_errors(self):
        client, _ = make(max_retries=0)
        with mock.patch("urllib.request.urlopen") as m:
            m.return_value = ok_response()
            client.complete("s", "u")
            self.assertNotIn(KEY, m.call_args[0][0].full_url)
        with mock.patch("urllib.request.urlopen", side_effect=http_error(500, "boom")):
            with self.assertRaises(lc.LLMTransientError) as ctx:
                client.complete("s", "u")
        self.assertNotIn(KEY, str(ctx.exception))

    def test_json_mode_off_and_temperature_none(self):
        client, _ = make(json_mode=False, temperature=None)
        body = client._build_body("s", "u")
        self.assertNotIn("response_format", body)
        self.assertNotIn("temperature", body)

    def test_usage_is_counted(self):
        client, _ = make()
        with mock.patch("urllib.request.urlopen", return_value=ok_response(prompt=6300, completion=650)):
            client.complete("s", "u")
        self.assertEqual((client.usage.calls, client.usage.prompt_tokens, client.usage.output_tokens), (1, 6300, 650))

    def test_fenced_text_is_returned_raw_for_the_parser(self):
        client, _ = make()
        with mock.patch("urllib.request.urlopen", return_value=ok_response('```json\n{"x": 1}\n```')):
            self.assertEqual(lc.extract_json(client.complete("s", "u")), {"x": 1})


class ErrorHandlingTests(unittest.TestCase):
    def test_429_with_short_retry_after_is_waited_and_retried(self):
        client, ft = make()
        side = [http_error(429, "rate limited", {"retry-after": "7"}), ok_response()]
        with mock.patch("urllib.request.urlopen", side_effect=side) as m:
            self.assertEqual(client.complete("s", "u"), '{"a": 1}')
        self.assertEqual(m.call_count, 2)
        self.assertEqual(ft.sleeps, [7.5])

    def test_429_with_try_again_text_in_milliseconds(self):
        client, ft = make()
        body = '{"error":{"message":"Rate limit reached on tokens per minute (TPM). Please try again in 850ms."}}'
        with mock.patch("urllib.request.urlopen", side_effect=[http_error(429, body), ok_response()]):
            client.complete("s", "u")
        self.assertAlmostEqual(ft.sleeps[0], 1.35, places=2)

    def test_429_without_hint_uses_backoff(self):
        client, ft = make()
        with mock.patch("urllib.request.urlopen", side_effect=[http_error(429, "slow down"), ok_response()]):
            client.complete("s", "u")
        self.assertGreaterEqual(ft.sleeps[0], 2.0)

    def test_daily_quota_via_header_stops_immediately(self):
        client, ft = make()
        with mock.patch("urllib.request.urlopen", side_effect=http_error(429, "x", {"retry-after": "7200"})) as m:
            with self.assertRaises(lc.LLMQuotaError) as ctx:
                client.complete("s", "u")
        self.assertEqual(m.call_count, 1)       # no wasted retries
        self.assertEqual(ft.sleeps, [])
        self.assertIn("quota", str(ctx.exception).lower())

    def test_daily_quota_via_text(self):
        client, _ = make()
        body = ('{"error":{"message":"Rate limit reached for model llama-3.3-70b-versatile on tokens per day '
                '(TPD): Limit 100000, Used 99000, Requested 7000. Please try again in 1h2m3.5s. Visit"}}')
        with mock.patch("urllib.request.urlopen", side_effect=http_error(429, body)) as m:
            with self.assertRaises(lc.LLMQuotaError):
                client.complete("s", "u")
        self.assertEqual(m.call_count, 1)

    def test_413_is_permanent_with_hint(self):
        client, _ = make()
        with mock.patch("urllib.request.urlopen", side_effect=http_error(413, "Request too large")) as m:
            with self.assertRaises(lc.LLMPermanentError) as ctx:
                client.complete("s", "u")
        self.assertEqual(m.call_count, 1)
        self.assertIn("tokens-per-minute", str(ctx.exception))

    def test_auth_and_bad_request_are_permanent(self):
        client, _ = make()
        for code in (400, 401, 403, 404):
            with mock.patch("urllib.request.urlopen", side_effect=http_error(code, "nope")) as m:
                with self.assertRaises(lc.LLMPermanentError):
                    client.complete("s", "u")
            self.assertEqual(m.call_count, 1, code)
        with mock.patch("urllib.request.urlopen", side_effect=http_error(401, "bad key")):
            with self.assertRaises(lc.LLMPermanentError) as ctx:
                client.complete("s", "u")
        self.assertIn("API key", str(ctx.exception))

    def test_json_mode_rejection_gets_a_hint(self):
        client, _ = make()
        with mock.patch("urllib.request.urlopen", side_effect=http_error(400, "response_format not supported")):
            with self.assertRaises(lc.LLMPermanentError) as ctx:
                client.complete("s", "u")
        self.assertIn("JSON mode", str(ctx.exception))

    def test_5xx_and_timeouts_retry_then_give_up(self):
        client, ft = make(max_retries=2)
        with mock.patch("urllib.request.urlopen", side_effect=[http_error(503, "busy"), TimeoutError("slow"),
                                                                http_error(500, "err")]) as m:
            with self.assertRaises(lc.LLMTransientError) as ctx:
                client.complete("s", "u")
        self.assertEqual(m.call_count, 3)
        self.assertIn("Gave up after 3 attempts", str(ctx.exception))
        self.assertLess(ft.sleeps[0], ft.sleeps[1])

    def test_empty_or_odd_responses_are_retried(self):
        client, _ = make(max_retries=1)
        class R:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps(self.body).encode()
        for payload in ({"choices": []}, {"choices": [{"message": {"content": "  "}}]}, {}):
            with mock.patch("urllib.request.urlopen", side_effect=[R(payload), R(payload)]) as m:
                with self.assertRaises(lc.LLMTransientError):
                    client.complete("s", "u")
            self.assertEqual(m.call_count, 2)


class PacingTests(unittest.TestCase):
    SYSTEM, USER = "s" * 9000, "u" * 9900   # estimated ~7,000 tokens per call

    def test_second_call_waits_for_the_token_window(self):
        client, ft = make(tpm_limit=12000, max_retries=0)
        with mock.patch("urllib.request.urlopen", return_value=ok_response(total=7000)):
            client.complete(self.SYSTEM, self.USER)
            self.assertEqual(ft.sleeps, [])                 # first call: no wait
            client.complete(self.SYSTEM, self.USER)
        self.assertEqual(len(ft.sleeps), 1)
        self.assertAlmostEqual(ft.sleeps[0], 60.5, places=1)  # waited for the 60s window to pass

    def test_small_calls_are_not_delayed(self):
        client, ft = make(tpm_limit=12000, max_retries=0)
        with mock.patch("urllib.request.urlopen", return_value=ok_response(total=500)):
            for _ in range(5):
                client.complete("s", "u")
        self.assertEqual(ft.sleeps, [])

    def test_no_pacing_when_disabled(self):
        client, ft = make(tpm_limit=None, max_retries=0)
        with mock.patch("urllib.request.urlopen", return_value=ok_response(total=7000)):
            client.complete(self.SYSTEM, self.USER)
            client.complete(self.SYSTEM, self.USER)
        self.assertEqual(ft.sleeps, [])

    def test_window_expires_with_time(self):
        client, ft = make(tpm_limit=12000, max_retries=0)
        with mock.patch("urllib.request.urlopen", return_value=ok_response(total=7000)):
            client.complete(self.SYSTEM, self.USER)
            ft.now += 61                                      # a minute passes on its own
            client.complete(self.SYSTEM, self.USER)
        self.assertEqual(ft.sleeps, [])


class QuotaInPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.cp_path = os.path.join(self.tmp, "cp.jsonl")

    class QuotaAfter(lc.BaseLLMClient):
        """Answers `good` times, then reports an exhausted daily quota."""

        def __init__(self, good):
            super().__init__()
            self.good, self.calls = good, 0

        def complete(self, system, user):
            self.calls += 1
            if self.calls > self.good:
                raise lc.LLMQuotaError("daily quota used up")
            return "{}"

    def test_extract_for_row_flags_quota(self):
        client = lc.MockClient([lc.LLMQuotaError("daily quota")])
        result = lc.extract_for_row(client, DATA[0], LINE_NEW)
        self.assertFalse(result.ok)
        self.assertTrue(result.quota_exhausted)
        self.assertEqual(result.attempts, 1)

    def test_other_failures_do_not_flag_quota(self):
        result = lc.extract_for_row(lc.MockClient([lc.LLMTransientError("x")]), DATA[0], LINE_NEW)
        self.assertFalse(result.quota_exhausted)

    def test_pipeline_stops_calling_after_quota_error_and_resumes_later(self):
        first = self.QuotaAfter(good=2)
        r1 = run_pipeline(DATA[:3], first, checkpoint=Checkpoint(self.cp_path), progress=lambda s: None)
        self.assertEqual(first.calls, 3)                    # 2 good + the one that hit the quota; no more
        self.assertTrue(r1.quota_stopped)
        self.assertEqual(len(r1.rows), 6)                   # every row still produced
        self.assertEqual(len(r1.failed), 4)
        self.assertEqual(len(Checkpoint(self.cp_path).cache), 2)
        self.assertTrue(all(r["PWP/GWP OFFER $ VALUE"] == "$30.00" for r in r1.rows))

        second = self.QuotaAfter(good=99)                   # next day: quota is back
        r2 = run_pipeline(DATA[:3], second, checkpoint=Checkpoint(self.cp_path), progress=lambda s: None)
        self.assertEqual(second.calls, 4)                   # only the 4 pending rows
        self.assertEqual(r2.cached_rows, 2)
        self.assertTrue(r2.complete)
        self.assertFalse(r2.quota_stopped)

    def test_cli_reports_quota_exhaustion(self):
        out_csv = os.path.join(self.tmp, "o.csv")
        args = ["--input", INPUT_PATH, "--output", out_csv, "--checkpoint", self.cp_path,
                "--issues", os.path.join(self.tmp, "i.log"), "--limit", "3"]
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(args, client=self.QuotaAfter(good=2), env_file=None)
        self.assertEqual(code, 2)
        self.assertIn("QUOTA EXHAUSTED", buf.getvalue())
        self.assertIn("4 row(s) are still pending", buf.getvalue())
        self.assertTrue(os.path.exists(out_csv))            # partial but well-formed CSV is still written


if __name__ == "__main__":
    unittest.main()