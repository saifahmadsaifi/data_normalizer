"""Shared test isolation: tests must never depend on, or leak into, the real environment.

Why this exists
  * A developer's real .env / shell variables (GEMINI_FALLBACK_MODEL, GROQ_API_KEY,
    LLM_PROVIDER, ...) must not change test outcomes.
  * No test may ever reach a real API (it would spend real quota). Any attempt to
    call urllib.request.urlopen without a test's own patch raises AssertionError.

Each test module does:
    _ISO = Isolation()
    def setUpModule(): _ISO.start()
    def tearDownModule(): _ISO.stop()
"""

import os
from unittest import mock

ENV_PREFIXES = ("GEMINI_", "GROQ_", "LLM_", "PROMPT_")


def _blocked_urlopen(*args, **kwargs):
    raise AssertionError("A test tried to make a REAL network call. Patch urllib.request.urlopen in the test.")


class Isolation:
    def start(self):
        self._env = mock.patch.dict(os.environ)       # restores the original environment on stop()
        self._env.start()
        for name in [n for n in os.environ if n.startswith(ENV_PREFIXES)]:
            del os.environ[name]
        self._net = mock.patch("urllib.request.urlopen", side_effect=_blocked_urlopen)
        self._net.start()

    def stop(self):
        self._net.stop()
        self._env.stop()