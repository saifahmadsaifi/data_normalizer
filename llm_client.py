"""LLM client (Step 6): talks to the model and returns parsed JSON.

Design goals
  * Provider-agnostic: every provider implements `BaseLLMClient.complete()`.
    Gemini is the default; swapping providers later means adding one small
    class and setting LLM_PROVIDER (nothing else in the pipeline changes).
  * Robust: timeouts, exponential-backoff retries on transient errors, no retry
    on permanent errors (bad key / bad request), and tolerant JSON parsing
    (code fences, stray text around the object).
  * Never crashes the whole run: `extract_for_row` returns an ExtractionResult
    with an error message instead of raising, so one bad row cannot stop the CSV.
  * Zero extra dependencies (standard library only).
  * Counts real token usage reported by the API, so cost can be reported.

Environment variables
  GEMINI_API_KEY      required for the Gemini provider (never printed or logged)
  GEMINI_MODEL        model id; default below. Check Google's models page for the
                      current Flash / Flash-Lite id and set it here if needed.
  GEMINI_TEMPERATURE  optional, default 0 (as deterministic as possible). Set to
                      "none" to omit it: some newer Gemini models reject it.
  GEMINI_FALLBACK_MODEL optional second model id, used automatically when the
                      main model stays unavailable (HTTP 503/429/timeouts).
  GEMINI_TIMEOUT      optional, seconds to wait for one response (default 90)
  GEMINI_MAX_RETRIES  optional, retries after a transient failure (default 3)
  LLM_PROVIDER        "gemini" (default), "groq", "openai-compatible", "dryrun" (no network)
                      or "mock" (tests). Groq settings are documented on GroqClient.
"""

import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from prompt_builder import build_messages

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class LLMError(Exception):
    """Base class for all client errors."""


class LLMConfigError(LLMError):
    """Missing/invalid configuration (e.g. no API key). Not retryable."""


class LLMTransientError(LLMError):
    """Timeouts, rate limits, 5xx. Safe to retry.

    `retry_after` is the server's own "try again in N seconds" hint, when it gave one.
    """

    def __init__(self, message: str = "", retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMQuotaError(LLMError):
    """A usage quota (e.g. tokens per DAY) is used up and will not recover soon.

    Retrying now only wastes requests, so the pipeline stops calling the API,
    keeps its saved progress, and tells the user to rerun after the reset.
    """


class LLMPermanentError(LLMError):
    """Bad request, bad key, blocked content. Retrying will not help."""


class LLMParseError(LLMError):
    """The model answered, but not with usable JSON."""


# --------------------------------------------------------------------------- #
# Tolerant JSON parsing
# --------------------------------------------------------------------------- #
def _strip_code_fences(text: str) -> str:
    """Remove a leading ```json / ``` fence and a trailing ``` fence if present."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1:] if first_newline != -1 else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _first_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} block, honouring quotes and escapes."""
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)  # unbalanced; try the next "{"
    return None


def extract_json(text: Any) -> Dict[str, Any]:
    """Parse a JSON object out of model output, tolerating fences and stray text.

    Raises LLMParseError if no JSON *object* can be recovered.
    """
    if not isinstance(text, str) or not text.strip():
        raise LLMParseError("Empty model response")

    cleaned = _strip_code_fences(text)
    candidates = [cleaned]
    block = _first_json_object(cleaned)
    if block and block != cleaned:
        candidates.append(block)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
        last_error = ValueError(f"Top-level JSON is {type(parsed).__name__}, expected object")
    raise LLMParseError(f"Could not parse JSON object from response: {last_error}")


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #
@dataclass
class Usage:
    """Running token totals as reported by the API."""
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    def summary(self) -> str:
        return (
            f"{self.calls} calls | prompt {self.prompt_tokens} | "
            f"output {self.output_tokens} | thinking {self.thinking_tokens} tokens"
        )


class BaseLLMClient(ABC):
    """Every provider implements `complete`: (system, user) -> raw text."""

    def __init__(self) -> None:
        self.usage = Usage()

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send one request and return the model's raw text. May raise LLMError."""


def _env_number(name: str, cast: Callable[[str], Any], default: Any) -> Any:
    """Read a numeric setting from the environment, falling back to `default`."""
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise LLMConfigError(f"{name} must be a number, got {raw!r}") from exc


def _parse_temperature(raw: Optional[str]) -> Optional[float]:
    """GEMINI_TEMPERATURE: number -> that value; 'none'/'off'/'omit' -> None (omit); unset -> 0."""
    if raw is None or raw.strip() == "":
        return 0.0
    if raw.strip().lower() in ("none", "off", "omit", "null"):
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise LLMConfigError(f"temperature setting must be a number or 'none', got {raw!r}") from exc


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"  # id shown in Google's current docs; override via GEMINI_MODEL
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}


class GeminiClient(BaseLLMClient):
    """Google Gemini via the REST `generateContent` endpoint (standard library only)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        backoff_base: float = 2.0,
        temperature: Optional[float] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__()
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise LLMConfigError(
                "GEMINI_API_KEY is not set. Set it as an environment variable "
                "(see README); never hard-code it in the source."
            )
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.timeout = timeout if timeout is not None else _env_number("GEMINI_TIMEOUT", float, 90.0)
        self.max_retries = max_retries if max_retries is not None else _env_number("GEMINI_MAX_RETRIES", int, 3)
        self.backoff_base = backoff_base
        self.temperature = temperature if temperature is not None else _parse_temperature(
            os.environ.get("GEMINI_TEMPERATURE")
        )
        fallback = fallback_model or os.environ.get("GEMINI_FALLBACK_MODEL") or None
        self.fallback_model = fallback if fallback and fallback != self.model else None
        self.last_model_used: Optional[str] = None
        self._sleep = sleep

    # -- request ---------------------------------------------------------- #
    def _build_body(self, system: str, user: str) -> Dict[str, Any]:
        generation_config: Dict[str, Any] = {"responseMimeType": "application/json"}
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }

    def _post_once(self, body: Dict[str, Any], model: str) -> Dict[str, Any]:
        """One HTTP attempt. Maps failures to Transient / Permanent errors."""
        url = f"{GEMINI_BASE_URL}/{model}:generateContent"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001 - best effort only
                pass
            finally:
                exc.close()
            message = f"HTTP {exc.code} from Gemini ({model}): {detail}"
            if exc.code == 400 and "temperature" in detail.lower():
                message += " | Hint: this model rejects 'temperature'. Set GEMINI_TEMPERATURE=none in .env"
            if exc.code in RETRYABLE_HTTP:
                raise LLMTransientError(message) from exc
            raise LLMPermanentError(message) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise LLMTransientError(f"Network/timeout error: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMTransientError(f"Gemini returned non-JSON HTTP body: {exc}") from exc

    # -- response --------------------------------------------------------- #
    def _read_text(self, payload: Dict[str, Any]) -> str:
        """Pull the text out of a generateContent response and record token usage."""
        meta = payload.get("usageMetadata") or {}
        self.usage.calls += 1
        self.usage.prompt_tokens += int(meta.get("promptTokenCount") or 0)
        self.usage.output_tokens += int(meta.get("candidatesTokenCount") or 0)
        self.usage.thinking_tokens += int(meta.get("thoughtsTokenCount") or 0)

        block_reason = (payload.get("promptFeedback") or {}).get("blockReason")
        if block_reason:
            raise LLMPermanentError(f"Prompt blocked by Gemini: {block_reason}")

        candidates = payload.get("candidates") or []
        if not candidates:
            raise LLMTransientError("Gemini returned no candidates")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
        if not text.strip():
            raise LLMTransientError(
                f"Gemini returned empty text (finishReason={candidate.get('finishReason')})"
            )
        return text

    # -- public ----------------------------------------------------------- #
    def _complete_with_retries(self, body: Dict[str, Any], model: str) -> str:
        """Retry transient failures for ONE model with exponential backoff."""
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            try:
                text = self._read_text(self._post_once(body, model))
                logger.info("Gemini call finished in %.1fs (model %s, attempt %d)",
                            time.monotonic() - started, model, attempt + 1)
                self.last_model_used = model
                return text
            except LLMTransientError as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                delay = self.backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "Gemini attempt %d/%d failed after %.1fs (%s). Retrying in %.1fs...",
                    attempt + 1, self.max_retries + 1, time.monotonic() - started, exc, delay,
                )
                self._sleep(delay)
        raise LLMTransientError(
            f"Gave up after {self.max_retries + 1} attempts on {model}: {last_error}"
        )

    def complete(self, system: str, user: str) -> str:
        """Ask the main model; if it stays unavailable, try the fallback model (if set).

        Only TRANSIENT failures trigger the fallback. Permanent errors (bad key, wrong
        model name, blocked prompt) are raised immediately so they are never hidden.
        """
        body = self._build_body(system, user)
        models = [self.model] + ([self.fallback_model] if self.fallback_model else [])
        errors: List[str] = []
        for position, model in enumerate(models):
            try:
                return self._complete_with_retries(body, model)
            except LLMTransientError as exc:
                errors.append(str(exc))
                if position + 1 < len(models):
                    logger.warning("Model %s is unavailable; switching to fallback model %s",
                                   model, models[position + 1])
        raise LLMTransientError(" || ".join(errors))


def parse_retry_delay(retry_after_header: Optional[str], body: str) -> Optional[float]:
    """Seconds the server asks us to wait: the Retry-After header, else 'try again in 1h2m3.5s'."""
    if retry_after_header:
        try:
            return max(float(retry_after_header), 0.0)
        except ValueError:
            pass
    match = re.search(r"try again in ([0-9hms.]+)", body or "", re.IGNORECASE)
    if match:
        units = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
        parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", match.group(1))
        if parts:
            return sum(float(value) * units[unit] for value, unit in parts)
    return None


class OpenAICompatibleClient(BaseLLMClient):
    """Any provider that speaks the OpenAI "chat/completions" format (Groq, Mistral,
    Cerebras, OpenRouter, ...). Standard library only.

    Rate-limit aware:
      * a rolling 60-second token budget (`tpm_limit`) paces calls BEFORE the server
        has to refuse them;
      * a 429 that says "try again in a few seconds" is waited out and retried;
      * a 429 asking for a long wait (> `quota_wait_threshold`, e.g. a daily token
        limit) raises LLMQuotaError instead of burning more requests.
    """

    provider_name = "openai-compatible"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 90.0,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        temperature: Optional[float] = 0.0,
        json_mode: bool = True,
        tpm_limit: Optional[int] = None,
        quota_wait_threshold: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.temperature = temperature
        self.json_mode = json_mode
        self.tpm_limit = tpm_limit
        self.quota_wait_threshold = quota_wait_threshold
        self._sleep = sleep
        self._clock = clock
        self._window: Deque[Tuple[float, int]] = deque()  # (time, tokens) in the last 60s

    # -- request ---------------------------------------------------------- #
    def _build_body(self, system: str, user: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body  # deliberately no max_tokens cap

    @staticmethod
    def _estimate_tokens(system: str, user: str) -> int:
        """Rough size of a call (~3 characters per token, plus room for the answer)."""
        return (len(system) + len(user)) // 3 + 700

    def _throttle(self, estimated: int) -> None:
        """Wait until the last-60-seconds token budget has room for this call."""
        if not self.tpm_limit:
            return
        while True:
            now = self._clock()
            while self._window and now - self._window[0][0] >= 60:
                self._window.popleft()
            used = sum(tokens for _, tokens in self._window)
            if not self._window or used + estimated <= self.tpm_limit:
                return
            wait = self._window[0][0] + 60 - now + 0.5
            logger.warning("Pacing: %d of %d tokens/minute already used; waiting %.0fs before the next call",
                           used, self.tpm_limit, wait)
            self._sleep(max(wait, 0.5))

    def _post_once(self, body: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "data-normalizer/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001 - best effort only
                pass
            finally:
                exc.close()
            raise self._map_http_error(exc, detail) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise LLMTransientError(f"Network/timeout error: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMTransientError(f"{self.provider_name} returned a non-JSON HTTP body: {exc}") from exc

    def _map_http_error(self, exc: "urllib.error.HTTPError", detail: str) -> LLMError:
        code = exc.code
        message = f"HTTP {code} from {self.provider_name} ({self.model}): {detail}"
        headers = exc.headers if exc.headers is not None else {}
        header = headers.get("Retry-After") or headers.get("retry-after")
        if code == 429:
            wait = parse_retry_delay(header, detail)
            if wait is not None and wait > self.quota_wait_threshold:
                return LLMQuotaError(f"{message} | The server asks for a {wait / 60:.0f}-minute wait: "
                                     "a daily quota is used up. Rerun later; saved progress is kept.")
            return LLMTransientError(message, retry_after=wait)
        if code in RETRYABLE_HTTP:
            return LLMTransientError(message)
        if code == 413:
            return LLMPermanentError(message + " | Hint: one request is bigger than your plan's "
                                     "tokens-per-minute limit. Check your limits page or shorten the prompt.")
        if code == 400 and "response_format" in detail.lower():
            message += " | Hint: this model rejects JSON mode. Turn JSON mode off in .env."
        if code in (401, 403):
            message += " | Hint: check the API key."
        return LLMPermanentError(message)

    # -- response --------------------------------------------------------- #
    def _read_text(self, payload: Dict[str, Any], estimated: int) -> Tuple[str, int]:
        usage = payload.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        self.usage.calls += 1
        self.usage.prompt_tokens += prompt
        self.usage.output_tokens += completion
        tokens = int(usage.get("total_tokens") or (prompt + completion) or estimated)

        choices = payload.get("choices") or []
        if not choices:
            raise LLMTransientError(f"{self.provider_name} returned no choices")
        content = (choices[0].get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMTransientError(
                f"{self.provider_name} returned empty text (finish_reason={choices[0].get('finish_reason')})"
            )
        return content, tokens

    # -- public ----------------------------------------------------------- #
    def complete(self, system: str, user: str) -> str:
        body = self._build_body(system, user)
        estimated = self._estimate_tokens(system, user)
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            self._throttle(estimated)
            started = self._clock()
            try:
                text, tokens = self._read_text(self._post_once(body), estimated)
                self._window.append((started, tokens))
                logger.info("%s call finished in %.1fs (model %s, attempt %d, %d tokens)",
                            self.provider_name, self._clock() - started, self.model, attempt + 1, tokens)
                return text
            except LLMTransientError as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                if exc.retry_after is not None:  # honour the server's own hint
                    delay = exc.retry_after + 0.5
                else:
                    delay = self.backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning("%s attempt %d/%d failed after %.1fs (%s). Retrying in %.1fs...",
                               self.provider_name, attempt + 1, self.max_retries + 1,
                               self._clock() - started, exc, delay)
                self._sleep(delay)
        raise LLMTransientError(
            f"Gave up after {self.max_retries + 1} attempts on {self.model}: {last_error}"
        )


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


class GroqClient(OpenAICompatibleClient):
    """Groq (OpenAI-compatible endpoint). Settings come from the environment:

      GROQ_API_KEY      required
      GROQ_MODEL        default llama-3.3-70b-versatile
      GROQ_TEMPERATURE  default 0 ("none" omits it)
      GROQ_TIMEOUT      seconds per request, default 90
      GROQ_MAX_RETRIES  default 3
      GROQ_TPM_LIMIT    your plan's tokens/minute (default 12000; 0 turns pacing off).
                        See console.groq.com/settings/limits for your real value.
      GROQ_JSON_MODE    "on" (default) or "off"
    """

    provider_name = "groq"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, **overrides: Any) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMConfigError(
                "GROQ_API_KEY is not set. Put it in .env (see .env.example); never hard-code it."
            )

        def pick(name: str, env: str, cast: Callable[[str], Any], default: Any) -> Any:
            return overrides[name] if name in overrides else _env_number(env, cast, default)

        tpm = pick("tpm_limit", "GROQ_TPM_LIMIT", int, 12000)
        json_raw = os.environ.get("GROQ_JSON_MODE", "on").strip().lower()
        temperature = overrides["temperature"] if "temperature" in overrides else _parse_temperature(
            os.environ.get("GROQ_TEMPERATURE")
        )
        super().__init__(
            api_key=key,
            model=model or os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL,
            base_url=GROQ_BASE_URL,
            timeout=pick("timeout", "GROQ_TIMEOUT", float, 90.0),
            max_retries=pick("max_retries", "GROQ_MAX_RETRIES", int, 3),
            backoff_base=overrides.get("backoff_base", 2.0),
            temperature=temperature,
            json_mode=(overrides["json_mode"] if "json_mode" in overrides
                       else json_raw not in ("off", "false", "0", "no")),
            tpm_limit=tpm or None,
            quota_wait_threshold=overrides.get("quota_wait_threshold", 120.0),
            sleep=overrides.get("sleep", time.sleep),
            clock=overrides.get("clock", time.monotonic),
        )


class MockClient(BaseLLMClient):
    """Offline stand-in for tests and dry runs. Returns scripted responses in order."""

    def __init__(self, responses: Optional[list] = None) -> None:
        super().__init__()
        self._responses = list(responses or [])
        self.requests: list = []  # (system, user) pairs received

    def complete(self, system: str, user: str) -> str:
        self.requests.append((system, user))
        self.usage.calls += 1
        if not self._responses:
            raise LLMPermanentError("MockClient has no more scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)


class DryRunClient(BaseLLMClient):
    """No-network client for dry runs: always answers with an empty JSON object.

    The pipeline then produces every row (base fields + deterministic PWP/GWP)
    but no LLM-derived offers. Used to test the full pipeline and CSV writer
    without an API key. Its results are never written to the resume checkpoint.
    """

    def complete(self, system: str, user: str) -> str:
        self.usage.calls += 1
        return "{}"


def get_client(provider: Optional[str] = None) -> BaseLLMClient:
    """Factory: choose the provider by argument or the LLM_PROVIDER env var."""
    name = (provider or os.environ.get("LLM_PROVIDER") or "gemini").strip().lower()
    if name == "gemini":
        return GeminiClient()
    if name == "groq":
        return GroqClient()
    if name in ("openai-compatible", "openai_compatible"):
        base, key, model = (os.environ.get(v) for v in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))
        if not (base and key and model):
            raise LLMConfigError("openai-compatible provider needs LLM_BASE_URL, LLM_API_KEY and LLM_MODEL")
        return OpenAICompatibleClient(api_key=key, model=model, base_url=base)
    if name == "mock":
        return MockClient()
    if name in ("dryrun", "dry-run", "dry_run"):
        return DryRunClient()
    raise LLMConfigError(f"Unknown LLM provider {name!r}. Supported: gemini, groq, openai-compatible, dryrun, mock")


# --------------------------------------------------------------------------- #
# Row-level extraction (bridge between prompt builder and the validator)
# --------------------------------------------------------------------------- #
@dataclass
class ExtractionResult:
    """Outcome for one (product, line type) row. `data` is None when it failed."""
    data: Optional[Dict[str, Any]]
    error: Optional[str] = None
    attempts: int = 0
    raw_text: Optional[str] = field(default=None, repr=False)
    quota_exhausted: bool = False  # True -> stop calling the API for now

    @property
    def ok(self) -> bool:
        return self.data is not None


def extract_for_row(
    client: BaseLLMClient,
    product: Dict[str, Any],
    line_type: str,
    max_parse_attempts: int = 3,
) -> ExtractionResult:
    """Ask the LLM about one product + line type and return parsed JSON.

    If the model answers with malformed JSON, the same request is repeated up to
    `max_parse_attempts` times. Transport-level retries happen inside the client.
    Never raises for model/network problems; the error is returned instead.
    """
    try:
        messages = build_messages(product, line_type)
    except Exception as exc:  # noqa: BLE001 - e.g. unknown line type, odd payload
        return ExtractionResult(data=None, error=f"Prompt build failed: {exc}")

    last_error = "unknown error"
    raw: Optional[str] = None
    for attempt in range(1, max_parse_attempts + 1):
        try:
            raw = client.complete(messages["system"], messages["user"])
            return ExtractionResult(
                data=extract_json(raw), attempts=attempt, raw_text=raw
            )
        except LLMParseError as exc:  # model replied, JSON unusable -> ask again
            last_error = f"Malformed JSON: {exc}"
        except LLMQuotaError as exc:  # daily quota gone -> the pipeline should stop calling
            return ExtractionResult(
                data=None, error=str(exc), attempts=attempt, raw_text=raw, quota_exhausted=True
            )
        except LLMError as exc:  # transport/permanent failure -> stop for this row
            return ExtractionResult(
                data=None, error=str(exc), attempts=attempt, raw_text=raw
            )
    return ExtractionResult(
        data=None, error=last_error, attempts=max_parse_attempts, raw_text=raw
    )