"""Connection test: ONE tiny request to your configured provider (LLM_PROVIDER), with timing.

Use this when check_step6/check_step7 hang or time out. It tells you whether the
problem is your network/key/model name (this script fails too) or the size of
the real prompt (this script works but the real one is slow).

Run:  python check_connection.py
"""

import logging
import time

from env_loader import load_env
from llm_client import LLMConfigError, LLMError, extract_json, get_client

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
load_env(".env")

try:
    client = get_client()
except LLMConfigError as exc:
    raise SystemExit(f"Config problem: {exc}")

print(f"Model: {getattr(client, 'model', 'n/a')} | fallback: {getattr(client, 'fallback_model', None) or 'none'} "
      f"| timeout: {getattr(client, 'timeout', 'n/a')}s | retries: {getattr(client, 'max_retries', 'n/a')}")
print("Sending a tiny request...")
started = time.monotonic()
try:
    text = client.complete("Reply with JSON only.", 'Return exactly this JSON: {"ok": true}')
    print(f"\nSUCCESS in {time.monotonic() - started:.1f}s using model "
          f"{getattr(client, 'last_model_used', 'n/a')} -> {extract_json(text)}")
    print("Token usage:", client.usage.summary())
except LLMError as exc:
    print(f"\nFAILED after {time.monotonic() - started:.1f}s:\n  {exc}")
    print("\nWhat this usually means:")
    print("  HTTP 400/404 -> wrong model name (check the model setting in .env)")
    print("  HTTP 401/403 -> wrong or disabled API key")
    print("  HTTP 429     -> rate limit / quota (the message says per-minute or per-day); see the provider's limits page")
    print("  HTTP 503     -> provider is busy; wait a few minutes (Gemini: GEMINI_FALLBACK_MODEL can help)")
    print("  HTTP 413     -> one request is larger than your tokens-per-minute limit (Groq): shorten the prompt")
    print("  timeout      -> network, VPN, proxy or firewall blocking the provider's API host")