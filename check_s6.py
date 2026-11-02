"""Step 6 live check: ONE real Gemini call (product 0, New Line).

Prerequisite: a .env file next to this script (copy .env.example to .env and
fill in GEMINI_API_KEY; optionally GEMINI_MODEL).
Run:  python .\\check_step6.py
"""

import json
import logging

from constants import LINE_NEW
from env_loader import load_env
from llm_client import LLMConfigError, extract_for_row, get_client

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
loaded = load_env(".env")  # real environment variables still take priority
print(f".env loaded: {sorted(loaded) if loaded else 'no .env found or nothing new to load'}")

with open("Input.json", encoding="utf-8") as f:
    data = json.load(f)

try:
    client = get_client()  # reads LLM_PROVIDER (default: gemini) and GEMINI_API_KEY
except LLMConfigError as exc:
    raise SystemExit(f"Config problem: {exc}")

product = data[0]
print(f"Model: {getattr(client, 'model', 'n/a')}")
print(f"Asking about: {product['product_name']} | {product['variant']} | {LINE_NEW}\n")

result = extract_for_row(client, product, LINE_NEW)

if result.ok:
    print("PARSED JSON FROM MODEL:")
    print(json.dumps(result.data, indent=2, ensure_ascii=False))
else:
    print("FAILED:", result.error)
    if result.raw_text:
        print("\nRaw model text (first 800 chars):\n", result.raw_text[:800])

print(f"\nAttempts: {result.attempts}")
print("Token usage:", client.usage.summary())