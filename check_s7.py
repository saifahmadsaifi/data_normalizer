"""Step 7 live check: 2 real Gemini calls (product 0: New Line + Upgrade),
then validation. Prints only the columns that ended up with a value, plus
every rejection reason.

Run:  python .\\check_step7.py
"""

import json
import logging

from constants import LINE_TYPES, COL_LINE_TYPE
from env_loader import load_env
from llm_client import LLMConfigError, extract_for_row, get_client
from mapper import map_base_fields
from validator import build_final_row

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
load_env(".env")

with open("Input.json", encoding="utf-8") as f:
    data = json.load(f)

try:
    client = get_client()
except LLMConfigError as exc:
    raise SystemExit(f"Config problem: {exc}")

product = data[0]
for line_type in LINE_TYPES:
    print("=" * 78)
    print(f"{product['product_name']} | {product['variant']} | {line_type}")
    print("=" * 78)
    base = map_base_fields(product)
    base[COL_LINE_TYPE] = line_type
    extraction = extract_for_row(client, product, line_type)
    row, issues = build_final_row(base, product, line_type, extraction)

    for column, value in row.items():
        if value is not None:
            print(f"  {column}: {value}")
    print("\n  Rejections / notes:")
    for issue in issues or ["(none)"]:
        print("   -", issue)
    print()

print("Token usage:", client.usage.summary())