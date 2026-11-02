import json

from constants import LINE_NEW, LINE_UPGRADE, LINE_TYPES
from prompt_builder import build_messages, collect_source_text

with open("Input.json", encoding="utf-8") as f:
    data = json.load(f)

# 1) Build all 24 prompts and report their size
total = 0
for i, product in enumerate(data):
    for line_type in LINE_TYPES:
        msg = build_messages(product, line_type)
        total += 1
        print(f"{total:2d}. {product['product_name']} | {product['variant']} | {line_type} "
              f"| user message: {len(msg['user'])} chars")

# 2) Save one full prompt (product 0, New Line) so you can read exactly what the LLM will see
msg = build_messages(data[0], LINE_NEW)
with open("sample_prompt_product0_new_line.txt", "w", encoding="utf-8") as f:
    f.write("=== SYSTEM ===\n" + msg["system"] + "\n\n=== USER ===\n" + msg["user"])
print("\nSaved: sample_prompt_product0_new_line.txt")

# 3) Confirm key amounts exist in the source text the LLM will see
src = collect_source_text(data[0])
for amount in ["$628.36", "$200", "$800", "$240", "$10"]:
    print(f"{amount!r} in source for product 0: {amount in src}")