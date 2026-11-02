import json
 
from constants import COL_LINE_TYPE, COL_PRODUCT_NAME, COL_VARIANT, COL_MSRP
from row_builder import build_all_rows
 
with open("Input.json", encoding="utf-8") as f:
    data = json.load(f)
 
rows = build_all_rows(data)
print(f"Products: {len(data)} | Rows: {len(rows)}\n")
for i, r in enumerate(rows, 1):
    print(f"{i:2d}. {r[COL_PRODUCT_NAME]} | {r[COL_VARIANT]} | {r[COL_MSRP]} | {r[COL_LINE_TYPE]}")
 