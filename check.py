import json
from mapper import map_base_fields

with open("Input.json", encoding="utf-8") as f:
    data = json.load(f)

row = map_base_fields(data[0])
for column, value in row.items():
    print(f"{column}: {value}")
    