"""Tiny .env loader (standard library only).

Reads KEY=VALUE lines from a .env file into os.environ so secrets such as
GEMINI_API_KEY stay out of the source code and out of Git.

Rules
  * Lines starting with '#' and blank lines are ignored.
  * An optional leading 'export ' is allowed.
  * Values may be wrapped in single or double quotes.
  * Variables that are ALREADY set in the real environment win (unless
    override=True), so a terminal/CI setting always beats the file.
  * A missing .env file is not an error (returns an empty dict).
"""

import os
from typing import Dict


def parse_env_line(line: str):
    """Return (key, value) for a valid KEY=VALUE line, otherwise None."""
    line = line.strip().lstrip("\ufeff")
    if not line or line.startswith("#") or "=" not in line:
        return None
    if line.startswith("export "):
        line = line[len("export "):].lstrip()
    key, _, value = line.partition("=")
    key, value = key.strip(), value.strip()
    if not key or " " in key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    elif " #" in value:  # inline comment on an unquoted value
        value = value.split(" #", 1)[0].rstrip()
    return key, value


def load_env(path: str = ".env", override: bool = False) -> Dict[str, str]:
    """Load `path` into os.environ. Returns the variables that were actually set."""
    applied: Dict[str, str] = {}
    if not os.path.isfile(path):
        return applied
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            parsed = parse_env_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            if override or key not in os.environ:
                os.environ[key] = value
                applied[key] = value
    return applied