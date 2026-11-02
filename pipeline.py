"""Pipeline (Step 8): runs every product through LLM extraction + validation.

Flow per product, per line type (New Line, Upgrade):
    base row (Step 3/4)  ->  prompt (Step 5)  ->  LLM (Step 6)
                         ->  validation (Step 7)  ->  final row

Reliability features
  * Resume: each successful LLM answer is appended to a checkpoint file right
    away. A rerun reuses saved answers, so an interrupted or partly failed run
    never repeats finished calls. The checkpoint stores the RAW LLM answer, so
    changes to validation rules take effect on rerun with zero new API calls.
  * Cache key = row position + line type + hash of the exact prompt. Editing the
    prompt or the input data automatically invalidates the old answer.
  * Failed rows are never cached (they are retried next run) and still produce a
    complete row (base fields + deterministic PWP/GWP) with an issue note.
  * If the API reports an exhausted quota (e.g. tokens per day), the run stops
    calling the API at once; remaining rows are marked incomplete, progress is kept.
  * Output integrity is verified before anything is written.
"""

import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from constants import (
    COL_LINE_TYPE, EXPECTED_COLUMN_COUNT, LINE_TYPES, MONEY_COLUMNS, OUTPUT_COLUMNS,
)
from llm_client import BaseLLMClient, ExtractionResult, extract_for_row
from prompt_builder import build_messages, source_trim_report
from row_builder import build_rows_for_product
from validator import build_final_row

Row = Dict[str, Optional[str]]

_MONEY_CELL = re.compile(r"^\$\d+(\.\d+)?$")
_PLACEHOLDERS = {"n/a", "na", "0", "-", "null", "none", "nan", "nil"}


# --------------------------------------------------------------------------- #
# Checkpoint (resume support)
# --------------------------------------------------------------------------- #
def make_key(index: int, line_type: str, system: str, user: str) -> str:
    """Stable cache key: row position + line type + hash of the exact prompt."""
    digest = hashlib.sha256((system + "\x00" + user).encode("utf-8")).hexdigest()[:20]
    return f"{index}|{line_type}|{digest}"


class Checkpoint:
    """Append-only JSON-lines store of raw LLM answers."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.skipped_lines = 0
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:  # a half-written last line (e.g. after Ctrl+C) is simply ignored
                    record = json.loads(line)
                    if isinstance(record, dict) and "key" in record and isinstance(record.get("data"), dict):
                        self.cache[record["key"]] = record["data"]
                    else:
                        self.skipped_lines += 1
                except json.JSONDecodeError:
                    self.skipped_lines += 1

    def get(self, key: Optional[str]) -> Optional[Dict[str, Any]]:
        return self.cache.get(key) if key else None

    def put(self, key: Optional[str], data: Dict[str, Any], model: Optional[str] = None) -> None:
        if not key:
            return
        self.cache[key] = data
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "model": model, "data": data}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


# --------------------------------------------------------------------------- #
# Pipeline run
# --------------------------------------------------------------------------- #
@dataclass
class PipelineResult:
    rows: List[Row] = field(default_factory=list)
    row_issues: List[Tuple[str, List[str]]] = field(default_factory=list)  # (row label, issues)
    failed: List[str] = field(default_factory=list)  # labels of rows whose LLM call failed
    live_calls: int = 0
    cached_rows: int = 0
    quota_stopped: bool = False  # the API quota ran out; remaining rows were not attempted

    @property
    def complete(self) -> bool:
        return not self.failed


def _label(position: int, total: int, row: Row) -> str:
    return (f"[{position}/{total}] {row.get('PRODUCT NAME')} | {row.get('VARIANT')} | "
            f"{row.get(COL_LINE_TYPE)}")


def run_pipeline(
    products: List[Dict[str, Any]],
    client: BaseLLMClient,
    checkpoint: Optional[Checkpoint] = None,
    pause: float = 0.0,
    progress: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> PipelineResult:
    """Process all products, two rows each (New Line first, then Upgrade)."""
    if not isinstance(products, list):
        raise TypeError(f"Expected a list of product objects, got {type(products).__name__}")

    result = PipelineResult()
    total = len(products) * len(LINE_TYPES)
    position = 0
    model = getattr(client, "model", None)

    for index, product in enumerate(products):
        for row in build_rows_for_product(product):  # New Line row, Upgrade row
            position += 1
            line_type = row[COL_LINE_TYPE]
            label = _label(position, total, row)

            key: Optional[str] = None
            try:
                messages = build_messages(product, line_type)
                key = make_key(index, line_type, messages["system"], messages["user"])
            except Exception:  # noqa: BLE001 - odd payload; extract_for_row reports it
                pass

            cached = checkpoint.get(key) if checkpoint else None
            started = time.monotonic()
            if cached is not None:
                extraction = ExtractionResult(data=cached, attempts=0)
                result.cached_rows += 1
                source = "resumed from checkpoint"
            elif result.quota_stopped:
                # Quota is gone: do not burn more requests. Saved rows are still reused above.
                extraction = ExtractionResult(
                    data=None, attempts=0, quota_exhausted=True,
                    error="skipped: API quota exhausted earlier in this run (rerun after it resets)",
                )
                source = "not attempted: quota exhausted"
            else:
                extraction = extract_for_row(client, product, line_type)
                result.live_calls += 1
                source = f"LLM {time.monotonic() - started:.1f}s, {extraction.attempts} attempt(s)"
                if extraction.ok and checkpoint:
                    checkpoint.put(key, extraction.data, model)
                if extraction.quota_exhausted:
                    result.quota_stopped = True
                if pause and position < total:
                    sleep(pause)

            final_row, issues = build_final_row(row, product, line_type, extraction)
            trim_note = source_trim_report(product)
            if trim_note:
                issues.insert(0, trim_note)
            result.rows.append(final_row)
            if issues:
                result.row_issues.append((label, issues))
            if not extraction.ok:
                result.failed.append(label)
            status = ("OK" if extraction.ok else
                      "SKIPPED (quota)" if extraction.quota_exhausted and extraction.attempts == 0
                      else "FAILED (see issues log)")
            reason = ""
            if not extraction.ok and not (extraction.quota_exhausted and extraction.attempts == 0):
                reason = f"\n      reason: {str(extraction.error)[:220]}"
            progress(f"{label}  {status}  [{source}; {len(issues)} note(s)]{reason}")

    return result


# --------------------------------------------------------------------------- #
# Output integrity + writers
# --------------------------------------------------------------------------- #
def check_output_integrity(rows: List[Row], product_count: int) -> List[str]:
    """Return a list of problems (empty list = the output meets the assignment's rules)."""
    problems: List[str] = []
    if len(rows) != 2 * product_count:
        problems.append(f"expected {2 * product_count} rows (2 per product), got {len(rows)}")
    for i, row in enumerate(rows, 1):
        if list(row.keys()) != OUTPUT_COLUMNS:
            problems.append(f"row {i}: columns differ from the required {EXPECTED_COLUMN_COUNT}-column schema/order")
            continue
        expected_line = LINE_TYPES[(i - 1) % 2]
        if row[COL_LINE_TYPE] != expected_line:
            problems.append(f"row {i}: LINE TYPE is {row[COL_LINE_TYPE]!r}, expected {expected_line!r}")
        for column in MONEY_COLUMNS:
            value = row[column]
            if value is not None and not _MONEY_CELL.match(value):
                problems.append(f"row {i}: {column}={value!r} is not formatted like $800 / $244.36")
        for column, value in row.items():
            if isinstance(value, str) and value.strip().lower() in _PLACEHOLDERS:
                problems.append(f"row {i}: {column} uses placeholder {value!r}; must be blank")
    return problems


def write_csv(rows: List[Row], path: str) -> None:
    """Write the CSV: exact headers, None -> blank cell, UTF-8 without BOM."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row[c] is None else row[c]) for c in OUTPUT_COLUMNS})


def write_issues_log(result: PipelineResult, path: str, usage_summary: str = "") -> None:
    """Human-readable audit trail: every dropped/adjusted value and failed row."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ISSUES LOG - every value dropped or adjusted by validation\n")
        handle.write(f"Rows: {len(result.rows)} | live LLM calls: {result.live_calls} | "
                     f"resumed from checkpoint: {result.cached_rows} | failed rows: {len(result.failed)}\n")
        if usage_summary:
            handle.write(f"Token usage this run: {usage_summary}\n")
        handle.write("=" * 78 + "\n")
        if not result.row_issues:
            handle.write("No issues.\n")
        for label, issues in result.row_issues:
            handle.write(f"\n{label}\n")
            for issue in issues:
                handle.write(f"  - {issue}\n")
        if result.failed:
            handle.write("\nINCOMPLETE ROWS (rerun the same command to retry only these):\n")
            for label in result.failed:
                handle.write(f"  - {label}\n")