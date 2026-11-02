"""Command-line entry point: Input.json -> normalized CSV.

Examples
  python main.py                      # full run (needs GEMINI_API_KEY in .env), resumes if interrupted
  python main.py --dry-run            # no network, no key: proves the pipeline/CSV structure
  python main.py --limit 1            # only the first product (2 rows), handy for a cheap live test
  python main.py --fresh              # ignore saved progress and start over

Exit codes: 0 = complete | 2 = CSV written but some rows have no LLM data (rerun to retry)
            1 = error (bad input, bad config, failed integrity check) | 130 = interrupted
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from env_loader import load_env
from llm_client import BaseLLMClient, DryRunClient, LLMConfigError, get_client
from pipeline import (
    Checkpoint, check_output_integrity, run_pipeline, write_csv, write_issues_log,
)

DEFAULT_OUTPUT = os.path.join("output", "normalized_output.csv")
DEFAULT_DRY_OUTPUT = os.path.join("output", "dry_run_output.csv")
DEFAULT_CHECKPOINT = os.path.join("output", "checkpoint.jsonl")
DEFAULT_ISSUES = os.path.join("output", "issues.log")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize retailer promo JSON into a 29-column CSV.")
    parser.add_argument("--input", default="Input.json", help="raw JSON file (default: Input.json)")
    parser.add_argument("--output", default=None, help=f"CSV path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="resume file for LLM answers")
    parser.add_argument("--issues", default=DEFAULT_ISSUES, help="audit log of dropped/adjusted values")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N products")
    parser.add_argument("--provider", default=None, help="LLM provider (default: LLM_PROVIDER or gemini)")
    parser.add_argument("--dry-run", action="store_true", help="no network/key; LLM-derived columns stay blank")
    parser.add_argument("--fresh", action="store_true", help="delete saved progress before running")
    parser.add_argument("--no-resume", action="store_true", help="do not read or write the checkpoint")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to wait between live calls")
    return parser.parse_args(argv)


def load_products(path: str) -> List[Dict[str, Any]]:
    """Read and sanity-check the input JSON. Raises ValueError with a clear message."""
    if not os.path.isfile(path):
        raise ValueError(f"Input file not found: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Input file is not valid JSON: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError("Input JSON must be a non-empty array of product objects")
    bad = [i for i, item in enumerate(data) if not isinstance(item, dict)]
    if bad:
        raise ValueError(f"Input items at positions {bad[:5]} are not JSON objects")
    return data


def main(argv: Optional[List[str]] = None, client: Optional[BaseLLMClient] = None,
         env_file: Optional[str] = ".env") -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")
    if env_file:
        load_env(env_file)  # real environment variables still take priority

    try:
        products = load_products(args.input)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    if args.limit is not None:
        products = products[: max(args.limit, 0)]
        if not products:
            print("ERROR: --limit left no products to process")
            return 1

    if client is None:
        try:
            client = get_client("dryrun" if args.dry_run else args.provider)
        except LLMConfigError as exc:
            print(f"ERROR: {exc}")
            return 1
    dry = isinstance(client, DryRunClient)
    output_path = args.output or (DEFAULT_DRY_OUTPUT if dry else DEFAULT_OUTPUT)

    checkpoint: Optional[Checkpoint] = None
    if not dry and not args.no_resume:
        if args.fresh and os.path.isfile(args.checkpoint):
            os.remove(args.checkpoint)
        checkpoint = Checkpoint(args.checkpoint)
        if checkpoint.cache:
            print(f"Found {len(checkpoint.cache)} saved answer(s) in {args.checkpoint}; they will be reused.")
        if checkpoint.skipped_lines:
            print(f"(ignored {checkpoint.skipped_lines} unreadable line(s) in the checkpoint)")

    print(f"Processing {len(products)} product(s) -> {2 * len(products)} rows"
          f"{' [DRY RUN: no LLM, offer columns will be blank]' if dry else ''}\n")

    try:
        result = run_pipeline(products, client, checkpoint=checkpoint, pause=args.pause)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; run the same command again to resume.")
        return 130

    problems = check_output_integrity(result.rows, len(products))
    if problems:
        print("\nOUTPUT INTEGRITY CHECK FAILED - CSV not written:")
        for problem in problems[:20]:
            print("  -", problem)
        return 1

    write_csv(result.rows, output_path)
    write_issues_log(result, args.issues, usage_summary=client.usage.summary())

    print("\n" + "=" * 70)
    print(f"CSV written:   {output_path}  ({len(result.rows)} rows x 29 columns)")
    print(f"Issues log:    {args.issues}")
    calls_label = "Dry-run calls (nothing sent)" if dry else "Live LLM calls"
    print(f"{calls_label}: {result.live_calls} | resumed from checkpoint: {result.cached_rows}")
    print(f"Token usage this run: {client.usage.summary()}")
    if not result.complete:
        if result.quota_stopped:
            print(f"\nAPI QUOTA EXHAUSTED: {len(result.failed)} row(s) are still pending. Progress is saved. "
                  "Rerun the same command after your quota resets (daily limits reset on the provider's "
                  "schedule) and only the pending rows will be processed.")
        else:
            print(f"\n{len(result.failed)} row(s) have NO LLM data (API failures). Rerun the same command "
                  "to retry only those; finished rows are reused.")
        return 2
    print("All rows complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    