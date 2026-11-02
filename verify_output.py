"""Step 9: verify the final CSV against the assignment's rules and against the source JSON.

Run:  python verify_output.py                       (defaults: output/normalized_output.csv, Input.json)
      python verify_output.py --csv path.csv --input Input.json

Three kinds of finding
  FAIL  a rule is broken or a value is wrong (exit code 1)
  WARN  a value that the source text supports is MISSING (a possible miss, never an invented value)
  PASS  checked and fine

What is checked
  Structure   exact 29 headers in order | 2 rows per product (New Line, then Upgrade) |
              base fields equal the JSON | no BOM/whitespace/placeholder cells
  Formatting  money cells look like $800 / $244.36 | paired columns are filled together
  PDF rules   Upgrade rows never carry a switcher | new-line-only offers never appear on Upgrade rows |
              no source snippet is used for two columns
  Grounding   every amount in a money column, and every $ amount inside a text cell, exists in
              that product's source JSON (nothing invented, nothing rounded)
  Recomputed  unique discount = amount after the product's OWN storage label | max trade-in = highest
  from source tier that names the model | finance = components (New Line $450, Upgrade $250) |
              PWP/GWP = was - now | rate-plan duration only where an installment offer exists
"""

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from constants import (
    COL_ADDITIONAL_OFFERS, COL_ADDITIONAL_VALUE, COL_BASE, COL_BOGO_OFFER, COL_BRAND,
    COL_CLEANED_PWP_GWP_OFFERS, COL_FINANCE_OFFER_DESC, COL_FINANCE_OFFER_VALUE,
    COL_FREE_MEMORY_UPGRADE, COL_LINE_TYPE, COL_MAX_TI, COL_MSRP, COL_PLAN_TYPE,
    COL_PRODUCT_NAME, COL_PWP_GWP_OFFERS, COL_PWP_GWP_VALUE, COL_RATEPLAN_DURATION,
    COL_RESELLER_LIST_PRICE, COL_SERVICE_LINE_OFFERS, COL_SERVICE_LINE_VALUE, COL_SHIPMENT,
    COL_STOCK, COL_SUBSCRIPTION_BUNDLE_OFFERS, COL_SUBSCRIPTION_BUNDLE_VALUE, COL_SWITCHER,
    COL_SWITCHER_OFFER_TEXT, COL_TRADE_IN_VALUE_TYPE, COL_UNIQUE_DISCOUNT, COL_VARIANT,
    LINE_NEW, LINE_UPGRADE, LINE_TYPES, MONEY_COLUMNS, OUTPUT_COLUMNS,
)
from mapper import format_money
from prompt_builder import normalize_variant
from pwp import extract_pwp
from validator import SourceIndex, _Ctx, amount_regex, amounts_in, applicable_trade_in_tiers, digits_of, norm

MONEY_RE = re.compile(r"^\$\d+(\.\d+)?$")
PLACEHOLDERS = {"n/a", "na", "0", "-", "null", "none", "nan", "nil"}
SKIP_GROUNDING = {COL_MSRP, COL_FINANCE_OFFER_VALUE, COL_PWP_GWP_VALUE}   # checked by dedicated rules
TEXT_COLUMNS = [COL_SWITCHER_OFFER_TEXT, COL_SERVICE_LINE_OFFERS, COL_ADDITIONAL_OFFERS, COL_FREE_MEMORY_UPGRADE,
                COL_BOGO_OFFER, COL_SUBSCRIPTION_BUNDLE_OFFERS, COL_FINANCE_OFFER_DESC, COL_PWP_GWP_OFFERS]
PAIRS = [(COL_SWITCHER, COL_SWITCHER_OFFER_TEXT), (COL_SERVICE_LINE_VALUE, COL_SERVICE_LINE_OFFERS),
         (COL_FINANCE_OFFER_VALUE, COL_FINANCE_OFFER_DESC), (COL_PWP_GWP_VALUE, COL_PWP_GWP_OFFERS),
         (COL_PWP_GWP_VALUE, COL_CLEANED_PWP_GWP_OFFERS), (COL_ADDITIONAL_VALUE, COL_ADDITIONAL_OFFERS),
         (COL_SUBSCRIPTION_BUNDLE_VALUE, COL_SUBSCRIPTION_BUNDLE_OFFERS)]
EXPECTED_BLANK = [COL_BASE, COL_RESELLER_LIST_PRICE, COL_TRADE_IN_VALUE_TYPE]   # no evidence in the data

_VARIANT_AMOUNT = re.compile(
    r"(\d+(?:\.\d+)?(?:tb|gb)) for \$[\d.]+/mo\. for \d+ mos\. after up to \$([\d.]+) in bill credits")
_SWITCHER = re.compile(r"get up to an? \$(\d+(?:\.\d+)?) at&t visa reward card when you switch")
_SERVICE = [re.compile(r"price after additional \$(\d+(?:\.\d+)?) in credits"),
            re.compile(r"get \$(\d+(?:\.\d+)?) off a new line")]
_CITI = {"one_time_welcome": re.compile(r"\$(\d+(?:\.\d+)?) statement credit after spending"),
         "recurring_annual": re.compile(r"earn up to \$(\d+(?:\.\d+)?) in statement credits every year"),
         "recurring_discount": re.compile(r"\$(\d+(?:\.\d+)?) per line monthly discount")}


@dataclass
class Report:
    findings: List[tuple] = field(default_factory=list)

    def add(self, level: str, check: str, detail: str = "") -> None:
        self.findings.append((level, check, detail))

    def count(self, level: str) -> int:
        return sum(1 for f in self.findings if f[0] == level)

    @property
    def ok(self) -> bool:
        return self.count("FAIL") == 0


def _strings(obj: Any) -> List[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def _money(value: str) -> Decimal:
    return Decimal(digits_of(value))


def _label(i: int, row: Dict[str, str]) -> str:
    return f"row {i} ({row.get(COL_PRODUCT_NAME)} | {row.get(COL_VARIANT)} | {row.get(COL_LINE_TYPE)})"


def verify(csv_path: str, input_path: str) -> Report:
    report = Report()
    with open(input_path, encoding="utf-8") as handle:
        products = json.load(handle)
    with open(csv_path, "rb") as handle:
        raw = handle.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        report.add("WARN", "encoding", "file starts with a UTF-8 BOM; the first header may read as '\\ufeffPLAN TYPE' in some tools")
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        data_rows = list(reader)

    # ---- structure ----------------------------------------------------------------- #
    if header == OUTPUT_COLUMNS:
        report.add("PASS", "headers", "29 columns, exact names and order")
    else:
        missing = [c for c in OUTPUT_COLUMNS if c not in header]
        extra = [c for c in header if c not in OUTPUT_COLUMNS]
        report.add("FAIL", "headers", f"columns differ (missing {missing}, extra {extra}, count {len(header)})")
        return report                                       # nothing else is meaningful
    bad_width = [i for i, r in enumerate(data_rows, 1) if len(r) != 29]
    if bad_width:
        report.add("FAIL", "row_width", f"rows with a wrong number of cells: {bad_width[:5]}")
        return report
    rows = [dict(zip(header, r)) for r in data_rows]
    expected_rows = 2 * len(products)
    if len(rows) == expected_rows:
        report.add("PASS", "row_count", f"{len(rows)} rows = 2 x {len(products)} products")
    else:
        report.add("FAIL", "row_count", f"expected {expected_rows} rows (2 per product), found {len(rows)}")
        return report

    bad = [i for i, r in enumerate(rows, 1) if r[COL_LINE_TYPE] != LINE_TYPES[(i - 1) % 2]]
    report.add("FAIL" if bad else "PASS", "line_type_order",
               f"rows with the wrong LINE TYPE (expected New Line, Upgrade, New Line, ...): {bad[:6]}" if bad
               else "every product has a New Line row followed by an Upgrade row")

    problems = []
    for i, row in enumerate(rows, 1):
        p = products[(i - 1) // 2]
        pricing = p.get("pricing") or {}
        expect = {COL_PRODUCT_NAME: p.get("product_name"), COL_BRAND: p.get("brand"), COL_VARIANT: p.get("variant"),
                  COL_PLAN_TYPE: p.get("plan_type"), COL_STOCK: p.get("stock"), COL_SHIPMENT: p.get("shipment"),
                  COL_MSRP: format_money(pricing.get("retail"))}
        for col, value in expect.items():
            if (row[col] or None) != (value or None):
                problems.append(f"{_label(i, row)}: {col} is {row[col]!r}, JSON says {value!r}")
    report.add("FAIL" if problems else "PASS", "base_fields",
               "; ".join(problems[:4]) if problems else "product name, brand, variant, plan type, stock, shipment and MSRP match the JSON for all rows")

    # ---- cell hygiene ---------------------------------------------------------------- #
    problems = []
    for i, row in enumerate(rows, 1):
        for col in MONEY_COLUMNS:
            if row[col] and not MONEY_RE.match(row[col]):
                problems.append(f"{_label(i, row)}: {col}={row[col]!r}")
    report.add("FAIL" if problems else "PASS", "money_format",
               "; ".join(problems[:4]) if problems else "all money cells look like $800 / $244.36 (leading $, exact decimals, no commas)")

    problems = [f"{_label(i, r)}: {c}={v!r}" for i, r in enumerate(rows, 1) for c, v in r.items()
                if v.strip().lower() in PLACEHOLDERS]
    report.add("FAIL" if problems else "PASS", "placeholders", "; ".join(problems[:4]) if problems
               else "no N/A, 0, -, null, None or nan placeholders (empty cells are truly blank)")
    problems = [f"{_label(i, r)}: {c}" for i, r in enumerate(rows, 1) for c, v in r.items() if v != v.strip()]
    report.add("FAIL" if problems else "PASS", "whitespace", "; ".join(problems[:4]) if problems
               else "no leading/trailing whitespace in any cell")

    problems = []
    for i, row in enumerate(rows, 1):
        for a, b in PAIRS:
            if bool(row[a]) != bool(row[b]):
                problems.append(f"{_label(i, row)}: {a} and {b} must be filled together")
    report.add("FAIL" if problems else "PASS", "paired_columns", "; ".join(problems[:4]) if problems
               else "each amount has its text and each text has its amount")

    problems = [f"{_label(i, r)}: {c} should be blank (no evidence in the data)" for i, r in enumerate(rows, 1)
                for c in EXPECTED_BLANK if r[c]]
    report.add("FAIL" if problems else "PASS", "undefined_columns_blank", "; ".join(problems[:3]) if problems
               else "BASE, RESELLER LIST PRICE and Trade-In Value Type are blank (not defined by the PDF/data)")

    # ---- per-row rules ---------------------------------------------------------------- #
    fails: Dict[str, List[str]] = {}
    warns: Dict[str, List[str]] = {}

    def fail(check, msg):
        fails.setdefault(check, []).append(msg)

    def warn(check, msg):
        warns.setdefault(check, []).append(msg)

    passed_counts: Dict[str, int] = {}
    cache: Dict[int, Dict[str, Any]] = {}      # per-product facts, computed once and shared by both rows

    for i, row in enumerate(rows, 1):
        pi = (i - 1) // 2
        p = products[pi]
        line = row[COL_LINE_TYPE]
        if pi not in cache:
            ctx = _Ctx(p, LINE_NEW)
            cache[pi] = {"index": ctx.src, "everything": norm(" ".join(_strings(p))),
                         "tiers": applicable_trade_in_tiers(ctx)[0], "pwp": extract_pwp(p)[0]}
        index = cache[pi]["index"]
        src = index.full
        everything = cache[pi]["everything"]
        who = _label(i, row)

        # PDF rule: a customer upgrading an existing line cannot be a switcher
        if line == LINE_UPGRADE and (row[COL_SWITCHER] or row[COL_SWITCHER_OFFER_TEXT]):
            fail("switcher_upgrade_blank", f"{who}: Upgrade row has a switcher offer")

        # new-line-only offers must not appear on Upgrade rows
        if line == LINE_UPGRADE:
            for col in (COL_UNIQUE_DISCOUNT, COL_SERVICE_LINE_VALUE):
                if row[col]:
                    groups = [g for g, _, text in index.blocks if amount_regex(digits_of(row[col])).search(text)]
                    if not groups or not all(index.line_gate(g)[1] for g in groups):
                        fail("gating_upgrade", f"{who}: {col}={row[col]} comes from a new-line-only offer")

        # grounding: amounts exist in the source, nothing rounded
        for col in MONEY_COLUMNS:
            if col in SKIP_GROUNDING or not row[col]:
                continue
            if not amount_regex(digits_of(row[col])).search(everything):
                fail("amount_grounding", f"{who}: {col}={row[col]} not found in the source JSON")
        for col in TEXT_COLUMNS:
            for amount in amounts_in(row[col]):
                if not amount_regex(amount).search(everything):
                    fail("text_grounding", f"{who}: ${amount} in {col} not found in the source JSON")

        # non-duplication: a snippet feeds one column only
        seen: Dict[str, str] = {}
        for col in TEXT_COLUMNS:
            if row[col]:
                key = norm(row[col])
                if key in seen:
                    fail("non_duplication", f"{who}: identical text in {seen[key]} and {col}")
                seen[key] = col

        # unique discount: amount after THIS product's storage label, New Line only
        labels: Dict[str, str] = {}
        for m in _VARIANT_AMOUNT.finditer(src):
            labels.setdefault(m.group(1), m.group(2))
        want_unique = f"${labels[normalize_variant(p['variant'])]}" if normalize_variant(p["variant"]) in labels and line == LINE_NEW else None
        if (row[COL_UNIQUE_DISCOUNT] or None) != want_unique:
            (warn if (not row[COL_UNIQUE_DISCOUNT] and want_unique) else fail)(
                "variant_anchoring", f"{who}: UNIQUE DISCOUNT is {row[COL_UNIQUE_DISCOUNT] or 'blank'}, source says {want_unique or 'blank'}")
        else:
            passed_counts["variant_anchoring"] = passed_counts.get("variant_anchoring", 0) + 1

        # max trade-in: highest tier that names this model (same for both line types)
        tiers = cache[pi]["tiers"]
        want_ti = format_money(max(tiers, key=Decimal)) if tiers else None
        if (row[COL_MAX_TI] or None) != want_ti:
            (warn if (not row[COL_MAX_TI] and want_ti) else fail)(
                "max_ti_tier", f"{who}: Max TI is {row[COL_MAX_TI] or 'blank'}, highest tier for this model is {want_ti or 'none'}")
        else:
            passed_counts["max_ti_tier"] = passed_counts.get("max_ti_tier", 0) + 1

        # switcher (New Line only) and service line: recall checks against the source text
        m = _SWITCHER.search(src)
        want_sw = f"${m.group(1)}" if m and line == LINE_NEW else None
        if row[COL_SWITCHER] and row[COL_SWITCHER] != want_sw and line == LINE_NEW:
            fail("switcher_value", f"{who}: SWITCHER {row[COL_SWITCHER]} but the source offers {want_sw}")
        elif want_sw and not row[COL_SWITCHER]:
            warn("switcher_missing", f"{who}: source offers {want_sw} but SWITCHER is blank")
        svc = next((f"${mm.group(1)}" for rx in _SERVICE if (mm := rx.search(src))), None)
        if line == LINE_NEW and svc and row[COL_SERVICE_LINE_VALUE] != svc:
            (warn if not row[COL_SERVICE_LINE_VALUE] else fail)(
                "service_line", f"{who}: SERVICE LINE is {row[COL_SERVICE_LINE_VALUE] or 'blank'}, source says {svc}")

        # finance: components valid for the line type, added exactly
        comps = {k: Decimal(m.group(1)) for k, rx in _CITI.items() if (m := rx.search(src))}
        if len(comps) == 3:
            want_fin = comps["recurring_annual"] + comps["recurring_discount"] + (comps["one_time_welcome"] if line == LINE_NEW else 0)
            got = row[COL_FINANCE_OFFER_VALUE]
            if not got:
                warn("finance_missing", f"{who}: source supports ${want_fin} but FINANCE value is blank")
            elif _money(got) != want_fin:
                fail("finance_values", f"{who}: FINANCE {got}, expected ${want_fin} ({'welcome + ' if line == LINE_NEW else ''}annual + discount)")
            if line == LINE_UPGRADE and re.search(r"welcome|sign-?up|bonus", row[COL_FINANCE_OFFER_DESC], re.I):
                fail("finance_values", f"{who}: Upgrade description mentions the welcome bonus")
            if got and _money(got) == want_fin:
                passed_counts["finance_values"] = passed_counts.get("finance_values", 0) + 1

        # PWP/GWP: was - now, same on every row
        pwp = cache[pi]["pwp"]
        if (row[COL_PWP_GWP_VALUE] or None) != pwp["value"]:
            fail("pwp", f"{who}: PWP value {row[COL_PWP_GWP_VALUE] or 'blank'}, computed {pwp['value']}")

        # rate-plan duration: stated in the source, and only with an installment offer
        if row[COL_RATEPLAN_DURATION]:
            n = row[COL_RATEPLAN_DURATION]
            if not n.isdigit() or not re.search(rf"(?<!\d){n}\s*(?:-?\s*)(?:mos?\b|months?\b)", src):
                fail("duration", f"{who}: RATEPLAN DURATION {n!r} is not stated in the source")
            if not any(row[c] for c in (COL_UNIQUE_DISCOUNT, COL_SERVICE_LINE_VALUE, COL_MAX_TI)):
                fail("duration", f"{who}: duration given but no installment offer on this row")

    for check in ("switcher_upgrade_blank", "gating_upgrade", "amount_grounding", "text_grounding",
                  "non_duplication", "variant_anchoring", "max_ti_tier", "switcher_value", "service_line",
                  "finance_values", "pwp", "duration"):
        if check in fails:
            report.add("FAIL", check, f"{len(fails[check])} problem(s): " + "; ".join(fails[check][:3]))
        else:
            detail = {"switcher_upgrade_blank": "no Upgrade row has a switcher",
                      "gating_upgrade": "no Upgrade row carries a new-line-only offer",
                      "amount_grounding": "every amount in a money column exists in its product's source",
                      "text_grounding": "every $ amount inside a text cell exists in its product's source",
                      "non_duplication": "no snippet is used for two columns",
                      "variant_anchoring": "each UNIQUE DISCOUNT matches the product's own storage size and New Line only",
                      "max_ti_tier": "Max TI equals the highest tier that names the model, on both line types",
                      "switcher_value": "SWITCHER amounts match the source",
                      "service_line": "SERVICE LINE amounts match the source",
                      "finance_values": "finance totals are exact (New Line = welcome + annual + discount; Upgrade = annual + discount)",
                      "pwp": "PWP/GWP equals was - now from the accessories data",
                      "duration": "rate-plan duration is stated in the source and tied to an installment offer"}[check]
            report.add("PASS", check, detail)
    for check, msgs in warns.items():
        report.add("WARN", check, f"{len(msgs)} possible miss(es): " + "; ".join(msgs[:3]))
    return report


def print_table(csv_path: str) -> None:
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cols = [("product", COL_PRODUCT_NAME), ("variant", COL_VARIANT), ("line", COL_LINE_TYPE), ("switcher", COL_SWITCHER),
            ("unique", COL_UNIQUE_DISCOUNT), ("service", COL_SERVICE_LINE_VALUE), ("maxTI", COL_MAX_TI),
            ("finance", COL_FINANCE_OFFER_VALUE), ("pwp", COL_PWP_GWP_VALUE), ("months", COL_RATEPLAN_DURATION)]
    widths = [24, 8, 10, 10, 10, 9, 8, 9, 8, 7]
    print("\n" + "".join(f"{name:<{w}}" for (name, _), w in zip(cols, widths)))
    for r in rows:
        print("".join(f"{(r[c].replace('Apple ', '') if name == 'product' else r[c]):<{w}}" for (name, c), w in zip(cols, widths)))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the normalized CSV against the assignment rules.")
    parser.add_argument("--csv", default=os.path.join("output", "normalized_output.csv"))
    parser.add_argument("--input", default="Input.json")
    parser.add_argument("--no-table", action="store_true", help="skip the 24-row summary table")
    args = parser.parse_args(argv)
    for path in (args.csv, args.input):
        if not os.path.isfile(path):
            print(f"ERROR: file not found: {path}")
            return 2
    report = verify(args.csv, args.input)
    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    for level, check, detail in sorted(report.findings, key=lambda f: order[f[0]]):
        print(f"[{level}] {check}: {detail}")
    print(f"\nSUMMARY: {report.count('PASS')} passed | {report.count('WARN')} warning(s) | {report.count('FAIL')} failure(s)")
    if not args.no_table and report.findings and not any(f[1] in ("headers", "row_width", "row_count") and f[0] == "FAIL" for f in report.findings):
        print_table(args.csv)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())