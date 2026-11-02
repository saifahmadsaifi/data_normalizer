"""Tests for verify_output.py: a clean CSV passes, and every kind of corruption is caught.

Run:  python -m unittest test_verify_output -v
"""

import csv
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

import verify_output as vo
from constants import *  # noqa: F401,F403
from pipeline import run_pipeline
from test_pipeline import CarelessOracleClient
from test_support import Isolation

_ISO = Isolation()


def setUpModule():
    _ISO.start()


def tearDownModule():
    _ISO.stop()


_PATHS = ["Input.json", "/mnt/user-data/uploads/Input.json"]
INPUT_PATH = next(p for p in _PATHS if os.path.exists(p))
with open(INPUT_PATH, encoding="utf-8") as _fh:
    DATA = json.load(_fh)

R_PM_NEW, R_PM_UP = 0, 1       # iPhone 17 Pro Max 1 TB (New Line, Upgrade)   -> 0-based row positions
R_17_NEW, R_17_UP = 10, 11     # iPhone 17 256 GB (New Line, Upgrade)


class VerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = run_pipeline(DATA, CarelessOracleClient(), progress=lambda s: None)
        cls.clean = [{c: (r[c] or "") for c in OUTPUT_COLUMNS} for r in result.rows]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, "o.csv")

    def check(self, mutate=None, header=None, bom=False, drop_last=False):
        rows = [dict(r) for r in self.clean]
        if mutate:
            mutate(rows)
        if drop_last:
            rows = rows[:-1]
        with open(self.path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header or OUTPUT_COLUMNS)
            for row in rows:
                writer.writerow([row[c] for c in OUTPUT_COLUMNS])
        return vo.verify(self.path, INPUT_PATH)

    def failed(self, report):
        return {c for level, c, _ in report.findings if level == "FAIL"}

    def warned(self, report):
        return {c for level, c, _ in report.findings if level == "WARN"}

    # ---- clean file ------------------------------------------------------------------ #
    def test_clean_csv_passes_everything(self):
        report = self.check()
        self.assertTrue(report.ok, report.findings)
        self.assertEqual((report.count("FAIL"), report.count("WARN")), (0, 0))
        self.assertGreaterEqual(report.count("PASS"), 20)

    # ---- structure ------------------------------------------------------------------- #
    def test_header_problems(self):
        swapped = OUTPUT_COLUMNS[1:2] + OUTPUT_COLUMNS[0:1] + OUTPUT_COLUMNS[2:]
        self.assertIn("headers", self.failed(self.check(header=swapped)))
        self.assertIn("headers", self.failed(self.check(header=[c.lower() for c in OUTPUT_COLUMNS])))
        self.assertIn("headers", self.failed(self.check(header=OUTPUT_COLUMNS[:-1] + ["EXTRA"])))

    def test_row_count_and_order(self):
        self.assertIn("row_count", self.failed(self.check(drop_last=True)))

        def swap(rows):
            rows[0], rows[1] = rows[1], rows[0]
        self.assertIn("line_type_order", self.failed(self.check(swap)))

    def test_base_fields_must_match_json(self):
        for col, value in ((COL_MSRP, "$1599.00"), (COL_VARIANT, "2 TB"), (COL_STOCK, "sold_out"), (COL_BRAND, "Samsung")):
            report = self.check(lambda rows, c=col, v=value: rows[R_PM_NEW].update({c: v}))
            self.assertIn("base_fields", self.failed(report), col)

    # ---- formatting ------------------------------------------------------------------ #
    def test_money_format(self):
        for bad in ("800", "$1,200", "$800.", "$ 800", "800 USD", "$8OO"):
            report = self.check(lambda rows, b=bad: rows[R_PM_NEW].update({COL_SWITCHER: b}))
            self.assertIn("money_format", self.failed(report), bad)

    def test_placeholders_and_whitespace(self):
        for bad in ("N/A", "0", "-", "null", "None", "nan"):
            report = self.check(lambda rows, b=bad: rows[R_PM_NEW].update({COL_BOGO_OFFER: b}))
            self.assertIn("placeholders", self.failed(report), bad)
        report = self.check(lambda rows: rows[R_PM_NEW].update({COL_SWITCHER_OFFER_TEXT: rows[R_PM_NEW][COL_SWITCHER_OFFER_TEXT] + " "}))
        self.assertIn("whitespace", self.failed(report))

    def test_paired_columns(self):
        report = self.check(lambda rows: rows[R_PM_NEW].update({COL_SWITCHER_OFFER_TEXT: ""}))
        self.assertIn("paired_columns", self.failed(report))
        report = self.check(lambda rows: rows[R_PM_NEW].update({COL_PWP_GWP_VALUE: ""}))
        self.assertIn("paired_columns", self.failed(report))

    def test_undefined_columns_must_stay_blank(self):
        for col in (COL_BASE, COL_RESELLER_LIST_PRICE, COL_TRADE_IN_VALUE_TYPE):
            report = self.check(lambda rows, c=col: rows[R_PM_NEW].update({c: "Enhanced" if c == COL_TRADE_IN_VALUE_TYPE else "$1.00"}))
            self.assertIn("undefined_columns_blank", self.failed(report), col)

    def test_bom_is_a_warning(self):
        report = self.check(bom=True)
        self.assertTrue(report.ok)
        self.assertIn("encoding", self.warned(report))

    # ---- PDF rules ------------------------------------------------------------------- #
    def test_upgrade_row_with_switcher(self):
        def mutate(rows):
            rows[R_PM_UP].update({COL_SWITCHER: "$800", COL_SWITCHER_OFFER_TEXT: "Get up to $800 when you switch."})
        self.assertIn("switcher_upgrade_blank", self.failed(self.check(mutate)))

    def test_new_line_only_offers_on_upgrade_rows(self):
        def mutate(rows):
            rows[R_PM_UP][COL_UNIQUE_DISCOUNT] = "$628.36"
        report = self.check(mutate)
        self.assertIn("gating_upgrade", self.failed(report))
        self.assertIn("variant_anchoring", self.failed(report))

        def mutate2(rows):
            rows[R_PM_UP].update({COL_SERVICE_LINE_VALUE: "$200", COL_SERVICE_LINE_OFFERS: "Price after additional $200 in credits"})
        self.assertIn("gating_upgrade", self.failed(self.check(mutate2)))

    def test_trade_in_offer_is_allowed_on_upgrade_rows(self):
        self.assertEqual(self.clean[R_17_UP][COL_MAX_TI], "$930")
        self.assertNotIn("gating_upgrade", self.failed(self.check()))

    def test_duplicate_snippet_in_two_columns(self):
        def mutate(rows):
            rows[R_PM_NEW].update({COL_ADDITIONAL_OFFERS: rows[R_PM_NEW][COL_SERVICE_LINE_OFFERS], COL_ADDITIONAL_VALUE: "$200"})
        self.assertIn("non_duplication", self.failed(self.check(mutate)))

    # ---- grounding / recomputed values ----------------------------------------------- #
    def test_wrong_variant_amount(self):
        for wrong in ("$812.36", "$660.36", "$644.36"):
            report = self.check(lambda rows, w=wrong: rows[R_PM_NEW].update({COL_UNIQUE_DISCOUNT: w}))
            self.assertIn("variant_anchoring", self.failed(report), wrong)

    def test_rounded_or_invented_amounts(self):
        report = self.check(lambda rows: rows[R_PM_NEW].update({COL_UNIQUE_DISCOUNT: "$628"}))
        self.assertIn("amount_grounding", self.failed(report))
        report = self.check(lambda rows: rows[R_PM_NEW].update({COL_SWITCHER: "$900"}))
        self.assertIn("amount_grounding", self.failed(report))
        report = self.check(lambda rows: rows[R_PM_NEW].update({COL_SWITCHER_OFFER_TEXT: "Get up to $999 when you switch."}))
        self.assertIn("text_grounding", self.failed(report))

    def test_wrong_trade_in_tier(self):
        report = self.check(lambda rows: rows[R_17_NEW].update({COL_MAX_TI: "$1200"}))   # 18 Pro tier on an iPhone 17
        self.assertIn("max_ti_tier", self.failed(report))
        report = self.check(lambda rows: rows[R_PM_NEW].update({COL_MAX_TI: "$930"}))    # no tier exists for the Pro Max
        self.assertIn("max_ti_tier", self.failed(report))

    def test_finance_values(self):
        self.assertIn("finance_values", self.failed(self.check(lambda rows: rows[R_PM_NEW].update({COL_FINANCE_OFFER_VALUE: "$400"}))))
        self.assertIn("finance_values", self.failed(self.check(lambda rows: rows[R_PM_UP].update({COL_FINANCE_OFFER_VALUE: "$450"}))))
        report = self.check(lambda rows: rows[R_PM_UP].update({COL_FINANCE_OFFER_DESC: "Earn a $200 statement credit welcome bonus plus $240 and $10"}))
        self.assertIn("finance_values", self.failed(report))

    def test_pwp(self):
        report = self.check(lambda rows: rows[R_PM_NEW].update({COL_PWP_GWP_VALUE: "$149.00"}))
        self.assertIn("pwp", self.failed(report))
        report = self.check(lambda rows: rows[R_PM_NEW].update({COL_PWP_GWP_VALUE: "", COL_PWP_GWP_OFFERS: "", COL_CLEANED_PWP_GWP_OFFERS: ""}))
        self.assertIn("pwp", self.failed(report))

    def test_duration(self):
        for bad in ("99", "abc", "36.0"):
            report = self.check(lambda rows, b=bad: rows[R_PM_NEW].update({COL_RATEPLAN_DURATION: b}))
            self.assertIn("duration", self.failed(report), bad)
        report = self.check(lambda rows: rows[R_PM_UP].update({COL_RATEPLAN_DURATION: "36"}))   # no installment offer there
        self.assertIn("duration", self.failed(report))

    # ---- misses are warnings, not failures ------------------------------------------- #
    def test_missing_values_warn_but_do_not_fail(self):
        def mutate(rows):
            rows[R_PM_NEW].update({COL_SWITCHER: "", COL_SWITCHER_OFFER_TEXT: "", COL_UNIQUE_DISCOUNT: "",
                                   COL_SERVICE_LINE_VALUE: "", COL_SERVICE_LINE_OFFERS: "",
                                   COL_FINANCE_OFFER_VALUE: "", COL_FINANCE_OFFER_DESC: "", COL_RATEPLAN_DURATION: ""})
            rows[R_17_NEW].update({COL_MAX_TI: ""})
        report = self.check(mutate)
        self.assertTrue(report.ok, [f for f in report.findings if f[0] == "FAIL"])
        self.assertTrue({"switcher_missing", "variant_anchoring", "service_line", "finance_missing", "max_ti_tier"} <= self.warned(report))

    # ---- command line ---------------------------------------------------------------- #
    def run_cli(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = vo.main(argv)
        return code, buf.getvalue()

    def test_cli_exit_codes_and_output(self):
        self.check()
        code, out = self.run_cli(["--csv", self.path, "--input", INPUT_PATH])
        self.assertEqual(code, 0)
        self.assertIn("SUMMARY:", out)
        self.assertIn("0 failure(s)", out)
        self.assertIn("iPhone 17 Pro Max", out)            # the review table
        self.check(lambda rows: rows[R_PM_UP].update({COL_SWITCHER: "$800", COL_SWITCHER_OFFER_TEXT: "x"}))
        code, out = self.run_cli(["--csv", self.path, "--input", INPUT_PATH, "--no-table"])
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] switcher_upgrade_blank", out)
        code, out = self.run_cli(["--csv", os.path.join(self.tmp, "missing.csv"), "--input", INPUT_PATH])
        self.assertEqual(code, 2)
        self.assertIn("ERROR", out)

    def test_wrong_cell_count_is_reported_not_crashed(self):
        with open(self.path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(OUTPUT_COLUMNS)
            writer.writerow(["only", "three", "cells"])
        report = vo.verify(self.path, INPUT_PATH)
        self.assertIn("row_width", self.failed(report))


if __name__ == "__main__":
    unittest.main()