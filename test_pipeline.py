"""End-to-end tests for pipeline.py and main.py (offline; needs Input.json in this folder).

A fake "careless" LLM answers every row as if it were a New Line, ignoring the
line type. If the pipeline still produces correct Upgrade rows, the Python
safety net (Step 7) is doing its job.

Run:  python -m unittest test_pipeline -v
"""

import csv
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout

import main as cli
from constants import *  # noqa: F401,F403
from llm_client import BaseLLMClient, DryRunClient, LLMPermanentError, LLMTransientError, MockClient
from pipeline import (Checkpoint, check_output_integrity, make_key, run_pipeline,
                      write_csv, write_issues_log)
from prompt_builder import normalize_variant
from validator import SourceIndex, norm

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


def _rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _text(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


class CarelessOracleClient(BaseLLMClient):
    """Test double: reads the real source text and proposes every offer it can find,
    always as if the customer were a New Line (it ignores the simulated line type)."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def _product(self, user):
        facts = json.loads(re.search(r"PRODUCT AND SIMULATED LINE TYPE:\n(\{.*?\})\n\nEvaluate", user, re.S).group(1))
        for p in DATA:
            if p["product_name"] == facts["product_name"] and p["variant"] == facts["variant"]:
                return p
        raise AssertionError("product not found")

    def complete(self, system, user):
        self.calls += 1
        product = self._product(user)
        index = SourceIndex(product)
        src = index.full
        ans = {}

        q = "Get up to an $800 AT&T Visa Reward card when you switch to AT&T or FirstNet and port-in a new line with eligible postpaid service."
        if norm(q) in src:
            ans["switcher"] = {"amount": "$800", "text": "Get up to $800 when you switch to AT&T and port in a new line.", "source_quote": q}

        label = normalize_variant(product["variant"])
        m = re.search(rf"{re.escape(label)} for \$[\d.]+/mo\. for 36 mos\. after up to \$([\d.]+) in bill credits", src)
        if m:
            ans["unique_discount"] = {"amount": f"${m.group(1)}", "source_quote": f"up to ${m.group(1)} in bill credits"}

        for q in ("Price after additional $200 in credits ($5.56/mo.) for new consumer lines only.", "Get $200 off a new line."):
            if norm(q) in src:
                ans["service_line"] = {"text": q, "amount": "$200", "source_quote": q}
                break

        tiers = []
        for mm in re.finditer(r"up to \$(\d+) off", src):
            group = [g for g in index.groups_containing(mm.group(0)) if "trade" in index.groups[g]]
            if group:
                tiers.append(int(mm.group(1)))
        if tiers:
            top = max(tiers)
            ans["max_trade_in"] = {"amount": f"${top}", "source_quote": f"up to ${top} off"}

        comps = []
        for kind, amount, quote in (
            ("one_time_welcome", "$200", "$200 statement credit after spending $1,000 on purchases in your first three months."),
            ("recurring_annual", "$240", "Earn up to $240 in statement credits every year"),
            ("recurring_discount", "$10", "$10 per line monthly discount on your AT&T wireless bill"),
        ):
            if norm(quote) in src:
                comps.append({"kind": kind, "amount": amount, "source_quote": quote})
        if comps:
            ans["finance"] = {"description": "Earn a $200 statement credit welcome bonus plus up to $240 back every year with the AT&T Points Plus Card from Citi and $10 discounts on your AT&T bills.",
                              "components": comps}
        ans["rateplan_duration_months"] = 36
        return json.dumps(ans)


def run_oracle():
    client = CarelessOracleClient()
    result = run_pipeline(DATA, client, progress=lambda s: None)
    return client, result


def by_index(result):
    """-> {product_index: (new_row, upgrade_row)}"""
    return {i: (result.rows[2 * i], result.rows[2 * i + 1]) for i in range(len(DATA))}


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client, cls.result = run_oracle()
        cls.rows = by_index(cls.result)

    def test_shape(self):
        self.assertEqual(len(self.result.rows), 24)
        self.assertEqual(self.client.calls, 24)
        self.assertEqual(check_output_integrity(self.result.rows, 12), [])
        self.assertTrue(self.result.complete)

    def test_upgrade_rows_never_get_new_line_only_offers(self):
        for i, (_, up) in self.rows.items():
            for col in (COL_SWITCHER, COL_SWITCHER_OFFER_TEXT, COL_UNIQUE_DISCOUNT,
                        COL_SERVICE_LINE_VALUE, COL_SERVICE_LINE_OFFERS):
                self.assertIsNone(up[col], f"product {i} Upgrade {col}")

    def test_unique_discount_new_line_only_and_variant_exact(self):
        expected = {0: "$628.36", 1: "$812.36", 2: "$740.36", 3: "$724.36", 4: "$708.36", 7: "$664.36", 8: "$648.36"}
        for i, (new, _) in self.rows.items():
            self.assertEqual(new[COL_UNIQUE_DISCOUNT], expected.get(i), f"product {i}")

    def test_max_trade_in_both_line_types(self):
        expected = {5: "$930", 6: "$930", 9: "$1200", 10: "$1200", 11: "$1200"}
        for i, (new, up) in self.rows.items():
            self.assertEqual(new[COL_MAX_TI], expected.get(i), f"product {i} New")
            self.assertEqual(up[COL_MAX_TI], expected.get(i), f"product {i} Upgrade")

    def test_wrong_trade_in_tier_is_corrected_and_logged(self):
        """The careless LLM proposes $1200 for the iPhone 17; Python corrects it to $930."""
        label_issues = {label: issues for label, issues in self.result.row_issues}
        key = next(k for k in label_issues if "iPhone 17 | 256 GB | New Line" in k)
        self.assertTrue(any("max_trade_in" in i and "$930" in i for i in label_issues[key]))

    def test_pdf_example_rules_for_product_0(self):
        new, up = self.rows[0]
        self.assertEqual((new[COL_SWITCHER], new[COL_SERVICE_LINE_VALUE], new[COL_FINANCE_OFFER_VALUE]),
                         ("$800", "$200", "$450"))
        self.assertEqual((up[COL_SWITCHER], up[COL_UNIQUE_DISCOUNT], up[COL_SERVICE_LINE_VALUE],
                          up[COL_FINANCE_OFFER_VALUE]), (None, None, None, "$250"))
        self.assertNotIn("welcome", up[COL_FINANCE_OFFER_DESC].lower())
        self.assertIsNone(new[COL_MAX_TI]); self.assertIsNone(up[COL_MAX_TI])

    def test_finance_new_vs_upgrade_everywhere(self):
        for i, (new, up) in self.rows.items():
            self.assertEqual(new[COL_FINANCE_OFFER_VALUE], "$450", f"product {i}")
            self.assertEqual(up[COL_FINANCE_OFFER_VALUE], "$250", f"product {i}")

    def test_pwp_on_every_row(self):
        for row in self.result.rows:
            self.assertEqual(row[COL_PWP_GWP_VALUE], "$30.00")

    def test_columns_with_no_evidence_stay_blank(self):
        for row in self.result.rows:
            for col in (COL_BASE, COL_RESELLER_LIST_PRICE, COL_FREE_MEMORY_UPGRADE, COL_BOGO_OFFER,
                        COL_TRADE_IN_VALUE_TYPE, COL_SUBSCRIPTION_BUNDLE_OFFERS, COL_SUBSCRIPTION_BUNDLE_VALUE,
                        COL_ADDITIONAL_OFFERS, COL_ADDITIONAL_VALUE):
                self.assertIsNone(row[col], col)

    def test_duration_only_where_an_installment_offer_survives(self):
        for i, (new, up) in self.rows.items():
            for row in (new, up):
                has_offer = any(row[c] for c in (COL_UNIQUE_DISCOUNT, COL_SERVICE_LINE_VALUE, COL_MAX_TI))
                self.assertEqual(row[COL_RATEPLAN_DURATION] is not None, has_offer, f"product {i} {row[COL_LINE_TYPE]}")

    def test_line_type_order_and_base_fields_identical(self):
        for i, (new, up) in self.rows.items():
            self.assertEqual((new[COL_LINE_TYPE], up[COL_LINE_TYPE]), (LINE_NEW, LINE_UPGRADE))
            for col in (COL_PRODUCT_NAME, COL_VARIANT, COL_MSRP, COL_STOCK, COL_SHIPMENT, COL_PLAN_TYPE, COL_BRAND):
                self.assertEqual(new[col], up[col])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, "cp.jsonl")

    def run_with(self, client, products=DATA[:2]):
        return run_pipeline(products, client, checkpoint=Checkpoint(self.path), progress=lambda s: None)

    def test_second_run_makes_no_calls(self):
        c1 = CarelessOracleClient()
        r1 = self.run_with(c1)
        c2 = CarelessOracleClient()
        r2 = self.run_with(c2)
        self.assertEqual((c1.calls, c2.calls), (4, 0))
        self.assertEqual((r2.live_calls, r2.cached_rows), (0, 4))
        self.assertEqual(r1.rows, r2.rows)  # identical output from cache

    def test_partial_failure_then_resume_only_retries_failed(self):
        good = CarelessOracleClient()
        flaky_answers = [good, None]

        class Flaky(CarelessOracleClient):
            def complete(inner, system, user):
                if "Upgrade" in user.split("simulated_line_type")[1][:20]:
                    raise LLMTransientError("Gave up after 4 attempts: HTTP 503")
                return super().complete(system, user)

        r1 = self.run_with(Flaky())
        self.assertEqual(len(r1.failed), 2)          # both Upgrade rows failed
        self.assertFalse(r1.complete)
        self.assertEqual(len(Checkpoint(self.path).cache), 2)  # only successes saved
        failed_row = r1.rows[1]
        self.assertEqual(failed_row[COL_LINE_TYPE], LINE_UPGRADE)
        self.assertEqual(failed_row[COL_PWP_GWP_VALUE], "$30.00")  # row still complete
        self.assertIsNone(failed_row[COL_FINANCE_OFFER_VALUE])
        c2 = CarelessOracleClient()
        r2 = self.run_with(c2)
        self.assertEqual(c2.calls, 2)                # retried only the failed Upgrade rows
        self.assertTrue(r2.complete)
        self.assertEqual(r2.rows[1][COL_FINANCE_OFFER_VALUE], "$250")

    def test_changed_prompt_invalidates_cache(self):
        a = make_key(0, LINE_NEW, "system v1", "user")
        b = make_key(0, LINE_NEW, "system v2", "user")
        self.assertNotEqual(a, b)
        self.assertNotEqual(make_key(0, LINE_NEW, "s", "u"), make_key(0, LINE_UPGRADE, "s", "u"))
        self.assertNotEqual(make_key(0, LINE_NEW, "s", "u"), make_key(1, LINE_NEW, "s", "u"))

    def test_corrupt_and_garbage_lines_are_skipped(self):
        self.run_with(CarelessOracleClient(), DATA[:1])
        with open(self.path, "a", encoding="utf-8") as f:
            f.write('{"key": "x", "data": {"a": 1\n')   # half-written line
            f.write("not json\n")
            f.write('{"key": "y", "data": "not a dict"}\n')
        cp = Checkpoint(self.path)
        self.assertEqual(len(cp.cache), 2)
        self.assertEqual(cp.skipped_lines, 3)

    def test_validation_changes_apply_to_cached_answers(self):
        """The checkpoint stores RAW LLM output, so validation always re-runs on resume."""
        self.run_with(CarelessOracleClient(), DATA[:1])
        raw = list(Checkpoint(self.path).cache.values())[0]
        self.assertIn("finance", raw)                 # raw answer, not the cleaned row
        self.assertIn("welcome", raw["finance"]["description"].lower())

    def test_dry_run_client_is_not_cached(self):
        out = os.path.join(self.tmp, "o.csv")
        cp = os.path.join(self.tmp, "dry_cp.jsonl")
        with redirect_stdout(io.StringIO()):
            code = cli.main(["--input", INPUT_PATH, "--dry-run", "--output", out, "--checkpoint", cp,
                             "--issues", os.path.join(self.tmp, "i.log")], env_file=None)
        self.assertEqual(code, 0)
        self.assertFalse(os.path.exists(cp))


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        _, self.result = run_oracle()

    def test_csv_roundtrip_exact_headers_blank_cells_no_bom(self):
        path = os.path.join(self.tmp, "o.csv")
        write_csv(self.result.rows, path)
        raw = _bytes(path)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        with open(path, encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], OUTPUT_COLUMNS)
        self.assertEqual(len(rows), 25)
        self.assertTrue(all(len(r) == 29 for r in rows))
        parsed = _rows(path)
        for original, back in zip(self.result.rows, parsed):
            for col in OUTPUT_COLUMNS:
                self.assertEqual(back[col], original[col] or "", col)   # commas/quotes survive, None -> ""
        self.assertEqual(parsed[1][COL_SWITCHER], "")
        self.assertNotIn(",N/A,", raw.decode()); self.assertNotIn(",-,", raw.decode())

    def test_integrity_detects_problems(self):
        rows = [dict(r) for r in self.result.rows]
        self.assertEqual(check_output_integrity(rows, 12), [])
        self.assertTrue(check_output_integrity(rows[:-1], 12))                       # wrong count
        bad = [dict(r) for r in rows]; bad[0][COL_SWITCHER] = "800"
        self.assertTrue(any("not formatted" in p for p in check_output_integrity(bad, 12)))
        bad = [dict(r) for r in rows]; bad[0][COL_BOGO_OFFER] = "N/A"
        self.assertTrue(any("placeholder" in p for p in check_output_integrity(bad, 12)))
        bad = [dict(r) for r in rows]; bad[0][COL_FREE_MEMORY_UPGRADE] = "0"
        self.assertTrue(any("placeholder" in p for p in check_output_integrity(bad, 12)))
        bad = [dict(r) for r in rows]; bad[0], bad[1] = bad[1], bad[0]
        self.assertTrue(any("LINE TYPE" in p for p in check_output_integrity(bad, 12)))
        bad = [dict(r) for r in rows]; bad[2].pop(COL_BASE)
        self.assertTrue(any("schema" in p for p in check_output_integrity(bad, 12)))

    def test_issues_log_written(self):
        path = os.path.join(self.tmp, "i.log")
        write_issues_log(self.result, path, usage_summary="x")
        text = _text(path)
        self.assertIn("ISSUES LOG", text)
        self.assertIn("Upgrade", text)
        self.assertIn("dropped", text)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = os.path.join(self.tmp, "o.csv")
        self.args = ["--input", INPUT_PATH, "--output", self.out,
                     "--checkpoint", os.path.join(self.tmp, "cp.jsonl"), "--issues", os.path.join(self.tmp, "i.log")]

    def call(self, extra, client=None, args=None):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main((args or self.args) + extra, client=client, env_file=None)
        return code, buf.getvalue()

    def test_dry_run_full_pipeline(self):
        code, out = self.call(["--dry-run"])
        self.assertEqual(code, 0)
        rows = _rows(self.out)
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[0][COL_PWP_GWP_VALUE], "$30.00")
        self.assertIn("DRY RUN", out)

    def test_dry_run_default_name_does_not_overwrite_real_output(self):
        self.assertNotEqual(cli.DEFAULT_DRY_OUTPUT, cli.DEFAULT_OUTPUT)

    def test_limit(self):
        code, _ = self.call(["--dry-run", "--limit", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(len(_rows(self.out)), 2)

    def test_api_down_still_writes_complete_csv_and_returns_2(self):
        class Down(BaseLLMClient):
            def complete(self, s, u):
                raise LLMTransientError("Gave up after 4 attempts: HTTP 503")
        code, out = self.call([], client=Down())
        self.assertEqual(code, 2)
        rows = _rows(self.out)
        self.assertEqual(len(rows), 24)
        self.assertIn("Rerun", out)
        self.assertIn("INCOMPLETE ROWS", _text(os.path.join(self.tmp, "i.log")))

    def test_permanent_error_does_not_crash(self):
        code, _ = self.call([], client=MockClient([LLMPermanentError("bad key")] * 30))
        self.assertEqual(code, 2)

    def test_ctrl_c_saves_progress_and_writes_no_csv(self):
        calls = {"n": 0}

        class Interrupt(CarelessOracleClient):
            def complete(inner, s, u):
                calls["n"] += 1
                if calls["n"] == 3:
                    raise KeyboardInterrupt
                return super().complete(s, u)
        code, out = self.call([], client=Interrupt())
        self.assertEqual(code, 130)
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(len(Checkpoint(os.path.join(self.tmp, "cp.jsonl")).cache), 2)
        code2, _ = self.call([], client=CarelessOracleClient())   # resume
        self.assertEqual(code2, 0)
        self.assertEqual(len(_rows(self.out)), 24)

    def test_fresh_discards_saved_progress(self):
        self.call([], client=CarelessOracleClient(), args=self.args + ["--limit", "1"])
        c = CarelessOracleClient()
        self.call(["--fresh", "--limit", "1"], client=c)
        self.assertEqual(c.calls, 2)

    def test_bad_inputs(self):
        missing = ["--input", os.path.join(self.tmp, "nope.json")]
        self.assertEqual(self.call(["--dry-run"] + missing[:0], args=missing + self.args[2:])[0], 1)
        for content in ("{not json", "{}", "[]", "[1, 2]"):
            p = os.path.join(self.tmp, "bad.json")
            with open(p, "w", encoding="utf-8") as handle:
                handle.write(content)
            code, out = self.call(["--dry-run"], args=["--input", p] + self.args[2:])
            self.assertEqual(code, 1, content)
            self.assertIn("ERROR", out)

    def test_missing_api_key_is_a_clean_error(self):
        code, out = self.call([], args=self.args)   # real provider, empty environment, .env ignored
        self.assertEqual(code, 1)
        self.assertIn("GEMINI_API_KEY", out)
        self.assertFalse(os.path.exists(self.out))

    def test_env_file_is_honoured_and_does_not_leak(self):
        env_path = os.path.join(self.tmp, "my.env")
        with open(env_path, "w", encoding="utf-8") as handle:
            handle.write("LLM_PROVIDER=dryrun\n")
        before = dict(os.environ)
        buf = io.StringIO()
        with mock.patch.dict(os.environ):
            with redirect_stdout(buf):
                code = cli.main(self.args, env_file=env_path)   # no client, no --dry-run: .env decides
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", buf.getvalue())
        self.assertEqual(dict(os.environ), before)               # patch.dict restored everything

    def test_env_file_none_never_touches_the_environment(self):
        before = dict(os.environ)
        self.call(["--dry-run"])
        self.assertEqual(dict(os.environ), before)

    def test_failed_rows_show_the_reason_in_the_console(self):
        class Boom(BaseLLMClient):
            def complete(self, s, u):
                raise LLMPermanentError("HTTP 413 from groq: Request too large (Limit 8000, Requested 9000)")
        code, out = self.call(["--limit", "1"], client=Boom())
        self.assertEqual(code, 2)
        self.assertIn("reason: HTTP 413", out)
        self.assertIn("Requested 9000", out)


if __name__ == "__main__":
    unittest.main()