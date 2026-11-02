"""Tests for validator.py and pwp.py (offline; needs Input.json in this folder).

Run:  python -m unittest test_validator -v
"""

import copy
import json
import os
import re
import unittest

from constants import *  # noqa: F401,F403
from llm_client import ExtractionResult
from mapper import map_base_fields
from prompt_builder import collect_source_blocks, normalize_variant
from pwp import extract_pwp
from validator import SourceIndex, build_final_row, validate_and_normalize

_PATHS = ["Input.json", "/mnt/user-data/uploads/Input.json"]
with open(next(p for p in _PATHS if os.path.exists(p)), encoding="utf-8") as _fh:
    DATA = json.load(_fh)

# The exact answer Gemini returned in the live check (product 0, New Line).
LIVE_NEW_LINE = {
    "switcher": {"amount": "$800",
                 "text": "Get up to an $800 AT&T Visa Reward Card per line when you switch to AT&T and port in a new line with eligible postpaid service.",
                 "source_quote": "Get up to an $800 AT&T Visa Reward card when you switch to AT&T or FirstNet and port-in a new line with eligible postpaid service."},
    "unique_discount": {"amount": "$628.36", "source_quote": "up to $628.36 in bill credits applied over the installment term"},
    "service_line": {"text": "Price after additional $200 in credits ($5.56/mo.) for new consumer lines only.",
                     "amount": "$200",
                     "source_quote": "Price after additional $200 in credits ($5.56/mo.) for new consumer lines only."},
    "max_trade_in": {"amount": None, "source_quote": None},
    "trade_in_value_type": None,
    "rateplan_duration_months": 36,
    "subscription_bundle": {"text": None, "amount": None, "source_quote": None},
    "free_memory_upgrade": {"text": None, "source_quote": None},
    "bogo": {"text": None, "source_quote": None},
    "additional_offers": {"text": None, "amount": None, "source_quote": None},
    "finance": {
        "description": "Cardholders can earn a $200 statement credit welcome bonus, up to $240 in annual statement credits, and $10 monthly discounts on AT&T bills.",
        "components": [
            {"kind": "one_time_welcome", "amount": "$200", "source_quote": "$200 statement credit after spending $1,000 on purchases in your first three months."},
            {"kind": "recurring_annual", "amount": "$240", "source_quote": "Earn up to $240 in statement credits every year"},
            {"kind": "recurring_discount", "amount": "$10", "source_quote": "$10 per line monthly discount on your AT&T wireless bill"},
        ]},
    "notes": "Trade-in program does not state a specific maximum dollar amount for this device.",
}


def find_quote(idx, pattern):
    """A verbatim snippet from product idx's real source text."""
    for _, text in collect_source_blocks(DATA[idx]):
        m = re.search(pattern, text, re.I)
        if m:
            return m.group(0)
    raise AssertionError(f"pattern {pattern!r} not found in product {idx}")


def run(idx, line_type, data):
    return validate_and_normalize(DATA[idx], line_type, data)


class LiveOutputTests(unittest.TestCase):
    def test_new_line_matches_expected(self):
        o = run(0, LINE_NEW, LIVE_NEW_LINE)
        c = o.columns
        self.assertEqual(c[COL_SWITCHER], "$800")
        self.assertEqual(c[COL_UNIQUE_DISCOUNT], "$628.36")
        self.assertEqual(c[COL_SERVICE_LINE_VALUE], "$200")
        self.assertIn("$200", c[COL_SERVICE_LINE_OFFERS])
        self.assertEqual(c[COL_FINANCE_OFFER_VALUE], "$450")  # 200 + 240 + 10, added by Python
        self.assertEqual(c[COL_RATEPLAN_DURATION], "36")
        self.assertNotIn(COL_MAX_TI, c)
        self.assertEqual(c[COL_PWP_GWP_VALUE], "$30.00")
        self.assertEqual(o.issues, [])

    def test_same_answer_forced_onto_upgrade_is_corrected_in_code(self):
        """Even if the LLM ignores line-type gating, Python enforces it."""
        o = run(0, LINE_UPGRADE, LIVE_NEW_LINE)
        c = o.columns
        for col in (COL_SWITCHER, COL_SWITCHER_OFFER_TEXT, COL_UNIQUE_DISCOUNT,
                    COL_SERVICE_LINE_VALUE, COL_SERVICE_LINE_OFFERS, COL_RATEPLAN_DURATION):
            self.assertNotIn(col, c, col)
        self.assertEqual(c[COL_FINANCE_OFFER_VALUE], "$250")  # 240 + 10, no welcome bonus
        self.assertNotIn("welcome", c[COL_FINANCE_OFFER_DESC].lower())
        self.assertEqual(c[COL_PWP_GWP_VALUE], "$30.00")  # not line-type dependent
        self.assertTrue(any("switch" in i for i in o.issues))

    def test_correct_upgrade_answer_keeps_model_description(self):
        data = copy.deepcopy(LIVE_NEW_LINE)
        data["finance"]["components"] = data["finance"]["components"][1:]
        data["finance"]["description"] = "Get up to $240 back every year with the AT&T Points Plus Card from Citi and $10 discounts on your AT&T bills."
        c = run(0, LINE_UPGRADE, data).columns
        self.assertEqual(c[COL_FINANCE_OFFER_VALUE], "$250")
        self.assertIn("Citi", c[COL_FINANCE_OFFER_DESC])


class HallucinationGuardTests(unittest.TestCase):
    def test_other_variants_amount_rejected(self):
        for wrong in ("$812.36", "$660.36", "$644.36"):  # 2TB / 256GB / 512GB amounts
            data = {"unique_discount": {"amount": wrong,
                    "source_quote": f"up to {wrong} in bill credits applied over the installment term"}}
            o = run(0, LINE_NEW, data)
            self.assertNotIn(COL_UNIQUE_DISCOUNT, o.columns, wrong)
            self.assertTrue(any("variant" in i for i in o.issues), wrong)

    def test_rounded_amount_rejected(self):
        data = {"unique_discount": {"amount": "$628", "source_quote": "up to $628.36 in bill credits applied over the installment term"}}
        o = run(0, LINE_NEW, data)
        self.assertNotIn(COL_UNIQUE_DISCOUNT, o.columns)
        self.assertTrue(any("rounding" in i for i in o.issues))

    def test_decimals_never_added(self):
        data = {"switcher": {"amount": "$800.00", "source_quote": LIVE_NEW_LINE["switcher"]["source_quote"]}}
        self.assertNotIn(COL_SWITCHER, run(0, LINE_NEW, data).columns)

    def test_invented_quote_rejected(self):
        data = {"switcher": {"amount": "$900", "source_quote": "Get up to $900 when you switch to AT&T"}}
        o = run(0, LINE_NEW, data)
        self.assertNotIn(COL_SWITCHER, o.columns)
        self.assertTrue(any("not found" in i for i in o.issues))

    def test_quote_without_amount_rejected(self):
        data = {"switcher": {"amount": "$500", "source_quote": LIVE_NEW_LINE["switcher"]["source_quote"]}}
        self.assertNotIn(COL_SWITCHER, run(0, LINE_NEW, data).columns)

    def test_bad_money_strings_rejected(self):
        for bad in ("800 dollars", "$8OO", "N/A", "0", "-", 800, 800.0, ["$800"]):
            data = {"switcher": {"amount": bad, "source_quote": LIVE_NEW_LINE["switcher"]["source_quote"]}}
            self.assertNotIn(COL_SWITCHER, run(0, LINE_NEW, data).columns, repr(bad))

    def test_text_with_invented_amount_falls_back_to_quote(self):
        data = copy.deepcopy(LIVE_NEW_LINE)
        data["switcher"]["text"] = "Get up to $999 when you switch."
        c = run(0, LINE_NEW, data).columns
        self.assertEqual(c[COL_SWITCHER_OFFER_TEXT], data["switcher"]["source_quote"])

    def test_duplicate_snippet_goes_to_one_column_only(self):
        q = LIVE_NEW_LINE["service_line"]["source_quote"]
        data = {"service_line": {"text": q, "amount": "$200", "source_quote": q},
                "additional_offers": {"text": q, "amount": "$200", "source_quote": q}}
        c = run(0, LINE_NEW, data).columns
        self.assertIn(COL_SERVICE_LINE_VALUE, c)
        self.assertNotIn(COL_ADDITIONAL_VALUE, c)

    def test_finance_never_double_counts_a_kind(self):
        data = copy.deepcopy(LIVE_NEW_LINE)
        data["finance"]["components"].append(copy.deepcopy(data["finance"]["components"][2]))
        self.assertEqual(run(0, LINE_NEW, data).columns[COL_FINANCE_OFFER_VALUE], "$450")

    def test_trade_in_value_type_needs_a_label_in_source(self):
        data = {"trade_in_value_type": "Enhanced"}
        o = run(0, LINE_NEW, data)
        self.assertNotIn(COL_TRADE_IN_VALUE_TYPE, o.columns)
        self.assertTrue(any("trade_in_value_type" in i for i in o.issues))

    def test_duration_rules(self):
        for bad in (99, True, "36", 0, 3.5, -1):
            data = copy.deepcopy(LIVE_NEW_LINE)
            data["rateplan_duration_months"] = bad
            self.assertNotIn(COL_RATEPLAN_DURATION, run(0, LINE_NEW, data).columns, repr(bad))


class RobustnessTests(unittest.TestCase):
    def test_garbage_llm_payloads_never_crash(self):
        for junk in (None, [], "text", 5, {}, {"switcher": "x", "finance": [1], "unique_discount": 3,
                     "max_trade_in": ["a"], "rateplan_duration_months": {}}):
            o = run(0, LINE_NEW, junk)
            self.assertEqual(o.columns.get(COL_PWP_GWP_VALUE), "$30.00")  # deterministic part survives

    def test_failed_extraction_still_gives_complete_row(self):
        base = map_base_fields(DATA[0])
        base[COL_LINE_TYPE] = LINE_NEW
        row, issues = build_final_row(base, DATA[0], LINE_NEW, ExtractionResult(data=None, error="timeout"))
        self.assertEqual(list(row.keys()), OUTPUT_COLUMNS)
        self.assertEqual(row[COL_PRODUCT_NAME], "Apple iPhone 17 Pro Max")
        self.assertEqual(row[COL_PWP_GWP_VALUE], "$30.00")
        self.assertIsNone(row[COL_SWITCHER])
        self.assertIn("extraction failed", issues[0])

    def test_final_row_keeps_29_columns_in_order(self):
        base = map_base_fields(DATA[0])
        base[COL_LINE_TYPE] = LINE_NEW
        row, _ = build_final_row(base, DATA[0], LINE_NEW, ExtractionResult(data=LIVE_NEW_LINE, attempts=1))
        self.assertEqual(list(row.keys()), OUTPUT_COLUMNS)
        self.assertEqual(row[COL_UNIQUE_DISCOUNT], "$628.36")

    def test_input_not_mutated(self):
        before = json.dumps(DATA[0], sort_keys=True)
        run(0, LINE_NEW, LIVE_NEW_LINE)
        self.assertEqual(before, json.dumps(DATA[0], sort_keys=True))


class RealDataGatingTests(unittest.TestCase):
    """Check the gating rules against EVERY product's real offer wording."""

    BILL_CREDIT = [0, 1, 2, 3, 4, 7, 8]   # "New customers activating new line"
    TRADE_IN = [5, 6, 9, 10, 11]          # "New and existing customers ... or upgrading"

    def test_bill_credit_offers_are_new_line_only(self):
        for i in self.BILL_CREDIT:
            src = SourceIndex(DATA[i])
            self.assertEqual(src.line_gate("OFFER 0"), (True, False), f"product {i}")

    def test_trade_in_offers_open_to_both(self):
        for i in self.TRADE_IN:
            src = SourceIndex(DATA[i])
            self.assertEqual(src.line_gate("OFFER 0"), (True, True), f"product {i}")

    def test_200_off_new_line_is_new_only(self):
        for i in self.TRADE_IN:
            self.assertEqual(SourceIndex(DATA[i]).line_gate("OFFER 2"), (True, False), f"product {i}")

    def test_unique_discount_passes_for_each_own_variant_and_only_that(self):
        """For every product with bill-credit text: its own amount passes, every other size's fails."""
        checked = 0
        for i in self.BILL_CREDIT:
            variant = normalize_variant(DATA[i]["variant"])
            full = SourceIndex(DATA[i]).full
            by_label = {}
            for m in re.finditer(r"(\d+(?:\.\d+)?(?:tb|gb)) for \$[\d.]+/mo\. for \d+ mos\. after up to \$([\d.]+) in bill credits", full):
                by_label.setdefault(m.group(1), m.group(2))
            self.assertIn(variant, by_label, f"product {i} variant {variant} not in text")
            for label, amount in by_label.items():
                data = {"unique_discount": {"amount": f"${amount}",
                        "source_quote": f"up to ${amount} in bill credits applied over the installment term"}}
                got = run(i, LINE_NEW, data).columns.get(COL_UNIQUE_DISCOUNT)
                if label == variant:
                    self.assertEqual(got, f"${amount}", f"product {i} own amount")
                    checked += 1
                else:
                    self.assertIsNone(got, f"product {i} {variant} accepted {label}'s ${amount}")
        self.assertGreaterEqual(checked, len(self.BILL_CREDIT))

    def test_max_trade_in_kept_for_both_line_types(self):
        cases = {5: ("$930", r"up to \$930 off"), 9: ("$1200", r"up to \$1,200 off")}
        for idx, (expected, pattern) in cases.items():
            quote = find_quote(idx, pattern)
            amount = "$1,200" if idx == 9 else "$930"
            data = {"max_trade_in": {"amount": amount, "source_quote": quote}}
            for lt in LINE_TYPES:
                self.assertEqual(run(idx, lt, data).columns.get(COL_MAX_TI), expected, f"{idx} {lt}")

    def test_200_off_new_line_blocked_for_upgrade(self):
        quote = find_quote(5, r"Get \$200 off a new line\.")
        data = {"service_line": {"text": quote, "amount": "$200", "source_quote": quote}}
        self.assertEqual(run(5, LINE_NEW, data).columns.get(COL_SERVICE_LINE_VALUE), "$200")
        self.assertNotIn(COL_SERVICE_LINE_VALUE, run(5, LINE_UPGRADE, data).columns)

    def test_pwp_present_on_all_12_products(self):
        for i, p in enumerate(DATA):
            pwp, issues = extract_pwp(p)
            self.assertEqual(pwp["value"], "$30.00", f"product {i}")
            self.assertIn("Was $179.00. Now $149.00.", pwp["raw"])
            self.assertEqual(pwp["cleaned"], "Apple AirPods 4 with Active Noise Cancellation for $149.00 (was $179.00)")


class TradeInTierTests(unittest.TestCase):
    def propose(self, idx, amount_digits, line_type=LINE_NEW):
        quote = f"up to ${amount_digits} off"
        data = {"max_trade_in": {"amount": f"${amount_digits}", "source_quote": quote}}
        return run(idx, line_type, data)

    def test_iphone17_correct_tier_kept(self):
        self.assertEqual(self.propose(5, "930").columns[COL_MAX_TI], "$930")

    def test_iphone17_gets_930_even_if_llm_picks_the_18_pro_tier(self):
        o = self.propose(5, "1200")
        self.assertEqual(o.columns[COL_MAX_TI], "$930")
        self.assertTrue(any("not a tier for Apple iPhone 17" in i for i in o.issues))

    def test_lower_applicable_tier_is_raised_to_the_maximum(self):
        for low in ("500", "350"):
            o = self.propose(6, low)
            self.assertEqual(o.columns[COL_MAX_TI], "$930", low)
            self.assertTrue(any("not the maximum tier" in i for i in o.issues), low)

    def test_18_pro_max_gets_1200(self):
        for idx in (9, 10, 11):
            for proposed in ("1200", "930", "500"):
                self.assertEqual(self.propose(idx, proposed).columns[COL_MAX_TI], "$1200", (idx, proposed))

    def test_both_line_types_get_the_corrected_tier(self):
        for lt in LINE_TYPES:
            self.assertEqual(self.propose(5, "1200", lt).columns[COL_MAX_TI], "$930", lt)

    def test_model_matching_is_exact(self):
        from validator import _Ctx, applicable_trade_in_tiers
        # iPhone 17 must not inherit the "iPhone 17 Pro"/"17e" wording, and vice versa
        tiers17, _ = applicable_trade_in_tiers(_Ctx(DATA[5], LINE_NEW))
        self.assertEqual(sorted(tiers17, key=float, reverse=True), ["930", "500", "350"])
        tiers_pm, _ = applicable_trade_in_tiers(_Ctx(DATA[9], LINE_NEW))
        self.assertEqual(sorted(tiers_pm, key=float, reverse=True), ["1200", "930", "500", "350"])
        for idx in (0, 1, 2, 3, 4, 7, 8):
            self.assertEqual(applicable_trade_in_tiers(_Ctx(DATA[idx], LINE_NEW))[0], {}, idx)

    def test_other_model_tier_dropped_when_text_names_only_other_models(self):
        import copy as _copy
        product = _copy.deepcopy(DATA[5])
        product["product_name"] = "Apple iPhone 99"   # a model the tiers never mention (and no 'all devices')
        # remove the generic "all eligible devices" tiers so only model-specific ones remain
        for offer in product["offers"]:
            details = offer.get("full_details") or {}
            for key in ("modal_raw_text",):
                if details.get(key):
                    details[key] = details[key].replace("all eligible devices", "some devices")
            for card in details.get("cards") or []:
                inner = card.get("full_details") or {}
                if inner.get("raw_text"):
                    inner["raw_text"] = inner["raw_text"].replace("all eligible devices", "some devices")
        quote = find_quote(5, r"up to \$930 off")
        data = {"max_trade_in": {"amount": "$930", "source_quote": quote}}
        o = validate_and_normalize(product, LINE_NEW, data)
        self.assertNotIn(COL_MAX_TI, o.columns)
        self.assertTrue(any("name other models" in i for i in o.issues))


class PwpUnitTests(unittest.TestCase):
    def acc(self, badge="On sale", cur="+$149.00", orig="$179.00", hidden="Was $179.00. Now $149.00.", title="Item"):
        return {"title": title, "badge": badge, "pricing": {"current_price": cur, "original_price": orig, "hidden_description": hidden}}

    def run_pwp(self, *items):
        return extract_pwp({"accessories_offers": list(items)})

    def test_basic_and_exact_decimals(self):
        pwp, _ = self.run_pwp(self.acc(cur="+$19.99", orig="$29.99", hidden="Was $29.99. Now $19.99."))
        self.assertEqual(pwp["value"], "$10.00")

    def test_multiple_items_are_added(self):
        pwp, _ = self.run_pwp(self.acc(title="A"), self.acc(title="B", cur="+$10.50", orig="$20.25", hidden="Was $20.25. Now $10.50."))
        self.assertEqual(pwp["value"], "$39.75")
        self.assertIn("A", pwp["cleaned"]); self.assertIn("B", pwp["cleaned"])

    def test_not_qualifying(self):
        for item in (self.acc(badge="New!"), self.acc(badge=None), self.acc(orig="$100.00", cur="+$149.00", hidden=None),
                     self.acc(cur="+$149.00", orig="$149.00"), self.acc(cur=None), self.acc(orig="abc"),
                     self.acc(hidden="Was $1.00. Now $2.00.")):
            pwp, _ = self.run_pwp(item)
            self.assertIsNone(pwp["value"], item)

    def test_bad_payloads(self):
        for product in ({}, {"accessories_offers": None}, {"accessories_offers": ["x", 1, None]}, None):
            self.assertIsNone(extract_pwp(product)[0]["value"])


if __name__ == "__main__":
    unittest.main()
    