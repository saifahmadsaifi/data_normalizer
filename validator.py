"""Validation layer (Step 7): the Python safety net behind the LLM.

The LLM proposes values; nothing reaches the CSV unless Python can prove it:

  1. Money format      - "$" + digits exactly as written; no rounding.
  2. Grounding         - every `source_quote` must exist verbatim (case/whitespace/
                         quote-style insensitive) inside ONE source block, and the
                         amount must appear in that quote (exact digits, so a
                         rounded "$628" is rejected for "$628.36").
  3. Variant anchoring - UNIQUE DISCOUNT must sit right after the product's own
                         storage label ("1TB"), not another size's label.
  4. Line-type gating  - re-derived from the offer's own wording ("new customers
                         activating new line", "for new consumer lines only", ...).
                         Switcher is always New Line only.
  5. Finance           - Python (not the LLM) picks valid components per line type,
                         keeps one per kind, and adds them with Decimal.
  6. No duplication    - one source snippet is mapped to only one column.
  7. Free text         - any dollar amount inside model-written text must exist in the source.
  8. Trade-in tiers    - a tier counts for a product only if its clause names that model (or says
                         "all eligible devices"); the highest applicable tier is the Max TI, and a
                         lower/other-model tier chosen by the LLM is corrected in code.

Anything rejected becomes blank, and a human-readable reason is recorded in
`issues` (written to a log by the pipeline) so every drop is auditable.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from constants import (
    COL_ADDITIONAL_OFFERS, COL_ADDITIONAL_VALUE, COL_BOGO_OFFER,
    COL_FINANCE_OFFER_DESC, COL_FINANCE_OFFER_VALUE, COL_FREE_MEMORY_UPGRADE,
    COL_MAX_TI, COL_PWP_GWP_OFFERS, COL_CLEANED_PWP_GWP_OFFERS, COL_PWP_GWP_VALUE,
    COL_RATEPLAN_DURATION, COL_SERVICE_LINE_OFFERS, COL_SERVICE_LINE_VALUE,
    COL_SUBSCRIPTION_BUNDLE_OFFERS, COL_SUBSCRIPTION_BUNDLE_VALUE, COL_SWITCHER,
    COL_SWITCHER_OFFER_TEXT, COL_TRADE_IN_VALUE_TYPE, COL_UNIQUE_DISCOUNT,
    LINE_NEW, LINE_UPGRADE,
)
from llm_client import ExtractionResult
from mapper import format_money
from prompt_builder import collect_source_blocks, normalize_variant
from pwp import extract_pwp

Row = Dict[str, Optional[str]]

# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
_SMART = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
                        "\u2013": "-", "\u2014": "-", "\u00a0": " "})


def norm(text: Any) -> str:
    """Lenient normal form for comparing quotes with the source."""
    if not isinstance(text, str):
        return ""
    text = text.translate(_SMART).replace("\u00ae", "").replace("\u2122", "")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 1,200 -> 1200
    return re.sub(r"\s+", " ", text).strip().lower()


def digits_of(amount: str) -> str:
    """'$1,200' / '$628.36' -> '1200' / '628.36'."""
    return amount.lstrip("$").replace(",", "")


def amount_regex(digits: str) -> "re.Pattern[str]":
    """Match the number exactly: not part of a longer number, not followed by more decimals."""
    return re.compile(r"(?<![\d.])" + re.escape(digits) + r"(?!\d)(?!\.\d)")


def amounts_in(text: str) -> List[str]:
    """All dollar amounts written in a piece of text, as digit strings."""
    return [m.replace(",", "") for m in re.findall(r"\$\s?(\d+(?:,\d{3})*(?:\.\d+)?)", text or "")]


_STORAGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)")
_EXISTING_RE = re.compile(r"existing customers?|upgrading an existing line")
_MONTHS_RE = r"(?:-?\s*)(?:mos?\b|months?\b|-mo\b)"


# --------------------------------------------------------------------------- #
# Source index
# --------------------------------------------------------------------------- #
class SourceIndex:
    """Normalised view of a product's source text, grouped by offer."""

    def __init__(self, product: Dict[str, Any]) -> None:
        self.blocks: List[Tuple[str, str, str]] = []  # (group, label, normalised text)
        groups: Dict[str, List[str]] = {}
        for label, text in collect_source_blocks(product):
            parts = label.split(" ")
            group = " ".join(parts[:2])  # "OFFER 2" / "PRICING OPTION"
            if label.startswith("PRICING OPTION"):
                group = label  # each pricing option is its own group
            normed = norm(text)
            self.blocks.append((group, label, normed))
            groups.setdefault(group, []).append(normed)
        self.groups = {g: "\n".join(t) for g, t in groups.items()}
        self.full = "\n".join(b[2] for b in self.blocks)

    def groups_containing(self, quote: str) -> List[str]:
        q = norm(quote)
        return sorted({g for g, _, text in self.blocks if q and q in text})

    def line_gate(self, group: str) -> Tuple[bool, bool]:
        """(New Line allowed, Upgrade allowed) from the offer's own wording."""
        text = self.groups.get(group, "")
        if _EXISTING_RE.search(text):
            return True, True
        # New-only wording ("new customers activating new line", "for new consumer
        # lines only") OR no statement at all: Upgrade eligibility is not proven.
        return True, False


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class ValidationOutcome:
    columns: Dict[str, Optional[str]] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


class _Ctx:
    """Shared state for validating one (product, line type)."""

    def __init__(self, product: Dict[str, Any], line_type: str) -> None:
        self.product = product
        self.line_type = line_type
        self.variant_key = normalize_variant(product.get("variant"))
        self.src = SourceIndex(product)
        self.issues: List[str] = []
        self.used_quotes: Dict[str, str] = {}

    def drop(self, name: str, reason: str) -> None:
        self.issues.append(f"{name}: dropped - {reason}")


# --------------------------------------------------------------------------- #
# Generic checks
# --------------------------------------------------------------------------- #
def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _check_amount(ctx: _Ctx, name: str, amount: Any, quote: Any,
                  require_variant: bool = False, require_word: Optional[str] = None,
                  gate_line: bool = True) -> Optional[Tuple[str, str]]:
    """Validate (amount, quote). Returns (formatted '$x', normalised quote) or None."""
    amount_s, quote_s = _as_str(amount), _as_str(quote)
    if amount_s is None:
        return None  # nothing proposed: not an issue
    if quote_s is None:
        ctx.drop(name, "no source_quote supplied")
        return None
    money = format_money(amount_s)
    if money is None:
        ctx.drop(name, f"amount {amount_s!r} is not a clean dollar value")
        return None
    nq = norm(quote_s)
    groups = ctx.src.groups_containing(quote_s)
    if not groups:
        ctx.drop(name, "source_quote not found verbatim in a single source block")
        return None
    if not amount_regex(digits_of(money)).search(nq):
        ctx.drop(name, f"amount {money} does not appear exactly in its source_quote (possible rounding)")
        return None
    if require_word and not any(require_word in ctx.src.groups[g] for g in groups):
        ctx.drop(name, f"offer text never mentions '{require_word}'")
        return None
    if require_variant and not _variant_anchored(ctx, digits_of(money)):
        ctx.drop(name, f"{money} is not listed against the {ctx.product.get('variant')} variant")
        return None
    if gate_line and ctx.line_type == LINE_UPGRADE:
        if not all(ctx.src.line_gate(g)[1] for g in groups):
            ctx.drop(name, "offer is not open to an Upgrade (new-line wording)")
            return None
    return money, nq


def _variant_anchored(ctx: _Ctx, digits: str) -> bool:
    """True if the amount appears right after this product's own storage label."""
    if not ctx.variant_key:
        return False
    pattern = amount_regex(digits)
    for match in pattern.finditer(ctx.src.full):
        window = ctx.src.full[max(0, match.start() - 300):match.start()]
        labels = _STORAGE_RE.findall(window)
        if labels and f"{labels[-1][0]}{labels[-1][1]}" == ctx.variant_key:
            return True
    return False


def _claim(ctx: _Ctx, name: str, nquote: Optional[str]) -> bool:
    """Non-duplication: a source snippet may feed only ONE column."""
    if nquote is None:
        return True
    owner = ctx.used_quotes.get(nquote)
    if owner and owner != name:
        ctx.drop(name, f"same source snippet already used for {owner}")
        return False
    ctx.used_quotes[nquote] = name
    return True


def _free_text(ctx: _Ctx, name: str, text: Any, fallback: Optional[str]) -> Optional[str]:
    """Model-written text is kept only if every $ amount in it exists in the source."""
    clean = _as_str(text)
    if clean is None:
        return fallback
    source_amounts = set(amounts_in(ctx.src.full))
    unknown = [a for a in amounts_in(clean) if a not in source_amounts]
    if unknown:
        ctx.issues.append(f"{name}: text had amounts not in source {unknown}; used source quote instead")
        return fallback
    return clean


# --------------------------------------------------------------------------- #
# Trade-in tier applicability
# --------------------------------------------------------------------------- #
_TIER_RE = re.compile(r"up to \$(\d+(?:\.\d+)?) off")
_SENTENCE_CUT = re.compile(r"[.\n]\s*")
_ALL_DEVICES = "all eligible devices"


def _model_regex(product: Dict[str, Any]) -> Optional["re.Pattern[str]"]:
    """'Apple iPhone 17 Pro' -> pattern for 'iphone 17 pro' that is NOT 'iphone 17 pro max' / 'iphone 17e'."""
    name = norm(product.get("product_name"))
    name = re.sub(r"^apple\s+", "", name)
    if not name:
        return None
    words = r"\s+".join(re.escape(w) for w in name.split())
    return re.compile(words + r"(?!\w)(?!\s+(?:pro|max|plus|air|mini)\b)")


def applicable_trade_in_tiers(ctx: "_Ctx") -> Tuple[Dict[str, str], bool]:
    """Find every 'up to $X off' trade-in tier and keep those that apply to this product.

    Returns ({amount_digits: verbatim_quote}, any_tier_names_a_model).
    A tier applies when its clause (the sentence start before it, up to the next tier or
    260 characters after it) names this product's model, or says 'all eligible devices'.
    """
    model = _model_regex(ctx.product)
    applicable: Dict[str, str] = {}
    names_a_model = False
    for group, text in ctx.src.groups.items():
        if "trade" not in text:
            continue
        matches = list(_TIER_RE.finditer(text))
        for i, m in enumerate(matches):
            nxt = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            after = text[m.end(): min(nxt, m.end() + 260)]
            before = _SENTENCE_CUT.split(text[max(0, m.start() - 80): m.start()])[-1]
            clause = before + m.group(0) + after
            if re.search(r"iphone \d", clause):
                names_a_model = True
            if _ALL_DEVICES in after or (model is not None and model.search(clause)):
                applicable.setdefault(m.group(1), m.group(0))
    return applicable, names_a_model


# --------------------------------------------------------------------------- #
# Field handlers
# --------------------------------------------------------------------------- #
def _simple_offer(ctx, out, data, key, name, text_col, value_col=None, text_key="text", **kw):
    block = _as_dict(data.get(key))
    text, amount = _as_str(block.get(text_key)), block.get("amount")
    if text is None and _as_str(amount) is None:
        return
    quote = _as_str(block.get("source_quote"))
    checked_money = None
    if _as_str(amount) is not None:
        checked = _check_amount(ctx, name, amount, quote, **kw)
        if checked is None:
            return
        checked_money, nq = checked
        if not _claim(ctx, name, nq):
            return
    else:  # text-only offer (free memory upgrade, BOGO): quote must still be real
        if quote is None or not ctx.src.groups_containing(quote):
            ctx.drop(name, "no verifiable source_quote for text-only offer")
            return
        groups = ctx.src.groups_containing(quote)
        if ctx.line_type == LINE_UPGRADE and not all(ctx.src.line_gate(g)[1] for g in groups):
            ctx.drop(name, "offer is not open to an Upgrade (new-line wording)")
            return
        if not _claim(ctx, name, norm(quote)):
            return
    out[text_col] = _free_text(ctx, name, text, quote)
    if value_col:
        out[value_col] = checked_money


def _handle_switcher(ctx, out, data):
    block = _as_dict(data.get("switcher"))
    if _as_str(block.get("amount")) is None:
        return
    if ctx.line_type != LINE_NEW:
        ctx.drop("switcher", "Upgrade lines cannot switch carriers")
        return
    checked = _check_amount(ctx, "switcher", block.get("amount"), block.get("source_quote"),
                            require_word="switch")
    if checked is None or not _claim(ctx, "switcher", checked[1]):
        return
    out[COL_SWITCHER] = checked[0]
    out[COL_SWITCHER_OFFER_TEXT] = _free_text(ctx, "switcher", block.get("text"),
                                             _as_str(block.get("source_quote")))


def _handle_unique(ctx, out, data):
    block = _as_dict(data.get("unique_discount"))
    checked = _check_amount(ctx, "unique_discount", block.get("amount"),
                            block.get("source_quote"), require_variant=True)
    if checked and _claim(ctx, "unique_discount", checked[1]):
        out[COL_UNIQUE_DISCOUNT] = checked[0]


def _handle_service(ctx, out, data):
    block = _as_dict(data.get("service_line"))
    checked = _check_amount(ctx, "service_line", block.get("amount"), block.get("source_quote"))
    if checked and _claim(ctx, "service_line", checked[1]):
        out[COL_SERVICE_LINE_VALUE] = checked[0]
        text = _as_str(block.get("text"))
        out[COL_SERVICE_LINE_OFFERS] = text if text and norm(text) in ctx.src.full \
            else _as_str(block.get("source_quote"))


def _handle_trade_in(ctx, out, data):
    block = _as_dict(data.get("max_trade_in"))
    checked = _check_amount(ctx, "max_trade_in", block.get("amount"),
                            block.get("source_quote"), require_word="trade")
    if checked:
        amount, quote = checked
        tiers, names_a_model = applicable_trade_in_tiers(ctx)
        if tiers:
            top = max(tiers, key=Decimal)
            if digits_of(amount) not in tiers:
                ctx.issues.append(f"max_trade_in: {amount} is not a tier for {ctx.product.get('product_name')}; "
                                  f"using its highest tier ${top}")
            elif Decimal(digits_of(amount)) < Decimal(top):
                ctx.issues.append(f"max_trade_in: {amount} is not the maximum tier for "
                                  f"{ctx.product.get('product_name')}; using ${top}")
            amount, quote = format_money(top), norm(tiers[top])
        elif names_a_model:
            ctx.drop("max_trade_in", f"trade-in tiers in the source name other models, none for "
                                     f"{ctx.product.get('product_name')}")
            amount = None
        if amount and _claim(ctx, "max_trade_in", quote):
            out[COL_MAX_TI] = amount
    label = _as_str(data.get("trade_in_value_type"))
    if label:
        if label in ("Enhanced", "Base") and label.lower() in ctx.src.full:
            out[COL_TRADE_IN_VALUE_TYPE] = label
        else:
            ctx.drop("trade_in_value_type", f"{label!r} is not labelled that way in the source")


def _handle_finance(ctx, out, data):
    block = _as_dict(data.get("finance"))
    components = block.get("components")
    if not isinstance(components, list) or not components:
        return
    allowed = {"recurring_annual", "recurring_discount"}
    if ctx.line_type == LINE_NEW:
        allowed.add("one_time_welcome")
    chosen: Dict[str, str] = {}
    for comp in components:
        comp = _as_dict(comp)
        kind = comp.get("kind")
        if kind not in ("one_time_welcome", "recurring_annual", "recurring_discount"):
            ctx.drop("finance", f"unknown component kind {kind!r}")
            continue
        if kind not in allowed:
            ctx.issues.append(f"finance: {kind} excluded for {ctx.line_type}")
            continue
        if kind in chosen:
            ctx.issues.append(f"finance: second {kind} component ignored (no double counting)")
            continue
        checked = _check_amount(ctx, f"finance.{kind}", comp.get("amount"),
                                comp.get("source_quote"), gate_line=False)
        if checked:
            chosen[kind] = checked[0]
    if not chosen:
        return
    total = sum((Decimal(digits_of(v)) for v in chosen.values()), Decimal(0))
    out[COL_FINANCE_OFFER_VALUE] = f"${format(total, 'f')}"
    out[COL_FINANCE_OFFER_DESC] = _finance_description(ctx, block.get("description"), chosen)


def _finance_description(ctx: _Ctx, text: Any, chosen: Dict[str, str]) -> str:
    """Use the model's sentence only if it is consistent with the validated components."""
    bad_for_upgrade = ctx.line_type == LINE_UPGRADE and re.search(r"welcome|sign-?up|bonus", str(text or ""), re.I)
    clean = None if bad_for_upgrade else _free_text(ctx, "finance", text, None)
    if clean is None:
        parts = []
        if "one_time_welcome" in chosen:
            parts.append(f"Earn a {chosen['one_time_welcome']} statement credit welcome bonus")
        if "recurring_annual" in chosen:
            parts.append(f"up to {chosen['recurring_annual']} back every year")
        if "recurring_discount" in chosen:
            parts.append(f"{chosen['recurring_discount']} discounts on your bills")
        clean = ", ".join(parts)
        ctx.issues.append("finance: description rebuilt from validated components")
    return clean


def _handle_duration(ctx, out, data):
    value = data.get("rateplan_duration_months")
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 120:
        ctx.drop("rateplan_duration_months", f"{value!r} is not a sensible whole number of months")
        return
    if not re.search(rf"(?<!\d){value}\s*{_MONTHS_RE}", ctx.src.full):
        ctx.drop("rateplan_duration_months", f"'{value} months' is not stated in the source")
        return
    if not any(out.get(c) for c in (COL_UNIQUE_DISCOUNT, COL_SERVICE_LINE_VALUE, COL_MAX_TI)):
        ctx.drop("rateplan_duration_months", "no installment offer is valid for this line type")
        return
    out[COL_RATEPLAN_DURATION] = str(value)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def validate_and_normalize(product: Dict[str, Any], line_type: str,
                           llm_data: Any) -> ValidationOutcome:
    """Validate one LLM answer and return only the columns Python can prove."""
    ctx = _Ctx(product, line_type)
    out: Dict[str, Optional[str]] = {}
    data = llm_data if isinstance(llm_data, dict) else {}
    if llm_data is not None and not isinstance(llm_data, dict):
        ctx.issues.append("llm: response was not a JSON object; all LLM fields blank")

    # Priority order decides who keeps a snippet that two fields both claim.
    _handle_switcher(ctx, out, data)
    _handle_unique(ctx, out, data)
    _handle_service(ctx, out, data)
    _handle_trade_in(ctx, out, data)
    _simple_offer(ctx, out, data, "subscription_bundle", "subscription_bundle",
                  COL_SUBSCRIPTION_BUNDLE_OFFERS, COL_SUBSCRIPTION_BUNDLE_VALUE)
    _simple_offer(ctx, out, data, "free_memory_upgrade", "free_memory_upgrade", COL_FREE_MEMORY_UPGRADE)
    _simple_offer(ctx, out, data, "bogo", "bogo", COL_BOGO_OFFER)
    _simple_offer(ctx, out, data, "additional_offers", "additional_offers",
                  COL_ADDITIONAL_OFFERS, COL_ADDITIONAL_VALUE)
    _handle_finance(ctx, out, data)
    _handle_duration(ctx, out, data)

    pwp, pwp_issues = extract_pwp(product)  # deterministic, independent of the LLM
    ctx.issues.extend(pwp_issues)
    out[COL_PWP_GWP_OFFERS], out[COL_CLEANED_PWP_GWP_OFFERS], out[COL_PWP_GWP_VALUE] = (
        pwp["raw"], pwp["cleaned"], pwp["value"])

    return ValidationOutcome(columns={k: v for k, v in out.items() if v is not None}, issues=ctx.issues)


def build_final_row(base_row: Row, product: Dict[str, Any], line_type: str,
                    extraction: ExtractionResult) -> Tuple[Row, List[str]]:
    """Merge validated LLM fields into a base row. A failed LLM call still yields
    a complete row (base fields + deterministic PWP/GWP) plus an issue note."""
    row = dict(base_row)
    outcome = validate_and_normalize(product, line_type, extraction.data if extraction.ok else None)
    issues = list(outcome.issues)
    if not extraction.ok:
        issues.insert(0, f"llm: extraction failed ({extraction.error}); LLM-derived fields left blank")
    row.update(outcome.columns)
    return row, issues
