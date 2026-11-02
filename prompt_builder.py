"""Prompt builder (Step 5): designs what the LLM sees for ONE product and ONE line type.

Design principles (all taken from the assignment PDF):
  * Use only the text supplied. No outside knowledge, no guessing, no rounding.
  * Strict variant matching: anchor discounts to the product's own storage size.
  * Line-type gating: an offer whose eligibility excludes the simulated line type
    must produce null for that line type.
  * Never sum values across columns. One value/snippet -> one column only.
  * Every amount must come with a verbatim `source_quote`, so the Python
    validation layer (Step 7) can prove the number really exists in the source.
  * Arithmetic is NOT done by the LLM: finance is returned as components and
    Python adds them up with Decimal.

This module only BUILDS prompts. It makes no API call (that is Step 6).
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from constants import LINE_NEW, LINE_UPGRADE

# --------------------------------------------------------------------------- #
# Variant helpers
# --------------------------------------------------------------------------- #
_VARIANT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)", re.IGNORECASE)


def normalize_variant(variant: Any) -> str:
    """'1 TB' / '1TB' / '1tb' -> '1tb'; '256 GB' -> '256gb'. Empty string if unparseable."""
    match = _VARIANT_RE.search(str(variant or ""))
    return f"{match.group(1)}{match.group(2).lower()}" if match else ""


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = r"""You are a precise data-extraction engine. You convert messy retailer promotional text
into a strict JSON object. You are given ONE product, ONE simulated line type, and the raw source text.
Output JSON only: no prose, no markdown, no code fences.

=== ABSOLUTE RULES ===
1. SOURCE ONLY. Use only the text in the SOURCE section. Never use outside knowledge, never guess a
   missing price, never infer an amount that is not written. If unsure, return null.
2. EXACT NUMBERS. Copy dollar amounts exactly as written, digit for digit. Never round, never drop or
   add decimals ("$660.36" stays "$660.36", "$800" stays "$800"). Remove thousands commas only
   ("$1,200" -> "$1200"). Always keep the leading "$".
3. PROOF. Every non-null amount must be paired with a `source_quote`: a short VERBATIM snippet
   copied from the SOURCE that contains that amount. If you cannot quote it, return null for it.
4. NO DUPLICATION. A single dollar value or text snippet may be placed in ONLY ONE field.
   Never add values from different fields together.
5. NULL, NOT PLACEHOLDERS. If an offer does not exist or the line type disqualifies the customer,
   use JSON null. Never write "N/A", "0", "-", or "none".
6. IGNORE spend requirements ("after spending $1,000"), fees, APRs, tax, down payments, monthly
   installment payment amounts, "Next Up Anytime" charges, and prices stated only for "retailer
   partner and business customers" when computing values. They are not offers for this customer.

=== STRICT VARIANT MATCHING ===
The product has a storage variant (e.g. "1 TB"). Treat "1 TB", "1TB" and "1tb" as identical; the same
for "256 GB"/"256GB". When the source lists several storage sizes with different amounts, take ONLY the
amount stated for this product's variant. Never use another size's amount. If this product's
variant is not listed, return null. Titles such as "Get iPhone 17 Pro Max 256GB for $9.43/mo."
describe the 256GB model only and must not be applied to other variants. Device model names must
also match (iPhone 17 Pro is not iPhone 17 Pro Max).

=== LINE-TYPE GATING (global eligibility gate) ===
You are told the simulated LINE TYPE: "New Line" or "Upgrade" (an existing customer upgrading an
already-active line). Read each offer's eligibility wording as a strict gate:
  - "Eligibility: New customers activating new line", "requires new line", "new line required",
    "for new consumer lines only", "port-in of a new line", "requires activation of a new line"
    => valid for New Line ONLY. For Upgrade, every value from that offer is null.
  - "existing customers", "upgrading an existing line", "New and existing customers activating a new
    line or upgrading an existing line" => valid for BOTH line types.
  - If the text does not state that the offer is open to the simulated line type, treat it as NOT
    eligible and return null.
Evaluate ONLY for the line type you are given. Do not mention the other line type.

=== FIELD DEFINITIONS ===
switcher: Offer triggered by switching carriers / porting a number in ("up to $800/line to break your
  contract"). A customer upgrading a line already on this carrier cannot switch, so Upgrade => null.
  `amount` = the stated maximum (e.g. "$800"). `text` = one clean sentence describing the offer, in
  your own words, using only facts from the source (e.g. "Get up to $800/line when you switch to
  AT&T and get a reward card to pay off your phone.").

unique_discount: The device-specific bill-credit discount for THIS variant (e.g. "up to $660.36 in
  bill credits" for 256GB). Use only the amount for this variant. Do NOT include the additional
  service-line credit ($200 / $5.56 per month) here, and do NOT include trade-in credits here.

service_line: The additional credit tied to the service line (e.g. "Price after additional $200 in
  credits ($5.56/mo.)", or "Up to $200 off after 36 monthly bill credits" for a new line). `text` is a
  verbatim snippet from the source. `amount` is the total dollar amount (e.g. "$200"). Never add it
  to unique_discount. The monthly figure ($5.56/mo.) is the SAME money as the $200 spread over 36
  months; never count it separately.

max_trade_in: A specific, explicitly stated MAXIMUM dollar amount tied to trading in a device (e.g.
  "Up to $1,200 off ... with trade-in"). Several tiers may be listed ($1,200 / $930 / $500 / $350):
  pick the highest tier that applies to THIS product model. If the text only describes that a
  trade-in program exists or lists device-condition requirements without a stated maximum dollar
  amount for this product, return null. Trade-in conditions and eligible-device lists are not amounts.

trade_in_value_type: "Enhanced" or "Base" ONLY if the source text itself labels the trade-in value
  that way. Otherwise null. Do not infer it.

rateplan_duration_months: The installment/rate-plan term in months, as an integer, ONLY if it is
  stated for an offer that is valid for this line type (e.g. "36 mos." -> 36). Otherwise null.

PWP/GWP (accessory offers) are NOT part of your task. Do not output them and do not use accessory
  prices for any field; they are computed separately.

subscription_bundle: A subscription (streaming, cloud, care plan, etc.) bundled with the purchase.
  Null unless explicitly stated in the source.

free_memory_upgrade: A free storage/memory upgrade ("get 512GB for the price of 256GB"). Null unless
  explicitly stated in the source.

bogo: A buy-one-get-one offer. Null unless explicitly stated in the source.

additional_offers: Any other eligible promotion in the source that does not belong to another field
  and has not already been used. Null if none. Do not repeat a value used elsewhere.

finance: Third-party bank / credit-card offers (e.g. the Citi AT&T Points Plus card), distinct from the
  carrier's own installment plan. Return the offer as COMPONENTS; do NOT add them up yourself:
    - kind "one_time_welcome": acquisition / welcome / sign-up bonus. Valid for New Line ONLY.
      Excluded for Upgrade.
    - kind "recurring_annual": recurring yearly credit (use the stated yearly total, e.g. "up to $240
      ... every year"). Valid for both line types. Do not also add the per-cycle figure it comes from.
    - kind "recurring_discount": a standing bill discount (e.g. "$10 discounts on your AT&T bills").
      Valid for both line types. Use the single stated amount ("$10").
  Spend requirements are ignored. `description` is one clean sentence for THIS line type. It must
  name the card and its issuer exactly as the source names them (for example "the AT&T Points Plus
  Card from Citi") and must not contain any dollar amount that is not in the source.
  For New Line it should cover the welcome bonus plus the recurring benefits; for Upgrade it must
  cover only the recurring benefits. Components must contain only amounts valid for this line type.

=== OUTPUT JSON SCHEMA (return exactly these keys; use null where nothing applies) ===
{
  "switcher":              {"amount": str|null, "text": str|null, "source_quote": str|null},
  "unique_discount":       {"amount": str|null, "source_quote": str|null},
  "service_line":          {"text": str|null, "amount": str|null, "source_quote": str|null},
  "max_trade_in":          {"amount": str|null, "source_quote": str|null},
  "trade_in_value_type":   "Enhanced" | "Base" | null,
  "rateplan_duration_months": int|null,
  "subscription_bundle":   {"text": str|null, "amount": str|null, "source_quote": str|null},
  "free_memory_upgrade":   {"text": str|null, "source_quote": str|null},
  "bogo":                  {"text": str|null, "source_quote": str|null},
  "additional_offers":     {"text": str|null, "amount": str|null, "source_quote": str|null},
  "finance": {
    "description": str|null,
    "components": [ {"kind": "one_time_welcome"|"recurring_annual"|"recurring_discount",
                     "amount": str, "source_quote": str} ]
  },
  "notes": str|null
}
`notes` is optional, at most one short sentence on why something is null. Return the JSON object only.

=== WORKED ILLUSTRATION (from the assignment; abbreviated) ===
Product: Apple iPhone 17 Pro Max, variant 256 GB. Source says: "Eligibility: New customers activating
new line ... iPhone 17 Pro Max 256GB for $9.43/mo. for 36 mos. after up to $660.36 in bill credits ...
Price after additional $200 in credits ($5.56/mo.) for new consumer lines only."
  New Line -> unique_discount "$660.36"; service_line amount "$200".
  Upgrade  -> unique_discount null; service_line null (the offer is gated to new lines).
Source says "Get up to $800/line to break your contract." -> New Line switcher "$800"; Upgrade null.
Source says "AT&T Trade-In Program ... Receive an AT&T Promotional Card" with no maximum amount
-> max_trade_in null for both line types.
Citi card: $200 welcome bonus + "up to $240 ... every year" + "$10 discounts on your AT&T bills".
  New Line -> components [$200 one_time_welcome, $240 recurring_annual, $10 recurring_discount].
  Upgrade  -> components [$240 recurring_annual, $10 recurring_discount] (no welcome bonus).
"""


# --------------------------------------------------------------------------- #
# Source-text collection
# --------------------------------------------------------------------------- #
def _add_block(blocks: List[Tuple[str, str]], seen: set, label: str, text: Any) -> None:
    """Append a (label, text) block once, skipping empty or duplicate text."""
    if not isinstance(text, str):
        return
    cleaned = text.strip()
    if not cleaned or cleaned in seen:
        return
    seen.add(cleaned)
    blocks.append((label, cleaned))


def _raw_source_blocks(product: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Gather every text snippet the LLM may use, de-duplicated, labelled by origin.

    Pulls from: pricing.options and each offer (title, description, modal text,
    and any nested cards' legal / full-detail text).

    Deliberately NOT sent to the LLM:
      * accessories_offers  -> PWP/GWP is computed in plain Python from the printed
        "Was $X. Now $Y." prices (exact arithmetic, no LLM needed).
      * additional_add_on_offers -> `offer_text` is always null; it only carries
        price tags for ~1,000 catalogue items (noise).
    """
    blocks: List[Tuple[str, str]] = []
    seen: set = set()

    pricing = product.get("pricing") if isinstance(product.get("pricing"), dict) else {}
    for k, option in enumerate(pricing.get("options") or []):
        _add_block(blocks, seen, f"PRICING OPTION {k}", option)

    for j, offer in enumerate(product.get("offers") or []):
        if not isinstance(offer, dict):
            continue
        tag = f"OFFER {j}"
        _add_block(blocks, seen, f"{tag} title", offer.get("title"))
        _add_block(blocks, seen, f"{tag} description", offer.get("description"))
        details = offer.get("full_details")
        if isinstance(details, dict):
            _add_block(blocks, seen, f"{tag} details", details.get("modal_raw_text"))
            for c, card in enumerate(details.get("cards") or []):
                if not isinstance(card, dict):
                    continue
                _add_block(blocks, seen, f"{tag} card {c} legal", card.get("legal"))
                inner = card.get("full_details")
                if isinstance(inner, dict):
                    _add_block(blocks, seen, f"{tag} card {c} full terms", inner.get("raw_text"))

    return blocks


# --------------------------------------------------------------------------- #
# Prompt budget: keep every request under the provider's per-minute token limit
# --------------------------------------------------------------------------- #
DEFAULT_PROMPT_CHAR_BUDGET = 24000   # ~6,700 tokens at ~3.6 chars/token (measured on Groq/Qwen)
_PROMPT_OVERHEAD = 400               # facts header + instructions around the source text
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_SEGMENT_START = re.compile(r"^(firstnet\b[^:]{0,80}customers:|business customers:)", re.IGNORECASE)
_SEGMENT_END = re.compile(r"^limitations and other terms", re.IGNORECASE)
_SIGNAL = re.compile(
    r"\$|%|new line|new customer|existing|upgrad|port-?in|switch|trade|credit|bill|discount|off\b|"
    r"mos\b|month|installment|eligib|activat|requires?|per line|/mo|\b\d+\s?(?:gb|tb)\b", re.IGNORECASE)


def get_prompt_char_budget() -> int:
    """PROMPT_CHAR_BUDGET env var (characters of system + user prompt); 0 turns trimming off."""
    raw = os.environ.get("PROMPT_CHAR_BUDGET")
    if raw is None or raw.strip() == "":
        return DEFAULT_PROMPT_CHAR_BUDGET
    try:
        return max(int(raw), 0)
    except ValueError:
        return DEFAULT_PROMPT_CHAR_BUDGET


def _estimate_prompt_chars(blocks: List[Tuple[str, str]]) -> int:
    return len(SYSTEM_PROMPT) + _PROMPT_OVERHEAD + sum(len(label) + len(text) + 4 for label, text in blocks)


def _drop_other_segments(parts: List[str]) -> List[str]:
    """Remove FirstNet / Business sections (terms for other customer types, which repeat the
    consumer tiers). Only acts when the section's end marker exists, so no tail is lost."""
    out: List[str] = []
    skipping = False
    for index, sentence in enumerate(parts):
        if not skipping and _SEGMENT_START.search(sentence) and any(_SEGMENT_END.search(s) for s in parts[index + 1:]):
            skipping = True
        if skipping and _SEGMENT_END.search(sentence):
            skipping = False
        if not skipping:
            out.append(sentence)
    return out


def _drop_device_lists(parts: List[str]) -> List[str]:
    """Remove long comma-separated device/model lists (no offer information in them)."""
    return [s for s in parts if not (len(s) > 300 and s.count(",") >= 15)]


def _rewrite(blocks: List[Tuple[str, str]], fn) -> List[Tuple[str, str]]:
    result = []
    for label, text in blocks:
        parts = [p for p in _SPLIT.split(text) if p.strip()]
        kept = fn(parts)
        result.append((label, " ".join(kept) if kept and len(kept) != len(parts) else text))
    return result


def _drop_low_signal(blocks: List[Tuple[str, str]], budget: int) -> List[Tuple[str, str]]:
    """Last resort: drop sentences with no dollar sign / offer keyword, longest first, until it fits."""
    split = [[p for p in _SPLIT.split(text) if p.strip()] for _, text in blocks]
    candidates = sorted(((len(s), bi, si) for bi, parts in enumerate(split)
                         for si, s in enumerate(parts) if not _SIGNAL.search(s)), reverse=True)
    removed = set()
    current = list(blocks)
    for _, bi, si in candidates:
        if _estimate_prompt_chars(current) <= budget:
            break
        removed.add((bi, si))
        current = [(label, " ".join(s for k, s in enumerate(split[i]) if (i, k) not in removed) or text)
                   for i, (label, text) in enumerate(blocks)]
    return current


def _fit_to_budget(blocks: List[Tuple[str, str]], budget: int) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Return (blocks, steps applied). Untouched when already within budget."""
    if budget <= 0 or _estimate_prompt_chars(blocks) <= budget:
        return blocks, []
    steps: List[str] = []
    current = _rewrite(blocks, _drop_other_segments)
    if current != blocks:
        steps.append("terms for other customer types (FirstNet/Business)")
    if _estimate_prompt_chars(current) > budget:
        after = _rewrite(current, _drop_device_lists)
        if after != current:
            steps.append("long device lists")
        current = after
    if _estimate_prompt_chars(current) > budget:
        after = _drop_low_signal(current, budget)
        if after != current:
            steps.append("low-signal sentences")
        current = after
    return current, steps


def collect_source_blocks(product: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Source blocks the LLM sees AND the validator checks against (always the same text).

    Prompts within the character budget are returned unchanged (so saved answers stay
    valid). Oversized prompts are trimmed, see _fit_to_budget.
    """
    blocks, _ = _fit_to_budget(_raw_source_blocks(product), get_prompt_char_budget())
    return blocks


def source_trim_report(product: Dict[str, Any]) -> Optional[str]:
    """A one-line note when this product's prompt had to be trimmed, else None."""
    raw = _raw_source_blocks(product)
    trimmed, steps = _fit_to_budget(raw, get_prompt_char_budget())
    if not steps:
        return None
    return (f"prompt trimmed from ~{_estimate_prompt_chars(raw)} to ~{_estimate_prompt_chars(trimmed)} characters "
            f"to fit the provider's per-minute token limit; removed: {', '.join(steps)}")


def collect_source_text(product: Dict[str, Any]) -> str:
    """All source text joined into one string (used by Step 7 to verify quotes)."""
    return "\n".join(text for _, text in collect_source_blocks(product))


# --------------------------------------------------------------------------- #
# User message
# --------------------------------------------------------------------------- #
def build_user_message(product: Dict[str, Any], line_type: str) -> str:
    """Build the per-row user message for one product and one line type."""
    if line_type not in (LINE_NEW, LINE_UPGRADE):
        raise ValueError(f"Unknown line type: {line_type!r}")

    variant = str(product.get("variant") or "").strip()
    facts = {
        "product_name": product.get("product_name"),
        "brand": product.get("brand"),
        "variant": variant,
        "variant_match_key": normalize_variant(variant),
        "simulated_line_type": line_type,
    }
    source = "\n\n".join(f"[{label}]\n{text}" for label, text in collect_source_blocks(product))
    return (
        "PRODUCT AND SIMULATED LINE TYPE:\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
        + f"\n\nEvaluate ONLY for LINE TYPE = \"{line_type}\" and ONLY for variant "
        f"\"{variant}\" of \"{product.get('product_name')}\".\n\n"
        "SOURCE (the only text you may use):\n"
        "<<<SOURCE\n" + source + "\nSOURCE>>>\n\n"
        "Return the JSON object only."
    )


def build_messages(product: Dict[str, Any], line_type: str) -> Dict[str, str]:
    """Return {'system': ..., 'user': ...} ready to send to the LLM client (Step 6)."""
    return {"system": SYSTEM_PROMPT, "user": build_user_message(product, line_type)}