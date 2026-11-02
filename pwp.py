"""PWP/GWP extractor (Step 7): computed in plain Python, no LLM.

Source: `accessories_offers`. Their prices are prefixed with "+" (added to the
order with the device), which makes them Pay-With-Purchase prices. An item
counts as an offer only when it is badged "On sale" AND the source prints both
an original and a current price with original > current.

  value   = original - current, computed with Decimal (exact; e.g. $30.00)
  raw     = the source's own wording ("<title> | On sale | Was $179.00. Now $149.00.")
  cleaned = short description ("<title> for $149.00 (was $179.00)")

If several items qualify, raw/cleaned are joined with "; " and the values are
added. The catalogue in `additional_add_on_offers` is intentionally not used.
"""

import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from mapper import clean_text

_PRICE_RE = re.compile(r"^\+?\$?(\d+(?:,\d{3})*(?:\.\d+)?)$")


def _parse_price(value: Any) -> Optional[Tuple[str, Decimal]]:
    """'+$149.00' -> ('149.00', Decimal('149.00')); None if not a clean price."""
    text = clean_text(value)
    if text is None:
        return None
    match = _PRICE_RE.match(text)
    if not match:
        return None
    digits = match.group(1).replace(",", "")
    return digits, Decimal(digits)


def extract_pwp(product: Dict[str, Any]) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """Return ({'raw','cleaned','value'}, issues). All three are None when nothing qualifies."""
    issues: List[str] = []
    raws: List[str] = []
    cleaned_items: List[str] = []
    total = Decimal(0)

    accessories = product.get("accessories_offers") if isinstance(product, dict) else None
    for index, item in enumerate(accessories or []):
        if not isinstance(item, dict):
            continue
        badge = (clean_text(item.get("badge")) or "").lower()
        if badge != "on sale":
            continue
        pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
        current = _parse_price(pricing.get("current_price"))
        original = _parse_price(pricing.get("original_price"))
        title = clean_text(item.get("title"))
        if not (title and current and original):
            issues.append(f"pwp: accessory #{index} on sale but price/title unreadable; skipped")
            continue
        if original[1] <= current[1]:
            issues.append(f"pwp: accessory #{index} not actually discounted; skipped")
            continue
        hidden = clean_text(pricing.get("hidden_description"))
        if hidden and (original[0] not in hidden or current[0] not in hidden):
            issues.append(f"pwp: accessory #{index} prices disagree with its description; skipped")
            continue

        saving = original[1] - current[1]
        total += saving
        raws.append(f"{title} | {clean_text(item.get('badge'))} | "
                    f"{hidden or 'Was $%s. Now $%s.' % (original[0], current[0])}")
        cleaned_items.append(f"{title} for ${current[0]} (was ${original[0]})")

    if not raws:
        return {"raw": None, "cleaned": None, "value": None}, issues
    return {
        "raw": "; ".join(raws),
        "cleaned": "; ".join(cleaned_items),
        "value": f"${format(total, 'f')}",
    }, issues
