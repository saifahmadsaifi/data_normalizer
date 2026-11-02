"""Base mapper (Step 3): fills the columns that need NO LLM.

Only fields that exist verbatim in the source JSON are copied here. Nothing is
guessed, rounded or inferred. Every other column starts as None and is filled
by later pipeline steps (LLM extraction + validation).
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

from constants import (
    COL_BRAND,
    COL_MSRP,
    COL_PLAN_TYPE,
    COL_PRODUCT_NAME,
    COL_SHIPMENT,
    COL_STOCK,
    COL_VARIANT,
    OUTPUT_COLUMNS,
)

_NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")


def clean_text(value: Any) -> Optional[str]:
    """Return a stripped string, or None if the value is missing/blank."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def format_money(value: Any) -> Optional[str]:
    """Turn a numeric string like '1599.99' into '$1599.99'.

    The digits are copied exactly as they appear in the source: no rounding,
    no added or dropped decimals ('800' -> '$800', '244.36' -> '$244.36').
    Returns None for missing or non-numeric input instead of guessing.
    """
    text = clean_text(value)
    if text is None:
        return None
    text = text.lstrip("$").replace(",", "")
    if not _NUMBER_RE.match(text):
        return None
    try:
        Decimal(text)  # sanity check only; the original string is kept as-is
    except InvalidOperation:
        return None
    return f"${text}"


def empty_row() -> Dict[str, Optional[str]]:
    """A row with all 29 columns present and set to None (blank in the CSV)."""
    return {column: None for column in OUTPUT_COLUMNS}


def map_base_fields(product: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Map one raw product object to a row holding only the non-LLM columns.

    Line type is NOT set here; Step 4 duplicates this row for New Line / Upgrade.
    Missing keys or an unexpected payload shape produce None, never an exception.
    """
    row = empty_row()
    if not isinstance(product, dict):
        return row

    pricing = product.get("pricing")
    retail = pricing.get("retail") if isinstance(pricing, dict) else None

    row[COL_PLAN_TYPE] = clean_text(product.get("plan_type"))
    row[COL_PRODUCT_NAME] = clean_text(product.get("product_name"))
    row[COL_BRAND] = clean_text(product.get("brand"))
    row[COL_VARIANT] = clean_text(product.get("variant"))
    row[COL_MSRP] = format_money(retail)  # from pricing.retail
    row[COL_SHIPMENT] = clean_text(product.get("shipment"))  # raw text, unchanged
    row[COL_STOCK] = clean_text(product.get("stock"))  # raw value, unchanged
    return row

   