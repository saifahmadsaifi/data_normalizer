"""Row builder (Step 4): implements the "Two-Row" Multiplication Rule.

Every product object in the source JSON yields exactly two rows:
  Row 1 -> LINE TYPE = "New Line"
  Row 2 -> LINE TYPE = "Upgrade"

Only the base fields and LINE TYPE are set here. Offer columns stay blank and
are filled later by the LLM extraction + validation steps, evaluated
independently for each line type.
"""

from typing import Any, Dict, List, Optional

from constants import COL_LINE_TYPE, LINE_TYPES
from mapper import map_base_fields

Row = Dict[str, Optional[str]]


def build_rows_for_product(product: Dict[str, Any]) -> List[Row]:
    """Return exactly two rows (New Line, then Upgrade) for one product.

    Each row is its own dict, so filling offers in one row can never leak
    into the other.
    """
    rows: List[Row] = []
    for line_type in LINE_TYPES:  # order: New Line, Upgrade
        row = map_base_fields(product)  # fresh dict each iteration
        row[COL_LINE_TYPE] = line_type
        rows.append(row)
    return rows


def build_all_rows(products: List[Dict[str, Any]]) -> List[Row]:
    """Return 2 x len(products) rows, keeping the input product order."""
    if not isinstance(products, list):
        raise TypeError(
            f"Expected a list of product objects, got {type(products).__name__}"
        )
    all_rows: List[Row] = []
    for product in products:
        all_rows.extend(build_rows_for_product(product))
    return all_rows