"""Project-wide constants.

The CSV schema is defined here once so every other module imports the same
column names. Order matters: it is the exact order from the assignment PDF
(Section 3, "Required CSV Columns").
"""

# --- Line types (the "Two-Row" Multiplication Rule) -------------------------
LINE_NEW = "New Line"
LINE_UPGRADE = "Upgrade"
LINE_TYPES = (LINE_NEW, LINE_UPGRADE)  # every product yields one row per entry

# --- Output column names (exact order from the PDF, items 1-29) -------------
COL_PLAN_TYPE = "PLAN TYPE"
COL_PRODUCT_NAME = "PRODUCT NAME"
COL_BRAND = "BRAND"
COL_VARIANT = "VARIANT"
COL_BASE = "BASE"
COL_MSRP = "MSRP"
COL_RESELLER_LIST_PRICE = "RESELLER LIST PRICE"
COL_SWITCHER = "SWITCHER"
COL_SWITCHER_OFFER_TEXT = "SWITCHER_OFFER_TEXT"
COL_UNIQUE_DISCOUNT = "UNIQUE DISCOUNT"
COL_FREE_MEMORY_UPGRADE = "FREE MEMORY UPGRADE"
COL_BOGO_OFFER = "BOGO OFFER"
COL_MAX_TI = "Max TI"
COL_TRADE_IN_VALUE_TYPE = "Trade-In Value Type"
COL_LINE_TYPE = "LINE TYPE"
COL_RATEPLAN_DURATION = "RATEPLAN DURATION (MONTHS)"
COL_SHIPMENT = "SHIPMENT"
COL_STOCK = "STOCK"
COL_PWP_GWP_OFFERS = "PWP/GWP OFFERS"
COL_CLEANED_PWP_GWP_OFFERS = "Cleaned PWP/GWP OFFERS"
COL_PWP_GWP_VALUE = "PWP/GWP OFFER $ VALUE"
COL_SUBSCRIPTION_BUNDLE_OFFERS = "SUBSCRIPTION BUNDLE OFFERS"
COL_SUBSCRIPTION_BUNDLE_VALUE = "SUBSCRIPTION BUNDLE OFFER $ VALUE"
COL_SERVICE_LINE_OFFERS = "SERVICE LINE OFFERS"
COL_SERVICE_LINE_VALUE = "SERVICE LINE OFFER $ VALUE"
COL_ADDITIONAL_OFFERS = "ADDITIONAL OFFERS"
COL_ADDITIONAL_VALUE = "ADDITIONAL OFFER $ VALUE"
COL_FINANCE_OFFER_DESC = "FINANCE OFFER DESCRIPTION 1"
COL_FINANCE_OFFER_VALUE = "FINANCE OFFER $ VALUE"

OUTPUT_COLUMNS = [
    COL_PLAN_TYPE,                   # 1
    COL_PRODUCT_NAME,                # 2
    COL_BRAND,                       # 3
    COL_VARIANT,                     # 4
    COL_BASE,                        # 5
    COL_MSRP,                        # 6
    COL_RESELLER_LIST_PRICE,         # 7
    COL_SWITCHER,                    # 8
    COL_SWITCHER_OFFER_TEXT,         # 9
    COL_UNIQUE_DISCOUNT,             # 10
    COL_FREE_MEMORY_UPGRADE,         # 11
    COL_BOGO_OFFER,                  # 12
    COL_MAX_TI,                      # 13
    COL_TRADE_IN_VALUE_TYPE,         # 14
    COL_LINE_TYPE,                   # 15
    COL_RATEPLAN_DURATION,           # 16
    COL_SHIPMENT,                    # 17
    COL_STOCK,                       # 18
    COL_PWP_GWP_OFFERS,              # 19
    COL_CLEANED_PWP_GWP_OFFERS,      # 20
    COL_PWP_GWP_VALUE,               # 21
    COL_SUBSCRIPTION_BUNDLE_OFFERS,  # 22
    COL_SUBSCRIPTION_BUNDLE_VALUE,   # 23
    COL_SERVICE_LINE_OFFERS,         # 24
    COL_SERVICE_LINE_VALUE,          # 25
    COL_ADDITIONAL_OFFERS,           # 26
    COL_ADDITIONAL_VALUE,            # 27
    COL_FINANCE_OFFER_DESC,          # 28
    COL_FINANCE_OFFER_VALUE,         # 29
]

# Columns whose cell content is a dollar amount (must look like "$800" / "$244.36").
MONEY_COLUMNS = [
    COL_MSRP,
    COL_RESELLER_LIST_PRICE,
    COL_SWITCHER,
    COL_UNIQUE_DISCOUNT,
    COL_MAX_TI,
    COL_PWP_GWP_VALUE,
    COL_SUBSCRIPTION_BUNDLE_VALUE,
    COL_SERVICE_LINE_VALUE,
    COL_ADDITIONAL_VALUE,
    COL_FINANCE_OFFER_VALUE,
]

EXPECTED_COLUMN_COUNT = 29
