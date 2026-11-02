# Data Normalization and LLM Extraction

This is my solution for the Python developer intern assignment. It takes the messy product JSON from a retailer (`Input.json`, 12 phones) and turns it into one clean CSV with 29 fixed columns.

The hard part of the data is the offer text. Things like "Get up to $800/line when you switch" or "for new consumer lines only" are written in long paragraphs, and some offers only work for a new line and not for an upgrade. So I use an LLM to read the text, and then plain Python to check everything the LLM says before it goes into the CSV.

For every product the script writes exactly two rows, one for **New Line** and one for **Upgrade**. 12 products give 24 rows.

## How it works

1. Read the product from the JSON.
2. Fill the simple columns straight from the JSON, no LLM (product name, brand, variant, MSRP, stock, shipment, plan type).
3. Make two rows from it, New Line and Upgrade.
4. Send the offer text to the LLM with strict rules and ask for JSON back. Each amount it returns has to come with the exact sentence it copied it from.
5. Check the answer in Python. If I can't prove a value from the source text, it is dropped and left blank.
6. Write the CSV and a log of everything that was dropped or fixed.

I don't trust the model for numbers. The checks in step 5 are the part I spent most time on. They do these things:

- The quote must really exist in the source text, and the amount must be in the quote with the exact digits. So `$628` is rejected when the text says `$628.36`.
- The unique discount must come right after the product's own storage size. The text lists 256GB, 512GB, 1TB and 2TB amounts together, so a 1 TB phone can't take the 2 TB amount.
- Line type rules are checked again in code. The switcher is never allowed on an Upgrade row. Offers that say "new line only" are dropped on Upgrade rows.
- The finance total is added up in Python, not by the model. New Line is $200 + $240 + $10 = $450. Upgrade has no welcome bonus, so it is $240 + $10 = $250.
- Max trade-in is the highest tier that names the phone model. The iPhone 17 text also lists a $1,200 tier, but that one is only for the iPhone 18 Pro and Pro Max, so the iPhone 17 gets $930.
- One sentence from the source can only be used in one column.

## Files

| File | What it does |
|---|---|
| `main.py` | The command to run. Reads the JSON, runs everything, writes the files. |
| `constants.py` | The 29 column names in the exact order from the PDF. |
| `mapper.py` | Fills the simple columns from the JSON. |
| `row_builder.py` | Makes the two rows (New Line, Upgrade) for each product. |
| `prompt_builder.py` | Builds the prompt: rules for the LLM plus the offer text. |
| `llm_client.py` | Talks to the LLM (Groq or Gemini). Retries, timeouts, JSON parsing, quota handling. |
| `validator.py` | Checks the LLM answer against the source text. |
| `pwp.py` | Works out the PWP/GWP offer from the accessories data, no LLM. |
| `pipeline.py` | Runs all rows, saves progress, checks the output, writes the CSV. |
| `env_loader.py` | Reads the `.env` file. |
| `verify_output.py` | Checks the final CSV against the assignment rules. |
| `check_connection.py` | Sends one tiny request to test the API key and model. |
| `test_*.py`, `test_support.py` | The tests (187). |
| `Input.json` | The input data. |
| `output/` | `normalized_output.csv` (the result) and `issues.log`. |

## How to run

You need Python 3.9 or newer. I tested it on Python 3.12. There is nothing to install, it only uses the standard library.

**1. Get the code**

```
git clone <the repo url>
cd <the repo folder>
```

**2. Make your `.env` file**
For Groq the important lines are:

```
LLM_PROVIDER=groq
GROQ_API_KEY=your key here
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_TPM_LIMIT=7000
```

For Gemini use `LLM_PROVIDER=gemini` and `GEMINI_API_KEY`. The `.env` file is in `.gitignore`, so the key is never pushed. Check the limits for your own account and model, they are different for every model.

**3. Run the tests (no internet, no key needed)**

```
python -m unittest discover
```

You should see `Ran 187 tests ... OK`. The tests never call a real API.

**4. Try it without the LLM**

```
python main.py --dry-run
```

This writes `output/dry_run_output.csv` with all 24 rows, but the offer columns stay empty because no LLM was used. It is only to see the structure.

**5. Real run**

```
python main.py
```

It writes `output/normalized_output.csv` and `output/issues.log`.

On a free plan it goes slowly. With a 7,000 tokens per minute limit it can only do about one call per minute, so the full run is around 25 minutes. If the daily limit runs out it stops and tells you. Run the same command again later and it carries on. It saves every answer in `output/checkpoint.jsonl` and never asks the same question twice.

**6. Check the result**

```
python verify_output.py
```

It prints PASS, WARN or FAIL for each check and then a table of the key columns. My run gave 21 passed, 0 warnings, 0 failures.

Useful options for `main.py`:

| Option | What it does |
|---|---|
| `--limit 1` | Only the first product (2 rows). Good for a cheap test. |
| `--fresh` | Throws away the saved answers and starts again. |
| `--dry-run` | No LLM, no key. |
| `--pause 5` | Waits 5 seconds between calls. |
| `--output file.csv` | Different name for the CSV. |

## Settings in `.env`

| Setting | Meaning | Default |
|---|---|---|
| `LLM_PROVIDER` | `groq`, `gemini` or `dryrun` | `gemini` |
| `GROQ_API_KEY`, `GROQ_MODEL` | Groq key and model name | model: `llama-3.3-70b-versatile` |
| `GROQ_TPM_LIMIT` | Tokens per minute your plan allows. The script waits so it doesn't go over. | 12000 |
| `PROMPT_CHAR_BUDGET` | Biggest prompt in characters. Bigger ones are trimmed. | 24000 |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Gemini key and model name | |
| `GEMINI_FALLBACK_MODEL` | Second Gemini model to use if the first stays busy | none |
| `*_TIMEOUT`, `*_MAX_RETRIES` | Seconds to wait for one answer, and how many retries | 90, 3 |

## Choices I made where the assignment wasn't clear

The PDF gives names for all 29 columns but only explains some of them. For the others I looked at the data. When the data had nothing, I left the column blank. The assignment says not to guess, so I didn't.

| Column | What I did |
|---|---|
| BASE, RESELLER LIST PRICE | Blank. The PDF doesn't say what they are. The only reseller number in the data is a monthly price for one storage size, which is not a list price. |
| Trade-In Value Type | Blank. The source text never says Enhanced or Base. |
| FREE MEMORY UPGRADE, BOGO OFFER, SUBSCRIPTION BUNDLE, ADDITIONAL OFFERS | Blank. There is nothing like this in the data. |
| PWP/GWP | The AirPods 4 are on sale in the accessories list (was $179.00, now $149.00). I count the saving, $30.00, worked out in code. It is the same on both line types. |
| RATEPLAN DURATION | 36, only on rows where an offer that mentions 36 months is valid for that line type. |
| Money format | `$` plus the exact digits. No commas, so `$1,200` becomes `$1200`, like the `$1000` in the PDF example. No `.00` added to whole numbers. |
| Max TI | The highest tier that names the phone model, same for New Line and Upgrade. Blank where the text has no tier for the model. |
| Unique discount vs service line | Kept apart and never added. The `$5.56/mo` credit is the same money as the `$200` spread over 36 months, so it is not counted again. |

One thing to know about the PDF example: it uses the iPhone 17 Pro Max 256 GB, but that size is not in my input file (only 1 TB and 2 TB). The 1 TB rows follow the same logic.

## Result

For my input: 24 rows, 29 columns. All New Line rows have the $800 switcher and the $200 service line offer. Seven products have a unique discount that matches their own storage size. The iPhone 17 and iPhone 18 Pro Max rows have a Max TI ($930 and $1200) on both line types. Finance is $450 on New Line and $250 on Upgrade. PWP is $30.00 everywhere.

The full table is in the PDF (`Solution_Document.pdf`).

## Problems I ran into

- **Gemini was busy (HTTP 503).** The model was overloaded for a while. I added retries with waiting and an optional second model to fall back to.
- **Quota limits (HTTP 429).** I ran out of the free quota. The client now reads how long the server asks to wait. Short waits are retried. If the wait is hours (a daily limit), it stops and keeps the saved answers, so no calls are wasted.
- **Prompt too big (HTTP 413).** My Groq model only allows 7,000 input tokens in one request, and the biggest prompts were about 8,300. Those were the iPhone 17 and iPhone 18 Pro Max rows. Their trade-in terms are long and repeat the same tiers for FirstNet and Business customers, and also list a lot of phone models. I now remove that text, but only for prompts over the size budget. Prompts that were already small are not changed, so the answers I had already paid for stayed valid.
- **My tests were using my real `.env`.** A test could end up calling the real API. The tests now ignore `.env` and the environment, and any real network call inside a test fails on purpose.

## What it does not do

- The sentences in the switcher text and the finance description are written by the model. I check the dollar amounts in them, not the wording.
- It was only tested on this one input file. A different retailer would need the prompt rules looked at again.
- The model that wrote each answer is saved next to it in `output/checkpoint.jsonl`. The final run used Groq with `qwen/qwen3.8-27b`. Some early rows may have come from Gemini while I was testing.
- Free plans are slow and have daily limits. A paid plan would finish in one go.
