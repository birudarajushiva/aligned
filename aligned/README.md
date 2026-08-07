# ALIGNED — autonomous account retention + upsell agent

B2B SaaS companies lose ARR because nobody notices an account going cold until
renewal. Aligned watches every account in a portfolio, flags churn risk and
upsell readiness with rules you can read, and drafts outreach grounded in that
specific customer's own history.

The pitch in one line: **it surfaced $60,468 of at-risk ARR across a 12-account
portfolio, against a $2,388/year subscription to Aligned itself ($199/mo) —
a 25.32x multiple.**

Three pieces:

| | what it does | how |
| --- | --- | --- |
| **Scoring** | flags churn risk / upsell readiness | deterministic rules, no LLM |
| **Memory** | remembers each customer's history | EverOS Memory API v2 (`api.evermind.ai`) |
| **Drafting** | writes the outreach email | Gemini 3.5 Flash, grounded in retrieved memory |

Scoring is deliberately dumb and explainable: an account is a churn risk if it
went quiet, its usage fell off a cliff, or its last two support tickets were
both unhappy. No model decides who gets flagged, so every flag has a number
behind it. This is `acc_001`'s actual `reason` string, in full:

> No login in 23 days; usage down 61% since March; last 2 tickets negative
> ("Third week without a fix on the export timeouts, escalating to their VP")

---

## Setup

Python 3.11 is the target; the code is written to run on 3.9+ unchanged.

```bash
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
```

Copy the example env file and fill in the two keys:

```bash
cp .env.example .env
```

### Environment variables

Read in exactly one place, `aligned/config.py`, and nowhere else.

| variable | required | what happens without it |
| --- | --- | --- |
| `EVERMIND_API_KEY` | for memory | `EVEROS_ENABLED=False`, a loud startup warning, `memories: []` everywhere, seeding refuses with a 503. Accounts and scoring still serve. |
| `EVERMIND_BASE_URL` | no | defaults to `https://api.evermind.ai`. Only set it to point at staging. |
| `GEMINI_API_KEY` | for drafting | `POST /api/scan` returns 503 naming the variable. Everything else still works. |

Get an EverOS key at [everos.evermind.ai](https://everos.evermind.ai) and a
Gemini key at [aistudio.google.com](https://aistudio.google.com).

The app boots fine with both blank — you just get a smaller demo.

---

## Running it, in order

**1. Start the server.**

```bash
./.venv/bin/uvicorn api:app --reload --port 8000
```

It prints a banner telling you what is live before you demo anything:

```
==========================================================================
  ALIGNED — autonomous retention + upsell agent
--------------------------------------------------------------------------
  accounts    12 loaded  (6 healthy / 3 churn_risk / 3 upsell_ready)
  portfolio   $138,420 ARR under management (simulated)
  EverOS      LIVE   https://api.evermind.ai
  Gemini      LIVE   gemini-2.5-flash
  memories    NOT YET -> POST /api/seed-memory
  actions     none yet -> POST /api/scan
==========================================================================
```

Nothing is seeded or scanned at startup, on purpose: both are network-bound,
and a flaky connection should never stop the server booting.

**2. Seed memory into EverOS.** Run once. Writes 97 memories — one profile
paragraph, one entry per feature plus a usage-trend entry, and one entry per
support ticket, all as natural prose because retrieval quality depends on it.

```bash
curl -s -X POST http://127.0.0.1:8000/api/seed-memory | python3 -m json.tool
```

This is the slow step. Writes are synchronous (`async_mode: false`) so the
memories are searchable the instant it returns — the alternative is a 202 and
memories that aren't there when you demo recall. Budget real time for it.

**3. Run the scan.** Drafts outreach for every flagged account, sequentially.

```bash
curl -s -X POST http://127.0.0.1:8000/api/scan | python3 -m json.tool
```

**4. The numbers.**

```bash
curl -s http://127.0.0.1:8000/api/metrics | python3 -m json.tool
```

### Endpoints

| method | path | |
| --- | --- | --- |
| `GET` | `/` | `dashboard.html` if present, else a JSON hint saying where to put it |
| `GET` | `/api/health` | `{"ok": true, "accounts": 12}` |
| `GET` | `/api/accounts` | all accounts, scored, riskiest first |
| `GET` | `/api/accounts/{id}` | one account, plus what EverOS remembers about it |
| `POST` | `/api/seed-memory` | write the portfolio into EverOS (run once) |
| `POST` | `/api/scan` | draft outreach for flagged accounts |
| `GET` | `/api/actions` | every action drafted this session, newest first |
| `GET` | `/api/metrics` | portfolio totals and the ROI multiple |
| `POST` | `/api/compare/{id}` | the A/B proof: same email with and without memory |
| `POST` | `/api/reset` | clear actions, keep accounts and memories |
| `GET` | `/pricing.html` | pricing page (no checkout connected) |

---

## What's simulated vs measured

Point a judge at this section. It is the honest answer.

### Simulated — invented by us, not observed from anything

- **Every account.** All 12 companies are made up. Northwind Logistics, Rivet
  Health, Ardent Labs — none of them exist.
- **All usage data.** Every login gap, every monthly usage count, every
  per-feature tally is authored in `aligned/seed.py`.
- **All support tickets.** Every ticket date, summary and sentiment is written
  by hand.
- **All ARR and MRR figures.** Including the $48,000 Northwind contract, the
  $138,420 portfolio total, the $60,468 at-risk figure, and every upsell
  expansion estimate.
- **The plan price ladder** used to size upsells (Starter $199 / Growth $800 /
  Enterprise $3,000 per month), and the assumption that a top-tier account
  expands by 25%.

- **Aligned's own $199/month price**, the ROI denominator. We made that up too.

Every API response carrying a money figure is tagged `"simulated": true` —
`/api/accounts`, `/api/accounts/{id}`, `/api/scan`, `/api/actions` and
`/api/metrics`. On accounts the flag is set twice, on the seed data *and* again
at the response boundary, so no future edit to the seed can quietly ship a
modeled number that reads as observed.

### Measured — real values from real API calls

- **Token counts.** Read from Gemini's `usageMetadata` on every response. Never
  estimated. If that field is missing, the call raises rather than guessing.
- **EverOS retrieval latency.** Timed around the actual HTTP call and logged in
  milliseconds.
- **Gemini call latency**, same.

### Computed — arithmetic on top of the above

- **`cost_usd`.** Measured tokens × Google's published paid-tier rates
  ($1.50 in / $9.00 out per 1M tokens for `gemini-3.5-flash`, from
  [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing),
  verified 2026-08-07). The rates live in one dict in `aligned/config.py`.

  **Be careful with this one on stage.** On the AI Studio free tier these calls
  bill **$0.00**. `cost_usd` is what the scan *would* cost at paid-tier rates —
  it is not money that changed hands. Say "would cost", not "cost".

- **`roi_multiple`.** At-risk ARR ÷ Aligned's own annual price
  ($60,468 ÷ $2,388 = 25.32). Not floored, not capped, not rounded up — the
  code divides and reports whatever comes out.

  **Both inputs to this number are simulated.** The $60,468 comes from invented
  accounts; the $2,388 is a price we made up for ourselves. Only the division
  is real. It belongs in this section as arithmetic, not as evidence — it
  demonstrates the shape of the argument on a portfolio we control, and says
  nothing yet about a real customer's book of business.

  It is also **at-risk ARR surfaced per dollar of subscription**, not ARR
  *saved*. Aligned flags the account and drafts the email; whether the customer
  stays depends on the human who sends it. If a judge asks "so you saved
  $60k?", the answer is no — we surfaced $60k of at-risk ARR for $2,388/yr.

### Grounding

`memory_used` on every action is the **verbatim** text EverOS returned. It is
assigned straight from the retrieval call and never passed through the model,
never paraphrased, never summarised — read `generate_action` in
`aligned/agent.py` and you can see there is no path that rewrites it.

Before a draft ships, a deterministic check counts how many specifics from the
retrieved snippets actually appear in it — a named feature or a real figure,
by plain string matching rather than a second model, so it can't be flaky on
stage. A draft citing fewer than two gets re-drafted once, with a prompt naming
the exact specifics it should have quoted.

**That check warns, it does not block.** If the second draft is still weakly
grounded the action ships anyway, with a loud `!!` line in the server log naming
the account. This is deliberate — one vague email should not kill a live scan —
but it means a low-confidence action can reach the UI. The `confidence` score
reflects it: grounding is half the formula, so an ungrounded action can never
score above 0.5.

---

## Layout

```
api.py                 FastAPI app, CORS wide open for the local UI
aligned/
  config.py            env vars, scoring thresholds, price ladder, Gemini rates
  seed.py              the 12-account simulated portfolio (fixed RNG seed)
  scoring.py           deterministic churn / upsell rules — no LLM
  memory.py            EverOS Cloud v1 client (write, recall, seed)
  agent.py             retrieve -> draft -> verify grounding -> Action
  metrics.py           portfolio totals and the ROI headline
  store.py             in-memory dicts + actions.jsonl audit log
```

The portfolio is generated from a fixed RNG seed and hand-authored profiles, so
it is byte-identical on every run — a rehearsal and the live demo produce the
same numbers. Status is never hardcoded: the account data is written so the
scoring rules reach 6 healthy / 3 churn_risk / 3 upsell_ready on their own, and
`verify_portfolio()` fails loudly at startup if that ever drifts.

---

## Verifying it

```bash
./.venv/bin/python verify_everos.py     # 6 memory checks, stops on first failure
./.venv/bin/python verify_gemini.py     # 6 Gemini checks incl. a live model probe
./.venv/bin/python test_aligned.py      # full 27-check acceptance suite
./.venv/bin/python test_aligned.py --fast   # skips the 4 Gemini checks, saves quota
```

## Demo flags

```bash
./.venv/bin/uvicorn api:app --port 8000                       # normal
./.venv/bin/python -c "import sys;sys.argv=['api.py','--demo'];import uvicorn,api;uvicorn.run(api.app,port=8000)"
./.venv/bin/python -c "import sys;sys.argv=['api.py','--offline'];import uvicorn,api;uvicorn.run(api.app,port=8000)"
```

`--demo` seeds memory and runs one scan at startup. `--offline` replays the last
good scan and A/B from `demo_cache.json` with zero network calls.

## Models

`gemini-2.5-flash` and `gemini-2.5-flash-lite` return 404 "no longer available
to new users" for this API key. The project uses **gemini-3.5-flash** for
drafting and **gemini-3.5-flash-lite** for classification and as the 503
fallback. `verify_gemini.py` probes the live candidate list, so run it if
drafting starts failing.
