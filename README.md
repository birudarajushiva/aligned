# Aligned
### Explainable account scoring with memory-grounded AI outreach

Aligned is a Python/FastAPI prototype that identifies churn risk and upsell opportunities in a **simulated** SaaS account portfolio. Deterministic rules flag accounts; EverOS retrieves account history; Gemini drafts outreach with the retrieved text exposed for inspection.

**All companies, usage histories, revenue figures, plan prices, and ROI examples are synthetic. This project does not demonstrate revenue saved or production customer outcomes.**

## Engineering highlights

- Readable scoring rules separate business decisions from model-generated prose.
- Retrieved memory is retained verbatim in `memory_used`, providing an inspectable connection between evidence and drafts.
- Startup avoids network-bound seeding and scanning, so the API can boot without provider credentials.
- Explicit offline replay uses saved results rather than presenting cached output as a fresh model call.
- Grounding checks, retry behavior, and an audit log make failure modes visible.

## Run locally

Python 3.11 is the documented target. From the repository root:

```sh
cd aligned
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python api.py
```

Open http://127.0.0.1:8000 for the dashboard and http://127.0.0.1:8000/docs for the API.

Accounts and scoring are available without keys. For live memory and drafting, create `aligned/.env` containing `EVERMIND_API_KEY` and `GEMINI_API_KEY`. The tracked template is [`aligned/.env.example`](aligned/.env.example); do not commit real credentials. Model selection and pricing assumptions live in [config.py](aligned/config.py).

Seed memory once, then scan:

```sh
curl -X POST http://127.0.0.1:8000/api/seed-memory
curl -X POST http://127.0.0.1:8000/api/scan
```

Seeding writes to the configured external memory service and is not idempotent. Scanning uses provider quota and may incur charges.

## Architecture

```text
Simulated accounts → deterministic scoring → flagged accounts
                                             ↓
                          EverOS retrieval → Gemini draft
                                             ↓
                              grounding checks → actions + dashboard
```

The implementation is entirely under [aligned/](aligned/): `api.py` serves HTTP and the dashboard, `scoring.py` owns rules, `memory.py` handles retrieval, `agent.py` drafts, and `store.py` holds session state and writes `actions.jsonl`.

## Demo and verification

For an offline replay, from `aligned/` after installing dependencies:

```sh
.venv/bin/python -c "import sys; sys.argv=['api.py','--offline']; import uvicorn,api; uvicorn.run(api.app,host='127.0.0.1',port=8000)"
```

This serves the committed `demo_cache.json`; it is recorded example output, not a live provider run.

The existing acceptance suite is `test_aligned.py`. It expects the API at port 8000. `--fast` skips Gemini-backed checks, but **is not a guarantee of zero network calls**; other checks exercise memory integration. Inspect it before running with provider credentials. Provider probes are `verify_everos.py` and `verify_gemini.py`. See the [detailed implementation guide](aligned/README.md) for commands and evidence boundaries.

## Limits

- Grounding checks warn rather than block after an unsuccessful retry; inspect every draft before use.
- In-process state and an append-only local audit file are prototype persistence.
- Broad CORS and unauthenticated endpoints make this a local demonstration, not a public service ready for customer data.
- Monetary outputs are modeled arithmetic; token counts and timings come from runtime calls when those calls actually occur.
- No repository-wide license is currently included.
