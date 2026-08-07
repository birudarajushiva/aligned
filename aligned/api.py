"""Aligned HTTP API.

Boots the simulated portfolio into memory, scores it with the deterministic
rules in aligned/scoring.py, and serves it. Memory lives in EverOS; outreach
is drafted by Gemini grounded in that memory.

Nothing is seeded or scanned at startup — both are network-bound and manual,
so a flaky connection can never stop the server booting.

Run it:

    ./.venv/bin/uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import config, store
from agent import GeminiError, generate_action_pair, run_scan
from memory import EverOSError, recall_account, seed_from_accounts
from metrics import compute_metrics
from scoring import Account, apply_score
from seed import generate_accounts

BASE_DIR = Path(__file__).resolve().parent
CACHE_PATH = BASE_DIR / "demo_cache.json"

# --offline serves everything from demo_cache.json and makes zero network
# calls. --demo seeds memory and runs one scan at startup so the dashboard is
# already populated before you present.
OFFLINE = "--offline" in sys.argv
DEMO = "--demo" in sys.argv


def _cache_read() -> Dict[str, Any]:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _cache_write(key: str, value: Any) -> None:
    """Remember the last good result so --offline can replay it."""
    data = _cache_read()
    data[key] = value
    try:
        CACHE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        print("[aligned] could not write %s: %s" % (CACHE_PATH, exc), file=sys.stderr)


def _cache_require(key: str, what: str) -> Any:
    """Offline replay, or a clear error naming what is missing."""
    data = _cache_read()
    if key not in data:
        raise HTTPException(
            status_code=503,
            detail=("Offline mode: no cached %s in %s. Run once online without "
                    "--offline to populate the cache." % (what, CACHE_PATH.name)),
        )
    return data[key]

DASHBOARD_CANDIDATES = ("dashboard.html", "dashboard_3.html")
"""Filenames to serve at /, in preference order.

dashboard_3.html is the name the frontend was committed under; dashboard.html
is here so renaming it later doesn't silently break the page. Serving the UI
from the API's own origin also means the browser never makes a cross-origin
request at all — CORS stays wide open for a teammate opening the file directly,
but the happy path doesn't depend on it.
"""


def _dashboard_path():
    """The dashboard file to serve, or None if neither name is present."""
    for name in DASHBOARD_CANDIDATES:
        candidate = BASE_DIR / name
        if candidate.exists():
            return candidate
    return None


def _banner() -> str:
    """One glance: what is live, what still needs doing, before you demo.

    Reads only in-process state — no network calls, so this can never be the
    reason the server fails to boot.
    """
    accounts = store.all_accounts()
    mix: Dict[str, int] = {}
    for account in accounts:
        mix[account["status"]] = mix.get(account["status"], 0) + 1
    portfolio_arr = sum(float(a["arr_usd"]) for a in accounts)

    everos = (
        "LIVE   %s" % config.EVERMIND_BASE_URL
        if config.EVEROS_ENABLED
        else "OFF    (EVERMIND_API_KEY is blank)"
    )
    gemini = (
        "LIVE   %s" % config.GEMINI_DRAFT_MODEL
        if config.GEMINI_API_KEY
        else "OFF    (GEMINI_API_KEY is blank)"
    )
    if store.MEMORIES_SEEDED:
        seeded = "%d written this session" % store.MEMORIES_SEEDED
    else:
        seeded = "NOT YET -> POST /api/seed-memory"
    actions = (
        "%d drafted this session" % len(store.ACTIONS)
        if store.ACTIONS
        else "none yet -> POST /api/scan"
    )

    width = 74
    rows = [
        ("accounts", "%d loaded  (%d healthy / %d churn_risk / %d upsell_ready)"
         % (len(accounts), mix.get("healthy", 0), mix.get("churn_risk", 0), mix.get("upsell_ready", 0))),
        ("portfolio", "${:,.0f} ARR under management (simulated)".format(portfolio_arr)),
        ("EverOS", everos),
        ("Gemini", gemini),
        ("memories", seeded),
        ("actions", actions),
    ]
    lines = ["", "=" * width, "  TRACTION — autonomous retention + upsell agent".ljust(width)]
    lines.append("-" * width)
    for label, value in rows:
        lines.append("  %-11s %s" % (label, value))
    lines.append("=" * width)
    return "\n".join(lines)


def _public(account: Account) -> Dict[str, Any]:
    """The wire form of an account.

    Every field here is modeled, and ``mrr_usd``/``arr_usd`` are money, so the
    payload is tagged ``simulated``. Set here as well as in seed.py on purpose:
    no future edit to the seed data can quietly ship a money figure that reads
    as observed.
    """
    payload = dict(account)
    payload["simulated"] = True
    return payload


@asynccontextmanager
async def lifespan(app: FastAPI):
    accounts = generate_accounts()
    for account in accounts:
        apply_score(account)
    store.load_accounts(accounts)

    # Deliberately NOT seeding or scanning here. Both are network-bound; if
    # either ran at startup a flaky connection would stop the server booting.
    if OFFLINE:
        print("[aligned] OFFLINE MODE — serving %s, zero network calls."
              % CACHE_PATH.name, file=sys.stderr)
        cached = _cache_read().get("scan") or []
        for action in cached:
            store.save_action(action)
        if cached:
            print("[aligned] replayed %d cached actions" % len(cached), file=sys.stderr)

    if DEMO and not OFFLINE:
        print("[aligned] --demo: seeding memory then running one scan...", file=sys.stderr)
        try:
            store.MEMORIES_SEEDED += seed_from_accounts(store.all_accounts())
        except EverOSError as exc:
            print("[aligned] --demo seeding failed: %s" % exc, file=sys.stderr)
        try:
            for action in run_scan(store.all_accounts()):
                store.save_action(action)
            _cache_write("scan", store.all_actions())
        except (GeminiError, EverOSError) as exc:
            print("[aligned] --demo scan failed: %s" % exc, file=sys.stderr)

    print(_banner(), file=sys.stderr)

    yield

    store.reset()


app = FastAPI(
    title="Aligned",
    description=(
        "Autonomous account retention + upsell agent for B2B SaaS. "
        "All accounts are simulated; responses carrying money figures are "
        "tagged \"simulated\": true."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Wide open: the UI is a local file, so its origin is "null" or a file:// path
# that no allowlist would match. Credentials must stay off for "*" to be valid.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    """The dashboard, or a hint about where to put it."""
    dashboard = _dashboard_path()
    if dashboard is not None:
        return FileResponse(dashboard, media_type="text/html")

    return JSONResponse(
        {
            "ok": False,
            "service": "aligned",
            "message": (
                "No dashboard found. Save it next to api.py as one of %s and "
                "reload this page. The API itself is up — try /api/accounts."
                % " or ".join(DASHBOARD_CANDIDATES)
            ),
            "expected_path": str(BASE_DIR / DASHBOARD_CANDIDATES[0]),
            "endpoints": [
                "GET  /api/health",
                "GET  /api/accounts",
                "GET  /api/accounts/{account_id}",
                "POST /api/seed-memory",
                "POST /api/scan",
                "GET  /api/actions",
                "GET  /api/metrics",
                "POST /api/compare/{account_id}",
                "POST /api/reset",
            ],
        }
    )


@app.get("/pricing.html", include_in_schema=False)
def pricing():
    """The pricing page. Static, and deliberately has no checkout wired to it."""
    path = BASE_DIR / "pricing.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="pricing.html is not next to api.py")
    return FileResponse(path, media_type="text/html")


@app.get("/api/health")
def health():
    """Liveness, account count, and which subsystems are configured."""
    return {
        "ok": True,
        "accounts": len(store.ACCOUNTS),
        "everos": bool(config.EVEROS_ENABLED),
        "gemini": bool(config.GEMINI_API_KEY),
        "offline": OFFLINE,
    }


@app.get("/api/accounts")
def list_accounts():
    """Every account, scored, riskiest first."""
    return [_public(account) for account in store.all_accounts()]


@app.post("/api/seed-memory")
def seed_memory():
    """Write every account's profile / usage / ticket memories into EverOS.

    Manual on purpose, and not idempotent — running it twice writes everything
    twice. Run it once after boot.
    """
    if OFFLINE:
        raise HTTPException(status_code=503,
                            detail="Offline mode: seeding needs the network. Restart without --offline.")
    started = time.perf_counter()
    try:
        written = seed_from_accounts(store.all_accounts())
    except EverOSError as exc:
        # 503 rather than 500: the API is fine, its memory backend is not.
        raise HTTPException(status_code=503, detail=str(exc))

    store.MEMORIES_SEEDED += written
    print(_banner(), file=sys.stderr)
    return {
        "memories_written": written,
        "elapsed_ms": int(round((time.perf_counter() - started) * 1000)),
    }


@app.get("/api/metrics")
def metrics():
    """Portfolio and scan numbers, including the ROI headline."""
    return compute_metrics(store.all_accounts(), store.all_actions())


@app.post("/api/scan")
def scan():
    """Score the portfolio and draft outreach for every flagged account.

    Sequential and Gemini-bound, so it takes a while — watch the server console
    for per-account progress.
    """
    if OFFLINE:
        actions = _cache_require("scan", "scan result")
        for action in actions:
            store.save_action(action)
        return actions

    try:
        actions = run_scan(store.all_accounts())
    except GeminiError as exc:
        # 503: the API is healthy, its model backend is not.
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        # config.require_env raises this when GEMINI_API_KEY is blank.
        raise HTTPException(status_code=503, detail=str(exc))

    for action in actions:
        store.save_action(action)
    _cache_write("scan", actions)
    return actions


@app.get("/api/actions")
def list_actions():
    """Every action drafted this session, newest first."""
    return sorted(
        store.all_actions(),
        key=lambda a: (a.get("triggered_at", ""), a.get("action_id", "")),
        reverse=True,
    )


@app.post("/api/compare/{account_id}")
def compare(account_id: str):
    """The A/B proof: same outreach drafted with and without EverOS memory.

    Same model, same temperature, same instructions. The only variable is
    whether the retrieved snippets were in the prompt.
    """
    account = store.get_account(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="No account with id '%s'." % account_id)

    cache_key = "compare:%s" % account_id
    if OFFLINE:
        return _cache_require(cache_key, "comparison for %s" % account_id)

    try:
        pair = generate_action_pair(account)
    except GeminiError as exc:
        raise HTTPException(status_code=503, detail="Gemini: %s" % exc)
    except EverOSError as exc:
        raise HTTPException(status_code=503, detail="EverOS: %s" % exc)

    _cache_write(cache_key, pair)
    return pair


@app.post("/api/reset")
def reset_actions():
    """Clear drafted actions so the scan can be re-run live and clean.

    Deliberately keeps accounts AND EverOS memories — only the actions table
    is emptied. actions.jsonl on disk is left alone as the audit trail.
    """
    cleared = len(store.ACTIONS)
    store.ACTIONS.clear()
    return {"ok": True, "actions_cleared": cleared,
            "accounts_kept": len(store.ACCOUNTS),
            "memories_kept": store.MEMORIES_SEEDED,
            "note": "EverOS memories untouched; re-run POST /api/scan"}


@app.get("/api/accounts/{account_id}")
def get_account(account_id: str):
    """One account by id, with whatever EverOS remembers about it."""
    account = store.get_account(account_id)
    if account is None:
        known = store.known_account_ids()
        raise HTTPException(
            status_code=404,
            detail=(
                "No account with id '%s'. %d accounts are loaded, ids %s through %s. "
                "GET /api/accounts for the full list."
                % (
                    account_id,
                    len(known),
                    known[0] if known else "(none)",
                    known[-1] if known else "(none)",
                )
            ),
        )

    payload = _public(account)
    # Empty list when EverOS is disabled or unreachable — never fatal.
    payload["memories"] = recall_account(account_id, account["company_name"], limit=8)
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
