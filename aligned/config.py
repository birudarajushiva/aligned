"""Configuration and tunable constants for Traction.

This is the ONLY module that reads environment variables. Everything else
imports from here, so there is exactly one place to look when a key is wrong.

Nothing is read at call time and nothing raises at import time: this stage
makes no outbound calls, so the whole app must boot with all three keys
blank. When a later stage actually needs a key, it calls ``require_env()``,
which fails loudly and names the exact variable that is missing.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Scoring knobs ──────────────────────────────────────────────────────────
# These are the numbers to reach for when tuning the demo. Every rule in
# scoring.py reads from here; no thresholds are inlined at the call site.

CHURN_INACTIVE_DAYS = 7
"""Days since last login above which an account is flagged as churn risk."""

CHURN_NEGATIVE_TICKETS = 2
"""How many consecutive most-recent tickets must be negative to flag churn."""

UPSELL_USAGE_THRESHOLD = 500
"""Actions in the last 30 days above which an account is upsell-ready."""

UPSELL_SEAT_UTILIZATION = 0.85
"""Seat utilization above which an account is upsell-ready.

Reserved, and deliberately not wired into scoring yet. The frozen Account
contract carries ``seats`` (seats purchased) but no count of seats actually
occupied, so there is no honest way to compute utilization from the data we
have. Inventing an occupied-seat number to make this knob work would be
exactly the kind of modeled-as-observed figure the project rules forbid.
Wire it up the day an ``active_seats`` field lands on the contract.
"""

PRICE_PER_MONTH = 199
"""What TRACTION itself costs a customer, per month.

This is our price, not our customers' prices. It is the denominator of
``roi_multiple``: $199 x 12 = $2,388/year of subscription, against the at-risk
ARR we surface. Do not confuse it with PLAN_MONTHLY_LIST_USD below, which is
the price book of the *simulated SaaS product* our demo accounts are buying.
The two are unrelated; that Starter also lists at $199 is a coincidence of the
demo data, not a link.
"""

# ── Plan ladder, for sizing an upsell ──────────────────────────────────────
# The price book of the simulated product our demo accounts subscribe to. Edit
# these two dicts to change every expansion number the agent produces. Like
# everything else in the demo portfolio they are modeled, not observed, which
# is why every Action carries "simulated": true.

PLAN_LADDER: Dict[str, Optional[str]] = {
    "Starter": "Growth",
    "Growth": "Enterprise",
    "Enterprise": None,  # top of the ladder — expansion is seats, not tier
}

PLAN_MONTHLY_LIST_USD: Dict[str, float] = {
    "Starter": 199.0,
    "Growth": 800.0,
    "Enterprise": 3000.0,
}

TOP_TIER_SEAT_EXPANSION = 0.25
"""How much an already-Enterprise account is modeled to expand by.

There is no higher tier to sell them, so expansion has to come from seats —
and the frozen Account contract has no signal for how many more seats they
need. Rather than invent one, this is a single explicit assumption (25% more
spend at their current rate) that you can see and change here.
"""

# ── Gemini pricing ─────────────────────────────────────────────────────────
# USD per 1,000,000 tokens, from https://ai.google.dev/gemini-api/docs/pricing
# (paid tier, text input). Verified 2026-08-07. One place to correct.
#
# IMPORTANT: on the AI Studio FREE tier these calls cost $0.00. The cost_usd
# on an Action is therefore "what this would have cost at paid-tier rates",
# not money that changed hands. It is a modeled figure like every other number
# in the demo — never present it as spend.

GEMINI_PRICING_USD_PER_MTOK: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}

GEMINI_DRAFT_MODEL = "gemini-2.5-flash"
"""Drafting outreach — the quality-sensitive call."""

GEMINI_CLASSIFY_MODEL = "gemini-2.5-flash-lite"
"""Cheap model for classification-shaped work. See traction/agent.py: stage 3
has no LLM classification step (grounding is checked deterministically), so
this is wired and ready rather than in use."""

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
GEMINI_RPM_SLEEP_SECONDS = 0.5
"""Free tier is ~10 RPM. Sleep this long between calls."""

# ── Trend calendar ─────────────────────────────────────────────────────────
# ``usage_trend`` is 6 monthly counts, oldest first. These labels let a reason
# string say "usage down 61% since March" instead of "since 5 months ago".
#
# They are fixed constants rather than derived from today's date on purpose:
# a demo rehearsed on Tuesday and given on Thursday must read identically.

TREND_MONTHS: List[str] = ["March", "April", "May", "June", "July", "August"]

# ── Environment ────────────────────────────────────────────────────────────

EVERMIND_API_KEY: str = os.getenv("EVERMIND_API_KEY", "").strip()
EVERMIND_BASE_URL: str = os.getenv("EVERMIND_BASE_URL", "").strip() or "https://api.evermind.ai"
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()

_ENV: Dict[str, str] = {
    "EVERMIND_API_KEY": EVERMIND_API_KEY,
    "EVERMIND_BASE_URL": EVERMIND_BASE_URL,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}

# ── EverOS availability ────────────────────────────────────────────────────
# Default on. Forced off when there is no API key, so the rest of the app can
# branch on one boolean instead of re-checking the key everywhere. With this
# False, recall returns an empty list and the stage 1 dashboard still serves
# every account — memory is an enhancement, not a hard dependency.

EVEROS_ENABLED = True

if not EVERMIND_API_KEY:
    EVEROS_ENABLED = False
    # stderr, not stdout: `python -c "...json.dumps(generate_accounts())" > out.json`
    # has to stay pipeable, and a banner on stdout would corrupt the JSON.
    _warning = [
        "=" * 78,
        "  WARNING: EVERMIND_API_KEY is blank — EverOS is DISABLED.",
        "",
        "  Accounts, scoring and the dashboard all still work.",
        "  What will NOT work:",
        "    - POST /api/seed-memory   returns 503 and writes nothing",
        '    - GET  /api/accounts/{id} returns "memories": []',
        "",
        "  Fix: put your key in .env next to api.py, then restart:",
        "      EVERMIND_API_KEY=sk-...",
        "  Get a key at https://everos.evermind.ai",
        "=" * 78,
    ]
    print("\n".join(_warning), file=sys.stderr)


def require_env(name: str) -> str:
    """Return the value of ``name``, or raise naming the exact variable.

    Call this at the point of use, not at import time, so that a stage which
    does not need a given key still runs with that key blank.
    """
    if name not in _ENV:
        raise RuntimeError(
            "Unknown environment variable %r. Traction reads only: %s"
            % (name, ", ".join(sorted(_ENV)))
        )
    value = _ENV[name]
    if not value:
        raise RuntimeError(
            "Missing required environment variable: %s. "
            "Set it in your shell or add it to the .env file next to api.py "
            "(see .env.example)." % name
        )
    return value


def missing_env() -> List[str]:
    """Names of the environment variables that are currently blank.

    Reporting only — nothing in this stage treats a blank key as an error.
    """
    return sorted(name for name, value in _ENV.items() if not value)
