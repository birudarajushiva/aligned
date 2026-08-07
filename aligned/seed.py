"""The simulated portfolio Aligned watches.

Everything in this module is invented. No account here corresponds to a real
company, and no number here was observed from a real product — which is why
every account carries ``"simulated": true`` from the moment it is created,
rather than having it bolted on at the API boundary.

Determinism is a hard requirement: the RNG is seeded with a fixed constant and
every account's status-bearing fields (activity, usage trend, tickets) are
authored literals, so a rehearsal and the live demo produce byte-identical
output. Nothing here reads today's date.

The shared data contract — these field names are frozen, a teammate's UI is
built against them:

    { "account_id": str,            # "acc_001"
      "company_name": str,
      "plan": "Starter"|"Growth"|"Enterprise",
      "mrr_usd": float,
      "arr_usd": float,             # mrr * 12
      "seats": int,
      "signup_date": "YYYY-MM-DD",
      "last_active_days": int,      # days since last login
      "usage_count_30d": int,       # actions taken in last 30 days
      "usage_trend": [int, ...],    # 6 monthly counts, oldest first
      "feature_usage": {feature: count},
      "support_tickets": [{"date","summary","sentiment"}],
      "status": "healthy"|"churn_risk"|"upsell_ready",
      "risk_score": int,            # 0-100
      "reason": str }               # human-readable trigger
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from scoring import Account, apply_score

SEED = 20260807
"""Fixed RNG seed. Change this and the demo changes. Don't change this."""

MRR_BANDS: Dict[str, Tuple[float, float]] = {
    "Starter": (99.0, 299.0),
    "Growth": (400.0, 1200.0),
    "Enterprise": (2000.0, 4000.0),
}

FEATURE_CATALOG = (
    "csv_export",
    "api_calls",
    "team_dashboard",
    "slack_integration",
    "sso_login",
    "custom_reports",
    "bulk_import",
    "webhooks",
)

FEATURE_SPLIT = (0.38, 0.24, 0.16, 0.11, 0.07, 0.04)
"""Share of an account's activity that lands on its 1st, 2nd, ... feature.

Real usage is lopsided — one or two features carry most of the volume — so a
flat split would look synthetic on screen.
"""

TARGET_MIX = {"healthy": 6, "churn_risk": 3, "upsell_ready": 3}
"""What the portfolio must score to. Enforced by verify_portfolio()."""


# ── The portfolio ──────────────────────────────────────────────────────────
#
# Status is never assigned here. Each profile is authored so that the rules in
# scoring.py reach the intended verdict on their own — a churning account is
# one whose numbers actually fell apart. verify_portfolio() fails loudly if the
# data drifts away from TARGET_MIX, which is the signal to fix the data rather
# than to loosen the scorer.
#
# acc_001 is the demo's hero: a $48k/yr Enterprise logistics account that has
# gone quiet, watched its usage fall 61% off its March peak, and filed two
# angry tickets in a row about the same unfixed bug.

PROFILES: List[Dict[str, Any]] = [
    {
        "account_id": "acc_001",
        "company_name": "Northwind Logistics",
        "plan": "Enterprise",
        "mrr_usd": 4000.0,
        "seats": 120,
        "signup_date": "2023-02-14",
        "last_active_days": 23,
        "usage_trend": [4210, 3980, 3420, 2610, 1980, 1640],
        "features": ["api_calls", "csv_export", "team_dashboard", "sso_login", "custom_reports"],
        "support_tickets": [
            {
                "date": "2026-05-19",
                "summary": "Asked whether scheduled exports can land straight in their S3 bucket",
                "sentiment": "neutral",
            },
            {
                "date": "2026-06-30",
                "summary": "Export to CSV timing out on files over 10k rows",
                "sentiment": "negative",
            },
            {
                "date": "2026-07-21",
                "summary": "Third week without a fix on the export timeouts, escalating to their VP",
                "sentiment": "negative",
            },
        ],
    },
    {
        "account_id": "acc_002",
        "company_name": "Rivet Health",
        "plan": "Growth",
        "mrr_usd": 890.0,
        "seats": 28,
        "signup_date": "2024-08-05",
        "last_active_days": 14,
        "usage_trend": [640, 690, 610, 470, 330, 300],
        "features": ["bulk_import", "team_dashboard", "custom_reports", "csv_export"],
        "support_tickets": [
            {
                "date": "2026-04-11",
                "summary": "Onboarding call went well, whole ops team is on the shared dashboard",
                "sentiment": "positive",
            },
            {
                "date": "2026-06-22",
                "summary": "Bulk import silently skipped 300 rows with no error shown",
                "sentiment": "negative",
            },
            {
                "date": "2026-07-15",
                "summary": "Still blocked on the bulk import issue, asked what our SLA actually is",
                "sentiment": "negative",
            },
        ],
    },
    {
        "account_id": "acc_003",
        "company_name": "Cobalt Freight Systems",
        "plan": "Starter",
        "mrr_usd": 149.0,
        "seats": 5,
        "signup_date": "2025-11-03",
        "last_active_days": 31,
        "usage_trend": [120, 132, 118, 96, 74, 61],
        "features": ["csv_export", "team_dashboard", "slack_integration"],
        "support_tickets": [
            {
                "date": "2026-03-28",
                "summary": "Asked how to invite a second admin to the workspace",
                "sentiment": "neutral",
            },
        ],
    },
    {
        "account_id": "acc_004",
        "company_name": "Ardent Labs",
        "plan": "Enterprise",
        "mrr_usd": 2750.0,
        "seats": 60,
        "signup_date": "2024-01-22",
        "last_active_days": 1,
        "usage_trend": [1840, 2010, 2260, 2540, 2880, 3120],
        "features": [
            "api_calls",
            "webhooks",
            "custom_reports",
            "sso_login",
            "team_dashboard",
            "csv_export",
        ],
        "support_tickets": [
            {
                "date": "2026-06-02",
                "summary": "Asked about SAML setup for their security review",
                "sentiment": "neutral",
            },
            {
                "date": "2026-07-28",
                "summary": "Wants a sandbox workspace so their QA team can test against the API",
                "sentiment": "neutral",
            },
        ],
    },
    {
        "account_id": "acc_005",
        "company_name": "Halcyon Retail Group",
        "plan": "Growth",
        "mrr_usd": 1150.0,
        "seats": 34,
        "signup_date": "2024-06-18",
        "last_active_days": 2,
        "usage_trend": [410, 455, 520, 590, 680, 742],
        "features": [
            "custom_reports",
            "team_dashboard",
            "csv_export",
            "slack_integration",
            "bulk_import",
        ],
        "support_tickets": [
            {
                "date": "2026-07-09",
                "summary": "Walked through scheduling custom reports, said it saves their Monday",
                "sentiment": "positive",
            },
        ],
    },
    {
        "account_id": "acc_006",
        "company_name": "Quill & Bright",
        "plan": "Starter",
        "mrr_usd": 249.0,
        "seats": 9,
        "signup_date": "2025-09-12",
        "last_active_days": 3,
        "usage_trend": [96, 128, 150, 205, 268, 341],
        "features": ["slack_integration", "bulk_import", "csv_export", "team_dashboard"],
        "support_tickets": [
            {
                "date": "2026-06-14",
                "summary": "Asked whether Slack alerts can be routed to more than one channel",
                "sentiment": "neutral",
            },
            {
                "date": "2026-07-25",
                "summary": "Said the new bulk import cut their weekly prep in half",
                "sentiment": "positive",
            },
        ],
    },
    {
        "account_id": "acc_007",
        "company_name": "Fernbrook Analytics",
        "plan": "Growth",
        "mrr_usd": 640.0,
        "seats": 22,
        "signup_date": "2024-11-08",
        "last_active_days": 2,
        "usage_trend": [312, 298, 330, 316, 341, 327],
        "features": ["api_calls", "webhooks", "custom_reports", "csv_export"],
        "support_tickets": [
            {
                "date": "2026-05-30",
                "summary": "Asked if webhook payloads include the account owner's email",
                "sentiment": "neutral",
            },
        ],
    },
    {
        "account_id": "acc_008",
        "company_name": "Trellis Payroll",
        "plan": "Growth",
        "mrr_usd": 780.0,
        "seats": 26,
        "signup_date": "2024-03-19",
        "last_active_days": 5,
        "usage_trend": [402, 388, 415, 430, 421, 436],
        "features": ["team_dashboard", "bulk_import", "csv_export", "sso_login", "custom_reports"],
        "support_tickets": [
            {
                "date": "2026-06-05",
                "summary": "Team dashboard crawling on Monday mornings when everyone logs in",
                "sentiment": "negative",
            },
            {
                "date": "2026-06-27",
                "summary": "Confirmed Monday dashboard load times are back to normal",
                "sentiment": "positive",
            },
        ],
    },
    {
        "account_id": "acc_009",
        "company_name": "Marlowe Interiors",
        "plan": "Starter",
        "mrr_usd": 99.0,
        "seats": 3,
        "signup_date": "2026-01-16",
        "last_active_days": 1,
        "usage_trend": [44, 58, 71, 66, 78, 74],
        "features": ["csv_export", "team_dashboard", "slack_integration"],
        "support_tickets": [
            {
                "date": "2026-02-02",
                "summary": "Wanted to confirm CSV export is included on the Starter plan",
                "sentiment": "neutral",
            },
        ],
    },
    {
        "account_id": "acc_010",
        "company_name": "Grayson Dental Partners",
        "plan": "Starter",
        "mrr_usd": 129.0,
        "seats": 6,
        "signup_date": "2025-07-24",
        "last_active_days": 6,
        "usage_trend": [188, 176, 195, 181, 190, 184],
        "features": ["bulk_import", "csv_export", "team_dashboard"],
        "support_tickets": [
            {
                "date": "2026-04-17",
                "summary": "Asked for help importing last year's patient list",
                "sentiment": "neutral",
            },
            {
                "date": "2026-04-21",
                "summary": "Import worked off the CSV template, thanked support",
                "sentiment": "positive",
            },
        ],
    },
    {
        "account_id": "acc_011",
        "company_name": "Solstice Media Co",
        "plan": "Starter",
        "mrr_usd": 179.0,
        "seats": 8,
        "signup_date": "2025-05-09",
        "last_active_days": 4,
        "usage_trend": [236, 254, 241, 262, 248, 259],
        "features": ["slack_integration", "custom_reports", "csv_export", "team_dashboard"],
        "support_tickets": [
            {
                "date": "2026-07-02",
                "summary": "Asked whether scheduled reports can go to a shared inbox",
                "sentiment": "neutral",
            },
        ],
    },
    {
        "account_id": "acc_012",
        "company_name": "Pinehurst Supply Co",
        "plan": "Growth",
        "mrr_usd": 520.0,
        "seats": 18,
        "signup_date": "2025-02-27",
        "last_active_days": 7,
        "usage_trend": [366, 341, 352, 338, 349, 344],
        "features": ["webhooks", "api_calls", "csv_export", "team_dashboard", "slack_integration"],
        "support_tickets": [
            {
                "date": "2026-05-21",
                "summary": "Frustrated that webhook retries aren't configurable",
                "sentiment": "negative",
            },
            {
                "date": "2026-06-18",
                "summary": "Asked for a status page link to monitor API uptime",
                "sentiment": "neutral",
            },
            {
                "date": "2026-07-30",
                "summary": "Renewal contact changed, updated the billing email on file",
                "sentiment": "neutral",
            },
        ],
    },
]


def _feature_usage(rng: random.Random, features: List[str], usage_30d: int) -> Dict[str, int]:
    """Lopsided cumulative per-feature counts consistent with monthly volume.

    Counts are lifetime, not last-30-days, so they run several times the
    monthly figure — which is why they are allowed to exceed usage_count_30d.
    """
    months_of_history = rng.randint(4, 9)
    pool = max(usage_30d, 1) * months_of_history
    usage: Dict[str, int] = {}
    for index, feature in enumerate(features):
        share = FEATURE_SPLIT[index] * rng.uniform(0.85, 1.15)
        usage[feature] = max(1, int(round(pool * share)))
    return usage


def _validate_profile(profile: Dict[str, Any]) -> None:
    """Catch authoring mistakes at generation time, not on stage."""
    plan = profile["plan"]
    if plan not in MRR_BANDS:
        raise ValueError("%s: unknown plan %r" % (profile["account_id"], plan))

    low, high = MRR_BANDS[plan]
    mrr = profile["mrr_usd"]
    if not low <= mrr <= high:
        raise ValueError(
            "%s: %s MRR $%.0f is outside the $%.0f-$%.0f band"
            % (profile["account_id"], plan, mrr, low, high)
        )

    features = profile["features"]
    if not 3 <= len(features) <= 6:
        raise ValueError(
            "%s: uses %d features, expected 3-6" % (profile["account_id"], len(features))
        )
    unknown = [f for f in features if f not in FEATURE_CATALOG]
    if unknown:
        raise ValueError("%s: features not in catalog: %s" % (profile["account_id"], unknown))

    if len(profile["usage_trend"]) != 6:
        raise ValueError(
            "%s: usage_trend has %d months, expected 6"
            % (profile["account_id"], len(profile["usage_trend"]))
        )

    if not 1 <= len(profile["support_tickets"]) <= 4:
        raise ValueError(
            "%s: has %d support tickets, expected 1-4"
            % (profile["account_id"], len(profile["support_tickets"]))
        )


def generate_accounts(n: int = 12) -> List[Account]:
    """Build the simulated portfolio, scored and ready to serve.

    Deterministic: same seed, same profiles, same output on every run.
    """
    if n < 1:
        raise ValueError("generate_accounts(n=%d): n must be at least 1" % n)
    if n > len(PROFILES):
        raise ValueError(
            "generate_accounts(n=%d): only %d authored profiles exist. "
            "Add another entry to aligned.seed.PROFILES rather than generating "
            "filler, so the portfolio keeps reading like a real book of business."
            % (n, len(PROFILES))
        )

    rng = random.Random(SEED)
    accounts: List[Account] = []

    for profile in PROFILES[:n]:
        _validate_profile(profile)

        trend = list(profile["usage_trend"])
        usage_30d = trend[-1]
        # Round once, then derive ARR from the rounded figure, so the frozen
        # contract's arr_usd == mrr_usd * 12 holds exactly for any MRR — not
        # just the 2dp literals the portfolio happens to use today.
        mrr = round(float(profile["mrr_usd"]), 2)

        account: Account = {
            "account_id": profile["account_id"],
            "company_name": profile["company_name"],
            "plan": profile["plan"],
            "mrr_usd": mrr,
            "arr_usd": round(mrr * 12, 2),
            "seats": int(profile["seats"]),
            "signup_date": profile["signup_date"],
            "last_active_days": int(profile["last_active_days"]),
            "usage_count_30d": usage_30d,
            "usage_trend": trend,
            "feature_usage": _feature_usage(rng, list(profile["features"]), usage_30d),
            "support_tickets": [dict(ticket) for ticket in profile["support_tickets"]],
            # Modeled, not observed. Travels with the record so it cannot be
            # lost on the way to the UI.
            "simulated": True,
        }
        apply_score(account)
        accounts.append(account)

    if n == len(PROFILES):
        verify_portfolio(accounts)

    return accounts


def verify_portfolio(accounts: List[Account]) -> Dict[str, int]:
    """Assert the scored portfolio lands on TARGET_MIX. Returns the counts.

    If this raises, the fix is in the account data above — not in scoring.py.
    """
    counts = {"healthy": 0, "churn_risk": 0, "upsell_ready": 0}
    for account in accounts:
        counts[account["status"]] = counts.get(account["status"], 0) + 1

    if counts != TARGET_MIX:
        raise AssertionError(
            "Portfolio scored %s, expected %s. Adjust the account data in "
            "aligned/seed.py, not the rules in aligned/scoring.py."
            % (counts, TARGET_MIX)
        )

    hero = [
        a
        for a in accounts
        if a["status"] == "churn_risk" and a["plan"] == "Enterprise" and a["arr_usd"] >= 40000
    ]
    if not hero:
        raise AssertionError(
            "No high-ARR Enterprise account is flagged churn_risk. The demo "
            "needs one — see acc_001 in aligned/seed.py."
        )

    return counts
