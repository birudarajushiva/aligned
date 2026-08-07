"""Portfolio-level numbers — the ones that go on the last slide.

Every figure here is derived from the scored portfolio and the actions a scan
actually produced. Nothing is floored, capped, or rounded up: if the ROI is
unimpressive this module says so, because finding that out on stage is worse
than finding it out now.

A note on what ``roi_multiple`` does and does not claim. It is:

    at-risk ARR *surfaced* per dollar of subscription

It is NOT "ARR saved". Aligned flags the account and drafts the email; whether
the customer stays is down to the human who sends it. That distinction is the
first thing a sharp judge will push on, so the honest framing is "we surfaced
$60k of at-risk ARR for $2,388/yr", not "we saved $60k".
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import config
from agent import arr_at_stake
from scoring import Account

Action = Dict[str, Any]


def compute_metrics(accounts: Sequence[Account], actions: Sequence[Action]) -> Dict[str, Any]:
    """Roll the portfolio and the last scan into one dict.

    Works before a scan has run: the ARR and status figures come from the
    accounts, so only the token/cost fields sit at zero until you scan.
    """
    accounts = list(accounts)
    actions = list(actions)

    status_counts = {"healthy": 0, "churn_risk": 0, "upsell_ready": 0}
    for account in accounts:
        status = account.get("status")
        if status in status_counts:
            status_counts[status] += 1

    arr_at_risk = sum(
        float(a["arr_usd"]) for a in accounts if a.get("status") == "churn_risk"
    )
    # Same expansion maths the agent puts on an Action — one definition, so the
    # headline number and the per-action numbers can never drift apart.
    arr_upsell_potential = sum(
        arr_at_stake(a, "upsell") for a in accounts if a.get("status") == "upsell_ready"
    )
    portfolio_arr = sum(float(a["arr_usd"]) for a in accounts)

    tokens_used = sum(int(action.get("tokens_used", 0)) for action in actions)
    # float() explicitly: sum([]) is int 0, and round(0, 6) stays int 0, which
    # would make cost_usd change JSON type depending on whether a scan has run.
    cost_usd = float(sum(float(action.get("cost_usd", 0.0)) for action in actions))

    accounts_total = len(accounts)
    cost_per_account = cost_usd / accounts_total if accounts_total else 0.0

    # The product's own annual price is the denominator: what a customer pays
    # us for a year of this. Guarded only against a zero price in config.
    annual_price = float(config.PRICE_PER_MONTH) * 12
    roi_multiple = (arr_at_risk / annual_price) if annual_price else 0.0

    return {
        "accounts_total": accounts_total,
        "healthy": status_counts["healthy"],
        "churn_risk": status_counts["churn_risk"],
        "upsell_ready": status_counts["upsell_ready"],
        "arr_at_risk_usd": round(arr_at_risk, 2),
        "arr_upsell_potential_usd": round(arr_upsell_potential, 2),
        "portfolio_arr_usd": round(portfolio_arr, 2),
        "actions_generated": len(actions),
        "tokens_used": tokens_used,
        "cost_usd": float(round(cost_usd, 6)),
        "cost_per_account_usd": float(round(cost_per_account, 6)),
        "price_per_month_usd": config.PRICE_PER_MONTH,
        "roi_multiple": round(roi_multiple, 2),
        "simulated": True,
    }
