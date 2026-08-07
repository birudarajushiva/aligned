"""Deterministic account scoring. No LLM anywhere in this file.

The one-sentence version: an account is a churn risk if it went quiet, its
usage fell off a cliff, or its last two support tickets were both unhappy —
and it is upsell-ready if it is doing none of that while using the product
harder every month.

Same input always produces the same status, the same integer, and the same
sentence. That matters twice: the demo has to be repeatable, and when a judge
asks "why was that one flagged?", the answer is a rule with a number in it,
not a model's opinion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from config import (
    CHURN_INACTIVE_DAYS,
    CHURN_NEGATIVE_TICKETS,
    TREND_MONTHS,
    UPSELL_USAGE_THRESHOLD,
)

Account = Dict[str, Any]
"""An account as described by the shared data contract (see aligned/seed.py)."""

# ── Tunables local to the risk formula ─────────────────────────────────────

CHURN_DECLINE_PCT = 40.0
"""Percent drop from the usage peak that counts as churn risk."""

UPSELL_RISING_MONTHS = 3
"""Consecutive month-over-month usage increases that count as upsell-ready."""

INACTIVITY_FULL_RISK_DAYS = 30
"""Days of silence that max out the inactivity component of risk_score."""

WEIGHT_INACTIVITY = 0.40
WEIGHT_DECLINE = 0.35
WEIGHT_SENTIMENT = 0.25

SENTIMENT_PRESSURE = {"positive": 0.0, "neutral": 0.4, "negative": 1.0}
"""How much each sentiment pushes risk up, before recency weighting."""

SENTIMENT_RECENCY_DECAY = 0.5
"""Each older ticket counts half as much as the one after it."""


# ── Signal helpers (public: the API layer and later stages reuse these) ─────


def tickets_newest_first(account: Account) -> List[Dict[str, Any]]:
    """Support tickets sorted newest first. Dates are ISO, so string sort is
    date sort."""
    tickets = account.get("support_tickets") or []
    return sorted(tickets, key=lambda t: str(t.get("date", "")), reverse=True)


def usage_decline(trend: Sequence[int]) -> Tuple[float, int]:
    """Percent the latest month is down from the series peak, and the peak's
    index.

    Returns ``(0.0, index)`` when the latest month *is* the peak — this never
    reports a negative decline, so a growing account contributes nothing to
    the decline component of risk.
    """
    if not trend:
        return 0.0, 0
    peak = max(trend)
    peak_index = list(trend).index(peak)
    if peak <= 0:
        return 0.0, peak_index
    drop = (peak - trend[-1]) / peak * 100.0
    return max(0.0, drop), peak_index


def rising_streak(trend: Sequence[int]) -> int:
    """How many consecutive month-over-month increases the series ends on.

    ``[100, 90, 120, 140, 160]`` -> 3. Trailing rather than best-anywhere:
    an account that grew last spring and has been flat since is not a live
    upsell.
    """
    streak = 0
    for older, newer in zip(reversed(trend[:-1]), reversed(trend[1:])):
        if newer > older:
            streak += 1
        else:
            break
    return streak


def negative_ticket_streak(account: Account) -> int:
    """How many of the most recent tickets in a row came back negative."""
    streak = 0
    for ticket in tickets_newest_first(account):
        if ticket.get("sentiment") == "negative":
            streak += 1
        else:
            break
    return streak


def sentiment_pressure(account: Account) -> float:
    """Recency-weighted unhappiness across all tickets, on a 0.0-1.0 scale.

    The newest ticket carries full weight, the one before it half, and so on,
    so a bad week last month does not outweigh a bad week yesterday.
    """
    tickets = tickets_newest_first(account)
    if not tickets:
        return 0.0
    total = 0.0
    weight_sum = 0.0
    for index, ticket in enumerate(tickets):
        weight = SENTIMENT_RECENCY_DECAY ** index
        total += weight * SENTIMENT_PRESSURE.get(ticket.get("sentiment", "neutral"), 0.4)
        weight_sum += weight
    return total / weight_sum


# ── Reason-string helpers ──────────────────────────────────────────────────


def _days_phrase(days: int) -> str:
    if days <= 0:
        return "today"
    if days == 1:
        return "1 day"
    return "%d days" % days


def _peak_label(peak_index: int, trend_length: int) -> str:
    """"March", when the trend is the 6 months we have labels for."""
    if trend_length == len(TREND_MONTHS) and 0 <= peak_index < len(TREND_MONTHS):
        return TREND_MONTHS[peak_index]
    return "its peak"


# ── The scorer ─────────────────────────────────────────────────────────────


def risk_score(account: Account) -> int:
    """0-100 blend of inactivity (40%), usage decline (35%), sentiment (25%)."""
    trend = account.get("usage_trend") or []
    last_active_days = int(account.get("last_active_days", 0))

    inactivity = min(last_active_days / INACTIVITY_FULL_RISK_DAYS, 1.0) * 100.0
    decline = min(usage_decline(trend)[0], 100.0)
    sentiment = sentiment_pressure(account) * 100.0

    blended = (
        WEIGHT_INACTIVITY * inactivity
        + WEIGHT_DECLINE * decline
        + WEIGHT_SENTIMENT * sentiment
    )
    return int(round(max(0.0, min(100.0, blended))))


def score_account(account: Account) -> Tuple[str, int, str]:
    """Classify one account.

    Returns ``(status, risk_score, reason)`` where reason names the actual
    trigger with the actual number in it — the string goes straight on screen.
    """
    trend: List[int] = list(account.get("usage_trend") or [])
    last_active_days = int(account.get("last_active_days", 0))
    usage_30d = int(account.get("usage_count_30d", 0))

    decline_pct, peak_index = usage_decline(trend)
    peak_label = _peak_label(peak_index, len(trend))
    streak_up = rising_streak(trend)
    streak_negative = negative_ticket_streak(account)
    newest_tickets = tickets_newest_first(account)

    score = risk_score(account)

    # ── churn_risk: any one of the three triggers is enough ──
    churn_reasons: List[str] = []
    if last_active_days > CHURN_INACTIVE_DAYS:
        churn_reasons.append("No login in %s" % _days_phrase(last_active_days))
    if decline_pct > CHURN_DECLINE_PCT:
        churn_reasons.append("usage down %.0f%% since %s" % (decline_pct, peak_label))
    if streak_negative >= CHURN_NEGATIVE_TICKETS:
        churn_reasons.append(
            'last %d tickets negative ("%s")'
            % (min(streak_negative, CHURN_NEGATIVE_TICKETS), newest_tickets[0].get("summary", ""))
        )

    if churn_reasons:
        return "churn_risk", score, _sentence(churn_reasons)

    # ── upsell_ready: only for accounts that are not at risk ──
    upsell_reasons: List[str] = []
    if usage_30d > UPSELL_USAGE_THRESHOLD:
        upsell_reasons.append(
            "%d actions in the last 30 days (upsell line is %d)"
            % (usage_30d, UPSELL_USAGE_THRESHOLD)
        )
    if streak_up >= UPSELL_RISING_MONTHS:
        upsell_reasons.append(
            "usage up %d months straight (%d -> %d)"
            % (streak_up, trend[-(streak_up + 1)], trend[-1])
        )

    if upsell_reasons:
        return "upsell_ready", score, _sentence(upsell_reasons)

    # ── healthy: still say something with numbers in it ──
    if decline_pct < 0.5:
        peak_clause = "holding at %d/mo" % (trend[-1] if trend else usage_30d)
    else:
        peak_clause = "%.0f%% off its %s peak" % (decline_pct, peak_label)
    active_clause = (
        "Active today"
        if last_active_days <= 0
        else "Active %s ago" % _days_phrase(last_active_days)
    )
    healthy_reason = "%s; %d actions in the last 30 days; %s" % (
        active_clause,
        usage_30d,
        peak_clause,
    )
    return "healthy", score, healthy_reason


def _sentence(clauses: Sequence[str]) -> str:
    """Join trigger clauses into the one line that goes on screen."""
    joined = "; ".join(clauses)
    return joined[:1].upper() + joined[1:]


def apply_score(account: Account) -> Account:
    """Write ``status``, ``risk_score`` and ``reason`` onto the account.

    Pure with respect to the scoring inputs, so calling it twice is a no-op.
    """
    status, score, reason = score_account(account)
    account["status"] = status
    account["risk_score"] = score
    account["reason"] = reason
    return account
