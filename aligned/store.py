"""In-memory store. No database, no ORM.

Accounts live in a module-level dict for the life of the process; actions are
kept in memory *and* appended to ``actions.jsonl`` next to api.py, so the audit
trail of what the agent did survives a restart even though the portfolio does
not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from scoring import Account

ACCOUNTS: Dict[str, Account] = {}
"""account_id -> account, as served by the API."""

ACTIONS: Dict[str, Dict[str, Any]] = {}
"""action_id -> action taken by the agent (drafts, sends, snoozes)."""

ACTIONS_LOG_PATH = Path(__file__).resolve().parents[1] / "actions.jsonl"
"""Append-only log, one JSON object per line, next to api.py."""

MEMORIES_SEEDED = 0
"""How many memories this process has written to EverOS.

Session-scoped, and deliberately not a check against EverOS itself — the
startup banner has to render without touching the network.
"""


def load_accounts(accounts: Iterable[Account]) -> int:
    """Replace the portfolio wholesale. Returns how many accounts are loaded."""
    ACCOUNTS.clear()
    for account in accounts:
        ACCOUNTS[account["account_id"]] = account
    return len(ACCOUNTS)


def get_account(account_id: str) -> Optional[Account]:
    """One account, or None if there is no such id."""
    return ACCOUNTS.get(account_id)


def all_accounts() -> List[Account]:
    """Every account, riskiest first.

    Ties break on account_id so the ordering is stable run to run — the demo
    scrolls the same list every time.
    """
    return sorted(
        ACCOUNTS.values(),
        key=lambda a: (-int(a.get("risk_score", 0)), a.get("account_id", "")),
    )


def known_account_ids() -> List[str]:
    """Sorted account ids, for building a useful 404."""
    return sorted(ACCOUNTS)


def save_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Record one action in memory and append it to actions.jsonl.

    Stores exactly what it is given apart from filling in a missing
    ``action_id`` — no hidden timestamps, so a caller that needs one records
    it deliberately and the log stays reproducible.
    """
    stored = dict(action)
    action_id = str(stored.get("action_id") or "act_%04d" % (len(ACTIONS) + 1))
    stored["action_id"] = action_id
    ACTIONS[action_id] = stored

    ACTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ACTIONS_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stored, ensure_ascii=False, sort_keys=True) + "\n")

    return stored


def get_action(action_id: str) -> Optional[Dict[str, Any]]:
    """One recorded action, or None."""
    return ACTIONS.get(action_id)


def all_actions() -> List[Dict[str, Any]]:
    """Every recorded action, in the order it was saved."""
    return list(ACTIONS.values())


def reset() -> None:
    """Empty both dicts. Does not touch actions.jsonl on disk."""
    global MEMORIES_SEEDED
    ACCOUNTS.clear()
    ACTIONS.clear()
    MEMORIES_SEEDED = 0
