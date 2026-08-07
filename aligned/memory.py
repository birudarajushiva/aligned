"""EverOS (Evermind Cloud) as Traction's persistent customer memory.

Talks to the **Cloud v1** API at https://api.evermind.ai — not the self-hosted
OSS server, and not the v2 endpoints.

Everything in here is built against the published OpenAPI contract for Cloud v1
(https://docs.evermind.ai/api-reference/openapi-cloud-v1.json). Three details
of that contract shape this module, and each one differs from what you might
expect from a plain key/value memory store:

1. There is NO ``metadata`` field. The string "metadata" does not appear
   anywhere in the v1 spec. Scoping is done with ``user_id`` (required, top
   level on write; inside ``filters`` on search) and ``session_id`` (optional,
   filterable with eq/in/gt/gte/lt/lte). So:

       user_id    = account_id     -> scopes retrieval to one customer
       session_id = event_type     -> scopes retrieval to usage/ticket/profile

   That is the documented mechanism and it does exactly what metadata was
   wanted for.

2. ``/api/v1/memories`` ingests *messages*, it does not store text. Each
   message needs ``role`` ("user" or "assistant"), ``timestamp`` (unix
   milliseconds) and ``content``. EverOS then runs extraction over them, so
   what comes back from search is extracted episodes/profiles, not the exact
   sentences written here. Writing good prose still matters — extraction
   quality depends on it — but recall is not verbatim playback.

3. ``async_mode`` defaults to **true** server-side, which returns 202 and
   queues the work. Anything written that way is not immediately searchable.
   Since the demo seeds and then immediately recalls, this module sends
   ``async_mode: false`` (see SYNC_WRITES) and takes the slower synchronous
   write in exchange for memories that are there when you ask for them.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

from traction import config
from traction.scoring import Account, rising_streak, usage_decline

# ── Wire contract (Cloud v1) ───────────────────────────────────────────────

ADD_PATH = "/api/v1/memories"
SEARCH_PATH = "/api/v1/memories/search"
GET_PATH = "/api/v1/memories/get"

EVENT_TYPES = ("usage", "ticket", "profile")
"""Values carried in session_id, so retrieval can scope to one kind of fact."""

SYNC_WRITES = True
"""Send async_mode=false.

Slower per write (extraction runs inline) but the memory is searchable the
moment the call returns. Flip to False only if you are willing to poll
GET /api/v1/tasks/{task_id} before demoing recall.
"""

SEARCH_METHOD = "hybrid"
"""One of keyword | vector | hybrid | agentic. Hybrid is the API default."""

SEARCH_MEMORY_TYPES = ("episodic_memory", "profile")
"""API default. Add "raw_message" to also get back unextracted text."""

# ── Politeness and failure handling ────────────────────────────────────────

WRITE_SLEEP_SECONDS = 0.2
MAX_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = (1, 2, 4, 8)
"""Waits between attempts. With MAX_ATTEMPTS=4 the 1/2/4 entries are used;
the 8 is the next step if MAX_ATTEMPTS is ever raised to 5."""

WRITE_TIMEOUT_SECONDS = 90.0
"""Generous: a synchronous write runs extraction before it answers."""

READ_TIMEOUT_SECONDS = 30.0

_client: Optional[httpx.Client] = None


class EverOSError(RuntimeError):
    """A call to EverOS failed. Raised loudly so seeding can never half-finish
    without you knowing."""


def _log(message: str) -> None:
    """Diagnostics go to stderr so stdout stays clean for piped JSON."""
    print("[everos] %s" % message, file=sys.stderr)


def _get_client() -> httpx.Client:
    """One pooled client for the whole process — seeding makes ~100 calls."""
    global _client
    if _client is None:
        _client = httpx.Client(base_url=config.EVERMIND_BASE_URL.rstrip("/"))
    return _client


def _post(path: str, payload: Dict[str, Any], *, label: str, timeout: float) -> Tuple[Dict[str, Any], float]:
    """POST with bearer auth, 429 backoff, and latency logging.

    Returns ``(response_json, elapsed_ms)``. Raises EverOSError on any
    non-retryable failure, on a 429 that outlives the retries, or on a
    network error that outlives them.
    """
    headers = {
        "Authorization": "Bearer %s" % config.require_env("EVERMIND_API_KEY"),
        "Content-Type": "application/json",
    }

    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            response = _get_client().post(path, json=payload, headers=headers, timeout=timeout)
        except httpx.RequestError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if attempt < MAX_ATTEMPTS:
                wait = RETRY_BACKOFF_SECONDS[attempt - 1]
                _log(
                    "%s network error after %.0fms (%s) — attempt %d/%d, retrying in %ds"
                    % (label, elapsed_ms, type(exc).__name__, attempt, MAX_ATTEMPTS, wait)
                )
                time.sleep(wait)
                continue
            raise EverOSError(
                "%s failed after %d attempts: %s: %s"
                % (label, MAX_ATTEMPTS, type(exc).__name__, exc)
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 429:
            if attempt < MAX_ATTEMPTS:
                wait = RETRY_BACKOFF_SECONDS[attempt - 1]
                _log(
                    "%s rate limited (429) after %.0fms — attempt %d/%d, retrying in %ds"
                    % (label, elapsed_ms, attempt, MAX_ATTEMPTS, wait)
                )
                time.sleep(wait)
                continue
            raise EverOSError(
                "%s still rate limited (429) after %d attempts. Seeding is "
                "incomplete — do not demo until this is resolved." % (label, MAX_ATTEMPTS)
            )

        if response.status_code >= 400:
            raise EverOSError(
                "%s failed with HTTP %d after %.0fms: %s"
                % (label, response.status_code, elapsed_ms, response.text[:500])
            )

        _log("%s -> %d in %.0fms" % (label, response.status_code, elapsed_ms))
        try:
            return response.json(), elapsed_ms
        except ValueError as exc:
            raise EverOSError(
                "%s returned HTTP %d with a non-JSON body: %s"
                % (label, response.status_code, response.text[:300])
            ) from exc

    raise EverOSError("%s exhausted %d attempts" % (label, MAX_ATTEMPTS))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _date_to_ms(date_str: str) -> Optional[int]:
    """"2026-06-30" -> unix ms at UTC midnight. None if unparseable."""
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return int(parsed.timestamp() * 1000)


# ── Write ──────────────────────────────────────────────────────────────────


def store_account_event(
    account_id: str,
    content: str,
    event_type: str,
    timestamp_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Write one memory about one account. POST /api/v1/memories.

    ``account_id`` becomes ``user_id`` and ``event_type`` becomes
    ``session_id`` — the two documented fields the search filter understands,
    which is what lets recall scope to a single customer.

    ``timestamp_ms`` defaults to now; pass the real event date for tickets so
    the memory sits at the right point on the customer's timeline.
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(
            "event_type %r is not one of %s" % (event_type, ", ".join(EVENT_TYPES))
        )
    if not config.EVEROS_ENABLED:
        raise EverOSError(
            "Cannot write to EverOS: EVERMIND_API_KEY is blank so EVEROS_ENABLED "
            "is False. Set the key in .env and restart before seeding."
        )

    payload = {
        "user_id": account_id,
        "session_id": event_type,
        "messages": [
            {
                "role": "user",
                "timestamp": timestamp_ms if timestamp_ms is not None else _now_ms(),
                "content": content,
            }
        ],
        "async_mode": not SYNC_WRITES,
    }

    body, elapsed_ms = _post(
        ADD_PATH,
        payload,
        label="write %s/%s" % (account_id, event_type),
        timeout=WRITE_TIMEOUT_SECONDS,
    )
    result = dict(body.get("data") or {})
    result["elapsed_ms"] = round(elapsed_ms, 1)
    return result


# ── Read ───────────────────────────────────────────────────────────────────


def _profile_snippet(profile: Dict[str, Any]) -> str:
    """Flatten a SearchProfileItem's profile_data into one readable line."""
    data = profile.get("profile_data")
    if isinstance(data, dict):
        parts = []
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                value = "; ".join(str(v) for v in value)
            parts.append("%s: %s" % (key, value))
        return " | ".join(parts)
    return str(data) if data else ""


def recall_account(account_id: str, query: str, limit: int = 5) -> List[str]:
    """Retrieve this account's memories. POST /api/v1/memories/search.

    Returns raw text snippets, best match first. Returns ``[]`` rather than
    raising when EverOS is disabled or unreachable — retrieval is on the read
    path that serves the dashboard, and stage 1's demo has to survive EverOS
    being down.
    """
    if not config.EVEROS_ENABLED:
        return []

    top_k = max(1, min(int(limit), 100))
    payload = {
        "query": query,
        "filters": {"user_id": account_id},
        "method": SEARCH_METHOD,
        "memory_types": list(SEARCH_MEMORY_TYPES),
        "top_k": top_k,
    }

    try:
        body, elapsed_ms = _post(
            SEARCH_PATH,
            payload,
            label="recall %s" % account_id,
            timeout=READ_TIMEOUT_SECONDS,
        )
    except EverOSError as exc:
        _log("recall %s failed, serving account without memories: %s" % (account_id, exc))
        return []

    data = body.get("data") or {}

    scored: List[Tuple[float, str]] = []
    for episode in data.get("episodes") or []:
        text = episode.get("episode") or episode.get("summary") or episode.get("subject")
        if text:
            scored.append((float(episode.get("score") or 0.0), str(text)))
    for profile in data.get("profiles") or []:
        text = _profile_snippet(profile)
        if text:
            scored.append((float(profile.get("score") or 0.0), text))
    for message in data.get("raw_messages") or []:
        text = message.get("content") or message.get("text")
        if isinstance(text, str) and text:
            scored.append((float(message.get("score") or 0.0), text))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    snippets = [text for _, text in scored][:top_k]
    _log("recall %s -> %d snippets in %.0fms" % (account_id, len(snippets), elapsed_ms))
    return snippets


# ── Prose. Retrieval quality lives or dies here. ───────────────────────────


def _money(value: float) -> str:
    return "${:,.0f}".format(value)


def _article(word: str) -> str:
    """"an Enterprise", "a Growth" — the prose is read by an LLM later, and
    later still by a customer, so it should not read like a mail merge."""
    return "an" if word[:1].upper() in "AEIOU" else "a"


def _possessive(name: str) -> str:
    """"Ardent Labs'" not "Ardent Labs's"."""
    return name + ("'" if name.endswith(("s", "S")) else "'s")


def _profile_prose(account: Account) -> str:
    return (
        "{company} is {article} {plan} plan customer. They pay {mrr} per month, which is "
        "{arr} per year, and they have {seats} seats on the account. They signed up "
        "on {signup}. Their account id in our system is {account_id}. "
        "(Simulated demo data for the Traction prototype — {company} is not a real "
        "customer and these figures are modeled, not observed.)"
    ).format(
        company=account["company_name"],
        article=_article(account["plan"]),
        plan=account["plan"],
        mrr=_money(account["mrr_usd"]),
        arr=_money(account["arr_usd"]),
        seats=account["seats"],
        signup=account["signup_date"],
        account_id=account["account_id"],
    )


def _feature_prose(account: Account) -> List[str]:
    """One entry per feature the account uses, ranked and phrased naturally."""
    usage = account.get("feature_usage") or {}
    if not usage:
        return []

    ranked = sorted(usage.items(), key=lambda item: item[1], reverse=True)
    total = sum(count for _, count in ranked) or 1
    company = account["company_name"]
    ordinals = ["most-used", "second most-used", "third most-used", "fourth most-used",
                "fifth most-used", "sixth most-used"]

    entries: List[str] = []
    for index, (feature, count) in enumerate(ranked):
        share = count / total * 100
        if share >= 30:
            intensity = "heavily"
        elif share >= 15:
            intensity = "regularly"
        else:
            intensity = "only occasionally"

        if index < len(ordinals):
            rank_clause = "It is their %s feature" % ordinals[index]
        else:
            rank_clause = "It is one of their least-used features"

        entries.append(
            "{company} uses {feature} {intensity} — {count:,} recorded uses to date, "
            "about {share:.0f}% of all their tracked feature activity. {rank}. "
            "They currently use {n} features in total.".format(
                company=company,
                feature=feature,
                intensity=intensity,
                count=count,
                share=share,
                rank=rank_clause,
                n=len(ranked),
            )
        )
    return entries


def _trend_prose(account: Account) -> str:
    """One sentence about direction, plus the actual six months."""
    trend = list(account.get("usage_trend") or [])
    company = account["company_name"]
    decline_pct, peak_index = usage_decline(trend)
    streak = rising_streak(trend)

    if decline_pct > 40:
        direction = "falling sharply"
    elif decline_pct > 10:
        direction = "drifting downward"
    elif streak >= 3:
        direction = "growing steadily"
    else:
        direction = "holding roughly steady"

    months = config.TREND_MONTHS
    if len(trend) == len(months):
        walk = ", ".join("%s %s" % (months[i], "{:,}".format(trend[i])) for i in range(len(trend)))
        peak_label = months[peak_index]
    else:
        walk = ", ".join("{:,}".format(value) for value in trend)
        peak_label = "their peak month"

    if decline_pct >= 1:
        movement = "That is down %.0f%% from their %s peak." % (decline_pct, peak_label)
    elif streak >= 1:
        movement = "They have risen %d month%s in a row." % (streak, "" if streak == 1 else "s")
    else:
        movement = "Month to month they have barely moved."

    return (
        "{company} product usage is {direction}. Their monthly activity over the last "
        "six months ran: {walk}. {movement} In the last 30 days they took {recent:,} "
        "actions in total, and their last login was {days} days ago."
    ).format(
        company=_possessive(company),
        direction=direction,
        walk=walk,
        movement=movement,
        recent=account.get("usage_count_30d", 0),
        days=account.get("last_active_days", 0),
    )


def _ticket_prose(account: Account) -> List[Tuple[str, Optional[int]]]:
    """One entry per support ticket, verbatim summary, with its own timestamp."""
    company = account["company_name"]
    entries: List[Tuple[str, Optional[int]]] = []
    for ticket in account.get("support_tickets") or []:
        date = ticket.get("date", "an unknown date")
        sentiment = ticket.get("sentiment", "neutral")
        entries.append(
            (
                'On {date}, {company} raised a support ticket with our team. In their '
                'words: "{summary}". The sentiment of that conversation was {sentiment}.'.format(
                    date=date,
                    company=company,
                    summary=ticket.get("summary", ""),
                    sentiment=sentiment,
                ),
                _date_to_ms(str(date)),
            )
        )
    return entries


def memories_for_account(account: Account) -> List[Tuple[str, str, Optional[int]]]:
    """Every memory to be written for one account, as (content, event_type, ts).

    Pure — no network. Exposed so the prose can be inspected (and tested)
    without an API key.
    """
    entries: List[Tuple[str, str, Optional[int]]] = [
        (_profile_prose(account), "profile", None),
    ]
    for text in _feature_prose(account):
        entries.append((text, "usage", None))
    entries.append((_trend_prose(account), "usage", None))
    for text, timestamp_ms in _ticket_prose(account):
        entries.append((text, "ticket", timestamp_ms))
    return entries


def seed_from_accounts(accounts: Iterable[Account]) -> int:
    """Write profile, usage and ticket memories for every account.

    Returns the total number of memories written. Raises EverOSError on the
    first failure that outlives its retries — a half-seeded memory store is
    worse than an obviously broken one, because you find out on stage.
    """
    accounts = list(accounts)
    if not config.EVEROS_ENABLED:
        raise EverOSError(
            "Refusing to seed: EVERMIND_API_KEY is blank so EVEROS_ENABLED is "
            "False. Nothing was written. Put the key in .env next to api.py and "
            "restart the server, then POST /api/seed-memory again."
        )

    planned = sum(len(memories_for_account(account)) for account in accounts)
    _log(
        "seeding %d memories across %d accounts (sync writes, %.1fs politeness "
        "sleep between each) — this takes a while, watch the per-account lines"
        % (planned, len(accounts), WRITE_SLEEP_SECONDS)
    )

    written = 0
    for account in accounts:
        account_id = account["account_id"]
        for content, event_type, timestamp_ms in memories_for_account(account):
            store_account_event(account_id, content, event_type, timestamp_ms=timestamp_ms)
            written += 1
            time.sleep(WRITE_SLEEP_SECONDS)
        _log("%s (%s) done — %d/%d memories written" % (account_id, account["company_name"], written, planned))

    _log("seeding complete: %d memories written" % written)
    return written


def list_account_memories(account_id: str, memory_type: str = "episodic_memory", page_size: int = 20) -> List[Dict[str, Any]]:
    """Raw listing for one account. POST /api/v1/memories/get.

    Not used by the dashboard — this is here for eyeballing what actually
    landed in EverOS after a seed, which is the fastest way to debug recall.
    """
    if not config.EVEROS_ENABLED:
        return []
    body, _ = _post(
        GET_PATH,
        {
            "memory_type": memory_type,
            "filters": {"user_id": account_id},
            "page": 1,
            "page_size": max(1, min(int(page_size), 100)),
            "rank_by": "timestamp",
            "rank_order": "desc",
        },
        label="list %s" % account_id,
        timeout=READ_TIMEOUT_SECONDS,
    )
    data = body.get("data") or {}
    for key in ("episodes", "profiles", "memories", "items"):
        if isinstance(data.get(key), list):
            return data[key]
    return []
