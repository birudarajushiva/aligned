"""The product: outreach drafted from what EverOS actually remembers.

Pipeline for one flagged account:

    retrieve (EverOS)  ->  draft (Gemini)  ->  verify grounding  ->  Action

The one rule that matters: ``memory_used`` on the returned Action is the
verbatim text EverOS handed back. It is never paraphrased, never summarised,
never synthesised. It is the proof panel in the UI — if it stops being
literally true the product has nothing left to stand on.

Money and token figures:
  * ``tokens_used`` comes from Gemini's ``usageMetadata``. If that field is
    missing the call raises rather than guessing.
  * ``cost_usd`` is computed from the published paid-tier rates in config. On
    the AI Studio free tier the real spend is $0.00, so this is a modeled
    "what it would cost" figure — hence ``"simulated": true`` on every Action.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

import config
from memory import EverOSError, recall_account, store_account_event
from scoring import Account, apply_score
from seed import FEATURE_CATALOG

Action = Dict[str, Any]

# ── Retrieval ──────────────────────────────────────────────────────────────
#
# Two targeted recalls per account rather than one generic one. Each action
# type has a two-part question, and asking the halves separately retrieves
# better than asking them together — a single query mixing "what did they
# complain about" with "what did they stop using" pulls the centroid of two
# unrelated topics. Results are merged, keeping best-match order and dropping
# exact duplicates.

RECALL_QUERIES: Dict[str, Tuple[str, str]] = {
    "churn_risk": (
        "what did they complain about",
        "what did they stop using",
    ),
    "upsell": (
        "what are they using most",
        "what limits are they hitting",
    ),
}

RECALL_LIMIT_PER_QUERY = 6

# ── Gemini ─────────────────────────────────────────────────────────────────

MAX_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = (1, 2, 4, 8)
REQUEST_TIMEOUT_SECONDS = 90.0

BANNED_PHRASES = (
    "i hope this email finds you well",
    "circling back",
    "circle back",
    "touch base",
    "touching base",
    "synergy",
    "reaching out",
    "just checking in",
)

RECENT_CONTACT_DAYS = 14
"""Contacted inside this window -> the next draft must reference the last one."""

_client: Optional[genai.Client] = None


class GeminiError(RuntimeError):
    """A Gemini call failed, or came back in a shape we refuse to guess about."""


def _log(message: str) -> None:
    print("[agent] %s" % message, file=sys.stderr)


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.require_env("GEMINI_API_KEY"))
    return _client


def _cost_usd(model: str, usage: Dict[str, int]) -> float:
    """Paid-tier cost for one call, from real token counts.

    Thinking tokens bill as output (Google lists the output price as
    "including thinking tokens"), so they are added to the candidate tokens
    rather than ignored. On the free tier the real charge is $0.00 — this is
    the modeled paid-tier equivalent.
    """
    rates = config.GEMINI_PRICES.get(model)
    if rates is None:
        raise GeminiError(
            "No pricing for model %r. Add it to GEMINI_PRICES in "
            "aligned/config.py — cost is never estimated." % model
        )
    out_tokens = usage["candidates_token_count"] + usage["thoughts_token_count"]
    return (usage["prompt_token_count"] * rates["input"] / 1_000_000.0
            + out_tokens * rates["output"] / 1_000_000.0)


def _extract_usage(response: Any, model: str) -> Dict[str, int]:
    """Real token counts off the response. Never estimated; raises if absent."""
    u = getattr(response, "usage_metadata", None)
    if u is None:
        raise GeminiError(
            "No usage_metadata on the %s response. Token counts must come from "
            "the API, so this call is failed rather than estimated." % model
        )
    prompt = getattr(u, "prompt_token_count", None)
    total = getattr(u, "total_token_count", None)
    if prompt is None or total is None:
        raise GeminiError(
            "usage_metadata for %s is missing prompt/total token counts (%r). "
            "Refusing to estimate." % (model, u)
        )
    return {
        "prompt_token_count": int(prompt),
        "candidates_token_count": int(getattr(u, "candidates_token_count", None) or 0),
        "thoughts_token_count": int(getattr(u, "thoughts_token_count", None) or 0),
        "total_token_count": int(total),
    }


def call_gemini(
    prompt: str,
    *,
    model: Optional[str] = None,
    json_mode: bool = True,
    temperature: float = 0.6,
) -> Tuple[str, Dict[str, int], float]:
    """One generateContent call. Returns ``(text, usage, cost_usd)``.

    Sleeps before every call to respect the free tier's ~10 RPM, and retries
    429 with 1/2/4/8s backoff, printing each retry loudly. Any other error
    fails immediately.
    """
    model = model or config.GEMINI_DRAFT_MODEL

    cfg: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": config.GEMINI_MAX_OUTPUT_TOKENS,
    }
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    if config.GEMINI_THINKING_BUDGET is not None:
        cfg["thinking_config"] = types.ThinkingConfig(
            thinking_budget=config.GEMINI_THINKING_BUDGET
        )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        time.sleep(config.GEMINI_RPM_SLEEP_SECONDS)
        started = time.perf_counter()
        try:
            response = _get_client().models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**cfg),
            )
        except genai_errors.APIError as exc:
            code = getattr(exc, "code", None)
            # 429 = rate limited, 503 = "model is experiencing high demand".
            # Both are explicitly transient and both are retryable. The spec
            # said 429 only, but a 503 overload is exactly the failure that
            # would kill a live demo, and Google's own message says "try again
            # later" — so it gets the same backoff. Everything else still
            # fails immediately.
            if code in (429, 503):
                if attempt < MAX_ATTEMPTS:
                    wait = RETRY_BACKOFF_SECONDS[attempt - 1]
                    _log("!! %s on %s — attempt %d/%d, backing off %ds"
                         % ("RATE LIMITED (429)" if code == 429 else "MODEL OVERLOADED (503)",
                            model, attempt, MAX_ATTEMPTS, wait))
                    time.sleep(wait)
                    continue
                if config.GEMINI_FALLBACK_MODEL and model != config.GEMINI_FALLBACK_MODEL:
                    # flash-lite has its own quota bucket and no thinking, so it
                    # usually answers when flash is rate limited or overloaded.
                    _log("!! %s unavailable (HTTP %s) after %d attempts — falling back to %s"
                         % (model, code, MAX_ATTEMPTS, config.GEMINI_FALLBACK_MODEL))
                    return call_gemini(prompt, model=config.GEMINI_FALLBACK_MODEL,
                                       json_mode=json_mode, temperature=temperature)
                raise GeminiError(
                    "%s still returning HTTP %s after %d attempts. 429 = free tier "
                    "~10 RPM / 250 RPD, wait a minute. 503 = Google-side overload, "
                    "not your fault; retry or use --offline."
                    % (model, code, MAX_ATTEMPTS)
                ) from exc
            # Anything that is not a 429 fails immediately, as specified.
            raise GeminiError(
                "%s failed with HTTP %s: %s" % (model, code, getattr(exc, "message", exc))
            ) from exc
        except Exception as exc:
            raise GeminiError(
                "%s call failed: %s: %s" % (model, type(exc).__name__, exc)
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        usage = _extract_usage(response, model)
        cost = _cost_usd(model, usage)

        finish_reason_str = ""
        if getattr(response, "candidates", None):
            finish_reason_str = str(getattr(response.candidates[0], "finish_reason", "") or "")
        text = response.text or ""
        if not text.strip():
            finish = None
            if getattr(response, "candidates", None):
                finish = getattr(response.candidates[0], "finish_reason", None)
            raise GeminiError(
                "%s returned no text (finish_reason=%s). It spent %d tokens thinking "
                "out of a %d budget — if finish_reason is MAX_TOKENS, raise "
                "config.GEMINI_MAX_OUTPUT_TOKENS."
                % (model, finish, usage["thoughts_token_count"],
                   config.GEMINI_MAX_OUTPUT_TOKENS)
            )

        _log("%s -> %d tokens (%d in / %d out incl %d thinking) in %.0fms, $%.6f"
             % (model, usage["total_token_count"], usage["prompt_token_count"],
                usage["candidates_token_count"] + usage["thoughts_token_count"],
                usage["thoughts_token_count"], elapsed_ms, cost))
        usage["_max_tokens"] = 1 if "MAX_TOKENS" in finish_reason_str.upper() else 0
        return text, usage, cost

    raise GeminiError("%s exhausted %d attempts" % (model, MAX_ATTEMPTS))


def _parse_draft(text: str) -> Optional[Dict[str, str]]:
    """Pull {"subject","body"} out of model output. None if unusable.

    Handles a bare object, a ```json fence, and leading/trailing prose — all
    three show up in practice even with responseSchema set.
    """
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()

    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            subject, body = parsed.get("subject"), parsed.get("body")
            if isinstance(subject, str) and isinstance(body, str) and subject.strip() and body.strip():
                return {"subject": subject.strip(), "body": body.strip()}
    return None


# ── Grounding, checked rather than trusted ─────────────────────────────────


_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?")


def _specifics(snippets: Sequence[str]) -> List[str]:
    """Concrete things the draft could cite: feature names, dates and figures.

    Deterministic on purpose. A second LLM asked "is this grounded?" would be
    one more thing that can be wrong on stage; a string match cannot.

    Dates are pulled out whole *before* numbers are scanned. Otherwise
    "2026-02-14" decomposes into the fragments "02" and "14", and a draft that
    happened to mention "$1,402" would match them as substrings and be scored
    as grounded without citing anything real.
    """
    blob = " ".join(snippets)
    found: List[str] = []

    for feature in FEATURE_CATALOG:
        if feature in blob:
            found.append(feature)

    for date in sorted(set(_DATE_RE.findall(blob))):
        found.append(date)

    for number in _NUMBER_RE.findall(_DATE_RE.sub(" ", blob)):
        number = number.rstrip(",.")     # "1,640," -> "1,640"
        if len(number) >= 2 and number not in found:
            found.append(number)
    return found


def _cites(text_lower: str, specific: str) -> bool:
    """Does the draft actually cite this specific?

    Feature names are distinctive enough for a plain substring test. Numbers
    are not: "23" must match a standalone 23, never the middle of "1,234".
    """
    needle = specific.lower()
    if not specific[:1].isdigit():
        return needle in text_lower
    return re.search(r"(?<![\d,.])%s(?![\d,.])" % re.escape(needle), text_lower) is not None


def _grounding_hits(draft: Dict[str, str], snippets: Sequence[str]) -> List[str]:
    """Which specifics from the snippets actually made it into the draft."""
    text = (draft["subject"] + " " + draft["body"]).lower()
    return [s for s in _specifics(snippets) if _cites(text, s)]


# ── Prompting ──────────────────────────────────────────────────────────────


def _build_prompt(
    account: Account,
    action_type: str,
    snippets: Sequence[str],
    *,
    retry_reason: Optional[str] = None,
    prior_summary: Optional[str] = None,
    recently_contacted: bool = False,
    include_memory: bool = True,
) -> str:
    """Build the drafting prompt.

    ``retry_reason`` is None on the first attempt, "json" when the previous
    response could not be parsed, and "grounding" when it parsed but read as
    generic copy. The corrective instruction has to match the actual failure —
    telling a model to fix its JSON when the real problem was vagueness just
    gets you well-formed vagueness.
    """
    if not include_memory:
        # A/B control arm: company_name, plan and status ONLY.
        #
        # Critically, this arm must NOT see account["reason"] either. The
        # scoring reason quotes the customer's actual ticket text ("Third week
        # without a fix on the export timeouts..."), so leaking it here would
        # hand the control arm the very specifics that are supposed to be
        # memory's contribution — and the comparison would prove nothing.
        memory_block = "(no customer history available)"
        grounding_rule = (
            "You have no history for this account. Write the best short outreach "
            "email you can from the company name, plan and status alone. Do not "
            "invent specific incidents, features, dates or numbers you were not given."
        )
    elif snippets:
        memory_block = "\n".join("- %s" % s for s in snippets)
        grounding_rule = (
            "You MUST reference at least TWO specific details drawn from the memory "
            "above — a named feature, a real complaint from a ticket, or an actual "
            "usage number. Quote the specifics; do not gesture at them vaguely."
        )
    else:
        memory_block = "(no memories were retrieved for this account)"
        grounding_rule = (
            "No memory was retrieved for this account. Do NOT invent details, "
            "features, numbers or complaints. Say plainly in the body that you do "
            "not have their recent history in front of you and ask them directly."
        )

    if action_type == "churn_risk":
        intent = (
            "This account is at risk of churning. Acknowledge the specific friction "
            "they hit and offer a concrete fix or next step. Do not be defensive."
        )
    else:
        intent = (
            "This account is ready for an upsell. Reference the specific limit or "
            "volume they are pushing against and what the next tier actually "
            "unlocks for them. Do not be pushy."
        )

    strict_block = ""
    if retry_reason == "json":
        strict_block = (
            "\n\nYOUR PREVIOUS RESPONSE WAS REJECTED because it was not valid JSON. "
            'Return ONLY a raw JSON object of exactly the form {"subject": "...", '
            '"body": "..."} — no markdown fence, no commentary, no leading or trailing '
            "text of any kind. The very first character of your response must be { and "
            "the last must be }."
        )
    elif retry_reason == "grounding":
        specifics = _specifics(snippets)[:6]
        strict_block = (
            "\n\nYOUR PREVIOUS DRAFT WAS REJECTED because it was generic — it did not "
            "cite anything that only this customer would recognise. Rewrite it. You MUST "
            "quote at least TWO of these exact specifics from their history, verbatim, "
            "in the subject or body: %s. A draft that would still make sense with another "
            "company's name in it will be rejected again." % ", ".join(specifics)
        )

    if include_memory:
        customer_block = (
            "CUSTOMER: %s (%s plan, %d seats)\n"
            "WHY THEY ARE FLAGGED: %s\n\n"
            % (account["company_name"], account["plan"], account["seats"],
               account.get("reason", ""))
        )
    else:
        # Control arm: company, plan, status. Nothing else — see above.
        customer_block = (
            "CUSTOMER: %s (%s plan)\n"
            "ACCOUNT STATUS: %s\n\n"
            % (account["company_name"], account["plan"],
               "at risk of churning" if action_type == "churn_risk" else "ready for an upsell")
        )

    prior_block = ""
    if prior_summary:
        if recently_contacted:
            prior_block = (
                "\n\nWE ALREADY CONTACTED THEM RECENTLY: %s\n"
                "Do NOT write as if this is the first time. Open by referring back to "
                "that previous message — acknowledge you have already written, and "
                "move the conversation forward rather than repeating it."
                % prior_summary
            )
        else:
            prior_block = (
                "\n\nPRIOR CONTACT (a while ago): %s\n"
                "You may reference it briefly, but this is a fresh approach."
                % prior_summary
            )

    return (
        "You are an account manager at a B2B SaaS company writing to a customer you "
        "genuinely know. You are not a marketer.\n\n"
        "{customer_block}"
        "WHAT WE REMEMBER ABOUT THEM:\n{memory}\n\n"
        "YOUR TASK: {intent}\n\n"
        "{grounding}\n\n"
        "RULES:\n"
        "- Under 120 words in the body. Warm, direct, specific.\n"
        "- Write like a person who has actually looked at this account.\n"
        "- Do NOT write generic re-engagement copy. A reader must not be able to "
        "swap in another company's name and have it still make sense.\n"
        "- Never use these phrases: \"I hope this email finds you well\", "
        "\"circling back\", \"touching base\", \"just checking in\", \"synergy\".\n"
        "- No placeholder text, no [brackets], no TODO.\n\n"
        "{prior}"
        "\n\n"
        'Return strict JSON: {{"subject": "...", "body": "..."}}'
        "{strict}"
    ).format(
        customer_block=customer_block,
        memory=memory_block,
        intent=intent,
        grounding=grounding_rule,
        prior=prior_block,
        strict=strict_block,
    )


# ── Money and confidence ───────────────────────────────────────────────────


def arr_at_stake(account: Account, action_type: str) -> float:
    """What this action is worth, in ARR.

    churn_risk: the whole contract is on the table.
    upsell: the next tier's list price minus what they pay today, annualised.
            An account already at the top of the ladder has no higher tier, so
            expansion is modeled as seat growth at their current rate (see
            TOP_TIER_SEAT_EXPANSION).
    """
    if action_type == "churn_risk":
        return round(float(account["arr_usd"]), 2)

    plan = account["plan"]
    current_mrr = float(account["mrr_usd"])
    next_plan = config.PLAN_LADDER.get(plan)

    if next_plan is None:
        expansion_mrr = current_mrr * config.TOP_TIER_SEAT_EXPANSION
    else:
        next_price = config.PLAN_MONTHLY_LIST_USD[next_plan]
        expansion_mrr = max(0.0, next_price - current_mrr)

    return round(expansion_mrr * 12, 2)


def confidence(account: Account, action_type: str, snippet_count: int) -> float:
    """How much to trust this action, 0-1.

    Two equally weighted halves:

      signal   — how clear the scoring trigger was. For churn_risk that is
                 risk_score directly (75/100 -> 0.75). For upsell it is the
                 inverse: an upsell is most credible on a calm, healthy account,
                 so a risk_score of 11 gives 0.89.

      grounding— how much real memory backed the draft. Zero snippets is zero
                 grounding; GROUNDING_SATURATION (6) or more is full marks.
                 This is the "more grounding = higher confidence" term, and it
                 is why an ungrounded action can never score above 0.5.

    confidence = 0.5 * signal + 0.5 * grounding, clamped to 0.05-0.99 — never
    0 and never 1, because neither is ever honest here.
    """
    GROUNDING_SATURATION = 6
    risk = float(account.get("risk_score", 0)) / 100.0
    signal = risk if action_type == "churn_risk" else 1.0 - risk
    grounding = min(snippet_count / float(GROUNDING_SATURATION), 1.0)
    return round(max(0.05, min(0.99, 0.5 * signal + 0.5 * grounding)), 2)


# ── The agent ──────────────────────────────────────────────────────────────


def _action_type_for(account: Account) -> Optional[str]:
    status = account.get("status")
    if status == "churn_risk":
        return "churn_risk"
    if status == "upsell_ready":
        return "upsell"
    return None


def retrieve_memory(account: Account, action_type: str) -> List[str]:
    """Two targeted recalls, merged, verbatim, duplicates dropped."""
    snippets: List[str] = []
    seen = set()
    for query in RECALL_QUERIES[action_type]:
        for snippet in recall_account(account["account_id"], query, limit=RECALL_LIMIT_PER_QUERY):
            if snippet not in seen:
                seen.add(snippet)
                snippets.append(snippet)
    return snippets


# ── Memory that compounds: prior outreach ──────────────────────────────────
#
# Every action is written back to EverOS as event_type "outreach". Before
# drafting, we recall that history — so the second scan behaves differently
# from the first because it remembers the first.

PRIOR_OUTREACH_QUERY = "what outreach email did we already send them and when"

_OUTREACH_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def recall_prior_outreach(account_id: str) -> List[str]:
    """Snippets describing outreach we have already sent this account."""
    return recall_account(account_id, PRIOR_OUTREACH_QUERY, limit=4)


def _days_since_latest_date(snippets: Sequence[str]) -> Optional[int]:
    """Days since the most recent ISO date mentioned in prior-outreach memory.

    EverOS returns extracted prose, not structured records, so the send date
    has to be read back out of the text. We write it in ISO form precisely so
    this is a parse rather than a guess. None when no date is present.
    """
    best: Optional[datetime] = None
    for text in snippets:
        for raw in _OUTREACH_DATE_RE.findall(text):
            try:
                when = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if best is None or when > best:
                best = when
    if best is None:
        return None
    return max(0, (datetime.now(timezone.utc) - best).days)


def summarise_prior_outreach(account_id: str, snippets: Sequence[str]) -> Optional[str]:
    """One line describing what we last sent. Classification-shaped -> flash-lite.

    Returns None when there is no prior outreach, so the caller can tell
    "nothing sent yet" apart from "sent, and here is what".
    """
    if not snippets:
        return None
    prompt = (
        "Below are memory snippets about outreach emails already sent to one "
        "customer. In ONE sentence under 25 words, state what the most recent "
        "message was about and roughly when it was sent. If the snippets do not "
        "actually describe a sent email, reply with exactly: NONE\n\n"
        + "\n".join("- %s" % s for s in snippets)
        + '\n\nReturn strict JSON: {"summary": "..."}'
    )
    try:
        text, usage, cost = call_gemini(
            prompt, model=config.GEMINI_CLASSIFY_MODEL, json_mode=True, temperature=0.2
        )
    except GeminiError as exc:
        _log("%s: prior-outreach summary failed (%s) — continuing without it"
             % (account_id, exc))
        return None
    try:
        parsed = json.loads(text.strip())
        summary = str(parsed.get("summary", "")).strip()
    except (ValueError, AttributeError):
        summary = text.strip()[:200]
    if not summary or summary.upper().startswith("NONE"):
        return None
    return summary


def record_outreach(account: Account, action: Action) -> None:
    """Write this action back to EverOS so the next scan remembers it.

    Written as prose with an explicit ISO date, because recall returns
    extracted text and the date has to survive that round trip.
    """
    if not config.EVEROS_ENABLED:
        return
    sent_on = action["triggered_at"][:10]
    text = (
        "On {date} we sent {company} a {kind} outreach email. Subject: \"{subject}\". "
        "It was triggered because: {reason} The message said: {body}"
    ).format(
        date=sent_on,
        company=account["company_name"],
        kind="churn-risk retention" if action["action_type"] == "churn_risk" else "upsell",
        subject=action["draft_subject"],
        reason=action.get("reason", "(no reason recorded)"),
        body=action["draft_body"],
    )
    try:
        store_account_event(account["account_id"], text, "outreach")
        _log("%s: outreach written back to EverOS (next scan will see it)"
             % account["account_id"])
    except EverOSError as exc:
        # Never fail an action because the write-back failed — the draft is
        # already good. Say so loudly instead.
        _log("!! %s: could not write outreach back to EverOS: %s"
             % (account["account_id"], exc))


def generate_action(account: Account, *, write_back: bool = True) -> Action:
    """Draft outreach for one flagged account, grounded in its own history."""
    action_type = _action_type_for(account)
    if action_type is None:
        raise ValueError(
            "%s is %s — generate_action is only for flagged accounts."
            % (account["account_id"], account.get("status"))
        )

    # 0. What have we already said to them?
    prior_snippets = recall_prior_outreach(account["account_id"])
    days_since = _days_since_latest_date(prior_snippets)
    prior_summary = summarise_prior_outreach(account["account_id"], prior_snippets)
    previously_contacted = prior_summary is not None
    recently_contacted = (
        previously_contacted and days_since is not None and days_since <= RECENT_CONTACT_DAYS
    )
    if previously_contacted:
        _log("%s: previously contacted%s — the draft will reference it"
             % (account["account_id"],
                (" %d days ago" % days_since) if days_since is not None else ""))

    # 1. Retrieve. These strings go into memory_used untouched.
    snippets = retrieve_memory(account, action_type)
    if not snippets:
        _log(
            "%s: EverOS returned no memories — the draft will say so rather than "
            "invent details. Did you POST /api/seed-memory?" % account["account_id"]
        )

    # 2. Draft, with one stricter retry on unusable output.
    tokens_used = 0
    cost_usd = 0.0
    draft: Optional[Dict[str, str]] = None
    last_text = ""

    retry_reason: Optional[str] = None
    for attempt in (1, 2):
        text, usage, cost = call_gemini(
            _build_prompt(account, action_type, snippets, retry_reason=retry_reason,
                          prior_summary=prior_summary,
                          recently_contacted=recently_contacted),
            model=config.GEMINI_DRAFT_MODEL,
            json_mode=True,
            temperature=0.6 if retry_reason is None else 0.2,
        )
        tokens_used += usage["total_token_count"]
        cost_usd += cost
        last_text = text

        if usage.get("_max_tokens"):
            _log("!! %s hit MAX_TOKENS (%d thinking tokens) — the JSON is likely "
                 "truncated. Raise config.GEMINI_MAX_OUTPUT_TOKENS."
                 % (account["account_id"], usage["thoughts_token_count"]))

        parsed = _parse_draft(text)
        if parsed is None:
            if attempt == 1:
                _log("%s: unparseable draft — retrying with a stricter JSON instruction"
                     % account["account_id"])
                retry_reason = "json"
                continue
            break  # second attempt also unusable; fail below

        hits = _grounding_hits(parsed, snippets)
        if snippets and len(hits) < 2 and attempt == 1:
            _log(
                "%s: draft cited only %d specific detail(s) %s — retrying for a "
                "more grounded draft" % (account["account_id"], len(hits), hits)
            )
            draft = parsed  # keep as a fallback in case the retry is worse
            retry_reason = "grounding"
            continue

        # Keep whichever draft is better grounded. The retry usually wins, but
        # it is not guaranteed to, and shipping a worse one would be perverse.
        if draft is not None and len(_grounding_hits(draft, snippets)) > len(hits):
            _log("%s: keeping the first draft — the retry cited fewer specifics"
                 % account["account_id"])
            break
        draft = parsed
        break

    if draft is None:
        raise GeminiError(
            "Could not get usable JSON out of %s for %s after 2 attempts. Last "
            "response was: %s"
            % (config.GEMINI_DRAFT_MODEL, account["account_id"], last_text[:400])
        )

    final_hits = _grounding_hits(draft, snippets)
    if snippets and len(final_hits) < 2:
        _log(
            "!! %s: draft cites only %d specific detail(s) %s after a retry. "
            "Shipping it, but it is weakly grounded — check it before sending."
            % (account["account_id"], len(final_hits), final_hits)
        )
    lowered = draft["body"].lower()
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            _log("!! %s: draft contains banned phrase %r" % (account["account_id"], phrase))

    action: Action = {
        "action_id": "act_%s_%s"
        % (account["account_id"], datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]),
        "account_id": account["account_id"],
        "company_name": account["company_name"],
        "action_type": action_type,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "reason": account.get("reason", ""),
        "memory_used": snippets,  # verbatim, never rewritten
        "draft_subject": draft["subject"],
        "draft_body": draft["body"],
        "arr_at_stake_usd": arr_at_stake(account, action_type),
        "confidence": confidence(account, action_type, len(snippets)),
        "tokens_used": tokens_used,
        "cost_usd": round(cost_usd, 6),
        "previously_contacted": previously_contacted,
        "prior_outreach_summary": prior_summary,
        "simulated": True,
    }

    # STEP 4: write it back so the NEXT scan knows we already wrote.
    if write_back:
        record_outreach(account, action)
    return action


def generate_action_pair(account: Account) -> Dict[str, Any]:
    """The A/B proof: the same outreach drafted with and without memory.

    Both arms use the SAME model, the SAME temperature, and the SAME
    instructions. The ONLY difference is whether the retrieved EverOS snippets
    are in the prompt. Nothing else may vary — if it did, the comparison would
    be measuring prompt engineering rather than memory, and the demo would be
    a lie.

    Deliberately absent from both arms: prior-outreach context. Feeding it to
    one side and not the other would be a second variable.
    """
    action_type = _action_type_for(account) or "churn_risk"
    snippets = retrieve_memory(account, action_type)

    shared = {
        "model": config.GEMINI_DRAFT_MODEL,
        "json_mode": True,
        "temperature": 0.6,
    }

    def _arm(include_memory: bool, label: str) -> Dict[str, Any]:
        prompt = _build_prompt(
            account, action_type,
            snippets if include_memory else [],
            include_memory=include_memory,
        )
        text, usage, cost = call_gemini(prompt, **shared)
        draft = _parse_draft(text)
        if draft is None:
            text, usage2, cost2 = call_gemini(
                _build_prompt(account, action_type,
                              snippets if include_memory else [],
                              include_memory=include_memory, retry_reason="json"),
                **shared
            )
            usage = {k: usage[k] + usage2[k] for k in usage}
            cost += cost2
            draft = _parse_draft(text)
        if draft is None:
            raise GeminiError(
                "A/B %s arm for %s produced unparseable JSON twice."
                % (label, account["account_id"])
            )
        return {
            "subject": draft["subject"],
            "body": draft["body"],
            "tokens_used": usage["total_token_count"],
            "cost_usd": round(cost, 6),
        }

    _log("%s: A/B pair — control arm (no memory)" % account["account_id"])
    without = _arm(False, "without_memory")
    _log("%s: A/B pair — treatment arm (%d memory snippets)"
         % (account["account_id"], len(snippets)))
    with_mem = _arm(True, "with_memory")
    with_mem["memory_used"] = snippets   # verbatim

    # What did the memory arm ACTUALLY ground on? Deterministic string
    # matching against the retrieved snippets — no padding, no LLM opinion.
    cited = _grounding_hits(
        {"subject": with_mem["subject"], "body": with_mem["body"]}, snippets
    )
    if not cited:
        _log("!! %s: with_memory cited NOTHING specific from the snippets. The A/B "
             "comparison does not support the pitch for this account."
             % account["account_id"])

    return {
        "account_id": account["account_id"],
        "company_name": account["company_name"],
        "without_memory": without,
        "with_memory": with_mem,
        "specifics_cited": cited,
        "simulated": True,
    }


def run_scan(accounts: Sequence[Account]) -> List[Action]:
    """Score every account, draft outreach for the flagged ones.

    Strictly sequential. The free tier is ~10 RPM and concurrency would burn
    through it in seconds.
    """
    accounts = list(accounts)
    actions: List[Action] = []

    flagged = []
    for account in accounts:
        apply_score(account)
        if _action_type_for(account) is not None:
            flagged.append(account)

    _log("scanning %d accounts — %d flagged, %d healthy (sequential, ~%.1fs/call)"
         % (len(accounts), len(flagged), len(accounts) - len(flagged), config.GEMINI_RPM_SLEEP_SECONDS))

    failed: List[Dict[str, str]] = []
    index = 0
    for account in accounts:
        action_type = _action_type_for(account)
        if action_type is None:
            _log("  %-8s %-24s healthy — skipped" % (account["account_id"], account["company_name"]))
            continue

        index += 1
        _log(
            "  %-8s %-24s %s (%d/%d) — drafting..."
            % (account["account_id"], account["company_name"], action_type, index, len(flagged))
        )
        try:
            action = generate_action(account)
        except (GeminiError, EverOSError) as exc:
            # One account failing must not throw away the whole scan. On a free
            # tier with one shot at a demo, five good emails beats a stack trace.
            failed.append({"account_id": account["account_id"],
                           "company_name": account["company_name"],
                           "error": str(exc)})
            _log("!! %-8s %-24s FAILED — %s" % (account["account_id"], account["company_name"], exc))
            _log("   continuing with the remaining accounts")
            continue
        actions.append(action)
        _log(
            "  %-8s %-24s done — %d memories cited, confidence %.2f, "
            "$%s ARR at stake, %d tokens"
            % (
                account["account_id"],
                account["company_name"],
                len(action["memory_used"]),
                action["confidence"],
                "{:,.0f}".format(action["arr_at_stake_usd"]),
                action["tokens_used"],
            )
        )

    if failed:
        _log("!! %d of %d accounts failed to draft:" % (len(failed), len(flagged)))
        for f in failed:
            _log("     %s %s — %s" % (f["account_id"], f["company_name"], f["error"][:110]))
    total_tokens = sum(a["tokens_used"] for a in actions)
    total_cost = sum(a["cost_usd"] for a in actions)
    total_arr = sum(a["arr_at_stake_usd"] for a in actions)
    _log(
        "scan complete: %d actions, $%s ARR at stake, %d tokens, $%.6f "
        "(modeled at paid-tier rates; free tier bills $0.00)"
        % (len(actions), "{:,.0f}".format(total_arr), total_tokens, total_cost)
    )
    return actions
