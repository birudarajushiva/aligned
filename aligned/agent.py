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

import httpx

from traction import config
from traction.memory import recall_account
from traction.scoring import Account, apply_score
from traction.seed import FEATURE_CATALOG

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

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {"subject": {"type": "string"}, "body": {"type": "string"}},
    "required": ["subject", "body"],
}

BANNED_PHRASES = (
    "i hope this email finds you well",
    "circling back",
    "circle back",
    "synergy",
    "touching base",
    "just checking in",
)

_client: Optional[httpx.Client] = None


class GeminiError(RuntimeError):
    """A Gemini call failed, or came back in a shape we refuse to guess about."""


def _log(message: str) -> None:
    print("[agent] %s" % message, file=sys.stderr)


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(base_url=config.GEMINI_BASE_URL.rstrip("/"))
    return _client


def _cost_usd(model: str, usage: Dict[str, int]) -> float:
    """Paid-tier cost for one call.

    Thinking tokens bill as output (Google lists the output price as
    "including thinking tokens"), so they are added to the candidate tokens
    rather than ignored.
    """
    rates = config.GEMINI_PRICING_USD_PER_MTOK.get(model)
    if rates is None:
        raise GeminiError(
            "No pricing for model %r. Add it to GEMINI_PRICING_USD_PER_MTOK in "
            "traction/config.py — cost is never estimated." % model
        )
    input_tokens = usage["promptTokenCount"]
    output_tokens = usage["candidatesTokenCount"] + usage.get("thoughtsTokenCount", 0)
    return (
        input_tokens * rates["input"] / 1_000_000.0
        + output_tokens * rates["output"] / 1_000_000.0
    )


def _extract_usage(body: Dict[str, Any], model: str) -> Dict[str, int]:
    """Pull usageMetadata, or raise. Never estimated, never defaulted to zero."""
    usage = body.get("usageMetadata")
    if not isinstance(usage, dict):
        raise GeminiError(
            "Gemini response for %s has no usageMetadata. Token counts must come "
            "from the API, so this call is being failed rather than estimated. "
            "Response keys: %s" % (model, sorted(body))
        )
    for field in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount"):
        if not isinstance(usage.get(field), int):
            raise GeminiError(
                "Gemini usageMetadata for %s is missing %s (got %r). Refusing to "
                "estimate token counts." % (model, field, usage.get(field))
            )
    return usage


def call_gemini(
    prompt: str,
    *,
    model: Optional[str] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0.7,
) -> Tuple[str, Dict[str, int], float]:
    """One generateContent call. Returns ``(text, usage, cost_usd)``.

    Sleeps before every call to stay under the free tier's ~10 RPM, and retries
    429 with 1/2/4/8s backoff, printing each retry.
    """
    model = model or config.GEMINI_DRAFT_MODEL
    api_key = config.require_env("GEMINI_API_KEY")

    generation_config: Dict[str, Any] = {"temperature": temperature}
    if json_schema is not None:
        # Structured output beats asking politely for JSON in the prompt. The
        # defensive parse downstream stays anyway — belt and braces.
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = json_schema

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    # Key in a header, never in the query string.
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    path = "/v1beta/models/%s:generateContent" % model

    for attempt in range(1, MAX_ATTEMPTS + 1):
        time.sleep(config.GEMINI_RPM_SLEEP_SECONDS)
        started = time.perf_counter()
        try:
            response = _get_client().post(
                path, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except httpx.RequestError as exc:
            if attempt < MAX_ATTEMPTS:
                wait = RETRY_BACKOFF_SECONDS[attempt - 1]
                _log(
                    "%s network error (%s) — attempt %d/%d, retrying in %ds"
                    % (model, type(exc).__name__, attempt, MAX_ATTEMPTS, wait)
                )
                time.sleep(wait)
                continue
            raise GeminiError(
                "%s failed after %d attempts: %s: %s"
                % (model, MAX_ATTEMPTS, type(exc).__name__, exc)
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 429:
            if attempt < MAX_ATTEMPTS:
                wait = RETRY_BACKOFF_SECONDS[attempt - 1]
                _log(
                    "!! RATE LIMITED (429) on %s after %.0fms — attempt %d/%d, "
                    "backing off %ds" % (model, elapsed_ms, attempt, MAX_ATTEMPTS, wait)
                )
                time.sleep(wait)
                continue
            raise GeminiError(
                "%s still rate limited (429) after %d attempts. The free tier is "
                "~10 RPM — wait a minute and re-run the scan." % (model, MAX_ATTEMPTS)
            )

        if response.status_code >= 400:
            raise GeminiError(
                "%s failed with HTTP %d after %.0fms: %s"
                % (model, response.status_code, elapsed_ms, response.text[:500])
            )

        body = response.json()
        usage = _extract_usage(body, model)
        cost = _cost_usd(model, usage)

        candidates = body.get("candidates") or []
        text = ""
        if candidates:
            for part in (candidates[0].get("content") or {}).get("parts") or []:
                if isinstance(part.get("text"), str):
                    text += part["text"]
        if not text.strip():
            finish = candidates[0].get("finishReason") if candidates else "no candidates"
            raise GeminiError(
                "%s returned no text (finishReason=%s). Full response: %s"
                % (model, finish, json.dumps(body)[:400])
            )

        _log(
            "%s -> %d tokens (%d in / %d out) in %.0fms, $%.6f"
            % (
                model,
                usage["totalTokenCount"],
                usage["promptTokenCount"],
                usage["candidatesTokenCount"] + usage.get("thoughtsTokenCount", 0),
                elapsed_ms,
                cost,
            )
        )
        return text, usage, cost

    raise GeminiError("%s exhausted %d attempts" % (model, MAX_ATTEMPTS))


# ── Parsing ────────────────────────────────────────────────────────────────


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
) -> str:
    """Build the drafting prompt.

    ``retry_reason`` is None on the first attempt, "json" when the previous
    response could not be parsed, and "grounding" when it parsed but read as
    generic copy. The corrective instruction has to match the actual failure —
    telling a model to fix its JSON when the real problem was vagueness just
    gets you well-formed vagueness.
    """
    if snippets:
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

    return (
        "You are an account manager at a B2B SaaS company writing to a customer you "
        "genuinely know. You are not a marketer.\n\n"
        "CUSTOMER: {company} ({plan} plan, {seats} seats)\n"
        "WHY THEY ARE FLAGGED: {reason}\n\n"
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
        'Return strict JSON: {{"subject": "...", "body": "..."}}'
        "{strict}"
    ).format(
        company=account["company_name"],
        plan=account["plan"],
        seats=account["seats"],
        reason=account.get("reason", ""),
        memory=memory_block,
        intent=intent,
        grounding=grounding_rule,
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


def generate_action(account: Account) -> Action:
    """Draft outreach for one flagged account, grounded in its own history."""
    action_type = _action_type_for(account)
    if action_type is None:
        raise ValueError(
            "%s is %s — generate_action is only for flagged accounts."
            % (account["account_id"], account.get("status"))
        )

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
            _build_prompt(account, action_type, snippets, retry_reason=retry_reason),
            model=config.GEMINI_DRAFT_MODEL,
            json_schema=DRAFT_SCHEMA,
            temperature=0.6 if retry_reason is None else 0.2,
        )
        tokens_used += usage["totalTokenCount"]
        cost_usd += cost
        last_text = text

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

    return {
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
        action = generate_action(account)
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

    total_tokens = sum(a["tokens_used"] for a in actions)
    total_cost = sum(a["cost_usd"] for a in actions)
    total_arr = sum(a["arr_at_stake_usd"] for a in actions)
    _log(
        "scan complete: %d actions, $%s ARR at stake, %d tokens, $%.6f "
        "(modeled at paid-tier rates; free tier bills $0.00)"
        % (len(actions), "{:,.0f}".format(total_arr), total_tokens, total_cost)
    )
    return actions
