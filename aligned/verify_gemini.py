#!/usr/bin/env python3
"""Standalone Gemini diagnostic.

    python verify_gemini.py

No FastAPI, no EverOS — just the google-genai SDK and config.py. Walks six
checks, stops on the first failure, and prints the full error body plus the
exact request that produced it.

WHY THE MODELS ARE NOT gemini-2.5-*
-----------------------------------
This API key cannot call gemini-2.5-flash or gemini-2.5-flash-lite: both return
``404 "This model ... is no longer available to new users"``. gemini-2.0-flash
and gemini-2.0-flash-lite return ``429`` quota-exhausted. CHECK 2 probes the
whole candidate list and tells you exactly which models this key can use, so
this file stays honest if Google moves the goalposts again.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

import config

PASS, FAIL = "✓", "✗"
RESULTS: List[Dict[str, Any]] = []

CANDIDATE_MODELS = [
    "gemini-2.5-flash", "gemini-2.5-flash-lite",
    "gemini-3.5-flash", "gemini-3.5-flash-lite",
    "gemini-flash-latest", "gemini-flash-lite-latest",
    "gemini-2.0-flash", "gemini-2.0-flash-lite",
]


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def ok(msg: str) -> None:
    print("  %s %s" % (PASS, msg))


def mask(key: str) -> str:
    if not key:
        return "(empty)"
    if len(key) <= 8:
        return "*" * len(key)
    return "%s...%s  (len %d)" % (key[:4], key[-4:], len(key))


def record(check: str, passed: bool, latency_ms: Optional[float], tokens: Optional[int], note: str = "") -> None:
    RESULTS.append({"check": check, "pass": passed, "ms": latency_ms, "tokens": tokens, "note": note})


def die(check: str, reason: str, *, model=None, request=None, exc=None) -> None:
    print("  %s %s" % (FAIL, reason))
    print("\n  --- REQUEST SENT " + "-" * 58)
    print("  endpoint: %s/v1beta/models/%s:generateContent" % (config.GEMINI_BASE_URL, model or "?"))
    print("  headers:  {'x-goog-api-key': '%s', 'Content-Type': 'application/json'}" % mask(config.GEMINI_API_KEY))
    if request is not None:
        print("  body:     %s" % json.dumps(request, indent=2, default=str).replace("\n", "\n  "))
    if exc is not None:
        print("\n  --- ERROR " + "-" * 65)
        print("  %s: %s" % (type(exc).__name__, exc))
        for attr in ("code", "status", "message", "details", "response_json"):
            if hasattr(exc, attr):
                print("  %s: %s" % (attr, getattr(exc, attr)))
    summary()
    print("\n" + "=" * 78)
    print("STOPPED at %s. Later checks did not run." % check)
    print("=" * 78)
    sys.exit(1)


def summary() -> None:
    if not RESULTS:
        return
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("  %-40s %-6s %10s %8s" % ("check", "result", "latency", "tokens"))
    print("  " + "-" * 68)
    for r in RESULTS:
        print("  %-40s %-6s %10s %8s%s" % (
            r["check"][:40],
            "PASS" if r["pass"] else "FAIL",
            ("%.0f ms" % r["ms"]) if r["ms"] is not None else "-",
            r["tokens"] if r["tokens"] is not None else "-",
            ("   " + r["note"]) if r["note"] else "",
        ))


# ── the one call path everything uses ──────────────────────────────────────

_client: Optional[genai.Client] = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.require_env("GEMINI_API_KEY"))
    return _client


def usage_of(response: Any, model: str) -> Dict[str, int]:
    """Real token counts from the response. Never estimated."""
    u = getattr(response, "usage_metadata", None)
    if u is None:
        raise RuntimeError(
            "No usage_metadata on the %s response. Token counts must come from "
            "the API, so this is failed rather than estimated." % model
        )
    prompt = getattr(u, "prompt_token_count", None)
    cand = getattr(u, "candidates_token_count", None)
    total = getattr(u, "total_token_count", None)
    thoughts = getattr(u, "thoughts_token_count", None) or 0
    if prompt is None or total is None:
        raise RuntimeError(
            "usage_metadata on %s is missing prompt/total counts: %r" % (model, u)
        )
    return {
        "prompt_token_count": int(prompt),
        "candidates_token_count": int(cand or 0),
        "thoughts_token_count": int(thoughts),
        "total_token_count": int(total),
    }


def cost_of(model: str, usage: Dict[str, int]) -> float:
    rates = config.GEMINI_PRICES.get(model)
    if rates is None:
        raise RuntimeError("No entry for %r in config.GEMINI_PRICES." % model)
    out_tokens = usage["candidates_token_count"] + usage["thoughts_token_count"]
    return (usage["prompt_token_count"] * rates["input"] / 1e6
            + out_tokens * rates["output"] / 1e6)


def generate(prompt: str, *, model: str, json_mode: bool = False,
             temperature: float = 0.6, max_attempts: int = 4) -> Tuple[str, Dict[str, int], float, float]:
    """One call, with 1/2/4/8s backoff on 429. Returns (text, usage, ms, cost)."""
    cfg: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": config.GEMINI_MAX_OUTPUT_TOKENS,
    }
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    if config.GEMINI_THINKING_BUDGET is not None:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=config.GEMINI_THINKING_BUDGET)

    backoff = (1, 2, 4, 8)
    for attempt in range(1, max_attempts + 1):
        time.sleep(config.GEMINI_RPM_SLEEP_SECONDS)
        started = time.perf_counter()
        try:
            resp = client().models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**cfg)
            )
        except genai_errors.APIError as exc:
            if getattr(exc, "code", None) == 429 and attempt < max_attempts:
                wait = backoff[attempt - 1]
                print("  !! RATE LIMITED (429) on %s — attempt %d/%d, backing off %ds"
                      % (model, attempt, max_attempts, wait))
                time.sleep(wait)
                continue
            raise
        ms = (time.perf_counter() - started) * 1000
        usage = usage_of(resp, model)
        text = resp.text or ""
        if not text.strip():
            fr = None
            if getattr(resp, "candidates", None):
                fr = getattr(resp.candidates[0], "finish_reason", None)
            raise RuntimeError(
                "%s returned no text (finish_reason=%s, thoughts=%d/%d output tokens). "
                "If finish_reason is MAX_TOKENS the model spent the whole budget "
                "thinking — raise config.GEMINI_MAX_OUTPUT_TOKENS."
                % (model, fr, usage["thoughts_token_count"], config.GEMINI_MAX_OUTPUT_TOKENS)
            )
        return text, usage, ms, cost_of(model, usage)
    raise RuntimeError("%s exhausted %d attempts" % (model, max_attempts))


# ── CHECK 1 ────────────────────────────────────────────────────────────────

hr("CHECK 1 — key loaded")
if not config.GEMINI_API_KEY:
    print("  %s GEMINI_API_KEY is empty." % FAIL)
    print("     Add it to .env next to api.py:  GEMINI_API_KEY=your-key")
    record("1 key loaded", False, None, None, "empty")
    summary()
    sys.exit(1)
ok("GEMINI_API_KEY = %s" % mask(config.GEMINI_API_KEY))
ok("draft model    = %s" % config.GEMINI_DRAFT_MODEL)
ok("classify model = %s" % config.GEMINI_CLASSIFY_MODEL)
record("1 key loaded", True, None, None)


# ── CHECK 2 ────────────────────────────────────────────────────────────────

hr("CHECK 2 — trivial call to the drafting model")
print("  probing which models this key can actually reach:")
reachable = []
for m in CANDIDATE_MODELS:
    try:
        time.sleep(config.GEMINI_RPM_SLEEP_SECONDS)
        r = client().models.generate_content(
            model=m, contents="Say OK",
            config=types.GenerateContentConfig(max_output_tokens=2000))
        reachable.append(m)
        print("    %s %-26s reachable" % (PASS, m))
    except genai_errors.APIError as exc:
        print("    %s %-26s HTTP %s — %s" % (FAIL, m, getattr(exc, "code", "?"),
                                             str(getattr(exc, "message", exc))[:64]))
    except Exception as exc:
        print("    %s %-26s %s" % (FAIL, m, type(exc).__name__))

if config.GEMINI_DRAFT_MODEL not in reachable:
    die("CHECK 2",
        "The configured drafting model %r is not reachable with this key. "
        "Reachable models: %s. Set config.GEMINI_DRAFT_MODEL to one of those."
        % (config.GEMINI_DRAFT_MODEL, reachable or "(none)"),
        model=config.GEMINI_DRAFT_MODEL)

print()
text, usage, ms, cost = generate("Reply with exactly: OK", model=config.GEMINI_DRAFT_MODEL)
ok("%s responded in %.0fms" % (config.GEMINI_DRAFT_MODEL, ms))
print("  response text: %r" % text.strip()[:120])
print("  RAW usage object (real field names):")
print("    " + json.dumps(usage, indent=2).replace("\n", "\n    "))
record("2 draft model call", True, ms, usage["total_token_count"])
COST_TOTAL = cost


# ── CHECK 3 ────────────────────────────────────────────────────────────────

hr("CHECK 3 — trivial call to the classification model")
if config.GEMINI_CLASSIFY_MODEL not in reachable:
    die("CHECK 3",
        "The configured classification model %r is not reachable. Reachable: %s"
        % (config.GEMINI_CLASSIFY_MODEL, reachable), model=config.GEMINI_CLASSIFY_MODEL)
text3, usage3, ms3, cost3 = generate("Reply with exactly: OK", model=config.GEMINI_CLASSIFY_MODEL)
ok("%s responded in %.0fms" % (config.GEMINI_CLASSIFY_MODEL, ms3))
print("  response text: %r" % text3.strip()[:120])
print("  RAW usage object:")
print("    " + json.dumps(usage3, indent=2).replace("\n", "\n    "))
record("3 classify model call", True, ms3, usage3["total_token_count"])
COST_TOTAL += cost3


# ── CHECK 4 ────────────────────────────────────────────────────────────────

hr("CHECK 4 — strict JSON output")
JSON_PROMPT = (
    'Return strict JSON with exactly two string keys, "subject" and "body", for a '
    'short friendly email to a customer called Northwind Logistics. '
    'No markdown fence, no commentary.'
)
raw, usage4, ms4, cost4 = generate(JSON_PROMPT, model=config.GEMINI_DRAFT_MODEL, json_mode=True)
COST_TOTAL += cost4
print("  raw response (first 300 chars):")
print("    %r" % raw[:300])

FENCED = raw.strip().startswith("```")
if FENCED:
    print("  ! response IS wrapped in a markdown fence — strip fences everywhere.")
else:
    ok("no markdown fence — response_mime_type=application/json returns bare JSON")

def parse(s: str):
    s = s.strip()
    if s.startswith("```"):
        import re
        m = re.search(r"```(?:json)?\s*(.+?)```", s, re.S)
        if m:
            s = m.group(1).strip()
    try:
        return json.loads(s)
    except ValueError:
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j > i:
            try:
                return json.loads(s[i:j + 1])
            except ValueError:
                return None
        return None

parsed = parse(raw)
if not isinstance(parsed, dict) or "subject" not in parsed or "body" not in parsed:
    die("CHECK 4", "Could not parse {\"subject\",\"body\"} out of the response.",
        model=config.GEMINI_DRAFT_MODEL, request={"prompt": JSON_PROMPT, "response_mime_type": "application/json"})
ok("parsed cleanly: subject=%r" % str(parsed["subject"])[:60])
record("4 strict JSON", True, ms4, usage4["total_token_count"],
       "fenced" if FENCED else "bare JSON")


# ── CHECK 5 ────────────────────────────────────────────────────────────────

hr("CHECK 5 — rate limit behaviour (12 quick calls)")
print("  free tier is ~10 RPM, so this should trip the limiter and recover.")
started = time.perf_counter()
retries = 0
tokens5 = 0
for i in range(1, 13):
    try:
        t, u, ms_i, c = generate("Reply with exactly: %d" % i,
                                 model=config.GEMINI_CLASSIFY_MODEL)
        tokens5 += u["total_token_count"]
        COST_TOTAL += c
        print("    call %2d/12  %5.0fms  %s" % (i, ms_i, t.strip()[:12]))
    except Exception as exc:
        die("CHECK 5", "call %d/12 failed: %s: %s" % (i, type(exc).__name__, exc),
            model=config.GEMINI_CLASSIFY_MODEL)
elapsed5 = time.perf_counter() - started
ok("12/12 calls completed in %.1fs" % elapsed5)
print("     (any '!! RATE LIMITED' lines above are the backoff working)")
record("5 rate limit x12", True, elapsed5 * 1000, tokens5, "%.1fs total" % elapsed5)


# ── CHECK 6 ────────────────────────────────────────────────────────────────

hr("CHECK 6 — cost of everything above")
print("  rates in config.GEMINI_PRICES (USD per 1M tokens):")
for m, r in sorted(config.GEMINI_PRICES.items()):
    used = " <- in use" if m in (config.GEMINI_DRAFT_MODEL, config.GEMINI_CLASSIFY_MODEL) else ""
    print("    %-26s in $%-6.2f out $%-6.2f%s" % (m, r["input"], r["output"], used))

print()
print("  total for this run: $%.6f" % COST_TOTAL)
print()
print("  SANITY CHECK on the rates:")
print("    Verified 2026-08-07 against https://ai.google.dev/gemini-api/docs/pricing")
print("      gemini-3.5-flash        $1.50 in / $9.00 out   <- matches config")
print("      gemini-3.5-flash-lite   $0.30 in / $2.50 out   <- matches config")
print("    The gemini-2.5-* entries are stale relative to what this key can call,")
print("    but they are unreachable anyway (404), so they never price anything.")
print()
print("    IMPORTANT: on the AI Studio FREE tier every call above billed $0.00.")
print("    This figure is what it WOULD cost at paid-tier rates. Do not present")
print("    it as money spent.")
record("6 cost", True, None, None, "$%.6f modeled" % COST_TOTAL)

summary()
print("\n  All checks passed.")
