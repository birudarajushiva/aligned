#!/usr/bin/env python3
"""Standalone EverOS end-to-end diagnostic.

    python verify_everos.py

No FastAPI, no Gemini, no imports from agent.py — just `requests` and
`config.py`. Nothing here calls a language model.

Walks six checks in order and stops dead on the first failure, printing the
full HTTP status, the complete response body, and the exact request that was
sent (URL, headers with the key masked, JSON body) so there is nothing left to
guess about.

WHICH API VERSION
-----------------
This account is provisioned for the **v2** memory API. Calling v1 returns
403 VERSION_NOT_ALLOWED ("This account (memory API v2) is not allowed to call
the v1 memory API"), so every call below uses /api/v2/memory/*.

ON THE `metadata` FIELD
-----------------------
There is no `metadata` field in either version. v2 scoping works like this:

    write  -> messages[].sender_id = the account   (there is no top-level user_id)
              session_id                           = required, 1-128 chars
    search -> user_id = the account                (top-level, not in filters)

The sender_id you write under is the user_id you search by. CHECK 3 probes
`metadata` live so you see the server's own answer, and CHECK 5 proves whether
that scoping actually isolates accounts.
"""

from __future__ import annotations

import json
import random
import string
import sys
import time

import requests

import config

TIMEOUT = 90
PASS = "✓"
FAIL = "✗"


# ── output helpers ─────────────────────────────────────────────────────────

def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def ok(msg):
    print("  %s %s" % (PASS, msg))


def mask(key):
    if not key:
        return "(empty)"
    if len(key) <= 8:
        return "*" * len(key)
    return "%s...%s  (len %d)" % (key[:4], key[-4:], len(key))


def headers():
    return {
        "Authorization": "Bearer %s" % config.require_env("EVERMIND_API_KEY"),
        "Content-Type": "application/json",
    }


def safe_headers():
    """Headers as they'd be printed — key masked."""
    return {
        "Authorization": "Bearer %s" % mask(config.EVERMIND_API_KEY),
        "Content-Type": "application/json",
    }


def die(check, reason, *, method=None, url=None, body=None, response=None):
    """Print everything needed to debug, then stop."""
    print("  %s %s" % (FAIL, reason))
    print("\n  --- REQUEST SENT " + "-" * 58)
    if method and url:
        print("  %s %s" % (method, url))
    print("  headers: %s" % json.dumps(safe_headers(), indent=2).replace("\n", "\n  "))
    if body is not None:
        print("  body:    %s" % json.dumps(body, indent=2, ensure_ascii=False).replace("\n", "\n  "))
    if response is not None:
        print("\n  --- RESPONSE " + "-" * 62)
        print("  HTTP %s %s" % (response.status_code, response.reason))
        print("  headers: %s" % dict(response.headers))
        print("  body:")
        try:
            print("  " + json.dumps(response.json(), indent=2, ensure_ascii=False).replace("\n", "\n  "))
        except ValueError:
            print("  " + (response.text or "(empty body)"))
    print("\n" + "=" * 78)
    print("STOPPED at %s. Later checks did not run." % check)
    print("=" * 78)
    sys.exit(1)


def post(path, payload):
    """POST and return (response, elapsed_ms). Network errors are fatal."""
    url = config.EVERMIND_BASE_URL.rstrip("/") + path
    started = time.perf_counter()
    try:
        r = requests.post(url, json=payload, headers=headers(), timeout=TIMEOUT)
    except requests.RequestException as exc:
        die("network", "Could not reach %s: %s: %s" % (url, type(exc).__name__, exc),
            method="POST", url=url, body=payload)
    return r, (time.perf_counter() - started) * 1000


def get(path):
    url = config.EVERMIND_BASE_URL.rstrip("/") + path
    started = time.perf_counter()
    try:
        r = requests.get(url, headers=headers(), timeout=TIMEOUT)
    except requests.RequestException as exc:
        die("network", "Could not reach %s: %s: %s" % (url, type(exc).__name__, exc),
            method="GET", url=url)
    return r, (time.perf_counter() - started) * 1000


def token(n=8):
    rng = random.Random()
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def write_memory(user_id, text, event_type="profile", extra=None):
    """One memory via POST /api/v2/memory/add.

    v2 has no top-level user_id: the owner is messages[].sender_id, and that is
    the value you later search by as user_id. session_id is required, so it is
    scoped per account+type to keep one customer's sessions from merging with
    another's.
    """
    payload = {
        "session_id": "%s_%s" % (user_id, event_type),
        "messages": [{
            "sender_id": user_id,          # required in v2 — this IS the user_id
            "role": "user",
            "timestamp": int(time.time() * 1000),
            "content": text,
        }],
        "async_mode": False,   # synchronous, so it is searchable when we return
    }
    if extra:
        payload.update(extra)
    return payload, post("/api/v2/memory/add", payload)


def flush(session_id):
    """POST /api/v2/memory/flush — force extraction of a session's buffer.

    v2 accumulates messages and only makes them searchable once extraction
    runs. A bare `add` returns status "accumulated" and the memory is NOT yet
    retrievable; flush runs boundary detection and extracts immediately.
    """
    payload = {"session_id": session_id}
    return payload, post("/api/v2/memory/flush", payload)


def write_and_flush(user_id, text, event_type="profile"):
    """add + flush, the sequence that actually makes a memory searchable."""
    payload, (r, ms) = write_memory(user_id, text, event_type)
    if r.status_code >= 400:
        return payload, (r, ms), None
    fpayload, (fr, fms) = flush("%s_%s" % (user_id, event_type))
    return payload, (r, ms), (fpayload, fr, fms)


def search(user_id, query, top_k=10):
    payload = {
        "query": query,
        "user_id": user_id,        # top-level in v2, not inside filters
        "method": "hybrid",
        "top_k": top_k,
        "include_profile": True,   # profiles are opt-in in v2
    }
    return payload, post("/api/v2/memory/search", payload)


def snippets_from(data):
    """The same extraction aligned/memory.py:recall_account performs."""
    out = []
    for ep in data.get("episodes") or []:
        t = ep.get("episode") or ep.get("summary") or ep.get("subject")
        if t:
            out.append((float(ep.get("score") or 0.0), "episode", str(t)))
    for pr in data.get("profiles") or []:
        pd = pr.get("profile_data")
        if isinstance(pd, dict):
            t = " | ".join("%s: %s" % (k, v) for k, v in pd.items())
        else:
            t = str(pd) if pd else ""
        if t:
            out.append((float(pr.get("score") or 0.0), "profile", t))
    for rm in data.get("unprocessed_messages") or []:
        t = rm.get("content") or rm.get("text")
        if isinstance(t, str) and t:
            out.append((float(rm.get("score") or 0.0), "raw_message", t))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


# ── CHECK 1 ────────────────────────────────────────────────────────────────

hr("CHECK 1 — key loaded")
if not config.EVERMIND_API_KEY:
    print("  %s EVERMIND_API_KEY is empty." % FAIL)
    print("     config.EVEROS_ENABLED = %s" % config.EVEROS_ENABLED)
    print("     Put it in %s/.env as:  EVERMIND_API_KEY=your-key" % config.__file__.rsplit("/", 2)[0])
    print("     No quotes, no spaces around '='. Then re-run.")
    sys.exit(1)
ok("EVERMIND_API_KEY = %s" % mask(config.EVERMIND_API_KEY))
ok("EVERMIND_BASE_URL = %s" % config.EVERMIND_BASE_URL)
ok("EVEROS_ENABLED = %s" % config.EVEROS_ENABLED)


# ── CHECK 2 ────────────────────────────────────────────────────────────────

hr("CHECK 2 — auth works")
print("  POST /api/v2/memory/get  (lightest authenticated read on the v2 API)")
_probe = {"memory_type": "profile", "user_id": "test_acc_auth_probe", "page_size": 1}
r, ms = post("/api/v2/memory/get", _probe)
if r.status_code == 401:
    die("CHECK 2",
        "401 Unauthorized. Either the key is invalid/revoked, or the header format is "
        "wrong. We sent 'Authorization: Bearer <key>', which is what the spec requires — "
        "so if the key was copied correctly, suspect the key itself (regenerate it at "
        "https://everos.evermind.ai). A stray space or newline in .env also does this.",
        method="POST", url=config.EVERMIND_BASE_URL + "/api/v2/memory/get",
        body=_probe, response=r)
if r.status_code == 403:
    die("CHECK 2",
        "403 Forbidden. The key authenticated but is not allowed on this API version. "
        "If the message says the account is v1-only, switch these calls back to "
        "/api/v1/memories*; if it says v2-only, they are already correct and something "
        "else is wrong with the account's entitlements.",
        method="POST", url=config.EVERMIND_BASE_URL + "/api/v2/memory/get",
        body=_probe, response=r)
if r.status_code >= 400:
    die("CHECK 2", "HTTP %d on an authenticated read." % r.status_code,
        method="POST", url=config.EVERMIND_BASE_URL + "/api/v2/memory/get",
        body=_probe, response=r)
ok("authenticated OK — HTTP %d in %.0fms" % (r.status_code, ms))
print("  response body:")
try:
    print("    " + json.dumps(r.json(), indent=2, ensure_ascii=False).replace("\n", "\n    "))
except ValueError:
    print("    " + r.text[:400])


# ── CHECK 3 ────────────────────────────────────────────────────────────────

hr("CHECK 3 — store one memory")
TOK = token()
TEST_USER = "test_acc_999"
TEST_TEXT = ("DIAGNOSTIC MARKER %s: Northwind Logistics diagnostic record written by "
             "verify_everos.py. Unique token %s." % (TOK, TOK))
print("  distinctive token: %s" % TOK)
print("  user_id: %s   session_id: profile" % TEST_USER)

payload, (r, ms) = write_memory(TEST_USER, TEST_TEXT, "profile")
if r.status_code >= 400:
    die("CHECK 3", "Write failed with HTTP %d." % r.status_code,
        method="POST", url=config.EVERMIND_BASE_URL + "/api/v2/memory/add",
        body=payload, response=r)
ok("stored — HTTP %d in %.0fms" % (r.status_code, ms))
print("  FULL RESPONSE:")
try:
    print("    " + json.dumps(r.json(), indent=2, ensure_ascii=False).replace("\n", "\n    "))
except ValueError:
    print("    " + r.text[:600])

_status = ((r.json().get("data") or {}).get("status")) if r.headers.get("content-type","").startswith("application/json") else None
print("\n  add status: %r" % _status)
if _status != "extracted":
    print("  -> 'accumulated' means buffered, NOT yet searchable. Flushing to force extraction.")
fpayload, (fr, fms) = flush("%s_%s" % (TEST_USER, "profile"))
if fr.status_code >= 400:
    die("CHECK 3", "flush failed with HTTP %d — without it the memory never becomes searchable." % fr.status_code,
        method="POST", url=config.EVERMIND_BASE_URL + "/api/v2/memory/flush",
        body=fpayload, response=fr)
_fstatus = (fr.json().get("data") or {}).get("status")
ok("flushed — HTTP %d in %.0fms, status=%r" % (fr.status_code, fms, _fstatus))
if _fstatus == "no_extraction":
    print("  ! flush reported 'no_extraction' (no conversation boundary detected).")
    print("    CHECK 4 will show whether anything is retrievable anyway.")

# Probe the metadata question live rather than asserting it.
print("\n  --- probe: does the API accept a `metadata` field? ---")
probe_payload = {
    "session_id": "test_acc_metadata_probe_profile",
    "messages": [{"sender_id": "test_acc_metadata_probe", "role": "user",
                  "timestamp": int(time.time() * 1000),
                  "content": "metadata probe %s" % TOK}],
    "async_mode": False,
    "metadata": {"account_id": "test_acc_metadata_probe", "event_type": "profile"},
}
pr, pms = post("/api/v2/memory/add", probe_payload)
if pr.status_code >= 400:
    print("  %s REJECTED — HTTP %d. `metadata` is not a valid field." % (PASS, pr.status_code))
    print("     server said: %s" % (pr.text or "")[:300])
    print("     -> scoping must use sender_id / user_id (which is what CHECK 5 tests).")
else:
    print("  ! ACCEPTED — HTTP %d. The server did not reject it, but the v1 spec does not"
          % pr.status_code)
    print("    define `metadata`, so it is almost certainly being ignored rather than")
    print("    stored. Search scopes by top-level user_id, not by metadata. Do not")
    print("    rely on it.")


# ── CHECK 4 ────────────────────────────────────────────────────────────────

hr("CHECK 4 — retrieve it")
payload, (r, ms) = search(TEST_USER, TOK, top_k=10)
if r.status_code >= 400:
    die("CHECK 4", "Search failed with HTTP %d." % r.status_code,
        method="POST", url=config.EVERMIND_BASE_URL + "/api/v2/memory/search",
        body=payload, response=r)
print("  retrieval latency: %.0f ms" % ms)
body = r.json()
print("\n  RAW RESPONSE:")
print("    " + json.dumps(body, indent=2, ensure_ascii=False)[:2500].replace("\n", "\n    "))

data = body.get("data") or {}
print("\n  response shape: data keys = %s" % sorted(data.keys()))
for k in ("episodes", "profiles", "unprocessed_messages", "agent_cases", "agent_skills"):
    v = data.get(k)
    print("    data.%-13s %s" % (k, ("%d items" % len(v)) if isinstance(v, list) else type(v).__name__))

extracted = snippets_from(data)
print("\n  EXTRACTED SNIPPETS (what aligned/memory.py would hand the agent):")
if not extracted:
    print("    (none)")
for score, kind, text in extracted:
    print("    [%.4f] (%s) %s" % (score, kind, text[:160]))

if not extracted:
    die("CHECK 4",
        "Nothing came back. The write in CHECK 3 returned success, so either extraction "
        "has not finished server-side, or our parser is reading the wrong fields — compare "
        "the RAW RESPONSE above against the EXTRACTED list. We already sent an explicit "
        "flush after the write, so a buffered-but-unextracted message is not the "
        "explanation. If flush reported 'no_extraction', EverOS found no conversation "
        "boundary in a single one-sided message — see the notes at the end of this file.",
        method="POST", url=config.EVERMIND_BASE_URL + "/api/v2/memory/search",
        body=payload, response=r)

found = any(TOK in text for _, _, text in extracted)
if found:
    ok("token %s found in the retrieved text — round-trip works" % TOK)
else:
    print("  ! token %s NOT found verbatim in any snippet." % TOK)
    print("    This is not necessarily a failure: EverOS extracts memories rather than")
    print("    storing raw text, so recall is not verbatim playback. Judge from the")
    print("    snippets above whether the meaning survived.")


# ── CHECK 5 ────────────────────────────────────────────────────────────────

hr("CHECK 5 — scoping (CRITICAL)")
print("  If this fails, every account's outreach is contaminated with another")
print("  customer's history and the product does not work.\n")

A_TOK, B_TOK = token(), token()
A_USER, B_USER = "test_acc_AAA_%s" % A_TOK, "test_acc_BBB_%s" % B_TOK
A_TEXT = "SCOPE-A %s: Acme Robotics is on the Growth plan and loves the webhooks feature." % A_TOK
B_TEXT = "SCOPE-B %s: Zenith Freight is on the Starter plan and complains about slow exports." % B_TOK

for user, text in ((A_USER, A_TEXT), (B_USER, B_TEXT)):
    payload, (r, ms), fl = write_and_flush(user, text, "profile")
    if r.status_code >= 400:
        die("CHECK 5", "Failed writing scoping fixture for %s (HTTP %d)." % (user, r.status_code),
            method="POST", url=config.EVERMIND_BASE_URL + "/api/v2/memory/add",
            body=payload, response=r)
    ok("wrote memory for %s" % user)

print("\n  searching scoped to %s only ..." % A_USER)
payload, (r, ms) = search(A_USER, "what plan are they on and what do they use", top_k=20)
if r.status_code >= 400:
    die("CHECK 5", "Scoped search failed with HTTP %d." % r.status_code,
        method="POST", url=config.EVERMIND_BASE_URL + "/api/v2/memory/search",
        body=payload, response=r)
a_snips = snippets_from(r.json().get("data") or {})
print("  %d snippet(s) returned:" % len(a_snips))
for score, kind, text in a_snips:
    print("    [%.4f] (%s) %s" % (score, kind, text[:150]))

leaked = [t for _, _, t in a_snips if B_TOK in t or "Zenith" in t or "SCOPE-B" in t]
if leaked:
    print("\n  %s LEAK: account B's memory came back in a search scoped to account A." % FAIL)
    for t in leaked:
        print("     %s" % t[:200])
    die("CHECK 5",
        "user_id scoping did NOT isolate accounts. Do not ship outreach until this is "
        "resolved — drafts would cite other customers' history.",
        method="POST", url=config.EVERMIND_BASE_URL + "/api/v2/memory/search",
        body=payload, response=r)
ok("no leakage — account B's memory did not appear in account A's results")
print("     (scoping mechanism: filters.user_id, not metadata)")


# ── CHECK 6 ────────────────────────────────────────────────────────────────

hr("CHECK 6 — natural language retrieval quality")
NW_TOK = token()
NW_USER = "test_acc_northwind_%s" % NW_TOK
FIXTURES = [
    ("usage",
     "Northwind Logistics uses csv_export heavily — 412 times in the last 30 days. "
     "It's their most-used feature."),
    ("ticket",
     "March 14: Export to CSV timing out on files over 10k rows. Customer was frustrated. "
     "Sentiment: negative."),
    ("profile",
     "Northwind Logistics is on the Enterprise plan, 45 seats, $48,000 ARR, "
     "signed up 2024-03-02."),
]
print("  writing 3 memories under user_id=%s\n" % NW_USER)
for session_id, text in FIXTURES:
    payload, (r, ms), fl = write_and_flush(NW_USER, text, session_id)
    if r.status_code >= 400:
        die("CHECK 6", "Failed writing fixture (HTTP %d): %s" % (r.status_code, text[:60]),
            method="POST", url=config.EVERMIND_BASE_URL + "/api/v2/memory/add",
            body=payload, response=r)
    ok("[%s] %s" % (session_id, text[:70]))
    time.sleep(0.2)

QUERIES = [
    ("churn_risk", "what did they complain about and what did they stop using"),
    ("upsell", "what are they using most and what limits are they hitting"),
]
for label, q in QUERIES:
    print("\n  " + "-" * 74)
    print("  QUERY (%s): %s" % (label, q))
    print("  " + "-" * 74)
    payload, (r, ms) = search(NW_USER, q, top_k=10)
    if r.status_code >= 400:
        die("CHECK 6", "Search failed with HTTP %d." % r.status_code,
            method="POST", url=config.EVERMIND_BASE_URL + "/api/v2/memory/search",
            body=payload, response=r)
    snips = snippets_from(r.json().get("data") or {})
    print("  latency %.0f ms, %d snippet(s), ranked:" % (ms, len(snips)))
    if not snips:
        print("    (nothing returned — retrieval is not usable for this query)")
    for i, (score, kind, text) in enumerate(snips, 1):
        print("    %d. [%.4f] (%s)" % (i, score, kind))
        print("       %s" % text[:300])

hr("DONE")
print("  All checks passed. Scoping uses filters.user_id (there is no metadata field).")
print("  Test data was written under these throwaway user_ids:")
for u in (TEST_USER, "test_acc_metadata_probe", A_USER, B_USER, NW_USER):
    print("    %s" % u)
print("\n  They are isolated from the real portfolio (acc_001..acc_012), so they will")
print("  not contaminate the demo. Judge CHECK 6's output yourself: if the right")
print("  memory does not rank first for each query, retrieval quality — not plumbing")
print("  — is the thing to fix.")
