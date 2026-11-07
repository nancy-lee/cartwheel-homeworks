"""Local web chat for the Cartwheel agent, going through the HW2 server.

Every chat is sent to the traced endpoint (`server.app`, default
http://127.0.0.1:8010) exactly as the HW3 scenario runner does, so each message
produces a Langfuse trace. The page also shows what the server's reply hides:
reasoning summaries, narration before a tool call, and each tool call with its
arguments and result. Those are read from `.sessions.db`, which the server
writes. An optional scenario id is attached to the trace.

(HW1's "record this chat" panel was removed; HW3/HW4 review tools come later.)

Run from the repo root, with Langfuse and the server already up:
    uv run uvicorn server.app:app --port 8010      # terminal 1
    uv run python scripts/chat_ui.py               # terminal 2, then open
                                                   # http://127.0.0.1:8765

It binds to localhost only and never sends or displays API keys.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agent.agent import prompt_version  # noqa: E402
from agent.cli import SESSIONS_DB  # noqa: E402
from agent.config import db_path  # noqa: E402
from observability.instrument import load_env  # noqa: E402
from scripts import record_session as rs  # noqa: E402

HOST, PORT = "127.0.0.1", 8765
SERVER_URL = "http://127.0.0.1:8010"  # the HW2 traced endpoint
LANGFUSE_URL = "http://localhost:3000/project/cartwheel-dev"
PAGE = Path(__file__).with_name("chat_ui.html")
# Append-only index of chats started from this page (who, which model, scenario ids).
# `.sessions.db` holds the transcripts but not who the chat was as. This file is
# kept out of git through .git/info/exclude.
HISTORY_FILE = REPO_ROOT / ".chat_ui_history.jsonl"

app = FastAPI(title="Cartwheel chat")
STATE: dict = {}  # one active chat at a time: session_id, token, role, user_id, model


class StartBody(BaseModel):
    role: str
    user_id: int
    model: str | None = None


class MessageBody(BaseModel):
    text: str
    scenario_id: str | None = None


def _log_history(entry: dict) -> None:
    entry = {**entry, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _read_history_index() -> dict[str, dict]:
    """session_id -> {role, user_id, model, scenario_ids}, from the append-only file."""
    meta: dict[str, dict] = {}
    if not HISTORY_FILE.exists():
        return meta
    for line in HISTORY_FILE.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        m = meta.setdefault(e.get("session_id", ""), {"scenario_ids": []})
        if e.get("type") == "start":
            m.update(role=e.get("role"), user_id=e.get("user_id"), model=e.get("model"))
        elif e.get("type") == "scenario" and e.get("scenario_id") not in m["scenario_ids"]:
            m["scenario_ids"].append(e["scenario_id"])
    return meta


def _call_server(method: str, path: str, **kwargs) -> dict:
    """Call the HW2 server; turn connection and HTTP errors into readable ones."""
    try:
        resp = httpx.request(method, f"{SERVER_URL}{path}", timeout=300, **kwargs)
    except httpx.ConnectError:
        raise HTTPException(
            502,
            f"Cannot reach the Cartwheel server at {SERVER_URL}. Start it with: "
            "uv run uvicorn server.app:app --port 8010",
        )
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}")
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise HTTPException(resp.status_code, f"server: {detail}")
    return resp.json()


def _require_chat() -> None:
    if "session_id" not in STATE:
        raise HTTPException(409, "start a chat first")


def _max_item_id() -> int:
    import sqlite3

    with sqlite3.connect(SESSIONS_DB) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM agent_messages WHERE session_id = ?",
            (STATE["session_id"],),
        ).fetchone()
    return row[0]


def _items_after(item_id: int) -> list[dict]:
    import sqlite3

    with sqlite3.connect(SESSIONS_DB) as conn:
        rows = conn.execute(
            "SELECT message_data FROM agent_messages WHERE session_id = ? AND id > ? ORDER BY id",
            (STATE["session_id"], item_id),
        ).fetchall()
    return [json.loads(r[0]) for r in rows]


def to_events(items: list[dict]) -> list[dict]:
    """Stored items -> display events, in the order the agent produced them."""
    outputs = {
        i["call_id"]: rs.parse_value(i.get("output"))
        for i in items
        if i.get("type") == "function_call_output"
    }
    events: list[dict] = []
    for item in items:
        kind = item.get("type")
        if kind == "reasoning":
            text = "\n".join(p.get("text", "") for p in item.get("summary") or [])
            if text.strip():
                events.append({"kind": "reasoning", "text": text})
        elif kind == "function_call":
            events.append(
                {
                    "kind": "tool",
                    "name": item["name"],
                    "arguments": rs.parse_value(item.get("arguments")),
                    "result": outputs.get(item["call_id"]),
                }
            )
        elif kind == "message" and item.get("role") == "assistant":
            events.append({"kind": "assistant", "text": rs.message_text(item)})
    # Text the agent wrote before a tool call is narration; the last assistant
    # message of the turn is the final reply.
    last_assistant = max(
        (n for n, e in enumerate(events) if e["kind"] == "assistant"), default=None
    )
    for n, event in enumerate(events):
        if event["kind"] == "assistant":
            later_tool = any(e["kind"] == "tool" for e in events[n + 1 :])
            event["kind"] = "narration" if later_tool and n != last_assistant else "final"
    return events


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE.read_text()


def _default_model() -> str:
    import os

    load_env()
    return os.environ.get("CARTWHEEL_MODEL") or "gpt-4o-mini"


@app.get("/api/status")
def status() -> dict:
    return {
        "model": _default_model(),
        "session_id": STATE.get("session_id"),
        "role": STATE.get("role"),
        "user_id": STATE.get("user_id"),
        "server_url": SERVER_URL,
        "langfuse_url": LANGFUSE_URL,
    }


@app.post("/api/start")
def start(body: StartBody) -> dict:
    # The server checks the user exists and that the role matches the database.
    data = _call_server("POST", "/sessions", json={"user_id": body.user_id, "role": body.role})
    STATE.clear()
    STATE.update(
        session_id=data["session_id"],
        token=data["token"],
        role=body.role,
        user_id=body.user_id,
        model=body.model or None,
    )
    _log_history(
        {"type": "start", "session_id": data["session_id"], "role": body.role,
         "user_id": body.user_id, "model": body.model or _default_model()}
    )
    return {
        "session_id": data["session_id"],
        "role": body.role,
        "user_id": body.user_id,
        "prompt_version": prompt_version(),
    }


# A plain `def` (not async) so FastAPI runs the slow server call in a worker
# thread and the page's other requests are not blocked while the agent thinks.
@app.post("/api/message")
def message(body: MessageBody) -> dict:
    _require_chat()
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty message")
    before = _max_item_id()
    payload = {"message": text, "model": STATE.get("model")}
    if body.scenario_id and body.scenario_id.strip():
        payload["scenario_id"] = body.scenario_id.strip()
        _log_history({"type": "scenario", "session_id": STATE["session_id"],
                      "scenario_id": payload["scenario_id"]})
    reply = _call_server(
        "POST",
        f"/sessions/{STATE['session_id']}/messages",
        json=payload,
        headers={"Authorization": f"Bearer {STATE['token']}"},
    )
    return {
        "events": to_events(_items_after(before)),
        "prompt_version": reply.get("prompt_version"),
    }


# ---------------------------------------------------------------------------
# Plan review (HW3 Part B, step B1): the draft pilot tuples and their answer keys
# Both files live in scenarios/results/, which is gitignored scratch space. Your
# decisions here are about the DRAFT PLAN; they are not the graded pilot_review.jsonl.
# ---------------------------------------------------------------------------

PLAN_FILE = REPO_ROOT / "scenarios" / "results" / "pilot_plan.json"
PLAN_REVIEW_FILE = REPO_ROOT / "scenarios" / "results" / "pilot_plan_review.json"


class PlanDecision(BaseModel):
    decision: str | None = None  # accept | revise | reject | None (clear)
    note: str = ""


MESSAGES_FILE = REPO_ROOT / "scenarios" / "results" / "pilot_messages.json"          # writer draft -> critic -> final
MESSAGES_REVIEW_FILE = REPO_ROOT / "scenarios" / "results" / "pilot_messages_review.json"


def _review_file(kind: str) -> Path:
    if kind not in ("plan", "messages"):
        raise HTTPException(400, "kind must be 'plan' or 'messages'")
    return MESSAGES_REVIEW_FILE if kind == "messages" else PLAN_REVIEW_FILE


def _read_plan_review(kind: str = "plan") -> dict:
    path = _review_file(kind)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except ValueError:
        return {}


@app.get("/api/plan")
def get_plan(kind: str = "plan") -> dict:
    if not PLAN_FILE.exists():
        raise HTTPException(404, "No draft plan yet (scenarios/results/pilot_plan.json).")
    plan = json.loads(PLAN_FILE.read_text())
    if kind == "messages":
        if not MESSAGES_FILE.exists():
            raise HTTPException(404, "No messages yet (scenarios/results/pilot_messages.json).")
        messages = json.loads(MESSAGES_FILE.read_text())
        for p in plan["plans"]:
            p["messages"] = messages.get(p["plan_id"])
    return {**plan, "decisions": _read_plan_review(kind)}


@app.post("/api/plan/{plan_id}/decision")
def set_plan_decision(plan_id: str, body: PlanDecision, kind: str = "plan") -> dict:
    plan = json.loads(PLAN_FILE.read_text()) if PLAN_FILE.exists() else {"plans": []}
    if plan_id not in {p["plan_id"] for p in plan["plans"]}:
        raise HTTPException(404, f"unknown plan item: {plan_id}")
    if body.decision not in (None, "accept", "revise", "reject"):
        raise HTTPException(400, "decision must be accept, revise, reject, or null")
    review = _read_plan_review(kind)
    if body.decision is None and not body.note.strip():
        review.pop(plan_id, None)
    else:
        review[plan_id] = {"decision": body.decision, "note": body.note.strip(),
                           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _review_file(kind).write_text(json.dumps(review, indent=2))
    return {"ok": True, "reviewed": sum(1 for v in review.values() if v.get("decision"))}


# ---------------------------------------------------------------------------
# Pilot results review (HW3 Part B, step B5)
# Shows each scenario's conversation, the traced tool calls, the database now, and the
# answer key, with a form that writes YOUR decisions to scenarios/pilot_review.jsonl.
# Nothing is pre-filled and nothing is judged for you.
# ---------------------------------------------------------------------------

PILOT_SCENARIOS = REPO_ROOT / "scenarios" / "pilot_scenarios.jsonl"
PILOT_RESULTS = REPO_ROOT / "scenarios" / "pilot-results.jsonl"
PILOT_REVIEW = REPO_ROOT / "scenarios" / "pilot_review.jsonl"
LANGFUSE_TRACE_URL = LANGFUSE_URL + "/traces/"


class ResultReview(BaseModel):
    scenario_valid: bool
    confirmed_failure: bool
    evidence: str
    scenario_change: str | None = None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _clickhouse(query: str) -> list[dict]:
    """Read-only query against Langfuse's ClickHouse (local docker). Returns [] if unavailable."""
    cmd = ["docker", "compose", "-f", str(REPO_ROOT / "observability" / "docker-compose.yml"), "exec", "-T",
           "clickhouse", "clickhouse-client", "--user", "clickhouse", "--password", "clickhouse",
           "--database", "default", "--query", query + " FORMAT JSONEachRow"]
    try:
        done = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=40)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if done.returncode != 0:
        return []
    return [json.loads(line) for line in done.stdout.splitlines() if line.strip()]


def _pilot_traces(turns_by_scenario: dict[str, int]) -> tuple[dict[str, list[dict]], bool]:
    """scenario_id -> [{trace_id, tools:[...]}] (the latest N traces, N = turns). Second value: store reachable."""
    rows = _clickhouse(
        "SELECT id, toString(timestamp) AS ts, JSONExtractString(metadata['attributes'], 'cartwheel.scenario_id') AS sid "
        "FROM traces WHERE sid LIKE 'pilot-%' ORDER BY timestamp")
    if not rows:
        return {}, False
    by_sid: dict[str, list[dict]] = {}
    for r in rows:
        by_sid.setdefault(r["sid"], []).append({"trace_id": r["id"], "tools": []})
    for sid, traces in by_sid.items():
        by_sid[sid] = traces[-turns_by_scenario.get(sid, len(traces)):]
    wanted = [t["trace_id"] for ts in by_sid.values() for t in ts]
    if wanted:
        ids = ",".join("'" + w + "'" for w in wanted)
        tools = _clickhouse(
            "SELECT trace_id, name, substring(input, 1, 700) AS i, substring(output, 1, 1100) AS o "
            f"FROM observations WHERE type = 'TOOL' AND trace_id IN ({ids}) ORDER BY start_time")
        index = {t["trace_id"]: t for ts in by_sid.values() for t in ts}
        for r in tools:
            index[r["trace_id"]]["tools"].append({"name": r["name"], "input": r["i"], "output": r["o"]})
    return by_sid, True


def _db_snapshot(order_ids: list[int]) -> dict:
    if not order_ids:
        return {"orders": [], "refunds": []}
    marks = ",".join("?" * len(order_ids))
    with _ro_conn() as conn:
        orders = [dict(r) for r in conn.execute(
            f"SELECT id, status, total_cents, delivered_at FROM orders WHERE id IN ({marks}) ORDER BY id", order_ids)]
        refunds = [dict(r) for r in conn.execute(
            f"SELECT id, order_id, amount_cents, status, created_at FROM refunds WHERE order_id IN ({marks}) ORDER BY id", order_ids)]
    return {"orders": orders, "refunds": refunds}


def _read_pilot_reviews() -> dict[str, dict]:
    return {r["scenario_id"]: r for r in _read_jsonl(PILOT_REVIEW)}


def _write_pilot_reviews(reviews: dict[str, dict]) -> None:
    ordered = [reviews[k] for k in sorted(reviews)]
    PILOT_REVIEW.write_text("".join(json.dumps(r) + "\n" for r in ordered))


@app.get("/api/results")
def get_results() -> dict:
    scenarios = {s["id"]: s for s in _read_jsonl(PILOT_SCENARIOS)}
    results = {r["scenario_id"]: r for r in _read_jsonl(PILOT_RESULTS)}
    if not results:
        raise HTTPException(404, "No pilot results yet (scenarios/pilot-results.jsonl).")
    traces, store_up = _pilot_traces({k: len(v["turns"]) for k, v in results.items()})
    plan = {p["plan_id"]: p for p in json.loads(PLAN_FILE.read_text())["plans"]} if PLAN_FILE.exists() else {}
    reviews = _read_pilot_reviews()
    items = []
    for sid, res in results.items():
        sc = scenarios.get(sid, {})
        t = sc.get("tuple", {})
        order_ids = [x for x in (t.get("order_id"), t.get("first_order_id")) if x]
        items.append({
            "scenario_id": sid, "scenario_group": res["scenario_group"], "status": res["status"],
            "tuple": t, "user_name": plan.get(sid, {}).get("user_name"),
            "record": plan.get(sid, {}).get("record"), "ui_hint": plan.get(sid, {}).get("ui_hint"),
            "turns": res["turns"], "expected": res["expected"], "duration_s": res["duration_s"],
            "traces": [{"trace_id": x["trace_id"], "url": LANGFUSE_TRACE_URL + x["trace_id"], "tools": x["tools"]}
                       for x in traces.get(sid, [])],
            "database_now": _db_snapshot(order_ids), "review": reviews.get(sid),
        })
    items.sort(key=lambda x: x["scenario_id"])
    return {"items": items, "trace_store_reachable": store_up, "model": next(iter(results.values())).get("model")}


@app.post("/api/results/{scenario_id}/review")
def save_result_review(scenario_id: str, body: ResultReview) -> dict:
    known = {r["scenario_id"] for r in _read_jsonl(PILOT_RESULTS)}
    if scenario_id not in known:
        raise HTTPException(404, f"unknown scenario: {scenario_id}")
    if not body.evidence.strip():
        raise HTTPException(422, "Evidence is required: name the database value, policy, tool result, or requirement behind your decision.")
    if body.confirmed_failure and not body.scenario_valid:
        raise HTTPException(422, "A failure only counts when the scenario is valid. Mark the scenario valid, or mark it not a failure.")
    reviews = _read_pilot_reviews()
    change = (body.scenario_change or "").strip() or None
    reviews[scenario_id] = {"scenario_id": scenario_id, "scenario_valid": body.scenario_valid,
                            "confirmed_failure": body.confirmed_failure, "evidence": body.evidence.strip(),
                            "scenario_change": change}
    _write_pilot_reviews(reviews)
    return _review_counts(reviews)


@app.delete("/api/results/{scenario_id}/review")
def delete_result_review(scenario_id: str) -> dict:
    reviews = _read_pilot_reviews()
    reviews.pop(scenario_id, None)
    _write_pilot_reviews(reviews)
    return _review_counts(reviews)


def _review_counts(reviews: dict[str, dict]) -> dict:
    return {"ok": True, "reviewed": len(reviews),
            "confirmed_failures": sum(1 for r in reviews.values() if r["scenario_valid"] and r["confirmed_failure"])}


# ---------------------------------------------------------------------------
# Final scenario review (HW3 Part C): read the generated conversations and record
# your accept / revise / reject decision. Writes scenarios/support_review.jsonl
# from YOUR entries only; nothing is pre-filled.
# ---------------------------------------------------------------------------

FINAL_SCENARIOS = REPO_ROOT / "scenarios" / "support_scenarios.jsonl"
FINAL_REVIEW = REPO_ROOT / "scenarios" / "support_review.jsonl"
FINAL_PLAN = REPO_ROOT / "scenarios" / "results" / "final_plan.json"
FINAL_SAMPLE = REPO_ROOT / "scenarios" / "results" / "final_sample.json"


class FinalReview(BaseModel):
    decision: str            # accept | revise | reject
    reason: str
    change: str | None = None


def _read_final_reviews() -> dict[str, dict]:
    return {r["scenario_id"]: r for r in _read_jsonl(FINAL_REVIEW)}


def _write_final_reviews(reviews: dict[str, dict]) -> None:
    ordered = [reviews[k] for k in sorted(reviews)]
    FINAL_REVIEW.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))


def _final_progress(items: list[dict]) -> dict:
    rev = [i for i in items if i["review"]]
    return {
        "reviewed": len(rev),
        "decisions": {d: sum(1 for i in rev if i["review"]["decision"] == d) for d in ("accept", "revise", "reject")},
        "groups": sorted({i["scenario_group"] for i in rev}),
        "roles": sorted({i["tuple"]["role"] for i in rev}),
        "intents": sorted({i["tuple"]["intent"] for i in rev}),
    }


@app.get("/api/final")
def get_final() -> dict:
    scenarios = _read_jsonl(FINAL_SCENARIOS)
    if not scenarios:
        raise HTTPException(404, "No final scenarios yet (scenarios/support_scenarios.jsonl).")
    plan = {p["plan_id"]: p for p in json.loads(FINAL_PLAN.read_text())["plans"]} if FINAL_PLAN.exists() else {}
    sample = set(json.loads(FINAL_SAMPLE.read_text())) if FINAL_SAMPLE.exists() else set()
    reviews = _read_final_reviews()
    items = []
    for s in scenarios:
        p = plan.get(s["id"], {})
        items.append({
            "scenario_id": s["id"], "scenario_group": s["scenario_group"], "tuple": s["tuple"],
            "data_quality_case_id": s["data_quality_case_id"],
            "opening_message": s["opening_message"], "followups": s["followups"], "expected": s["expected"],
            "user_name": p.get("user_name"), "record": p.get("record"), "ui_hint": p.get("ui_hint"),
            "writer_was_told": p.get("user_visible_facts"), "state_changing": p.get("state_changing", False),
            "in_sample": s["id"] in sample, "review": reviews.get(s["id"]),
        })
    return {"items": items, "progress": _final_progress(items)}


@app.post("/api/final/{scenario_id}/review")
def save_final_review(scenario_id: str, body: FinalReview) -> dict:
    known = {s["id"] for s in _read_jsonl(FINAL_SCENARIOS)}
    if scenario_id not in known:
        raise HTTPException(404, f"unknown scenario: {scenario_id}")
    if body.decision not in ("accept", "revise", "reject"):
        raise HTTPException(422, "decision must be accept, revise, or reject.")
    if not body.reason.strip():
        raise HTTPException(422, "Give a reason for your decision.")
    change = (body.change or "").strip() or None
    if body.decision == "revise" and not change:
        raise HTTPException(422, "A revise decision needs the change you want (what should the message say or do differently?).")
    reviews = _read_final_reviews()
    reviews[scenario_id] = {"scenario_id": scenario_id, "decision": body.decision,
                            "reason": body.reason.strip(), "change": change}
    _write_final_reviews(reviews)
    return {"ok": True, "reviewed": len(reviews)}


@app.delete("/api/final/{scenario_id}/review")
def delete_final_review(scenario_id: str) -> dict:
    reviews = _read_final_reviews()
    reviews.pop(scenario_id, None)
    _write_final_reviews(reviews)
    return {"ok": True, "reviewed": len(reviews)}


# ---------------------------------------------------------------------------
# Chat history (read-only): transcripts come from .sessions.db, "who" from the index
# ---------------------------------------------------------------------------


def _user_text(item: dict) -> str | None:
    if item.get("role") != "user" or item.get("type") in ("function_call", "function_call_output"):
        return None
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _session_items(session_id: str) -> list[dict]:
    import sqlite3

    with sqlite3.connect(f"file:{SESSIONS_DB}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT message_data FROM agent_messages WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    return [json.loads(r[0]) for r in rows]


def transcript(items: list[dict]) -> list[dict]:
    """Whole chat -> display events: each user message, then that turn's agent events."""
    events: list[dict] = []
    turn: list[dict] = []

    def flush() -> None:
        if turn:
            events.extend(to_events(turn))
            turn.clear()

    for item in items:
        text = _user_text(item)
        if text is not None:
            flush()
            events.append({"kind": "user", "text": text})
        else:
            turn.append(item)
    flush()
    return events


@app.get("/api/history")
def history() -> dict:
    import sqlite3

    index = _read_history_index()
    with sqlite3.connect(f"file:{SESSIONS_DB}?mode=ro", uri=True) as conn:
        sessions = conn.execute(
            "SELECT session_id, created_at FROM agent_sessions ORDER BY created_at DESC, rowid DESC LIMIT 200"
        ).fetchall()
    out = []
    for session_id, created_at in sessions:
        items = _session_items(session_id)
        users = [t for t in map(_user_text, items) if t is not None]
        if not users:
            continue  # chat was started but never used
        meta = index.get(session_id, {})
        out.append(
            {
                "session_id": session_id,
                "created_at": created_at,  # UTC, "YYYY-MM-DD HH:MM:SS"
                "first_message": users[0],
                "turns": len(users),
                "role": meta.get("role"),
                "user_id": meta.get("user_id"),
                "model": meta.get("model"),
                "scenario_ids": meta.get("scenario_ids", []),
                "active": session_id == STATE.get("session_id"),
            }
        )
    return {"chats": out}


@app.get("/api/history/{session_id}")
def history_one(session_id: str) -> dict:
    items = _session_items(session_id)
    if not items:
        raise HTTPException(404, "no stored messages for that session")
    meta = _read_history_index().get(session_id, {})
    return {"session_id": session_id, "events": transcript(items), "meta": meta,
            "active": session_id == STATE.get("session_id")}


# ---------------------------------------------------------------------------
# People browser (read-only view of the seeded database, for choosing who to chat as)
# This is a developer's view of the data. It is NOT what the agent can see: the
# agent only sees what AUTH-1 allows the chosen role.
# ---------------------------------------------------------------------------


def _ro_conn():
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/api/users")
def list_users(role: str = "shopper", q: str = "") -> dict:
    if role not in ("shopper", "merchant", "support"):
        raise HTTPException(400, f"unknown role: {role!r}")
    like = f"%{q.strip().lower()}%"
    with _ro_conn() as conn:
        rows = conn.execute(
            """
            SELECT u.id, u.name, u.role, u.store_id, s.name AS store,
                   s.return_window_days_override AS window_override,
                   s.restocking_fee_opt_in AS restocking_fee,
                   CASE u.role
                     WHEN 'shopper' THEN (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id)
                     WHEN 'merchant' THEN (SELECT COUNT(*) FROM orders o WHERE o.store_id = u.store_id)
                     ELSE NULL END AS order_count
            FROM users u LEFT JOIN stores s ON s.id = u.store_id
            WHERE u.role = ?
              AND (LOWER(u.name) LIKE ? OR CAST(u.id AS TEXT) LIKE ? OR LOWER(COALESCE(s.name, '')) LIKE ?)
            ORDER BY order_count DESC, u.id
            LIMIT 100
            """,
            (role, like, like, like),
        ).fetchall()
    return {"users": [dict(r) for r in rows]}


@app.get("/api/users/{user_id}/orders")
def user_orders(user_id: int) -> dict:
    with _ro_conn() as conn:
        user = conn.execute("SELECT id, name, role, store_id FROM users WHERE id = ?", (user_id,)).fetchone()
        if user is None:
            raise HTTPException(404, f"unknown user: {user_id}")
        if user["role"] == "support":
            return {"user": dict(user), "orders": [], "total": 0,
                    "note": "Support can view any order, so there is no personal order list."}
        where, arg = ("o.user_id = ?", user_id) if user["role"] == "shopper" else ("o.store_id = ?", user["store_id"])
        total = conn.execute(f"SELECT COUNT(*) FROM orders o WHERE {where}", (arg,)).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT o.id, p.title AS product, s.name AS store, o.quantity,
                   o.total_cents, o.status, o.ordered_at, o.delivered_at, o.refund_eligible,
                   (SELECT case_id FROM data_quality_cases d
                     WHERE d.entity_type = 'order' AND d.entity_id = o.id) AS dq_case
            FROM orders o
            JOIN products p ON p.id = o.product_id
            JOIN stores s ON s.id = o.store_id
            WHERE {where}
            ORDER BY o.ordered_at DESC, o.id DESC
            LIMIT 60
            """,
            (arg,),
        ).fetchall()
    return {"user": dict(user), "orders": [dict(r) for r in rows], "total": total}


@app.post("/api/reseed")
def reseed() -> dict:
    """Restore the original seeded world (undoes refunds, cancellations, tickets)."""
    done = subprocess.run(
        [sys.executable, "-m", "seed.generate"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if done.returncode != 0:
        raise HTTPException(500, done.stderr[-500:])
    return {"ok": True, "output": done.stdout.strip().splitlines()[0] if done.stdout else ""}


if __name__ == "__main__":
    print(f"Cartwheel chat at http://{HOST}:{PORT}  (Ctrl-C to stop)")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
