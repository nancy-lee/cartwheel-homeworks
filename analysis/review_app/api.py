"""Conversation review API for Homework 4 (open coding).

One conversation per ``cartwheel.session_id``: the turns in time order, each
turn's tool calls and results attached to the turn that made them. Traces come
from live Langfuse (the canonical store). The Homework 3 export is used only
when Langfuse is unavailable, and the response says so.

Notes are free text. A note is anchored to the evidence it is about: a span of
text, a whole message, a tool call, or the whole conversation. "No failure
observed" is a session-level marker, so a reviewed trace can be told apart from
an unreviewed one. Notes are saved to ``analysis/state/annotations.json``.

The scenario's answer key and its dimensions are not part of the default
responses: the reviewer reads the trace first. They are available from
``/api/review/scenario/{scenario_id}`` when the reviewer asks for them.

Mounted by ``scripts/chat_ui.py`` (page at ``/review``).
"""

from __future__ import annotations

import ast
import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from analysis.helpers import _state, selection

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ANNOTATIONS_FILE = REPO_ROOT / "analysis" / "state" / "annotations.json"
EXPORT_FILE = REPO_ROOT / "traces" / "support_traces.json"
SCENARIOS_FILE = REPO_ROOT / "scenarios" / "support_scenarios.jsonl"
PLAN_FILE = REPO_ROOT / "scenarios" / "results" / "final_plan.json"
EXCLUDED_FILE = REPO_ROOT / "analysis" / "state" / "excluded_scenarios.json"

ANCHOR_KINDS = ("session", "message", "tool", "text")
NOTE_KINDS = ("failure", "no_failure")

router = APIRouter()
_lock = threading.Lock()
_cache: dict[str, Any] = {"sessions": None, "traces": [], "source": None, "reason": None}


# ---------------------------------------------------------------------------
# small file helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _excluded() -> dict[str, dict[str, Any]]:
    """Scenarios left out of the review set because they are unrealistic (see excluded_scenarios.json)."""
    return {e["scenario_id"]: e for e in _read_json(EXCLUDED_FILE, {}).get("excluded", [])}


def _scenarios() -> dict[str, dict[str, Any]]:
    if not SCENARIOS_FILE.exists():
        return {}
    out = {}
    for line in SCENARIOS_FILE.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


# ---------------------------------------------------------------------------
# loading traces (live Langfuse first, the export only as a fallback)
# ---------------------------------------------------------------------------


def _fetch_traces() -> tuple[list[dict[str, Any]], str, str | None]:
    reason: str | None = None
    try:
        from observability.instrument import load_env

        load_env()
    except Exception:  # pragma: no cover - env loading is best effort
        pass
    from analysis.helpers import langfuse_io

    if langfuse_io.is_configured():
        try:
            return langfuse_io.fetch_traces(limit=2000), "langfuse", None
        except Exception as exc:  # network or auth problem: say so, fall back
            reason = f"Langfuse could not be reached ({type(exc).__name__}: {exc})"
    else:
        reason = "Langfuse is not configured (LANGFUSE_* variables missing)"

    from analysis.helpers.normalization import normalize_trace

    raw = _read_json(EXPORT_FILE, {}).get("traces", [])
    return [normalize_trace(t) for t in raw], "export", reason


def _system_prompt(trace: dict[str, Any]) -> str | None:
    for obs in trace.get("observations") or []:
        if obs.get("type") != "GENERATION":
            continue
        inp = obs.get("input")
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except ValueError:
                try:
                    inp = ast.literal_eval(inp)
                except (ValueError, SyntaxError):
                    inp = None
        if isinstance(inp, dict):
            for msg in inp.get("messages") or []:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    parts = [p.get("content", "") for p in msg.get("parts") or [] if isinstance(p, dict)]
                    text = "\n".join(p for p in parts if p)
                    if text:
                        return text
        return None
    return None


def _turn(trace: dict[str, Any]) -> dict[str, Any]:
    """One user turn: user message, then the steps in order, then the reply."""
    items: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for m in trace.get("trace") or []:
        role = m.get("role")
        if role == "user":
            items.append({"type": "user", "text": m.get("text", "")})
        elif role == "assistant":
            items.append({"type": "assistant", "text": m.get("text", "")})
        elif role == "tool_call":
            pending = {"type": "tool", "name": m.get("name"), "arguments": m.get("arguments"), "result": None}
            items.append(pending)
        elif role == "tool_result":
            if pending is not None and pending["name"] == m.get("name") and pending["result"] is None:
                pending["result"] = m.get("content")
            else:
                items.append({"type": "tool", "name": m.get("name"), "arguments": None, "result": m.get("content")})
    last_assistant = max((i for i, it in enumerate(items) if it["type"] == "assistant"), default=None)
    tid = trace["trace_id"]
    for i, it in enumerate(items):
        it["ref"] = f"{tid}:{i}"
        if it["type"] == "assistant":
            it["final"] = i == last_assistant
    return {"trace_id": tid, "timestamp": trace.get("timestamp"), "items": items}


def _build_sessions(traces: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    final_ids = set(_scenarios())
    grouped: dict[str, list[dict[str, Any]]] = {}
    for t in traces:
        meta = t.get("meta") or {}
        sid = meta.get("scenario_id")
        if sid not in final_ids:  # the review set is the final scenarios, not the pilot
            continue
        grouped.setdefault(meta.get("session_id") or sid, []).append(t)
    sessions: dict[str, dict[str, Any]] = {}
    for key, ts in grouped.items():
        ts.sort(key=lambda t: t.get("timestamp") or "")
        first = ts[0]
        md = first.get("metadata") or {}
        sessions[key] = {
            "session_id": key,
            "scenario_id": first["meta"]["scenario_id"],
            "role": first["meta"].get("role"),
            "user_id": md.get("cartwheel.user_id"),
            "system_prompt": _system_prompt(first),
            "turns": [_turn(t) for t in ts],
        }
    return sessions


def _sessions(refresh: bool = False) -> dict[str, dict[str, Any]]:
    with _lock:
        if _cache["sessions"] is None or refresh:
            traces, source, reason = _fetch_traces()
            _cache.update(sessions=_build_sessions(traces), traces=traces, source=source, reason=reason)
        return _cache["sessions"]


# ---------------------------------------------------------------------------
# annotations
# ---------------------------------------------------------------------------


def _read_annotations() -> list[dict[str, Any]]:
    data = _read_json(ANNOTATIONS_FILE, {"annotations": []})
    if isinstance(data, dict):
        data = data.get("annotations", [])
    return [a for a in data if isinstance(a, dict)] if isinstance(data, list) else []


def _write_annotations(items: list[dict[str, Any]], deleted: list[dict[str, Any]] | None = None) -> None:
    """Keep the file's other keys (the 'deleted' archive) when the notes list is rewritten."""
    raw = _read_json(ANNOTATIONS_FILE, {})
    out = dict(raw) if isinstance(raw, dict) else {}
    out["annotations"] = items
    if deleted is not None:
        out["deleted"] = deleted
    _write_json(ANNOTATIONS_FILE, out)


def _active(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Notes the page works with. A superseded 'no failure observed' marker stays in the file as history."""
    return [a for a in items if not a.get("superseded_at")]


def _supersede_no_failure(items: list[dict[str, Any]], session_id: str, by_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for a in items:
        if a.get("session_id") == session_id and a.get("kind") == "no_failure" and not a.get("superseded_at"):
            a["superseded_at"], a["superseded_by"] = now, by_id


class Anchor(BaseModel):
    kind: str = "session"          # session | message | tool | text
    ref: str | None = None         # "<trace_id>:<item index>" for message, tool, text
    quote: str | None = None       # the selected text, for a text anchor
    start: int | None = None
    end: int | None = None


class NoteIn(BaseModel):
    session_id: str
    kind: str = "failure"          # failure | no_failure
    note: str = ""
    anchor: Anchor = Anchor()


class NoteEdit(BaseModel):
    note: str


def _status(annotations: list[dict[str, Any]], session_id: str) -> tuple[str, int]:
    mine = [a for a in _active(annotations) if a.get("session_id") == session_id]
    notes = [a for a in mine if a.get("kind", "failure") == "failure"]
    if notes:
        return "failure", len(notes)
    if any(a.get("kind") == "no_failure" for a in mine):
        return "no_failure", 0
    return "unreviewed", 0


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@router.get("/review", response_class=HTMLResponse)
def review_page() -> str:
    return (HERE / "index.html").read_text()


@router.get("/api/review/sessions")
def list_sessions(refresh: bool = False, order: str = "shuffled", include_excluded: bool = False) -> dict[str, Any]:
    sessions = _sessions(refresh)
    ann = _read_annotations()
    excluded = _excluded()
    batch_of = _batch_index()
    rows = []
    for s in sessions.values():
        if s["scenario_id"] in excluded and not include_excluded:
            continue
        status, n = _status(ann, s["session_id"])
        first_user = next((it["text"] for it in s["turns"][0]["items"] if it["type"] == "user"), "")
        rows.append({
            "session_id": s["session_id"], "scenario_id": s["scenario_id"], "role": s["role"],
            "preview": first_user[:110], "status": status, "notes": n,
            "batch": batch_of.get(s["session_id"], {}).get("batch"), "why": batch_of.get(s["session_id"], {}).get("reason"),
            "trace_ids": [t["trace_id"] for t in s["turns"]],
        })
    if order == "scenario":
        rows.sort(key=lambda r: r["scenario_id"])
    else:  # a fixed pseudo-random order, so the list does not start with one scenario group
        rows.sort(key=lambda r: hashlib.sha1(r["session_id"].encode()).hexdigest())
    return {"source": _cache["source"], "reason": _cache["reason"], "total": len(rows),
            "excluded": len([x for x in sessions.values() if x["scenario_id"] in excluded]),
            "batches": [{"name": b["name"], "count": len(b["picks"])} for b in _manifest().get("batches", [])], "sessions": rows}


@router.get("/api/review/excluded")
def list_excluded() -> dict[str, Any]:
    d = _read_json(EXCLUDED_FILE, {})
    return {"about": d.get("about"), "excluded": d.get("excluded", []), "borderline_kept": d.get("borderline_kept", [])}


@router.get("/api/review/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    s = _sessions().get(session_id)
    if s is None:
        raise HTTPException(404, f"unknown session: {session_id}")
    ann = [a for a in _active(_read_annotations()) if a.get("session_id") == session_id]
    return {**s, "annotations": ann}


@router.get("/api/review/scenario/{scenario_id}")
def scenario_info(scenario_id: str) -> dict[str, Any]:
    sc = _scenarios().get(scenario_id)
    if sc is None:
        raise HTTPException(404, f"unknown scenario: {scenario_id}")
    plan = {p["plan_id"]: p for p in _read_json(PLAN_FILE, {}).get("plans", [])}.get(scenario_id, {})
    return {"scenario_id": scenario_id, "group": sc["scenario_group"], "tuple": sc["tuple"],
            "data_quality_case_id": sc["data_quality_case_id"], "expected": sc["expected"],
            "record": plan.get("record")}


@router.get("/api/review/annotations")
def get_annotations() -> dict[str, Any]:
    return {"annotations": _read_annotations()}


@router.post("/api/review/annotations")
def add_annotation(body: NoteIn) -> dict[str, Any]:
    s = _sessions().get(body.session_id)
    if s is None:
        raise HTTPException(404, f"unknown session: {body.session_id}")
    if body.kind not in NOTE_KINDS:
        raise HTTPException(422, "kind must be failure or no_failure")
    if body.anchor.kind not in ANCHOR_KINDS:
        raise HTTPException(422, "anchor kind must be session, message, tool or text")
    refs = {it["ref"] for t in s["turns"] for it in t["items"]}
    if body.anchor.kind != "session" and body.anchor.ref not in refs:
        raise HTTPException(422, "the anchor does not point at anything in this conversation")
    if body.kind == "failure" and not body.note.strip():
        raise HTTPException(422, "Write a note describing the failure.")
    items = _read_annotations()
    mine = [a for a in _active(items) if a.get("session_id") == body.session_id]
    new_id = uuid.uuid4().hex[:12]
    if body.kind == "no_failure":
        if any(a.get("kind", "failure") == "failure" for a in mine):
            raise HTTPException(422, "This conversation already has failure notes. Delete them first to mark it as no failure observed.")
        anchor = Anchor()
    else:
        anchor = body.anchor
    # an earlier "no failure observed" marker is kept as history and marked superseded, never deleted
    _supersede_no_failure(items, body.session_id, new_id)
    trace_id = anchor.ref.split(":")[0] if anchor.kind != "session" and anchor.ref else s["turns"][-1]["trace_id"]
    rec = {
        "id": new_id, "session_id": body.session_id, "scenario_id": s["scenario_id"],
        "trace_id": trace_id, "kind": body.kind, "anchor": anchor.model_dump(), "note": body.note.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    items.append(rec)
    _write_annotations(items)
    status, n = _status(items, body.session_id)
    return {"ok": True, "annotation": rec, "status": status, "notes": n}


@router.put("/api/review/annotations/{annotation_id}")
def edit_annotation(annotation_id: str, body: NoteEdit) -> dict[str, Any]:
    items = _read_annotations()
    for a in items:
        if a.get("id") == annotation_id:
            if a.get("kind", "failure") == "failure" and not body.note.strip():
                raise HTTPException(422, "A failure note cannot be empty. Delete it instead.")
            a.setdefault("history", []).append({"note": a.get("note", ""), "at": a.get("edited_at") or a.get("created_at")})
            a["note"] = body.note.strip()
            a["edited_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _write_annotations(items)
            return {"ok": True, "annotation": a}
    raise HTTPException(404, f"unknown annotation: {annotation_id}")


@router.delete("/api/review/annotations/{annotation_id}")
def delete_annotation(annotation_id: str) -> dict[str, Any]:
    items = _read_annotations()
    gone = next((a for a in items if a.get("id") == annotation_id), None)
    if gone is None:
        raise HTTPException(404, f"unknown annotation: {annotation_id}")
    items = [a for a in items if a.get("id") != annotation_id]
    archive = _read_json(ANNOTATIONS_FILE, {}).get("deleted", [])
    archive.append({**gone, "deleted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    _write_annotations(items, deleted=archive)      # a deleted note moves to "deleted" in the same file
    status, n = _status(items, gone["session_id"])
    return {"ok": True, "status": status, "notes": n}


# ---------------------------------------------------------------------------
# The error-discovery skill's state API (same paths and files as the reference
# server), so the skill's review loop can run against this app:
#   GET/POST /api/samples      the traces chosen for review
#   GET      /api/annotations  your notes (the loop polls the file behind it)
#   GET      /api/graph        2D projection of all traces, for the Map view
#   GET/POST /api/patterns     the taxonomy the agent groups from your notes
#   GET/POST /api/suggestions  suggestions waiting for your accept or reject
# ---------------------------------------------------------------------------


def _sf(name: str) -> Path:
    return _state.state_path(name)


@router.get("/api/samples")
def get_samples() -> Any:
    return _read_json(_sf("samples.json"), [])


@router.post("/api/samples")
async def post_samples(request: Request) -> dict[str, Any]:
    data = await request.json()
    if not isinstance(data, list):
        raise HTTPException(400, "expected a JSON list of samples")
    _write_json(_sf("samples.json"), data)
    return {"ok": True, "count": len(data)}


@router.get("/api/annotations")
def annotations_for_loop() -> dict[str, Any]:
    return {"annotations": _read_annotations()}


@router.get("/api/patterns")
def get_patterns() -> Any:
    return _read_json(_sf("patterns.json"), {"modes": []})


@router.post("/api/patterns")
async def post_patterns(request: Request) -> dict[str, Any]:
    data = await request.json()
    if not isinstance(data, dict):
        raise HTTPException(400, "expected a JSON object")
    _write_json(_sf("patterns.json"), data)
    return {"ok": True, "count": len(data.get("modes", data))}


@router.get("/api/suggestions")
def get_suggestions() -> Any:
    return _read_json(_sf("suggestions.json"), [])


@router.post("/api/suggestions")
async def post_suggestions(request: Request) -> dict[str, Any]:
    data = await request.json()
    if not isinstance(data, list):
        raise HTTPException(400, "expected a JSON list of suggestions")
    _write_json(_sf("suggestions.json"), data)
    return {"ok": True, "count": len(data)}


class Decision(BaseModel):
    decision: str      # accept | reject


@router.post("/api/review/suggestions/{suggestion_id}/decision")
def decide_suggestion(suggestion_id: str, body: Decision) -> dict[str, Any]:
    """Explicit accept or reject. Accepting promotes the suggestion to one of your notes."""
    if body.decision not in ("accept", "reject"):
        raise HTTPException(422, "decision must be accept or reject")
    items = _read_json(_sf("suggestions.json"), [])
    s = next((x for x in items if str(x.get("id")) == suggestion_id), None)
    if s is None:
        raise HTTPException(404, f"unknown suggestion: {suggestion_id}")
    s["status"] = "accepted" if body.decision == "accept" else "rejected"
    s["decided_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if body.decision == "accept":
        sess = next((x for x in _sessions().values() if any(t["trace_id"] == s.get("trace_id") for t in x["turns"])), None)
        if sess is not None:
            ann = _read_annotations()
            new_id = uuid.uuid4().hex[:12]
            _supersede_no_failure(ann, sess["session_id"], new_id)
            ann.append({"id": new_id, "session_id": sess["session_id"], "scenario_id": sess["scenario_id"],
                        "trace_id": s["trace_id"], "kind": "failure", "source": "accepted_suggestion", "mode": s.get("mode"),
                        "anchor": {"kind": "session", "ref": None, "quote": s.get("quote"), "start": None, "end": None},
                        "note": s.get("note") or f"(accepted suggestion: {s.get('mode')}) {s.get('quote') or ''}".strip(),
                        "created_at": s["decided_at"]})
            _write_annotations(ann)
    _write_json(_sf("suggestions.json"), items)
    return {"ok": True, "status": s["status"]}


SEARCH_MODES = {   # depth-search label -> the candidate mode it looks for
    "POS policy": "platform_policy_not_store_policy",
    "POS lookup": "lookup_not_used_before_replying",
    "POS write": "write_action_beyond_request",
    "POS support caller": "wrong_human_handoff",
    "POS negative price": "negative_price_presented_as_valid",
}


def build_suggestions() -> int:
    """One pending suggestion per trace that a depth search retrieved as a possible positive. Not a label."""
    items = _read_json(_sf("suggestions.json"), [])
    have = {x.get("id") for x in items}
    n = 0
    for b in _manifest().get("batches", []):
        for p in b.get("picks", []):
            reason = p.get("reason", "")
            if not reason.startswith("depth search:"):
                continue
            for part in reason.removeprefix("depth search:").split(";"):
                part = part.strip()
                mode = next((m for k, m in SEARCH_MODES.items() if part.startswith(k)), None)
                sid = f"s-{p['scenario_id']}-{mode}"
                if mode and sid not in have:
                    items.append({"id": sid, "trace_id": p["trace_id"], "session_id": p["session_id"], "scenario_id": p["scenario_id"],
                                  "mode": mode, "search": part, "batch": b["name"], "status": "pending", "source": "depth_search",
                                  "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                    have.add(sid); n += 1
    _write_json(_sf("suggestions.json"), items)
    return n


class Answer(BaseModel):
    answer: str      # yes | no


@router.post("/api/review/suggestions/{suggestion_id}/answer")
def answer_suggestion(suggestion_id: str, body: Answer) -> dict[str, Any]:
    """Your yes/no on a retrieved trace. Records the decision only; it never writes a note for you."""
    if body.answer not in ("yes", "no"):
        raise HTTPException(422, "answer must be yes or no")
    items = _read_json(_sf("suggestions.json"), [])
    s = next((x for x in items if str(x.get("id")) == suggestion_id), None)
    if s is None:
        raise HTTPException(404, f"unknown suggestion: {suggestion_id}")
    s["status"] = "accepted" if body.answer == "yes" else "rejected"
    s["decided_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(_sf("suggestions.json"), items)
    return {"ok": True, "status": s["status"]}


# ---- samples, batches and the map ------------------------------------------


def _manifest() -> dict[str, Any]:
    m = _read_json(_sf("sample_manifest.json"), {})
    return m if isinstance(m, dict) else {}


def _batch_index() -> dict[str, dict[str, str]]:
    idx: dict[str, dict[str, str]] = {}
    for b in _manifest().get("batches", []):
        for p in b.get("picks", []):
            idx[p["session_id"]] = {"batch": b["name"], "reason": p["reason"]}
    return idx


def _pool(exclude_sessions: set[str]) -> list[dict[str, Any]]:
    """Traces that may be reviewed: final scenarios, not left out as unrealistic, not already in a batch."""
    final, left_out, out = set(_scenarios()), _excluded(), []
    for t in _cache["traces"]:
        meta = t.get("meta") or {}
        sid = meta.get("scenario_id")
        if sid not in final or sid in left_out:
            continue
        if (meta.get("session_id") or sid) in exclude_sessions:
            continue
        out.append(t)
    return out


class BatchIn(BaseModel):
    uniform: int = 15
    clusters: int = 15
    name: str | None = None


@router.post("/api/review/batch")
def create_batch(body: BatchIn) -> dict[str, Any]:
    """Batch 1 of Part B: uniformly sampled traces plus cluster representatives (skill Phase 4)."""
    _sessions()
    manifest = _manifest()
    batches = manifest.get("batches", [])
    name = body.name or f"batch{len(batches) + 1}"
    if any(b["name"] == name for b in batches):
        raise HTTPException(422, f"{name} already exists")
    used = {p["session_id"] for b in batches for p in b["picks"]}
    pool = _pool(used)
    seed = 7
    k = (body.clusters * 3 + 1) // 2                         # the helper returns two thirds as representatives
    reps = [p for p in selection.select(pool, k=k, strategy="diversity", exclude_ids=set(), seed=seed)
            if "representative" in p["reason"]][: body.clusters]
    rep_ids = {p["trace_id"] for p in reps}
    uniform = selection.select([t for t in pool if t["id"] not in rep_ids], k=body.uniform + 5, strategy="random", exclude_ids=set(), seed=seed)
    by_id = {t["id"]: t for t in pool}
    picks, seen_sessions = [], set()
    for want, group, label in ((body.clusters, reps, None), (body.uniform, uniform, "uniform random pick")):
        got = 0
        for p in group:
            t = by_id[p["trace_id"]]
            sess = (t["meta"].get("session_id") or t["meta"]["scenario_id"])
            if sess in seen_sessions or got >= want:
                continue
            seen_sessions.add(sess); got += 1
            picks.append({"trace_id": t["id"], "session_id": sess, "scenario_id": t["meta"]["scenario_id"], "reason": label or p["reason"]})
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    samples = _read_json(_sf("samples.json"), [])
    for p in picks:
        t = by_id[p["trace_id"]]
        samples.append({"trace_id": t["trace_id"], "reason": p["reason"], "batch": name, "trace": t["trace"], "text": t["text"],
                        "features": t["features"], "meta": t["meta"], "permalink": t.get("permalink"), "flags": []})
    batches.append({"name": name, "created_at": now, "picks": picks})
    manifest.update(source="langfuse (live)" if _cache["source"] == "langfuse" else f"export ({_cache['reason']})",
                    strategy=f"{name}: {body.uniform} uniform + {body.clusters} cluster representatives", seed=seed,
                    selected_at=now, batches=batches, picks=[p for b in batches for p in b["picks"]],
                    left_out_as_unrealistic=sorted(_excluded()))
    _write_json(_sf("samples.json"), samples)
    _write_json(_sf("sample_manifest.json"), manifest)
    _build_graph()
    return {"ok": True, "batch": name, "sessions": len(picks), "cluster_representatives": sum("representative" in p["reason"] for p in picks),
            "uniform": sum(p["reason"].startswith("uniform") for p in picks)}


class ScenarioBatchIn(BaseModel):
    name: str
    scenario_ids: list[str]
    dimension: str = "intent"
    reasons: dict[str, str] = {}      # optional per-scenario reason, e.g. which depth search found it


@router.post("/api/review/batch/scenarios")
def create_batch_from_scenarios(body: ScenarioBatchIn) -> dict[str, Any]:
    """Batch 2 of Part B: the user picked a product dimension and the scenarios spread across its values."""
    _sessions()
    manifest = _manifest()
    batches = manifest.get("batches", [])
    if any(b["name"] == body.name for b in batches):
        raise HTTPException(422, f"{body.name} already exists")
    used = {p["session_id"] for b in batches for p in b["picks"]}
    pool = _pool(used)
    by_scenario: dict[str, dict[str, Any]] = {}
    for t in pool:
        by_scenario.setdefault(t["meta"]["scenario_id"], t)
    missing = [s for s in body.scenario_ids if s not in by_scenario]
    if missing:
        raise HTTPException(422, f"not available (excluded, already in a batch, or no trace): {missing}")
    scen = _scenarios()
    picks = []
    for sid in body.scenario_ids:
        t = by_scenario[sid]
        value = (scen.get(sid, {}).get("tuple") or {}).get(body.dimension, "?")
        picks.append({"trace_id": t["id"], "session_id": t["meta"].get("session_id") or sid, "scenario_id": sid,
                      "reason": body.reasons.get(sid) or f"{body.dimension} = {value}"})
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    samples = _read_json(_sf("samples.json"), [])
    for p in picks:
        t = by_scenario[p["scenario_id"]]
        samples.append({"trace_id": t["trace_id"], "reason": p["reason"], "batch": body.name, "trace": t["trace"], "text": t["text"],
                        "features": t["features"], "meta": t["meta"], "permalink": t.get("permalink"), "flags": []})
    batches.append({"name": body.name, "created_at": now, "dimension": body.dimension, "picks": picks})
    manifest.update(batches=batches, picks=[p for b in batches for p in b["picks"]], selected_at=now,
                    strategy=f"{manifest.get('strategy', '')}; {body.name}: {len(picks)} across {body.dimension}")
    _write_json(_sf("samples.json"), samples)
    _write_json(_sf("sample_manifest.json"), manifest)
    _build_graph()
    return {"ok": True, "batch": body.name, "sessions": len(picks)}


# ---- labels (Part E): one present or absent decision per conversation and final mode ----------
# A label is 1 when the failure mode is present, 0 when absent (the helper convention). One line is appended
# to analysis/state/labels/<mode>.jsonl per decision; a change appends a new line that supersedes the old one,
# and nothing is overwritten. A conversation is labelled against its last trace's id.


def _final_modes() -> list[tuple[str, dict[str, Any]]]:
    pats = _read_json(_sf("patterns.json"), {})
    items = [(n, p) for n, p in pats.items() if isinstance(p, dict) and p.get("status") == "final_candidate"]
    return sorted(items, key=lambda x: -x[1].get("count", 0))


def _label_file(mode: str) -> Path:
    return _state.state_path("labels", f"{mode}.jsonl")


def _latest_labels(mode: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rec in _state.read_jsonl(_label_file(mode)):
        out[rec["trace_id"]] = rec          # lines are appended in order, so the last one for a trace wins
    return out


@router.get("/api/review/labels")
def get_labels() -> dict[str, Any]:
    sessions, ann, excluded, modes = _sessions(), _active(_read_annotations()), _excluded(), _final_modes()
    saved = {m: _latest_labels(m) for m, _ in modes}
    rows = []
    for s in sessions.values():
        sid, sc = s["session_id"], s["scenario_id"]
        mine = [a for a in ann if a.get("session_id") == sid]
        if not mine or sc in excluded:
            continue
        status, _n = _status(ann, sid)
        trace_id = s["turns"][-1]["trace_id"]
        cells = {}
        for m, p in modes:
            lab = saved[m].get(trace_id)
            proposed = 1 if sc in p.get("scenarios", []) else (0 if status == "no_failure" else None)
            cells[m] = {"label": lab["label"] if lab else None, "proposed": proposed}
        rows.append({"session_id": sid, "scenario_id": sc, "trace_id": trace_id, "status": status,
                     "note": " / ".join(a.get("note", "") for a in mine if a.get("kind") == "failure"), "cells": cells})
    rows.sort(key=lambda r: r["scenario_id"])
    counts = {m: {"present": sum(r["cells"][m]["label"] == 1 for r in rows), "absent": sum(r["cells"][m]["label"] == 0 for r in rows),
                  "undecided": sum(r["cells"][m]["label"] is None for r in rows)} for m, _ in modes}
    return {"modes": [{"name": m, "definition": p.get("definition", ""), "boundary": p.get("boundary", ""), "notes": p.get("count", 0)} for m, p in modes],
            "rows": rows, "counts": counts, "total": len(rows)}


class LabelIn(BaseModel):
    session_id: str
    mode: str
    label: int          # 1 = present, 0 = absent


@router.post("/api/review/labels")
def set_label(body: LabelIn) -> dict[str, Any]:
    if body.mode not in dict(_final_modes()):
        raise HTTPException(422, f"{body.mode} is not a final mode")
    if body.label not in (0, 1):
        raise HTTPException(422, "label must be 1 (present) or 0 (absent)")
    s = _sessions().get(body.session_id)
    if s is None:
        raise HTTPException(404, f"unknown session: {body.session_id}")
    trace_id = s["turns"][-1]["trace_id"]
    path = _label_file(body.mode)
    prior = [r for r in _state.read_jsonl(path) if r["trace_id"] == trace_id]
    if prior and prior[-1]["label"] == body.label:
        return {"ok": True, "unchanged": True}
    rec = {"trace_id": trace_id, "session_id": body.session_id, "scenario_id": s["scenario_id"], "mode": body.mode, "label": body.label,
           "source": "human", "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label_id": f"{trace_id}#{len(prior)}"}
    if prior:
        rec["supersedes"] = prior[-1]["label_id"]
    _state.append_jsonl(path, rec)
    return {"ok": True, "label_id": rec["label_id"]}


class SyncIn(BaseModel):
    dry_run: bool = True


@router.post("/api/review/labels/sync")
def sync_labels(body: SyncIn) -> dict[str, Any]:
    """Write the latest label per conversation and mode to Langfuse as scores named after the mode (0 or 1).
    A dry run (the default) only reports what would be written."""
    from analysis.helpers import langfuse_io

    synced = _read_json(_sf("label_sync.json"), {})
    todo = [(m, rec) for m, _ in _final_modes() for rec in _latest_labels(m).values() if f"{m}:{rec['label_id']}" not in synced]
    by_mode: dict[str, int] = {}
    for m, _rec in todo:
        by_mode[m] = by_mode.get(m, 0) + 1
    if body.dry_run:
        return {"dry_run": True, "to_write": len(todo), "by_mode": by_mode, "langfuse_configured": langfuse_io.is_configured()}
    if not langfuse_io.is_configured():
        raise HTTPException(503, "Langfuse is not configured (check the LANGFUSE_* settings in .env)")
    done = 0
    try:
        client = langfuse_io._client()
        for m in by_mode:
            langfuse_io.ensure_score_config(m, client=client)  # the score config is created only when a mode has labels to write
        for m, rec in todo:
            # a fixed score id per label means writing the same label again overwrites the score instead of duplicating it
            client.create_score(name=m, value=int(rec["label"]), trace_id=langfuse_io.logical_to_langfuse_id(rec["trace_id"], client),
                                data_type="NUMERIC", score_id=f"{m}-{rec['label_id']}".replace("#", "-"))
            synced[f"{m}:{rec['label_id']}"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            done += 1
        client.flush()
    except Exception as exc:       # show the real reason instead of a bare "Internal Server Error"
        raise HTTPException(502, f"Langfuse write failed after {done} scores: {type(exc).__name__}: {exc}") from exc
    finally:
        _write_json(_sf("label_sync.json"), synced)
    return {"dry_run": False, "written": done, "by_mode": by_mode}


def _build_graph() -> dict[str, Any]:
    """2D projection (PCA of the same standardized features the clustering uses) for the Map view."""
    import hashlib as _h

    import numpy as np

    traces = _pool(set())
    if not traces:
        return {"nodes": [], "clusters": []}
    X = np.array(selection._standardize([selection._feature_vector(t) for t in traces]), dtype=float)
    assign = selection._kmeans(X.tolist(), k=min(8, len(traces)), seed=7)
    Xc = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ vt[:2].T
    nodes = []
    for i, t in enumerate(traces):   # a small fixed jitter, so identical traces do not sit exactly on top of each other
        h = int(_h.sha1(t["id"].encode()).hexdigest()[:8], 16)
        jx, jy = ((h % 1000) / 1000 - 0.5) * 0.25, (((h // 1000) % 1000) / 1000 - 0.5) * 0.25
        meta = t.get("meta") or {}
        nodes.append({"trace_id": t["id"], "session_id": meta.get("session_id") or meta.get("scenario_id"),
                      "scenario_id": meta.get("scenario_id"), "x": float(proj[i, 0]) + jx, "y": float(proj[i, 1]) + jy, "cluster": int(assign[i])})
    g = {"nodes": nodes, "clusters": sorted(set(assign))}
    _write_json(_sf("graph.json"), g)
    return g


@router.get("/api/graph")
def get_graph() -> Any:
    g = _read_json(_sf("graph.json"), None)
    if g is None:
        _sessions()
        g = _build_graph()
    return g
