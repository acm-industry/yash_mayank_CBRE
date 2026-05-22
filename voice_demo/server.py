from __future__ import annotations

import difflib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from langgraph.types import Command
from pydantic import BaseModel

from cbre_agent.agent import _GRAPH
from voice_demo.conversation import decide_next_step, final_dispatch_response
from voice_demo.session_store import CallSession, SessionStore
from voice_demo.speech import (
    SpeechError,
    fetch_twilio_recording,
    mock_transcript,
    synthesize_elevenlabs,
    transcribe_deepgram,
)
from voice_demo.twiml import (
    incoming_call_prompt,
    play_audio_and_hangup,
    play_audio_then_record,
    reprompt_record,
    silent_record_only,
)
from voice_demo.workorder import persist_call_artifact


class Settings(BaseModel):
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "")
    demo_mode: bool = os.getenv("VOICE_DEMO_MODE", "false").lower() == "true"
    max_turns_per_call: int = int(os.getenv("VOICE_DEMO_MAX_TURNS", "7"))
    max_clarification_rounds: int = int(os.getenv("VOICE_DEMO_MAX_CLARIFICATION_ROUNDS", "2"))
    max_classify_attempts: int = int(os.getenv("VOICE_DEMO_MAX_CLASSIFY_ATTEMPTS", "2"))
    min_caller_turns_before_classify: int = int(os.getenv("VOICE_DEMO_MIN_CALLER_TURNS_BEFORE_CLASSIFY", "2"))
    min_agent_turns_before_classify: int = int(os.getenv("VOICE_DEMO_MIN_AGENT_TURNS_BEFORE_CLASSIFY", "1"))
    outputs_dir: Path = Path(os.getenv("VOICE_DEMO_OUTPUTS_DIR", "demo_outputs"))


SETTINGS = Settings()
SESSIONS = SessionStore()
app = FastAPI(title="CBRE Live Voice Demo")
LIVE_HITL_REVIEWS: Dict[str, Dict[str, Any]] = {}
_DONE_PATTERNS = ("that's all", "that is all", "that's it", "that is it", "no that's all")
_OPENING_PROMPT = (
    "Thank you for calling CBRE maintenance intake. "
    "I will ask a few quick questions, then submit your work order."
)
_ROOT = Path(__file__).resolve().parent.parent
_BUILDINGS_PATH = _ROOT / "operational" / "buildings.json"
_BUILDING_NAMES = [b.get("name", "").strip() for b in json.loads(_BUILDINGS_PATH.read_text()) if b.get("name")]
_BUILDING_NAMES_BY_LC = {name.lower(): name for name in _BUILDING_NAMES}


def _h(value: Any) -> str:
    raw = "" if value is None else str(value)
    return (
        raw.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _public_audio_url(filename: str) -> str:
    if SETTINGS.public_base_url:
        return f"{SETTINGS.public_base_url}/audio/{filename}"
    return f"/audio/{filename}"


def _save_tts_audio(audio_bytes: bytes) -> str:
    out_dir = SETTINGS.outputs_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.mp3"
    (out_dir / filename).write_bytes(audio_bytes)
    return filename


def _caller_done(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in _DONE_PATTERNS)


def _merge_slots_from_decision(session: CallSession, decision_slots: Dict[str, Optional[str]]) -> None:
    for key, value in decision_slots.items():
        if not value:
            continue
        if key == "building_name":
            value = _canonicalize_building_name(value) or value
        if not session.intake_slots.get(key):
            session.intake_slots[key] = value


def _essential_slots_present(session: CallSession) -> bool:
    return bool(
        session.intake_slots.get("issue")
        and session.intake_slots.get("building_name")
        and session.intake_slots.get("floor")
    )


def _context_signal_present(session: CallSession) -> bool:
    """At least one of scope / active_status / safety_signal was gathered."""
    return bool(
        session.intake_slots.get("impact_scope")
        or session.intake_slots.get("active_status")
        or session.intake_slots.get("safety_signal")
    )


def _agent_turn_count(session: CallSession) -> int:
    return sum(1 for turn in session.turns if turn.get("speaker") == "agent")


def _caller_opening_was_short(session: CallSession) -> bool:
    """True if the very first caller utterance was <= 10 words."""
    first_caller = next((t for t in session.turns if t.get("speaker") == "caller"), None)
    if not first_caller:
        return False
    return len((first_caller.get("text") or "").split()) <= 10


def _min_agent_turns_required(session: CallSession) -> int:
    """If the caller's opening was short, require an extra agent probe before classify."""
    base = SETTINGS.min_agent_turns_before_classify
    if _caller_opening_was_short(session):
        return max(base, 2)
    return base


def _has_min_dialogue_for_first_classify(session: CallSession) -> bool:
    return (
        session.turn_count >= SETTINGS.min_caller_turns_before_classify
        and _agent_turn_count(session) >= _min_agent_turns_required(session)
    )


def _ensure_question(text: str) -> str:
    clean = (text or "").strip()
    if not clean:
        return "Before I submit this, could you share one more detail?"
    if "?" in clean:
        return clean
    return f"{clean} Could you share one more detail?"


def _extract_floor(text: str) -> Optional[str]:
    patterns = [
        r"\b(ground floor)\b",
        r"\b(basement)\b",
        r"\b(floor\s*\d{1,2})\b",
        r"\b(level\s*\d{1,2})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).strip().title()
    return None


_GENERIC_BUILDING_WORDS = {
    "the", "a", "an", "at", "in", "on", "of",
    "center", "centre", "plaza", "park", "tower", "building", "hub", "campus",
    "north", "south", "east", "west", "professional", "medical", "logistics",
    "commerce", "tech", "office", "complex", "court", "place",
}


def _building_distinct_tokens(name: str) -> set[str]:
    return {tok.lower() for tok in name.split() if len(tok) > 3 and tok.lower() not in _GENERIC_BUILDING_WORDS}


_BUILDING_DISTINCT_TOKENS = {name: _building_distinct_tokens(name) for name in _BUILDING_NAMES}


def _extract_building_name(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()

    # 1) Direct substring match against the full canonical name.
    for name_lc, canonical in _BUILDING_NAMES_BY_LC.items():
        if name_lc in lowered:
            return canonical

    # 2) Distinct-token match: at least one DISTINCT (non-generic) token matches AND
    #    that token uniquely identifies a single building.
    candidate_buildings: list[str] = []
    for canonical, tokens in _BUILDING_DISTINCT_TOKENS.items():
        if tokens and any(tok in lowered for tok in tokens):
            candidate_buildings.append(canonical)
    if len(candidate_buildings) == 1:
        return candidate_buildings[0]

    # 3) Fuzzy match over multi-word phrases (high cutoff, prefer 3+ word phrases).
    words = lowered.split()
    best: Optional[Tuple[float, str]] = None
    for size in (5, 4, 3):
        if len(words) < size:
            continue
        for start in range(0, len(words) - size + 1):
            phrase = " ".join(words[start : start + size])
            close = difflib.get_close_matches(phrase, list(_BUILDING_NAMES_BY_LC.keys()), n=1, cutoff=0.7)
            if close:
                ratio = difflib.SequenceMatcher(None, phrase, close[0]).ratio()
                if best is None or ratio > best[0]:
                    best = (ratio, _BUILDING_NAMES_BY_LC[close[0]])
    return best[1] if best else None


def _canonicalize_building_name(value: Optional[str]) -> Optional[str]:
    """Map an LLM- or STT-provided building name to a canonical name when close enough."""
    if not value:
        return None
    if value in _BUILDING_NAMES:
        return value
    lowered = value.lower().strip()
    if lowered in _BUILDING_NAMES_BY_LC:
        return _BUILDING_NAMES_BY_LC[lowered]
    # Try substring match against canonical names.
    for name_lc, canonical in _BUILDING_NAMES_BY_LC.items():
        if name_lc in lowered or lowered in name_lc:
            return canonical
    # Distinct-token uniqueness check.
    candidates = [
        canonical for canonical, tokens in _BUILDING_DISTINCT_TOKENS.items()
        if tokens and any(tok in lowered for tok in tokens)
    ]
    if len(candidates) == 1:
        return candidates[0]
    # Fuzzy match against canonical list (high cutoff to avoid false positives).
    close = difflib.get_close_matches(lowered, list(_BUILDING_NAMES_BY_LC.keys()), n=1, cutoff=0.75)
    if close:
        return _BUILDING_NAMES_BY_LC[close[0]]
    return value


def _update_slots_from_caller_text(session: CallSession, text: str) -> None:
    clean = text.strip()
    if not clean:
        return
    if not session.intake_slots.get("issue") and len(clean) >= 8:
        session.intake_slots["issue"] = clean
    if not session.intake_slots.get("building_name"):
        session.intake_slots["building_name"] = _extract_building_name(clean)
    if not session.intake_slots.get("floor"):
        session.intake_slots["floor"] = _extract_floor(clean)
    if not session.intake_slots.get("urgency"):
        lowered = clean.lower()
        if any(term in lowered for term in ("fire", "smoke", "gas", "threat", "trapped", "injury", "flood")):
            session.intake_slots["urgency"] = "emergency"
        elif any(term in lowered for term in ("urgent", "asap", "immediately")):
            session.intake_slots["urgency"] = "urgent"
        else:
            session.intake_slots["urgency"] = "normal"


def _can_attempt_classify(session: CallSession) -> bool:
    """Silent guard: returns True only when it is safe to actually invoke classify().

    Requires: min dialogue, all 3 essential slots, AND at least one context signal
    (scope, active_status, or safety_signal). This forces the intake LLM to
    probe for severity/impact before we ever call classify().
    """
    if not _has_min_dialogue_for_first_classify(session):
        return False
    if not _essential_slots_present(session):
        return False
    if not _context_signal_present(session):
        return False
    return True


def _build_clarification_question(prediction: Dict[str, object]) -> str:
    subcategory = (prediction.get("subcategory") or "").lower()
    if subcategory in {"lighting", "parking_lighting", "power_outage"}:
        return "Before I submit this, can you confirm how large the affected area is and whether any critical area is dark?"
    if subcategory in {"restroom_fixture", "pipe_leak", "drainage_backup"}:
        return "Before I submit this, can you confirm if water is overflowing and whether anyone is at immediate risk?"
    if not prediction.get("building_name"):
        return "Before I submit this, what is the exact building name?"
    if not prediction.get("floor"):
        return "Before I submit this, what floor or area is impacted?"
    return "Before I submit this, could you clarify the exact symptom and impact area?"


def _build_graph_initial_state(turns: list[dict], caller_phone: str | None) -> dict:
    return {
        "turns": turns,
        "caller_phone": caller_phone,
        "transcript_text": "",
        "rag_query": "",
        "rag_rewrite_count": 0,
        "caller_profile": None,
        "retrieved_records": [],
        "rag_top_score": None,
        "classification": None,
        "building_info": None,
        "vendor_id": None,
        "dispatched_emergency": False,
        "human_override": None,
        "hitl_decision": None,
        "routing_decision": None,
        "final_decision": None,
        "rag_documents_relevant": None,
        "node_timings": [],
    }


def _prediction_from_graph_final(final_values: dict) -> dict:
    final_decision = dict(final_values.get("final_decision") or {})
    classification = dict(final_values.get("classification") or {})
    return {
        "category": final_decision.get("category", classification.get("category", "")),
        "subcategory": final_decision.get("subcategory", classification.get("subcategory", "")),
        "risk_level": final_decision.get("risk_level", classification.get("risk_level", "LOW")),
        "needs_human_review": bool(classification.get("needs_human_review", False)),
        "needs_clarification": bool(classification.get("needs_clarification", False)),
        "building_name": final_decision.get("building_name", classification.get("building_name")),
        "address": final_decision.get("address", classification.get("address")),
        "floor": final_decision.get("floor", classification.get("floor")),
        "dispatched_vendor_id": final_decision.get("dispatched_vendor_id", final_values.get("vendor_id")),
        "dispatched_emergency_services": bool(
            final_decision.get("dispatched_emergency_services", final_values.get("dispatched_emergency", False))
        ),
        "call_summary": classification.get("call_summary", ""),
    }


def _run_graph_with_hitl(session: CallSession, call_sid: str) -> Tuple[str, Dict[str, Any]]:
    thread_id = f"voice-{call_sid}-{uuid.uuid4().hex[:8]}"
    session.active_graph_thread_id = thread_id
    config = {"configurable": {"thread_id": thread_id}}

    _GRAPH.invoke(_build_graph_initial_state(session.turns, session.caller_phone), config)
    paused_state = _GRAPH.get_state(config)
    pre_review = dict(paused_state.values.get("classification") or {})

    if paused_state.next:
        review_id = uuid.uuid4().hex
        LIVE_HITL_REVIEWS[review_id] = {
            "status": "pending",
            "review_id": review_id,
            "call_sid": call_sid,
            "thread_id": thread_id,
            "config": config,
            "pre_review": pre_review,
            "transcript_text": paused_state.values.get("transcript_text", ""),
            "classification": pre_review,
            "created_at": time.time(),
        }
        session.pending_hitl_review_id = review_id
        return "pending_review", {"review_id": review_id, "pre_review": pre_review}

    final_values = _GRAPH.get_state(config).values
    return "final", {
        "final_values": final_values,
        "prediction": _prediction_from_graph_final(final_values),
    }


def _speak(text: str, continue_recording: bool) -> Tuple[str, float]:
    if SETTINGS.demo_mode:
        if continue_recording:
            return reprompt_record(text), 0.0
        return play_audio_and_hangup(audio_url=None, outro_text=text), 0.0

    t0 = time.perf_counter()
    audio_bytes = synthesize_elevenlabs(
        text=text,
        elevenlabs_api_key=SETTINGS.elevenlabs_api_key,
        elevenlabs_voice_id=SETTINGS.elevenlabs_voice_id,
    )
    elapsed = time.perf_counter() - t0
    filename = _save_tts_audio(audio_bytes)
    audio_url = _public_audio_url(filename)
    if continue_recording:
        return (
            play_audio_then_record(
                audio_url=audio_url,
                follow_up_text=None,
            ),
            elapsed,
        )
    return (
        play_audio_and_hangup(
            audio_url=audio_url,
            outro_text=None,
        ),
        elapsed,
    )


def _reprompt_with_voice(message: str) -> str:
    if SETTINGS.demo_mode:
        return reprompt_record(message)
    try:
        xml, _ = _speak(message, continue_recording=True)
        return xml
    except SpeechError:
        # Last-resort fallback if ElevenLabs itself fails: keep recording without Twilio voice.
        return silent_record_only()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/audio/{filename}")
def serve_audio(filename: str) -> Response:
    path = SETTINGS.outputs_dir / "audio" / filename
    if not path.exists():
        return PlainTextResponse("Audio file not found.", status_code=404)
    return FileResponse(path, media_type="audio/mpeg")


_LIVE_HITL_CSS = """
:root {
  --bg: #f4f6fb;
  --card: #ffffff;
  --text: #1a1f36;
  --muted: #65708a;
  --primary: #2f6fed;
  --primary-dark: #2154b6;
  --ok: #12805c;
  --warn: #b4690e;
  --danger: #b53b3b;
  --border: #d9deea;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.container { max-width: 1100px; margin: 0 auto; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }
.title { margin: 0; font-size: 30px; }
.subtitle { margin: 6px 0 0; color: var(--muted); font-size: 14px; }
.pill { padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; background: #e6efff; color: #2349aa; }
.pill.live { background: #ffeaea; color: #b53b3b; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(19, 24, 38, 0.05); }
.card h2, .card h3 { margin-top: 0; }
.meta { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.meta .item { background: #f8faff; border: 1px solid #e4eaf7; border-radius: 8px; padding: 10px; }
.meta .label { font-size: 12px; color: var(--muted); margin-bottom: 3px; display: block; }
.meta .value { font-size: 14px; font-weight: 600; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.btn { border: 0; border-radius: 8px; padding: 10px 14px; font-size: 14px; font-weight: 700; cursor: pointer; background: var(--primary); color: white; text-decoration: none; display: inline-block; }
.btn:hover { background: var(--primary-dark); }
.btn-secondary { background: #f2f6ff; color: #22479e; border: 1px solid #cdd9fb; }
label { font-size: 12px; color: var(--muted); font-weight: 600; display: block; margin-bottom: 6px; }
input { width: 100%; padding: 10px; border: 1px solid var(--border); border-radius: 8px; font-size: 14px; }
.status { padding: 10px 12px; border-radius: 8px; background: #ecfff7; color: var(--ok); border: 1px solid #cdeede; margin-bottom: 12px; }
.status.warn { background: #fff8eb; color: var(--warn); border-color: #f2deb8; }
.status.live { background: #ffeaea; color: var(--danger); border-color: #f3c8c8; }
pre { margin: 0; white-space: pre-wrap; background: #f9fbff; border: 1px solid #e4eaf7; border-radius: 8px; padding: 12px; max-height: 420px; overflow: auto; font-size: 13px; line-height: 1.45; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 14px; }
th { background: #f8faff; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
.radio-row { display: flex; gap: 18px; margin: 4px 0 12px; }
.radio-row label { font-size: 14px; color: var(--text); font-weight: 600; display: flex; align-items: center; gap: 6px; }
.footer-link { color: #3559b7; font-weight: 600; text-decoration: none; }
.footer-link:hover { text-decoration: underline; }
.empty { padding: 22px; text-align: center; color: var(--muted); }
@media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } .meta { grid-template-columns: 1fr; } }
"""


def _live_page(title: str, body: str, refresh_seconds: int | None = None) -> str:
    meta_refresh = f'<meta http-equiv="refresh" content="{refresh_seconds}" />' if refresh_seconds else ""
    return f"""
    <html>
    <head>
      <title>{_h(title)}</title>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      {meta_refresh}
      <style>{_LIVE_HITL_CSS}</style>
    </head>
    <body>
      <div class="container">{body}</div>
    </body>
    </html>
    """


@app.get("/hitl/live", response_class=HTMLResponse)
def live_hitl_dashboard() -> str:
    pending = [r for r in LIVE_HITL_REVIEWS.values() if r.get("status") == "pending"]
    resolved = [r for r in LIVE_HITL_REVIEWS.values() if r.get("status") == "resolved"][-10:]

    if pending:
        pending_rows = "".join(
            f"<tr><td><code>{_h(r.get('review_id', '')[:10])}</code></td>"
            f"<td><code>{_h(r.get('call_sid', ''))}</code></td>"
            f"<td>{_h((r.get('classification') or {}).get('category', '—'))} / "
            f"{_h((r.get('classification') or {}).get('subcategory', '—'))}</td>"
            f"<td>{_h((r.get('classification') or {}).get('risk_level', '—'))}</td>"
            f"<td><a class='btn' href='/hitl/live/{_h(r.get('review_id', ''))}'>Open Review</a></td></tr>"
            for r in pending
        )
        pending_html = f"<table>{pending_rows and '<thead><tr><th>Review</th><th>Call SID</th><th>Predicted</th><th>Risk</th><th>Action</th></tr></thead>'}<tbody>{pending_rows}</tbody></table>"
    else:
        pending_html = "<div class='empty'>No pending live HITL reviews right now. New reviews will appear here automatically.</div>"

    if resolved:
        resolved_rows = "".join(
            f"<tr><td><code>{_h(r.get('review_id', '')[:10])}</code></td>"
            f"<td><code>{_h(r.get('call_sid', ''))}</code></td>"
            f"<td>{_h((r.get('final_values') or {}).get('routing_decision', '—'))}</td></tr>"
            for r in resolved
        )
        resolved_html = f"<table><thead><tr><th>Review</th><th>Call SID</th><th>Routing</th></tr></thead><tbody>{resolved_rows}</tbody></table>"
    else:
        resolved_html = "<div class='empty'>No resolved reviews yet.</div>"

    body = f"""
      <div class="header">
        <div>
          <h1 class="title">Live Call HITL Queue</h1>
          <p class="subtitle">Approve or override agent classifications interrupted mid-call. Caller is on hold until you submit.</p>
        </div>
        <span class="pill live">Live · auto-refresh 4s</span>
      </div>
      <div class="card">
        <h2>Pending Reviews ({len(pending)})</h2>
        {pending_html}
      </div>
      <div class="card">
        <h3>Recently Resolved</h3>
        {resolved_html}
      </div>
    """
    return _live_page("Live HITL Queue", body, refresh_seconds=4)


@app.get("/hitl/live/{review_id}", response_class=HTMLResponse)
def live_hitl_review(review_id: str) -> str:
    ctx = LIVE_HITL_REVIEWS.get(review_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Review not found.")
    if ctx.get("status") != "pending":
        body = (
            "<div class='card'><h2>Review already resolved.</h2>"
            "<p><a class='footer-link' href='/hitl/live'>← Back to live queue</a></p></div>"
        )
        return _live_page("Live HITL Review", body)

    prediction = ctx.get("classification") or {}
    transcript_text = ctx.get("transcript_text") or ""
    body = f"""
      <div class="header">
        <div>
          <h1 class="title">Live Call HITL Review</h1>
          <p class="subtitle">Caller is on hold. Confirm the model prediction or override before dispatch.</p>
        </div>
        <span class="pill live">Review {_h(review_id[:8])}</span>
      </div>
      <div class="status live">A live caller is waiting for your decision. Please review and submit.</div>
      <div class="card">
        <div class="meta">
          <div class="item"><span class="label">Call SID</span><span class="value">{_h(ctx.get("call_sid"))}</span></div>
          <div class="item"><span class="label">Predicted Subcategory</span><span class="value">{_h(prediction.get("subcategory", "—"))}</span></div>
          <div class="item"><span class="label">Predicted Category</span><span class="value">{_h(prediction.get("category", "—"))}</span></div>
          <div class="item"><span class="label">Risk Level</span><span class="value">{_h(prediction.get("risk_level", "—"))}</span></div>
          <div class="item"><span class="label">Building</span><span class="value">{_h(prediction.get("building_name", "—"))}</span></div>
          <div class="item"><span class="label">Floor</span><span class="value">{_h(prediction.get("floor", "—"))}</span></div>
        </div>
      </div>
      <div class="grid-2">
        <div class="card">
          <h3>Pre-Review Prediction</h3>
          <pre>{_h(json.dumps(prediction, indent=2))}</pre>
        </div>
        <div class="card">
          <h3>Conversation Transcript</h3>
          <pre>{_h(transcript_text)}</pre>
        </div>
      </div>
      <div class="card">
        <h3>Decision</h3>
        <form method="post" action="/hitl/live/{_h(review_id)}">
          <div class="radio-row">
            <label><input type="radio" name="decision" value="approve" checked/> Approve as predicted</label>
            <label><input type="radio" name="decision" value="override"/> Override fields</label>
          </div>
          <p class="subtitle" style="margin-top:0;">Fill any override fields below only if you chose Override. Leave blank to keep predicted values.</p>
          <div class="meta">
            <div><label>Override Subcategory</label><input type="text" name="override_subcategory" placeholder="e.g. pipe_leak" /></div>
            <div><label>Override Category</label><input type="text" name="override_category" placeholder="e.g. plumbing" /></div>
            <div><label>Override Risk Level</label><input type="text" name="override_risk_level" placeholder="LOW | MEDIUM | HIGH" /></div>
            <div><label>Override Building Name</label><input type="text" name="override_building_name" placeholder="Building name" /></div>
            <div><label>Override Address</label><input type="text" name="override_address" placeholder="Street address" /></div>
            <div><label>Override Floor</label><input type="text" name="override_floor" placeholder="e.g. Floor 3" /></div>
          </div>
          <div style="margin-top:12px;"><label>Override Reason (optional)</label><input type="text" name="override_reason" placeholder="Short reason for the override" /></div>
          <div style="margin-top:14px; display:flex; gap:10px;">
            <button class="btn" type="submit">Submit Decision</button>
            <a class="btn btn-secondary" href="/hitl/live">Back to queue</a>
          </div>
        </form>
      </div>
    """
    return _live_page("Live HITL Review", body)


@app.post("/hitl/live/{review_id}", response_class=HTMLResponse)
def submit_live_hitl_review(
    review_id: str,
    decision: str = Form(...),
    override_subcategory: str = Form(default=""),
    override_category: str = Form(default=""),
    override_risk_level: str = Form(default=""),
    override_building_name: str = Form(default=""),
    override_address: str = Form(default=""),
    override_floor: str = Form(default=""),
    override_reason: str = Form(default=""),
) -> str:
    ctx = LIVE_HITL_REVIEWS.get(review_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Review not found.")
    if ctx.get("status") != "pending":
        return "<html><body><p>Review already resolved.</p><a href='/hitl/live'>Back</a></body></html>"

    if decision.lower() == "approve":
        resume_payload: Dict[str, Any] = {"approved": True}
    else:
        resume_payload = {"approved": False}
        if override_subcategory.strip():
            resume_payload["override_subcategory"] = override_subcategory.strip()
        if override_category.strip():
            resume_payload["override_category"] = override_category.strip()
        if override_risk_level.strip():
            resume_payload["override_risk_level"] = override_risk_level.strip()
        if override_building_name.strip():
            resume_payload["override_building_name"] = override_building_name.strip()
        if override_address.strip():
            resume_payload["override_address"] = override_address.strip()
        if override_floor.strip():
            resume_payload["override_floor"] = override_floor.strip()
        if override_reason.strip():
            resume_payload["override_reason"] = override_reason.strip()
        if len(resume_payload) == 1:
            resume_payload = {"approved": True}

    _GRAPH.invoke(Command(resume=resume_payload), ctx["config"])
    final_values = _GRAPH.get_state(ctx["config"]).values
    ctx["status"] = "resolved"
    ctx["final_values"] = final_values
    routing = (final_values or {}).get("routing_decision") or "dispatched"
    body = f"""
      <div class="header">
        <div>
          <h1 class="title">Decision Submitted</h1>
          <p class="subtitle">The caller will hear the final outcome on their next turn.</p>
        </div>
        <span class="pill">Resolved</span>
      </div>
      <div class="status">Routing decision recorded: <b>{_h(routing)}</b></div>
      <div class="card">
        <p><a class="btn" href="/hitl/live">Back to live queue</a></p>
      </div>
    """
    return _live_page("Decision Submitted", body)


@app.post("/voice/incoming")
async def voice_incoming() -> Response:
    if SETTINGS.demo_mode:
        return Response(content=incoming_call_prompt(), media_type="application/xml")
    xml, _ = _speak(_OPENING_PROMPT, continue_recording=True)
    return Response(content=xml, media_type="application/xml")


@app.post("/voice/process")
async def voice_process(
    CallSid: str = Form(...),
    From: Optional[str] = Form(default=None),
    SpeechResult: Optional[str] = Form(default=None),
    Confidence: Optional[str] = Form(default=None),
    RecordingUrl: Optional[str] = Form(default=None),
    RecordingDuration: Optional[str] = Form(default=None),
) -> Response:
    started = time.perf_counter()
    session = SESSIONS.get_or_create(CallSid, From)

    try:
        transcript = ""
        stt_seconds = 0.0

        if SETTINGS.demo_mode:
            transcript = mock_transcript()
        elif SpeechResult is not None and SpeechResult.strip():
            # Twilio Gather already did STT for us — zero extra round-trip.
            transcript = SpeechResult.strip()
        elif RecordingUrl:
            # Legacy <Record> fallback: download from Twilio and run Deepgram.
            stt_t0 = time.perf_counter()
            audio_bytes = fetch_twilio_recording(
                recording_url=RecordingUrl,
                twilio_account_sid=SETTINGS.twilio_account_sid,
                twilio_auth_token=SETTINGS.twilio_auth_token,
            )
            transcript = transcribe_deepgram(audio_bytes, SETTINGS.deepgram_api_key)
            stt_seconds = time.perf_counter() - stt_t0

        # If a HITL review is pending for this call, check that path FIRST so the
        # caller's silence on hold does not trigger a re-prompt or extra logic.
        if session.pending_hitl_review_id:
            review_ctx = LIVE_HITL_REVIEWS.get(session.pending_hitl_review_id)
            if review_ctx and review_ctx.get("status") == "resolved":
                prediction = _prediction_from_graph_final(review_ctx.get("final_values") or {})
                session.pending_hitl_review_id = None
                session.pending_clarification = False
                session.clarification_focus = None
                session.clarification_prompt_caller_turn = 0
                session.hitl_hold_poll_count = 0
                response_text = final_dispatch_response(prediction, session.turns)
                session.append_turn("agent", response_text)
                xml, tts_seconds = _speak(response_text, continue_recording=False)
                timings = {
                    "stt_seconds": round(stt_seconds, 4),
                    "classify_seconds": 0.0,
                    "tts_seconds": round(tts_seconds, 4),
                    "request_total_seconds": round(time.perf_counter() - started, 4),
                    "phase": "hitl_resolved_finalize",
                    "intake_slots": dict(session.intake_slots),
                }
                persist_call_artifact(
                    out_dir=SETTINGS.outputs_dir / "workorders",
                    call_sid=CallSid,
                    caller_phone=From,
                    turns=session.turns,
                    prediction=prediction,
                    timings=timings,
                )
                SESSIONS.end(CallSid)
                return Response(content=xml, media_type="application/xml")

            # Still pending — keep the call alive. To avoid repeating the hold
            # message every ~5s (Gather poll cadence), only speak periodically.
            session.hitl_hold_poll_count += 1
            speak_reminder = session.hitl_hold_poll_count % 4 == 0  # ~every 20s
            tts_seconds = 0.0
            if speak_reminder:
                hold_text = "Still verifying with the supervisor. Please stay on the line — almost there."
                session.append_turn("agent", hold_text)
                xml, tts_seconds = _speak(hold_text, continue_recording=True)
            else:
                # Silent listen — keeps Twilio Gather active without re-speaking.
                xml = silent_record_only()
            timings = {
                "stt_seconds": round(stt_seconds, 4),
                "classify_seconds": 0.0,
                "tts_seconds": round(tts_seconds, 4),
                "request_total_seconds": round(time.perf_counter() - started, 4),
                "phase": "hitl_pending_wait",
                "hold_poll_count": session.hitl_hold_poll_count,
                "spoke_reminder": speak_reminder,
            }
            print(f"Call {CallSid} waiting on live HITL review timings={timings}")
            return Response(content=xml, media_type="application/xml")

        if not transcript.strip():
            # Empty speech result from Gather (silence) or no audio at all.
            prompt = "I did not catch that. Could you say that one more time?"
            return Response(
                content=_reprompt_with_voice(prompt),
                media_type="application/xml",
            )

        session.append_turn("caller", transcript)
        _update_slots_from_caller_text(session, transcript)

        reached_turn_cap = session.turn_count >= SETTINGS.max_turns_per_call

        # Fast path: if we already have everything we need AND enough dialogue,
        # skip the extra LLM intake call and go straight to classification.
        if _can_attempt_classify(session):
            pass
        else:
            decision = decide_next_step(
                turns=session.turns,
                intake_slots=session.intake_slots,
                turn_count=session.turn_count,
                max_turns=SETTINGS.max_turns_per_call,
                pending_clarification=session.pending_clarification,
                clarification_focus=session.clarification_focus,
                known_building_names=_BUILDING_NAMES,
                building_ask_count=session.building_ask_count,
            )
            _merge_slots_from_decision(
                session,
                {
                    "issue": decision.captured_issue,
                    "building_name": decision.captured_building_name,
                    "floor": decision.captured_floor,
                    "urgency": decision.captured_urgency,
                    "impact_scope": decision.captured_impact_scope,
                    "active_status": decision.captured_active_status,
                    "safety_signal": decision.captured_safety_signal,
                },
            )

            # Track how often the LLM has asked about the building so it can adapt.
            llm_text_lower = (decision.agent_response or "").lower()
            if "building" in llm_text_lower or "property" in llm_text_lower or "location" in llm_text_lower:
                session.building_ask_count += 1

            # SAFETY OVERRIDE: if the LLM tries to finalize but essential location slots are
            # missing, force a question for the missing slot instead. The LLM should not be
            # finalizing without building/floor.
            missing_building = not session.intake_slots.get("building_name")
            missing_floor = not session.intake_slots.get("floor")
            overridden_question: Optional[str] = None
            if decision.should_finalize and (missing_building or missing_floor):
                if missing_building and missing_floor:
                    overridden_question = "Got it. Which building and floor are you calling from?"
                elif missing_building:
                    overridden_question = "Got it. Which building are you calling from?"
                else:
                    overridden_question = "Got it. What floor or area is impacted?"

            # If we are near the turn cap and location is still missing, force the ask now
            # regardless of what the LLM wanted to say next.
            near_cap_force_location = (
                not reached_turn_cap
                and session.turn_count >= (SETTINGS.max_turns_per_call - 2)
                and (missing_building or missing_floor)
            )
            if near_cap_force_location and overridden_question is None:
                if missing_building and missing_floor:
                    overridden_question = "And just so I can route this, which building and floor are you calling from?"
                elif missing_building:
                    overridden_question = "And which building are you calling from?"
                else:
                    overridden_question = "And what floor or area is impacted?"

            # Allow finalize only when:
            #  - the LLM says so AND we have a real context signal AND building+floor are known, OR
            #  - we hit turn cap, OR caller indicates they are done, OR we are stuck on building.
            llm_ready_to_finalize = bool(
                decision.should_finalize
                and _essential_slots_present(session)
                and _context_signal_present(session)
            )
            can_finalize = (
                llm_ready_to_finalize
                or reached_turn_cap
                or _caller_done(transcript)
                or (session.building_ask_count >= 3 and session.intake_slots.get("issue"))
            )
            if not can_finalize and not reached_turn_cap:
                question = overridden_question or _ensure_question(decision.agent_response)
                session.append_turn("agent", question)
                xml, tts_seconds = _speak(question, continue_recording=True)
                timings = {
                    "stt_seconds": round(stt_seconds, 4),
                    "classify_seconds": 0.0,
                    "tts_seconds": round(tts_seconds, 4),
                    "request_total_seconds": round(time.perf_counter() - started, 4),
                    "phase": "llm_intake_turn",
                    "overridden": overridden_question is not None,
                }
                print(
                    f"Call {CallSid} intake_slots={session.intake_slots} timings={timings} "
                    f"llm_finalize={decision.should_finalize} pending_clarification={session.pending_clarification} "
                    f"building_ask_count={session.building_ask_count} missing_building={missing_building} "
                    f"missing_floor={missing_floor}"
                )
                return Response(content=xml, media_type="application/xml")

        # Important: classify only on pre-final conversation turns.
        turns_for_classification = list(session.turns)
        clf_t0 = time.perf_counter()
        run_status, run_payload = _run_graph_with_hitl(session, CallSid)
        classify_seconds = time.perf_counter() - clf_t0
        session.classify_attempts += 1

        if run_status == "pending_review":
            review_id = run_payload["review_id"]
            hold_text = (
                "Thanks for those details. Please hold for just a moment while a "
                "human supervisor verifies this before we dispatch."
            )
            session.append_turn("agent", hold_text)
            xml, tts_seconds = _speak(hold_text, continue_recording=True)
            timings = {
                "stt_seconds": round(stt_seconds, 4),
                "classify_seconds": round(classify_seconds, 4),
                "tts_seconds": round(tts_seconds, 4),
                "request_total_seconds": round(time.perf_counter() - started, 4),
                "phase": "hitl_interrupt_wait",
                "review_id": review_id,
            }
            print(f"Call {CallSid} hitl_interrupt review_id={review_id} timings={timings}")
            return Response(content=xml, media_type="application/xml")

        prediction = run_payload["prediction"]

        if (
            prediction.get("needs_clarification")
            and session.turn_count < SETTINGS.max_turns_per_call
            and session.clarification_rounds < SETTINGS.max_clarification_rounds
            and session.classify_attempts < SETTINGS.max_classify_attempts
        ):
            question = _build_clarification_question(prediction)
            session.clarification_rounds += 1
            session.pending_clarification = True
            session.clarification_focus = question
            session.clarification_prompt_caller_turn = session.turn_count
            session.append_turn("agent", question)
            xml, tts_seconds = _speak(question, continue_recording=True)
            timings = {
                "stt_seconds": round(stt_seconds, 4),
                "classify_seconds": round(classify_seconds, 4),
                "tts_seconds": round(tts_seconds, 4),
                "request_total_seconds": round(time.perf_counter() - started, 4),
                "phase": "clarification_loop",
                "clarification_round": session.clarification_rounds,
            }
            print(f"Call {CallSid} clarification_loop timings={timings}")
            return Response(content=xml, media_type="application/xml")

        session.pending_clarification = False
        session.clarification_focus = None
        session.clarification_prompt_caller_turn = 0
        response_text = final_dispatch_response(prediction, turns_for_classification)
        session.append_turn("agent", response_text)
        xml, tts_seconds = _speak(response_text, continue_recording=False)

        timings = {
            "stt_seconds": round(stt_seconds, 4),
            "classify_seconds": round(classify_seconds, 4),
            "tts_seconds": round(tts_seconds, 4),
            "request_total_seconds": round(time.perf_counter() - started, 4),
            "phase": "finalize",
            "intake_slots": dict(session.intake_slots),
        }
        persist_call_artifact(
            out_dir=SETTINGS.outputs_dir / "workorders",
            call_sid=CallSid,
            caller_phone=From,
            turns=session.turns,
            prediction=prediction,
            timings=timings,
        )
        SESSIONS.end(CallSid)
        return Response(content=xml, media_type="application/xml")
    except SpeechError as exc:
        print(f"SpeechError in call {CallSid}: {exc}")
        xml = _reprompt_with_voice("Audio processing failed. Please try once more.")
        return Response(content=xml, media_type="application/xml")
    except Exception as exc:  # noqa: BLE001 - keep webhook resilient in demo mode.
        print(f"Unhandled /voice/process error: {exc}")
        return Response(
            content=_reprompt_with_voice("We hit a temporary error. Please repeat the issue."),
            media_type="application/xml",
        )


@app.get("/")
def root() -> PlainTextResponse:
    return PlainTextResponse("CBRE voice demo server is running.")

