from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel

from cbre_agent.agent import classify
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
    max_turns_per_call: int = int(os.getenv("VOICE_DEMO_MAX_TURNS", "5"))
    max_clarification_rounds: int = int(os.getenv("VOICE_DEMO_MAX_CLARIFICATION_ROUNDS", "2"))
    max_classify_attempts: int = int(os.getenv("VOICE_DEMO_MAX_CLASSIFY_ATTEMPTS", "2"))
    min_caller_turns_before_classify: int = int(os.getenv("VOICE_DEMO_MIN_CALLER_TURNS_BEFORE_CLASSIFY", "2"))
    min_agent_turns_before_classify: int = int(os.getenv("VOICE_DEMO_MIN_AGENT_TURNS_BEFORE_CLASSIFY", "1"))
    outputs_dir: Path = Path(os.getenv("VOICE_DEMO_OUTPUTS_DIR", "demo_outputs"))


SETTINGS = Settings()
SESSIONS = SessionStore()
app = FastAPI(title="CBRE Live Voice Demo")
_DONE_PATTERNS = ("that's all", "that is all", "that's it", "that is it", "no that's all")
_OPENING_PROMPT = (
    "Thank you for calling CBRE maintenance intake. "
    "I will ask a few quick questions, then submit your work order."
)
_ROOT = Path(__file__).resolve().parent.parent
_BUILDINGS_PATH = _ROOT / "operational" / "buildings.json"
_BUILDING_NAMES = [b.get("name", "").strip() for b in json.loads(_BUILDINGS_PATH.read_text()) if b.get("name")]
_BUILDING_NAMES_BY_LC = {name.lower(): name for name in _BUILDING_NAMES}


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
        if value and not session.intake_slots.get(key):
            session.intake_slots[key] = value


def _essential_slots_present(session: CallSession) -> bool:
    return bool(
        session.intake_slots.get("issue")
        and session.intake_slots.get("building_name")
        and session.intake_slots.get("floor")
    )


def _agent_turn_count(session: CallSession) -> int:
    return sum(1 for turn in session.turns if turn.get("speaker") == "agent")


def _has_min_dialogue_for_first_classify(session: CallSession) -> bool:
    return (
        session.turn_count >= SETTINGS.min_caller_turns_before_classify
        and _agent_turn_count(session) >= SETTINGS.min_agent_turns_before_classify
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


def _extract_building_name(text: str) -> Optional[str]:
    lowered = text.lower()
    for name_lc, canonical in _BUILDING_NAMES_BY_LC.items():
        if name_lc in lowered:
            return canonical
    return None


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


def _deterministic_gate_question(session: CallSession) -> Optional[str]:
    if not _has_min_dialogue_for_first_classify(session):
        return "Before I submit this, could you add one more detail about the issue impact?"
    if not session.intake_slots.get("issue"):
        return "Could you briefly describe the exact issue you are seeing?"
    if not session.intake_slots.get("building_name"):
        return "What is the building name for this issue?"
    if not session.intake_slots.get("floor"):
        return "What floor or area is impacted?"
    if (
        session.pending_clarification
        and session.turn_count <= session.clarification_prompt_caller_turn
    ):
        return "Thanks. Could you share one more clarifying detail before I submit this?"
    return None


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
    RecordingUrl: Optional[str] = Form(default=None),
    RecordingDuration: Optional[str] = Form(default=None),
) -> Response:
    started = time.perf_counter()
    session = SESSIONS.get_or_create(CallSid, From)

    try:
        if SETTINGS.demo_mode:
            transcript = mock_transcript()
            stt_seconds = 0.0
        else:
            if not RecordingUrl:
                return Response(
                    content=_reprompt_with_voice("I did not receive your audio. Please repeat the issue."),
                    media_type="application/xml",
                )
            stt_t0 = time.perf_counter()
            audio_bytes = fetch_twilio_recording(
                recording_url=RecordingUrl,
                twilio_account_sid=SETTINGS.twilio_account_sid,
                twilio_auth_token=SETTINGS.twilio_auth_token,
            )
            transcript = transcribe_deepgram(audio_bytes, SETTINGS.deepgram_api_key)
            stt_seconds = time.perf_counter() - stt_t0

        if not transcript.strip():
            duration_seconds = int(RecordingDuration or "0") if (RecordingDuration or "").isdigit() else 0
            prompt = (
                "I did not catch that clearly. Please repeat that response."
                if duration_seconds > 0
                else "I did not hear anything. Please respond after the beep."
            )
            return Response(
                content=_reprompt_with_voice(prompt),
                media_type="application/xml",
            )

        session.append_turn("caller", transcript)
        _update_slots_from_caller_text(session, transcript)
        reached_turn_cap = session.turn_count >= SETTINGS.max_turns_per_call
        gate_question = _deterministic_gate_question(session)
        if gate_question and not reached_turn_cap:
            question = gate_question
            session.append_turn("agent", question)
            xml, tts_seconds = _speak(question, continue_recording=True)
            timings = {
                "stt_seconds": round(stt_seconds, 4),
                "classify_seconds": 0.0,
                "tts_seconds": round(tts_seconds, 4),
                "request_total_seconds": round(time.perf_counter() - started, 4),
                "phase": "llm_clarification_intake_turn" if session.pending_clarification else "llm_intake_turn",
            }
            print(
                f"Call {CallSid} intake_slots={session.intake_slots} timings={timings} "
                f"pending_clarification={session.pending_clarification} gate_question=True"
            )
            return Response(content=xml, media_type="application/xml")

        decision = decide_next_step(
            turns=session.turns,
            intake_slots=session.intake_slots,
            turn_count=session.turn_count,
            max_turns=SETTINGS.max_turns_per_call,
            pending_clarification=session.pending_clarification,
            clarification_focus=session.clarification_focus,
        )
        _merge_slots_from_decision(
            session,
            {
                "issue": decision.captured_issue,
                "building_name": decision.captured_building_name,
                "floor": decision.captured_floor,
                "urgency": decision.captured_urgency,
            },
        )

        can_finalize = decision.should_finalize or reached_turn_cap or _caller_done(transcript)
        if not can_finalize and not reached_turn_cap:
            question = _ensure_question(decision.agent_response)
            session.append_turn("agent", question)
            xml, tts_seconds = _speak(question, continue_recording=True)
            timings = {
                "stt_seconds": round(stt_seconds, 4),
                "classify_seconds": 0.0,
                "tts_seconds": round(tts_seconds, 4),
                "request_total_seconds": round(time.perf_counter() - started, 4),
                "phase": "llm_intake_turn",
            }
            print(
                f"Call {CallSid} intake_slots={session.intake_slots} timings={timings} "
                f"llm_finalize={decision.should_finalize} pending_clarification={session.pending_clarification}"
            )
            return Response(content=xml, media_type="application/xml")

        # Important: classify only on pre-final conversation turns.
        turns_for_classification = list(session.turns)
        clf_t0 = time.perf_counter()
        prediction = classify(turns_for_classification, session.caller_phone)
        classify_seconds = time.perf_counter() - clf_t0
        session.classify_attempts += 1

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

