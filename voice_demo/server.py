from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel

from cbre_agent.agent import classify
from voice_demo.session_store import SessionStore
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
)
from voice_demo.workorder import needs_follow_up, persist_call_artifact, render_dispatch_message


class Settings(BaseModel):
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "")
    demo_mode: bool = os.getenv("VOICE_DEMO_MODE", "false").lower() == "true"
    max_turns_per_call: int = int(os.getenv("VOICE_DEMO_MAX_TURNS", "3"))
    outputs_dir: Path = Path(os.getenv("VOICE_DEMO_OUTPUTS_DIR", "demo_outputs"))


SETTINGS = Settings()
SESSIONS = SessionStore()
app = FastAPI(title="CBRE Live Voice Demo")


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
    return Response(content=incoming_call_prompt(), media_type="application/xml")


@app.post("/voice/process")
async def voice_process(
    CallSid: str = Form(...),
    From: Optional[str] = Form(default=None),
    RecordingUrl: Optional[str] = Form(default=None),
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
                    content=reprompt_record("I did not receive your audio. Please repeat the issue."),
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
            return Response(
                content=reprompt_record("I could not understand that. Please describe the issue again."),
                media_type="application/xml",
            )

        session.append_turn("caller", transcript)

        clf_t0 = time.perf_counter()
        prediction = classify(session.turns, session.caller_phone)
        classify_seconds = time.perf_counter() - clf_t0

        response_text = render_dispatch_message(prediction)
        if prediction.get("needs_clarification"):
            response_text += " I still need one more detail to proceed."

        tts_t0 = time.perf_counter()
        if SETTINGS.demo_mode:
            # In mock mode we still return TwiML with spoken fallback via <Say>.
            tts_audio = b""
            tts_seconds = 0.0
        else:
            tts_audio = synthesize_elevenlabs(
                text=response_text,
                elevenlabs_api_key=SETTINGS.elevenlabs_api_key,
                elevenlabs_voice_id=SETTINGS.elevenlabs_voice_id,
            )
            tts_seconds = time.perf_counter() - tts_t0

        continue_call = needs_follow_up(
            prediction=prediction,
            max_turns=SETTINGS.max_turns_per_call,
            current_turn_count=session.turn_count,
        )

        timings = {
            "stt_seconds": round(stt_seconds, 4),
            "classify_seconds": round(classify_seconds, 4),
            "tts_seconds": round(tts_seconds, 4),
            "request_total_seconds": round(time.perf_counter() - started, 4),
        }
        persist_call_artifact(
            out_dir=SETTINGS.outputs_dir / "workorders",
            call_sid=CallSid,
            caller_phone=From,
            turns=session.turns,
            prediction=prediction,
            timings=timings,
        )

        session.append_turn("agent", response_text)
        if continue_call:
            if SETTINGS.demo_mode:
                xml = reprompt_record(f"{response_text} Please share the missing detail after the beep.")
            else:
                filename = _save_tts_audio(tts_audio)
                xml = play_audio_then_record(
                    audio_url=_public_audio_url(filename),
                    follow_up_text="Please provide one more detail after the beep.",
                )
            return Response(content=xml, media_type="application/xml")

        SESSIONS.end(CallSid)
        if SETTINGS.demo_mode:
            xml = play_audio_and_hangup(
                audio_url="",
                outro_text=f"{response_text} Thank you. Your request has been logged.",
            )
            return Response(content=xml, media_type="application/xml")

        filename = _save_tts_audio(tts_audio)
        xml = play_audio_and_hangup(
            audio_url=_public_audio_url(filename),
            outro_text="Thank you. Your request has been logged.",
        )
        return Response(content=xml, media_type="application/xml")
    except SpeechError as exc:
        xml = reprompt_record(f"Audio processing failed: {exc}. Please try once more.")
        return Response(content=xml, media_type="application/xml")
    except Exception as exc:  # noqa: BLE001 - keep webhook resilient in demo mode.
        print(f"Unhandled /voice/process error: {exc}")
        return Response(
            content=reprompt_record("We hit a temporary error. Please repeat the issue."),
            media_type="application/xml",
        )


@app.get("/")
def root() -> PlainTextResponse:
    return PlainTextResponse("CBRE voice demo server is running.")

