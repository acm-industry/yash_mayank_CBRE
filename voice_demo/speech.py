from __future__ import annotations

import os
from typing import Any, Dict

import requests


class SpeechError(RuntimeError):
    pass


def _recording_wav_url(recording_url: str) -> str:
    if recording_url.endswith(".wav"):
        return recording_url
    if recording_url.endswith(".mp3"):
        return recording_url
    return f"{recording_url}.wav"


def fetch_twilio_recording(
    recording_url: str,
    twilio_account_sid: str,
    twilio_auth_token: str,
    timeout_s: int = 20,
) -> bytes:
    url = _recording_wav_url(recording_url)
    response = requests.get(
        url,
        auth=(twilio_account_sid, twilio_auth_token),
        timeout=timeout_s,
    )
    if response.status_code >= 400:
        raise SpeechError(f"Failed to download Twilio recording: HTTP {response.status_code}")
    return response.content


def transcribe_deepgram(audio_bytes: bytes, deepgram_api_key: str, timeout_s: int = 30) -> str:
    if not deepgram_api_key:
        raise SpeechError("DEEPGRAM_API_KEY is missing.")
    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true"
    headers = {
        "Authorization": f"Token {deepgram_api_key}",
        "Content-Type": "audio/wav",
    }
    response = requests.post(url, headers=headers, data=audio_bytes, timeout=timeout_s)
    if response.status_code >= 400:
        raise SpeechError(f"Deepgram transcription failed: HTTP {response.status_code} {response.text[:180]}")
    data: Dict[str, Any] = response.json()
    try:
        return (
            data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise SpeechError(f"Unexpected Deepgram response shape: {data}") from exc


def synthesize_elevenlabs(
    text: str,
    elevenlabs_api_key: str,
    elevenlabs_voice_id: str,
    timeout_s: int = 30,
) -> bytes:
    if not elevenlabs_api_key:
        raise SpeechError("ELEVENLABS_API_KEY is missing.")
    if not elevenlabs_voice_id:
        raise SpeechError("ELEVENLABS_VOICE_ID is missing.")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}"
    headers = {
        "xi-api-key": elevenlabs_api_key,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
    }
    body = {
        "text": text,
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.8},
    }
    response = requests.post(url, headers=headers, json=body, timeout=timeout_s)
    if response.status_code >= 400:
        raise SpeechError(f"ElevenLabs TTS failed: HTTP {response.status_code} {response.text[:180]}")
    return response.content


def mock_transcript() -> str:
    return os.getenv(
        "VOICE_DEMO_MOCK_TRANSCRIPT",
        "There is a pipe leak on floor 3 at 200 Market Street.",
    )

