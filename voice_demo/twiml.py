from __future__ import annotations

from twilio.twiml.voice_response import VoiceResponse


def incoming_call_prompt() -> str:
    """Initial call prompt for inbound calls."""
    response = VoiceResponse()
    response.say(
        "Thank you for calling CBRE maintenance intake. "
        "I will ask a few quick questions and then submit your work order. "
        "After the beep, please describe the issue.",
        voice="alice",
    )
    response.record(
        action="/voice/process",
        method="POST",
        max_length=30,
        timeout=3,
        play_beep=True,
        recording_status_callback_method="POST",
    )
    response.say("We did not hear anything. Please call back when ready.", voice="alice")
    response.hangup()
    return str(response)


def reprompt_record(message: str) -> str:
    """Play message then record next caller turn."""
    response = VoiceResponse()
    response.say(message, voice="alice")
    response.record(
        action="/voice/process",
        method="POST",
        max_length=30,
        timeout=3,
        play_beep=True,
        recording_status_callback_method="POST",
    )
    response.say("We still could not capture audio. Ending the call for now.", voice="alice")
    response.hangup()
    return str(response)


def silent_record_only() -> str:
    """Fallback record-only TwiML with no spoken prompt."""
    response = VoiceResponse()
    response.record(
        action="/voice/process",
        method="POST",
        max_length=30,
        timeout=5,
        play_beep=True,
        recording_status_callback_method="POST",
    )
    return str(response)


def play_audio_and_hangup(audio_url: str | None, outro_text: str | None = None) -> str:
    response = VoiceResponse()
    if audio_url:
        response.play(audio_url)
    if outro_text:
        response.say(outro_text, voice="alice")
    response.hangup()
    return str(response)


def play_audio_then_record(audio_url: str | None, follow_up_text: str | None = None) -> str:
    response = VoiceResponse()
    if audio_url:
        response.play(audio_url)
    if follow_up_text:
        response.say(follow_up_text, voice="alice")
    response.record(
        action="/voice/process",
        method="POST",
        max_length=30,
        timeout=5,
        play_beep=True,
        recording_status_callback_method="POST",
    )
    return str(response)

