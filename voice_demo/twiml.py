from __future__ import annotations

from twilio.twiml.voice_response import VoiceResponse

# Latency-tuned defaults for Twilio <Gather input="speech">.
# speechTimeout="auto" lets Twilio detect end-of-speech as soon as the caller
# pauses; this is dramatically faster than <Record> which waits for a full
# silence timeout. The phone_call model is tuned for telephony audio.
_GATHER_KWARGS = {
    "input": "speech",
    "action": "/voice/process",
    "method": "POST",
    "speech_timeout": "auto",
    "speech_model": "phone_call",
    "enhanced": True,
    "timeout": 5,
    "language": "en-US",
    "action_on_empty_result": True,
    "barge_in": True,
}


def _add_speech_gather(response: VoiceResponse) -> None:
    response.gather(**_GATHER_KWARGS)


def incoming_call_prompt() -> str:
    """Initial call prompt for inbound calls."""
    response = VoiceResponse()
    gather = response.gather(**_GATHER_KWARGS)
    gather.say(
        "Thank you for calling CBRE maintenance intake. "
        "How can I help you today?",
        voice="alice",
    )
    response.redirect("/voice/process", method="POST")
    return str(response)


def reprompt_record(message: str) -> str:
    """Play message then listen for next caller turn (low-latency Gather)."""
    response = VoiceResponse()
    gather = response.gather(**_GATHER_KWARGS)
    gather.say(message, voice="alice")
    response.redirect("/voice/process", method="POST")
    return str(response)


def silent_record_only() -> str:
    """Fallback listen-only TwiML with no spoken prompt."""
    response = VoiceResponse()
    _add_speech_gather(response)
    response.redirect("/voice/process", method="POST")
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
    gather = response.gather(**_GATHER_KWARGS)
    if audio_url:
        gather.play(audio_url)
    if follow_up_text:
        gather.say(follow_up_text, voice="alice")
    response.redirect("/voice/process", method="POST")
    return str(response)

