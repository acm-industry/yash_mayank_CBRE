from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CallSession:
    call_sid: str
    caller_phone: Optional[str]
    turns: List[dict] = field(default_factory=list)
    turn_count: int = 0
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def append_turn(self, speaker: str, text: str) -> None:
        clean_text = (text or "").strip()
        if not clean_text:
            return
        self.turns.append({"speaker": speaker, "text": clean_text})
        if speaker == "caller":
            self.turn_count += 1
        self.updated_at = _utc_now_iso()


class SessionStore:
    """In-memory session store keyed by Twilio Call SID."""

    def __init__(self) -> None:
        self._sessions: Dict[str, CallSession] = {}

    def get_or_create(self, call_sid: str, caller_phone: Optional[str]) -> CallSession:
        existing = self._sessions.get(call_sid)
        if existing:
            return existing
        session = CallSession(call_sid=call_sid, caller_phone=caller_phone)
        self._sessions[call_sid] = session
        return session

    def end(self, call_sid: str) -> Optional[CallSession]:
        return self._sessions.pop(call_sid, None)

