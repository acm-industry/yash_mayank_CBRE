from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CallSession:
    call_sid: str
    caller_phone: Optional[str]
    turns: List[dict] = field(default_factory=list)
    turn_count: int = 0
    clarification_rounds: int = 0
    classify_attempts: int = 0
    pending_clarification: bool = False
    clarification_focus: Optional[str] = None
    clarification_prompt_caller_turn: int = 0
    pending_hitl_review_id: Optional[str] = None
    active_graph_thread_id: Optional[str] = None
    building_ask_count: int = 0
    intake_slots: Dict[str, Optional[str]] = field(
        default_factory=lambda: {
            "issue": None,
            "building_name": None,
            "floor": None,
            "urgency": None,
            "impact_scope": None,
        }
    )
    asked_question_keys: Set[str] = field(default_factory=set)
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

