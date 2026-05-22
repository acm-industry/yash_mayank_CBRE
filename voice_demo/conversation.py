from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class IntakeDecision(BaseModel):
    should_finalize: bool = Field(
        description="True when enough context is gathered and intake can proceed to classification."
    )
    agent_response: str = Field(
        description="Single concise sentence to speak back to the caller."
    )
    captured_issue: Optional[str] = None
    captured_building_name: Optional[str] = None
    captured_floor: Optional[str] = None
    captured_urgency: Optional[str] = None


_MODEL = os.getenv("VOICE_CONVERSATION_MODEL", "gpt-4o-mini")
_LLM = ChatOpenAI(model=_MODEL, temperature=0.2)
_LLM_STRUCTURED = _LLM.with_structured_output(IntakeDecision)


def _turns_to_text(turns: List[dict], max_turns: int = 12) -> str:
    trimmed = turns[-max_turns:]
    return "\n".join(f"[{t.get('speaker', 'caller').upper()}] {t.get('text', '')}" for t in trimmed)


def decide_next_step(
    turns: List[dict],
    intake_slots: Dict[str, Optional[str]],
    turn_count: int,
    max_turns: int,
    pending_clarification: bool = False,
    clarification_focus: Optional[str] = None,
) -> IntakeDecision:
    system = SystemMessage(
        content=(
            "You are a CBRE phone intake operator. Drive a natural, concise back-and-forth to gather "
            "enough context before work-order submission. Ask at most one focused follow-up question per turn. "
            "Only set should_finalize=true when issue, location/building, and impacted area/floor are sufficiently clear, "
            "or when turn_count reached max_turns. Prefer clarifying the issue details before asking for building/floor "
            "when the caller statement is vague. "
            "If pending_clarification is true, continue gathering clarification details first and avoid finalizing "
            "until the ambiguity is clearly resolved."
        )
    )
    payload = {
        "turn_count": turn_count,
        "max_turns": max_turns,
        "pending_clarification": pending_clarification,
        "clarification_focus": clarification_focus,
        "known_slots": intake_slots,
        "recent_conversation": _turns_to_text(turns),
    }
    user = HumanMessage(content=json.dumps(payload, indent=2))
    return _LLM_STRUCTURED.invoke([system, user])


def clarification_question(prediction: Dict[str, Any], turns: List[dict]) -> str:
    system = SystemMessage(
        content=(
            "You are a CBRE dispatcher. The model flagged needs_clarification=true. "
            "Ask exactly one short clarification question that resolves the highest-risk ambiguity "
            "for dispatch accuracy."
        )
    )
    payload = {
        "prediction": prediction,
        "recent_conversation": _turns_to_text(turns),
    }
    user = HumanMessage(content=json.dumps(payload, indent=2))
    text = _LLM.invoke([system, user]).content
    if isinstance(text, str) and text.strip():
        return text.strip()
    return "Before I submit this, what is the exact symptom and impacted area right now?"


def final_dispatch_response(prediction: Dict[str, Any], turns: List[dict]) -> str:
    system = SystemMessage(
        content=(
            "You are a CBRE dispatch operator closing a phone intake call. "
            "Write a concise final response (2-3 sentences) confirming submission and next step. "
            "Mention key details: problem type, location, risk, and dispatch/escalation outcome. "
            "Do not invent fields; rely only on provided prediction and transcript."
        )
    )
    payload = {
        "prediction": prediction,
        "recent_conversation": _turns_to_text(turns),
    }
    user = HumanMessage(content=json.dumps(payload, indent=2))
    text = _LLM.invoke([system, user]).content
    if isinstance(text, str) and text.strip():
        return text.strip()

    category = prediction.get("category") or "maintenance"
    subcategory = prediction.get("subcategory") or "issue"
    risk = prediction.get("risk_level") or "LOW"
    building = prediction.get("building_name") or "the property"
    floor = prediction.get("floor") or "unspecified area"
    vendor = prediction.get("dispatched_vendor_id") or "dispatch team"
    return (
        f"Thank you for the details. I am submitting your work order now for {category} / {subcategory}. "
        f"It is logged at {building}, {floor}, risk {risk}, and dispatch is going to {vendor}."
    )

