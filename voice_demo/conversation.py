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
    known_building_names: Optional[List[str]] = None,
    building_ask_count: int = 0,
) -> IntakeDecision:
    system = SystemMessage(
        content=(
            "You are a friendly CBRE phone intake operator handling a live call. "
            "Drive a natural, warm back-and-forth conversation, one focused question at a time, "
            "to gather enough context before submitting a work order. "
            "Speak like a human, not a form-filler: vary phrasing, acknowledge what the caller said, "
            "and never repeat the same canned line. "
            "\n\n"
            "RULES:\n"
            "- Ask at most ONE question per turn.\n"
            "- Set should_finalize=true ONLY when you have a clear issue description AND a building AND a "
            "floor/area (or when turn_count >= max_turns - 1).\n"
            "- If the caller mentions a building name that does not exactly match the canonical list, "
            "infer the closest match and return it as captured_building_name. Phone STT is imperfect, "
            "so be tolerant of partial or fuzzy matches like 'west park' for 'Westpark Professional Center'.\n"
            "- If you have already asked about the building 2 or more times and still are unsure, ACCEPT "
            "the caller's best statement (use captured_building_name as their last building utterance) "
            "and move on to the floor or finalize. Do not loop on the same slot.\n"
            "- If pending_clarification is true, ask one short clarifying question targeted at the focus.\n"
            "- Avoid robotic phrases like 'before I submit this, could you add one more detail'. "
            "Use phrases like 'Got it' / 'Okay' / 'Thanks for that' to make the conversation feel natural.\n"
            "- When you do have enough info, set should_finalize=true and put a brief acknowledgement in "
            "agent_response (e.g. 'Got it — let me get someone dispatched right away.')."
        )
    )
    payload = {
        "turn_count": turn_count,
        "max_turns": max_turns,
        "pending_clarification": pending_clarification,
        "clarification_focus": clarification_focus,
        "known_slots": intake_slots,
        "building_ask_count": building_ask_count,
        "canonical_building_names": known_building_names or [],
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

