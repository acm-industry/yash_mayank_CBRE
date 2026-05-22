from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class IntakeDecision(BaseModel):
    should_finalize: bool = Field(
        description=(
            "True only when you have a clear understanding of the issue, location, "
            "scope/impact, and whether anything is actively escalating. NEVER finalize "
            "just because issue + building + floor slots are filled."
        )
    )
    agent_response: str = Field(
        description=(
            "Single concise sentence to speak back to the caller. If finalizing, this "
            "is a brief acknowledgement like 'Got it — let me get someone dispatched right away.'"
        )
    )
    captured_issue: Optional[str] = Field(
        default=None, description="The specific symptom/problem in the caller's own terms."
    )
    captured_building_name: Optional[str] = Field(
        default=None, description="Building name canonicalized to the provided list when possible."
    )
    captured_floor: Optional[str] = Field(
        default=None, description="Floor, suite, or area impacted."
    )
    captured_urgency: Optional[str] = Field(
        default=None, description="emergency | urgent | normal — derived from caller statements."
    )
    captured_impact_scope: Optional[str] = Field(
        default=None,
        description=(
            "How widespread the issue is. Examples: 'one fixture', 'back row of parking', "
            "'whole floor', 'one suite', 'multiple suites', 'common area only'."
        ),
    )
    captured_active_status: Optional[str] = Field(
        default=None,
        description=(
            "Whether the problem is happening right now or already resolved. Examples: "
            "'active and spreading', 'active', 'intermittent', 'stopped now', 'tripping repeatedly'."
        ),
    )
    captured_safety_signal: Optional[str] = Field(
        default=None,
        description=(
            "Any safety-critical detail mentioned: alarm, smoke, chemical smell, trapped person, "
            "injury, electrical risk, water spreading, etc. Empty when none reported."
        ),
    )


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
            "You are a friendly CBRE phone intake operator handling a live maintenance call. "
            "Drive a natural, warm back-and-forth, one focused question at a time. Your goal is to "
            "fully understand the ISSUE first, then ask for the location LAST. Never lead with "
            "'what building are you in?'.\n"
            "\n"
            "CONVERSATION STYLE (mimic these example operator phrasings):\n"
            "- 'Okay, got it. How many fixtures are out?'\n"
            "- 'Got it. Is anyone in the elevator right now?'\n"
            "- 'How bad is the smell — is it in one room or the whole floor?'\n"
            "- 'Did any alarm go off, or do you see smoke?'\n"
            "- 'Is the water still actively spreading or did it stop?'\n"
            "- 'Is the breaker tripping repeatedly or did the power just go out once?'\n"
            "Acknowledge what the caller said before asking the next question. Vary phrasing. "
            "Never use canned lines like 'before I submit this, could you add one more detail'.\n"
            "\n"
            "CONVERSATION PLAN — you have a strict turn budget. Stick to this script:\n"
            "  Stage A (issue context, 1-2 caller turns max): probe 1-2 of these in any order, "
            "  picking whichever is most relevant to the caller's complaint:\n"
            "      - Scope / impact area (one fixture vs whole floor, etc.)\n"
            "      - Active status (happening now, stopped, spreading, intermittent)\n"
            "      - Safety signal (alarm sounded, smoke, chemical smell, anyone trapped/hurt)\n"
            "  Stage B (location, 1-2 caller turns max): ask BOTH building AND floor. Prefer to "
            "  combine them into ONE question, e.g. 'And which building and floor are you "
            "  calling from?'. Split only if the caller answers partially.\n"
            "  Stage C: finalize with a short acknowledgement.\n"
            "\n"
            "HARD CONSTRAINTS ON ORDER:\n"
            "- NEVER lead with 'what building are you in?'. Always probe the issue first.\n"
            "- If the caller volunteered the building or floor in their statement, RECORD it "
            "silently in the captured slots and keep going with Stage A — do not re-ask.\n"
            "- Once you have 1-2 context signals captured (any of scope/active/safety), MOVE TO "
            "Stage B and ask for building + floor on the very next turn.\n"
            "- By turn_count >= max_turns - 2, if building is unknown you MUST ask for it now.\n"
            "- By turn_count >= max_turns - 1, if floor is unknown you MUST ask for it now.\n"
            "\n"
            "FINALIZE RULES:\n"
            "- Set should_finalize=true ONLY when:\n"
            "    (a) you have a clear issue description, AND\n"
            "    (b) at least one of scope / active_status / safety_signal has been captured, AND\n"
            "    (c) building AND floor are BOTH known.\n"
            "- NEVER set should_finalize=true while building or floor is still unknown — instead "
            "  your agent_response MUST be the question asking for whichever is missing.\n"
            "- If pending_clarification is true, ask one short clarifying question targeted at the focus.\n"
            "\n"
            "BUILDING NAME HANDLING:\n"
            "- canonical_building_names contains the valid building list. If the caller mentions a "
            "partial / mis-heard name (e.g. 'west park' or 'westpark'), map it to the closest canonical "
            "name in captured_building_name. Phone STT is imperfect.\n"
            "- If you have asked about the building 2+ times (building_ask_count >= 2) and still unsure, "
            "ACCEPT the caller's best statement and move on. Do NOT loop on the building slot.\n"
            "\n"
            "OUTPUT FIELDS:\n"
            "- agent_response: the next thing to SAY to the caller (one sentence). When finalizing, "
            "use a brief acknowledgement like 'Got it — let me get someone dispatched right away.'.\n"
            "- captured_issue / captured_building_name / captured_floor / captured_urgency: slot updates.\n"
            "- captured_impact_scope: how widespread (e.g. 'one suite', 'whole back row of parking').\n"
            "- captured_active_status: 'active', 'intermittent', 'stopped', 'spreading', etc.\n"
            "- captured_safety_signal: a short phrase like 'no alarm, no smoke', 'fire alarm sounded', "
            "'someone trapped inside', 'water spreading to next room', or empty when not yet asked."
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

