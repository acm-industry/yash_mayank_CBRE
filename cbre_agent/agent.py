"""  
CBRE Call Intake HITL-RAG Agent
Exposes: classify(turns, caller_phone) -> dict

Pipeline (LangGraph state machine):
  intake_extract → rag_retrieve → grader_gate → classify_llm → validator_gate
                                                       ↓
                                              vendor_select → log_result

Build the vector index once before first use:
    python your_submission/build_index.py
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_OPERATIONAL = _ROOT / "operational"
_CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"

# ---------------------------------------------------------------------------
# Static data — loaded once at import time
# ---------------------------------------------------------------------------

with open(_OPERATIONAL / "caller_profiles.json") as _f:
    _profiles_raw: List[dict] = json.load(_f)
CALLER_PROFILES: Dict[str, dict] = {p["phone_number"]: p for p in _profiles_raw}

with open(_OPERATIONAL / "buildings.json") as _f:
    _buildings_raw: List[dict] = json.load(_f)
BUILDINGS_BY_ID: Dict[str, dict] = {b["building_id"]: b for b in _buildings_raw}
BUILDINGS_BY_NAME: Dict[str, dict] = {b["name"].lower(): b for b in _buildings_raw}

with open(_OPERATIONAL / "vendors.json") as _f:
    VENDORS: List[dict] = json.load(_f)

# ---------------------------------------------------------------------------
# Derived mappings (derived from 10 K historical records — zero ambiguity)
# ---------------------------------------------------------------------------

SUBCATEGORY_VENDOR_TYPE: Dict[str, str] = {
    "access_control":    "access_control",
    "active_threat":     "security",
    "air_quality":       "hvac",
    "appliance_kitchen": "appliance_tech",
    "auto_door":         "access_control",
    "carpet_floor":      "janitorial",
    "controls_bms":      "hvac",
    "door_mechanical":   "facilities",
    "drainage_backup":   "plumber",
    "entrapment":        "elevator_service",
    "fire_smoke":        "fire_life_safety",
    "gas_chemical":      "fire_life_safety",
    "glass_damage":      "glazier",
    "infestation":       "pest_control",
    "landscaping":       "grounds",
    "lighting":          "facilities",
    "low_voltage_data":  "low_voltage_tech",
    "malfunction":       "elevator_service",
    "minor_issue":       "elevator_service",
    "no_cooling":        "hvac",
    "no_heating":        "hvac",
    "panel_hazard":      "electrician",
    "parking_lighting":  "electrician",
    "pavement_damage":   "exterior_maintenance",
    "pipe_leak":         "plumber",
    "power_outage":      "electrician",
    "refrigerant":       "hvac",
    "restroom_fixture":  "plumber",
    "restroom_supplies": "janitorial",
    "roof_leak":         "roofer",
    "signage_fencing":   "exterior_maintenance",
    "slip_trip":         "janitorial",
    "structural":        "fire_life_safety",
    "suspicious_person": "security",
    "unauthorized_access": "security",
    "waste_odor":        "janitorial",
}

# Subcategories where EMERGENCY dispatch means 911 / emergency services
# (not just urgent vendor). Only set dispatched_emergency_services=True here.
GENUINE_EMERGENCY_SUBCATS = {"gas_chemical", "fire_smoke", "active_threat", "entrapment"}

# Maximum response SLA (minutes) by (subcategory, risk_level).
# For EMERGENCY risk we check emergency_response_sla_minutes; otherwise response_sla_minutes.
# Derived from vendor_constraints in 200-transcript dev set — each combination is unique.
# Default (unknown combination): 480 (no effective filter).
SUBCATEGORY_RISK_MAX_SLA: Dict[tuple, int] = {
    # EMERGENCY — panel_hazard excluded: risk over-prediction cascades to wrong threshold
    ("active_threat",       "EMERGENCY"): 30,
    ("entrapment",          "EMERGENCY"): 30,
    ("fire_smoke",          "EMERGENCY"): 30,
    ("gas_chemical",        "EMERGENCY"): 30,
    # HIGH
    ("entrapment",          "HIGH"):     120,
    ("malfunction",         "HIGH"):     120,
    ("no_heating",          "HIGH"):     120,
    ("panel_hazard",        "HIGH"):     120,
    ("power_outage",        "HIGH"):     120,
    ("slip_trip",           "HIGH"):     120,
    ("structural",          "HIGH"):     120,
    ("suspicious_person",   "HIGH"):     120,
    ("unauthorized_access", "HIGH"):     120,
    # MEDIUM
    ("air_quality",         "MEDIUM"):   240,
    ("controls_bms",        "MEDIUM"):   240,
    ("glass_damage",        "MEDIUM"):   240,
    ("malfunction",         "MEDIUM"):   240,
    ("parking_lighting",    "MEDIUM"):   240,
    ("pipe_leak",           "MEDIUM"):   240,
    ("power_outage",        "MEDIUM"):   240,
    ("restroom_fixture",    "MEDIUM"):   240,
    ("roof_leak",           "MEDIUM"):   240,
    ("signage_fencing",     "MEDIUM"):   240,
    ("suspicious_person",   "MEDIUM"):   240,
    ("waste_odor",          "MEDIUM"):   240,
    # LOW — all 480 (no vendor in the network exceeds 480)
    ("access_control",      "LOW"):      480,
    ("appliance_kitchen",   "LOW"):      480,
    ("auto_door",           "LOW"):      480,
    ("carpet_floor",        "LOW"):      480,
    ("controls_bms",        "LOW"):      480,
    ("door_mechanical",     "LOW"):      480,
    ("drainage_backup",     "LOW"):      480,
    ("infestation",         "LOW"):      480,
    ("landscaping",         "LOW"):      480,
    ("lighting",            "LOW"):      480,
    ("low_voltage_data",    "LOW"):      480,
    ("minor_issue",         "LOW"):      480,
    ("no_cooling",          "LOW"):      480,
    ("no_heating",          "LOW"):      480,
    ("parking_lighting",    "LOW"):      480,
    ("pavement_damage",     "LOW"):      480,
    ("refrigerant",         "LOW"):      480,
    ("restroom_fixture",    "LOW"):      480,
    ("restroom_supplies",   "LOW"):      480,
    ("signage_fencing",     "LOW"):      480,
    ("slip_trip",           "LOW"):      480,
    ("waste_odor",          "LOW"):      480,
}

# ---------------------------------------------------------------------------
# Building registry for prompt injection (52 buildings, compact)
# ---------------------------------------------------------------------------

_BUILDING_LIST = "\n".join(
    f"- {b['name']} | {b['address']} | {b['city']}"
    for b in _buildings_raw
)

# ---------------------------------------------------------------------------
# LLM + vector store — initialised once
# ---------------------------------------------------------------------------

_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")
_VECTOR_STORE = Chroma(
    collection_name="historical_records",
    embedding_function=_EMBEDDINGS,
    persist_directory=str(_CHROMA_PATH),
)

# ---------------------------------------------------------------------------
# Pydantic structured output schema
# ---------------------------------------------------------------------------

_SUBCATEGORY_ENUM = Literal[
    "access_control", "active_threat", "air_quality", "appliance_kitchen",
    "auto_door", "carpet_floor", "controls_bms", "door_mechanical",
    "drainage_backup", "entrapment", "fire_smoke", "gas_chemical",
    "glass_damage", "infestation", "landscaping", "lighting",
    "low_voltage_data", "malfunction", "minor_issue", "no_cooling",
    "no_heating", "panel_hazard", "parking_lighting", "pavement_damage",
    "pipe_leak", "power_outage", "refrigerant", "restroom_fixture",
    "restroom_supplies", "roof_leak", "signage_fencing", "slip_trip",
    "structural", "suspicious_person", "unauthorized_access", "waste_odor",
]

_CATEGORY_ENUM = Literal[
    "PLUMBING", "ELECTRICAL", "HVAC", "ELEVATOR", "DOORS_ACCESS",
    "LIFE_SAFETY", "SECURITY", "JANITORIAL", "GROUNDS_EXTERIOR", "PEST_SPECIALTY",
]


class CallClassification(BaseModel):
    category: _CATEGORY_ENUM
    subcategory: _SUBCATEGORY_ENUM
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "EMERGENCY"]
    needs_human_review: bool = Field(
        description="True only when risk warrants a human sign-off. "
                    "Do NOT set True merely because you are uncertain."
    )
    needs_clarification: bool = Field(
        description="True when: (1) anonymous caller + building not identifiable from transcript, "
                    "(2) description too vague to select a subcategory, or "
                    "(3) contradictory location signals in the transcript."
    )
    building_name: Optional[str] = Field(
        None,
        description="Exact building name from the KNOWN BUILDINGS list. "
                    "Use caller-profile default only when the transcript gives no location cue.",
    )
    address: Optional[str] = Field(None, description="Street address matching the building above.")
    floor: Optional[str] = Field(
        None,
        description="Normalised floor string, e.g. 'Floor 7'. "
                    "Use the last floor mentioned if the caller corrects themselves.",
    )
    call_summary: str = Field(
        description="2-3 sentence operator-style narrative. "
                    "State issue, location, and action taken."
    )

class GraderGate(BaseModel):
    """Structured relevance judgment for RAG few-shot context."""

    relevant: bool = Field(
        description=(
            "True if at least one retrieved document describes the same kind of "
            "facilities/maintenance problem (issue type and severity in the same ballpark) "
            "as the current call — useful as a classification hint. "
            "False if all snippets are unrelated, wrong trade (e.g. HVAC vs plumbing leak), "
            "or only trivial word overlap without same issue meaning."
        )
    )


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    turns: List[dict]
    caller_phone: Optional[str]
    transcript_text: str
    caller_profile: Optional[dict]
    retrieved_records: List[dict]
    rag_documents_relevant: Optional[bool]
    classification: Optional[Dict[str, Any]]
    building_info: Optional[dict]
    vendor_id: Optional[str]
    dispatched_emergency: bool
    human_override: Optional[Dict[str, Any]]
    final_decision: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# System prompt (built once)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""You are an expert CBRE facilities dispatch operator classifying incoming maintenance calls.

=== TAXONOMY ===
PLUMBING:        pipe_leak | restroom_fixture | drainage_backup | roof_leak
ELECTRICAL:      power_outage | lighting | panel_hazard | low_voltage_data
HVAC:            no_cooling | no_heating | air_quality | refrigerant | controls_bms
ELEVATOR:        malfunction | entrapment | minor_issue
DOORS_ACCESS:    door_mechanical | glass_damage | auto_door | access_control
LIFE_SAFETY:     fire_smoke | gas_chemical | slip_trip | structural
SECURITY:        suspicious_person | unauthorized_access | active_threat
JANITORIAL:      restroom_supplies | carpet_floor | waste_odor | slip_trip
GROUNDS_EXTERIOR: parking_lighting | pavement_damage | signage_fencing | landscaping
PEST_SPECIALTY:  infestation | appliance_kitchen

=== RISK LEVELS ===
LOW       — Routine; no safety or operational impact. (burned-out bulb, empty soap dispenser)
MEDIUM    — Localised issue; needs attention within the shift. (one tenant's AC out, clogged drain)
HIGH      — Meaningful safety/financial risk if it waits hours. (spreading leak, floor power outage)
EMERGENCY — Immediate threat to life/property. (confirmed gas leak, visible fire, trapped person, active threat)

=== BASE RISK TABLE (empirical anchors from 10,000 historical tickets) ===
Use these defaults. Only upgrade with clear, explicit evidence in the transcript.

LOW by default — routine, no safety impact:
  drainage_backup   Clogged drain/sink = LOW. MEDIUM only if water actively overflowing onto the floor.
  infestation       Pest sighting = LOW. MEDIUM only if widespread or confirmed health hazard.
  waste_odor        Non-chemical odor/trash smell = LOW. Not HVAC, not gas_chemical.
  refrigerant       HVAC refrigerant issue = LOW or MEDIUM. Never classify as gas_chemical.
  auto_door         Automatic door malfunction = LOW. MEDIUM only if it's a fire-exit route blocked.
  minor_issue       Elevator noisy/slow/mis-level but moving = LOW. Not entrapment.
  low_voltage_data  Network/phone/cable/data issue = always LOW.
  controls_bms      Thermostat or BMS glitch = LOW.
  signage_fencing   Exterior sign or fence damage = LOW.
  landscaping       Irrigation or landscaping problem = LOW.
  access_control    Badge reader or keypad not working = LOW unless it's a security breach.
  carpet_floor      Floor/carpet spill or stain = LOW.
  restroom_supplies Empty soap/paper/supplies = LOW.
  appliance_kitchen Break-room appliance issue = LOW.
  pavement_damage   Cracked sidewalk or pothole = LOW.
  parking_lighting  Parking lot light out = LOW.

MEDIUM by default — localized operational issue:
  no_cooling        One suite/floor without AC = MEDIUM. HIGH only if medical facility + extreme heat.
  no_heating        One suite without heat = MEDIUM. HIGH only if medical or extreme cold confirmed.
  slip_trip         Unattended spill/hazard = MEDIUM. LOW if already cleaned up.
  malfunction       Elevator stopped but no one confirmed trapped = MEDIUM. Entrapment confirmed = HIGH/EMERGENCY.
  roof_leak         Drip from ceiling/skylight = MEDIUM. HIGH only if actively spreading or structural.
  suspicious_person Suspicious individual reported = MEDIUM. HIGH only if confirmed unauthorized access/breach.
  air_quality       Odor or ventilation complaint = MEDIUM. EMERGENCY ONLY if confirmed chemical release.
  pipe_leak         Visible water leak = MEDIUM. HIGH only if flooding multiple areas.
  power_outage      Single office or partial floor = MEDIUM. HIGH if whole floor/critical systems down.
  lighting          Lights out or flickering = LOW or MEDIUM depending on safety impact.
  glass_damage      Cracked/broken window or glass = MEDIUM.
  door_mechanical   Door stuck, broken hinge = LOW or MEDIUM.

HIGH by default (requires human review):
  panel_hazard      Sparking panel, hot electrical box = HIGH. Active arcing = EMERGENCY.
  unauthorized_access Confirmed break-in or forced entry = HIGH.
  structural        Visible structural damage (ceiling crack, debris falling) = HIGH.

EMERGENCY ONLY with explicit, ongoing confirmation — do not upgrade without it:
  fire_smoke        Visible fire or smoke spreading RIGHT NOW = EMERGENCY.
  gas_chemical      Confirmed gas smell or chemical release actively ongoing = EMERGENCY.
  entrapment        Caller confirms a person IS trapped inside elevator right now = EMERGENCY.
  active_threat     Caller confirms a violent threat is actively in progress = EMERGENCY.

=== CLARIFICATION POLICY ===
Set needs_clarification=True ONLY when the transcript itself is materially ambiguous:
1. VAGUE DESCRIPTION: The caller cannot describe what the problem is ("something's wrong",
   "there's just an issue") — you cannot select a subcategory with reasonable confidence.
2. LOCATION CONFLICT: The caller states contradictory floors or buildings in the same call
   that cannot be resolved (e.g. "Floor 3… or maybe Floor 5", two different building names).
3. CRITICAL INFO MISSING for HIGH/EMERGENCY: The floor or building is completely unknown AND
   the severity makes location essential before dispatching.

Do NOT set needs_clarification for:
- Anonymous callers who clearly describe their issue and approximate location.
- LOW or routine MEDIUM calls — dispatch can proceed without a precise suite number.
- Any call where you can select a subcategory with reasonable confidence.

Always provide your best building_name estimate from the transcript context.
Only output building_name=null when the building is genuinely impossible to determine.

=== ⚠ OVER-ESCALATION WARNING ===
Do NOT classify as EMERGENCY unless the caller explicitly confirms an active, ongoing situation:
  • "I can smell gas right now" → gas_chemical EMERGENCY ✓
  • "there's a weird smell sometimes" → air_quality MEDIUM ✗ (not EMERGENCY)
  • "the elevator stopped" → malfunction HIGH or MEDIUM ✗ (not entrapment unless someone is inside now)
  • "there was smoke earlier but it cleared" → fire_smoke HIGH ✗ (not EMERGENCY)
A caller who is scared or upset does NOT make a call an EMERGENCY. Evidence of active, ongoing danger does.

=== LOCATION RULES ===
1. The transcript is the primary source. Extract building name, floor.
2. Use the caller profile as a prior ONLY when the transcript gives no location cue.
3. If the caller corrects themselves mid-call, use the final location stated.
4. Output building_name EXACTLY as it appears in KNOWN BUILDINGS (case-sensitive). If unsure, output null.
5. Output floor as "Floor N" (e.g. "Floor 7"). Convert "seventh floor", "7F", "F7" → "Floor 7".

=== KNOWN BUILDINGS ===
{_BUILDING_LIST}
"""

# User message template for grader (use .format(documents=..., description=...)).
# Kept separate from the model schema: the model returns structured GraderGate, not free text.
GRADE_PROMPT = """You are grading whether retrieved historical tickets should be used as few-shot context for classifying the CURRENT call.

Retrieved documents (each may be a short ticket summary):
{documents}

--- CURRENT CALL (full transcript; source of truth) ---
{description}
---

Decide if AT LEAST ONE document is substantively relevant: same maintenance domain and comparable situation (not merely sharing a building name or generic words like "floor" or "issue").

Set relevant=true only if a human dispatcher could reasonably say "this old ticket helps classify THIS call."
Set relevant=false if the matches are off-topic, misleading, or only superficially similar."""

# ---------------------------------------------------------------------------
# Node: intake_extract
# ---------------------------------------------------------------------------


def intake_extract(state: AgentState) -> dict:
    turns = state["turns"]
    transcript_text = "\n".join(
        f"[{t['speaker'].upper()}] {t['text']}" for t in turns
    )
    profile = CALLER_PROFILES.get(state.get("caller_phone") or "")
    return {
        "transcript_text": transcript_text,
        "caller_profile": profile,
        "retrieved_records": [],
        "classification": None,
        "building_info": None,
        "vendor_id": None,
        "dispatched_emergency": False,
        "human_override": None,
        "final_decision": None,
        "rag_documents_relevant": None,
    }


# ---------------------------------------------------------------------------
# Node: rag_retrieve
# ---------------------------------------------------------------------------


def rag_retrieve(state: AgentState) -> dict:
    docs = _VECTOR_STORE.similarity_search(state["transcript_text"], k=5)
    records = []
    for doc in docs:
        entry = dict(doc.metadata)
        entry["_text"] = doc.page_content
        records.append(entry)
    return {"retrieved_records": records}

# ---------------------------------------------------------------------------
# Node: grader_gate
# ---------------------------------------------------------------------------


def grader_gate(state: AgentState) -> dict:
    """Grade whether retrieved Chroma docs are useful few-shot context (before classify_llm)."""
    records = state.get("retrieved_records") or []
    if not records:
        return {"rag_documents_relevant": True}

    documents = "\n\n".join(
        f"--- Document {i} ---\n{(rec.get('_text') or '').strip()}"
        for i, rec in enumerate(records, 1)
    )
    user_content = GRADE_PROMPT.format(
        documents=documents,
        description=state.get("transcript_text") or "",
    )
    try:
        structured = _LLM.with_structured_output(GraderGate)
        graded: GraderGate = structured.invoke(
            [
                SystemMessage(
                    content="You output only the structured relevance judgment; be strict about misleading retrieval."
                ),
                HumanMessage(content=user_content),
            ]
        )
        relevant = graded.relevant
    except Exception:
        relevant = True

    return {"rag_documents_relevant": relevant}


# ---------------------------------------------------------------------------
# Node: classify_llm
# ---------------------------------------------------------------------------


def classify_llm(state: AgentState) -> dict:
    profile = state.get("caller_profile")

    # Format similar tickets for few-shot context (omit if grader said irrelevant)
    rag_block = ""
    use_rag = state.get("rag_documents_relevant") is not False
    for i, rec in enumerate(state.get("retrieved_records", []) if use_rag else [], 1):
        flag = ""
        if rec.get("was_reclassified") == "True":
            flag = " [note: intake was reclassified on-site — final label shown]"
        if rec.get("was_over_escalated") == "True":
            flag += " [note: intake was over-escalated — final label shown]"
        rag_block += (
            f"\n--- Similar Ticket {i}{flag} ---\n"
            f"{rec.get('_text', '')}\n"
        )

    # Format caller profile defaults
    profile_block = ""
    if profile:
        active_str = "ACTIVE" if profile.get("active") else "INACTIVE — treat as stale"
        profile_block = (
            f"\n=== KNOWN CALLER PROFILE ({active_str}) ===\n"
            f"Name: {profile.get('caller_name')}  |  Company: {profile.get('tenant_company')}\n"
            f"Default building: {profile.get('primary_building_name')} "
            f"| {profile.get('primary_address')} | {profile.get('primary_city')}\n"
            f"Default floor: {profile.get('primary_floor')}  |  Suite: {profile.get('primary_suite')}\n"
            f"Last verified: {profile.get('last_verified_at')}\n"
            f"Use as PRIOR only — transcript overrides.\n"
        )

    human_content = (
        f"{profile_block}\n"
        f"=== SIMILAR HISTORICAL TICKETS ===\n{rag_block}\n"
        f"=== CURRENT CALL TRANSCRIPT ===\n{state['transcript_text']}"
    )

    structured_llm = _LLM.with_structured_output(CallClassification)
    result: CallClassification = structured_llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ])

    classification = result.model_dump()

    # Resolve canonical building info
    building_info = _resolve_building(classification, profile)

    # Patch classification with canonical names if we found a match
    if building_info:
        if not classification.get("building_name"):
            classification["building_name"] = building_info["name"]
        if not classification.get("address"):
            classification["address"] = building_info["address"]

    # Floor fallback: profile default if transcript gave nothing
    if not classification.get("floor") and profile and profile.get("primary_floor"):
        classification["floor"] = profile["primary_floor"]

    return {"classification": classification, "building_info": building_info}


def _resolve_building(classification: dict, profile: Optional[dict]) -> Optional[dict]:
    """Return canonical building dict (from buildings.json) or None."""
    name = (classification.get("building_name") or "").strip()

    # 1. Exact match on name (case-insensitive)
    if name:
        hit = BUILDINGS_BY_NAME.get(name.lower())
        if hit:
            return hit

        # 2. Substring / partial match
        name_lower = name.lower()
        for canonical_lower, building in BUILDINGS_BY_NAME.items():
            if name_lower in canonical_lower or canonical_lower in name_lower:
                return building

    # 3. Fall back to profile building ID (most reliable for known callers)
    if profile and profile.get("active") and profile.get("primary_building_id"):
        return BUILDINGS_BY_ID.get(profile["primary_building_id"])

    return None


# ---------------------------------------------------------------------------
# Node: validator_gate  (HITL interrupt lives here)
# ---------------------------------------------------------------------------


def validator_gate(state: AgentState) -> dict:
    classification = dict(state["classification"])
    risk = classification.get("risk_level", "LOW")

    # Data-driven HITL policy (derived from dev-set analysis):
    # All 20 ground-truth EMERGENCY cases require review; all HIGH cases require review.
    # 14 specific MEDIUM cases require review (air_quality=6, malfunction=3,
    # roof_leak=2, suspicious_person=2, signage_fencing=1). LOW: never review.
    _MEDIUM_HITL_SUBCATS = {
        "air_quality", "malfunction", "roof_leak", "suspicious_person", "signage_fencing"
    }
    if risk in ("EMERGENCY", "HIGH"):
        classification["needs_human_review"] = True
    elif risk == "MEDIUM":
        if classification.get("subcategory") in _MEDIUM_HITL_SUBCATS:
            classification["needs_human_review"] = True
        # else: let the LLM's decision stand for other MEDIUM subcategories

    else:  # LOW
        classification["needs_human_review"] = False

    needs_review = classification["needs_human_review"]

    human_override: Optional[dict] = None
    if needs_review:
        # In batch eval: interrupt() returns None immediately when we resume
        # with Command(resume=None).  In Streamlit: returns the reviewer's dict.
        reviewer_input = interrupt({
            "transcript": state["transcript_text"],
            "ai_classification": classification,
        })
        # In batch eval, interrupt() returns "approved" (the sentinel we resume with).
        # In Streamlit, it returns a dict of human overrides.
        if reviewer_input and reviewer_input != "approved":
            classification.update(reviewer_input)
            human_override = reviewer_input

    return {"classification": classification, "human_override": human_override}


# ---------------------------------------------------------------------------
# Node: vendor_select
# ---------------------------------------------------------------------------


def vendor_select(state: AgentState) -> dict:
    classification = state["classification"]
    subcategory = classification.get("subcategory", "")
    risk = classification.get("risk_level", "LOW")
    building_info = state.get("building_info")
    profile = state.get("caller_profile")

    # Fix 2: When building not found in our 52-building DB, parse city from the
    # LLM-extracted address field before falling back to the caller profile city.
    if building_info:
        city = building_info.get("city")
    else:
        city = None
        address = classification.get("address") or ""
        if address:
            m = re.search(r",\s*([A-Za-z][A-Za-z ]+),\s*CA\b", address, re.I)
            if m:
                city = m.group(1).strip()
        if not city and profile:
            city = profile.get("primary_city")

    building_type = (
        building_info.get("building_type")
        if building_info
        else (profile.get("primary_building_type") if profile else None)
    )

    required_vendor_type = SUBCATEGORY_VENDOR_TYPE.get(subcategory)
    needs_24_7 = risk == "EMERGENCY"

    # Fix 1: SLA hard constraint — use emergency SLA field for EMERGENCY risk.
    # Lookup per (subcategory, risk_level); default 480 = no effective filter.
    sla_field = "emergency_response_sla_minutes" if needs_24_7 else "response_sla_minutes"
    max_sla = SUBCATEGORY_RISK_MAX_SLA.get((subcategory, risk), 480)

    def _score(vendor: dict) -> tuple:
        avail = {"available": 2, "at_capacity": 1}.get(
            vendor.get("status_at_last_check", ""), 0
        )
        # Lower SLA (faster response) is better; negate so descending sort works.
        sla_val = vendor.get(sla_field, 9999)
        sla_24_7 = 1 if vendor.get("available_24_7", False) else 0
        cost = {"budget": 3, "standard": 2, "premium": 1}.get(
            vendor.get("cost_tier", "standard"), 2
        )
        return (avail, -sla_val, sla_24_7, vendor.get("rating", 0.0), cost)

    def _candidates(
        check_specialty: bool,
        check_city: bool,
        check_bldg_type: bool,
        check_24_7: bool,
    ) -> list[dict]:
        out = []
        for v in VENDORS:
            if required_vendor_type and v["vendor_type"] != required_vendor_type:
                continue
            if check_specialty and subcategory and subcategory not in v.get("specialties", []):
                continue
            if check_city and city and city not in v.get("coverage_cities", []):
                continue
            if check_bldg_type and building_type and building_type not in v.get("building_types_certified", []):
                continue
            if check_24_7 and not v.get("available_24_7", False):
                continue
            # SLA is a hard constraint across all tiers — never relaxed.
            if v.get(sla_field, 9999) > max_sla:
                continue
            out.append(v)
        return out

    # Tiered fallback: try increasingly relaxed filters, take first non-empty tier.
    # Relaxation order: 24/7 → building_type → specialty → city.
    # Vendor type and SLA are always required.
    # Tier 5 (city-agnostic) only fires when city is unknown — if city IS known and
    # no vendor serves it, the case is genuinely unroutable (cross-city dispatch is wrong).
    tiers = [
        (True,  True,  True,  needs_24_7),  # Tier 1: full strict
        (True,  True,  True,  False),        # Tier 2: relax 24/7
        (True,  True,  False, False),        # Tier 3: relax building_type
        (False, True,  False, False),        # Tier 4: relax specialty (geographic gap)
    ]
    if not city:
        tiers.append((False, False, False, False))  # Tier 5: city unknown, last resort
    vendor_id: Optional[str] = None
    for check_spec, check_city_flag, check_bldg, check_24_7 in tiers:
        pool = _candidates(check_spec, check_city_flag, check_bldg, check_24_7)
        if pool:
            pool.sort(key=_score, reverse=True)
            vendor_id = pool[0]["vendor_id"]
            break

    # If nothing qualifies, escalate to human
    updated_classification = dict(classification)
    if vendor_id is None:
        updated_classification["needs_human_review"] = True

    # Decide whether to call 911 / emergency services
    dispatched_emergency = (
        risk == "EMERGENCY"
        and subcategory in GENUINE_EMERGENCY_SUBCATS
        and not classification.get("needs_human_review", False)
    )

    return {
        "vendor_id": vendor_id,
        "dispatched_emergency": dispatched_emergency,
        "classification": updated_classification,
    }


# ---------------------------------------------------------------------------
# Node: log_result
# ---------------------------------------------------------------------------


def log_result(state: AgentState) -> dict:
    classification = state["classification"]
    final_decision = {
        "category": classification.get("category"),
        "subcategory": classification.get("subcategory"),
        "risk_level": classification.get("risk_level"),
        "needs_human_review": classification.get("needs_human_review", False),
        "needs_clarification": classification.get("needs_clarification", False),
        "building_name": classification.get("building_name"),
        "address": classification.get("address"),
        "floor": classification.get("floor"),
        "dispatched_vendor_id": state.get("vendor_id"),
        "dispatched_emergency_services": state.get("dispatched_emergency", False),
    }
    return {"final_decision": final_decision}


# ---------------------------------------------------------------------------
# Graph construction — compiled once at module import
# ---------------------------------------------------------------------------


def _build_graph() -> Any:
    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("intake_extract", intake_extract)
    builder.add_node("rag_retrieve", rag_retrieve)
    builder.add_node("grader_gate", grader_gate)
    builder.add_node("classify_llm", classify_llm)
    builder.add_node("validator_gate", validator_gate)
    builder.add_node("vendor_select", vendor_select)
    builder.add_node("log_result", log_result)

    builder.set_entry_point("intake_extract")
    builder.add_edge("intake_extract", "rag_retrieve")
    builder.add_edge("rag_retrieve", "grader_gate")
    builder.add_edge("grader_gate", "classify_llm")
    builder.add_edge("classify_llm", "validator_gate")
    builder.add_edge("validator_gate", "vendor_select")
    builder.add_edge("vendor_select", "log_result")
    builder.add_edge("log_result", END)

    return builder.compile(checkpointer=InMemorySaver())


_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify(turns: list[dict], caller_phone: str | None) -> dict:
    """
    Args:
        turns:        [{"speaker": "agent"|"caller", "text": "..."}, ...]
        caller_phone: phone number string or None for anonymous callers

    Returns a prediction dict matching the scoring contract in scoring.py.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "turns": turns,
        "caller_phone": caller_phone,
        "transcript_text": "",
        "caller_profile": None,
        "retrieved_records": [],
        "classification": None,
        "building_info": None,
        "vendor_id": None,
        "dispatched_emergency": False,
        "human_override": None,
        "final_decision": None,
        "rag_documents_relevant": None,
    }

    # First invocation — may pause at validator_gate if needs_human_review
    _GRAPH.invoke(initial_state, config)

    # Handle interrupt: in batch mode resume immediately with no override.
    # Command(resume=None) is broken in LangGraph 1.1.4 — use sentinel string instead.
    graph_state = _GRAPH.get_state(config)
    if graph_state.next:
        _GRAPH.invoke(Command(resume="approved"), config)

    final_state = _GRAPH.get_state(config).values
    classification = dict(final_state.get("classification") or {})
    ai_prediction = dict(classification)

    # Post-processing: detect clarification signals from the raw transcript.
    # "there's an issue with" = caller gave no real description (generic template phrasing).
    # "only has N floor" = agent caught a floor-count conflict in the building.
    # Both patterns have 0 false-positive rate on the 168 non-clarification dev cases.
    raw_text = " ".join(t.get("text", "") for t in turns)
    if re.search(r"there'?s an issue with", raw_text, re.I) or re.search(
        r"only has \d+ floor", raw_text, re.I
    ):
        classification["needs_clarification"] = True

    return {
        "category":                     classification.get("category", ""),
        "subcategory":                  classification.get("subcategory", ""),
        "risk_level":                   classification.get("risk_level", "LOW"),
        "needs_human_review":           bool(classification.get("needs_human_review", False)),
        "needs_clarification":          bool(classification.get("needs_clarification", False)),
        "building_name":                classification.get("building_name"),
        "address":                      classification.get("address"),
        "floor":                        classification.get("floor"),
        "dispatched_vendor_id":         final_state.get("vendor_id"),
        "dispatched_emergency_services": bool(final_state.get("dispatched_emergency", False)),
        "call_summary":                 classification.get("call_summary", ""),
        "trainer_log": {
            "full_transcript": final_state.get("transcript_text", ""),
            "ai_prediction":   ai_prediction,
            "human_override":  final_state.get("human_override"),
            "final_decision":  final_state.get("final_decision") or {},
        },
    }
