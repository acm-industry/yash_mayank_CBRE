"""  
CBRE Call Intake HITL-RAG Agent
Exposes: classify(turns, caller_phone) -> dict

Pipeline (LangGraph state machine):
  intake_extract → rag_retrieve ⇄ grader_gate → classify_llm → validator_gate → hitl_review
       (rag_query copy of transcript; optional rewrite_rag_query loop if grader fails)
                                                                      ↓
                                                             vendor_select → log_result

Build the vector index once before first use:
    python your_submission/build_index.py
"""
from __future__ import annotations

import json
import os
import re
import time
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

# Max times we run rewrite_rag_query after a failed grade (each run re-embeds and re-retrieves).
_MAX_RAG_REWRITE_ATTEMPTS = 2

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
# Canonical category for each subcategory (used for consistency enforcement in validator_gate).
# slip_trip is omitted — it can be LIFE_SAFETY or JANITORIAL depending on context.
_CANONICAL_CATEGORY: Dict[str, str] = {
    "pipe_leak": "PLUMBING", "restroom_fixture": "PLUMBING",
    "drainage_backup": "PLUMBING", "roof_leak": "PLUMBING",
    "power_outage": "ELECTRICAL", "lighting": "ELECTRICAL",
    "panel_hazard": "ELECTRICAL", "low_voltage_data": "ELECTRICAL",
    "no_cooling": "HVAC", "no_heating": "HVAC",
    "air_quality": "HVAC", "refrigerant": "HVAC", "controls_bms": "HVAC",
    "malfunction": "ELEVATOR", "entrapment": "ELEVATOR", "minor_issue": "ELEVATOR",
    "door_mechanical": "DOORS_ACCESS", "glass_damage": "DOORS_ACCESS",
    "auto_door": "DOORS_ACCESS", "access_control": "DOORS_ACCESS",
    "fire_smoke": "LIFE_SAFETY", "gas_chemical": "LIFE_SAFETY", "structural": "LIFE_SAFETY",
    "suspicious_person": "SECURITY", "unauthorized_access": "SECURITY", "active_threat": "SECURITY",
    "restroom_supplies": "JANITORIAL", "carpet_floor": "JANITORIAL", "waste_odor": "JANITORIAL",
    "parking_lighting": "GROUNDS_EXTERIOR", "pavement_damage": "GROUNDS_EXTERIOR",
    "signage_fencing": "GROUNDS_EXTERIOR",
    "landscaping": "PEST_SPECIALTY",
    "infestation": "PEST_SPECIALTY", "appliance_kitchen": "PEST_SPECIALTY",
}

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
        description=(
            "True for EMERGENCY/HIGH risk. "
            "For MEDIUM risk, True ONLY when escalation is present: "
            "malfunction=True if someone was inside the elevator (even if out now); "
            "suspicious_person=True if active yelling/arguing/threatening (NOT passive loitering); "
            "roof_leak=True if ceiling tile fell or structural damage visible (NOT just dripping from rain); "
            "air_quality=True for HVAC chemical smells affecting multiple people. "
            "LOW risk: always False. Do NOT set True merely because you are uncertain."
        )
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
    address: Optional[str] = Field(
        None,
        description="Street address only — no city, state, or zip. E.g. '800 Summit Blvd'.",
    )
    floor: Optional[str] = Field(
        None,
        description="Normalised floor string. Use 'Floor N' for numbered floors (e.g. 'Floor 7'). "
                    "For named areas use the name as-is: 'Rooftop', 'Basement', 'Lobby', 'Mezzanine'. "
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

class RewriteRAGQuery(BaseModel):
    """Rewrite the RAG query to be more specific to the current call."""
    query: str = Field(description="The rewritten RAG query.")



# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    turns: List[dict]
    caller_phone: Optional[str]
    transcript_text: str
    # Embedding search text: copy of transcript at intake; rewrite_rag_query may replace it.
    rag_query: str
    # Number of completed rewrite_rag_query runs (caps retrieve→grade loops).
    rag_rewrite_count: int
    caller_profile: Optional[dict]
    retrieved_records: List[dict]
    # Best (lowest) L2 distance from the most recent rag_retrieve call.
    rag_top_score: Optional[float]
    rag_documents_relevant: Optional[bool]
    classification: Optional[Dict[str, Any]]
    building_info: Optional[dict]
    vendor_id: Optional[str]
    dispatched_emergency: bool
    human_override: Optional[Dict[str, Any]]
    hitl_decision: Optional[Dict[str, Any]]
    routing_decision: Optional[str]
    final_decision: Optional[Dict[str, Any]]
    # Per-node wall times (seconds); multiple entries when a node runs more than once (e.g. RAG loop).
    node_timings: List[Dict[str, Any]]


def _merge_with_timing(
    state: AgentState,
    node: str,
    elapsed_s: float,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    """Append one timing event and merge `updates` into the graph state."""
    out = dict(updates)
    ev = list(state.get("node_timings") or [])
    ev.append({"node": node, "seconds": round(elapsed_s, 6)})
    out["node_timings"] = ev
    return out


# ---------------------------------------------------------------------------
# System prompt (built once)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""You are an expert CBRE facilities dispatch operator classifying incoming maintenance calls.
.
=== TAXONOMY ===
PLUMBING:        pipe_leak | restroom_fixture | drainage_backup | roof_leak
ELECTRICAL:      power_outage | lighting | panel_hazard | low_voltage_data
HVAC:            no_cooling | no_heating | air_quality | refrigerant | controls_bms
ELEVATOR:        malfunction | entrapment | minor_issue
DOORS_ACCESS:    door_mechanical | glass_damage | auto_door | access_control
LIFE_SAFETY:     fire_smoke | gas_chemical | slip_trip | structural
SECURITY:        suspicious_person | unauthorized_access | active_threat
JANITORIAL:      restroom_supplies | carpet_floor | waste_odor | slip_trip
GROUNDS_EXTERIOR: parking_lighting | pavement_damage | signage_fencing
PEST_SPECIALTY:  infestation | appliance_kitchen | landscaping

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
  signage_fencing   Exterior sign or fence damage = LOW. MEDIUM only if structurally unsafe (leaning, about to fall, posing fall risk).
  landscaping       Irrigation or landscaping problem = LOW. Category: PEST_SPECIALTY (not GROUNDS_EXTERIOR).
  access_control    Badge reader or keypad not working = LOW. Always LOW unless the failure enabled a confirmed security breach.
  carpet_floor      Floor/carpet spill or stain = LOW.
  restroom_supplies Empty soap/paper/supplies = LOW.
  appliance_kitchen Break-room appliance issue = LOW.
  pavement_damage   Cracked sidewalk or pothole = LOW.
  parking_lighting  Parking lot light out = LOW.

MEDIUM by default — localized operational issue:
  no_cooling        Standard single-suite comfort complaint = LOW. MEDIUM if multiple suites affected or HVAC actively malfunctioning. HIGH only if medical facility + extreme heat.
  no_heating        Standard single-suite heat complaint = LOW. MEDIUM if HVAC actively blowing wrong-temp air (system fault) or multiple suites. HIGH only if medical facility or extreme cold confirmed.
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
  panel_hazard      Sparking panel, hot electrical box = HIGH. Active arcing OR smell of burning (electrical burning smell) = EMERGENCY.
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

=== ⚠ OVER-ESCALATION WARNINGS ===

LOW → MEDIUM: Do NOT upgrade because the caller sounds urgent, mentions a meeting, or says "please hurry."
  access_control: badge reader/keypad failure = always LOW. "It won't scan" / "nothing is registering" = LOW.
  no_heating: "heat isn't working" for a single suite = LOW. Only MEDIUM if HVAC actively blowing cold air (system fault) or multi-suite.
  no_cooling: "AC isn't working" / "it's hot" for a single suite = LOW. Only MEDIUM if multi-suite or HVAC malfunction.
  drainage_backup, infestation, minor_issue, refrigerant, waste_odor, landscaping, auto_door — LOW by default.
  Upgrade ONLY with explicit, objective evidence (water overflowing onto floor, confirmed health hazard, etc.).

MEDIUM → HIGH: Only upgrade when the specific HIGH condition below is met — not by default:
  • pipe_leak: MEDIUM unless water spreading to MULTIPLE rooms/floors OR threatening electrical.
    A single dripping pipe, slow leak in one spot, leak under a sink = MEDIUM, not HIGH.
  • power_outage: MEDIUM unless an ENTIRE floor has zero power OR a critical system is down.
    One office, one breaker, one circuit = MEDIUM.
  • glass_damage: Always MEDIUM. Broken glass alone never qualifies for HIGH.
  • suspicious_person: MEDIUM unless caller confirms active physical threat OR social engineering
    (asking employees for badge numbers / credentials) OR confirmed no-badge in a secured area.
    Passive loitering without those signals = MEDIUM.
  • drainage_backup: LOW unless water actively overflowing across the floor and spreading.

HIGH → EMERGENCY: Only when the caller EXPLICITLY confirms an active, ongoing situation right now:
  • "I can smell gas right now" → gas_chemical EMERGENCY ✓
  • "there's a weird smell sometimes" → air_quality MEDIUM ✗
  • "the elevator stopped" → malfunction MEDIUM ✗ (NOT entrapment unless someone is confirmed inside now)
  • "I was stuck in the elevator earlier" → malfunction HIGH ✗ (past tense — not EMERGENCY)
  • "there was smoke earlier but it cleared" → fire_smoke HIGH ✗
  • panel_hazard sparking/hot = HIGH. Only EMERGENCY if active arcing or confirmed fire started.
  • entrapment/unauthorized_access/panel_hazard that seem serious but person is SAFE = HIGH, not EMERGENCY.
A caller who is scared or upset does NOT make a call an EMERGENCY.

=== SUBCATEGORY DISAMBIGUATION ===

waste_odor vs air_quality vs fire_smoke:
  • waste_odor: Bad smell from a PHYSICAL SOURCE — garbage, trash, dumpster, sewage, restroom,
    burnt food (toast, microwave), candle smoke. Even if a smoke alarm briefly triggered, if the
    source is identified as burnt food or a candle → waste_odor (not fire_smoke, not air_quality).
    Vendor: janitorial.
  • air_quality: HVAC/ventilation complaint — stale air, stuffiness, solvent/chemical smell from
    unknown source, smell from vents. Generic "chemical smell" without confirmed gas = air_quality.
    Never classify a non-sulfur, non-confirmed-gas chemical smell as gas_chemical.
  • gas_chemical: ONLY when caller explicitly smells gas OR sulfur ("rotten eggs") AND confirms
    it is ongoing. "Chemical smell" alone = air_quality, not gas_chemical.
  • Rule: known physical source (trash/burnt food/candle) → waste_odor.
    Vent/HVAC/unknown indoor smell → air_quality. Confirmed gas/sulfur → gas_chemical.

minor_issue vs malfunction (ELEVATOR):
  • minor_issue: Elevator IS moving but slow, noisy, jerky, mis-leveling, or door slow.
  • malfunction: Elevator STOPPED and does NOT respond. No one confirmed trapped.
  • entrapment: Someone IS inside a non-moving elevator right now → EMERGENCY.

auto_door vs door_mechanical:
  • auto_door: POWERED/automatic door — sliding lobby doors, sensor-activated, handicap button,
    fire-exit doors with automatic openers. If the MAIN ENTRANCE or lobby doors are "being weird",
    stuck, or not working → auto_door (main entrance doors are almost always automatic). Vendor: access_control.
  • door_mechanical: MANUAL door with broken HARDWARE — hinges, closer, latch, lock, handle. Vendor: facilities.
  • Rule: main entrance / lobby door issue → auto_door. Manual interior door hardware broken → door_mechanical.

drainage_backup vs pipe_leak:
  • drainage_backup: Water NOT draining (clog) — sink, drain, toilet backing up.
  • pipe_leak: Water coming FROM a pipe — dripping joint, burst pipe, active leak from source.
  • Rule: water not going down → drainage_backup. Water coming out of pipe → pipe_leak.

slip_trip vs pavement_damage:
  • slip_trip: Any surface hazard that could cause someone to slip/fall — wet floor, spill, ice,
    snow, or frost on a walkway or entrance. Ice/snow on sidewalk or main entrance = slip_trip HIGH.
  • pavement_damage: Physical structural damage to pavement — cracks, potholes, crumbling concrete.
  • Rule: slippery surface (ice, wet, spill) → slip_trip. Broken/cracked pavement → pavement_damage.

=== CALLER SELF-CORRECTIONS ===
Callers often correct themselves mid-call. Always use the FINAL version of what was said.
- Issue corrections: "I think it's rodents… actually no, it's the trash smell" → classify as waste_odor, not infestation.
- Location corrections: "Floor 3… sorry, I mean Floor 5" → use Floor 5.
- Scope corrections: "half the suite… well, the whole area" → use the corrected scope.
The first thing a caller says is often a guess; their correction is the accurate report.

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


REWRITE_PROMPT = """You are rewriting the RAG query to be more specific to the current call.

Current RAG query:
{query}

Current call transcript:
{transcript}

Rewrite the query to be more specific to the current call."""



# ---------------------------------------------------------------------------
# Node: intake_extract
# ---------------------------------------------------------------------------


def intake_extract(state: AgentState) -> dict:
    t0 = time.perf_counter()
    turns = state["turns"]
    transcript_text = "\n".join(
        f"[{t['speaker'].upper()}] {t['text']}" for t in turns
    )
    profile = CALLER_PROFILES.get(state.get("caller_phone") or "")
    dt = time.perf_counter() - t0
    return _merge_with_timing(
        state,
        "intake_extract",
        dt,
        {
            "transcript_text": transcript_text,
            "rag_query": transcript_text,
            "rag_rewrite_count": 0,
            "caller_profile": profile,
            "retrieved_records": [],
            "rag_top_score": None,
            "classification": None,
            "building_info": None,
            "vendor_id": None,
            "dispatched_emergency": False,
            "human_override": None,
            "final_decision": None,
            "rag_documents_relevant": None,
        },
    )


# ---------------------------------------------------------------------------
# Node: rag_retrieve
# ---------------------------------------------------------------------------


def rag_retrieve(state: AgentState) -> dict:
    t0 = time.perf_counter()
    q = (state.get("rag_query") or "").strip() or (state.get("transcript_text") or "")
    results = _VECTOR_STORE.similarity_search_with_score(q, k=5)
    records = []
    scores = []
    for doc, score in results:
        entry = dict(doc.metadata)
        entry["_text"] = doc.page_content
        records.append(entry)
        scores.append(score)
    top_score = min(scores) if scores else None
    dt = time.perf_counter() - t0
    return _merge_with_timing(state, "rag_retrieve", dt, {
        "retrieved_records": records,
        "rag_top_score": top_score,
    })


# L2 distance threshold: docs with score below this are considered clearly relevant,
# so we skip the LLM grader call entirely.
# Empirically, all retrieved docs fall in [0.55, 0.70] for this corpus — scores
# below ~0.63 indicate a strong domain match; above it the grader inspects further.
_RAG_RELEVANCE_SCORE_THRESHOLD = 0.63


def _route_after_retrieve(state: AgentState) -> str:
    """Skip the LLM grader when the top retrieved doc is already a strong match."""
    top_score = state.get("rag_top_score")
    if top_score is not None and top_score < _RAG_RELEVANCE_SCORE_THRESHOLD:
        # High similarity — docs are clearly on-topic, no need to grade.
        return "classify_llm"
    # Low / uncertain similarity — let the LLM grader decide, then possibly rewrite.
    return "grader_gate"

# ---------------------------------------------------------------------------
# Node: grader_gate
# ---------------------------------------------------------------------------


def grader_gate(state: AgentState) -> dict:
    """Grade whether retrieved Chroma docs are useful few-shot context (before classify_llm)."""
    t0 = time.perf_counter()
    records = state.get("retrieved_records") or []
    if not records:
        dt = time.perf_counter() - t0
        return _merge_with_timing(
            state, "grader_gate", dt, {"rag_documents_relevant": True}
        )

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

    dt = time.perf_counter() - t0
    return _merge_with_timing(
        state, "grader_gate", dt, {"rag_documents_relevant": relevant}
    )


def _route_after_grader(state: AgentState) -> str:
    if state.get("rag_documents_relevant") is not False:
        return "classify_llm"
    if (state.get("rag_rewrite_count") or 0) < _MAX_RAG_REWRITE_ATTEMPTS:
        return "rewrite_rag_query"
    return "classify_llm"


# ---------------------------------------------------------------------------
# Node: rewrite_rag_query
# ---------------------------------------------------------------------------
def rewrite_rag_query(state: AgentState) -> dict:
    """LLM-rewritten retrieval string; always bump rag_rewrite_count to avoid infinite loops."""
    t0 = time.perf_counter()
    prior = state.get("rag_rewrite_count") or 0
    next_count = prior + 1
    query = (state.get("rag_query") or "").strip() or (state.get("transcript_text") or "")
    transcript = state.get("transcript_text") or ""
    user_content = REWRITE_PROMPT.format(query=query, transcript=transcript)
    try:
        structured = _LLM.with_structured_output(RewriteRAGQuery)
        result: RewriteRAGQuery = structured.invoke(
            [
                SystemMessage(
                    content="You rewrite text for vector search only; do not invent facts not supported by the transcript."
                ),
                HumanMessage(content=user_content),
            ]
        )
        new_q = (result.query or "").strip() or query
    except Exception:
        new_q = query
    dt = time.perf_counter() - t0
    return _merge_with_timing(
        state,
        "rewrite_rag_query",
        dt,
        {"rag_query": new_q, "rag_rewrite_count": next_count},
    )


# ---------------------------------------------------------------------------
# Node: classify_llm
# ---------------------------------------------------------------------------


def classify_llm(state: AgentState) -> dict:
    t0 = time.perf_counter()
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

    # Always overwrite with canonical name/address when we resolved the building —
    # handles partial matches, hallucinated names, and profile fallbacks.
    if building_info:
        classification["building_name"] = building_info["name"]
        classification["address"] = building_info["address"]

    # Floor fallback: profile default if transcript gave nothing
    if not classification.get("floor") and profile and profile.get("primary_floor"):
        classification["floor"] = profile["primary_floor"]

    dt = time.perf_counter() - t0
    return _merge_with_timing(
        state,
        "classify_llm",
        dt,
        {"classification": classification, "building_info": building_info},
    )


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
    t0 = time.perf_counter()
    classification = dict(state["classification"])
    risk = classification.get("risk_level", "LOW")
    subcategory = classification.get("subcategory", "")
    # Full transcript (for detecting escalation evidence in any turn)
    tx = (state.get("transcript_text") or "").lower()
    # Caller-only text (for downgrade rules — agent questions must not trigger them)
    caller_tx = " ".join(
        t.get("text", "").lower()
        for t in (state.get("turns") or [])
        if t.get("speaker") == "caller"
    )

    # ── Step 0: Subcategory corrections ────────────────────────────────────
    # These run first so subsequent risk rules use the correct subcategory.

    # gas_chemical → air_quality when caller never confirmed gas or sulfur smell,
    # OR when caller explicitly identifies a benign source ("smells like gas but it's nail polish").
    _NON_GAS_SOURCES = ["nail polish", "acetone", "cleaning fluid", "bleach", "perfume",
                         "candle", "burnt food", "burnt toast", "coffee", "marker", "paint"]
    if subcategory == "gas_chemical":
        no_gas_confirmed = not any(p in caller_tx for p in ["gas", "sulfur", "rotten egg", "natural gas", "propane"])
        benign_source_identified = any(p in caller_tx for p in _NON_GAS_SOURCES)
        if no_gas_confirmed or benign_source_identified:
            classification["subcategory"] = "air_quality"
            classification["category"] = "HVAC"
            subcategory = "air_quality"
            if risk == "EMERGENCY":
                classification["risk_level"] = "MEDIUM"
                risk = "MEDIUM"

    # power_outage → panel_hazard when caller reports burning smell near electrical.
    # Caller self-corrections ("this isn't just a power outage") signal a reclassification.
    if subcategory == "power_outage":
        has_burn = any(p in caller_tx for p in ["burn", "burning", "burnt"])
        has_electrical = any(p in caller_tx for p in ["electrical", "closet", "panel", "box", "breaker"])
        if has_burn and has_electrical:
            classification["subcategory"] = "panel_hazard"
            subcategory = "panel_hazard"

    # structural → roof_leak when caller explicitly says "nothing structural" / ceiling tile / water damage.
    if subcategory == "structural":
        if any(p in caller_tx for p in ["nothing structural", "not structural", "ceiling tile",
                                         "drop-ceiling", "water-damaged", "water damage",
                                         "just a leak", "just the tile", "just water"]):
            classification["subcategory"] = "roof_leak"
            classification["category"] = "PLUMBING"
            subcategory = "roof_leak"
            if risk == "HIGH":
                classification["risk_level"] = "MEDIUM"
                risk = "MEDIUM"

    # pipe_leak → roof_leak when caller describes dripping from ceiling / above.
    if subcategory == "pipe_leak":
        if any(p in caller_tx for p in ["from up top", "from the ceiling", "from above",
                                         "ceiling is leaking", "ceiling drip", "roof is leaking",
                                         "coming from the ceiling"]):
            classification["subcategory"] = "roof_leak"
            subcategory = "roof_leak"

    # drainage_backup → restroom_fixture when caller reports a sink/toilet in a restroom.
    if subcategory == "drainage_backup":
        if any(p in tx for p in ["restroom", "bathroom", "washroom", "lavatory"]):
            if any(p in caller_tx for p in ["sink", "toilet", "commode", "basin", "fixture"]):
                classification["subcategory"] = "restroom_fixture"
                classification["category"] = "PLUMBING"
                subcategory = "restroom_fixture"
                classification["risk_level"] = "LOW"
                risk = "LOW"

    # air_quality → waste_odor when the caller identifies a physical source (burnt food/candle).
    if subcategory == "air_quality":
        if any(p in caller_tx for p in ["burnt toast", "burned toast", "burnt food", "burned food",
                                         "candle", "toaster", "microwave", "cooking smell"]):
            classification["subcategory"] = "waste_odor"
            classification["category"] = "JANITORIAL"
            subcategory = "waste_odor"
            if risk not in ("LOW",):
                classification["risk_level"] = "LOW"
                risk = "LOW"

    # no_cooling → refrigerant when AC is still running but performance is degraded.
    # "Still running but not cold / performance dropped" = refrigerant depletion, not system failure.
    if subcategory == "no_cooling":
        running = any(p in caller_tx for p in ["still running", "still on", "running but",
                                                 "unit is running", "still working"])
        degraded = any(p in caller_tx for p in ["not cold", "not as cold", "performance drop",
                                                  "performance dropped", "not efficient",
                                                  "acting funny", "acting weird"])
        if running and degraded:
            classification["subcategory"] = "refrigerant"
            subcategory = "refrigerant"
            classification["risk_level"] = "LOW"
            risk = "LOW"

    # slip_trip category fix: the LLM occasionally picks GROUNDS_EXTERIOR.
    # slip_trip belongs to LIFE_SAFETY (outdoor/high-risk) or JANITORIAL (indoor).
    if subcategory == "slip_trip":
        if classification.get("category") not in ("LIFE_SAFETY", "JANITORIAL"):
            if any(p in tx for p in ["ice", "icy", "snow", "frost", "outside", "exterior",
                                      "sidewalk", "entrance", "parking"]) or risk == "HIGH":
                classification["category"] = "LIFE_SAFETY"
            else:
                classification["category"] = "JANITORIAL"

    # Category consistency: enforce category matches subcategory via taxonomy.
    # Handles LLM anchoring on the initial wrong category after a self-correction
    # (e.g., waste_odor with PEST_SPECIALTY, or fire_smoke category after sub changes to waste_odor).
    _correct_cat = _CANONICAL_CATEGORY.get(subcategory)
    if _correct_cat and classification.get("category") != _correct_cat:
        classification["category"] = _correct_cat

    # ── Step 1: Rule-based risk correction ─────────────────────────────────
    # The LLM persistently over-escalates certain subcategories despite prompt
    # warnings. Downgrade rules use caller_tx so the agent's questions don't
    # falsely trigger the phrases (e.g. agent asking "is water overflowing?").

    # pipe_leak HIGH → MEDIUM unless caller confirms multi-area flooding or electrical threat
    if subcategory == "pipe_leak" and risk == "HIGH":
        if not any(p in caller_tx for p in ["multiple floor", "multiple room", "flooding the",
                                             "electrical", "burst", "several floor", "spreading to"]):
            classification["risk_level"] = "MEDIUM"
            risk = "MEDIUM"

    # power_outage HIGH → MEDIUM unless caller confirms building-wide outage, critical system,
    # or a repeatedly tripping breaker (which indicates an ongoing electrical fault, not a one-off).
    # "whole floor" / "entire floor" are intentionally excluded: callers use them to mean their
    # suite's floor space ("the whole of our area"), not the building floor.
    if subcategory == "power_outage" and risk == "HIGH":
        if not any(p in caller_tx for p in ["entire building", "whole building", "building-wide",
                                             "server room", "all floors", "multiple floor",
                                             "everywhere", "critical system", "all suite",
                                             "all offices", "tripping", "keeps tripping",
                                             "keep tripping", "trip again"]):
            classification["risk_level"] = "MEDIUM"
            risk = "MEDIUM"

    # glass_damage HIGH → MEDIUM unless caller confirms a person was actually injured or glass fell on someone.
    # "no injuries" / "no one injured" must NOT keep it HIGH — check for positive injury language only.
    if subcategory == "glass_damage" and risk == "HIGH":
        has_injury = any(p in caller_tx for p in ["someone was injured", "person injured",
                                                    "got cut", "cut by glass", "was hit",
                                                    "fell on", "structural damage"]) and \
                     not any(p in caller_tx for p in ["no injur", "no one injur", "nobody injur",
                                                       "no injuries", "not injured"])
        if not has_injury:
            classification["risk_level"] = "MEDIUM"
            risk = "MEDIUM"

    # roof_leak HIGH → MEDIUM unless actively spreading to multiple areas or structural collapse.
    # A simple ceiling drip is MEDIUM by default; we also reclassify pipe_leak→roof_leak in Step 0
    # which means the original HIGH (from pipe_leak over-escalation) won't be caught by the
    # pipe_leak downgrade rule below — this rule picks it up.
    if subcategory == "roof_leak" and risk == "HIGH":
        if not any(p in caller_tx for p in ["spreading", "multiple room", "multiple area",
                                              "everywhere", "structural damage", "ceiling collapse",
                                              "floors below", "another floor"]):
            classification["risk_level"] = "MEDIUM"
            risk = "MEDIUM"

    # drainage_backup MEDIUM → LOW unless the CALLER confirms active overflow/flooding
    if subcategory == "drainage_backup" and risk == "MEDIUM":
        if not any(p in caller_tx for p in ["overflow", "overflowing", "flooding",
                                             "spilling", "water all over"]):
            classification["risk_level"] = "LOW"
            risk = "LOW"

    # infestation MEDIUM → LOW unless widespread or confirmed health hazard
    if subcategory == "infestation" and risk == "MEDIUM":
        if not any(p in caller_tx for p in ["widespread", "health hazard", "swarming",
                                             "colony", "everywhere", "making people sick",
                                             "nauseated", "sick from"]):
            classification["risk_level"] = "LOW"
            risk = "LOW"

    # waste_odor MEDIUM → LOW unless causing illness, affecting large/common area, or in a hallway.
    if subcategory == "waste_odor" and risk == "MEDIUM":
        if not any(p in caller_tx for p in ["making people sick", "people are sick",
                                             "nauseated", "health hazard", "throughout",
                                             "whole building", "entire floor"]) and \
           not any(p in tx for p in ["hallway", "corridor", "common area", "lobby",
                                      "shared space", "main area"]):
            classification["risk_level"] = "LOW"
            risk = "LOW"

    # access_control MEDIUM → LOW: badge/keypad failure is always routine unless security breach confirmed
    if subcategory == "access_control" and risk == "MEDIUM":
        if not any(p in caller_tx for p in ["breach", "broke in", "forced entry", "unauthorized",
                                             "intruder", "tailgate", "someone got in"]):
            classification["risk_level"] = "LOW"
            risk = "LOW"

    # no_heating MEDIUM → LOW for routine single-suite complaints. Keep MEDIUM if HVAC actively
    # malfunctioning (blowing cold air when set to heat), multi-suite, medical, or extreme cold.
    if subcategory == "no_heating" and risk == "MEDIUM":
        has_severity = any(p in caller_tx for p in [
            "blowing cold", "pushing cold", "cold air coming", "medical", "hospital", "patient",
            "clinic", "freezing", "frozen", "pipes", "dangerous", "health", "multiple suite",
            "entire floor", "whole floor", "all suite", "every office",
        ])
        if not has_severity:
            classification["risk_level"] = "LOW"
            risk = "LOW"

    # no_cooling MEDIUM → LOW for routine single-suite complaints. Keep MEDIUM if multi-suite,
    # HVAC malfunction, medical, or extreme-heat health risk.
    if subcategory == "no_cooling" and risk == "MEDIUM":
        has_severity = any(p in caller_tx for p in [
            "medical", "hospital", "patient", "clinic", "faint", "heat stroke", "dangerous",
            "health hazard", "multiple suite", "entire floor", "whole floor", "all suite",
            "every office", "blowing hot", "pushing hot",
        ])
        if not has_severity:
            classification["risk_level"] = "LOW"
            risk = "LOW"

    # signage_fencing LOW → MEDIUM when the fence/sign is structurally unsafe (leaning badly).
    if subcategory == "signage_fencing" and risk == "LOW":
        if any(p in caller_tx for p in ["leaning", "lean", "about to fall", "toppling",
                                         "falling over", "unstable", "collapsed"]):
            classification["risk_level"] = "MEDIUM"
            risk = "MEDIUM"

    # landscaping is always LOW — irrigation runoff, plant issues, etc. are routine grounds work.
    if subcategory == "landscaping" and risk == "MEDIUM":
        classification["risk_level"] = "LOW"
        risk = "LOW"

    # restroom_fixture LOW → MEDIUM when a fixture is continuously running (water damage risk).
    if subcategory == "restroom_fixture" and risk == "LOW":
        if any(p in caller_tx for p in ["won't stop running", "keeps running", "running constantly",
                                         "can't stop it", "non-stop", "running all night"]):
            classification["risk_level"] = "MEDIUM"
            risk = "MEDIUM"

    # slip_trip MEDIUM → HIGH when outdoor ice/snow creates hazard at building entrance.
    # Ice on sidewalk or main entrance = immediate large-scale public hazard.
    if subcategory == "slip_trip" and risk == "MEDIUM":
        if any(p in caller_tx for p in ["ice", "icy", "frozen", "snow", "frost"]):
            if any(p in tx for p in ["sidewalk", "entrance", "outside", "exterior",
                                      "parking", "main door", "front door", "front entrance"]):
                classification["risk_level"] = "HIGH"
                risk = "HIGH"

    # slip_trip MEDIUM → LOW when hazard is already being attended
    if subcategory == "slip_trip" and risk == "MEDIUM":
        if any(p in caller_tx for p in ["already cleaned", "cleaning it", "cleaned up",
                                         "put up a sign", "caution sign", "wet floor sign",
                                         "already mopped", "mopping it",
                                         "cone", "put a cone", "set up a cone", "cones up"]):
            classification["risk_level"] = "LOW"
            risk = "LOW"

    # entrapment EMERGENCY → HIGH when occupants are communicating safely (not imminent danger)
    if subcategory == "entrapment" and risk == "EMERGENCY":
        resolved = any(p in tx for p in ["nobody was in", "no one was in", "got out",
                                          "out of the elevator", "they're out", "made it out",
                                          "already out", "evacuated", "they got out"])
        still_trapped = any(p in caller_tx for p in ["still inside", "still trapped",
                                                      "can't get out", "stuck inside",
                                                      "door won't open"])
        communicating_safely = any(p in tx for p in ["pressing the call button", "press the call button",
                                                       "call button", "banging", "knocking",
                                                       "can hear them", "i can hear"])
        no_medical = not any(p in caller_tx for p in ["injur", "hurt", "medical", "can't breathe",
                                                        "unconscious", "heart", "diabetic"])
        if (resolved and not still_trapped) or (communicating_safely and no_medical and not still_trapped):
            classification["risk_level"] = "HIGH"
            risk = "HIGH"

    # panel_hazard EMERGENCY → HIGH unless active arcing, fire, or burning smell confirmed by caller
    if subcategory == "panel_hazard" and risk == "EMERGENCY":
        if not any(p in caller_tx for p in ["fire", "arcing", "smoking", "sparking now",
                                             "on fire", "flames", "burn", "burning smell",
                                             "smells like burn", "electrical smell"]):
            classification["risk_level"] = "HIGH"
            risk = "HIGH"

    # unauthorized_access EMERGENCY → HIGH unless weapon/active violence in caller text
    if subcategory == "unauthorized_access" and risk == "EMERGENCY":
        if not any(p in caller_tx for p in ["weapon", "gun", "knife", "shooting",
                                             "stabbing", "attacking"]):
            classification["risk_level"] = "HIGH"
            risk = "HIGH"

    # auto_door MEDIUM/HIGH → LOW: automatic door malfunction is always routine.
    # Even a blocked fire-exit auto_door is LOW — manual push-open override is always available.
    if subcategory == "auto_door" and risk in ("MEDIUM", "HIGH"):
        classification["risk_level"] = "LOW"
        risk = "LOW"

    # refrigerant MEDIUM → LOW: refrigerant service is routine HVAC maintenance.
    if subcategory == "refrigerant" and risk == "MEDIUM":
        if not any(p in caller_tx for p in ["hissing", "hiss", "health", "sick", "evacuate",
                                              "spreading", "fire", "emergency"]):
            classification["risk_level"] = "LOW"
            risk = "LOW"

    # suspicious_person HIGH → MEDIUM unless physical threat, forced entry, OR social engineering
    # (asking for badge numbers / credentials) OR confirmed no-badge in secured area.
    if subcategory == "suspicious_person" and risk == "HIGH":
        if not any(p in caller_tx for p in ["threatening", "threatened", "weapon", "gun", "knife",
                                             "hitting", "attacking", "forced", "broke in",
                                             "breaking in", "violence", "violent",
                                             "badge number", "asking for badge", "no badge",
                                             "without a badge", "doesn't have a badge",
                                             "requesting access", "credentials"]):
            classification["risk_level"] = "MEDIUM"
            risk = "MEDIUM"

    # suspicious_person MEDIUM → HIGH when caller reports confirmed no-badge / social engineering.
    # The LLM sometimes outputs MEDIUM directly for these cases (bypass of the downgrade above).
    if subcategory == "suspicious_person" and risk == "MEDIUM":
        if any(p in caller_tx for p in ["no badge", "without a badge", "doesn't have a badge",
                                         "don't have a badge", "asking for badge",
                                         "badge number", "asking employees", "no id",
                                         "without id"]):
            classification["risk_level"] = "HIGH"
            risk = "HIGH"

    # air_quality EMERGENCY → MEDIUM unless caller explicitly confirms gas, sulfur, toxic fumes,
    # or active evacuation. "Chemical smell" / "people with headaches" alone = MEDIUM, not EMERGENCY.
    if subcategory == "air_quality" and risk == "EMERGENCY":
        if not any(p in caller_tx for p in ["gas", "sulfur", "rotten egg", "evacuate", "evacuating",
                                             "toxic fume", "chemical release", "chemical leak",
                                             "spreading", "overcome", "unconscious"]):
            classification["risk_level"] = "MEDIUM"
            risk = "MEDIUM"

    # ── Step 2: HITL policy ─────────────────────────────────────────────────
    # EMERGENCY/HIGH always need human sign-off.
    # MEDIUM: subcategory-specific rules below; default False to prevent LLM over-firing.
    # LOW: never (except special signal rules below).

    if risk in ("EMERGENCY", "HIGH"):
        needs_review = True
    elif risk == "MEDIUM":
        needs_review = False  # default; overridden by subcategory rules below

        # malfunction: review only when someone was ever inside the stalled elevator.
        if subcategory == "malfunction":
            needs_review = any(p in tx for p in [
                "someone inside", "person inside", "someone in it", "person in it",
                "with someone inside", "they're out", "they got out", "made it out",
                "was in the elevator", "was inside", "stuck with",
            ])

        # suspicious_person MEDIUM: review only when caller reports active confrontation
        # (yelling/arguing). Passive loitering → auto-resolve.
        if subcategory == "suspicious_person":
            needs_review = any(p in caller_tx for p in [
                "yelling", "yelled", "shouting", "shouted", "arguing", "argument",
                "confrontation", "heated", "getting aggressive", "threatening behaviour",
            ])

    else:  # LOW
        needs_review = False

    # ── Special signal overrides (any risk level) ───────────────────────────
    # Refrigerant actively leaking or audible hissing → needs verification.
    # Exclude past-tense/secondhand reports (e.g. "tech said there was a refrigerant leak last week").
    if subcategory == "refrigerant":
        stale_report = any(p in caller_tx for p in ["last week", "last month", "tech said",
                                                      "technician said", "said it was", "mentioned"])
        has_active_leak = (
            any(p in caller_tx for p in ["leaking refrigerant", "refrigerant leak", "refrigerant is leaking"])
            and not stale_report
        )
        if has_active_leak or "hissing" in caller_tx:
            needs_review = True

    # Cleaning chemical / bleach smell in a shared area → HITL even when source is identified.
    # The over-escalation trap: don't call it gas_chemical, but still verify chemical exposure.
    if subcategory == "air_quality" and risk == "MEDIUM":
        if any(p in caller_tx for p in ["bleach", "cleaning fluid", "cleaning solution",
                                         "cleaning chemical", "ammonia", "disinfectant"]):
            needs_review = True

    # Smoke alarm triggered but classified below fire_smoke → human must verify all-clear.
    if subcategory not in ("fire_smoke", "gas_chemical") and risk in ("LOW", "MEDIUM"):
        if any(p in tx for p in ["smoke alarm", "fire alarm beeped", "alarm went off",
                                   "alarm triggered", "alarm beeped", "alarm sounded"]):
            needs_review = True

    # Visible smoke confirmed by caller → human must verify, even if source is benign (candle/toast).
    if subcategory not in ("fire_smoke", "gas_chemical") and risk in ("LOW", "MEDIUM"):
        if any(p in caller_tx for p in ["see smoke", "there's smoke", "seeing smoke",
                                         "visible smoke", "smoke coming", "smoke in the"]):
            needs_review = True

    # Roof leak with a physically fallen ceiling piece → physical hazard warrants verification.
    # "ceiling tile" alone is too broad (just water dripping through tile = normal roof_leak).
    if subcategory == "roof_leak" and risk == "MEDIUM":
        if any(p in caller_tx for p in ["fell", "fallen", "collapse", "came down",
                                         "piece of ceiling", "tile fell", "ceiling fell"]):
            needs_review = True

    classification["needs_human_review"] = needs_review

    dt = time.perf_counter() - t0
    return _merge_with_timing(
        state,
        "validator_gate",
        dt,
        {"classification": classification},
    )


# ---------------------------------------------------------------------------
# Node: hitl_review
# ---------------------------------------------------------------------------


def hitl_review(state: AgentState) -> dict:
    """Node: pause for conditional human review and apply approve/override."""
    t0 = time.perf_counter()
    classification = dict(state["classification"])
    needs_review = bool(classification.get("needs_human_review", False))
    original_classification = dict(classification)

    human_override: Optional[dict] = None
    hitl_decision: Dict[str, Any] = {
        "status": "auto_routed",
        "reason": "No human review required",
    }
    routing_decision = (
        f"Auto-routed -> {classification.get('subcategory', 'unknown')} "
        f"(risk {classification.get('risk_level', 'LOW')})"
    )

    if needs_review:
        review_reason = f"risk={classification.get('risk_level', 'LOW')}"
        reviewer_input = interrupt({
            "type": "review_required",
            "reason": review_reason,
            "transcript": state["transcript_text"],
            "ai_classification": classification,
        })

        # Approve path:
        # - batch eval sentinel: "approved"
        # - interactive payload: {"approved": True}
        approved = (
            reviewer_input == "approved"
            or (isinstance(reviewer_input, dict) and reviewer_input.get("approved") is True)
        )

        if approved:
            hitl_decision = {"status": "approved", "reason": review_reason}
            routing_decision = (
                f"Human approved -> {classification.get('subcategory', 'unknown')} "
                f"(risk {classification.get('risk_level', 'LOW')})"
            )
        elif isinstance(reviewer_input, dict) and reviewer_input:
            # Control keys are metadata, not direct classification fields.
            key_aliases = {
                "override_category": "category",
                "override_subcategory": "subcategory",
                "override_risk_level": "risk_level",
                "override_building_name": "building_name",
                "override_address": "address",
                "override_floor": "floor",
            }
            override_payload: Dict[str, Any] = {}
            for key, value in reviewer_input.items():
                if key in {"approved", "decision", "action", "override_reason", "override_code"}:
                    continue
                target_key = key_aliases.get(key, key)
                override_payload[target_key] = value

            # Support a compact override contract.
            # In this codebase we route by taxonomy fields rather than a single selected code.
            if reviewer_input.get("override_code") and not override_payload.get("subcategory"):
                override_payload["subcategory"] = reviewer_input["override_code"]

            if override_payload:
                classification.update(override_payload)
                classification["needs_human_review"] = False
                human_override = reviewer_input
                hitl_decision = {
                    "status": "overridden",
                    "reason": review_reason,
                    "override_reason": reviewer_input.get("override_reason"),
                }
                routing_decision = (
                    f"Human override -> {classification.get('subcategory', 'unknown')} "
                    f"(risk {classification.get('risk_level', 'LOW')})"
                )
            else:
                # Safety fallback: no concrete overrides supplied.
                hitl_decision = {"status": "approved", "reason": review_reason}
                routing_decision = (
                    f"Human approved -> {classification.get('subcategory', 'unknown')} "
                    f"(risk {classification.get('risk_level', 'LOW')})"
                )
        else:
            # Defensive fallback for unexpected resume payloads.
            hitl_decision = {"status": "approved", "reason": review_reason}
            routing_decision = (
                f"Human approved -> {classification.get('subcategory', 'unknown')} "
                f"(risk {classification.get('risk_level', 'LOW')})"
            )

    if classification != original_classification:
        hitl_decision["original_classification"] = original_classification

    dt = time.perf_counter() - t0
    return _merge_with_timing(
        state,
        "hitl_review",
        dt,
        {
            "classification": classification,
            "human_override": human_override,
            "hitl_decision": hitl_decision,
            "routing_decision": routing_decision,
        },
    )


# ---------------------------------------------------------------------------
# Node: vendor_select
# ---------------------------------------------------------------------------


def vendor_select(state: AgentState) -> dict:
    t0 = time.perf_counter()
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

    dt = time.perf_counter() - t0
    return _merge_with_timing(
        state,
        "vendor_select",
        dt,
        {
            "vendor_id": vendor_id,
            "dispatched_emergency": dispatched_emergency,
            "classification": updated_classification,
        },
    )


# ---------------------------------------------------------------------------
# Node: log_result
# ---------------------------------------------------------------------------


def log_result(state: AgentState) -> dict:
    t0 = time.perf_counter()
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
        "hitl_decision": state.get("hitl_decision"),
        "routing_decision": state.get("routing_decision"),
    }
    dt = time.perf_counter() - t0
    return _merge_with_timing(state, "log_result", dt, {"final_decision": final_decision})


# ---------------------------------------------------------------------------
# Graph construction — compiled once at module import
# ---------------------------------------------------------------------------


def _build_graph() -> Any:
    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("intake_extract", intake_extract)
    builder.add_node("rag_retrieve", rag_retrieve)
    builder.add_node("grader_gate", grader_gate)
    builder.add_node("rewrite_rag_query", rewrite_rag_query)
    builder.add_node("classify_llm", classify_llm)
    builder.add_node("validator_gate", validator_gate)
    builder.add_node("hitl_review", hitl_review)
    builder.add_node("vendor_select", vendor_select)
    builder.add_node("log_result", log_result)

    builder.set_entry_point("intake_extract")
    builder.add_edge("intake_extract", "rag_retrieve")
    # Route by similarity score: strong match → skip grader; weak match → grader decides
    builder.add_conditional_edges(
        "rag_retrieve",
        _route_after_retrieve,
        {"classify_llm": "classify_llm", "grader_gate": "grader_gate"},
    )
    builder.add_conditional_edges(
        "grader_gate",
        _route_after_grader,
        {"classify_llm": "classify_llm", "rewrite_rag_query": "rewrite_rag_query"},
    )
    builder.add_edge("rewrite_rag_query", "rag_retrieve")
    builder.add_edge("classify_llm", "validator_gate")
    builder.add_edge("validator_gate", "hitl_review")
    builder.add_edge("hitl_review", "vendor_select")
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

    Each row also includes ``_latency`` (wall clock + per-node timings) for dev
    analysis; strip before official submission if your grader forbids extra keys.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "turns": turns,
        "caller_phone": caller_phone,
        "transcript_text": "",
        "rag_query": "",
        "rag_rewrite_count": 0,
        "caller_profile": None,
        "retrieved_records": [],
        "rag_top_score": None,
        "classification": None,
        "building_info": None,
        "vendor_id": None,
        "dispatched_emergency": False,
        "human_override": None,
        "hitl_decision": None,
        "routing_decision": None,
        "final_decision": None,
        "rag_documents_relevant": None,
        "node_timings": [],
    }

    wall_t0 = time.perf_counter()
    # First invocation — may pause at hitl_review if needs_human_review
    _GRAPH.invoke(initial_state, config)

    # Handle interrupt: in batch mode resume immediately with no override.
    # Command(resume=None) is broken in LangGraph 1.1.4 — use sentinel string instead.
    graph_state = _GRAPH.get_state(config)
    if graph_state.next:
        _GRAPH.invoke(Command(resume="approved"), config)

    wall_seconds = time.perf_counter() - wall_t0

    final_state = _GRAPH.get_state(config).values
    classification = dict(final_state.get("classification") or {})
    ai_prediction = dict(classification)

    raw_text = " ".join(t.get("text", "") for t in turns)

    # Post-processing: clarification signals.
    _clarif_patterns = [
        r"there'?s an issue with",          # vague template phrasing
        r"only has \d+ floor",              # agent catches floor-count conflict
        r"floor\s*1\b.{0,60}\bground\s*floor\b",  # floor-1 vs ground-floor conflict
        r"ground\s*floor\b.{0,60}\bfloor\s*1\b",
    ]
    if any(re.search(p, raw_text, re.I) for p in _clarif_patterns):
        classification["needs_clarification"] = True

    # Location-impossibility check: a high floor number paired with an outdoor ground-level
    # area (lawn, garden, parking exterior) is physically impossible → flag for clarification.
    _floor_match = re.search(r"\bfloor\s*(\d+)\b", raw_text, re.I)
    _outdoor_terms = ["lawn", "grass", "garden", "flooding the lawn", "parking lot exterior",
                      "exterior sprinkler", "outside sprinkler", "landscaping outside"]
    if _floor_match and int(_floor_match.group(1)) >= 5:
        if any(t in raw_text.lower() for t in _outdoor_terms):
            classification["needs_clarification"] = True

    # Post-processing: floor normalization.
    floor_val = classification.get("floor") or ""
    if re.match(r"^floor\s*(g|0)$", floor_val, re.I):
        classification["floor"] = "Ground Floor"
    elif re.match(r"^ground\s*floor$", floor_val, re.I):
        classification["floor"] = "Ground Floor"

    timings = list(final_state.get("node_timings") or [])
    by_node: Dict[str, float] = {}
    for ev in timings:
        name = ev.get("node") or "unknown"
        by_node[name] = by_node.get(name, 0.0) + float(ev.get("seconds") or 0.0)

    sum_timed_nodes = sum(by_node.values())

    latency_payload = {
        "wall_clock_seconds": round(wall_seconds, 4),
        "sum_timed_nodes_seconds": round(sum_timed_nodes, 4),
        "wall_minus_sum_nodes": round(wall_seconds - sum_timed_nodes, 4),
        "node_timings": timings,
        "seconds_by_node": {k: round(v, 4) for k, v in sorted(by_node.items())},
        "grader_gate_visits": sum(1 for e in timings if e.get("node") == "grader_gate"),
        "rewrite_rag_query_visits": sum(1 for e in timings if e.get("node") == "rewrite_rag_query"),
        "rag_retrieve_visits": sum(1 for e in timings if e.get("node") == "rag_retrieve"),
        "final_rag_rewrite_count": final_state.get("rag_rewrite_count") or 0,
    }

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
            "hitl_decision":   final_state.get("hitl_decision"),
            "routing_decision": final_state.get("routing_decision"),
            "was_overridden": bool(final_state.get("human_override")),
            "final_decision":  final_state.get("final_decision") or {},
        },
        "_latency": latency_payload,
    }
