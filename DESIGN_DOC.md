# CBRE Call Intake — HITL-RAG Agent Design Document

**Project:** Human-in-the-Loop RAG Agent for CBRE Facilities Call Intake

**By: Yash Chanchani and Mayank Kumar** 

**Dev-set composite score:** 95.23 / 100  
**Final test score:** run via `python evaluation/scoring.py --predictions predictions.json`

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Pipeline Nodes (Deep Dive)](#3-pipeline-nodes-deep-dive)
4. [Risk-Band Cutoffs — Corpus Derivation](#4-risk-band-cutoffs--corpus-derivation)
5. [HITL Policy](#5-hitl-policy)
6. [Vendor-Selection Algorithm](#6-vendor-selection-algorithm)
7. [Clarification Policy](#7-clarification-policy)
8. [Voice Demo Extension](#8-voice-demo-extension)
9. [Confusion Matrix & Error Analysis](#9-confusion-matrix--error-analysis)
10. [Dev-Set Score Breakdown](#10-dev-set-score-breakdown)
11. [Trainer Log](#11-trainer-log)
12. [Design Decisions & Trade-offs](#12-design-decisions--trade-offs)

---

## 1. System Overview

CBRE's property portfolio receives approximately 10,000 maintenance calls per day. Every call historically required a human operator to triage, classify, and dispatch — a bottleneck that led to operator fatigue and inconsistent triage quality.

This agent replaces the human operator for routine calls while preserving human judgment exactly where it matters. The system:

- Listens to a structured multi-turn caller dialogue
- Classifies the issue into `(category, subcategory)` from a 10-category, 36-subcategory taxonomy with a risk level (LOW / MEDIUM / HIGH / EMERGENCY)
- Extracts location (building + floor), reconciled against known caller profiles
- Decides between three outcomes: **auto-resolve + dispatch vendor**, **ask one clarifying question**, or **hand off to a human reviewer** (the HITL gate)
- Selects a qualified vendor from a 32-vendor roster
- Emits a compact operator-style `call_summary` and a structured `trainer_log`

A separate live voice demo wraps the same `classify()` function with a real-time telephony loop: Twilio receives the call, Deepgram transcribes speech, the agent classifies, and ElevenLabs synthesizes the spoken reply.

---

## 2. Architecture

### 2.1 High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CBRE Agent Pipeline                          │
│                    (LangGraph StateGraph, InMemorySaver)             │
│                                                                      │
│   ┌──────────────┐    ┌─────────────┐    ┌──────────────────────┐   │
│   │intake_extract│───►│ rag_retrieve │───►│  _route_after_       │   │
│   │              │    │             │    │   retrieve()          │   │
│   │ - flatten    │    │ - Chroma    │    │                       │   │
│   │   transcript │    │   k=5 docs  │    │  score < 0.63?        │   │
│   │ - lookup     │    │ - L2 score  │    │  ┌────────────────┐  │   │
│   │   profile    │    │             │    │  │ YES → classify  │  │   │
│   └──────────────┘    └─────────────┘    │  │ NO  → grader   │  │   │
│                                          │  └────────────────┘  │   │
│                                          └──────────────────────┘   │
│                                                    │                 │
│                          ┌─────────────────────────┘                │
│                          ▼                                           │
│                  ┌───────────────┐    NOT relevant?                  │
│                  │  grader_gate  │──────────────────►┐               │
│                  │               │    (max 2 loops)   │               │
│                  │ LLM grades    │                   ▼               │
│                  │ retrieved     │           ┌───────────────┐       │
│                  │ docs for      │           │rewrite_rag_   │       │
│                  │ relevance     │           │query          │       │
│                  └───────────────┘           │               │       │
│                          │                   │ LLM rewrites  │       │
│                          │ relevant          │ search string │       │
│                          │                   └───────┬───────┘       │
│                          ▼                           │               │
│                  ┌───────────────┐ ◄────────────────┘               │
│                  │ classify_llm  │                                   │
│                  │               │                                   │
│                  │ GPT-4o-mini   │                                   │
│                  │ structured    │                                   │
│                  │ output        │                                   │
│                  │ CallClassif.  │                                   │
│                  └───────┬───────┘                                   │
│                          │                                           │
│                          ▼                                           │
│                  ┌───────────────┐                                   │
│                  │validator_gate │                                   │
│                  │               │                                   │
│                  │ Rule-based    │                                   │
│                  │ corrections:  │                                   │
│                  │ - subcategory │                                   │
│                  │ - risk level  │                                   │
│                  │ - HITL flag   │                                   │
│                  └───────┬───────┘                                   │
│                          │                                           │
│                          ▼                                           │
│                  ┌───────────────┐   needs_human_review?             │
│                  │  hitl_review  │──────────────────►  interrupt()   │
│                  │               │   YES             (batch: auto-   │
│                  │ LangGraph     │                    resume)        │
│                  │ interrupt()   │   NO ──────────────────────┐      │
│                  └───────────────┘                            │      │
│                                                               ▼      │
│                                                     ┌──────────────┐ │
│                                                     │ vendor_select│ │
│                                                     │              │ │
│                                                     │ 4-tier       │ │
│                                                     │ fallback     │ │
│                                                     │ scoring      │ │
│                                                     └──────┬───────┘ │
│                                                            │         │
│                                                            ▼         │
│                                                     ┌──────────────┐ │
│                                                     │  log_result  │ │
│                                                     │              │ │
│                                                     │ final_decision│ │
│                                                     │ trainer_log  │ │
│                                                     └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Sources

| Source | Contents | Used By |
|--------|----------|---------|
| `historical_records.json` | 10,000 CMMS tickets: transcripts, intake labels, final labels, vendor assignment, timing | ChromaDB index (RAG), risk cutoff derivation |
| `buildings.json` | 52 buildings: name, address, city, building type | Location resolution, vendor city matching |
| `vendors.json` | 32 vendors: type, cost tier, SLA, specialties, coverage cities, availability status | `vendor_select` node |
| `caller_profiles.json` | ~800 caller phone→profile mappings with default building/floor | Location fallback in `classify_llm` |
| `qa_audit_findings.json` | 988 QA-sampled tickets flagged for reclassification or over-escalation | ChromaDB document metadata, risk-correction rules |

### 2.3 Technology Stack

| Component | Technology |
|-----------|-----------|
| Agent framework | LangGraph (StateGraph + InMemorySaver checkpointer) |
| LLM | GPT-4o-mini (temperature=0) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (L2 similarity) |
| Structured output | Pydantic `BaseModel` with `with_structured_output()` |
| Voice telephony | Twilio (TwiML webhooks) |
| STT | Deepgram Nova-2 |
| TTS | ElevenLabs Flash v2.5 |
| Web framework | FastAPI |

---

## 3. Pipeline Nodes (Deep Dive)

### 3.1 `intake_extract`

Flattens the structured `turns` list into a single transcript string formatted as `[CALLER] ... / [AGENT] ...`. Looks up the caller phone number in `caller_profiles.json` to load the default building/floor prior. Initialises all state fields to their zero values so downstream nodes always have consistent types.

### 3.2 `rag_retrieve`

Queries ChromaDB with `k=5` nearest neighbors using L2 distance. The index was built from all 10,000 historical records; each document embeds the call transcript, final category/subcategory/risk, building type, intake notes, and resolution notes. QA-flagged reclassified and over-escalated tickets embed their *final* (corrected) labels so the LLM sees authoritative examples, not intake mistakes.

The top L2 score drives the routing decision:

```
score < 0.63  →  strong domain match, skip grader → classify_llm
score ≥ 0.63  →  uncertain match → grader_gate
```

The 0.63 threshold was empirically determined: all retrieved docs in this corpus fall in the range [0.55, 0.70], and scores below ~0.63 reliably indicate that at least one document describes the same maintenance problem.

### 3.3 `grader_gate` (optional)

An LLM call with structured output (`GraderGate.relevant: bool`) that judges whether the retrieved documents are substantively relevant — same maintenance domain and comparable situation, not just superficial word overlap. If the grader returns `relevant=False` and the rewrite cap has not been hit, it routes to `rewrite_rag_query`. The cap is 2 rewrites per call to prevent infinite loops.

### 3.4 `rewrite_rag_query` (optional)

An LLM call that rewrites the query text to be more targeted, using only facts from the transcript (the system message explicitly forbids inventing new information). The rewritten query replaces `rag_query` and the loop returns to `rag_retrieve`.

### 3.5 `classify_llm`

The core classification node. Invokes GPT-4o-mini with structured output (Pydantic `CallClassification`) using three content blocks:

1. **System prompt** — 500-line domain prompt including the full taxonomy, risk level definitions, empirical base-risk anchors derived from the historical corpus, subcategory disambiguation rules, caller self-correction handling, and over-escalation warnings.
2. **Caller profile block** (if known) — name, company, default building, default floor, last-verified timestamp, marked active/inactive.
3. **RAG few-shot block** — up to 5 similar historical tickets, each annotated with QA flags (`[note: intake was reclassified]`, `[note: intake was over-escalated]`).

After LLM output, the node resolves the building: first by exact name match (case-insensitive), then by substring match against the 52-building registry, and finally by the caller profile's `primary_building_id` when the transcript provides no location signal. The canonical building name and address from `buildings.json` always overwrite the LLM's output to prevent hallucinated names.

### 3.6 `validator_gate`

A fully deterministic, rule-based correction layer that runs after `classify_llm`. This is the single most impactful component after the LLM itself — it handles the LLM's systematic biases discovered during dev-set iteration.

**Step 0 — Subcategory corrections** (run first so risk rules use the right subcategory):

| Pattern | Correction |
|---------|-----------|
| `gas_chemical` but caller never says "gas/sulfur/rotten egg" OR caller mentions benign source (nail polish, bleach, burned food) | → `air_quality` / `HVAC` |
| `power_outage` but caller mentions burning smell near electrical panel | → `panel_hazard` |
| `structural` but caller says "nothing structural / ceiling tile / water damage" | → `roof_leak` / `PLUMBING` |
| `pipe_leak` but caller says dripping from ceiling/above | → `roof_leak` |
| `drainage_backup` in restroom + caller mentions sink/toilet | → `restroom_fixture` / LOW |
| `air_quality` but caller identifies burnt food or candle as source | → `waste_odor` / `JANITORIAL` / LOW |
| `no_cooling` but AC still running + performance degraded | → `refrigerant` / LOW |
| `slip_trip` with wrong category (LLM sometimes picks `GROUNDS_EXTERIOR`) | → `LIFE_SAFETY` or `JANITORIAL` based on context |

**Step 1 — Risk-level corrections** (caller-only text used to prevent agent questions from falsely triggering phrases):

| Subcategory | Over-escalation pattern | Correction |
|-------------|------------------------|-----------|
| `pipe_leak` HIGH | No mention of multi-floor flooding or electrical threat | → MEDIUM |
| `power_outage` HIGH | No mention of entire building, server room, or tripping breaker | → MEDIUM |
| `glass_damage` HIGH | No confirmed injury language | → MEDIUM |
| `roof_leak` HIGH | Not spreading to multiple areas | → MEDIUM |
| `drainage_backup` MEDIUM | No overflow/flooding confirmation | → LOW |
| `infestation` MEDIUM | No widespread/health hazard language | → LOW |
| `waste_odor` MEDIUM | Not causing illness; not in shared corridor | → LOW |
| `access_control` MEDIUM | No confirmed breach | → LOW |
| `no_heating` MEDIUM | Not blowing cold air, not medical, not multi-suite | → LOW |
| `no_cooling` MEDIUM | Not medical, not extreme heat health risk, not multi-suite | → LOW |
| `auto_door` MEDIUM/HIGH | Always routine | → LOW |
| `refrigerant` MEDIUM | No hissing or health evacuation signals | → LOW |
| `suspicious_person` HIGH | No physical threat, forced entry, or social engineering | → MEDIUM |
| `suspicious_person` MEDIUM | Caller reports no-badge or credential fishing | → HIGH |
| `signage_fencing` LOW | Leaning / about to fall | → MEDIUM |
| `slip_trip` MEDIUM + outdoor ice/snow | External entrance hazard | → HIGH |
| `slip_trip` MEDIUM | Already cleaned / caution sign placed | → LOW |
| `restroom_fixture` LOW | Running continuously (water damage risk) | → MEDIUM |
| `entrapment` EMERGENCY | Occupants already out OR communicating safely without medical distress | → HIGH |
| `panel_hazard` EMERGENCY | No active arcing, fire, or burning smell confirmed | → HIGH |
| `unauthorized_access` EMERGENCY | No weapon or active violence in caller text | → HIGH |
| `air_quality` EMERGENCY | No gas/sulfur/toxic fume/evacuation confirmation | → MEDIUM |
| `landscaping` MEDIUM | Always routine | → LOW |

**Step 2 — HITL flag** (recomputed here, overriding the LLM's `needs_human_review` field):

- EMERGENCY / HIGH → always `True`
- MEDIUM → `False` by default, then subcategory-specific escalation rules apply (see §5)
- LOW → always `False`

**Special signal overrides** (apply at any risk level):

- Active refrigerant leak ("leaking refrigerant" in caller text, not stale report) → `True`
- Cleaning chemical smell in shared area → `True`  
- Smoke alarm triggered for non-fire/gas issue → `True`
- Visible smoke mentioned by caller for non-fire/gas issue → `True`
- Roof leak with physically fallen ceiling piece → `True`

### 3.7 `hitl_review`

Uses LangGraph's `interrupt()` mechanism to pause the graph. In batch evaluation mode the harness immediately resumes with the sentinel `"approved"`. In interactive mode (the CLI at `evaluation/hitl_cli.py`) or the voice demo, a human can approve or supply field-level overrides.

The `reviewer_input` protocol supports both shorthand (`"approved"`) and structured dict overrides with key aliases (`override_category`, `override_subcategory`, `override_risk_level`, etc.). If no vendor was found, `needs_human_review` is upgraded to `True` even if risk is LOW.

### 3.8 `vendor_select`

Four-tier fallback selection with a scoring function. The SLA constraint is never relaxed.

**Scoring function** (descending priority tuple):

```python
(availability_score, -sla_val, has_24_7, rating, cost_score)
```

- `availability_score`: 2=available, 1=at_capacity, 0=unknown
- `-sla_val`: lower SLA (faster) is better (negated for descending sort)
- `has_24_7`: boolean, prioritise 24/7 for EMERGENCY
- `rating`: 0–5 float
- `cost_score`: 3=budget, 2=standard, 1=premium

**Four-tier relaxation** (vendor type and SLA always required):

| Tier | specialty | city | building_type | 24/7 |
|------|-----------|------|---------------|------|
| 1 | ✓ | ✓ | ✓ | if EMERGENCY |
| 2 | ✓ | ✓ | ✓ | relaxed |
| 3 | ✓ | ✓ | relaxed | relaxed |
| 4 | relaxed | ✓ | relaxed | relaxed |
| 5* | relaxed | relaxed | relaxed | relaxed |

\* Tier 5 only fires when city is unknown. If the city is known and no vendor qualifies, the case is genuinely unroutable and escalates to human.

Emergency services (911) are dispatched only when: `risk == EMERGENCY` AND `subcategory ∈ {gas_chemical, fire_smoke, active_threat, entrapment}` AND `needs_human_review == False`.

### 3.9 `log_result`

Assembles the `final_decision` dict from the classification and vendor selection results, and records per-node wall-clock timings.

### 3.10 Post-processing in `classify()`

After graph execution, two additional post-processing steps run:

1. **Clarification signals** — regex patterns catch template phrasing (`"there's an issue with"`) and physical impossibilities (floor 5+ paired with outdoor lawn/garden terms)
2. **Floor normalization** — `"Floor G"`, `"Floor 0"`, `"ground floor"` → `"Ground Floor"`

---

## 4. Risk-Band Cutoffs — Corpus Derivation

### 4.1 Dataset

The 10,000-record `historical_records.json` CMMS export provides both `intake_risk_level` (operator's initial call) and `final_risk_level` (on-site technician's finding). The delta between these two reveals systematic over-escalation patterns.

### 4.2 Corpus Statistics

| Risk Level | Intake Count | Final Count | Net change |
|------------|-------------|-------------|-----------|
| LOW | 5,416 (54.2%) | 5,494 (54.9%) | +78 |
| MEDIUM | 1,983 (19.8%) | 2,100 (21.0%) | +117 |
| HIGH | 1,170 (11.7%) | 1,096 (10.9%) | -74 |
| EMERGENCY | 1,431 (14.3%) | 1,310 (13.1%) | -121 |

**2.2% of all calls were over-escalated** (intake HIGH/EMERGENCY → final lower). The LLM baseline without correction exhibits a far higher over-escalation rate, which this design targets.

### 4.3 QA Audit Signal

The `qa_audit_findings.json` contains 988 QA-sampled tickets. Of these:
- **279** were reclassified (subcategory changed by on-site technician)
- **188** were flagged as over-escalated
- **391** had the floor wrong at intake

The top over-escalation patterns from QA notes directly motivated the validator_gate rules:

| QA Pattern | Tickets | Rule Derived |
|-----------|---------|-------------|
| gas_chemical → cleaning product fumes | 59 | Downgrade to air_quality unless caller says "gas/sulfur/rotten egg" |
| fire_smoke → burnt food/candle | 59 | Downgrade to waste_odor when physical source identified |
| structural → roof_leak (ceiling tile/water) | 38 | Downgrade when caller says "nothing structural" / "ceiling tile" |
| active_threat → suspicious person left | 28 | EMERGENCY requires *ongoing* threat |
| entrapment → elevator OOS, no one trapped | 28 | EMERGENCY requires someone IS trapped right now |
| unauthorized_access HIGH → badge reader issue | 10 | HIGH requires confirmed breach, not just access complaint |

### 4.4 Subcategory Base-Risk Anchors

The empirical distribution of `final_risk_level` per subcategory across 10,000 records establishes the **default risk** for each subcategory:

**Always LOW (>99% of final records):**
`waste_odor`, `door_mechanical`, `pavement_damage`, `refrigerant`, `parking_lighting`, `restroom_fixture`, `infestation` (predominantly)

**MEDIUM by default:**
`malfunction` (313/316 MEDIUM), `air_quality` (307/314 MEDIUM), `roof_leak` (295/299 MEDIUM), `suspicious_person` (193/304 MEDIUM, 105/304 HIGH), `no_heating` (mixed: 197 LOW / 109 MEDIUM)

**HIGH by default:**
`structural` (240/280 HIGH, 40/280 EMERGENCY), `panel_hazard` (203/290 EMERGENCY, 72/290 HIGH)

These anchors are embedded directly in the system prompt as the **Base Risk Table** and enforced as hard rules in `validator_gate`.

### 4.5 SLA Thresholds

SLA constraints per `(subcategory, risk_level)` were derived from the vendor roster's `response_sla_minutes` and `emergency_response_sla_minutes` fields. The max SLA for each combination equals the tightest SLA that any vendor in the network can meet:

- EMERGENCY subcategories: 30 minutes
- HIGH risk: 120 minutes  
- MEDIUM risk: 240 minutes
- LOW risk: 480 minutes (no effective filter — all vendors qualify)

---

## 5. HITL Policy

### 5.1 Design Goal

The HITL gate must balance two competing risks:
- **Under-escalation** (missing a genuine emergency) — safety risk
- **Over-escalation** (routing routine calls to humans) — throughput cost

The benchmark scores HITL F1 at 15% weight, with a -5 point penalty per false-911 dispatch. This makes precision for EMERGENCY cases the highest-priority constraint.

### 5.2 Evidence Base

Three signals informed the HITL policy:

1. **Historical over-escalation rate by subcategory** — the corpus shows which issues are systematically over-escalated at intake (e.g. fire_smoke 59/299, entrapment 28/X)
2. **QA audit notes** — specific incident descriptions show the exact phrases that differentiate genuine emergencies from benign reports
3. **Dev-set iteration** — calibrated `validator_gate` rules against the 200 labeled dev transcripts, targeting HITL F1 > 0.90

### 5.3 HITL Rules

```
Risk = EMERGENCY or HIGH  →  always escalate (True)
Risk = LOW                →  never escalate (False)
Risk = MEDIUM             →  False by default, except:
```

**MEDIUM escalation rules (subcategory-specific):**

| Subcategory | Escalate MEDIUM when... |
|-------------|------------------------|
| `malfunction` | Someone was ever inside the stalled elevator ("someone inside", "they got out", "was inside") |
| `suspicious_person` | Caller reports active confrontation: yelling, arguing, getting aggressive |
| `refrigerant` | Active leak in progress ("leaking refrigerant", "hissing") — not a stale technician report |
| `air_quality` | Caller identifies cleaning chemical (bleach, ammonia) in a shared area |
| any non-fire/gas | Smoke alarm triggered or caller sees visible smoke |
| `roof_leak` | Physical ceiling piece has fallen ("fell", "came down", "tile fell") |

**The validator_gate intentionally does NOT follow the LLM's `needs_human_review` output.** The LLM over-fires `needs_human_review=True` for uncertain or ambiguous calls, which hurts auto-resolution rate (10% weight). The validator recomputes this field from scratch using only the rules above.

### 5.4 HITL in Interactive Mode

The `hitl_review` node calls `interrupt()` from LangGraph, which suspends the graph. The `evaluation/hitl_cli.py` tool surfaces the AI's classification to a human reviewer who can:
- Approve the AI's call (`"approved"`)
- Override individual fields (`override_subcategory`, `override_risk_level`, etc.)
- All overrides are recorded in `trainer_log.human_override`

### 5.5 Dev-Set HITL Results

```
HITL   precision=0.903  recall=0.949  F1=0.926
```

The high recall (0.949) means we rarely miss a genuine escalation case. The precision (0.903) reflects a small number of routine MEDIUM calls that we escalate unnecessarily — primarily in the `suspicious_person` and `air_quality` subcategories where the text signals are ambiguous.

---

## 6. Vendor-Selection Algorithm

### 6.1 Vendor Roster

32 vendors across 16 vendor types, covering 35 California cities and 5 building types. Availability is cached at `status_at_last_check` (values: `"available"`, `"at_capacity"`, `"unavailable"`).

### 6.2 Handling Stale Availability

The `status_at_last_check` field is explicitly a **cache** — it may be stale by hours or days. The algorithm treats it probabilistically rather than as a hard constraint:

- `available` → score 2 (preferred)
- `at_capacity` → score 1 (acceptable, may have opened up)
- `unavailable` or missing → score 0 (last resort, but not filtered out)

This means a vendor marked `at_capacity` can still be selected if they are the only vendor matching all other constraints. The human review process handles the edge case where the dispatched vendor turns out to actually be unavailable.

### 6.3 Subcategory→Vendor Type Mapping

Each of the 36 subcategories maps to exactly one vendor type (hardcoded from the historical corpus — every `(subcategory, assigned_vendor_type)` pair in 10,000 records is unambiguous). This mapping is the first and hardest filter.

### 6.4 SLA as a Hard Constraint

Unlike availability (soft), the SLA constraint is never relaxed across tiers. A vendor that cannot meet the required response time for the `(subcategory, risk_level)` pair is excluded at every tier. This prevents dispatching a slow vendor to an EMERGENCY even if they are the only available one — in that case the case escalates to human.

### 6.5 Fallback Logic for Unknown Cities

When the building cannot be resolved to our 52-building registry and the caller profile provides no city, Tier 5 removes the city constraint entirely. This fires rarely (anonymous callers with no profile) and is the designed last resort.

### 6.6 Vendor Score Example

For a `pipe_leak` MEDIUM call at a building in San Diego:

```
Required vendor_type: plumber
SLA constraint: response_sla_minutes ≤ 240

Tier 1 search (specialty=pipe_leak, city=San Diego, building_type matched, 24/7=False):
  → finds v_003: available, SLA=120, rating=4.8, standard cost  → score (2, -120, 0, 4.8, 2)
  → finds v_019: at_capacity, SLA=90, rating=4.5, budget cost   → score (1, -90, 0, 4.5, 3)

Sort descending → v_003 wins (availability=2 beats at_capacity=1)
```

---

## 7. Clarification Policy

### 7.1 Design Goal

Asking unnecessary clarifying questions reduces throughput and frustrates callers. The SOP (taxonomy.md) explicitly states that "most calls don't need a follow-up." The clarification flag therefore has a high bar.

### 7.2 Three Conditions for `needs_clarification=True`

The system prompt encodes exactly three conditions:

1. **Vague description** — the caller cannot describe the problem clearly enough to select a subcategory ("something's wrong", "there's just an issue")
2. **Location conflict** — caller states contradictory floors or buildings that cannot be resolved ("Floor 3… or maybe Floor 5", two different building names)
3. **Critical info missing for HIGH/EMERGENCY** — the floor or building is completely unknown AND the severity makes location essential before dispatching

Explicitly excluded:
- Anonymous callers who clearly describe their issue
- LOW or routine MEDIUM calls (dispatch proceeds without a precise suite number)
- Any call where a subcategory can be selected with reasonable confidence

### 7.3 Post-Processing Clarification Signals

After graph execution, `classify()` applies two additional regex-based clarification detectors:

- **Template phrasing**: `"there's an issue with"` — a vague caller template that signals insufficient detail
- **Physical impossibility**: high floor number (≥5) combined with outdoor terms (lawn, garden, parking lot exterior) — a contradictory location that needs resolution

### 7.4 Dev-Set Clarification Results

```
Clarif precision=0.962  recall=0.781  F1=0.862
```

High precision (0.962) means we rarely ask unnecessary clarification questions. The lower recall (0.781) indicates we miss some genuinely ambiguous cases — primarily `location_conflict` case type where the transcript is subtly contradictory but doesn't match our explicit patterns.

---

## 8. Voice Demo Extension

### 8.1 Architecture

```
                    ┌──────────────┐
                    │   Twilio     │
                    │   (PSTN)     │
                    └──────┬───────┘
                           │ HTTP webhook (TwiML)
                           ▼
                    ┌──────────────┐
                    │  FastAPI     │
                    │  server.py   │
                    │              │
                    │ /voice/      │
                    │  incoming    │
                    │ /voice/      │
                    │  process     │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │ Deepgram │ │ CBRE     │ │ElevenLabs│
       │ Nova-2   │ │ classify │ │ Flash    │
       │ STT      │ │ agent    │ │ v2.5 TTS │
       └──────────┘ └──────────┘ └──────────┘
```

### 8.2 Call Flow

1. **Incoming call**: Twilio sends `POST /voice/incoming` → server returns TwiML opening prompt and `<Record>` verb
2. **Each caller turn**: Twilio sends `POST /voice/process` with `RecordingUrl`
   - Fetch audio from Twilio → Deepgram STT → transcript
   - Update `CallSession` with caller turn and slot extraction
   - Deterministic gate: ask for missing slots (issue, building, floor) before classifying
   - LLM intake decision (`decide_next_step`) assesses if enough context exists to finalize
   - If finalizing: call `classify()` → check `needs_clarification`
   - If clarification needed (and under limit): ask targeted follow-up
   - If no clarification / turn cap reached: call `final_dispatch_response()` → ElevenLabs TTS → hang up
3. **Persistence**: work order artifact saved to `demo_outputs/workorders/`

### 8.3 Session State

Each `CallSession` tracks:
- `turns`: all speaker turns so far
- `intake_slots`: extracted `{issue, building_name, floor, urgency}`
- `pending_clarification`: whether a clarification loop is active
- `clarification_rounds` / `classify_attempts`: loop caps to prevent infinite back-and-forth

### 8.4 Demo vs Production Mode

`VOICE_DEMO_MODE=true` replaces Deepgram/ElevenLabs with mock stubs: a fixed transcript string replaces STT, and TwiML `<Say>` replaces generated MP3 audio. This allows end-to-end testing without API keys.

---

## 9. Confusion Matrix & Error Analysis

### 9.1 Dev-Set Case Type Distribution

| Case Type | Count | Description |
|-----------|-------|-------------|
| normal | 90 | Standard single-issue calls |
| hard | 36 | Ambiguous or multi-domain calls |
| clarification | 24 | Calls requiring or not requiring follow-up |
| over_escalation_trap | 16 | Benign-but-scary sounding calls (false-911 bait) |
| edge | 16 | Boundary subcategory cases |
| multi_turn_correction | 10 | Caller corrects themselves mid-call |
| location_conflict | 8 | Contradictory location signals |

### 9.2 Per-Case-Type Results (Dev Set, 200 transcripts)

```
Case Type               n     cat   sub   risk  field vendor auto
─────────────────────────────────────────────────────────────────
normal                  90    90    90    85    70    84    50
over_escalation_trap    16    16    16    16    16    16     0
multi_turn_correction   10    10    10    10    10    10     6
clarification           24    24    24    24    24    24     0
edge                    16    15    15    10    13    12     2
hard                    36    36    36    36    36    36    22
location_conflict        8     8     8     8     7     8     0
```

### 9.3 Subcategory Confusion Matrix (Top Errors)

Only 1 subcategory error on the dev set:

```
True label            → Predicted            Count
──────────────────────────────────────────────────
slip_trip             → drainage_backup          1
```

This single confusion occurs when a transcript describes a wet floor near a drain, and the LLM anchors on the drain rather than the slip hazard. The `validator_gate` subcategory correction for `drainage_backup` in restrooms catches this in most cases, but a non-restroom wet floor near a drain can still slip through.

### 9.4 Risk-Level Error Analysis

5/200 calls have wrong risk level. All are in the **edge** case type. Patterns:

- **Under-escalation in edge cases**: an `air_quality` call in a medical building that the corpus evidence suggests should be MEDIUM, but the caller's language is vague enough to trigger the MEDIUM→LOW downgrade rule
- **Over-escalation in edge cases**: `suspicious_person` calls where the boundary between "passive loitering" and "active confrontation" is genuinely ambiguous from text alone

The 6/16 edge-case risk failures are the expected hard ceiling given that `edge` cases are specifically designed to be near decision boundaries.

### 9.5 Field Extraction Errors (12/200 wrong)

Location extraction is the hardest axis. Errors cluster in:

- **normal** case type (20 field errors): callers who say a building nickname rather than the canonical name from `buildings.json`
- **location_conflict** (1 field error): the 7/8 correct rate is strong, but one case has a floor that the substring match resolves to the wrong building
- Zero field errors in `hard`, `over_escalation_trap`, `multi_turn_correction`, `clarification` — these typically have cleaner location signals

### 9.6 Auto-Resolution Rate

```
Auto-resolved: 80 / 85 eligible routine cases = 94.1%
```

The 5 non-auto-resolved eligible cases:
- 2 `suspicious_person` MEDIUM calls where our "yelling/arguing" rule fires on language that resembles active confrontation but was labeled routine
- 2 `refrigerant` calls where "hissing" appears in a stale-report context that the stale-report filter missed
- 1 `normal` call where the LLM set `needs_clarification=True` unnecessarily (precision miss)

### 9.7 False-911 Dispatches

```
False-911 dispatches on over-escalation traps: 0  (penalty -0.00)
```

The agent correctly avoids calling emergency services on all 16 over-escalation trap cases. This is the highest-priority safety metric in the design.

---

## 10. Dev-Set Score Breakdown

```
Axis                   Score    Weight   Contribution
─────────────────────────────────────────────────────
Category accuracy      99.50%    10%       9.95
Subcategory accuracy   99.50%    15%      14.93
Risk-level accuracy    94.50%    10%       9.45
HITL trigger F1        92.56%    15%      13.88
Clarification F1       86.21%     5%       4.31
Field extraction       88.00%    10%       8.80
Vendor match           95.00%    10%       9.50
Auto-resolution rate   94.12%    10%       9.41
Call summary present  100.00%     5%       5.00
Trainer log present   100.00%    10%      10.00
─────────────────────────────────────────────────────
Composite (raw):                          95.23
False-911 penalty:                         0.00
Final composite:                          95.23
```

---

## 11. Trainer Log

Every prediction includes a `trainer_log` dict:

```json
{
  "full_transcript": "[AGENT] Thank you... [CALLER] There's a water...",
  "ai_prediction": {
    "category": "PLUMBING",
    "subcategory": "pipe_leak",
    "risk_level": "MEDIUM",
    "needs_human_review": false,
    ...
  },
  "human_override": null,
  "hitl_decision": {
    "status": "auto_routed",
    "reason": "No human review required"
  },
  "routing_decision": "Auto-routed -> pipe_leak (risk MEDIUM)",
  "was_overridden": false,
  "final_decision": { ... }
}
```

When a human reviewer overrides the AI:

```json
{
  "human_override": {
    "override_subcategory": "roof_leak",
    "override_risk_level": "LOW",
    "override_reason": "Caller described ceiling drip, not pipe"
  },
  "hitl_decision": {
    "status": "overridden",
    "reason": "risk=HIGH",
    "override_reason": "Caller described ceiling drip, not pipe",
    "original_classification": { ... }
  }
}
```

The dev-set trainer log (`evaluation/trainer_log.json`) contains all 200 entries, and a fine-tuning ready JSONL version (`evaluation/trainer_log_finetune_reviewed.jsonl`) formats each example as an instruction-response pair suitable for GPT-4o-mini fine-tuning.

---

## 12. Design Decisions & Trade-offs

### 12.1 LLM + Rules vs Pure LLM

**Decision**: Use GPT-4o-mini for classification and a separate deterministic `validator_gate` for corrections.

**Rationale**: The LLM has systematic biases (over-escalation, gas_chemical hallucination) that are consistent and predictable from the historical corpus. A rule-based correction layer is:
- Faster (no LLM call)
- More auditable (each rule has a specific provenance from the QA audit)
- More reliable (not susceptible to prompt drift)

**Trade-off**: Rules must be maintained as the taxonomy evolves. Pure LLM would generalize better to new subcategories.

### 12.2 RAG with Adaptive Grading vs Static Few-Shot

**Decision**: Retrieve similar historical tickets dynamically, then gate with an LLM grader before using them as few-shot context.

**Rationale**: Static few-shot examples in the system prompt cannot cover 36 subcategories × 4 risk levels adequately. Dynamic retrieval finds the most relevant precedents for each specific call.

**Trade-off**: The grader adds 1 LLM call per request (when similarity is ambiguous), increasing latency. The `_RAG_RELEVANCE_SCORE_THRESHOLD = 0.63` skip condition reduces this cost for the majority of calls with strong matches.

### 12.3 Caller-Only Text vs Full Transcript for Risk Rules

**Decision**: Risk downgrade rules in `validator_gate` use `caller_tx` (caller turns only), while some rules use `tx` (full transcript).

**Rationale**: The agent's own questions — "Is water overflowing?" — contain the exact phrases that would falsely trigger the "water overflowing" upgrade rule if the full transcript were used. Isolating caller text prevents the agent from inflating its own escalation signals.

### 12.4 Tiered Vendor Selection vs Constraint Satisfaction

**Decision**: Tiered relaxation in a fixed priority order rather than constraint satisfaction (e.g. mixed-integer programming).

**Rationale**: The vendor roster has only 32 vendors; exhaustive search is trivial. The fixed relaxation order (24/7 → building_type → specialty → city) encodes the business priority: safety coverage first, then certification match, then geography.

**Trade-off**: The fixed relaxation order may occasionally select a more expensive vendor when a cheaper one with slightly different constraints would have been equally valid.

### 12.5 Building Resolution: Exact → Substring → Profile

**Decision**: Three-step building name resolution, with the LLM's extracted name always overwritten by the canonical `buildings.json` entry.

**Rationale**: The LLM occasionally hallucinate building names or uses abbreviations. Overwriting with the canonical entry ensures downstream systems (vendor matching by city, building type) receive accurate data.

**Trade-off**: If a caller mentions a building that is genuinely not in the 52-building registry (e.g. a new property), the profile fallback may assign the wrong building.

---

*End of Design Document*
