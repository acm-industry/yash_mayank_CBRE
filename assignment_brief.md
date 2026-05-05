# Assignment Brief — Human-in-the-Loop RAG for CBRE Call Intake

## The task

You are building an agentic voice bot that triages incoming facilities calls
for CBRE's property portfolio. Today, every one of those calls hits a human
operator — roughly 10K/day across the portfolio, and the operators are
overwhelmed. The business ask:

> **Auto-resolve routine calls. Escalate to a human only when it genuinely
> requires judgment.**

Your agent must:

1. Listen to a multi-turn caller dialogue (provided as structured `turns`).
2. Classify the issue into a `(category, subcategory)` from the domain
   taxonomy, with a risk level.
3. Extract location (building + floor + suite) from the dialogue, reconciling
   against `caller_profiles.json` when the caller is known.
4. Decide whether to **auto-resolve and dispatch** a vendor, **ask one
   clarifying question**, or **hand off to a human reviewer** — this is the
   HITL gate.
5. Pick a qualified vendor from `vendors.json` (or correctly flag that none
   qualify and a human should reroute).
6. Emit a compact, operator-style `call_summary` plus a `trainer_log` that
   captures what the AI decided, what a human changed (if anything), and the
   final outcome.

## What you're given

| Folder | Purpose |
|---|---|
| `operational/` | CBRE's live ops data: the CMMS export (`historical_records.json`), building registry, vendor roster, caller profiles, and the intake SOP (`taxonomy.md`). This is what a real CBRE ops team would hand you. |
| `evaluation/`  | Your benchmark: 200 labeled dev transcripts (`eval_transcripts_dev.json` + `dev_labels.json`) to iterate on, 800 unlabeled test transcripts (`eval_transcripts_test.json`) used for final grading, plus a post-hoc QA audit file (`qa_audit_findings.json`), the scorer, the agent contract, and the Prediction dataclass. |

## What the grader expects

Your `classify(turns, caller_phone) -> dict` must return (at minimum):

```python
{
    "category":                       "PLUMBING",
    "subcategory":                    "pipe_leak",
    "risk_level":                     "MEDIUM",
    "needs_human_review":             False,
    "needs_clarification":            False,
    "building_name":                  "Pacific Ridge Medical Plaza",
    "address":                        "3200 Pacific Coast Hwy",
    "floor":                          "Floor 7",
    "dispatched_vendor_id":           "v_002",
    "dispatched_emergency_services":  False,
    "call_summary":                   "2-3 sentence operator-style narrative.",
    "trainer_log": {
        "full_transcript":  "...",
        "ai_prediction":    { ... },
        "human_override":   None,
        "final_decision":   { ... },
    },
}
```

Scoring is weighted across subcategory, risk level, HITL-F1, clarification-F1,
location, and vendor match. Full weights and the hard-fail penalty are
documented in `evaluation/scoring.py`'s docstring.

## Things you have to design (not prescribed)

- **Numeric thresholds** that separate LOW / MEDIUM / HIGH / EMERGENCY.
  Derive them from the historical corpus — `historical_records.json` has
  `intake_risk_level` + `final_risk_level` for 10K past tickets.
- **HITL policy**: when does the agent pause for a human rather than
  auto-resolve? The `qa_audit_findings.json` shows which past tickets a human
  would have caught — use that signal.
- **Clarification policy**: when to ask a follow-up vs. commit to a label.
- **Vendor ranking**: `vendors.json` gives you cost, rating, SLA, and a
  (possibly stale) availability cache — how you balance those is your call.
- **How to use profile data** — profiles are a "last-known-default" snapshot,
  not live location. Some are `active=false`, some have a stale building FK.

## Project-level deliverables

1. Working agent exposing `classify(turns, caller_phone) -> dict`.
2. `predictions.json` produced by running `evaluation/run_eval.py` on the
   dev set (for self-grading) and on the test set (for the final TA run).
3. Design document covering:
   - How you derived risk-band cutoffs from the historical corpus.
   - Your HITL policy and the evidence you used to pick it.
   - Your vendor-selection algorithm and how it handles stale availability.
   - Your clarification policy.
   - A confusion-matrix / error analysis on the dev set.
4. Trainer log from your dev-set run — this is the data a next-gen model
   would learn from.

## The scoring contract in one sentence

**Your dev-set score is what you iterate against. Your test-set score
(same scorer, held-out labels) is what goes on the board.**
