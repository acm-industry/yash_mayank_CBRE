# UCSB Agentic AI — Final Project

Build a Human-in-the-Loop (HITL) agentic voice bot that automates an international
commercial-real-estate call center. Read **`final-assignment.html`** for the full
brief and **`assignment_brief.md`** for a one-page recap.

## What's in this bundle

```
final_project/
├── final-assignment.html      ← the full assignment (open in browser)
├── assignment_brief.md         ← one-page recap
├── operational/                ← CBRE's live-ops data (treat as production)
│   ├── taxonomy.md             ← intake SOP — labels and risk tiers in prose
│   ├── caller_profiles.json    ← 250 known-caller records
│   ├── buildings.json          ← 52 properties (intentionally sparse)
│   ├── vendors.json            ← 32 dispatch vendors with SLAs
│   └── historical_records.json ← 10,000 past tickets (your live RAG knowledge base)
└── evaluation/                 ← your benchmark
    ├── eval_transcripts_dev.json   ← 200 calls, labeled, for self-eval
    ├── dev_labels.json             ← matching ground truth for dev
    ├── eval_transcripts_test.json  ← 800 calls, no labels (final grading set)
    ├── qa_audit_findings.json      ← post-hoc QA notes for the historical knowledge base
    ├── prediction.py               ← Prediction + TrainerLog dataclasses (your output contract)
    ├── run_eval.py                 ← batch harness (loops over transcripts, calls your agent)
    └── scoring.py                  ← composite scorer; same one we use for final grading
```

## Quick start

```bash
# 1. Set up a Python env (Python ≥ 3.9)
python3 -m venv .venv && source .venv/bin/activate
# Install whatever your agent needs (LangGraph, OpenAI, etc.).

# 2. Build your agent. It must expose a single callable:
#       def classify(turns: list[dict], caller_phone: str | None) -> dict
#    Returning the fields documented in evaluation/prediction.py.
#
#    Note: each eval row also carries caller_known_in_profiles: bool — true when
#    caller_phone matches an entry in operational/caller_profiles.json. The
#    harness only forwards `turns` and `caller_phone` to your callable; if you
#    want to use the flag, look it up yourself from caller_profiles.json.

# 3. Run on the dev set (with labels — for self-evaluation)
python evaluation/run_eval.py \
    --agent your_module.your_agent:classify \
    --eval evaluation/eval_transcripts_dev.json \
    --out predictions.json

# 4. Score yourself
python evaluation/scoring.py \
    --eval evaluation/eval_transcripts_dev.json \
    --ground-truth evaluation/dev_labels.json \
    --predictions predictions.json

# 5. When you're ready to submit, generate predictions on the test set:
python evaluation/run_eval.py \
    --agent your_module.your_agent:classify \
    --eval evaluation/eval_transcripts_test.json \
    --out predictions.json
# Submit predictions.json + your repo. We grade with the same scoring.py
# against held-out labels.
```

## What you submit

1. **Source repo** — your agent code, dependencies, README with run instructions.
2. **`predictions.json`** — output from running your agent over `eval_transcripts_test.json`.
3. **Design document (PDF)** — see the brief for required appendices (derived policy,
   HITL design, RAG strategy, vendor-selection logic, trainer-log spec, scale thought-experiment).

The full evaluation rubric, axis weights, and hard-cost rules are in
`final-assignment.html` → "What We Grade".
