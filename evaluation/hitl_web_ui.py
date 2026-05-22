from __future__ import annotations

import html
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Form, HTTPException, Query
from fastapi.responses import HTMLResponse
from langgraph.types import Command

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cbre_agent.agent import _GRAPH

DEFAULT_EVAL = HERE / "eval_transcripts_dev.json"
DEFAULT_TRAINER_LOG = HERE / "trainer_log_ui.json"

app = FastAPI(title="CBRE HITL Review UI")

# Pending review contexts keyed by review_id.
PENDING_REVIEWS: Dict[str, dict] = {}


def _build_initial_state(turns: list[dict], caller_phone: str | None) -> dict:
    return {
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


def _load_rows(eval_path: Path = DEFAULT_EVAL) -> list[dict]:
    data = json.loads(eval_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{eval_path} must contain a JSON list.")
    return data


def _load_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def _append_log(path: Path, entry: dict) -> None:
    rows = _load_log(path)
    rows.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


def _derive_review_outcome(final_values: dict) -> str:
    status = ((final_values.get("hitl_decision") or {}).get("status") or "").lower()
    if status == "approved":
        return "human_approved"
    if status == "overridden":
        return "human_override"
    return "auto_routed"


def _h(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _page_template(title: str, body: str) -> str:
    return f"""
    <html>
    <head>
      <title>{_h(title)}</title>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        :root {{
          --bg: #f4f6fb;
          --card: #ffffff;
          --text: #1a1f36;
          --muted: #65708a;
          --primary: #2f6fed;
          --primary-dark: #2154b6;
          --ok: #12805c;
          --warn: #b4690e;
          --border: #d9deea;
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          padding: 24px;
          background: var(--bg);
          color: var(--text);
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 18px;
        }}
        .title {{ margin: 0; font-size: 30px; }}
        .subtitle {{ margin: 6px 0 0; color: var(--muted); font-size: 14px; }}
        .pill {{
          padding: 6px 10px;
          border-radius: 999px;
          font-size: 12px;
          font-weight: 600;
          background: #e6efff;
          color: #2349aa;
        }}
        .card {{
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 16px;
          margin-bottom: 14px;
          box-shadow: 0 1px 2px rgba(19, 24, 38, 0.05);
        }}
        .card h2, .card h3 {{ margin-top: 0; }}
        .grid-2 {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 14px;
        }}
        .meta {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
        }}
        .meta .item {{
          background: #f8faff;
          border: 1px solid #e4eaf7;
          border-radius: 8px;
          padding: 10px;
        }}
        .meta .label {{
          font-size: 12px;
          color: var(--muted);
          margin-bottom: 3px;
          display: block;
        }}
        .meta .value {{ font-size: 14px; font-weight: 600; }}
        .toolbar {{
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          align-items: end;
        }}
        .field {{ min-width: 200px; flex: 1; }}
        label {{ font-size: 12px; color: var(--muted); font-weight: 600; display: block; margin-bottom: 6px; }}
        input, select {{
          width: 100%;
          padding: 10px;
          border: 1px solid var(--border);
          border-radius: 8px;
          font-size: 14px;
        }}
        .btn {{
          border: 0;
          border-radius: 8px;
          padding: 10px 14px;
          font-size: 14px;
          font-weight: 700;
          cursor: pointer;
          background: var(--primary);
          color: white;
        }}
        .btn:hover {{ background: var(--primary-dark); }}
        .btn-secondary {{
          background: #f2f6ff;
          color: #22479e;
          border: 1px solid #cdd9fb;
        }}
        .status {{
          padding: 10px 12px;
          border-radius: 8px;
          background: #ecfff7;
          color: var(--ok);
          border: 1px solid #cdeede;
          margin-bottom: 12px;
        }}
        .status.warn {{
          background: #fff8eb;
          color: var(--warn);
          border-color: #f2deb8;
        }}
        pre {{
          margin: 0;
          white-space: pre-wrap;
          background: #f9fbff;
          border: 1px solid #e4eaf7;
          border-radius: 8px;
          padding: 12px;
          max-height: 420px;
          overflow: auto;
          font-size: 13px;
          line-height: 1.45;
        }}
        .radio-row {{ display: flex; gap: 18px; margin: 4px 0 12px; }}
        .radio-row label {{ font-size: 14px; color: var(--text); font-weight: 600; display: flex; align-items: center; gap: 6px; }}
        .footer-link {{ color: #3559b7; font-weight: 600; text-decoration: none; }}
        .footer-link:hover {{ text-decoration: underline; }}
        @media (max-width: 900px) {{
          .grid-2 {{ grid-template-columns: 1fr; }}
          .meta {{ grid-template-columns: 1fr; }}
        }}
      </style>
    </head>
    <body>
      <div class="container">{body}</div>
    </body>
    </html>
    """


def _render_index_page(rows: list[dict], selected_index: int, message: str = "") -> str:
    selected_index = max(0, min(selected_index, len(rows) - 1))
    row = rows[selected_index]
    transcript_id = row.get("transcript_id")
    transcript_preview = "\n".join(
        f"[{t.get('speaker', '').upper()}] {t.get('text', '')}" for t in row.get("turns", [])
    )
    options = "\n".join(
        f'<option value="{idx}">{idx:03d} - {_h(r.get("transcript_id", "unknown"))}</option>'
        for idx, r in enumerate(rows)
    )
    msg_html = f"<div class='status'>{_h(message)}</div>" if message else ""
    body = f"""
      <div class="header">
        <div>
          <h1 class="title">CBRE HITL Review Console</h1>
          <p class="subtitle">Run one dev transcript through the graph and review when HITL is triggered.</p>
        </div>
        <span class="pill">Dev Set Size: {len(rows)}</span>
      </div>
      {msg_html}
      <div class="card">
        <h2 style="margin-bottom:12px;">Start A Review</h2>
        <form method="post" action="/run" class="toolbar">
          <div class="field">
            <label for="indexPick">Transcript (index - id)</label>
            <select id="indexPick" name="index">{options}</select>
          </div>
          <div class="field" style="max-width:180px;">
            <label for="indexNum">Current Index</label>
            <input id="indexNum" type="number" value="{selected_index}" min="0" max="{len(rows)-1}" />
          </div>
          <button class="btn" type="submit">Run Through Agent</button>
        </form>
        <script>
          const indexPick = document.getElementById("indexPick");
          const indexNum = document.getElementById("indexNum");
          if (indexPick && indexNum) {{
            indexPick.value = "{selected_index}";
            indexPick.addEventListener("change", () => {{
              indexNum.value = indexPick.value;
            }});
            indexNum.addEventListener("input", () => {{
              const raw = Number(indexNum.value);
              if (!Number.isNaN(raw) && raw >= 0 && raw <= {len(rows)-1}) {{
                indexPick.value = String(raw);
              }}
            }});
          }}
        </script>
      </div>
      <div class="card">
        <h3 style="margin-bottom:10px;">Selected Transcript Preview</h3>
        <div class="meta" style="margin-bottom:10px;">
          <div class="item"><span class="label">Transcript ID</span><span class="value">{_h(transcript_id)}</span></div>
          <div class="item"><span class="label">Caller Phone</span><span class="value">{_h(row.get("caller_phone"))}</span></div>
        </div>
        <pre>{_h(transcript_preview)}</pre>
      </div>
    """
    return _page_template("CBRE HITL UI", body)


def _render_review_page(review_id: str, row: dict, state_values: dict, pre_review: dict) -> str:
    transcript_text = state_values.get("transcript_text", "")
    body = f"""
      <div class="header">
        <div>
          <h1 class="title">HITL Review Required</h1>
          <p class="subtitle">Review the model prediction and confirm or override before routing.</p>
        </div>
        <span class="pill">Review ID: {_h(review_id[:8])}</span>
      </div>
      <div class="status warn">Human review was triggered by the agent for this transcript.</div>
      <div class="card">
        <div class="meta">
          <div class="item"><span class="label">Transcript ID</span><span class="value">{_h(row.get("transcript_id"))}</span></div>
          <div class="item"><span class="label">Caller Phone</span><span class="value">{_h(row.get("caller_phone"))}</span></div>
        </div>
      </div>
      <div class="grid-2">
        <div class="card">
          <h3>Model Prediction (Pre-Review)</h3>
          <pre>{_h(json.dumps(pre_review, indent=2))}</pre>
        </div>
        <div class="card">
          <h3>Transcript</h3>
          <pre>{_h(transcript_text)}</pre>
        </div>
      </div>
      <div class="card">
        <h3>Reviewer Decision</h3>
        <form method="post" action="/review/{_h(review_id)}">
          <div class="radio-row">
            <label><input type="radio" name="decision" value="approve" checked/> Approve</label>
            <label><input type="radio" name="decision" value="override"/> Override</label>
          </div>
          <div class="grid-2">
            <div><label>Override Subcategory</label><input type="text" name="override_subcategory" /></div>
            <div><label>Override Category</label><input type="text" name="override_category" /></div>
            <div><label>Override Risk Level</label><input type="text" name="override_risk_level" /></div>
            <div><label>Override Building Name</label><input type="text" name="override_building_name" /></div>
            <div><label>Override Address</label><input type="text" name="override_address" /></div>
            <div><label>Override Floor</label><input type="text" name="override_floor" /></div>
          </div>
          <div style="margin-top:12px;">
            <label>Override Reason</label>
            <input type="text" name="override_reason" />
          </div>
          <div style="margin-top:14px; display:flex; gap:10px;">
            <button class="btn" type="submit">Submit Review</button>
            <a href="/" class="btn btn-secondary" style="text-decoration:none; display:inline-flex; align-items:center;">Cancel</a>
          </div>
        </form>
      </div>
    """
    return _page_template("HITL Review", body)


def _render_result_page(transcript_id: str, final_values: dict, review_outcome: str, log_path: Path) -> str:
    status_class = "status" if review_outcome != "human_override" else "status warn"
    body = f"""
      <div class="header">
        <div>
          <h1 class="title">Review Complete</h1>
          <p class="subtitle">The graph has resumed and produced a final routing decision.</p>
        </div>
        <span class="pill">{_h(review_outcome)}</span>
      </div>
      <div class="{status_class}">
        Transcript {_h(transcript_id)} processed. Routing decision: {_h(final_values.get("routing_decision"))}
      </div>
      <div class="card">
        <div class="meta">
          <div class="item"><span class="label">Transcript ID</span><span class="value">{_h(transcript_id)}</span></div>
          <div class="item"><span class="label">Review Outcome</span><span class="value">{_h(review_outcome)}</span></div>
        </div>
      </div>
      <div class="grid-2">
        <div class="card">
          <h3>HITL Decision</h3>
          <pre>{_h(json.dumps(final_values.get("hitl_decision"), indent=2))}</pre>
        </div>
        <div class="card">
          <h3>Final Decision</h3>
          <pre>{_h(json.dumps(final_values.get("final_decision"), indent=2))}</pre>
        </div>
      </div>
      <div class="card">
        <p>Saved log entry to: <code>{_h(log_path)}</code></p>
        <a class="footer-link" href="/">Run another transcript</a>
      </div>
    """
    return _page_template("HITL Result", body)


@app.get("/", response_class=HTMLResponse)
def index(selected_index: int = Query(default=0, ge=0)) -> str:
    rows = _load_rows(DEFAULT_EVAL)
    return _render_index_page(rows, selected_index)


@app.post("/run", response_class=HTMLResponse)
def run_transcript(index: int = Form(...)) -> str:
    rows = _load_rows(DEFAULT_EVAL)
    if index < 0 or index >= len(rows):
        raise HTTPException(status_code=400, detail=f"index must be in [0, {len(rows)-1}]")

    row = rows[index]
    transcript_id = row.get("transcript_id", f"row-{index}")
    thread_id = f"hitl-ui-{transcript_id}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    _GRAPH.invoke(_build_initial_state(row["turns"], row.get("caller_phone")), config)
    paused_state = _GRAPH.get_state(config)
    pre_review = dict(paused_state.values.get("classification") or {})

    # HITL path: show UI review form.
    if paused_state.next:
        review_id = uuid.uuid4().hex
        PENDING_REVIEWS[review_id] = {
            "config": config,
            "row": row,
            "thread_id": thread_id,
            "pre_review": pre_review,
        }
        return _render_review_page(review_id, row, paused_state.values, pre_review)

    # No HITL needed: still log and show result.
    final_values = _GRAPH.get_state(config).values
    entry = {
        "timestamp": datetime.now().isoformat(),
        "thread_id": thread_id,
        "transcript_id": transcript_id,
        "review_outcome": _derive_review_outcome(final_values),
        "routing_decision": final_values.get("routing_decision"),
        "hitl_decision": final_values.get("hitl_decision"),
        "ai_prediction_pre_review": pre_review,
        "human_override": final_values.get("human_override"),
        "final_decision": final_values.get("final_decision"),
        "full_transcript": final_values.get("transcript_text"),
    }
    _append_log(DEFAULT_TRAINER_LOG, entry)
    return _render_result_page(
        transcript_id=transcript_id,
        final_values=final_values,
        review_outcome=entry["review_outcome"],
        log_path=DEFAULT_TRAINER_LOG,
    )


@app.post("/review/{review_id}", response_class=HTMLResponse)
def submit_review(
    review_id: str,
    decision: str = Form(...),
    override_subcategory: str = Form(default=""),
    override_category: str = Form(default=""),
    override_risk_level: str = Form(default=""),
    override_building_name: str = Form(default=""),
    override_address: str = Form(default=""),
    override_floor: str = Form(default=""),
    override_reason: str = Form(default=""),
) -> str:
    ctx = PENDING_REVIEWS.get(review_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Review session not found or already submitted.")

    config = ctx["config"]
    row = ctx["row"]
    pre_review = ctx["pre_review"]
    transcript_id = row.get("transcript_id", "unknown")

    if decision.lower() == "approve":
        resume_payload: Dict[str, Any] = {"approved": True}
    else:
        resume_payload = {"approved": False}
        if override_subcategory.strip():
            resume_payload["override_subcategory"] = override_subcategory.strip()
        if override_category.strip():
            resume_payload["override_category"] = override_category.strip()
        if override_risk_level.strip():
            resume_payload["override_risk_level"] = override_risk_level.strip()
        if override_building_name.strip():
            resume_payload["override_building_name"] = override_building_name.strip()
        if override_address.strip():
            resume_payload["override_address"] = override_address.strip()
        if override_floor.strip():
            resume_payload["override_floor"] = override_floor.strip()
        if override_reason.strip():
            resume_payload["override_reason"] = override_reason.strip()
        if len(resume_payload) == 1:
            # Override selected but no fields supplied -> treat as approve.
            resume_payload = {"approved": True}

    _GRAPH.invoke(Command(resume=resume_payload), config)
    final_values = _GRAPH.get_state(config).values

    entry = {
        "timestamp": datetime.now().isoformat(),
        "thread_id": ctx["thread_id"],
        "transcript_id": transcript_id,
        "review_outcome": _derive_review_outcome(final_values),
        "routing_decision": final_values.get("routing_decision"),
        "hitl_decision": final_values.get("hitl_decision"),
        "ai_prediction_pre_review": pre_review,
        "human_override": final_values.get("human_override"),
        "final_decision": final_values.get("final_decision"),
        "full_transcript": final_values.get("transcript_text"),
    }
    _append_log(DEFAULT_TRAINER_LOG, entry)
    PENDING_REVIEWS.pop(review_id, None)

    return _render_result_page(
        transcript_id=transcript_id,
        final_values=final_values,
        review_outcome=entry["review_outcome"],
        log_path=DEFAULT_TRAINER_LOG,
    )

