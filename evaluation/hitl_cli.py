from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langgraph.types import Command

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cbre_agent.agent import _GRAPH

DEFAULT_EVAL = HERE / "eval_transcripts_dev.json"
DEFAULT_TRAINER_LOG = HERE / "trainer_log.json"
DEFAULT_DEV_LABELS = HERE / "dev_labels.json"


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


def _load_log(path: Path) -> List[dict]:
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


def _load_dev_labels(path: Path) -> Dict[str, dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by transcript_id.")
    if "by_id" in data and isinstance(data["by_id"], dict):
        return data["by_id"]
    return data


def _print_review_screen(row: dict, state_values: dict) -> None:
    c = state_values.get("classification") or {}
    print("\n" + "=" * 72)
    print("HITL REVIEW")
    print("=" * 72)
    print(f"transcript_id: {row.get('transcript_id')}")
    print(f"caller_phone:  {row.get('caller_phone')}")
    print(f"category:      {c.get('category')}")
    print(f"subcategory:   {c.get('subcategory')}")
    print(f"risk_level:    {c.get('risk_level')}")
    print(f"building_name: {c.get('building_name')}")
    print(f"address:       {c.get('address')}")
    print(f"floor:         {c.get('floor')}")
    print("-" * 72)
    print("transcript:")
    print(state_values.get("transcript_text", ""))
    print("=" * 72)


def _prompt_override_payload() -> Dict[str, Any]:
    print("\nEnter override fields (leave blank to skip a field):")
    payload: Dict[str, Any] = {}

    override_subcategory = input("override_subcategory: ").strip()
    override_category = input("override_category: ").strip()
    override_risk_level = input("override_risk_level (LOW/MEDIUM/HIGH/EMERGENCY): ").strip()
    override_building_name = input("override_building_name: ").strip()
    override_address = input("override_address: ").strip()
    override_floor = input("override_floor: ").strip()
    override_reason = input("override_reason: ").strip()

    if override_subcategory:
        payload["override_subcategory"] = override_subcategory
    if override_category:
        payload["override_category"] = override_category
    if override_risk_level:
        payload["override_risk_level"] = override_risk_level
    if override_building_name:
        payload["override_building_name"] = override_building_name
    if override_address:
        payload["override_address"] = override_address
    if override_floor:
        payload["override_floor"] = override_floor
    if override_reason:
        payload["override_reason"] = override_reason

    if not payload:
        print("No override fields entered. Treating as approve.")
        return {"approved": True}
    payload["approved"] = False
    return payload


def _select_rows(rows: list[dict], transcript_id: Optional[str], index: Optional[int], run_all: bool) -> Iterable[tuple[int, dict]]:
    if run_all:
        return enumerate(rows)
    if transcript_id is not None:
        for i, row in enumerate(rows):
            if row.get("transcript_id") == transcript_id:
                return [(i, row)]
        raise ValueError(f"transcript_id={transcript_id!r} not found.")
    if index is None:
        index = 0
    if index < 0 or index >= len(rows):
        raise ValueError(f"index must be between 0 and {len(rows) - 1}.")
    return [(index, rows[index])]


def _derive_review_outcome(final_values: dict) -> str:
    status = ((final_values.get("hitl_decision") or {}).get("status") or "").lower()
    if status == "approved":
        return "human_approved"
    if status == "overridden":
        return "human_override"
    return "auto_routed"


def _build_auto_resume_payload_from_dev_label(transcript_id: str, pre_review: dict, dev_label: dict) -> Any:
    """Return approve payload if aligned, otherwise an override payload."""
    target = {
        "category": dev_label.get("true_category"),
        "subcategory": dev_label.get("true_subcategory"),
        "risk_level": dev_label.get("true_risk_level"),
        "building_name": dev_label.get("ground_truth_building_name"),
        "address": dev_label.get("ground_truth_address"),
        "floor": dev_label.get("ground_truth_floor"),
    }
    current = {
        "category": pre_review.get("category"),
        "subcategory": pre_review.get("subcategory"),
        "risk_level": pre_review.get("risk_level"),
        "building_name": pre_review.get("building_name"),
        "address": pre_review.get("address"),
        "floor": pre_review.get("floor"),
    }
    if current == target:
        print(f"[auto] {transcript_id}: prediction matches dev labels -> approve")
        return {"approved": True}

    payload: Dict[str, Any] = {"approved": False}
    if target["category"] is not None and current["category"] != target["category"]:
        payload["override_category"] = target["category"]
    if target["subcategory"] is not None and current["subcategory"] != target["subcategory"]:
        payload["override_subcategory"] = target["subcategory"]
    if target["risk_level"] is not None and current["risk_level"] != target["risk_level"]:
        payload["override_risk_level"] = target["risk_level"]
    if target["building_name"] is not None and current["building_name"] != target["building_name"]:
        payload["override_building_name"] = target["building_name"]
    if target["address"] is not None and current["address"] != target["address"]:
        payload["override_address"] = target["address"]
    if target["floor"] is not None and current["floor"] != target["floor"]:
        payload["override_floor"] = target["floor"]

    payload["override_reason"] = "Auto override from dev_labels.json ground truth."
    print(f"[auto] {transcript_id}: mismatch vs dev labels -> override")
    return payload


def _run_one(
    row: dict,
    row_idx: int,
    trainer_log_path: Path,
    thread_prefix: str,
    auto_from_dev_labels: bool,
    dev_labels: Optional[Dict[str, dict]],
) -> bool:
    transcript_id = row.get("transcript_id", f"row-{row_idx}")
    thread_id = f"{thread_prefix}-{transcript_id}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    _GRAPH.invoke(_build_initial_state(row["turns"], row.get("caller_phone")), config)
    paused_state = _GRAPH.get_state(config)
    pre_review = dict(paused_state.values.get("classification") or {})

    if paused_state.next:
        if auto_from_dev_labels:
            if not dev_labels:
                raise ValueError("--auto-from-dev-labels requires --dev-labels.")
            transcript_id_key = row.get("transcript_id")
            dev_label = dev_labels.get(transcript_id_key or "")
            if not dev_label:
                raise ValueError(f"No dev label found for transcript_id={transcript_id_key!r}")
            resume_payload = _build_auto_resume_payload_from_dev_label(
                transcript_id=str(transcript_id_key),
                pre_review=pre_review,
                dev_label=dev_label,
            )
        else:
            _print_review_screen(row, paused_state.values)

            while True:
                decision = input("Decision [approve/override]: ").strip().lower()
                if decision in {"approve", "a", "yes", "y"}:
                    resume_payload = {"approved": True}
                    break
                if decision == "override":
                    resume_payload = _prompt_override_payload()
                    break
                print("Please enter approve or override.")

        _GRAPH.invoke(Command(resume=resume_payload), config)

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
    _append_log(trainer_log_path, entry)
    print(
        f"Logged {transcript_id}: {entry['review_outcome']} "
        f"-> {trainer_log_path}"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporary interactive HITL runner that appends to trainer_log.json."
    )
    parser.add_argument("--eval", default=str(DEFAULT_EVAL), help="Path to transcripts JSON.")
    parser.add_argument("--trainer-log", default=str(DEFAULT_TRAINER_LOG), help="Path to trainer log JSON.")
    parser.add_argument("--transcript-id", default=None, help="Run a single transcript by transcript_id.")
    parser.add_argument("--index", type=int, default=None, help="Run a single transcript by 0-based index.")
    parser.add_argument("--all", action="store_true", help="Run all transcripts in --eval.")
    parser.add_argument("--limit", type=int, default=None, help="Max transcripts to process (use with --all).")
    parser.add_argument("--thread-prefix", default="hitl", help="Prefix for generated thread IDs.")
    parser.add_argument(
        "--auto-from-dev-labels",
        action="store_true",
        help="Auto approve/override interrupted cases using dev_labels.json ground truth.",
    )
    parser.add_argument(
        "--dev-labels",
        default=str(DEFAULT_DEV_LABELS),
        help="Path to dev labels JSON (used with --auto-from-dev-labels).",
    )
    args = parser.parse_args()

    rows = json.loads(Path(args.eval).read_text())
    if not isinstance(rows, list):
        raise ValueError("--eval file must contain a JSON list of transcripts.")

    selected = list(_select_rows(rows, args.transcript_id, args.index, args.all))
    if args.limit is not None:
        selected = selected[: args.limit]

    dev_labels: Optional[Dict[str, dict]] = None
    if args.auto_from_dev_labels:
        dev_labels = _load_dev_labels(Path(args.dev_labels))

    trainer_log_path = Path(args.trainer_log)
    written = 0
    skipped = 0
    for row_idx, row in selected:
        ok = _run_one(
            row=row,
            row_idx=row_idx,
            trainer_log_path=trainer_log_path,
            thread_prefix=args.thread_prefix,
            auto_from_dev_labels=args.auto_from_dev_labels,
            dev_labels=dev_labels,
        )
        if ok:
            written += 1
        else:
            skipped += 1

    print("\nDone.")
    print(f"Processed: {len(selected)}")
    print(f"Logged:    {written}")
    print(f"Skipped:   {skipped}")
    print(f"Trainer log path: {trainer_log_path}")


if __name__ == "__main__":
    main()
