from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_dispatch_message(prediction: Dict[str, Any]) -> str:
    category = prediction.get("category") or "maintenance"
    subcategory = prediction.get("subcategory") or "issue"
    risk_level = prediction.get("risk_level") or "LOW"
    building_name = prediction.get("building_name") or "the property"
    floor = prediction.get("floor") or "unspecified floor"
    vendor_id = prediction.get("dispatched_vendor_id") or "dispatch team"

    return (
        f"We classified this as {category} / {subcategory}, risk {risk_level}. "
        f"Location noted as {building_name}, {floor}. "
        f"Dispatch target is {vendor_id}."
    )


def needs_follow_up(prediction: Dict[str, Any], max_turns: int, current_turn_count: int) -> bool:
    if current_turn_count >= max_turns:
        return False
    return bool(prediction.get("needs_clarification"))


def persist_call_artifact(
    out_dir: Path,
    call_sid: str,
    caller_phone: Optional[str],
    turns: List[Dict[str, str]],
    prediction: Dict[str, Any],
    timings: Dict[str, float],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "call_sid": call_sid,
        "caller_phone": caller_phone,
        "created_at_utc": _utc_now_iso(),
        "turns": turns,
        "prediction": prediction,
        "timings_seconds": timings,
    }
    path = out_dir / f"{call_sid}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path

