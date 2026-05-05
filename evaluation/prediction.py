"""
Prediction schema for the HITL-RAG agent.

Use this as the return contract for your `classify()` function. The scorer
(`scoring.py`) and the batch harness (`run_eval.py`) both read predictions
in this shape; missing fields will be counted as incorrect for the
corresponding axis.

You don't have to import this dataclass — returning a plain dict with these
keys is fine. It's provided for type safety and IDE autocomplete.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "EMERGENCY"]


@dataclass
class TrainerLog:
    """The per-call record your agent emits for offline review / retraining.

    - `full_transcript`: the flattened call text (strings joined)
    - `ai_prediction`:   what the model produced BEFORE any human override
    - `human_override`:  `None` if no override; else the corrected fields
    - `final_decision`:  what actually went out (same as ai_prediction unless
                         a human overrode)
    """
    full_transcript: str
    ai_prediction:   Dict[str, Any]
    human_override:  Optional[Dict[str, Any]]
    final_decision:  Dict[str, Any]


@dataclass
class Prediction:
    transcript_id:                   str
    category:                        str                     # e.g. "PLUMBING"
    subcategory:                     str                     # e.g. "pipe_leak"
    risk_level:                      RiskLevel
    needs_human_review:              bool
    needs_clarification:             bool
    building_name:                   Optional[str]
    address:                         Optional[str]
    floor:                           Optional[str]
    dispatched_vendor_id:            Optional[str]           # None = no dispatch (escalated)
    dispatched_emergency_services:   bool                    # True = you called 911
    call_summary:                    str                     # 2-3 sentence operator-style narrative
    trainer_log:                     TrainerLog              # see TrainerLog above

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
