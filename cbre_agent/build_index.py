"""
Build the ChromaDB vector index from historical_records.json.

Run once before using the agent:
    python your_submission/build_index.py

Embed each record as a rich document (transcript + final labels + notes) so
the retriever can surface both semantically similar calls AND correctly-resolved
examples. QA-flagged reclassified/over-escalated tickets use their *final*
(authoritative) labels in the document body so the LLM sees corrected context.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
OPERATIONAL = ROOT / "operational"
EVALUATION = ROOT / "evaluation"
CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"

BATCH_SIZE = 500  # embeddings API batch size


def _load_qa_flags() -> tuple[set[str], set[str]]:
    qa_path = EVALUATION / "qa_audit_findings.json"
    with open(qa_path) as f:
        audits = json.load(f)
    reclassified = {a["ticket_id"] for a in audits if a.get("was_reclassified")}
    over_escalated = {a["ticket_id"] for a in audits if a.get("was_over_escalated")}
    return reclassified, over_escalated


def _make_document(rec: dict, reclassified: set[str], over_escalated: set[str]) -> Document:
    tid = rec["ticket_id"]

    # Always embed final (authoritative) labels — for reclassified tickets
    # the final_* fields have already been corrected by the on-site technician.
    text_parts = [
        f"Call Transcript: {rec.get('call_recording_transcript', '').strip()}",
        f"Category: {rec.get('final_category', '')} / {rec.get('final_subcategory', '')}",
        f"Risk Level: {rec.get('final_risk_level', '')}",
        f"Building Type: {rec.get('building_type', '')}",
    ]
    if rec.get("intake_notes"):
        text_parts.append(f"Intake Notes: {rec['intake_notes']}")
    if rec.get("resolution_notes"):
        text_parts.append(f"Resolution: {rec['resolution_notes']}")

    page_content = "\n".join(text_parts)

    metadata = {
        "ticket_id": tid,
        "final_category": rec.get("final_category", ""),
        "final_subcategory": rec.get("final_subcategory", ""),
        "final_risk_level": rec.get("final_risk_level", ""),
        "building_type": rec.get("building_type", ""),
        "city": rec.get("city", ""),
        "assigned_vendor_type": rec.get("assigned_vendor_type", ""),
        "assigned_vendor_id": rec.get("assigned_vendor_id", ""),
        "was_reclassified": str(tid in reclassified),
        "was_over_escalated": str(tid in over_escalated),
    }

    return Document(page_content=page_content, metadata=metadata)


def build() -> None:
    print("Loading historical records …")
    with open(OPERATIONAL / "historical_records.json") as f:
        records = json.load(f)

    reclassified, over_escalated = _load_qa_flags()
    print(
        f"  QA flags: {len(reclassified)} reclassified, "
        f"{len(over_escalated)} over-escalated"
    )

    docs = [_make_document(r, reclassified, over_escalated) for r in records]
    print(f"  Prepared {len(docs)} documents.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print(f"Building ChromaDB index at {CHROMA_PATH} …")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    store: Chroma | None = None
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        if store is None:
            store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name="historical_records",
                persist_directory=str(CHROMA_PATH),
            )
        else:
            store.add_documents(batch)
        indexed = min(i + BATCH_SIZE, len(docs))
        print(f"  Indexed {indexed}/{len(docs)} …")

    print(f"✓  Index complete — {len(docs)} records in {CHROMA_PATH}")


if __name__ == "__main__":
    build()
