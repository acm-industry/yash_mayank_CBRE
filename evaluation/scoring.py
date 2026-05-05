"""
Auto-grader for the Human-in-the-Loop RAG agent.

Inputs (defaults shown):
    --eval         evaluation/eval_transcripts_dev.json   (200 labeled dev transcripts)
    --ground-truth evaluation/dev_labels.json             (matching labels for dev)
    --predictions  predictions.json                       (your agent's output)

Student predictions must be a JSON array of dicts keyed by transcript_id:
    {
      "transcript_id":         "EVAL-0001",
      "category":              "PLUMBING",
      "subcategory":           "pipe_leak",
      "risk_level":            "MEDIUM",
      "needs_human_review":    false,
      "needs_clarification":   false,
      "building_name":         "Pacific Ridge Medical Plaza",
      "address":               "3200 Pacific Coast Hwy",
      "floor":                 "Floor 7",
      "dispatched_vendor_id":  "v_002",
      "dispatched_emergency_services": false,
      "call_summary":          "2-3 sentence operator-style narrative.",
      "trainer_log": {
          "full_transcript":  "...",
          "ai_prediction":    { ... },
          "human_override":   null,
          "final_decision":   { ... }
      }
    }

Grading axes (weighted, total = 100%):
    1. Category accuracy            10%
    2. Subcategory accuracy         15%
    3. Risk-level accuracy          10%
    4. HITL trigger F1              15%   (knowing when to escalate)
    5. Clarification F1              5%
    6. Field-extraction accuracy    10%   (building + address + floor)
    7. Vendor-match accuracy        10%
    8. Auto-resolution rate         10%   (% of routine cases auto-resolved)
    9. Call-summary present          5%
   10. Trainer-log present          10%

Hard-fail penalty:
    -5 points per `false-911` dispatch on an over_escalation_trap case.

Prints per-case-type breakdown and final composite (%).

Run (student, on dev):
    python evaluation/scoring.py --predictions my_preds.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(p: Path) -> Any:
    return json.loads(Path(p).read_text())

def index_preds(preds: List[dict]) -> Dict[str, dict]:
    return {p["transcript_id"]: p for p in preds if "transcript_id" in p}

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def eq_ci(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return a.strip().lower() == b.strip().lower()

def f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    score = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(score, 4)}

# ---------------------------------------------------------------------------
# Core grading
# ---------------------------------------------------------------------------

AXIS_WEIGHTS = {
    "category":            10,
    "subcategory":         15,
    "risk_level":          10,
    "hitl_f1":             15,
    "clarif_f1":            5,
    "fields":              10,
    "vendor":              10,
    "auto_resolution":     10,
    "call_summary":         5,
    "trainer_log":         10,
}
FALSE_911_PENALTY_PER_CASE = 5  # points deducted per false-911 dispatch

def grade(preds_map: Dict[str, dict],
          eval_rows: List[dict],
          gt_map: Dict[str, dict]) -> Dict[str, Any]:
    n_total = len(eval_rows)
    axes = {k: 0 for k in [
        "cat_hit", "sub_hit", "risk_hit", "field_hit", "vendor_hit",
        "auto_resolved_hits", "auto_resolved_eligible",
        "call_summary_hit", "trainer_log_hit",
    ]}
    hitl = {"tp": 0, "fp": 0, "fn": 0}
    clarif = {"tp": 0, "fp": 0, "fn": 0}
    false_911 = 0
    by_case: Dict[str, Counter] = defaultdict(Counter)
    confusion: Counter = Counter()
    missing_preds = 0

    for row in eval_rows:
        tid = row["transcript_id"]
        gt = gt_map.get(tid)
        pr = preds_map.get(tid)
        if gt is None:
            continue
        ct = gt["case_type"]
        by_case[ct]["n"] += 1
        if pr is None:
            missing_preds += 1
            continue

        # 1. category
        if eq_ci(pr.get("category"), gt["true_category"]):
            axes["cat_hit"] += 1; by_case[ct]["cat"] += 1
        # 2. subcategory
        if eq_ci(pr.get("subcategory"), gt["true_subcategory"]):
            axes["sub_hit"] += 1; by_case[ct]["sub"] += 1
        else:
            confusion[(gt["true_subcategory"], pr.get("subcategory") or "<missing>")] += 1
        # 3. risk level
        if eq_ci(pr.get("risk_level"), gt["true_risk_level"]):
            axes["risk_hit"] += 1; by_case[ct]["risk"] += 1

        # 4. HITL F1
        pr_h = bool(pr.get("needs_human_review", False))
        gt_h = bool(gt["true_needs_human_review"])
        if pr_h and gt_h: hitl["tp"] += 1
        elif pr_h and not gt_h: hitl["fp"] += 1
        elif (not pr_h) and gt_h: hitl["fn"] += 1

        # 5. clarification F1
        pr_c = bool(pr.get("needs_clarification", False))
        gt_c = bool(gt["true_needs_clarification"])
        if pr_c and gt_c: clarif["tp"] += 1
        elif pr_c and not gt_c: clarif["fp"] += 1
        elif (not pr_c) and gt_c: clarif["fn"] += 1

        # 6. field extraction (building/address/floor)
        fields_ok = (eq_ci(pr.get("building_name"), gt["ground_truth_building_name"])
                     and eq_ci(pr.get("address"),       gt["ground_truth_address"])
                     and eq_ci(pr.get("floor"),         gt["ground_truth_floor"]))
        if fields_ok:
            axes["field_hit"] += 1; by_case[ct]["field"] += 1

        # 7. vendor match
        dispatched = pr.get("dispatched_vendor_id")
        acceptable = gt.get("acceptable_vendor_ids", [])
        if gt.get("unroutable", False):
            # correct answer is: don't dispatch + escalate to human
            if (dispatched in (None, "")) and pr_h:
                axes["vendor_hit"] += 1; by_case[ct]["vendor"] += 1
        else:
            if dispatched and dispatched in acceptable:
                axes["vendor_hit"] += 1; by_case[ct]["vendor"] += 1

        # 8. auto-resolution rate — eligible = gt.should_auto_resolve (True)
        if gt.get("should_auto_resolve", False):
            axes["auto_resolved_eligible"] += 1
            # auto-resolved = pred didn't flag human review + didn't flag clarification
            if not pr_h and not pr_c:
                axes["auto_resolved_hits"] += 1
                by_case[ct]["auto"] += 1

        # 9. call_summary presence + non-trivial length
        cs = (pr.get("call_summary") or "").strip()
        if len(cs) >= 30:
            axes["call_summary_hit"] += 1; by_case[ct]["summary"] += 1

        # 10. trainer_log presence + minimum structure
        tl = pr.get("trainer_log")
        if isinstance(tl, dict) and "full_transcript" in tl and "final_decision" in tl:
            axes["trainer_log_hit"] += 1; by_case[ct]["trainer"] += 1

        # hard-fail: false-911 on over_escalation_trap
        if gt.get("is_over_escalation_trap") and pr.get("dispatched_emergency_services") is True:
            false_911 += 1

    # Final axis percentages
    def pct(hits: int, n: int) -> float:
        return 100.0 * hits / n if n else 0.0

    hitl_f1  = f1(hitl["tp"], hitl["fp"], hitl["fn"])
    clarif_f1 = f1(clarif["tp"], clarif["fp"], clarif["fn"])
    auto_pct = pct(axes["auto_resolved_hits"], axes["auto_resolved_eligible"])

    axis_pct = {
        "category":        pct(axes["cat_hit"],         n_total),
        "subcategory":     pct(axes["sub_hit"],         n_total),
        "risk_level":      pct(axes["risk_hit"],        n_total),
        "hitl_f1":         hitl_f1["f1"] * 100,
        "clarif_f1":       clarif_f1["f1"] * 100,
        "fields":          pct(axes["field_hit"],       n_total),
        "vendor":          pct(axes["vendor_hit"],      n_total),
        "auto_resolution": auto_pct,
        "call_summary":    pct(axes["call_summary_hit"], n_total),
        "trainer_log":     pct(axes["trainer_log_hit"],  n_total),
    }

    # Composite (weighted)
    composite = sum(axis_pct[k] * (w/100.0) for k, w in AXIS_WEIGHTS.items())
    penalty = false_911 * FALSE_911_PENALTY_PER_CASE
    composite_after = max(0.0, composite - penalty)

    return {
        "n_total":         n_total,
        "n_missing":       missing_preds,
        "axes":            axes,
        "hitl":            hitl_f1,
        "clarification":   clarif_f1,
        "axis_pct":        axis_pct,
        "by_case":         {k: dict(v) for k, v in by_case.items()},
        "confusion_top":   confusion.most_common(15),
        "false_911_count": false_911,
        "penalty":         penalty,
        "composite_raw":   round(composite, 2),
        "composite":       round(composite_after, 2),
    }


def print_report(res: Dict[str, Any]) -> None:
    n = res["n_total"]
    axp = res["axis_pct"]
    print(f"\nGraded {n} transcripts ({res['n_missing']} predictions missing).\n")
    print("Per-axis percentages:")
    for k, w in AXIS_WEIGHTS.items():
        print(f"  {k:<20} {axp[k]:6.2f}%  (weight {w:>2}%)")

    print(f"\nHITL   precision={res['hitl']['precision']:.3f}  recall={res['hitl']['recall']:.3f}  F1={res['hitl']['f1']:.3f}")
    print(f"Clarif precision={res['clarification']['precision']:.3f}  recall={res['clarification']['recall']:.3f}  F1={res['clarification']['f1']:.3f}")

    print(f"\nAuto-resolution: {res['axes']['auto_resolved_hits']} / {res['axes']['auto_resolved_eligible']} eligible routine cases")
    print(f"False-911 dispatches on over-escalation traps: {res['false_911_count']}  (penalty -{res['penalty']:.2f})")

    print("\nBy case_type:")
    for ct, stats in res["by_case"].items():
        n_ct = stats.get("n", 0)
        print(f"  {ct:<22}  n={n_ct:>4}  "
              f"cat={stats.get('cat',0)}  sub={stats.get('sub',0)}  risk={stats.get('risk',0)}  "
              f"field={stats.get('field',0)}  vendor={stats.get('vendor',0)}  "
              f"auto={stats.get('auto',0)}")

    print("\nTop confusions (true → predicted, count):")
    for (t, p), cnt in res["confusion_top"]:
        print(f"  {t:<22} → {p:<22} {cnt}")

    print(f"\nComposite (before penalty): {res['composite_raw']:.2f}")
    print(f"Final composite:            {res['composite']:.2f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--eval",
                    default=str(here / "eval_transcripts_dev.json"),
                    help="transcripts file to score against (default: dev)")
    ap.add_argument("--ground-truth", "-gt",
                    default=str(here / "dev_labels.json"),
                    help="labels file matching --eval (default: dev_labels.json)")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    args = ap.parse_args()

    eval_rows = load_json(Path(args.eval))
    gt_raw    = load_json(Path(args.ground_truth))
    gt_map    = gt_raw.get("by_id", gt_raw)  # tolerate older flat layout
    preds     = load_json(Path(args.predictions))
    preds_map = index_preds(preds)

    res = grade(preds_map, eval_rows, gt_map)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print_report(res)


if __name__ == "__main__":
    main()
