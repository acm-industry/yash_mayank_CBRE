"""
Batch harness — run a student `classify()` against every eval transcript,
writing predictions.json ready for scoring.py.

Usage:
    # agent_module must expose classify(turns, caller_phone) -> dict
    python evaluation/run_eval.py --agent my_agent:classify --out predictions.json
    # By default runs on eval_transcripts_dev.json (labeled, visible).
    # For the final test run:
    python evaluation/run_eval.py --agent my_agent:classify \
        --eval evaluation/eval_transcripts_test.json --out predictions.json

The agent callable receives:
    turns: list[dict]           structured dialogue ([{"speaker":"agent"|"caller", "text":...}, ...])
    caller_phone: str | None    phone number if the line identified the caller

Each eval row in the JSON file also carries `caller_known_in_profiles: bool`
(true when caller_phone is in caller_profiles.json). The harness does not
forward this flag — look it up yourself from caller_profiles.json if needed.

(Flatten the turns yourself however you like — e.g. "\n".join(f"[{t['speaker'].upper()}] {t['text']}" for t in turns).
 We don't pre-flatten for you.)

It must return a dict with the fields scoring.py expects:
    category, subcategory, risk_level, needs_human_review, needs_clarification,
    building_name, address, floor, dispatched_vendor_id,
    dispatched_emergency_services, call_summary, trainer_log

Missing fields → scoring.py will count as incorrect for those axes.

Agents may attach ``_latency`` (e.g. ``cbre_agent.agent``) — when present, a
short aggregate summary is printed after the run.
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_EVAL = HERE / "eval_transcripts_dev.json"
DEFAULT_TIMEOUT_S = 30.0


def load_agent(spec: str) -> Callable[..., Dict[str, Any]]:
    """spec = 'my_package.agent:classify' or 'path/to/agent.py:classify'"""
    if ":" not in spec:
        raise ValueError(f"--agent must be 'module_or_path:callable', got: {spec}")
    mod_spec, fn_name = spec.rsplit(":", 1)
    # Make the current working directory importable so students can place their
    # agent module at the bundle root and run `python evaluation/run_eval.py ...`.
    cwd = str(Path.cwd().resolve())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    if mod_spec.endswith(".py") or "/" in mod_spec:
        p = Path(mod_spec).resolve()
        sys.path.insert(0, str(p.parent))
        mod = importlib.import_module(p.stem)
    else:
        mod = importlib.import_module(mod_spec)
    return getattr(mod, fn_name)


def run(agent: Callable[..., Dict[str, Any]],
        eval_rows: List[dict],
        limit: Optional[int] = None,
        verbose: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S) -> List[dict]:
    preds: List[dict] = []
    n_timeouts = 0
    t0 = time.time()
    # One worker — calls run sequentially, but the future lets us bound each one.
    with ThreadPoolExecutor(max_workers=1) as pool:
        for i, row in enumerate(eval_rows):
            if limit is not None and i >= limit:
                break
            tid = row["transcript_id"]
            turns = row["turns"]
            phone = row.get("caller_phone")
            future = pool.submit(agent, turns, phone)
            try:
                result = future.result(timeout=timeout_s)
            except FuturesTimeoutError:
                n_timeouts += 1
                if verbose:
                    print(f"  [{tid}] timed out after {timeout_s:.0f}s — recording missing prediction")
                # Thread can't be killed; it'll keep running in the background
                # until the agent returns, but we move on.
                result = {"error": f"timeout after {timeout_s:.0f}s"}
            except Exception as e:
                if verbose:
                    traceback.print_exc()
                result = {"error": str(e)}
            if not isinstance(result, dict):
                result = {"error": f"agent returned {type(result).__name__}, expected dict"}
            result["transcript_id"] = tid
            preds.append(result)
            if verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed else 0
                print(f"  [{i+1}/{len(eval_rows)}]  {rate:.1f}/s")
    if n_timeouts:
        print(f"  {n_timeouts} call(s) exceeded the {timeout_s:.0f}s timeout and were recorded as missing.")
    return preds


def print_latency_aggregate(preds: List[dict]) -> None:
    """When predictions include ``_latency`` from the agent, print mean/median wall time and per-node means."""
    rows: List[Dict[str, Any]] = []
    for p in preds:
        if not isinstance(p, dict) or "error" in p:
            continue
        L = p.get("_latency")
        if isinstance(L, dict) and "wall_clock_seconds" in L:
            rows.append(L)
    if not rows:
        print(
            "\nLatency: no `_latency` field on predictions "
            "(instrumented agents add this for per-node breakdown)."
        )
        return

    walls = [float(r["wall_clock_seconds"]) for r in rows]
    n = len(rows)
    mean_w = statistics.mean(walls)
    med_w = statistics.median(walls)

    with_rewrite = sum(1 for r in rows if int(r.get("rewrite_rag_query_visits") or 0) > 0)
    grader_visits = [int(r.get("grader_gate_visits") or 0) for r in rows]

    node_names = sorted({k for r in rows for k in (r.get("seconds_by_node") or {})})
    node_means: Dict[str, float] = {}
    for name in node_names:
        vals = [(r.get("seconds_by_node") or {}).get(name, 0.0) for r in rows]
        node_means[name] = statistics.mean(vals)

    print("\n--- Latency aggregate (`_latency` on predictions) ---")
    print(f"  transcripts:              {n}")
    print(f"  wall_clock mean / median: {mean_w:.3f}s / {med_w:.3f}s")
    print(f"  grader_gate visits/call:  mean {statistics.mean(grader_visits):.2f}  (max {max(grader_visits)})")
    print(f"  calls with ≥1 rewrite:    {with_rewrite} ({100.0 * with_rewrite / n:.1f}%)")
    print("  mean seconds_by_node (summed per call, then averaged across calls):")
    for name in sorted(node_means, key=lambda k: -node_means[k]):
        tag = ""
        if name in ("grader_gate", "rewrite_rag_query"):
            tag = "  ← RAG gate / rewrite loop"
        print(f"    {name:22s} {node_means[name]:.3f}s{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True,
                    help="module:function, e.g. my_agent:classify or ./my_agent.py:classify")
    ap.add_argument("--eval", default=str(DEFAULT_EVAL))
    ap.add_argument("--out",  default="predictions.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of transcripts (debug).")
    ap.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S,
                    help=f"Per-call timeout in seconds (default {DEFAULT_TIMEOUT_S:.0f}). "
                         "Calls exceeding this are recorded as missing predictions.")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    agent = load_agent(args.agent)
    eval_rows = json.loads(Path(args.eval).read_text())

    print(f"Running agent on {len(eval_rows)} transcripts "
          f"(limit={args.limit}, timeout={args.timeout_s:.0f}s) ...")
    preds = run(agent, eval_rows, limit=args.limit,
                verbose=args.verbose, timeout_s=args.timeout_s)
    Path(args.out).write_text(json.dumps(preds, indent=2))
    print(f"Wrote {len(preds)} predictions → {args.out}")
    print_latency_aggregate(preds)


if __name__ == "__main__":
    main()
