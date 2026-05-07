import json
import sys
import pprint

sys.path.insert(0, ".")
from cbre_agent.agent import classify

with open("evaluation/eval_transcripts_dev.json") as f:
    dev = json.load(f)

row = dev[0]
print(f"=== Transcript ID: {row['transcript_id']} ===")
print(f"Caller phone: {row.get('caller_phone')}")
print(f"Known in profiles: {row.get('caller_known_in_profiles')}")
print("\nTurns:")
for t in row["turns"]:
    print(f"  [{t['speaker'].upper()}] {t['text']}")

print("\n=== Running classify() ===\n")
result = classify(row["turns"], row.get("caller_phone"))

# Print cleanly without the full trainer log transcript
display = {k: v for k, v in result.items() if k != "trainer_log"}
pprint.pprint(display)

print("\n--- trainer_log ---")
tl = result.get("trainer_log", {})
print(f"  full_transcript (first 200 chars): {tl.get('full_transcript','')[:200]}")
print(f"  ai_prediction:  {tl.get('ai_prediction')}")
print(f"  human_override: {tl.get('human_override')}")
print(f"  final_decision: {tl.get('final_decision')}")
