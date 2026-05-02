"""
Evaluate agent accuracy against the full sample_support_tickets.csv.
Prints per-field accuracy and a breakdown of mismatches.

    python code/eval.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set.")
    sys.exit(1)

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from retriever import Retriever
from agent import triage_ticket

SAMPLE_CSV = REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"

retriever = Retriever()
df = pd.read_csv(SAMPLE_CSV)

total = len(df)
status_correct = 0
type_correct = 0
mismatches = []

print(f"\nEvaluating {total} sample tickets…\n")

for i, row in tqdm(df.iterrows(), total=total, unit="ticket"):
    issue = str(row.get("Issue", ""))
    subject = str(row.get("Subject", ""))
    company = str(row.get("Company", ""))

    expected_status = str(row.get("Status", "")).strip().lower()
    expected_type = str(row.get("Request Type", "")).strip().lower()

    try:
        result = triage_ticket(issue, subject, company, retriever)
    except Exception as exc:
        tqdm.write(f"  Error on row {i}: {exc}")
        result = {"status": "escalated", "request_type": "product_issue",
                  "product_area": "", "response": "", "justification": ""}

    got_status = result["status"]
    got_type = result["request_type"]

    s_ok = got_status == expected_status
    t_ok = got_type == expected_type

    if s_ok:
        status_correct += 1
    if t_ok:
        type_correct += 1

    if not s_ok or not t_ok:
        mismatches.append({
            "row": i + 1,
            "subject": str(subject)[:50],
            "company": company,
            "expected_status": expected_status,
            "got_status": got_status,
            "status_ok": "✓" if s_ok else "✗",
            "expected_type": expected_type,
            "got_type": got_type,
            "type_ok": "✓" if t_ok else "✗",
        })

print(f"\n{'=' * 65}")
print(f"RESULTS  ({total} tickets)")
print(f"  status accuracy       : {status_correct}/{total} = {status_correct/total*100:.1f}%")
print(f"  request_type accuracy : {type_correct}/{total} = {type_correct/total*100:.1f}%")
print(f"  both correct          : {total - len(mismatches)}/{total} = {(total-len(mismatches))/total*100:.1f}%")
print(f"{'=' * 65}")

if mismatches:
    print(f"\nMismatches ({len(mismatches)}):\n")
    for m in mismatches:
        print(f"  Row {m['row']:3d} | {m['company']:<12} | {m['subject']}")
        if m["status_ok"] == "✗":
            print(f"          status : expected={m['expected_status']:<10} got={m['got_status']}")
        if m["type_ok"] == "✗":
            print(f"          type   : expected={m['expected_type']:<18} got={m['got_type']}")
