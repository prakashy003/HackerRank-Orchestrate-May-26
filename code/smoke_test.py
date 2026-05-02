"""
Quick smoke test — runs the first 3 sample tickets and prints results.
Use this to verify setup before running the full batch.

    python code/smoke_test.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set. Edit .env and add your key.")
    sys.exit(1)

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from retriever import Retriever
from agent import triage_ticket

SAMPLE_CSV = REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"
N = 3

retriever = Retriever()
df = pd.read_csv(SAMPLE_CSV).head(N)

print(f"\nRunning {N} sample tickets…\n{'=' * 70}\n")

for i, row in df.iterrows():
    issue = str(row.get("Issue", ""))
    subject = str(row.get("Subject", ""))
    company = str(row.get("Company", ""))

    result = triage_ticket(issue, subject, company, retriever)

    expected_status = str(row.get("Status", "")).lower()
    expected_type = str(row.get("Request Type", "")).lower()

    status_match = "✓" if result["status"] == expected_status else "✗"
    type_match = "✓" if result["request_type"] == expected_type else "✗"

    print(f"Ticket #{i + 1}: {subject[:60]}")
    print(f"  Company     : {company}")
    print(f"  Status      : {result['status']:12} (expected: {expected_status}) {status_match}")
    print(f"  Request type: {result['request_type']:16} (expected: {expected_type}) {type_match}")
    print(f"  Product area: {result['product_area']}")
    print(f"  Response    : {result['response'][:120]}…")
    print(f"  Justification: {result['justification'][:100]}")
    print()
