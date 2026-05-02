"""
Entry point for the HackerRank Orchestrate support triage agent.

Usage:
    python main.py                        # run on support_tickets.csv → output.csv
    python main.py --sample               # run on sample_support_tickets.csv (for testing)
    python main.py --reindex              # force rebuild of the vector index
    python main.py --input PATH --output PATH
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Resolve paths relative to repo root, not the code/ directory
REPO_ROOT = Path(__file__).parent.parent
INPUT_DEFAULT = REPO_ROOT / "support_tickets" / "support_tickets.csv"
SAMPLE_CSV = REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"
OUTPUT_DEFAULT = REPO_ROOT / "support_tickets" / "output.csv"

OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]


def parse_args():
    p = argparse.ArgumentParser(description="Support triage agent")
    p.add_argument("--input", default=str(INPUT_DEFAULT))
    p.add_argument("--output", default=str(OUTPUT_DEFAULT))
    p.add_argument("--reindex", action="store_true", help="Rebuild the vector index from scratch")
    p.add_argument("--sample", action="store_true", help="Run against sample_support_tickets.csv")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        print("  Copy .env.example → .env and add your key, or export the variable.")
        sys.exit(1)

    if args.sample:
        args.input = str(SAMPLE_CSV)

    # Import here so missing deps surface with a clean error before heavy work
    from retriever import Retriever
    from agent import triage_ticket

    print("=== HackerRank Orchestrate — Support Triage Agent ===\n")
    retriever = Retriever(force_reindex=args.reindex)

    df = pd.read_csv(args.input)
    print(f"\nProcessing {len(df)} ticket(s) from {args.input} …\n")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), unit="ticket"):
        issue = str(row.get("Issue") or row.get("issue") or "")
        subject = str(row.get("Subject") or row.get("subject") or "")
        company = str(row.get("Company") or row.get("company") or "None")

        try:
            result = triage_ticket(issue, subject, company, retriever)
        except Exception as exc:
            tqdm.write(f"  ⚠  Error on ticket '{subject[:60]}': {exc}")
            result = {
                "status": "escalated",
                "product_area": "unknown",
                "response": "This ticket could not be processed automatically. A human agent will follow up.",
                "justification": f"Processing error: {exc}",
                "request_type": "product_issue",
            }

        results.append({col: result.get(col, "") for col in OUTPUT_COLUMNS})

    out_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    out_df.to_csv(args.output, index=False)
    print(f"\n✓ Done. {len(results)} rows written to {args.output}")


if __name__ == "__main__":
    main()