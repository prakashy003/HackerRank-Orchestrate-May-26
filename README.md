# HackerRank Orchestrate — Support Triage Agent

> Built for the **HackerRank Orchestrate 24-hour Hackathon** · May 1–2, 2026

A production-grade AI agent that automatically triages customer support tickets across three product ecosystems — **HackerRank**, **Claude (Anthropic)**, and **Visa** — using Retrieval Augmented Generation (RAG) and Claude's structured tool-use API.

Achieved **100% accuracy** on both classification metrics on the provided evaluation set.

---

## What It Does

Given a raw support ticket (issue text + company), the agent:

1. **Retrieves** the most relevant documentation from a 774-file offline corpus using semantic search
2. **Reasons** over the retrieved context using Claude Haiku
3. **Outputs** a fully structured triage decision — no hallucination, strictly grounded in the corpus

Every output row contains exactly five fields:

| Field | Description | Values |
|---|---|---|
| `status` | Routing decision | `replied` or `escalated` |
| `product_area` | Support domain / category | e.g. `billing`, `account-management` |
| `response` | User-facing answer grounded in corpus | — |
| `justification` | Internal reasoning for the decision | — |
| `request_type` | Ticket classification | `product_issue`, `feature_request`, `bug`, `invalid` |

---

## Results

| Metric | Score |
|---|---|
| Status accuracy (`replied` vs `escalated`) | **10 / 10 = 100%** |
| Request-type accuracy | **10 / 10 = 100%** |
| Production tickets processed | **29 tickets, 0 errors** |

**Production output distribution:**

```
status        →  replied: 18   escalated: 11
request_type  →  product_issue: 23   bug: 5   invalid: 1
```

---

## Architecture

```
support_tickets.csv
        │
        ▼
    main.py                         ← CLI — reads CSV, writes output.csv
        │
        ▼
  Retriever.retrieve()
   ├─ Pass 1: company-filtered cosine search  (precision, n=8)
   └─ Pass 2: global cosine search            (recall blend, n=5)
        │
        │  top-8 merged corpus chunks by score
        ▼
  agent.triage_ticket()
   ├─ Builds prompt: ticket + retrieved docs as context
   ├─ Calls Claude Haiku via Anthropic tool_use API
   └─ Extracts structured JSON — guaranteed by tool_choice schema
        │
        ▼
  output.csv
```

### Component Breakdown

**`retriever.py` — The RAG Engine**
- Walks all 774 markdown files across `data/hackerrank/`, `data/claude/`, `data/visa/`
- Chunks each document at 2000 characters with 200-character overlap → **4,404 total chunks**
- Embeds every chunk using `all-MiniLM-L6-v2` (sentence-transformers, CPU-only, no API cost)
- Persists a flat numpy index: `embeddings.npy` (float32 matrix) + `metadata.json`
- Retrieval: single matrix dot product — brute-force cosine similarity, ~40ms per query
- Company filter: non-matching chunks receive a −2.0 score penalty, effectively excluding them

**`agent.py` — The Triage Brain**
- System prompt with three explicit sections: REQUEST TYPE GUIDE, ESCALATION RULES, GROUNDING RULES
- `tool_choice={"type":"tool"}` forces Claude to always return the exact 5-field schema — zero free-text, zero parse errors
- Key escalation logic: `replied` = give the user any useful info (even "contact X at Y"); `escalated` = zero actionable info, full hand-off to a human
- Temperature = 0 for deterministic, reproducible outputs

**`main.py` — The CLI**
- Reads any input CSV, runs every row through the retriever + agent, writes `output.csv`
- Flags: `--sample` (run against ground-truth sample), `--reindex` (rebuild index), `--input`/`--output` (custom paths)
- 
---

## Prompt Engineering Journey

The agent reached 100% accuracy through iterative, eval-driven prompt tuning:

| Round | Change | Status Acc | Type Acc |
|---|---|---|---|
| 1 | Initial prompt | 60% | 100% |
| 2 | Added REQUEST TYPE GUIDE (bug vs product_issue distinction) | 60% | 100% |
| 3 | Fixed incomplete index (ChromaDB → numpy, all 3 companies indexed) | 70% | 100% |
| 4 | Added CRITICAL DISTINCTION block to escalation rules | **100%** | **100%** |

**The key insight (Round 4):**

The model was treating *"the user needs a process that involves human action"* as a reason to escalate. For example — "my card was stolen, what do I do?" got escalated because fraud resolution requires a human. But the expected answer was `replied`: give the user the process/contact info from the corpus.

The fix was a single conceptual clarification added to the system prompt:

```
"replied"   = You give the user information, steps, or contact details IN YOUR RESPONSE.
              Even "please contact support at X" or "fill in this form at Y" is a REPLY.
"escalated" = You have ZERO actionable information. A human acts with no message from you.
Use "escalated" ONLY when the corpus provides zero guidance AND a human must act directly.
```

This single block fixed all remaining failures and brought accuracy to 100% on both metrics.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| LLM | Claude Haiku (`claude-haiku-4-5-20251001`) via Anthropic SDK |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector store | NumPy flat index (brute-force cosine similarity) |
| Structured output | Anthropic tool_use API (`tool_choice={"type":"tool"}`) |
| CSV processing | pandas |
| Dev tooling | Claude Code (primary AI assistant), GitHub Copilot (numpy rewrite session) |

---

## How AI Tools Were Used

This hackathon explicitly allowed — and encouraged — the use of AI coding assistants. Development was conducted collaboratively with **Claude Code** (Anthropic's CLI) as the primary tool, and **GitHub Copilot** for a focused session during the numpy rewrite.

**What the AI implemented:**
- Full codebase scaffolding from an architecture brief
- The retriever, agent, CLI, evaluator, and smoke test
- The numpy replacement for ChromaDB after the deadlock was diagnosed
- Bug fixes (eval.py counter, filepath leak in a heuristic quick-rule shortcut)

**What I designed and directed:**
- The overall RAG architecture and two-pass retrieval strategy
- The decision to investigate and replace ChromaDB after diagnosing the root cause
- Every prompt engineering iteration — I analysed the failing ticket justifications and identified the conceptual gap in the escalation logic
- The CRITICAL DISTINCTION block that fixed 30% of failures
- The decision to remove a heuristic quick-rule that was leaking internal file paths into user-facing responses

The full AI collaboration transcript was submitted as part of the hackathon deliverables, documenting the iterative back-and-forth across the 24-hour session.

---

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/prakashy003/HackerRank-Orchestrate-May-26.git
cd HackerRank-Orchestrate-May-26

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Open .env and add:  ANTHROPIC_API_KEY=sk-ant-...

# 5. Build the vector index (one-time, ~60s)
python code/build_index.py

# 6. Verify against sample (should print 10/10 on both metrics)
python code/main.py --sample

# 7. Run on production tickets
python code/main.py
# → writes to support_tickets/output.csv
```

---

## Project Structure

```
.
├── code/
│   ├── agent.py            # Triage agent — system prompt + Claude tool_use call
│   ├── retriever.py        # Numpy RAG retriever — chunk, embed, index, retrieve
│   ├── main.py             # CLI entry point
│   ├── build_index.py      # One-shot index builder
│   ├── eval.py             # Accuracy evaluator against sample ground truth
│   └── smoke_test.py       # Quick 3-ticket sanity check
├── support_tickets/
│   ├── support_tickets.csv          # Production input (29 tickets)
│   ├── sample_support_tickets.csv   # 10 tickets with ground truth (dev/eval)
│   └── output.csv                   # Agent predictions — submitted output
├── problem_statement.md             # Original hackathon challenge spec
├── evalutation_criteria.md          # Scoring rubric
├── requirements.txt
└── .env.example
```

> `data/` (the 774-file support corpus provided by HackerRank) and `code/.cache/` (generated embeddings) are gitignored. Run `python code/build_index.py` after cloning to rebuild the index locally.

---

## Key Design Decisions

**Why numpy over a proper vector database?**
ChromaDB deadlocked on macOS at doc 468/774 due to a multiprocessing semaphore conflict in PyTorch's tokenizer. Replacing it with a self-contained numpy implementation eliminated the C++ dependency entirely. Brute-force cosine similarity over 4,404 chunks takes ~40ms per query — fast enough for batch processing, zero external dependencies, and fully reproducible. For a production system with millions of documents a proper ANN index (FAISS, Pinecone) would be the right call.

**Why Claude Haiku and not a larger model?**
The task is grounded extraction and classification, not open-ended reasoning. Haiku is ~3× faster than Sonnet for batch processing and sufficient for this scope. `tool_choice={"type":"tool"}` enforces the exact output schema at the API level, so there is no free-text risk regardless of model size.

**Why two retrieval passes?**
A company-filtered pass gives precision — Visa tickets retrieve Visa docs. A global pass gives recall — it catches cross-domain documents and handles tickets where the company field is unknown or `None`. The top 8 merged by score go into the Claude prompt.

---

*Built in 24 hours · May 1–2, 2026 · HackerRank Orchestrate Hackathon*
