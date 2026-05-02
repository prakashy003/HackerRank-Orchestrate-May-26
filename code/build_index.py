"""
One-time index builder. Run this once before main.py or smoke_test.py.

    python code/build_index.py
"""
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from retriever import Retriever

Retriever(force_reindex=True)
print("\nIndex ready. You can now run: python code/smoke_test.py")
