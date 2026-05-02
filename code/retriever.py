"""
Corpus indexer and retriever.

Walks all .md files in data/, embeds them with sentence-transformers, and saves
a flat numpy index to disk.  Retrieval is a brute-force cosine similarity —
fast enough for ~3000 chunks and avoids ChromaDB's HNSW rebuild stalls on macOS.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# Prevents HuggingFace tokenizer deadlock on macOS (fork + semaphore conflict)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = Path(__file__).parent / ".cache"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP = 200
ENCODE_BATCH = 32


def _path_metadata(filepath: Path) -> dict:
    parts = filepath.relative_to(DATA_DIR).parts
    company = parts[0]
    product_area = parts[1] if len(parts) > 2 else parts[0]
    return {
        "company": company,
        "product_area": product_area,
        "filepath": str(filepath),
    }


def _chunk(text: str) -> list[str]:
    if len(text) <= CHUNK_MAX_CHARS:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_MAX_CHARS, len(text))
        if end < len(text):
            nl = text.rfind("\n", start + CHUNK_OVERLAP, end)
            if nl > start:
                end = nl
        chunks.append(text[start:end])
        next_start = end - CHUNK_OVERLAP
        if next_start <= start:
            # No forward progress — emit the rest and stop
            if end < len(text):
                chunks.append(text[end:])
            break
        start = next_start
    return chunks


class Retriever:
    def __init__(self, force_reindex: bool = False):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        emb_path = CACHE_DIR / "embeddings.npy"
        meta_path = CACHE_DIR / "metadata.json"

        if force_reindex or not emb_path.exists() or not meta_path.exists():
            # Read all docs BEFORE loading the model — model threads conflict with file I/O on macOS
            all_texts, all_meta = self._collect_chunks()
            self.model = SentenceTransformer(EMBED_MODEL)
            self._embed_and_save(all_texts, all_meta, emb_path, meta_path)
        else:
            self.model = SentenceTransformer(EMBED_MODEL)

        self._embeddings = np.load(str(emb_path))
        with open(meta_path, encoding="utf-8") as f:
            self._meta = json.load(f)

        print(f"Using cached index ({len(self._meta)} chunks).")

    def _collect_chunks(self) -> tuple[list[str], list[dict]]:
        docs = sorted(DATA_DIR.rglob("*.md"))
        print(f"Reading {len(docs)} corpus documents…", flush=True)
        all_texts, all_meta = [], []
        n = len(docs)
        for idx, fp in enumerate(docs):
            text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            meta = _path_metadata(fp)
            for chunk in _chunk(text):
                all_texts.append(chunk)
                all_meta.append({**meta, "text": chunk})
            if idx % 200 == 0:
                print(f"  {idx}/{n} docs read ({len(all_texts)} chunks)", flush=True)
        print(f"Read complete: {len(all_texts)} chunks from {n} docs.", flush=True)
        return all_texts, all_meta

    def _embed_and_save(
        self,
        all_texts: list[str],
        all_meta: list[dict],
        emb_path: Path,
        meta_path: Path,
    ) -> None:
        print(f"Embedding {len(all_texts)} chunks…", flush=True)
        all_embeddings = []
        for i in range(0, len(all_texts), ENCODE_BATCH):
            batch = all_texts[i : i + ENCODE_BATCH]
            embs = self.model.encode(batch, batch_size=ENCODE_BATCH, show_progress_bar=False)
            all_embeddings.append(embs)
            if i % (ENCODE_BATCH * 10) == 0:
                pct = int(100 * i / len(all_texts))
                print(f"  embed {pct}% ({i}/{len(all_texts)})", flush=True)

        embeddings = np.vstack(all_embeddings).astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)

        np.save(str(emb_path), embeddings)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(all_meta, f, ensure_ascii=False)
        print(f"Index saved: {len(all_meta)} chunks → {CACHE_DIR}", flush=True)

    def retrieve(
        self,
        query: str,
        company: Optional[str] = None,
        n: int = 5,
    ) -> list[dict]:
        q_emb = self.model.encode([query], show_progress_bar=False)[0].astype("float32")
        q_emb = q_emb / max(np.linalg.norm(q_emb), 1e-10)

        scores = self._embeddings @ q_emb  # cosine similarity (normalised)

        # Apply company filter by masking
        if company and company.lower() not in ("none", ""):
            co = company.lower()
            mask = np.array([1.0 if m["company"] == co else -2.0 for m in self._meta])
            scores = scores + mask  # push non-matching well below threshold

        top_idx = np.argpartition(scores, -n)[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            {
                "content": self._meta[i]["text"],
                "company": self._meta[i]["company"],
                "product_area": self._meta[i]["product_area"],
                "filepath": self._meta[i]["filepath"],
                "score": round(float(scores[i]), 4),
            }
            for i in top_idx
        ]
