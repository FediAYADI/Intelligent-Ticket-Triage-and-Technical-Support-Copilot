"""Retriever + answer generator for the RAG copilot.

Backends:
  - extractive (default, CPU-friendly): the suggested answer is assembled
    from the top retrieved historical resolution / FAQ answer.
  - hf: optional generative model (Phi-3.5-mini / Gemma-2-2B) via
    transformers, only if the hardware permits. Configured in config.yaml.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config


class KnowledgeBase:
    """Loads the FAISS index + document store once."""

    def __init__(self, index_dir: str | Path | None = None, cfg: dict | None = None):
        cfg = cfg or load_config()
        self.cfg = cfg
        self.rcfg = cfg["rag"]
        if index_dir is None:
            index_dir = Path(cfg["paths"]["models_dir"]) / "embeddings" / "rag_index"
        self.index_dir = Path(index_dir)
        self.docs = pd.read_parquet(self.index_dir / "kb_docs.parquet")
        with open(self.index_dir / "meta.json") as f:
            self.meta = json.load(f)

        self._embedder = None
        self._index = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.meta["embedding_model"])
        return self._embedder

    @property
    def index(self):
        if self._index is None:
            import faiss
            self._index = faiss.read_index(str(self.index_dir / "kb.index"))
        return self._index

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return top-k docs with cosine similarity scores, best first."""
        top_k = top_k or int(self.rcfg["top_k"])
        q = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss = __import__("faiss")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            row = self.docs.iloc[int(idx)]
            results.append({
                "doc_id": row["doc_id"],
                "kind": row["kind"],
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
                "team": row["team"],
                "similarity": float(score),
            })
        return results


class Copilot:
    """Retrieval + generation + grounding check."""

    def __init__(self, kb: KnowledgeBase | None = None, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.rcfg = self.cfg["rag"]
        self.kb = kb or KnowledgeBase(cfg=self.cfg)
        self._llm = None

    # ---------------- generation backends ----------------
    def _extractive_answer(self, query: str, sources: list[dict]) -> tuple[str, bool]:
        """Build a suggested reply from retrieved resolutions.

        Returns (answer_text, grounded). grounded=False when no reliable
        source exists — the UI must then show 'insufficient information'.
        """
        min_sim = float(self.rcfg["min_similarity"])
        good = [s for s in sources if s["similarity"] >= min_sim]
        if not good:
            return (
                "I could not find a reliable answer in the knowledge base for "
                "this ticket. Please consult a senior agent or escalate.",
                False,
            )
        best = good[0]
        if best["kind"] == "faq":
            body = best["answer"]
        else:
            body = (
                f"Based on a similar resolved ticket ({best['doc_id']}), "
                f"the recommended resolution was: {best['answer']}"
            )
        grounded = best["similarity"] >= min_sim + 0.15
        return body, grounded

    def _hf_answer(self, query: str, sources: list[dict]) -> tuple[str, bool]:
        """Generative backend (Phi-3.5-mini / Gemma-2-2B) — GPU recommended."""
        if self._llm is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            name = self.rcfg.get("llm_model_name", "microsoft/Phi-3.5-mini-instruct")
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self._llm = (tok, model)
        tok, model = self._llm
        context = "\n\n".join(
            f"[Source {i+1}: {s['question'][:200]}]\n{s['answer'][:400]}"
            for i, s in enumerate(sources[:3])
        )
        prompt = (
            "You are a technical support copilot. Answer the ticket using ONLY "
            "the provided sources. If the sources do not contain the answer, "
            "say so explicitly.\n\n"
            f"SOURCES:\n{context}\n\nTICKET: {query}\n\nSUGGESTED RESPONSE:"
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=int(self.rcfg.get("max_new_tokens", 256)),
            do_sample=False,
        )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip(), True

    def answer(self, ticket_text: str) -> dict:
        """Full RAG pipeline: retrieve -> generate -> return sources."""
        sources = self.kb.search(ticket_text)
        backend = self.rcfg.get("llm_backend", "extractive")
        if backend == "hf":
            try:
                text, grounded = self._hf_answer(ticket_text, sources)
            except Exception:
                text, grounded = self._extractive_answer(ticket_text, sources)
        else:
            text, grounded = self._extractive_answer(ticket_text, sources)
        return {
            "answer": text,
            "grounded": grounded,
            "sources": sources,
            "backend": backend,
        }
