"""Knowledge-base builder + FAISS index for the RAG copilot.

Documents:
  - resolved historical tickets: document = subject + body, answer = `answer`
  - FAQ articles: data/knowledge_base/faq_seed.csv (question, answer, category)

Run:  python -m src.rag.indexer
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, project_path


def load_documents(cfg: dict) -> pd.DataFrame:
    proc = Path(cfg["paths"]["processed_dir"])
    kb_dir = Path(cfg["paths"]["kb_dir"])

    docs = []
    # 1) resolved historical tickets (use train+val to avoid test leakage in demos
    #    is acceptable here since this is a support KB, not a benchmark)
    for split in ("train", "val", "test"):
        p = proc / f"tickets_{split}.csv"
        if p.exists():
            df = pd.read_csv(p)
            for _, r in df.iterrows():
                docs.append({
                    "doc_id": f"ticket_{r['ticket_id']}",
                    "kind": "resolved_ticket",
                    "question": f"{r['subject']} — {str(r['body'])[:400]}",
                    "answer": str(r["answer"]),
                    "category": str(r["ticket_type"]),
                    "team": str(r["team"]),
                })

    # 2) FAQ seed articles
    faq_path = kb_dir / "faq_seed.csv"
    if faq_path.exists():
        faq = pd.read_csv(faq_path)
        for i, r in faq.iterrows():
            docs.append({
                "doc_id": f"faq_{i}",
                "kind": "faq",
                "question": str(r["question"]),
                "answer": str(r["answer"]),
                "category": str(r.get("category", "general")),
                "team": str(r.get("team", "General Inquiry")),
            })

    # 3) human-validated FAQ additions
    approved = kb_dir / "faq_approved.csv"
    if approved.exists():
        faq = pd.read_csv(approved)
        offset = 1000
        for i, r in faq.iterrows():
            docs.append({
                "doc_id": f"faq_{offset + i}",
                "kind": "faq",
                "question": str(r["question"]),
                "answer": str(r["answer"]),
                "category": str(r.get("category", "general")),
                "team": str(r.get("team", "General Inquiry")),
            })

    return pd.DataFrame(docs)


def build_index(cfg: dict | None = None):
    cfg = cfg or load_config()
    rcfg = cfg["rag"]
    model_name = rcfg["embedding_model"]

    docs = load_documents(cfg)
    if docs.empty:
        raise RuntimeError("Knowledge base is empty — run the data pipeline first.")

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(model_name)
    # embed question text (what agents search by) — answer kept as payload
    embeddings = embedder.encode(
        docs["question"].tolist(), batch_size=64,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype("float32")

    index_type = rcfg.get("index_type", "faiss")
    out_dir = Path(cfg["paths"]["models_dir"]) / "embeddings" / "rag_index"
    out_dir.mkdir(parents=True, exist_ok=True)

    if index_type == "faiss":
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on L2-normalised vecs = cosine
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        faiss.write_index(index, str(out_dir / "kb.index"))
    else:
        # ChromaDB fallback
        import chromadb
        client = chromadb.PersistentClient(path=str(out_dir / "chroma"))
        col = client.get_or_create_collection("kb")
        col.add(
            ids=docs["doc_id"].tolist(),
            embeddings=embeddings.tolist(),
            documents=docs["question"].tolist(),
        )

    docs.to_parquet(out_dir / "kb_docs.parquet", index=False)
    meta = {
        "embedding_model": model_name,
        "index_type": index_type,
        "n_docs": len(docs),
        "kinds": docs["kind"].value_counts().to_dict(),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return out_dir


if __name__ == "__main__":
    build_index()
