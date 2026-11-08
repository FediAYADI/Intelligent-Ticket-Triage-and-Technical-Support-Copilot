"""FAQ-gap detection.

Finds clusters of incoming ticket questions that are NOT well covered by the
knowledge base (low retrieval similarity) and suggests new FAQ entries for
human validation. Suggestions are written to
data/knowledge_base/faq_suggestions.csv and consumed by the Streamlit UI.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from src.utils import load_config


def find_uncovered_questions(copilot, ticket_texts: list[str],
                             top_k: int = 3) -> pd.DataFrame:
    """Return DataFrame of questions whose best KB similarity is low."""
    # a question is "uncovered" when even its best KB hit is below the
    # FAQ-gap similarity bar (looser than the hard insufficient-info floor)
    gap_sim = float(copilot.rcfg.get("faq_gap_similarity", 0.55))
    rows = []
    for text in ticket_texts:
        hits = copilot.kb.search(text, top_k=top_k)
        best = hits[0]["similarity"] if hits else 0.0
        if best < gap_sim:
            rows.append({"question": text, "best_similarity": round(best, 4)})
    return pd.DataFrame(rows)


def suggest_faqs(copilot, uncovered: pd.DataFrame,
                 min_cluster_size: int | None = None,
                 similarity_threshold: float | None = None) -> pd.DataFrame:
    """Cluster uncovered questions; each sizeable cluster -> FAQ suggestion."""
    cfg = copilot.cfg
    rcfg = copilot.rcfg
    min_cluster_size = int(min_cluster_size or rcfg.get("faq_gap_min_cluster_size", 5))
    sim_thr = float(similarity_threshold or rcfg.get("faq_gap_similarity", 0.55))

    if uncovered.empty or len(uncovered) < min_cluster_size:
        return pd.DataFrame(columns=[
            "question", "draft_answer", "cluster_size", "mean_similarity",
            "category", "status", "created_at",
        ])

    embedder = copilot.kb.embedder
    embs = embedder.encode(
        uncovered["question"].tolist(), batch_size=64,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    # distance = 1 - cosine similarity; cluster radius is configurable and
    # intentionally looser than the coverage bar so paraphrases group together
    cluster_dist = float(rcfg.get("faq_gap_cluster_distance", 0.60))
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=cluster_dist,
        metric="cosine", linkage="average",
    )
    labels = clustering.fit_predict(embs)
    uncovered = uncovered.assign(cluster=labels)

    suggestions = []
    today = dt.date.today().isoformat()
    for cid, grp in uncovered.groupby("cluster"):
        if len(grp) < min_cluster_size:
            continue
        # representative question = medoid (closest to centroid)
        idxs = grp.index.to_numpy()
        sub = embs[uncovered.index.get_indexer(idxs)]
        centroid = sub.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        medoid_pos = int(np.argmax(sub @ centroid))
        rep_q = str(grp.iloc[medoid_pos]["question"])[:500]
        # draft answer: best KB hit for the representative question (may be weak)
        hits = copilot.kb.search(rep_q, top_k=1)
        draft = hits[0]["answer"][:500] if hits else ""
        suggestions.append({
            "question": rep_q,
            "draft_answer": draft,
            "cluster_size": int(len(grp)),
            "mean_similarity": round(float(grp["best_similarity"].mean()), 4),
            "category": hits[0]["category"] if hits else "general",
            "status": "pending",
            "created_at": today,
        })
    return pd.DataFrame(suggestions)


def save_suggestions(suggestions: pd.DataFrame, cfg) -> Path:
    out = Path(cfg["paths"]["kb_dir"]) / "faq_suggestions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if suggestions.empty:
        if not out.exists():
            pd.DataFrame(columns=[
                "question", "draft_answer", "cluster_size", "mean_similarity",
                "category", "status", "created_at",
            ]).to_csv(out, index=False)
        return out
    # merge with existing pending suggestions, dedupe on question
    if out.exists():
        old = pd.read_csv(out)
        old = old[old["status"] != "pending"]
        if not old.empty:
            suggestions = pd.concat([old, suggestions], ignore_index=True)
        suggestions = suggestions.drop_duplicates(subset=["question"], keep="last")
    suggestions.to_csv(out, index=False)
    return out
