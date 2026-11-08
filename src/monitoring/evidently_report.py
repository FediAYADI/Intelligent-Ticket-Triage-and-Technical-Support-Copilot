"""EvidentlyAI drift + data-quality report.

Compares a reference window (first part of processed tickets) with a current
window (recent tickets) on:
  - embedding drift (sentence-transformers embeddings, cosine/Wasserstein)
  - data quality: missing fields, empty text, invalid labels, duplicates

Run:  python -m src.monitoring.evidently_report
Output: reports/evidently_drift_report.html + reports/evidently_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config

TEXT_COLS = ["subject", "body", "ticket_text", "ticket_type", "urgency",
             "sentiment", "team"]
CATEGORICAL = ["ticket_type", "urgency", "sentiment", "team"]


def data_quality_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "rows": int(n),
        "missing_subject_pct": round(df["subject"].isna().mean() * 100, 2),
        "missing_body_pct": round(df["body"].isna().mean() * 100, 2),
        "empty_ticket_text_pct": round((df["ticket_text"].str.len() < 20).mean() * 100, 2),
        "duplicate_text_pct": round(df["ticket_text"].duplicated().mean() * 100, 2),
        "invalid_urgency_pct": round((~df["urgency"].isin(["low", "medium", "high"])).mean() * 100, 2),
        "invalid_team_pct": round(df["team"].isna().mean() * 100, 2),
    }


def embedding_drift(reference_texts: list[str], current_texts: list[str],
                    model_name: str) -> dict:
    """Mean cosine-distance drift between two embedding clouds (sampled)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    rng = np.random.default_rng(0)
    ref = rng.choice(reference_texts, size=min(1000, len(reference_texts)), replace=False)
    cur = rng.choice(current_texts, size=min(1000, len(current_texts)), replace=False)
    e_ref = model.encode(ref.tolist(), batch_size=64, normalize_embeddings=True)
    e_cur = model.encode(cur.tolist(), batch_size=64, normalize_embeddings=True)
    # centroid drift + mean pairwise distance between clouds
    centroid_drift = float(np.linalg.norm(e_ref.mean(0) - e_cur.mean(0)))
    # average distance from each current point to its nearest reference neighbour
    sims = e_cur @ e_ref.T
    nn = sims.max(axis=1)
    mean_nn_sim = float(nn.mean())
    return {
        "centroid_drift": round(centroid_drift, 4),
        "mean_nearest_reference_similarity": round(mean_nn_sim, 4),
        "drift_detected": bool(mean_nn_sim < 0.55),
    }


def run_report(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc = Path(cfg["paths"]["processed_dir"])
    df = pd.read_csv(proc / "tickets_all.csv")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)

    split_idx = int(len(df) * 0.7)
    reference, current = df.iloc[:split_idx], df.iloc[split_idx:]

    summary = {
        "reference_window": {"rows": len(reference),
                             "from": str(reference["created_at"].min()),
                             "to": str(reference["created_at"].max())},
        "current_window": {"rows": len(current),
                           "from": str(current["created_at"].min()),
                           "to": str(current["created_at"].max())},
        "data_quality_reference": data_quality_metrics(reference),
        "data_quality_current": data_quality_metrics(current),
    }

    try:
        summary["embedding_drift"] = embedding_drift(
            reference["ticket_text"].tolist(),
            current["ticket_text"].tolist(),
            cfg["rag"]["embedding_model"],
        )
    except Exception as e:  # evidently/embedding optional at report time
        summary["embedding_drift"] = {"error": str(e)}

    # ---- Evidently HTML report (if available) ----
    html_path = Path(cfg["paths"]["reports_dir"]) / "evidently_drift_report.html"
    try:
        try:  # evidently >= 0.7 moved the classic API under .legacy
            from evidently.legacy.report import Report
            from evidently.legacy.metric_preset import (
                DataDriftPreset, DataQualityPreset)
        except ImportError:  # evidently < 0.7
            from evidently.report import Report
            from evidently.metric_preset import (
                DataDriftPreset, DataQualityPreset)

        ref_sample = reference[TEXT_COLS].head(5000)
        cur_sample = current[TEXT_COLS].head(5000)
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=ref_sample, current_data=cur_sample)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(html_path))
        summary["evidently_html"] = str(html_path)
    except Exception as e:
        summary["evidently_html"] = None
        summary["evidently_error"] = str(e)

    out = Path(cfg["paths"]["reports_dir"]) / "evidently_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run_report()
