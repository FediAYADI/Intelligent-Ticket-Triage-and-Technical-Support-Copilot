"""Anomaly / incident detection.

Two complementary signals:
  1. Volume spikes: ticket counts per (team x hour) bucket are z-scored
     against the bucket's own history; z > threshold -> anomaly.
  2. Topical coherence: tickets in an anomalous bucket are embedded and
     clustered; the dominant cluster + its tickets form the incident.

Alerts are appended to data/processed/incidents.csv for the UI/dashboard.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.utils import load_config


def build_hourly_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ticket volume per team per hour."""
    d = df.copy()
    d["created_at"] = pd.to_datetime(d["created_at"])
    d["hour"] = d["created_at"].dt.floor("h")
    return (
        d.groupby(["team", "hour"])
        .agg(volume=("ticket_id", "count"),
             critical=("urgency", lambda s: int((s == "high").sum())),
             negative=("sentiment", lambda s: int((s == "negative").sum())))
        .reset_index()
    )


def detect_volume_anomalies(buckets: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Z-score each team's hourly volume against that team's history."""
    acfg = cfg["anomaly"]
    thr = float(acfg["zscore_threshold"])
    min_vol = int(acfg["min_bucket_volume"])
    rows = []
    for team, g in buckets.groupby("team"):
        mu, sigma = g["volume"].mean(), g["volume"].std()
        if sigma == 0 or np.isnan(sigma):
            continue
        g = g.assign(z=(g["volume"] - mu) / sigma)
        for _, r in g[g["z"] > thr].iterrows():
            if r["volume"] >= min_vol:
                rows.append({
                    "team": team, "hour": r["hour"], "volume": int(r["volume"]),
                    "zscore": round(float(r["z"]), 3),
                    "baseline_mean": round(float(mu), 2),
                    "negative": int(r["negative"]),
                })
    return pd.DataFrame(rows)


def describe_incident(anomaly_row: dict, tickets: pd.DataFrame,
                      embedder, cfg: dict) -> dict:
    """Cluster the tickets inside an anomalous window; pick dominant topic."""
    acfg = cfg["anomaly"]
    hour = pd.Timestamp(anomaly_row["hour"])
    window = tickets[
        (tickets["team"] == anomaly_row["team"]) &
        (pd.to_datetime(tickets["created_at"]).dt.floor("h") == hour)
    ].copy()
    if window.empty:
        window = tickets[
            (tickets["team"] == anomaly_row["team"]) &
            (pd.to_datetime(tickets["created_at"]).dt.floor("D") == hour.floor("D"))
        ].copy()
    if window.empty:
        return {"topic": "unknown", "ticket_ids": [], "sample_subjects": []}

    texts = (window["subject"].fillna("") + " " + window["body"].fillna("")).tolist()
    embs = embedder.encode(texts, batch_size=32, convert_to_numpy=True,
                           normalize_embeddings=True)
    k = min(int(acfg["n_clusters"]), max(2, len(window) // 3))
    km = KMeans(n_clusters=k, n_init=10, random_state=int(cfg["project"]["seed"]))
    labels = km.fit_predict(embs)
    counts = pd.Series(labels).value_counts()
    dominant = int(counts.idxmax())
    dom = window[labels == dominant]
    topics = dom["ticket_type"].value_counts().to_dict()
    top_topic = max(topics, key=topics.get) if topics else "unknown"
    return {
        "topic": f"{top_topic} — {dom['subject'].mode().iat[0] if len(dom) else ''}"[:120],
        "dominant_cluster_size": int(counts.max()),
        "window_size": int(len(window)),
        "ticket_type_dist": topics,
        "urgency_dist": dom["urgency"].value_counts().to_dict(),
        "ticket_ids": dom["ticket_id"].tolist()[:50],
        "sample_subjects": dom["subject"].dropna().unique().tolist()[:5],
    }


def run_detection(tickets: pd.DataFrame, cfg: dict | None = None,
                  embedder=None) -> pd.DataFrame:
    """Full detection pass over a ticket dataframe. Returns alerts."""
    cfg = cfg or load_config()
    buckets = build_hourly_buckets(tickets)
    anomalies = detect_volume_anomalies(buckets, cfg)
    if anomalies.empty:
        return pd.DataFrame(columns=[
            "alert_id", "detected_at", "team", "hour", "volume", "zscore",
            "topic", "dominant_cluster_size", "window_size",
            "ticket_type_dist", "urgency_dist", "ticket_ids", "sample_subjects",
            "status",
        ])
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(cfg["anomaly"]["embedding_model"])

    alerts = []
    cooldown = pd.Timedelta(hours=int(cfg["anomaly"].get("alert_cooldown_hours", 6)))
    last_alert: dict[str, pd.Timestamp] = {}
    now = dt.datetime.now().isoformat()
    for i, row in anomalies.iterrows():
        team = row["team"]
        ts = pd.Timestamp(row["hour"])
        if team in last_alert and ts - last_alert[team] < cooldown:
            continue
        last_alert[team] = ts
        desc = describe_incident(row.to_dict(), tickets, embedder, cfg)
        alerts.append({
            "alert_id": f"INC-{i+1:04d}",
            "detected_at": now,
            "team": team,
            "hour": str(row["hour"]),
            "volume": int(row["volume"]),
            "zscore": float(row["zscore"]),
            "baseline_mean": float(row["baseline_mean"]),
            "negative": int(row["negative"]),
            "status": "open",
            **desc,
        })
    return pd.DataFrame(alerts)


def save_alerts(alerts: pd.DataFrame, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    out = Path(cfg["paths"]["processed_dir"]) / "incidents.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if alerts.empty:
        if not out.exists():
            alerts.to_csv(out, index=False)
        return out
    if out.exists():
        old = pd.read_csv(out)
        alerts = pd.concat([old, alerts], ignore_index=True)
        alerts = alerts.drop_duplicates(subset=["alert_id"], keep="last")
    alerts.to_csv(out, index=False)
    return out


def run_on_processed(cfg: dict | None = None) -> pd.DataFrame:
    """Convenience entry: load processed tickets, detect, persist alerts."""
    cfg = cfg or load_config()
    proc = Path(cfg["paths"]["processed_dir"])
    tickets = pd.read_csv(proc / "tickets_all.csv")
    alerts = run_detection(tickets, cfg)
    save_alerts(alerts, cfg)
    return alerts


if __name__ == "__main__":
    alerts = run_on_processed()
    print(json.dumps(alerts.head(10).to_dict("records"), indent=2, default=str))
