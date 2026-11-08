"""Unit + integration tests. Run: pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing.sentiment import frustration_score, label_sentiment, score_text  # noqa: E402
from src.routing.router import compute_priority_score, route_ticket  # noqa: E402
from src.utils import load_config  # noqa: E402


# ---------------- sentiment ----------------
def test_sentiment_negative():
    assert label_sentiment("This is terrible, everything is broken and I am furious!") == "negative"


def test_sentiment_positive():
    assert label_sentiment("Thank you so much, the fix was perfect and fast!") == "positive"


def test_sentiment_neutral():
    assert label_sentiment("Please update my billing address to the new office location.") == "neutral"


def test_frustration_bounds():
    assert 0.0 <= frustration_score("hello world") <= 1.0
    assert frustration_score("URGENT!!! This is broken ASAP!!!") > frustration_score("hi")


# ---------------- routing ----------------
def _fake_prediction(team="Technical Support", urgency="high", conf=0.9):
    return {
        "ticket_type": {"label": "Problem", "confidence": conf},
        "urgency": {"label": urgency, "confidence": conf},
        "sentiment": {"label": "negative", "confidence": conf},
        "team": {"label": team, "confidence": conf},
        "overall_confidence": conf,
        "needs_human_review": conf < 0.6,
    }


def test_route_auto_vs_review():
    cfg = load_config()
    auto = route_ticket(_fake_prediction(conf=0.9), "text works", {}, cfg)
    review = route_ticket(_fake_prediction(conf=0.4), "text works", {}, cfg)
    assert auto["queue"] == "auto"
    assert review["queue"] == "human_review"


def test_priority_score_bounds():
    cfg = load_config()
    s = compute_priority_score(_fake_prediction(), "broken urgent!!!", {}, cfg)
    assert 0.0 <= s <= 1.0


# ---------------- storage ----------------
def test_storage_roundtrip(tmp_path, monkeypatch):
    from src import storage
    cfg = load_config()
    # point DB at tmp
    monkeypatch.setitem(cfg["paths"], "data_file", str(tmp_path / "t.db"))
    storage.init_db(cfg)
    with storage.connect(cfg) as conn:
        storage.insert_ticket(conn, {
            "ticket_id": "T1", "subject": "s", "body": "b", "team": "IT Support",
            "queue": "auto", "ticket_type": "Problem", "urgency": "high",
            "sentiment": "negative", "confidence": 0.8, "priority_score": 0.7,
            "created_at": "2026-01-01T00:00:00",
        })
        storage.record_feedback(conn, "T1", "a", "b", "accepted")
        rows = storage.fetch_all(conn, "tickets")
        fb = storage.fetch_all(conn, "copilot_feedback")
    assert rows[0]["ticket_id"] == "T1"
    assert fb[0]["action"] == "accepted"


# ---------------- anomaly detection ----------------
def test_anomaly_detection_finds_burst():
    from src.anomaly_detection.detector import build_hourly_buckets, detect_volume_anomalies
    cfg = load_config()
    # synthetic: 30 days of ~2 tickets/hour for one team + one huge spike
    rows = []
    for day in range(30):
        for h in range(24):
            for i in range(2):
                rows.append({"ticket_id": f"{day}-{h}-{i}", "team": "IT Support",
                             "created_at": f"2026-01-{day+1:02d}T{h:02d}:00:00",
                             "urgency": "low", "sentiment": "neutral"})
    # spike: 30 tickets in one hour
    for i in range(30):
        rows.append({"ticket_id": f"spike-{i}", "team": "IT Support",
                     "created_at": "2026-01-15T10:00:00",
                     "urgency": "high", "sentiment": "negative"})
    df = pd.DataFrame(rows)
    buckets = build_hourly_buckets(df)
    anomalies = detect_volume_anomalies(buckets, cfg)
    assert not anomalies.empty
    assert str(anomalies.iloc[0]["hour"]).startswith("2026-01-15 10")
