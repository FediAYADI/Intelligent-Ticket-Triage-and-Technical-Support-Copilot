"""Seed the SQLite store with historical tickets for dashboards/demos.

Optionally runs the trained classifier on a sample to attach real
predictions (slower). Usage:
    python scripts/seed_db.py [--predict-sample 300]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage import connect, init_db, insert_ticket
from src.utils import load_config, project_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict-sample", type=int, default=0,
                    help="run classifier on N tickets to attach predictions")
    args = ap.parse_args()

    cfg = load_config()
    init_db(cfg)
    proc = Path(cfg["paths"]["processed_dir"])
    df = pd.read_csv(proc / "tickets_all.csv")

    predictions = {}
    if args.predict_sample > 0:
        from src.classification.predict import TicketClassifier
        clf = TicketClassifier()
        sample = df.sample(n=min(args.predict_sample, len(df)),
                           random_state=int(cfg["project"]["seed"]))
        for _, r in sample.iterrows():
            predictions[r["ticket_id"]] = clf.predict_one(r["ticket_text"])
        print(f"scored {len(predictions)} tickets")

    with connect(cfg) as conn:
        for _, r in df.iterrows():
            pred = predictions.get(r["ticket_id"])
            ticket = {
                "ticket_id": r["ticket_id"],
                "subject": r["subject"], "body": r["body"],
                "team": r["team"], "queue": "historical",
                "ticket_type": r["ticket_type"], "urgency": r["urgency"],
                "sentiment": r["sentiment"],
                "confidence": pred["overall_confidence"] if pred else None,
                "priority_score": None,
                "needs_review": 0 if pred is None else int(pred["needs_human_review"]),
                "status": "resolved" if r["split"] != "test" else "new",
                "source": "historical",
                "created_at": r["created_at"],
                "predictions": {c: pred[c]["label"] for c in
                                ("ticket_type", "urgency", "sentiment", "team")}
                               if pred else {},
            }
            insert_ticket(conn, ticket)
    print(f"seeded {len(df)} tickets")


if __name__ == "__main__":
    main()
