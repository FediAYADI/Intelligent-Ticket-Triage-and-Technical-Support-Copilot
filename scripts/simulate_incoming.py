"""Simulate incoming live tickets end-to-end.

- Samples tickets from the test split (real text, treated as 'live').
- Optionally injects an incident burst (rephrased payment-outage tickets)
  to demonstrate anomaly detection.
- Runs classifier -> routing -> stores in SQLite, records Prometheus metrics.

Usage:  python scripts/simulate_incoming.py [--n 25] [--burst]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.predict import TicketClassifier
from src.monitoring import metrics
from src.routing.router import route_ticket
from src.storage import connect, init_db, insert_ticket, update_workload
from src.utils import load_config

BURST_TEMPLATES = [
    "URGENT: Payment service is completely down. Customers cannot check out. "
    "This is the third time today. We are losing sales every minute!",
    "Payment gateway not working since this morning. All transactions are "
    "failing with error 503. Please fix ASAP, this is a disaster.",
    "Cannot process any payments. Customers are flooding our support. "
    "Payment outage is costing us heavily. When will it be resolved?",
    "Our payment service has been unavailable for hours. Every customer "
    "is complaining. This is unacceptable — escalate immediately!",
]


def make_burst_tickets(n: int, team: str = "Billing and Payments") -> list[dict]:
    now = dt.datetime.now()
    tickets = []
    for i in range(n):
        body = BURST_TEMPLATES[i % len(BURST_TEMPLATES)]
        tickets.append({
            "ticket_id": f"LIVE-BURST-{now.strftime('%H%M%S')}-{i}",
            "subject": "Payment service outage - critical",
            "body": body,
            "ticket_text": f"Payment service outage - critical {body}",
            "created_at": (now - dt.timedelta(minutes=i)).isoformat(),
            "team": team,
        })
    return tickets


def process_ticket(clf, text: str, meta: dict, team_backlog: dict) -> dict:
    t0 = time.time()
    pred = clf.predict_one(text)
    decision = route_ticket(pred, text, team_backlog)
    metrics.observe_latency(time.time() - t0)
    metrics.record_ticket(decision["team"], decision["predictions"].get("urgency", "medium"))
    if decision["queue"] == "human_review":
        metrics.record_review()
    return {**meta, **decision, "predictions_detail": pred}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--burst", type=int, default=0,
                    help="inject N synthetic payment-outage tickets")
    args = ap.parse_args()

    cfg = load_config()
    init_db(cfg)
    metrics.start_metrics_server()

    clf = TicketClassifier()
    proc = Path(cfg["paths"]["processed_dir"])
    test_df = pd.read_csv(proc / "tickets_test.csv")
    sample = test_df.sample(n=min(args.n, len(test_df)),
                            random_state=int(cfg["project"]["seed"]))

    # current backlog per team from DB
    with connect(cfg) as conn:
        rows = conn.execute(
            "SELECT team, COUNT(*) c FROM tickets WHERE status='new' GROUP BY team"
        ).fetchall()
    team_backlog = {r["team"]: r["c"] for r in rows}

    incoming = []
    for _, r in sample.iterrows():
        incoming.append({
            "ticket_id": f"LIVE-{r['ticket_id']}",
            "subject": r["subject"], "body": r["body"],
            "ticket_text": r["ticket_text"],
            "created_at": dt.datetime.now().isoformat(),
        })
    if args.burst:
        incoming.extend(make_burst_tickets(args.burst))

    stored = []
    with connect(cfg) as conn:
        for t in incoming:
            res = process_ticket(clf, t["ticket_text"], t, team_backlog)
            ticket = {
                "ticket_id": res["ticket_id"],
                "subject": res.get("subject"), "body": res.get("body"),
                "team": res["team"], "queue": res["queue"],
                "ticket_type": res["predictions"].get("ticket_type"),
                "urgency": res["predictions"].get("urgency"),
                "sentiment": res["predictions"].get("sentiment"),
                "confidence": res["confidence"],
                "priority_score": res["priority_score"],
                "needs_review": int(res["queue"] == "human_review"),
                "status": "new",
                "source": "simulated" if "BURST" in res["ticket_id"] else "live",
                "created_at": res["created_at"],
                "predictions": res["predictions"],
            }
            insert_ticket(conn, ticket)
            update_workload(conn, res["team"], 1)
            stored.append(ticket)

    rev = sum(1 for t in stored if t["queue"] == "human_review")
    print(f"processed {len(stored)} tickets | {rev} sent to human review")
    print("metrics server on :8000/metrics")


if __name__ == "__main__":
    main()
