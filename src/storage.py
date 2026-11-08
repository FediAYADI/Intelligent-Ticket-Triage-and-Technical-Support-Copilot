"""SQLite store for runtime state: incoming tickets, review queue,
copilot feedback (accept/modify/reject), and team workloads."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.utils import load_config

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    subject TEXT,
    body TEXT,
    team TEXT,
    queue TEXT,               -- 'auto' | 'human_review'
    ticket_type TEXT,
    urgency TEXT,
    sentiment TEXT,
    confidence REAL,
    priority_score REAL,
    needs_review INTEGER DEFAULT 0,
    status TEXT DEFAULT 'new',   -- new|in_review|resolved|corrected
    source TEXT DEFAULT 'live',  -- live|historical|simulated
    created_at TEXT,
    predictions_json TEXT
);
CREATE TABLE IF NOT EXISTS copilot_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT,
    suggested_answer TEXT,
    final_answer TEXT,
    action TEXT,              -- accepted|modified|rejected
    agent TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS team_workload (
    team TEXT PRIMARY KEY,
    open_tickets INTEGER DEFAULT 0
);
"""


@contextmanager
def connect(cfg: dict | None = None):
    cfg = cfg or load_config()
    db = Path(cfg["paths"]["data_file"])
    if not db.is_absolute():
        db = Path(cfg["paths"]["processed_dir"]).parent.parent / db
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(cfg: dict | None = None):
    cfg = cfg or load_config()
    db = Path(cfg["paths"]["data_file"])
    if not db.is_absolute():
        db = Path(cfg["paths"]["processed_dir"]).parent.parent / db
    conn = sqlite3.connect(str(db))
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def insert_ticket(conn: sqlite3.Connection, ticket: dict):
    conn.execute(
        """INSERT OR REPLACE INTO tickets
        (ticket_id, subject, body, team, queue, ticket_type, urgency,
         sentiment, confidence, priority_score, needs_review, status,
         source, created_at, predictions_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            ticket["ticket_id"], ticket.get("subject"), ticket.get("body"),
            ticket.get("team"), ticket.get("queue"), ticket.get("ticket_type"),
            ticket.get("urgency"), ticket.get("sentiment"), ticket.get("confidence"),
            ticket.get("priority_score"), int(ticket.get("needs_review", 0)),
            ticket.get("status", "new"), ticket.get("source", "live"),
            ticket.get("created_at"), json.dumps(ticket.get("predictions", {})),
        ),
    )


def record_feedback(conn: sqlite3.Connection, ticket_id: str,
                    suggested: str, final: str, action: str, agent: str = "agent"):
    conn.execute(
        """INSERT INTO copilot_feedback
        (ticket_id, suggested_answer, final_answer, action, agent)
        VALUES (?,?,?,?,?)""",
        (ticket_id, suggested, final, action, agent),
    )


def update_workload(conn: sqlite3.Connection, team: str, delta: int):
    conn.execute(
        """INSERT INTO team_workload (team, open_tickets) VALUES (?, ?)
        ON CONFLICT(team) DO UPDATE SET open_tickets = open_tickets + ?""",
        (team, delta, delta),
    )


def fetch_all(conn: sqlite3.Connection, table: str,
              where: str = "", params: tuple = ()) -> list[dict]:
    cur = conn.execute(f"SELECT * FROM {table} {where}", params)
    return [dict(r) for r in cur.fetchall()]
