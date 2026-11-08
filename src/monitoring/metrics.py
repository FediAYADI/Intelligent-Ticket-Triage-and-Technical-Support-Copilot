"""Prometheus metrics for the triage pipeline.

Grafana scrapes http://localhost:8000/metrics. Call `start_metrics_server()`
once (e.g. from the Streamlit app or a small service script).
"""
from __future__ import annotations

from prometheus_client import (CollectorRegistry, Counter, Gauge,
                               Histogram, start_http_server)

REGISTRY = CollectorRegistry()

TICKETS_TOTAL = Counter(
    "tickets_total", "Tickets processed", ["team", "urgency", "source"],
    registry=REGISTRY,
)
TICKETS_REVIEW = Counter(
    "tickets_human_review_total", "Tickets sent to human review",
    registry=REGISTRY,
)
ROUTING_ERRORS = Counter(
    "routing_errors_total", "Tickets manually corrected after auto-routing",
    ["team"], registry=REGISTRY,
)
COPILOT_ACTIONS = Counter(
    "copilot_suggestions_total", "Copilot suggestion outcomes",
    ["action"], registry=REGISTRY,   # accepted|modified|rejected
)
ANOMALIES_TOTAL = Counter(
    "anomalies_detected_total", "Incident alerts raised", ["team"],
    registry=REGISTRY,
)
PIPELINE_LATENCY = Histogram(
    "pipeline_latency_seconds", "End-to-end ticket processing latency",
    registry=REGISTRY,
)
LLM_TOKENS = Counter(
    "llm_tokens_total", "Estimated LLM token usage", ["model"],
    registry=REGISTRY,
)
ACTIVE_TICKETS = Gauge(
    "active_tickets", "Open tickets per team", ["team"], registry=REGISTRY,
)

METRICS_PORT = 8000
_server_started = False


def start_metrics_server(port: int | None = None):
    global _server_started
    if _server_started:
        return
    start_http_server(port or METRICS_PORT, registry=REGISTRY)
    _server_started = True


def record_ticket(team: str, urgency: str, source: str = "live"):
    TICKETS_TOTAL.labels(team=team, urgency=urgency, source=source).inc()


def record_review():
    TICKETS_REVIEW.inc()


def record_routing_error(team: str):
    ROUTING_ERRORS.labels(team=team).inc()


def record_copilot_action(action: str):
    COPILOT_ACTIONS.labels(action=action).inc()


def record_anomaly(team: str):
    ANOMALIES_TOTAL.labels(team=team).inc()


def observe_latency(seconds: float):
    PIPELINE_LATENCY.observe(seconds)


def record_tokens(model: str, tokens: int):
    LLM_TOKENS.labels(model=model).inc(tokens)


def set_active_tickets(team: str, n: int):
    ACTIVE_TICKETS.labels(team=team).set(n)
