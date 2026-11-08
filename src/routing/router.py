"""Routing & prioritisation.

- Team assignment comes from the classifier's `team` head prediction.
- Tickets below the confidence threshold go to the human-review queue.
- Priority score = w_urgency*urgency + w_sentiment*frustration + w_workload*load,
  where load is the current backlog of the target team normalised by capacity.
"""
from __future__ import annotations

from src.data_processing.sentiment import frustration_score
from src.utils import load_config


def urgency_level_to_score(level: str, cfg: dict | None = None) -> float:
    cfg = cfg or load_config()
    return float(cfg["routing"]["urgency_scores"].get(level.lower(), 0.5))


def compute_priority_score(prediction: dict, text: str,
                           team_backlog: dict[str, int],
                           cfg: dict | None = None) -> float:
    """Return a priority score in [0, 1]. Higher = handle sooner."""
    cfg = cfg or load_config()
    w = cfg["routing"]["weights"]
    urgency = urgency_level_to_score(prediction["urgency"]["label"], cfg)
    frustration = frustration_score(text)
    team = prediction["team"]["label"]
    capacity = float(cfg["routing"]["team_capacity"].get(team, 5))
    load = min(1.0, team_backlog.get(team, 0) / max(capacity * 10, 1))
    score = (w["urgency"] * urgency
             + w["sentiment"] * frustration
             + w["workload"] * (1.0 - load))  # less-loaded teams get slight boost
    return round(min(1.0, max(0.0, score)), 4)


def route_ticket(prediction: dict, text: str,
                 team_backlog: dict[str, int] | None = None,
                 cfg: dict | None = None) -> dict:
    """Turn raw predictions into a routing decision."""
    cfg = cfg or load_config()
    team_backlog = team_backlog or {}
    team = prediction["team"]["label"]
    review = bool(prediction.get("needs_human_review"))
    return {
        "team": team,
        "queue": "human_review" if review else "auto",
        "priority_score": compute_priority_score(prediction, text, team_backlog, cfg),
        "confidence": prediction["overall_confidence"],
        "predictions": {c: prediction[c]["label"] for c in prediction if isinstance(prediction[c], dict)},
    }
