"""Sentiment labeling heuristic.

The source dataset (Tobi-Bueck/customer-support-tickets) does NOT provide
sentiment labels. We derive a coarse positive/neutral/negative label with a
documented lexicon + punctuation heuristic. This is intentionally simple and
inspectable; it is NOT a trained model.
"""
from __future__ import annotations

import re

POSITIVE = {
    "thank", "thanks", "great", "excellent", "happy", "pleased", "appreciate",
    "awesome", "wonderful", "love", "best", "good", "helpful", "perfect",
    "resolved", "fixed", "works", "glad", "nice", "smooth", "fast",
}
NEGATIVE = {
    "angry", "furious", "frustrat", "terrible", "awful", "hate", "worst",
    "broken", "fail", "failed", "error", "bug", "crash", "crashed", "down",
    "outage", "unacceptable", "disappointed", "disappointing", "ridiculous",
    "useless", "slow", "problem", "issue", "urgent", "asap", "immediately",
    "disaster", "nightmare", "complaint", "refund", "scam", "fraud",
}
INTENSIFIERS = {"very", "extremely", "really", "so", "absolutely", "totally"}
NEGATION = {"not", "no", "never", "n't", "cannot", "can't", "don't", "doesn't"}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def score_text(text: str) -> float:
    """Return a sentiment score in [-1, 1]."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = _WORD_RE.findall(text.lower())
    score = 0.0
    prev_neg = False
    for i, w in enumerate(words):
        stem = w.rstrip("s")
        is_pos = any(p in stem for p in POSITIVE)
        is_neg = any(n in stem for n in NEGATIVE)
        if not (is_pos or is_neg):
            prev_neg = w in NEGATION
            continue
        weight = 2.0 if words[max(0, i - 1)] in INTENSIFIERS else 1.0
        if prev_neg:
            weight *= -1.0  # negated sentiment flips
            prev_neg = False
        score += weight if is_pos and not is_neg else (-weight if is_neg else weight)
    # normalize by sqrt of length to keep scores bounded
    norm = max(len(words) ** 0.5, 1.0)
    return max(-1.0, min(1.0, score / norm))


def label_sentiment(text: str) -> str:
    """Map score to {negative, neutral, positive}."""
    s = score_text(text)
    if s <= -0.08:
        return "negative"
    if s >= 0.08:
        return "positive"
    return "neutral"


def frustration_score(text: str) -> float:
    """0..1 frustration estimate used by the priority scorer.

    Combines negative sentiment with urgency cues (caps, exclamation marks,
    time-pressure words).
    """
    s = score_text(text)
    neg = max(0.0, -s)
    urgency_hits = sum(text.lower().count(w) for w in
                       ("asap", "urgent", "immediately", "critical", "emergency"))
    caps = sum(1 for ch in text if ch.isupper()) / max(len(text), 1)
    punct = text.count("!") + text.count("?")
    raw = neg + 0.15 * min(urgency_hits, 5) + 0.3 * min(caps, 0.3) + 0.05 * min(punct, 10)
    return min(1.0, raw)
