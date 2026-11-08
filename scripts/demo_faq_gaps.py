"""Demo: FAQ-gap detection on simulated uncovered live questions.

Generates a set of recurring questions on topics absent from the knowledge
base, runs clustering-based gap detection, and writes suggestions to
data/knowledge_base/faq_suggestions.csv for human validation in the UI.

Usage:  python scripts/demo_faq_gaps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.copilot import Copilot
from src.rag.faq_gaps import (find_uncovered_questions, save_suggestions,
                              suggest_faqs)
from src.utils import load_config

# Simulated recurring live questions on KB gaps (SSO, mobile app,
# webhooks, GDPR/compliance). Each theme repeated with paraphrases.
QUESTIONS = {
    "sso": [
        "How do I configure SSO with Okta for our organization?",
        "SSO login fails with Okta, what are the requirements?",
        "Can we enforce SSO-only login for all employees?",
        "Our Okta SAML integration returns an assertion error.",
        "How do I map Okta groups to roles in the platform?",
        "SSO with Azure AD keeps redirecting in a loop, how to fix?",
        "Where do I enter the Okta client ID and secret?",
        "Employees cannot log in after we enabled SSO, help!",
    ],
    "mobile": [
        "The mobile app drains battery excessively after the latest update.",
        "Push notifications stopped working on Android devices.",
        "How do I enable biometric login in the mobile app?",
        "App crashes when uploading photos on iOS 18.",
        "Is there a dark mode in the mobile application?",
        "Mobile app shows a blank screen after login on Android 14.",
        "How do I log out of the mobile app on a shared device?",
        "Offline mode is not available in the mobile app, is that planned?",
    ],
    "webhooks": [
        "Webhook deliveries are failing with HTTP 500 errors.",
        "How do I rotate webhook signing secrets?",
        "Can webhooks retry failed deliveries automatically?",
        "Where can I see the webhook delivery logs?",
        "How do I test a webhook endpoint before going live?",
        "Our webhook endpoint times out after 5 seconds, what is the limit?",
        "Do you support webhook payload versioning?",
        "Webhook events arrive out of order, is that expected?",
    ],
    "gdpr": [
        "How do I export all customer data for a GDPR request?",
        "Does the platform support data residency in the EU?",
        "How long are audit logs retained for compliance?",
        "Can we anonymize user data without deleting accounts?",
        "Where is the subprocessor list published?",
        "We need a DPA before procurement, who do we contact?",
        "How do we handle a data breach notification under GDPR?",
        "Is customer data encrypted with keys we control (BYOK)?",
    ],
}


def main():
    cfg = load_config()
    copilot = Copilot(cfg=cfg)
    questions = [q for items in QUESTIONS.values() for q in items]
    uncovered = find_uncovered_questions(copilot, questions)
    print(f"uncovered questions: {len(uncovered)}/{len(questions)}")
    suggestions = suggest_faqs(copilot, uncovered)
    print(f"FAQ suggestions: {len(suggestions)}")
    if len(suggestions):
        print(suggestions[["question", "cluster_size",
                           "mean_similarity"]].to_string(index=False))
    path = save_suggestions(suggestions, cfg)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
