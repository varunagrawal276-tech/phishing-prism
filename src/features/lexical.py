"""
Lexical feature extraction for PRISM-Phish.

Computes text-based features that indicate phishing characteristics.
All features are deterministic and computed from the email text only.
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Urgency lexicon
URGENCY_LEXICON = [
    "urgent", "immediately", "action required", "verify your account",
    "suspend", "expire", "within 24 hours", "limited time", "act now",
    "final warning", "your account will be", "unauthorized", "unusual activity",
    "security alert", "verify now", "click here immediately", "time sensitive",
    "respond immediately", "failure to", "will result in",
]

# Credential-request lexicon
CREDENTIAL_LEXICON = [
    "password", "login", "credential", "verify your identity",
    "confirm your account", "social security", "bank account", "credit card",
    "ssn", "pin number", "security question", "username", "sign in",
    "log in", "reset your password", "update your information",
]

# Financial-action lexicon
FINANCIAL_LEXICON = [
    "wire transfer", "bank transfer", "payment", "invoice", "refund",
    "transaction", "billing", "purchase", "order", "shipping",
    "gift card", "bitcoin", "cryptocurrency", "paypal",
]

# Call-to-action lexicon
CTA_LEXICON = [
    "click here", "click below", "click the link", "download",
    "open the attachment", "follow the link", "visit", "go to",
    "access your", "log into", "sign into", "update now",
]


def count_lexicon_matches(text: str, lexicon: list[str]) -> int:
    """Count case-insensitive lexicon matches in text."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(1 for term in lexicon if term in text_lower)


def compute_lexical_features(
    text: str,
    subject: Optional[str] = None,
) -> dict:
    """
    Compute lexical features from email text.

    All features are deterministic.
    """
    if not isinstance(text, str):
        text = ""

    features = {}

    # Basic length features
    features["email_length"] = len(text)
    features["subject_length"] = len(str(subject)) if subject and isinstance(subject, str) else 0
    features["word_count"] = len(text.split())

    # Sentence count (approximate)
    sentences = re.split(r'[.!?]+', text)
    features["sentence_count"] = max(1, len([s for s in sentences if s.strip()]))

    # Character ratio features
    total_chars = max(len(text), 1)
    features["uppercase_ratio"] = sum(c.isupper() for c in text) / total_chars
    features["digit_ratio"] = sum(c.isdigit() for c in text) / total_chars
    features["punctuation_ratio"] = sum(
        not c.isalnum() and not c.isspace() for c in text
    ) / total_chars

    # Specific punctuation counts
    features["exclamation_count"] = text.count("!")
    features["question_count"] = text.count("?")

    # Lexicon-based features
    combined = (str(subject) + " " + text) if subject else text
    features["urgency_lexicon_count"] = count_lexicon_matches(combined, URGENCY_LEXICON)
    features["credential_request_lexicon_count"] = count_lexicon_matches(
        combined, CREDENTIAL_LEXICON
    )
    features["financial_action_lexicon_count"] = count_lexicon_matches(
        combined, FINANCIAL_LEXICON
    )
    features["call_to_action_count"] = count_lexicon_matches(combined, CTA_LEXICON)

    return features
