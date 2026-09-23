"""
Structural feature extraction for PRISM-Phish.

Extracts formatting and structural features from emails.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def compute_structural_features(
    body: str,
    subject: Optional[str] = None,
) -> dict:
    """
    Compute structural/formatting features from email content.

    These complement the lexical features with format-based signals.
    """
    if not isinstance(body, str):
        body = ""

    features = {}

    # Line-based features
    lines = body.split("\n")
    features["line_count"] = len(lines)
    features["empty_line_ratio"] = (
        sum(1 for l in lines if l.strip() == "") / max(len(lines), 1)
    )
    features["avg_line_length"] = (
        sum(len(l) for l in lines) / max(len(lines), 1)
    )

    # Whitespace features
    features["whitespace_ratio"] = (
        sum(c.isspace() for c in body) / max(len(body), 1)
    )
    features["tab_count"] = body.count("\t")

    # Special character patterns
    features["dollar_sign_count"] = body.count("$")
    features["at_sign_count"] = body.count("@")
    features["hash_count"] = body.count("#")

    # ALL CAPS words
    words = body.split()
    all_caps_words = sum(
        1 for w in words if w.isupper() and len(w) > 1 and w.isalpha()
    )
    features["all_caps_word_count"] = all_caps_words
    features["all_caps_word_ratio"] = all_caps_words / max(len(words), 1)

    # Email address patterns in body
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    features["email_addresses_in_body"] = len(email_pattern.findall(body))

    # Phone number patterns
    phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    features["phone_numbers_in_body"] = len(phone_pattern.findall(body))

    return features
