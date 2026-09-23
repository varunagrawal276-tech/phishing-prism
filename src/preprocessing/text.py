"""
Text preprocessing for PRISM-Phish.

Maintains TWO representations:
- Representation A: Minimally processed (for Transformer models)
  Preserves URLs, punctuation, casing, suspicious spacing, Unicode
- Representation B: Normalized (for classical NLP baselines)
  Lowercase, whitespace-normalized, optional stopword removal
"""

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)


def preprocess_transformer(text: str) -> str:
    """
    Representation A — Minimal preprocessing for Transformer input.

    Preserves:
    - URLs (phishing signals)
    - Punctuation patterns (urgency cues)
    - Casing (emphasis)
    - Suspicious spacing/Unicode
    - HTML-derived textual clues

    Only removes:
    - Null bytes and control characters (except newlines/tabs)
    - Excessive repeated whitespace (> 5 consecutive)
    """
    if not isinstance(text, str):
        return ""

    # Remove null bytes and most control chars (keep \n, \r, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse extreme whitespace runs (> 5 consecutive) but preserve mild spacing
    text = re.sub(r"[ \t]{6,}", "     ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Truncation safety — don't let a single email be absurdly long
    # (actual truncation is handled by tokenizer max_length)
    text = text.strip()

    return text


def preprocess_classical(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = False,
    remove_numbers: bool = False,
) -> str:
    """
    Representation B — Normalized preprocessing for classical models.

    - Lowercase
    - Unicode NFKD normalization
    - Whitespace normalization
    - Optional URL removal
    - Optional number removal
    """
    if not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # Lowercase
    if lowercase:
        text = text.lower()

    # Optionally remove URLs (but keep URL-related features separately)
    if remove_urls:
        text = re.sub(r"https?://\S+", " URL ", text)
        text = re.sub(r"www\.\S+", " URL ", text)

    # Optionally remove numbers
    if remove_numbers:
        text = re.sub(r"\b\d+\b", " NUM ", text)

    # Whitespace normalization
    text = re.sub(r"\s+", " ", text).strip()

    return text


def combine_subject_body(
    subject: Optional[str],
    body: Optional[str],
    separator: str = " [SEP] ",
) -> str:
    """
    Combine subject and body for model input.

    For Transformer: subject [SEP] body
    The separator allows the model to learn subject vs body patterns.
    """
    parts = []
    if subject and isinstance(subject, str) and subject.strip():
        parts.append(subject.strip())
    if body and isinstance(body, str) and body.strip():
        parts.append(body.strip())
    return separator.join(parts) if parts else ""
