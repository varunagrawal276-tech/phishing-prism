"""
HTML preprocessing for PRISM-Phish.

Extracts text and phishing-relevant signals from HTML email content.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_BS4_AVAILABLE = None


def _check_bs4():
    global _BS4_AVAILABLE
    if _BS4_AVAILABLE is None:
        try:
            from bs4 import BeautifulSoup
            _BS4_AVAILABLE = True
        except ImportError:
            _BS4_AVAILABLE = False
    return _BS4_AVAILABLE


def strip_html_tags(text: str) -> str:
    """Simple HTML tag removal without BeautifulSoup."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_html_features(text: str) -> dict:
    """
    Extract HTML-related features from email content.

    Features extracted:
    - html_presence: Whether the text contains HTML tags
    - text_html_ratio: Ratio of text to HTML markup
    - anchor_count: Number of <a> tags
    - form_count: Number of <form> tags
    - hidden_text_indicators: Count of hidden/invisible text patterns
    - external_resource_count: Number of external resource references
    """
    if not isinstance(text, str):
        return {
            "html_presence": False,
            "text_html_ratio": 1.0,
            "anchor_count": 0,
            "form_count": 0,
            "hidden_text_indicators": 0,
            "external_resource_count": 0,
        }

    has_html = bool(re.search(r"<[a-zA-Z][^>]*>", text))

    if not has_html:
        return {
            "html_presence": False,
            "text_html_ratio": 1.0,
            "anchor_count": 0,
            "form_count": 0,
            "hidden_text_indicators": 0,
            "external_resource_count": 0,
        }

    # Count tags
    anchor_count = len(re.findall(r"<a\b", text, re.IGNORECASE))
    form_count = len(re.findall(r"<form\b", text, re.IGNORECASE))

    # Hidden text indicators
    hidden_patterns = [
        r'display\s*:\s*none',
        r'visibility\s*:\s*hidden',
        r'font-size\s*:\s*0',
        r'color\s*:\s*(?:white|#fff|#ffffff|transparent)',
        r'opacity\s*:\s*0',
    ]
    hidden_count = sum(
        len(re.findall(p, text, re.IGNORECASE)) for p in hidden_patterns
    )

    # External resources
    ext_patterns = [
        r'src\s*=\s*["\']https?://',
        r'href\s*=\s*["\']https?://',
        r'background\s*=\s*["\']https?://',
    ]
    ext_count = sum(
        len(re.findall(p, text, re.IGNORECASE)) for p in ext_patterns
    )

    # Text/HTML ratio
    plain_text = strip_html_tags(text)
    ratio = len(plain_text) / max(len(text), 1)

    return {
        "html_presence": True,
        "text_html_ratio": round(ratio, 4),
        "anchor_count": anchor_count,
        "form_count": form_count,
        "hidden_text_indicators": hidden_count,
        "external_resource_count": ext_count,
    }
