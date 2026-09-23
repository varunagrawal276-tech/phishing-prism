"""
URL extraction and preprocessing for PRISM-Phish.

Extracts URLs from email text and computes URL-based features.
IMPORTANT: Never clicks, resolves, or queries extracted URLs.
"""

import logging
import math
import re
from collections import Counter
from typing import Optional
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__)

# URL extraction pattern
URL_PATTERN = re.compile(
    r'(?:https?://|www\.)'       # Protocol or www
    r'[^\s<>"\')\]]+',           # Non-space, non-delimiter characters
    re.IGNORECASE,
)

# IP address pattern
IP_PATTERN = re.compile(
    r'^(?:\d{1,3}\.){3}\d{1,3}$'
)

# Suspicious URL tokens
SUSPICIOUS_TOKENS = [
    "login", "signin", "verify", "update", "confirm", "secure",
    "account", "banking", "password", "credential", "authenticate",
    "suspend", "alert", "notification", "urgent", "paypal",
    "ebay", "amazon", "microsoft", "apple", "google",
    ".exe", ".zip", ".scr", ".bat", ".cmd",
]


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text. Does NOT resolve or visit them."""
    if not isinstance(text, str):
        return []
    return URL_PATTERN.findall(text)


def url_entropy(url: str) -> float:
    """Compute Shannon entropy of URL characters."""
    if not url:
        return 0.0
    freq = Counter(url)
    length = len(url)
    entropy = -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )
    return round(entropy, 4)


def compute_url_features(urls: list[str]) -> dict:
    """
    Compute security-relevant features from a list of URLs.

    All analysis is purely lexical/structural — no external queries.
    """
    features = {
        "url_count": len(urls),
        "total_url_chars": 0,
        "mean_url_length": 0.0,
        "max_url_length": 0,
        "hostname_length_mean": 0.0,
        "path_length_mean": 0.0,
        "query_length_mean": 0.0,
        "num_subdomains_mean": 0.0,
        "digit_ratio_url": 0.0,
        "special_char_ratio_url": 0.0,
        "percent_encoding_count": 0,
        "ip_address_hostname": 0,
        "punycode_indicator": 0,
        "suspicious_token_count": 0,
        "url_entropy_mean": 0.0,
    }

    if not urls:
        return features

    url_lengths = []
    hostname_lengths = []
    path_lengths = []
    query_lengths = []
    subdomain_counts = []
    digit_ratios = []
    special_ratios = []
    entropies = []

    for raw_url in urls:
        # Ensure scheme for parsing
        url_str = raw_url if "://" in raw_url else f"http://{raw_url}"

        try:
            parsed = urlparse(url_str)
        except Exception:
            continue

        url_len = len(raw_url)
        url_lengths.append(url_len)
        features["total_url_chars"] += url_len

        # Hostname analysis
        hostname = parsed.hostname or ""
        hostname_lengths.append(len(hostname))

        # Path analysis
        path_lengths.append(len(parsed.path or ""))

        # Query analysis
        query_lengths.append(len(parsed.query or ""))

        # Subdomain count
        parts = hostname.split(".")
        subdomain_counts.append(max(0, len(parts) - 2))

        # Digit ratio in URL
        digits = sum(c.isdigit() for c in raw_url)
        digit_ratios.append(digits / max(url_len, 1))

        # Special character ratio
        special = sum(not c.isalnum() and c not in "./-_:" for c in raw_url)
        special_ratios.append(special / max(url_len, 1))

        # Percent encoding
        features["percent_encoding_count"] += raw_url.count("%")

        # IP address as hostname
        if IP_PATTERN.match(hostname):
            features["ip_address_hostname"] += 1

        # Punycode (internationalized domain names)
        if "xn--" in hostname.lower():
            features["punycode_indicator"] += 1

        # Suspicious tokens
        url_lower = raw_url.lower()
        for token in SUSPICIOUS_TOKENS:
            if token in url_lower:
                features["suspicious_token_count"] += 1

        # Entropy
        entropies.append(url_entropy(raw_url))

    n = len(url_lengths)
    if n > 0:
        features["mean_url_length"] = round(sum(url_lengths) / n, 2)
        features["max_url_length"] = max(url_lengths)
        features["hostname_length_mean"] = round(sum(hostname_lengths) / n, 2)
        features["path_length_mean"] = round(sum(path_lengths) / n, 2)
        features["query_length_mean"] = round(sum(query_lengths) / n, 2)
        features["num_subdomains_mean"] = round(sum(subdomain_counts) / n, 2)
        features["digit_ratio_url"] = round(sum(digit_ratios) / n, 4)
        features["special_char_ratio_url"] = round(sum(special_ratios) / n, 4)
        features["url_entropy_mean"] = round(sum(entropies) / n, 4)

    return features
