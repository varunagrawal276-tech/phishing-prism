"""
URL-based feature extraction for PRISM-Phish.

Re-exports from src.preprocessing.urls for the feature pipeline.
"""

from src.preprocessing.urls import compute_url_features, extract_urls

__all__ = ["compute_url_features", "extract_urls"]
