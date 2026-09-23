"""Tests for feature extraction."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.lexical import compute_lexical_features, count_lexicon_matches
from src.features.structural import compute_structural_features
from src.features.header_features import compute_header_features, extract_email_domain
from src.preprocessing.urls import extract_urls, compute_url_features, url_entropy


class TestLexicalFeatures:
    def test_basic_features(self):
        features = compute_lexical_features("Hello World! This is urgent.")
        assert features["email_length"] > 0
        assert features["word_count"] == 5
        assert features["exclamation_count"] == 1

    def test_empty_text(self):
        features = compute_lexical_features("")
        assert features["email_length"] == 0
        assert features["word_count"] == 0

    def test_urgency_detection(self):
        features = compute_lexical_features("This is urgent, act now immediately")
        assert features["urgency_lexicon_count"] >= 2

    def test_credential_detection(self):
        features = compute_lexical_features("Please enter your password and login")
        assert features["credential_request_lexicon_count"] >= 2


class TestStructuralFeatures:
    def test_basic_features(self):
        text = "Line 1\nLine 2\n\nLine 4"
        features = compute_structural_features(text)
        assert features["line_count"] == 4
        assert features["empty_line_ratio"] > 0

    def test_all_caps(self):
        features = compute_structural_features("THIS IS VERY IMPORTANT")
        assert features["all_caps_word_count"] >= 3


class TestHeaderFeatures:
    def test_sender_present(self):
        features = compute_header_features(sender="user@gmail.com")
        assert features["sender_present"] is True
        assert features["free_email_provider"] == 1

    def test_no_sender(self):
        features = compute_header_features(sender=None)
        assert features["sender_present"] is False

    def test_domain_extraction(self):
        assert extract_email_domain("user@example.com") == "example.com"
        assert extract_email_domain("invalid") is None

    def test_corporate_sender(self):
        features = compute_header_features(sender="ceo@bigcorp.com")
        assert features["free_email_provider"] == 0


class TestURLFeatures:
    def test_extract_urls(self):
        text = "Visit http://example.com and https://test.org/page"
        urls = extract_urls(text)
        assert len(urls) == 2

    def test_no_urls(self):
        urls = extract_urls("No URLs here")
        assert len(urls) == 0

    def test_url_features(self):
        urls = ["http://example.com/login?user=admin"]
        features = compute_url_features(urls)
        assert features["url_count"] == 1
        assert features["suspicious_token_count"] >= 1  # "login"

    def test_ip_detection(self):
        urls = ["http://192.168.1.1/phish"]
        features = compute_url_features(urls)
        assert features["ip_address_hostname"] == 1

    def test_entropy(self):
        e = url_entropy("aaaaaa")
        assert e == 0.0  # All same char
        e2 = url_entropy("abcdef")
        assert e2 > 0  # Mixed chars
