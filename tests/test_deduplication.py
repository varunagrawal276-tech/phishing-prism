"""Tests for deduplication and near-duplicate detection."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.deduplication import (
    compute_duplicate_hashes,
    find_exact_duplicates,
    hash_text,
    normalize_text,
)


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("HELLO") == "hello"

    def test_whitespace_collapse(self):
        assert normalize_text("hello   world") == "hello world"

    def test_strip(self):
        assert normalize_text("  hello  ") == "hello"

    def test_non_string(self):
        assert normalize_text(None) == ""
        assert normalize_text(123) == ""


class TestHashText:
    def test_deterministic(self):
        h1 = hash_text("hello")
        h2 = hash_text("hello")
        assert h1 == h2

    def test_different_texts(self):
        h1 = hash_text("hello")
        h2 = hash_text("world")
        assert h1 != h2


class TestComputeDuplicateHashes:
    def test_adds_hash_columns(self):
        df = pd.DataFrame({
            "body": ["hello", "world"],
            "subject": ["sub1", "sub2"],
        })
        result = compute_duplicate_hashes(df)
        assert "body_hash_raw" in result.columns
        assert "body_hash_norm" in result.columns
        assert "subj_body_hash" in result.columns

    def test_identical_bodies_same_hash(self):
        df = pd.DataFrame({
            "body": ["hello world", "hello world"],
            "subject": ["a", "b"],
        })
        result = compute_duplicate_hashes(df)
        assert result["body_hash_raw"].iloc[0] == result["body_hash_raw"].iloc[1]

    def test_normalized_duplicates(self):
        df = pd.DataFrame({
            "body": ["Hello  World", "hello world"],
            "subject": ["a", "b"],
        })
        result = compute_duplicate_hashes(df)
        # Normalized hashes should be the same
        assert result["body_hash_norm"].iloc[0] == result["body_hash_norm"].iloc[1]
        # Raw hashes should differ
        assert result["body_hash_raw"].iloc[0] != result["body_hash_raw"].iloc[1]


class TestFindExactDuplicates:
    def test_finds_duplicates(self):
        df = pd.DataFrame({
            "body": ["hello", "hello", "world"],
            "subject": ["a", "a", "b"],
        })
        df = compute_duplicate_hashes(df)
        result = find_exact_duplicates(df, "body_hash_norm")
        assert result["is_duplicate"].sum() == 2

    def test_no_duplicates(self):
        df = pd.DataFrame({
            "body": ["hello", "world", "foo"],
            "subject": ["a", "b", "c"],
        })
        df = compute_duplicate_hashes(df)
        result = find_exact_duplicates(df, "body_hash_norm")
        assert result["is_duplicate"].sum() == 0
