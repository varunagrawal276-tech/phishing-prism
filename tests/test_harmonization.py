"""Tests for data harmonization."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.harmonization import discover_schema, harmonize_labels
from src.data.schemas import KNOWN_SOURCES, validate_canonical_df


class TestDiscoverSchema:
    def test_discovers_label_column(self):
        df = pd.DataFrame({"body": ["hello"], "label": [1]})
        schema = discover_schema(df, "test")
        assert schema.label_column == "label"
        assert schema.label_values == [1]

    def test_discovers_body_column(self):
        df = pd.DataFrame({"body": ["hello"], "label": [1]})
        schema = discover_schema(df, "test")
        assert schema.body_column == "body"

    def test_discovers_subject_column(self):
        df = pd.DataFrame({"subject": ["hi"], "body": ["hello"], "label": [1]})
        schema = discover_schema(df, "test")
        assert schema.subject_column == "subject"

    def test_case_insensitive_discovery(self):
        df = pd.DataFrame({"Body": ["hello"], "Label": [1]})
        schema = discover_schema(df, "test")
        assert schema.body_column is not None
        assert schema.label_column is not None

    def test_records_row_count(self):
        df = pd.DataFrame({"body": ["a", "b", "c"], "label": [0, 1, 1]})
        schema = discover_schema(df, "test")
        assert schema.row_count == 3


class TestHarmonizeLabels:
    def test_binary_numeric_labels(self):
        df = pd.DataFrame({"label": [0, 1, 0, 1]})
        schema = discover_schema(df, "test")
        result, mapping = harmonize_labels(df, schema, "test")
        assert set(result["label"].dropna()) == {0, 1}

    def test_text_labels(self):
        df = pd.DataFrame({"label": ["phishing", "legitimate", "phishing"]})
        schema = discover_schema(df, "test")
        result, mapping = harmonize_labels(df, schema, "test")
        assert mapping["phishing"] == 1
        assert mapping["legitimate"] == 0


class TestValidateCanonical:
    def test_valid_df(self):
        df = pd.DataFrame({
            "sample_id": ["Ling_0"],
            "dataset_id": ["Ling"],
            "label": [1],
            "raw_row_id": [0],
        })
        errors = validate_canonical_df(df)
        assert len(errors) == 0

    def test_missing_required_column(self):
        df = pd.DataFrame({"dataset_id": ["Ling"]})
        errors = validate_canonical_df(df)
        assert any("sample_id" in e for e in errors)

    def test_invalid_label(self):
        df = pd.DataFrame({
            "sample_id": ["Ling_0"],
            "dataset_id": ["Ling"],
            "label": [5],
            "raw_row_id": [0],
        })
        errors = validate_canonical_df(df)
        assert any("Invalid label" in e for e in errors)


class TestKnownSources:
    def test_seven_sources(self):
        assert len(KNOWN_SOURCES) == 7

    def test_expected_sources(self):
        expected = {"Ling", "Enron", "Assassin", "TREC-05", "TREC-06", "TREC-07", "CEAS-08"}
        assert set(KNOWN_SOURCES) == expected
