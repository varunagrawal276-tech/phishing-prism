"""
Email canonicalization for PRISM-Phish.

Converts raw email data into standardized representations for downstream processing.
"""

import logging
import re
from typing import Optional

import pandas as pd

from src.preprocessing.text import preprocess_transformer, preprocess_classical, combine_subject_body

logger = logging.getLogger(__name__)


def canonicalize_email(
    row: pd.Series,
    representation: str = "transformer",
) -> str:
    """
    Create a canonicalized text representation of an email.

    Args:
        row: A row from the canonical DataFrame.
        representation: 'transformer' for Representation A, 'classical' for B.

    Returns:
        Canonicalized text string.
    """
    subject = row.get("subject", "")
    body = row.get("body", "")

    if representation == "transformer":
        subject_clean = preprocess_transformer(str(subject)) if pd.notna(subject) else ""
        body_clean = preprocess_transformer(str(body)) if pd.notna(body) else ""
        return combine_subject_body(subject_clean, body_clean, separator=" [SEP] ")
    elif representation == "classical":
        subject_clean = preprocess_classical(str(subject)) if pd.notna(subject) else ""
        body_clean = preprocess_classical(str(body)) if pd.notna(body) else ""
        return combine_subject_body(subject_clean, body_clean, separator=" ")
    else:
        raise ValueError(f"Unknown representation: {representation}")


def add_canonicalized_text(
    df: pd.DataFrame,
    representations: list[str] = None,
) -> pd.DataFrame:
    """Add canonicalized text columns to the DataFrame."""
    if representations is None:
        representations = ["transformer", "classical"]

    df = df.copy()
    for rep in representations:
        col_name = f"text_{rep}"
        logger.info(f"Creating {col_name} representation...")
        df[col_name] = df.apply(lambda row: canonicalize_email(row, rep), axis=1)
        non_empty = (df[col_name].str.len() > 0).sum()
        logger.info(f"  {col_name}: {non_empty}/{len(df)} non-empty")

    return df
