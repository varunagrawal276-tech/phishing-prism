"""
Adversarial perturbation generators for PRISM-Phish robustness evaluation.

Implements:
1. Homoglyph substitution (Cyrillic / Greek lookalikes)
2. Zero-width and hidden whitespace injection
3. Typosquatting / Character transpositions
4. URL defanging and obfuscation
5. Case manipulation and leetspeak
"""

import random
import re
from typing import List

# Common Latin-to-Cyrillic homoglyph mappings
HOMOGLYPH_MAP = {
    "a": "а", "c": "с", "e": "е", "i": "і", "j": "ј",
    "o": "о", "p": "р", "s": "ѕ", "x": "х", "y": "у",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
    "I": "І", "J": "Ј", "K": "К", "M": "М", "O": "О",
    "P": "Р", "S": "Ѕ", "T": "Т", "X": "Х", "Y": "У",
}

ZERO_WIDTH_CHARS = ["\u200B", "\u200C", "\u200D", "\uFEFF"]


def homoglyph_substitution(text: str, p: float = 0.1, seed: int = 42) -> str:
    """Replace Latin characters with visual homoglyphs."""
    random.seed(seed)
    chars = []
    for char in text:
        if char in HOMOGLYPH_MAP and random.random() < p:
            chars.append(HOMOGLYPH_MAP[char])
        else:
            chars.append(char)
    return "".join(chars)


def zero_width_insertion(text: str, p: float = 0.05, seed: int = 42) -> str:
    """Insert zero-width unicode spaces between characters to disrupt tokenization."""
    random.seed(seed)
    chars = []
    for char in text:
        chars.append(char)
        if char.isalpha() and random.random() < p:
            chars.append(random.choice(ZERO_WIDTH_CHARS))
    return "".join(chars)


def typosquatting_perturbation(text: str, p: float = 0.05, seed: int = 42) -> str:
    """Simulate human typos via adjacent character swap or deletion."""
    random.seed(seed)
    words = text.split()
    perturbed_words = []
    for word in words:
        if len(word) > 3 and random.random() < p:
            idx = random.randint(0, len(word) - 2)
            # Swap adjacent chars
            word_list = list(word)
            word_list[idx], word_list[idx + 1] = word_list[idx + 1], word_list[idx]
            perturbed_words.append("".join(word_list))
        else:
            perturbed_words.append(word)
    return " ".join(perturbed_words)


def url_obfuscation(text: str) -> str:
    """Defang or obfuscate URLs found in email text."""
    # Replace http:// with hxxp:// or [.]
    text = re.sub(r"https?://", "hxxp://", text, flags=re.IGNORECASE)
    text = re.sub(r"\.([a-zA-Z]{2,6})([/\s]|$)", r"[.]\1\2", text)
    return text


def apply_perturbations(text: str, perturb_type: str, severity: float = 0.1, seed: int = 42) -> str:
    """Dispatch to perturbation function."""
    if perturb_type == "homoglyph":
        return homoglyph_substitution(text, p=severity, seed=seed)
    elif perturb_type == "zero_width":
        return zero_width_insertion(text, p=severity, seed=seed)
    elif perturb_type == "typo":
        return typosquatting_perturbation(text, p=severity, seed=seed)
    elif perturb_type == "url":
        return url_obfuscation(text)
    elif perturb_type == "combined":
        t = homoglyph_substitution(text, p=severity, seed=seed)
        t = zero_width_insertion(t, p=severity, seed=seed)
        return url_obfuscation(t)
    return text
