"""
Inference engine for PRISM-Phish API.
"""

import re
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.sparse import hstack

from src.api.schemas import EmailAnalysisRequest, EmailAnalysisResponse, FeatureHighlights

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINES_DIR = PROJECT_ROOT / "artifacts" / "baselines"

URGENCY_KEYWORDS = [
    "urgent", "immediately", "account suspended", "verify your account",
    "password expired", "security alert", "action required", "wire transfer",
    "login to claim", "unauthorized access", "billing problem", "gift card",
]


class PhishingInferenceEngine:
    def __init__(self):
        self.vectorizers = None
        self.model = None
        self._load_models()

    def _load_models(self):
        vec_path = BASELINES_DIR / "tfidf_vectorizers.joblib"
        model_path = BASELINES_DIR / "logistic_regression.joblib"

        if vec_path.exists() and model_path.exists():
            self.vectorizers = joblib.load(vec_path)
            self.model = joblib.load(model_path)

    def is_ready(self) -> bool:
        return self.vectorizers is not None and self.model is not None

    def analyze_email(self, req: EmailAnalysisRequest) -> EmailAnalysisResponse:
        start_time = time.time()

        subject = req.subject or ""
        body = req.body or ""
        full_text = f"{subject} {body}".strip()

        # 1. Feature highlights extraction
        url_matches = re.findall(r"https?://[^\s<>\"']+", body)
        num_urls = len(url_matches)
        has_ip_url = any(re.search(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", u) for u in url_matches)
        suspicious_tld = any(re.search(r"\.(xyz|top|work|click|info|cc|tk|ml|ga|cf|gq)/?", u, re.I) for u in url_matches)
        
        urgency_hits = [kw for kw in URGENCY_KEYWORDS if kw in full_text.lower()]
        urgency_score = min(1.0, len(urgency_hits) * 0.25)

        features = FeatureHighlights(
            num_urls=num_urls,
            has_ip_url=has_ip_url,
            suspicious_tld=suspicious_tld,
            body_length=len(body),
            urgency_score=urgency_score,
        )

        # 2. Model Prediction
        if self.is_ready():
            X_word = self.vectorizers["word"].transform([full_text])
            X_char = self.vectorizers["char"].transform([full_text])
            X = hstack([X_word, X_char])
            probs = self.model.predict_proba(X)[0]
            phish_prob = float(probs[1])
        else:
            # Fallback heuristic score if models not yet on disk
            phish_prob = 0.85 if (has_ip_url or urgency_score >= 0.5) else (0.1 if num_urls == 0 else 0.45)

        prediction = 1 if phish_prob >= 0.5 else 0
        label_name = "Phishing" if prediction == 1 else "Legitimate"
        confidence = round(float(phish_prob if prediction == 1 else 1.0 - phish_prob), 4)

        if phish_prob >= 0.85:
            risk_level = "CRITICAL"
        elif phish_prob >= 0.60:
            risk_level = "HIGH"
        elif phish_prob >= 0.35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Top signals
        signals = []
        if has_ip_url:
            signals.append("Direct IP address found in hyperlink target")
        if suspicious_tld:
            signals.append("Link points to high-risk anomalous top-level domain (.xyz, .top, etc.)")
        if urgency_hits:
            signals.append(f"Urgency / coercive language triggers: {', '.join(urgency_hits[:3])}")
        if num_urls > 3:
            signals.append(f"High link density ({num_urls} URLs extracted)")

        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        return EmailAnalysisResponse(
            prediction=prediction,
            label_name=label_name,
            phishing_probability=round(phish_prob, 4),
            confidence=confidence,
            risk_level=risk_level,
            features=features,
            top_signals=signals,
            inference_time_ms=inference_time_ms,
        )
