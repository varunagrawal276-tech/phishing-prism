"""
Pydantic schemas for PRISM-Phish REST API.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class EmailAnalysisRequest(BaseModel):
    subject: Optional[str] = Field(default="", description="Email subject line")
    body: str = Field(..., description="Email body content (text or HTML)")
    sender: Optional[str] = Field(default="", description="Sender email address (From header)")
    receiver: Optional[str] = Field(default="", description="Recipient email address (To header)")


class FeatureHighlights(BaseModel):
    num_urls: int = Field(default=0)
    has_ip_url: bool = Field(default=False)
    suspicious_tld: bool = Field(default=False)
    body_length: int = Field(default=0)
    urgency_score: float = Field(default=0.0)


class EmailAnalysisResponse(BaseModel):
    prediction: int = Field(..., description="0 for Legitimate, 1 for Phishing")
    label_name: str = Field(..., description="'Legitimate' or 'Phishing'")
    phishing_probability: float = Field(..., description="Predicted probability [0.0, 1.0]")
    confidence: float = Field(..., description="Confidence score [0.5, 1.0]")
    risk_level: str = Field(..., description="'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'")
    features: FeatureHighlights
    top_signals: List[str] = Field(default_factory=list, description="Top detected suspicious indicators")
    inference_time_ms: float = Field(..., description="End-to-end inference latency in ms")


class HealthCheckResponse(BaseModel):
    status: str
    models_loaded: List[str]
    version: str = "1.0.0"
