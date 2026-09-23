import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_analyze_endpoint():
    payload = {
        "subject": "URGENT: Password expired - verify your account now!",
        "body": "Dear employee, click http://192.168.1.1/login to verify your password immediately.",
        "sender": "security-alert@suspicious-bank.com",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["label_name"] == "Phishing"
    assert data["phishing_probability"] > 0.5
    assert data["risk_level"] in ["HIGH", "CRITICAL"]
    assert "top_signals" in data
    assert len(data["top_signals"]) > 0


def test_analyze_benign_endpoint():
    payload = {
        "subject": "Quarterly financial review meeting notes",
        "body": "Hi team, attached are the meeting notes from yesterday's sync. Best regards, Alice.",
        "sender": "alice@company.com",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["label_name"] == "Legitimate"
    assert 0.0 <= data["phishing_probability"] <= 0.5

