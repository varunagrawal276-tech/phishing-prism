#!/usr/bin/env python3
"""
PRISM-Phish Interactive Demo & Quick Test
Run this file to test phishing email detection in real-time.
Usage:
    python demo.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.inference import PhishingInferenceEngine
from src.api.schemas import EmailAnalysisRequest

def print_separator(title=""):
    print("=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)

def display_result(result):
    print(f"\n>> PREDICTION:       [{result.label_name.upper()}]")
    print(f">> CONFIDENCE:       {result.confidence * 100:.2f}%")
    print(f">> PHISHING PROB:    {result.phishing_probability * 100:.2f}%")
    print(f">> RISK LEVEL:       {result.risk_level}")
    print(f">> INFERENCE TIME:   {result.inference_time_ms:.2f} ms")
    
    print("\n[Extracted Threat Features]")
    print(f"  - URLs Detected:       {result.features.num_urls}")
    print(f"  - IP-Address in Link:  {'YES (High Risk!)' if result.features.has_ip_url else 'No'}")
    print(f"  - Suspicious Domain:   {'YES (.xyz/.top etc)' if result.features.suspicious_tld else 'No'}")
    print(f"  - Urgency Coercion:    {result.features.urgency_score * 100:.0f}%")
    
    if result.top_signals:
        print("\n[Security Flags / Red Flags]")
        for s in result.top_signals:
            print(f"  * {s}")

def main():
    print_separator("PRISM-Phish: Live Phishing Detection System")
    print("Initializing inference engine and loading trained model weights...")
    engine = PhishingInferenceEngine()
    
    if not engine.is_ready():
        print("\n[WARNING] Baseline artifacts not found on disk. Using fallback heuristic.")
    else:
        print("[SUCCESS] Loaded model weights and TF-IDF n-gram vectorizers!")

    sample_phish = {
        "sender": "Security Team <security-alert@paypal-account-update.xyz>",
        "subject": "URGENT: Your account has been suspended! Verify identity immediately",
        "body": "Dear customer, We detected unauthorized access to your banking profile. Please click http://192.168.1.105/login to verify your account within 24 hours or your profile will be terminated."
    }
    
    sample_ham = {
        "sender": "John Doe <john.doe@company.org>",
        "subject": "Updated meeting agenda for Thursday quarterly review",
        "body": "Hi team, Attached is the slide deck for Thursday's quarterly architecture review. Please review slides 5-10 before our 2 PM sync in conference room B. Thanks!"
    }

    print("\nChoose an option:")
    print("  1. Test Sample PHISHING email (Simulated attack with IP link & urgency)")
    print("  2. Test Sample LEGITIMATE email (Corporate correspondence)")
    print("  3. Enter custom email (type or paste your own subject and body)")
    print("  4. Exit")
    
    choice = input("\nEnter choice [1-4] (default=1): ").strip() or "1"
    
    if choice == "1":
        print_separator("Testing Sample PHISHING Email")
        print(f"Sender:  {sample_phish['sender']}")
        print(f"Subject: {sample_phish['subject']}")
        print(f"Body:    {sample_phish['body']}")
        req = EmailAnalysisRequest(**sample_phish)
        res = engine.analyze_email(req)
        display_result(res)
    elif choice == "2":
        print_separator("Testing Sample LEGITIMATE Email")
        print(f"Sender:  {sample_ham['sender']}")
        print(f"Subject: {sample_ham['subject']}")
        print(f"Body:    {sample_ham['body']}")
        req = EmailAnalysisRequest(**sample_ham)
        res = engine.analyze_email(req)
        display_result(res)
    elif choice == "3":
        print_separator("Custom Email Input")
        sender = input("Sender (optional): ").strip()
        subject = input("Subject: ").strip()
        print("Body (press Enter when done):")
        body = input().strip()
        req = EmailAnalysisRequest(sender=sender, subject=subject, body=body)
        res = engine.analyze_email(req)
        display_result(res)
    else:
        print("Exiting.")
        return

    print("\n" + "=" * 70)
    print("Want to see the Interactive Web Dashboard?")
    print("Run this command in terminal:")
    print("    python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000")
    print("Then open your browser at: http://localhost:8000/")
    print("=" * 70)

if __name__ == "__main__":
    main()
