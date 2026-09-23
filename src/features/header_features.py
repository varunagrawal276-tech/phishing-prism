"""
Header/sender-based feature extraction for PRISM-Phish.

Extracts features from sender, receiver, and email header information.
Only computes features from fields that are actually present.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Known free email providers
FREE_EMAIL_PROVIDERS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "mail.com", "protonmail.com", "yandex.com", "zoho.com", "icloud.com",
    "live.com", "msn.com", "gmx.com", "inbox.com", "fastmail.com",
}


def extract_email_domain(email_addr: str) -> Optional[str]:
    """Extract domain from an email address."""
    if not isinstance(email_addr, str):
        return None
    match = re.search(r'@([A-Za-z0-9.-]+\.[A-Za-z]{2,})', email_addr)
    return match.group(1).lower() if match else None


def extract_display_name(email_addr: str) -> Optional[str]:
    """Extract display name from 'Name <email>' format."""
    if not isinstance(email_addr, str):
        return None
    match = re.match(r'^"?([^"<]+)"?\s*<', email_addr.strip())
    return match.group(1).strip() if match else None


def compute_header_features(
    sender: Optional[str] = None,
    receiver: Optional[str] = None,
) -> dict:
    """
    Compute header/sender-based features.

    Only uses fields that are actually present — never fabricates metadata.
    """
    features = {
        "sender_present": False,
        "sender_domain_length": 0,
        "free_email_provider": 0,
        "display_name_email_mismatch": 0,
        "sender_receiver_mismatch": 0,
        "receiver_present": False,
    }

    # Sender features
    if sender and isinstance(sender, str) and sender.strip():
        features["sender_present"] = True

        sender_domain = extract_email_domain(sender)
        if sender_domain:
            features["sender_domain_length"] = len(sender_domain)
            features["free_email_provider"] = int(
                sender_domain in FREE_EMAIL_PROVIDERS
            )

        # Display name vs email address mismatch
        display_name = extract_display_name(sender)
        if display_name and sender_domain:
            # Check if display name contains a different domain
            display_domain = extract_email_domain(display_name)
            if display_domain and display_domain != sender_domain:
                features["display_name_email_mismatch"] = 1

    # Receiver features
    if receiver and isinstance(receiver, str) and receiver.strip():
        features["receiver_present"] = True

        # Sender-receiver domain mismatch
        if features["sender_present"]:
            sender_domain = extract_email_domain(sender)
            receiver_domain = extract_email_domain(receiver)
            if sender_domain and receiver_domain:
                features["sender_receiver_mismatch"] = int(
                    sender_domain != receiver_domain
                )

    return features
