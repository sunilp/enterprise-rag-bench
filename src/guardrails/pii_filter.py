"""PII detection and redaction for RAG pipelines.

Combines regex patterns for structured PII (SSN, credit cards, emails)
with optional NER-based detection for names and addresses. Critical
for enterprise RAG where retrieved documents may contain customer data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set


class PIIType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IBAN = "iban"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"


@dataclass
class PIIMatch:
    """A detected PII instance."""
    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float


# Compiled regex patterns for structured PII
_PII_PATTERNS: Dict[PIIType, re.Pattern] = {
    PIIType.EMAIL: re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    PIIType.PHONE: re.compile(
        r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    PIIType.SSN: re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    PIIType.CREDIT_CARD: re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),
    PIIType.IBAN: re.compile(
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})?\b"
    ),
    PIIType.IP_ADDRESS: re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
}


class PIIFilter:
    """Detect and redact PII in text.

    Usage:
        pii_filter = PIIFilter()
        clean_text = pii_filter.redact(document_text)
        matches = pii_filter.detect(document_text)
    """

    def __init__(
        self,
        pii_types: Optional[Set[PIIType]] = None,
        redaction_char: str = "X",
        custom_patterns: Optional[Dict[str, re.Pattern]] = None,
    ):
        self.pii_types = pii_types or set(PIIType)
        self.redaction_char = redaction_char
        self.custom_patterns = custom_patterns or {}

    def detect(self, text: str) -> List[PIIMatch]:
        """Detect PII instances in text."""
        matches: List[PIIMatch] = []

        for pii_type in self.pii_types:
            pattern = _PII_PATTERNS.get(pii_type)
            if pattern is None:
                continue
            for match in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                    )
                )

        for name, pattern in self.custom_patterns.items():
            for match in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        pii_type=PIIType.EMAIL,  # placeholder
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,
                    )
                )

        return sorted(matches, key=lambda m: m.start)

    def redact(self, text: str, replacement: Optional[str] = None) -> str:
        """Redact all detected PII from text."""
        matches = self.detect(text)
        if not matches:
            return text

        result = list(text)
        for match in reversed(matches):
            redacted = replacement or f"[{match.pii_type.value.upper()}]"
            result[match.start : match.end] = list(redacted)

        return "".join(result)

    def has_pii(self, text: str) -> bool:
        """Quick check: does the text contain any PII?"""
        return len(self.detect(text)) > 0

    def validate_credit_card(self, number: str) -> bool:
        """Luhn algorithm validation for credit card numbers."""
        digits = [int(d) for d in number if d.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return False
        checksum = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
        return checksum % 10 == 0
