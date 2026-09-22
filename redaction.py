import re
from typing import Dict, Pattern

class ClinicalDataRedactor:
    """
    Lightweight, zero-dependency Data Loss Prevention (DLP) utility
    to sanitize PHI/PII from clinical reports prior to external LLM egress.
    """

    PATTERNS: Dict[str, Pattern[str]] = {
        # Social Security Numbers / National IDs
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        # Medical Record Numbers / Account IDs (common clinical prefixes)
        "MRN": re.compile(r"\b(?:MRN|ID|RECORD|ACC|PATIENT\s*ID)[:#\s]+[A-Z0-9-]{4,15}\b", re.IGNORECASE),
        # Telephone and Fax numbers (standard US/international formats)
        "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        # Email Addresses
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        # Dates (ISO, slash, dot formats, excluding standalone 4-digit years)
        "DATE": re.compile(
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
            re.IGNORECASE,
        ),
        # ZIP / Postal Codes (5-digit or 5+4 format)
        "ZIP": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
        # Explicit Patient/Doctor Name labels common in EHR headers
        "LABELED_NAME": re.compile(
            r"(?:Patient(?:\s+Name)?|Provider|Physician|Doctor|Dr\.)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
            re.IGNORECASE,
        ),
    }

    @classmethod
    def redact(cls, text: str) -> str:
        """
        Scans and sanitizes known PHI patterns from untrusted clinical text.
        """
        if not text or not isinstance(text, str):
            return text

        sanitized = text

        # Handle labeled name contexts first to avoid colliding with secondary tokens
        sanitized = cls.PATTERNS["LABELED_NAME"].sub(r"Patient: [REDACTED_NAME]", sanitized)

        # Apply standard categorical token masking
        for token_type, pattern in cls.PATTERNS.items():
            if token_type == "LABELED_NAME":
                continue
            sanitized = pattern.sub(f"[REDACTED_{token_type}]", sanitized)

        return sanitized
