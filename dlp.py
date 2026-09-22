import re

def redact_phi(text: str) -> str:
    if not text:
        return text
    # MRN / IDs
    text = re.sub(r'(?i)\b(?:MRN|ID|RECORD|ACC|PATIENT\s*ID)[:#\s]+[A-Z0-9-]{4,15}\b', r'[REDACTED_MRN]', text)
    # SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', r'[REDACTED_SSN]', text)
    # Phone numbers
    text = re.sub(r'(?:(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4})\b', r'[REDACTED_PHONE]', text)
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', r'[REDACTED_EMAIL]', text)
    # Dates
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', r'[REDACTED_DATE]', text)
    text = re.sub(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b', r'[REDACTED_DATE]', text, flags=re.IGNORECASE)
    # Names
    text = re.sub(r'(?i)(?:Patient\s*Name|Patient)[:\s]+[A-Z][a-z]+\s+[A-Z][a-z]+', r'Patient: [REDACTED_NAME]', text)
    text = re.sub(r'\b(?:Dr\.|Doctor)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', r'Dr. [REDACTED_NAME]', text)
    # ZIP codes
    text = re.sub(r'\b\d{5}(?:-\d{4})?\b', r'[REDACTED_ZIP]', text)
    return text
