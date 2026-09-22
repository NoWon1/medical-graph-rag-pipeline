import unittest
from dlp import redact_phi

class TestRedaction(unittest.TestCase):
    def test_mrn(self):
        self.assertEqual(redact_phi("Patient ID: 1234-5678"), "[REDACTED_MRN]")
        self.assertEqual(redact_phi("MRN: AB12345"), "[REDACTED_MRN]")

    def test_ssn(self):
        self.assertEqual(redact_phi("SSN is 123-45-6789."), "SSN is [REDACTED_SSN].")

    def test_phone(self):
        self.assertEqual(redact_phi("Call (555) 123-4567"), "Call [REDACTED_PHONE]")
        self.assertEqual(redact_phi("Call 555-123-4567"), "Call [REDACTED_PHONE]")
        self.assertEqual(redact_phi("Call +1-555-123-4567"), "Call [REDACTED_PHONE]")

    def test_email(self):
        self.assertEqual(redact_phi("Email test@example.com."), "Email [REDACTED_EMAIL].")

    def test_date(self):
        self.assertEqual(redact_phi("Date: 12/05/2023"), "Date: [REDACTED_DATE]")
        self.assertEqual(redact_phi("Date: Jan 5, 2023"), "Date: [REDACTED_DATE]")

    def test_zip(self):
        self.assertEqual(redact_phi("Beverly Hills 90210"), "Beverly Hills [REDACTED_ZIP]")

    def test_names(self):
        self.assertEqual(redact_phi("Patient Name: John Doe"), "Patient: [REDACTED_NAME]")
        self.assertEqual(redact_phi("Patient: Jane Smith"), "Patient: [REDACTED_NAME]")
        self.assertEqual(redact_phi("Dr. House is here"), "Dr. [REDACTED_NAME] is here")

    def test_preserves_clinical_data(self):
        clinical_text = "Blood pressure 120/80 mmHg, tumor 2.3 cm. T2N0M0 staging. 50 mg dosage. 1234. 2024."
        self.assertEqual(redact_phi(clinical_text), clinical_text)

if __name__ == '__main__':
    unittest.main()
