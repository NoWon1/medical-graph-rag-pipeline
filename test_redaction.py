import unittest
from redaction import ClinicalDataRedactor

class TestClinicalDataRedactor(unittest.TestCase):
    def test_redact_ssn(self):
        text = "Patient SSN is 123-45-6789."
        redacted = ClinicalDataRedactor.redact(text)
        self.assertIn("[REDACTED_SSN]", redacted)
        self.assertNotIn("123-45-6789", redacted)

    def test_redact_mrn(self):
        text = "Patient MRN: 123-456-789."
        redacted = ClinicalDataRedactor.redact(text)
        self.assertIn("[REDACTED_MRN]", redacted)
        self.assertNotIn("123-456-789", redacted)

    def test_redact_phone(self):
        text = "Call at (555) 123-4567."
        redacted = ClinicalDataRedactor.redact(text)
        self.assertIn("[REDACTED_PHONE]", redacted)
        self.assertNotIn("(555) 123-4567", redacted)

    def test_legitimate_medical_data_retained(self):
        text = "Stage HER2 2+, tumor 3.5 cm."
        redacted = ClinicalDataRedactor.redact(text)
        self.assertIn("HER2 2+", redacted)
        self.assertIn("3.5 cm", redacted)

    def test_empty_string(self):
        self.assertEqual(ClinicalDataRedactor.redact(""), "")

    def test_none_input(self):
        self.assertIsNone(ClinicalDataRedactor.redact(None))

if __name__ == '__main__':
    unittest.main()
