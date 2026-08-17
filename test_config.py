import unittest
from config import detect_cancer_type, CANCER_KEYWORDS

class TestDetectCancerType(unittest.TestCase):
    def test_detect_cancer_type_happy_path(self):
        # Test mapping of explicit keywords to expected labels based on CANCER_KEYWORDS
        # Current keys are: osteosarcoma, leukemia, melanoma, breast, lung, skin
        self.assertEqual(detect_cancer_type("patient_osteosarcoma_scan.pdf"), "osteosarcoma")
        self.assertEqual(detect_cancer_type("leukemia_report.txt"), "leukemia")
        self.assertEqual(detect_cancer_type("melanoma_doc.docx"), "melanoma")
        self.assertEqual(detect_cancer_type("breast_cancer.pdf"), "breast")
        self.assertEqual(detect_cancer_type("lung_cancer_results.txt"), "lung")
        self.assertEqual(detect_cancer_type("skin_lesion_analysis.pdf"), "skin")

    def test_detect_cancer_type_case_insensitive(self):
        # Function should convert to lower case and still match
        self.assertEqual(detect_cancer_type("PATIENT_OSTEOSARCOMA_SCAN.PDF"), "osteosarcoma")
        self.assertEqual(detect_cancer_type("Leukemia_Report.txt"), "leukemia")
        self.assertEqual(detect_cancer_type("BReast_CANCER.pdf"), "breast")

    def test_detect_cancer_type_fallback(self):
        # Should return 'general' when no keywords are present
        self.assertEqual(detect_cancer_type("blood_test_results.pdf"), "general")
        self.assertEqual(detect_cancer_type("annual_physical.docx"), "general")

    def test_detect_cancer_type_edge_cases(self):
        # Empty string
        self.assertEqual(detect_cancer_type(""), "general")

        # Incomplete words or partial matches - it's a simple `in` check, so "osteosarcomas" should match "osteosarcoma"
        self.assertEqual(detect_cancer_type("osteosarcomas_report.pdf"), "osteosarcoma")

        # Multiple keywords present (should return the first one matched in CANCER_KEYWORDS list)
        # Assuming CANCER_KEYWORDS order is: osteosarcoma, leukemia, melanoma, breast, lung, skin
        # "osteosarcoma" is first, "lung" is later
        self.assertEqual(detect_cancer_type("lung_and_osteosarcoma_analysis.pdf"), "osteosarcoma")

if __name__ == '__main__':
    unittest.main()
