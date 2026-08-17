import unittest
import hashlib
from cancer_app import _report_hash

class TestCancerApp(unittest.TestCase):
    def test_report_hash_standard(self):
        text = "sample patient report"
        expected = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        self.assertEqual(_report_hash(text), expected)

    def test_report_hash_empty_string(self):
        self.assertEqual(_report_hash(""), "")

    def test_report_hash_none(self):
        self.assertEqual(_report_hash(None), "")

if __name__ == '__main__':
    unittest.main()
