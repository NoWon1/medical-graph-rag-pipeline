import unittest
from unittest.mock import patch
import config

class TestGetSourceUrl(unittest.TestCase):

    @patch.dict('config.SOURCE_URLS', {'test-source': 'https://example.com/123'})
    def test_get_source_url_existing_key(self):
        """Test getting the URL for an existing source name."""
        self.assertEqual(
            config.get_source_url("test-source"),
            "https://example.com/123"
        )

    def test_get_source_url_missing_key(self):
        """Test getting the URL for a non-existent source name."""
        self.assertEqual(config.get_source_url("non-existent-review"), "")

    def test_get_source_url_empty_string(self):
        """Test getting the URL with an empty string."""
        self.assertEqual(config.get_source_url(""), "")

if __name__ == '__main__':
    unittest.main()
