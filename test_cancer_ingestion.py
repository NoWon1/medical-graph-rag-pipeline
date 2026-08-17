import unittest
from cancer_ingestion import clean_text

class TestCancerIngestionCleanText(unittest.TestCase):

    def test_empty_string(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   "), "")
        self.assertEqual(clean_text(None), "")

    def test_math_digit_repair(self):
        # 0 and 1 in mathematical bold
        self.assertEqual(clean_text("Number \U0001D7CE and \U0001D7CF"), "Number 0 and 1")

    def test_unicode_normalization(self):
        # NFKC normalization test
        self.assertEqual(clean_text("\u2160"), "I")

    def test_raw_glyph_codes(self):
        # The regex is r'\s*/uniFB01\s*', so it replaces surrounding spaces as well
        self.assertEqual(clean_text("word /uniFB01/ test"), "wordfi/ test")
        self.assertEqual(clean_text("ef /uniFB01 ciency"), "efficiency")
        self.assertEqual(clean_text("50 /uniF642 "), "50%")

    def test_unicode_ligatures(self):
        self.assertEqual(clean_text("e\ufb03ciency"), "efficiency")
        self.assertEqual(clean_text("te\ufb05"), "test")

    def test_broken_word_stitching(self):
        # rule 1: r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)'
        self.assertEqual(clean_text("ef fi ciency"), "efficiency")

        # rule 2: r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)(?=[a-zA-Z])'
        self.assertEqual(clean_text("ef ficiency"), "efficiency")

        # rule 3: r'(?<=[a-zA-Z])(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)'
        self.assertEqual(clean_text("afi ciency"), "aficiency")

        # non-matching example
        self.assertEqual(clean_text("fi ciency"), "fi ciency")

    def test_docling_placeholder_tokens(self):
        # The regex is `re.sub(r'', '', text, flags=re.IGNORECASE)` in the source
        # But `r''` doesn't do much (it replaces empty strings with empty strings).
        # We just verify it does not crash or corrupt text.
        self.assertEqual(clean_text("Normal text"), "Normal text")

    def test_running_headers_footers(self):
        # 7. Running journal headers/footers
        text = "Some normal text\nPage 1 of 10\nMore text"
        self.assertEqual(clean_text(text), "Some normal text\nMore text")
        text2 = "Some normal text\n© 2023 Something\nMore text"
        self.assertEqual(clean_text(text2), "Some normal text\nMore text")
        text3 = "guideline version 2.0\nNormal text"
        self.assertEqual(clean_text(text3), "Normal text")

    def test_author_et_al_headers(self):
        text = "Normal text\nSmith et al. 2023\nMore text"
        self.assertEqual(clean_text(text), "Normal text\nMore text")

    def test_doi_url_lines(self):
        text = "Text\nhttps://example.com/test\nMore text"
        self.assertEqual(clean_text(text), "Text\nMore text")
        text2 = "Text\ndoi: 10.1000/182\nMore text"
        self.assertEqual(clean_text(text2), "Text\nMore text")
        text3 = "www.website.com/path\nText"
        self.assertEqual(clean_text(text3), "Text")

    def test_references_section_cutoff(self):
        text = "Main content here.\n## References\n1. Smith 2020\n2. Doe 2021"
        self.assertEqual(clean_text(text), "Main content here.")
        text2 = "Main content.\n*Works Cited*\nSmith 2020"
        self.assertEqual(clean_text(text2), "Main content.")
        text3 = "Main content.\nBibliography\n1. Author"
        self.assertEqual(clean_text(text3), "Main content.")

    def test_excess_whitespace(self):
        text = "This   has    too  much space."
        self.assertEqual(clean_text(text), "This has too much space.")
        text2 = "Line 1\n\n\n\nLine 2"
        self.assertEqual(clean_text(text2), "Line 1\n\nLine 2")
        text3 = "Tab\t\t\tseparated"
        self.assertEqual(clean_text(text3), "Tab separated")

    def test_isolated_page_numbers(self):
        text = "Text\n  42  \nMore text"
        self.assertEqual(clean_text(text), "Text\nMore text")
        text2 = "999\nText"
        self.assertEqual(clean_text(text2), "Text")

if __name__ == '__main__':
    unittest.main()
