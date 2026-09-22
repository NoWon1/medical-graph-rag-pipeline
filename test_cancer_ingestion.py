from cancer_ingestion import detect_content_type

def test_detect_content_type():
    # 1. Figure captions
    assert detect_content_type("Figure 1 shows the results.") == "figure_caption"
    assert detect_content_type("Fig. 2 is an illustration.") == "figure_caption"
    assert detect_content_type("Please refer to fig 10a.") == "figure_caption"

    # 2. Table captions
    assert detect_content_type("Table 1 summarizes the data.") == "table_caption"
    assert detect_content_type("Results in table 45.") == "table_caption"

    # 3. Statistical methods
    assert detect_content_type("The p-value was significant.") == "statistical_methods"
    assert detect_content_type("We used chi-square test.") == "statistical_methods"
    assert detect_content_type("The hazard ratio is 1.5.") == "statistical_methods"
    assert detect_content_type("p < 0.05 was considered significant.") == "statistical_methods"

    # 4. Clinical recommendations
    assert detect_content_type("This is the standard of care.") == "clinical_recommendation"
    assert detect_content_type("Treatment plan includes surgery.") == "clinical_recommendation"
    assert detect_content_type("This drug is contraindicated.") == "clinical_recommendation"

    # 5. Prognosis data
    assert detect_content_type("The 5-year survival rate is high.") == "prognosis_data"
    assert detect_content_type("Patient outcome improved.") == "prognosis_data"
    assert detect_content_type("Risk of recurrence is low.") == "prognosis_data"

    # 6. Fallback (clinical_text)
    assert detect_content_type("The patient presented with a headache.") == "clinical_text"
    assert detect_content_type("") == "clinical_text"

    # 7. Edge Cases / Priority (Returns first matched, order matters in code)
    # figure_caption > table_caption > statistical_methods > clinical_recommendation > prognosis_data
    # "Figure 1 has a p-value" -> figure_caption
    assert detect_content_type("Figure 1 has a p-value") == "figure_caption"
    # "Table 2 describes the survival rate" -> table_caption
    assert detect_content_type("Table 2 describes the survival rate") == "table_caption"

    # 8. Case insensitivity
    assert detect_content_type("P-VALUE IS 0.05") == "statistical_methods"
    assert detect_content_type("STANDARD OF CARE") == "clinical_recommendation"
    assert detect_content_type("OVERALL SURVIVAL") == "prognosis_data"

from unittest.mock import MagicMock
from cancer_ingestion import extract_table_data

def test_extract_table_data():
    doc = MagicMock()
    page1 = MagicMock()

    # block[4] is the text. block length must be at least 5.

    valid_text_1 = "Valid row 1 with numbers 100, 200, 300 and definitely longer than 40 characters."
    valid_text_2 = "Valid row 2 with   spaces  1.5, 2.5, 3.5  and definitely longer than 40 characters."
    short_text = "Short 1 2 3" # < 40 chars
    no_num_text = "This text is very long and has more than forty characters but no digits."
    too_few_nums = "This text is very long and has more than forty characters but only 10."

    invalid_block_short = (0, 1, 2) # len < 5

    page1.get_text.return_value = [
        (0, 0, 0, 0, valid_text_1),
        (0, 0, 0, 0, valid_text_2),
        (0, 0, 0, 0, short_text),
        (0, 0, 0, 0, no_num_text),
        (0, 0, 0, 0, too_few_nums),
        (0, 0, 0, 0, valid_text_1), # Duplicate
        invalid_block_short
    ]

    doc.__len__.return_value = 1
    doc.__getitem__.side_effect = lambda idx: page1 if idx == 0 else None

    result = extract_table_data(doc)

    assert "## Key Numerical Data" in result
    assert valid_text_1 in result

    # Spaces should be cleaned up in valid_text_2
    cleaned_valid_text_2 = "Valid row 2 with spaces 1.5, 2.5, 3.5 and definitely longer than 40 characters."
    assert cleaned_valid_text_2 in result
    assert short_text not in result
    assert no_num_text not in result
    assert too_few_nums not in result

    # Should only appear once (duplicates filtered)
    assert result.count(valid_text_1) == 1

    # Test empty document / no matches
    empty_doc = MagicMock()
    empty_doc.__len__.return_value = 0
    assert extract_table_data(empty_doc) == ""

    no_match_doc = MagicMock()
    no_match_page = MagicMock()
    no_match_page.get_text.return_value = [(0, 0, 0, 0, short_text)]
    no_match_doc.__len__.return_value = 1
    no_match_doc.__getitem__.return_value = no_match_page
    assert extract_table_data(no_match_doc) == ""

    # Test > 25 limit
    large_doc = MagicMock()
    large_page = MagicMock()
    blocks = []
    for i in range(30):
        # make sure it's valid: 3 numbers, > 40 chars
        blocks.append((0, 0, 0, 0, f"Valid row {i} with numbers 100, 200, 300 and definitely longer than 40 characters. PAD PAD PAD PAD"))

    large_page.get_text.return_value = blocks
    large_doc.__len__.return_value = 1
    large_doc.__getitem__.return_value = large_page

    large_result = extract_table_data(large_doc)

    # Check limit of 25
    assert "Valid row 24" in large_result
    assert "Valid row 25" not in large_result
