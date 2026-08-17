import pytest
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
