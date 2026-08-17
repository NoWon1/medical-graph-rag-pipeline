import pytest
from cancer_retrieval import detect_query_intent
from config import (
    INTENT_NON_CHEMO_INTERACTION,
    INTENT_PROTOCOL_DETAIL,
    INTENT_FOOD_GUIDANCE,
    INTENT_CANCER_DRUGS_EFFECTS,
    INTENT_GENERAL_GRAPH
)

def test_detect_query_intent_general_graph():
    result = detect_query_intent("Hello, how are you?")
    assert result["intent"] == INTENT_GENERAL_GRAPH
    assert result["cancer_name"] is None
    assert result["chemo_drug"] is None
    assert result["non_chemo_drug"] is None
    assert result["protocol"] is None
    assert result["eating_effect"] is None
    assert result["has_food_signal"] is False
    assert result["has_interaction_signal"] is False

def test_detect_query_intent_non_chemo_interaction():
    result = detect_query_intent("I am taking aspirin with cisplatin.")
    assert result["intent"] == INTENT_NON_CHEMO_INTERACTION
    assert result["non_chemo_drug"] == "aspirin"
    assert result["chemo_drug"] == "cisplatin"

def test_detect_query_intent_protocol_detail():
    result = detect_query_intent("What is the map protocol?")
    assert result["intent"] == INTENT_PROTOCOL_DETAIL
    assert result["protocol"] == "map"

def test_detect_query_intent_food_guidance_chemo():
    result = detect_query_intent("What foods should I avoid with cisplatin?")
    assert result["intent"] == INTENT_FOOD_GUIDANCE
    assert result["chemo_drug"] == "cisplatin"
    assert result["has_food_signal"] is True

def test_detect_query_intent_cancer_drugs_effects():
    result = detect_query_intent("What is the best treatment for lung cancer?")
    assert result["intent"] == INTENT_CANCER_DRUGS_EFFECTS
    assert result["cancer_name"] == "lung cancer"

def test_detect_query_intent_eating_effect():
    result = detect_query_intent("I have nausea.")
    assert result["intent"] == INTENT_GENERAL_GRAPH
    assert result["eating_effect"] == "nausea"

def test_detect_query_intent_patient_report():
    # If the user has both cancer and food intent, without a specific chemo drug it should be cancer_drugs_effects
    result = detect_query_intent("Should I avoid food?", patient_report="Patient has lung cancer.")
    assert result["intent"] == INTENT_CANCER_DRUGS_EFFECTS
    assert result["cancer_name"] == "lung cancer"
    assert result["chemo_drug"] is None
    assert result["has_food_signal"] is True

def test_detect_query_intent_order_precedence():
    # non_chemo with chemo should take precedence over protocol or food or cancer
    result = detect_query_intent("I take aspirin and cisplatin for lung cancer with food map")
    assert result["intent"] == INTENT_NON_CHEMO_INTERACTION

    # protocol takes precedence over chemo+food
    result = detect_query_intent("map protocol with cisplatin and food")
    assert result["intent"] == INTENT_PROTOCOL_DETAIL

    # chemo+food takes precedence over cancer+food
    result = detect_query_intent("cisplatin and food for lung cancer")
    assert result["intent"] == INTENT_FOOD_GUIDANCE

    # cancer+chemo takes precedence over just chemo
    result = detect_query_intent("cisplatin for lung cancer")
    assert result["intent"] == INTENT_CANCER_DRUGS_EFFECTS

def test_detect_query_intent_non_chemo_interaction_keywords():
    result = detect_query_intent("aspirin interaction")
    assert result["intent"] == INTENT_NON_CHEMO_INTERACTION
    assert result["non_chemo_drug"] == "aspirin"
    assert result["has_interaction_signal"] is True
