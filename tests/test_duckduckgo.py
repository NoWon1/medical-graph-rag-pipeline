import pytest
from unittest.mock import patch, MagicMock
import cancer_retrieval

@patch("cancer_retrieval.DDGS")
def test_duckduckgo_search_success_context_manager(mock_ddgs):
    """Test successful search using the DDGS context manager."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance

    mock_ddgs_instance.text.return_value = [
        {"title": "Test Title", "href": "https://test.com", "body": "Test Body"}
    ]

    results = cancer_retrieval._duckduckgo_search("cancer treatment", max_results=1)

    assert len(results) == 1
    assert results[0]["title"] == "Test Title"
    assert results[0]["url"] == "https://test.com"
    assert results[0]["snippet"] == "Test Body"

    # Verify text was called with the primary query (since "cancer" is in the query)
    mock_ddgs_instance.text.assert_called_once_with("cancer treatment", max_results=1)

@patch("cancer_retrieval.DDGS")
def test_duckduckgo_search_fallback_query(mock_ddgs):
    """Test search falling back to broad query when primary query yields no results."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance

    # First call returns empty, second call returns results
    mock_ddgs_instance.text.side_effect = [
        [], # First call (primary query)
        [{"title": "Fallback Title", "href": "https://fallback.com", "body": "Fallback Body"}] # Second call
    ]

    # Query without medical terms to trigger suffix appending
    results = cancer_retrieval._duckduckgo_search("unknown topic", max_results=1)

    assert len(results) == 1
    assert results[0]["title"] == "Fallback Title"

    # Verify text was called twice: once with appended terms, once with original query
    assert mock_ddgs_instance.text.call_count == 2
    mock_ddgs_instance.text.assert_any_call("unknown topic cancer oncology", max_results=1)
    mock_ddgs_instance.text.assert_any_call("unknown topic", max_results=1)

@patch("cancer_retrieval.DDGS")
def test_duckduckgo_search_exception_fallback(mock_ddgs):
    """Test fallback to direct object instantiation when context manager raises exception."""
    # Setup context manager to raise exception
    mock_ddgs_instance = MagicMock()
    mock_ddgs.return_value.__enter__.side_effect = Exception("Context manager failed")

    # Setup the direct object instance
    mock_ddgs.return_value.text.return_value = [
        {"title": "Direct Object Title", "href": "https://direct.com", "body": "Direct Body"}
    ]

    results = cancer_retrieval._duckduckgo_search("cancer", max_results=1)

    assert len(results) == 1
    assert results[0]["title"] == "Direct Object Title"

    # Verify the direct object text was called
    mock_ddgs.return_value.text.assert_called_once_with("cancer", max_results=1)

@patch("cancer_retrieval.DDGS")
def test_duckduckgo_search_complete_failure(mock_ddgs):
    """Test graceful handling when both context manager and direct object fail."""
    # Setup context manager to raise exception
    mock_ddgs.return_value.__enter__.side_effect = Exception("Context manager failed")

    # Setup direct object to also raise exception
    mock_ddgs.return_value.text.side_effect = Exception("Direct object failed")

    results = cancer_retrieval._duckduckgo_search("cancer", max_results=1)

    # Should return empty list on complete failure
    assert results == []

@patch("cancer_retrieval._DDG_AVAILABLE", False)
def test_duckduckgo_search_not_available():
    """Test early return when DDG is not available."""
    results = cancer_retrieval._duckduckgo_search("cancer", max_results=1)
    assert results == []
