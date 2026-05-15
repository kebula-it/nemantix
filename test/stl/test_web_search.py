import pytest
from nemantix.core import Toolset
from nemantix.stl.web_search.base import WebSearchToolset


class TestWebSearchToolset:
    # --- Web Search Tests ---

    @pytest.mark.external
    def test_search_web_success(self):
        """Test a successful live web search returning real results."""
        ts_search = Toolset.get_tool(
            tool_name="WebSearchToolset.search_web", instance_args=("us-en",)
        )

        # Execute a real query that is guaranteed to have results
        results = ts_search(
            query="Python programming language", max_results=2, backend="duckduckgo"
        )

        # Assertions
        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 2

        # Verify the structure of the real response matches our expected mapping
        first_result = results[0]
        assert "title" in first_result
        assert "link" in first_result
        assert "snippet" in first_result
        assert first_result["link"].startswith("http")

    # --- News Search Tests ---

    @pytest.mark.external
    def test_search_news_success(self):
        """Test a successful live news search."""
        ts_news = Toolset.get_tool(
            tool_name="WebSearchToolset.search_news", instance_args=("us-en",)
        )

        # Query a broad topic guaranteed to be in the news cycle
        results = ts_news(query="Technology", max_results=2, backend="duckduckgo")

        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 2

        # Verify the structure of the real news response
        first_result = results[0]
        assert "title" in first_result
        assert "source" in first_result
        assert "link" in first_result
        assert first_result["link"].startswith("http")

    # --- Initialization Tests ---

    def test_region_config(self):
        """Test that the toolset initializes directly with the correct region."""
        ts = WebSearchToolset(region="uk-en")
        assert ts.region == "uk-en"
