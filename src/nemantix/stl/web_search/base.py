from ddgs import DDGS
from typing import List, Dict
from nemantix.core import tool, Toolset


class WebSearchToolset(Toolset):
    """
    A toolset for searching the live web.
    Does not require an API key.

    Args:
        region (str): Region code (e.g., 'us-en', 'uk-en', 'wt-wt' for world). Defaults to "wt-wt".
    """

    def __init__(self, region: str = "wt-wt"):
        super().__init__()
        self.region = region

    @tool
    def search_web(self,
                   query: str,
                   max_results: int = 5,
                   backend: str = "auto") -> List[Dict[str, str]]:
        """
        Performs a general web search with adjustable result limits.

        Args:
            query (str): The search terms or question to look up.
            max_results (int, optional): Number of results to return. Defaults to 5.
            backend (str, optional): The search backend to use. Defaults to 'auto'.
        Returns:
            List[Dict[str, str]]: A list of dictionaries containing title, link, and snippet.

        Example call:
            search_web(
                query="Python Metaclasses",
                max_results=3
            )
        """
        results = []
        try:
            with DDGS() as ddgs:
                ddgs_gen = ddgs.text(
                    query=query,
                    region=self.region,
                    max_results=max_results,
                    backend=backend,
                )
                for r in ddgs_gen:
                    results.append({
                        "title": r.get("title", ""),
                        "link": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]

        return results

    @tool
    def search_news(self,
                    query: str,
                    max_results: int = 5,
                    backend: str = "auto") -> List[Dict[str, str]]:
        """
        Searches strictly for news articles with adjustable result limits.

        Args:
            query (str): The topic or current event to search for.
            max_results (int, optional): Number of results to return. Defaults to 5.
            backend (str, optional): The search backend to use. Defaults to 'auto'.
        Returns:
            List[Dict[str, str]]: A list of dictionaries containing title, link, source, date, and snippet.

        Example call:
            search_news(
                query="renewable energy breakthroughs",
                max_results=10
            )
        """
        results = []
        try:
            with DDGS() as ddgs:
                ddgs_gen = ddgs.news(
                    query=query,
                    region=self.region,
                    max_results=max_results,
                    backend=backend,
                )
                for r in ddgs_gen:
                    results.append({
                        "title": r.get("title", ""),
                        "link": r.get("url", ""),
                        "source": r.get("source", ""),
                        "date": r.get("date", ""),
                        "snippet": r.get("body", ""),
                    })
        except Exception as e:
            return [{"error": f"News search failed: {str(e)}"}]

        return results
