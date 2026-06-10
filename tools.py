from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

search = DuckDuckGoSearchRun()

@tool
def search_tool(query: str) -> str:
    """Searches the web for information about a given query."""
    return search.run(query)