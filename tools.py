from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# Gereken Langchain (LLM/Agentic AI framework) kütüphanelerini ekledik. 

search = DuckDuckGoSearchRun()


@tool
def search_tool(query: str) -> str:
    """Searches the web for information about a given query."""
    return search.run(query)


# String alıp string göndüren bir fonksiyon oluşturuyoruz,  DuckDuckGoSearchRun() fonksiyonunu çalıştırıyoruz.
# Query'den aldığımız inputla aratma işlemini yapıyoruz.

 
 