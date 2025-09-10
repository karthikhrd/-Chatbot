import requests

def simple_web_search(query: str) -> str:
    """Fallback web search using DuckDuckGo API"""
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    try:
        res = requests.get(url).json()
        if "RelatedTopics" in res and res["RelatedTopics"]:
            return res["RelatedTopics"][0].get("Text", "No result found")
    except Exception:
        return "Error fetching search results"
    return "No result found"
