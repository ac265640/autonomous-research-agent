"""
tools/web_tools.py
------------------
Three tools used by the ResearchAgent.

Function-calling JSON schemas live in agents/research_agent.py so Groq
can see them; the actual Python callables live here.
"""

import asyncio
import httpx
from bs4 import BeautifulSoup
from groq import Groq, RateLimitError


# ---------------------------------------------------------------------------
# TOOL 1 — web_search
# JSON schema (for reference / documentation):
# {
#   "type": "function",
#   "function": {
#     "name": "web_search",
#     "description": "Search the web using DuckDuckGo and return a list of results.",
#     "parameters": {
#       "type": "object",
#       "properties": {
#         "query":       {"type": "string",  "description": "Search query"},
#         "max_results": {"type": "integer", "description": "Max results to return", "default": 5}
#       },
#       "required": ["query"]
#     }
#   }
# }
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search DuckDuckGo HTML and return [{title, url, snippet}, ...].
    Falls back to an error dict on any exception.
    """
    url = "https://duckduckgo.com/html/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        with httpx.Client(timeout=15, headers=headers, follow_redirects=True) as client:
            res = client.post(url, data={"q": query})

        soup = BeautifulSoup(res.text, "html.parser")

        results: list[dict] = []
        for item in soup.select(".result")[:max_results]:
            title_tag = item.select_one(".result__a")
            snippet_tag = item.select_one(".result__snippet")
            url_tag = item.select_one(".result__url")

            if not title_tag:
                continue

            raw_href = title_tag.get("href", "")
            # DuckDuckGo wraps URLs in a redirect — extract the real one
            if "uddg=" in raw_href:
                from urllib.parse import urlparse, parse_qs, unquote
                qs = parse_qs(urlparse(raw_href).query)
                real_url = unquote(qs.get("uddg", [raw_href])[0])
            else:
                real_url = raw_href

            results.append({
                "title":   title_tag.get_text(strip=True),
                "url":     real_url,
                "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
            })

        return results if results else [{"title": "No results", "url": "", "snippet": ""}]

    except Exception as exc:
        return [{"title": "Error", "url": "", "snippet": str(exc)}]


# ---------------------------------------------------------------------------
# TOOL 2 — fetch_and_summarise  (async)
# JSON schema (for reference / documentation):
# {
#   "type": "function",
#   "function": {
#     "name": "fetch_and_summarise",
#     "description": "Fetch a URL, extract the text, and return a ~300-word summary.",
#     "parameters": {
#       "type": "object",
#       "properties": {
#         "url":       {"type": "string",  "description": "URL to fetch"},
#         "max_words": {"type": "integer", "description": "Approx word budget for summary", "default": 300}
#       },
#       "required": ["url"]
#     }
#   }
# }
# ---------------------------------------------------------------------------

async def fetch_and_summarise(url: str, max_words: int = 300) -> str:
    """
    Fetch *url* with httpx (async), extract visible text, summarise with Groq.
    Handles: timeouts, non-HTML, rate-limits, empty pages.
    Multiple calls can be parallelised with asyncio.gather().
    """
    if not url or not url.startswith("http"):
        return f"Skipped invalid URL: {url!r}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    # --- Fetch ---
    try:
        async with httpx.AsyncClient(
            timeout=15,
            headers=headers,
            follow_redirects=True,
        ) as http_client:
            response = await http_client.get(url)

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return f"Skipped non-HTML response ({content_type}) for: {url}"

    except httpx.TimeoutException:
        return f"Timeout fetching: {url}"
    except Exception as exc:
        return f"Fetch error for {url}: {exc}"

    # --- Extract text ---
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove nav/footer/script noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    text = " ".join(paragraphs)

    if not text.strip():
        # Fallback: grab all body text
        text = soup.get_text(" ", strip=True)

    # Truncate to avoid overwhelming the LLM (≈ 4 000 chars ~ 1 000 tokens)
    text = text[:4000]

    if not text.strip():
        return f"No readable text found on: {url}"

    # --- Summarise ---
    groq_client = Groq()
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Summarise the following web page content in at most {max_words} words. "
                        "Be factual and concise. Focus on the key ideas."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=500,
        )
        return completion.choices[0].message.content or "Empty summary returned."

    except RateLimitError:
        # Wait briefly and retry once
        await asyncio.sleep(5)
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": f"Summarise in {max_words} words."},
                    {"role": "user", "content": text[:2000]},
                ],
                max_tokens=400,
            )
            return completion.choices[0].message.content or "Empty summary (retry)."
        except Exception as exc2:
            return f"Rate-limit retry failed: {exc2}"

    except Exception as exc:
        return f"Summarisation error: {exc}"