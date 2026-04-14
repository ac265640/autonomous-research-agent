import httpx
from bs4 import BeautifulSoup
import httpx



def web_search(query: str, max_results: int = 5):
    url = "https://duckduckgo.com/html/"
    
    try:
        with httpx.Client(timeout=10) as client:
            res = client.post(url, data={"q": query})

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(res.text, "html.parser")

        results = []
        for a in soup.select(".result__a")[:max_results]:
            results.append({
                "title": a.get_text(),
                "url": a["href"],
                "snippet": ""
            })

        return results

    except Exception as e:
        return [{"title": "Error", "url": "", "snippet": str(e)}]




async def fetch_and_summarise(url: str, max_words: int = 300):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(url)

        if "text/html" not in res.headers.get("Content-Type", ""):
            return f"Skipped non-HTML: {url}"

        soup = BeautifulSoup(res.text, "html.parser")
        text = " ".join(p.get_text() for p in soup.find_all("p"))
        text = text[:5000]

        from groq import Groq
        client = Groq()

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Summarise this content"},
                {"role": "user", "content": text}
            ]
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"