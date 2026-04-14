import json
import asyncio
from groq import Groq

from tools.web_tools import web_search, fetch_and_summarise
from tools.file_tools import save_report
from config import SYSTEM_PROMPT


# -------- TOOL SCHEMAS --------

web_search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    }
}

fetch_tool = {
    "type": "function",
    "function": {
        "name": "fetch_and_summarise",
        "description": "Fetch URL and summarise",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_words": {"type": "integer", "default": 300}
            },
            "required": ["url"]
        }
    }
}

save_tool = {
    "type": "function",
    "function": {
        "name": "save_report",
        "description": "Save research report to disk. Always provide full structured markdown content.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {
                    "type": "string",
                    "description": "Full markdown report including Executive Summary, Key Findings, Sources"
                },
                "filename": {"type": "string"}
            },
            "required": ["title", "content", "filename"]
        }
    }
}


class ResearchAgent:
    def __init__(self, model="llama-3.1-8b-instant", max_iterations=10):
        self.client = Groq()
        self.model = model
        self.max_iterations = max_iterations

        self.tools = [web_search_tool, fetch_tool, save_tool]

        self.tool_map = {
            "web_search": web_search,
            "fetch_and_summarise": fetch_and_summarise,
            "save_report": save_report
        }

    def research(self, topic: str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": topic}
        ]

        return asyncio.run(self._run_loop(messages))

    async def _run_loop(self, messages):
        for _ in range(self.max_iterations):

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
    # fallback: save manually if model forgot
                result = save_report(
                    title="Research Report",
                    content=msg.content or "No content generated",
                    filename="auto_report.md"
                )
                return result

            tasks = []

            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)

                if name == "fetch_and_summarise":
                    tasks.append(fetch_and_summarise(**args))
                else:
                    result = self.tool_map[name](**args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": str(result)
                    })

            if tasks:
                results = await asyncio.gather(*tasks)

                for call, res in zip(msg.tool_calls, results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": res
                    })

        return "Max iterations reached"