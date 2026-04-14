"""
agents/research_agent.py
------------------------
ResearchAgent — uses Groq function-calling to orchestrate:
    - web_search        (sync)
    - fetch_and_summarise (async, parallelised with asyncio.gather)
    - save_report       (sync)

Memory:
    - Short-term : ConversationBuffer  (truncation + summarisation)
    - Long-term  : LongTermMemory / ChromaDB (retrieved before every research run)
"""
from __future__ import annotations

import json
import asyncio
import uuid

from groq import Groq, BadRequestError

from tools.web_tools import web_search, fetch_and_summarise
from tools.file_tools import save_report
from config import SYSTEM_PROMPT

from memory.conversation_buffer import ConversationBuffer
from memory.long_term_memory import LongTermMemory


# ---------------------------------------------------------------------------
# FUNCTION-CALLING TOOL SCHEMAS
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web using DuckDuckGo. "
            "Returns a list of {title, url, snippet} dicts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results (default 5)"},
            },
            "required": ["query"],
        },
    },
}

FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_and_summarise",
        "description": (
            "Fetch a URL and return a ~300-word factual summary of its content. "
            "Use this on the top 2 URLs from each search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url":       {"type": "string",  "description": "Full URL to fetch"},
                "max_words": {"type": "integer", "description": "Word budget for summary (default 300)"},
            },
            "required": ["url"],
        },
    },
}

SAVE_TOOL = {
    "type": "function",
    "function": {
        "name": "save_report",
        "description": (
            "Save the final structured research report as a Markdown file. "
            "Call this ONLY ONCE after all research is complete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title":    {"type": "string", "description": "Report title"},
                "content":  {"type": "string", "description": "Full Markdown body of the report"},
                "filename": {"type": "string", "description": "File name (no extension, no spaces)"},
            },
            "required": ["title", "content", "filename"],
        },
    },
}

ALL_TOOLS = [WEB_SEARCH_TOOL, FETCH_TOOL, SAVE_TOOL]


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """
    Autonomous research agent powered by Groq function-calling.

    Usage
    -----
    >>> agent = ResearchAgent()
    >>> path = agent.research("transformer architecture in LLMs")
    >>> print(path)   # e.g.  Saved to ./reports/transformer_architecture.md
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile", max_iterations: int = 10):
        self.client         = Groq()
        self.model          = model
        self.max_iterations = max_iterations

        # Short-term memory
        self.buffer = ConversationBuffer(max_tokens=4000)

        # Long-term memory (ChromaDB)
        self.long_memory = LongTermMemory()
        self.session_id  = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def research(self, topic: str) -> str:
        """
        Research *topic* and return the path to the saved report.
        Injects relevant past sessions from long-term memory into the system prompt.
        """
        print(f"\n{'='*60}")
        print(f"🔍 Researching: {topic}")
        print(f"{'='*60}\n")

        # --- Long-term memory retrieval ---
        past_sessions = self.long_memory.retrieve(topic, top_k=3)

        memory_block = ""
        if past_sessions:
            print(f"📚 Retrieved {len(past_sessions)} past session(s) from long-term memory.")
            memory_block = "\n\nFrom past research, you already know:\n"
            for i, ctx in enumerate(past_sessions, 1):
                # Show first 300 chars of each past session to keep it readable
                preview = ctx[:300].replace("\n", " ")
                memory_block += f"{i}. {preview}...\n"
                print(f"  [Memory {i}] {preview[:80]}...")
        else:
            print("📭 No relevant long-term memory found — starting fresh.")

        system_content = SYSTEM_PROMPT + memory_block

        # --- Build initial messages list ---
        # This list is the SINGLE SOURCE OF TRUTH throughout _run_loop.
        messages = [
            {"role": "system",  "content": system_content},
            {"role": "user",    "content": topic},
        ]

        # Track in short-term buffer as well
        self.buffer.add_message("user", topic)

        # --- Run the tool-calling loop ---
        result = asyncio.run(self._run_loop(messages, topic))

        # --- Persist this session to long-term memory ---
        self.long_memory.store(
            session_id=self.session_id,
            query=topic,
            response=str(result),
        )

        return result

    # ------------------------------------------------------------------
    # Async tool-calling loop
    # ------------------------------------------------------------------

    async def _run_loop(self, messages: list[dict], topic: str) -> str:
        """
        Core agentic loop.

        KEY DESIGN:
        -----------
        - `messages` is passed in and mutated in-place every iteration.
        - Tool results (role="tool") are appended to `messages` immediately,
          so the LLM always sees the full, correct conversation history.
        - fetch_and_summarise calls are parallelised with asyncio.gather().
        - The loop exits when the LLM produces no tool_calls (final answer)
          or when max_iterations is reached.
        """
        iteration = 0

        # Max chars kept per tool result before appending to messages.
        # Keeps the context window manageable and prevents the 400
        # 'tool_use_failed' error that occurs when Groq receives a prompt
        # that is too long for the model to handle structured tool calls.
        TOOL_RESULT_MAX_CHARS = 800

        # Max parallel fetch_and_summarise calls per iteration.
        # Running 6+ in parallel produces a huge context spike → 400 error.
        MAX_PARALLEL_FETCH = 2

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Optionally compress old history every 4 iterations
            if iteration % 4 == 0:
                self._maybe_summarise_buffer(messages)

            # --- Call Groq (with retry on context-overflow 400) ---
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=ALL_TOOLS,
                    tool_choice="auto",
                )
            except BadRequestError as e:
                print(f"  ⚠️  Groq 400 error (context too large / bad format). Truncating and retrying...")
                print(f"      Detail: {str(e)[:200]}")
                # Aggressively truncate: keep system msg + last 6 messages only
                sys_msgs   = [m for m in messages if m["role"] == "system"]
                other_msgs = [m for m in messages if m["role"] != "system"]
                messages[:] = sys_msgs + other_msgs[-6:]
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=ALL_TOOLS,
                        tool_choice="auto",
                    )
                except BadRequestError as e2:
                    print(f"  ❌  Retry also failed: {e2}")
                    path = save_report(
                        title=f"Research Report (error): {topic}",
                        content=f"Agent stopped due to API error: {e2}",
                        filename="error_" + topic.lower().replace(" ", "_")[:40],
                    )
                    return path

            msg = response.choices[0].message

            # Build the assistant message dict (content may be None when tool_calls present)
            assistant_msg: dict = {
                "role":    "assistant",
                "content": msg.content or "",
            }

            # Attach tool_calls if present (REQUIRED by Groq protocol)
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id":   tc.id,
                        "type": "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            # Append the assistant message BEFORE any tool results
            messages.append(assistant_msg)

            # --- No tool calls → agent is done ---
            if not msg.tool_calls:
                print("✅ Agent finished (no tool calls). Saving fallback report...")
                final_content = msg.content or "No content generated."
                path = save_report(
                    title=f"Research Report: {topic}",
                    content=final_content,
                    filename=topic.lower().replace(" ", "_")[:50],
                )
                return path

            # --- Process tool calls ---
            # Separate async (fetch) from sync (search, save) calls
            sync_calls:  list[tuple] = []   # (tool_call_object, name, args)
            async_calls: list[tuple] = []   # (tool_call_object, name, args)

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                print(f"  🔧 Tool call: {name}({args})")

                if name == "fetch_and_summarise":
                    async_calls.append((tc, name, args))
                else:
                    sync_calls.append((tc, name, args))

            # Execute sync tools
            for tc, name, args in sync_calls:
                if name == "web_search":
                    result = web_search(**args)
                    full_str = json.dumps(result, ensure_ascii=False)
                    # Truncate search results: keep top entries up to char limit
                    result_str = full_str[:TOOL_RESULT_MAX_CHARS]
                    if len(full_str) > TOOL_RESULT_MAX_CHARS:
                        result_str += "... [truncated]"
                elif name == "save_report":
                    result_str = save_report(**args)
                    # save_report signals completion — append tool result and exit
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result_str,
                    })
                    print(f"  💾 Report saved: {result_str}")
                    return result_str
                else:
                    result_str = f"Unknown tool: {name}"

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_str,
                })

            # Execute async tools in PARALLEL — capped at MAX_PARALLEL_FETCH
            # Running too many in parallel floods the context window in one shot.
            if async_calls:
                # Process in batches to keep context size predictable
                for batch_start in range(0, len(async_calls), MAX_PARALLEL_FETCH):
                    batch = async_calls[batch_start : batch_start + MAX_PARALLEL_FETCH]
                    coroutines = [fetch_and_summarise(**args) for _, _, args in batch]
                    print(f"  ⚡ Running {len(coroutines)} fetch_and_summarise call(s) in parallel (batch {batch_start // MAX_PARALLEL_FETCH + 1})...")
                    results = await asyncio.gather(*coroutines, return_exceptions=True)

                    for (tc, _, _), res in zip(batch, results):
                        if isinstance(res, Exception):
                            res = f"Error: {res}"
                        # Truncate summary to keep context window manageable
                        res_str = str(res)[:TOOL_RESULT_MAX_CHARS]
                        if len(str(res)) > TOOL_RESULT_MAX_CHARS:
                            res_str += "... [truncated]"
                        messages.append({
                            "role":         "tool",
                            "tool_call_id": tc.id,
                            "content":      res_str,
                        })

        # Max iterations reached
        print("⚠️  Max iterations reached. Saving partial report...")
        path = save_report(
            title=f"Research Report (partial): {topic}",
            content="Max iterations reached. Partial data collected.",
            filename="partial_" + topic.lower().replace(" ", "_")[:40],
        )
        return path

    # ------------------------------------------------------------------
    # Short-term memory helper
    # ------------------------------------------------------------------

    def _maybe_summarise_buffer(self, messages: list[dict]):
        """
        Compress early conversation turns to save context-window space.
        Replaces the oldest user/assistant messages with a summary injected
        after the system prompt.

        Tradeoff:
        - Keeping full history → accurate context, but burns tokens fast.
        - Summarising → cheaper, but may lose specific URL/fact details.
        """
        # Identify non-system messages
        sys_msgs    = [m for m in messages if m["role"] == "system"]
        other_msgs  = [m for m in messages if m["role"] != "system"]

        if len(other_msgs) < 6:
            return  # Not enough to worth summarising yet

        # Summarise oldest 4 turns (user/assistant only — skip tool)
        to_compress = [m for m in other_msgs[:6] if m["role"] in ("user", "assistant")]
        remaining   = other_msgs[6:]

        if not to_compress:
            return

        summary_lines = ["[Compressed history]"]
        for m in to_compress:
            role    = m["role"]
            snippet = str(m.get("content", ""))[:120].replace("\n", " ")
            summary_lines.append(f"{role}: {snippet}...")

        summary_msg = {
            "role":    "system",
            "content": "\n".join(summary_lines),
        }

        # Rebuild messages: system(s) → summary → remaining
        messages.clear()
        messages.extend(sys_msgs)
        messages.append(summary_msg)
        messages.extend(remaining)

        print("  📝 Compressed early history into summary.")