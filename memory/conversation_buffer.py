"""
memory/conversation_buffer.py
------------------------------
Short-term in-context memory using a sliding message buffer.

Tradeoffs:
  - More history  → richer context, more accurate, but burns context-window tokens fast
                    and increases API cost every call.
  - Less history  → cheaper, faster, but agent may forget earlier search results.

Two strategies implemented:
  1. Truncation    — drop the OLDEST messages when over the token budget.
  2. Summarisation — compress old messages into a single summary system message
                     (better: preserves key facts in fewer tokens).
"""

from typing import Dict, List, Optional


class ConversationBuffer:
    """
    Lightweight in-memory conversation history with token-aware truncation
    and summarisation.

    Note: This buffer is used for optional history compression only.
    The agent's authoritative `messages` list lives in `_run_loop` and
    includes tool messages; this buffer tracks user/assistant turns only.
    """

    def __init__(self, max_tokens: int = 4000):
        self.messages:   List[Dict] = []
        self.max_tokens: int        = max_tokens

    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a user or assistant turn."""
        self.messages.append({
            "role":    str(role),
            "content": str(content),
        })

    # ------------------------------------------------------------------

    def _estimate_tokens(self, messages: List[Dict]) -> int:
        """Rough token estimate: 1 token ≈ 4 characters (GPT rule of thumb)."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4

    # ------------------------------------------------------------------

    def get_messages(self, max_tokens: Optional[int] = None) -> List[Dict]:
        """
        Strategy 1 — TRUNCATION:
        Return messages that fit within the token budget, dropping the oldest first.

        Only user/assistant/system roles are returned (tool messages are excluded
        because they must stay adjacent to their tool_call in the main messages list).
        """
        budget = max_tokens or self.max_tokens

        safe = [
            m for m in self.messages
            if m.get("role") in ("user", "assistant", "system")
        ]

        # Drop oldest until we're within budget
        while self._estimate_tokens(safe) > budget and safe:
            safe.pop(0)

        return safe

    # ------------------------------------------------------------------

    def summarise_old_messages(self) -> str:
        """
        Strategy 2 — SUMMARISATION:
        When the buffer grows beyond 5 messages, compress the oldest 3 into
        a single summary system message and replace them.

        Returns the summary string (useful for debugging / logging).

        Why better than truncation:
          Truncation silently drops information. Summarisation distils key
          facts (which URLs were fetched, what was found) into a compact form
          that the agent can still refer to.
        """
        if len(self.messages) < 5:
            return ""

        old_msgs = self.messages[:3]
        rest     = self.messages[3:]

        lines = ["Summary of earlier conversation:"]
        for m in old_msgs:
            role    = str(m.get("role", "?"))
            snippet = str(m.get("content", ""))[:150].replace("\n", " ")
            lines.append(f"  [{role}] {snippet}")

        summary = "\n".join(lines)

        # Replace first 3 messages with the compressed summary
        self.messages = [{"role": "system", "content": summary}] + rest

        return summary