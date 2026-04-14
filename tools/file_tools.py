"""
tools/file_tools.py
-------------------
Tool 3 — save_report

JSON schema (for reference / documentation):
{
  "type": "function",
  "function": {
    "name": "save_report",
    "description": "Save a research report as a Markdown file under ./reports/.",
    "parameters": {
      "type": "object",
      "properties": {
        "title":    {"type": "string", "description": "Report title"},
        "content":  {"type": "string", "description": "Full Markdown body"},
        "filename": {"type": "string", "description": "File name without extension"}
      },
      "required": ["title", "content", "filename"]
    }
  }
}
"""

import os
from datetime import datetime


def save_report(title: str, content: str, filename: str) -> str:
    """
    Save *content* (Markdown) as ./reports/{filename}.md.

    - Creates the reports/ directory if it doesn't exist.
    - Prepends an H1 heading and a timestamp.
    - Returns the relative path string on success.
    """
    os.makedirs("reports", exist_ok=True)

    # Sanitise filename — keep alphanumeric, hyphens, underscores
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in filename
    ).strip("_") or "report"

    path = f"reports/{safe_name}.md"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"*Generated: {timestamp}*\n\n")
        f.write(content)

    return f"Saved to ./{path}"