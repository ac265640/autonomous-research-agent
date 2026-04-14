"""
main.py
-------
Entry point for the ResearchAgent.

Usage
-----
    python main.py

The script:
  1. Loads environment variables from .env (GROQ_API_KEY must be set there).
  2. Prompts for a research topic.
  3. Runs the agent and prints the saved report path.

Re-running with the SAME topic demonstrates long-term memory injection —
the second run will show "Retrieved N past session(s)" and produce a
richer, non-redundant report.
"""

import os
from dotenv import load_dotenv

# Load .env FIRST — before any Groq imports that read the API key
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Create a .env file with:\n  GROQ_API_KEY=your_key_here"
    )

from agents.research_agent import ResearchAgent


def main():
    agent = ResearchAgent(
        model          = "llama-3.1-8b-instant",
        max_iterations = 10,
    )

    print("\n🤖 ResearchAgent ready.")
    print("   (Run with the SAME topic twice to see long-term memory in action.)\n")

    topic = input("Enter research topic: ").strip()
    if not topic:
        print("No topic entered. Exiting.")
        return

    result = agent.research(topic)

    print(f"\n{'='*60}")
    print(f"✅ Done!  →  {result}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()