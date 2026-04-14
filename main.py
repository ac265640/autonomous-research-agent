from agents.research_agent import ResearchAgent

if __name__ == "__main__":
    agent = ResearchAgent()

    topic = input("Enter research topic: ")
    result = agent.research(topic)

    print("\nFinal Output:\n", result)