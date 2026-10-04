"""
web_search.py
~~~~~~~~~~~~~

Demonstrates live Google Serper web-search-augmented Q&A with AnswerKit.
"""

import asyncio
from answerkit import AnswerKit


async def main() -> None:
    # Initialize with web search enabled and Serper API Key
    ai = AnswerKit(
        groq_api_key="YOUR_GROQ_API_KEY",
        serper_api_key="YOUR_SERPER_API_KEY",
        enable_web_search=True,
    )

    query = "What are the latest AI breakthroughs announced this month?"
    print(f"Query: {query}\n")

    # Ask with rich metadata (sources and search status)
    result = await ai.ask_with_metadata(query)

    print("--- Generated Answer ---")
    print(result.content)
    print("\n--- Search Details ---")
    print(f"Web Search Triggered: {result.used_search}")
    print(f"Model Used: {result.model}")
    print(f"Sources Found: {len(result.sources)}")
    for s in result.sources:
        print(f" - [{s.title}] ({s.link})")

    await ai.close()


if __name__ == "__main__":
    asyncio.run(main())
