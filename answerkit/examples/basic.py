"""
basic.py
~~~~~~~~

Demonstrates basic, asynchronous question answering with AnswerKit.
"""

import asyncio
from answerkit import AnswerKit


async def main() -> None:
    # Initialize AnswerKit (will read GROQ_API_KEY from environment if not passed explicitly)
    ai = AnswerKit(
        groq_api_key="YOUR_GROQ_API_KEY",  # Or set GROQ_API_KEY environment variable
        model="openai/gpt-oss-20b",
    )

    print("Sending prompt to AnswerKit...")
    answer = await ai.ask("Explain Retrieval Augmented Generation (RAG) in simple terms.")

    print("\n--- Answer ---")
    print(answer)

    # Prompt polishing example
    raw_draft = "tell me how to write tests in python"
    polished = await ai.polish_prompt(raw_draft)
    print(f"\nOriginal: '{raw_draft}'")
    print(f"Polished: '{polished}'")

    await ai.close()


if __name__ == "__main__":
    asyncio.run(main())
