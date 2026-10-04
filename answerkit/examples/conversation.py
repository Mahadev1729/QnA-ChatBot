"""
conversation.py
~~~~~~~~~~~~~~~

Demonstrates multi-turn conversation memory with AnswerKit without requiring a database.
"""

import asyncio
from answerkit import AnswerKit


async def main() -> None:
    ai = AnswerKit(
        groq_api_key="YOUR_GROQ_API_KEY",
    )

    # Maintain conversation state in standard Python lists/dicts
    history = [
        {"role": "user", "content": "What is RAG in AI?"},
        {
            "role": "assistant",
            "content": "RAG stands for Retrieval-Augmented Generation, combining search with generative LLMs.",
        },
    ]

    follow_up = "Why is it superior to fine-tuning for rapidly changing company documentation?"
    print(f"Follow-up: {follow_up}\n")

    answer = await ai.ask(follow_up, conversation_history=history)
    print("Response with Context:")
    print(answer)

    # Append to history for subsequent turns
    history.append({"role": "user", "content": follow_up})
    history.append({"role": "assistant", "content": answer})

    await ai.close()


if __name__ == "__main__":
    asyncio.run(main())
