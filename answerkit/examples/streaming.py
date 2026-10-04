"""
streaming.py
~~~~~~~~~~~~

Demonstrates asynchronous real-time token streaming with AnswerKit.
"""

import asyncio
from answerkit import AnswerKit


async def main() -> None:
    ai = AnswerKit(
        groq_api_key="YOUR_GROQ_API_KEY",  # Or set GROQ_API_KEY environment variable
    )

    prompt = "Explain how JWT authentication works step-by-step."
    print(f"Prompt: {prompt}\nStreaming response:\n")

    # Framework-independent async generator
    async for chunk in ai.stream(prompt):
        print(chunk, end="", flush=True)

    print("\n\n[Stream complete]")
    await ai.close()


if __name__ == "__main__":
    asyncio.run(main())
