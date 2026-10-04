"""
answerkit.prompts
~~~~~~~~~~~~~~~~~

System prompts, templates, and message formatting helpers for AnswerKit.
"""

from typing import Any, Dict, List, Optional, Union

from answerkit.models import ChatMessage, SearchResult

DEFAULT_SYSTEM_PROMPT = (
    "You are AnswerKit, an intelligent, helpful, and concise AI assistant. "
    "You provide clear, well-structured, and accurate responses. "
    "When live search results or additional context are provided, "
    "incorporate them seamlessly to deliver fresh, verified answers."
)

POLISH_SYSTEM_PROMPT = (
    "You are an expert prompt engineer. Your job is to transform vague, brief, or draft user prompts "
    "into highly effective, structured, and context-rich prompts for advanced LLMs.\n\n"
    "Rules:\n"
    "1. Clarify the objective, structure, constraints, edge cases, and best practices.\n"
    "2. Preserve the user's core intent and subject matter.\n"
    "3. Return ONLY the polished prompt text. Do NOT add preamble like 'Here is your prompt:' or conversational filler.\n"
    "4. Keep it concise, punchy, and actionable (1 to 3 sentences maximum)."
)


def format_search_context(search_results: List[SearchResult]) -> str:
    """Formats structured search results into a clean context prompt for the LLM."""
    if not search_results:
        return ""

    formatted_items = []
    for idx, item in enumerate(search_results, start=1):
        formatted_items.append(
            f"[{idx}] Title: {item.title}\n"
            f"    URL: {item.link}\n"
            f"    Snippet: {item.snippet}"
        )

    joined_results = "\n\n".join(formatted_items)
    return (
        f"[Live Google Search Context]:\n"
        f"{joined_results}\n\n"
        f"Instructions: Use the above live web search context to provide an accurate, "
        f"up-to-date answer. If referencing specific sources, mention relevant titles or links."
    )


def build_chat_messages(
    prompt: str,
    system_prompt: Optional[str] = None,
    conversation_history: Optional[List[Union[Dict[str, Any], ChatMessage]]] = None,
    search_context: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Constructs the standard message array for LLM chat completion APIs.
    """
    messages: List[Dict[str, str]] = []

    # 1. System Prompt
    sys_content = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    if sys_content.strip():
        messages.append({"role": "system", "content": sys_content.strip()})

    # 2. Conversation History
    if conversation_history:
        for item in conversation_history:
            if isinstance(item, ChatMessage):
                messages.append(item.to_dict())
            elif isinstance(item, dict) and "role" in item and "content" in item:
                messages.append({
                    "role": str(item["role"]),
                    "content": str(item["content"]),
                })

    # 3. Web Search Context (if any)
    if search_context and search_context.strip():
        messages.append({
            "role": "system",
            "content": search_context.strip(),
        })

    # 4. Current User Prompt
    messages.append({"role": "user", "content": prompt.strip()})

    return messages
