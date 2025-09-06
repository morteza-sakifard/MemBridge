import json
import os
import re
from datetime import datetime, timezone
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv

from models import Conversation, Memory, Turn
from store import JSONStore

load_dotenv()

try:
    client = openai.OpenAI(
        api_key=os.environ["LITELLM_API_KEY"],
        base_url=os.environ["LITELLM_API_BASE"],
    )
except KeyError:
    raise ConnectionError("API key or base URL not found. Please check your config.env file.")

MODEL_NAME = "gemini-2.5-pro"

SYSTEM_PROMPT_TEMPLATE = """
You are an expert AI system designed to extract key facts from conversations. Your task is to identify **new pieces of information** or **updates to existing information** and structure them as memories.

You have been provided with a list of memories already extracted from this conversation.
**Your primary goal is to avoid redundancy.** Do not extract a fact if it is already present in the existing memories unless the new information contradicts or significantly refines it.

--- EXISTING_MEMORIES ---
{existing_memories}

--- RULES ---
1.  Analyze the **most recent turn** of the conversation in the context of the full conversation history and the existing memories.
2.  Extract facts that represent **new** or **updated** declarative statements about the user or the world.
3.  If a new fact is an update to an existing memory (e.g., "User's favorite color was blue" -> "User's favorite color is now green"), extract the new fact.
4.  For each extracted fact, provide a confidence score from 0.0 to 1.0.
5.  Your response MUST be a valid JSON object containing a single key "facts" which is a list of extracted fact objects.
6.  If no new or updated facts are found in the last turn, return an empty list: {{"facts": []}}.

--- EXAMPLE ---
EXISTING_MEMORIES:
[
  {{"content": "User works at OpenAI."}}
]

CONVERSATION:
[
  {{"role": "user", "content": "My name is Alice and I work at OpenAI"}},
  {{"role": "assistant", "content": "Nice to meet you, Alice!"}},
  {{"role": "user", "content": "Actually, I just switched jobs to Anthropic"}}
]

YOUR RESPONSE:
{{
  "facts": [
    {{
      "content": "User works at Anthropic.",
      "confidence": 0.98
    }}
  ]
}}
"""


def format_conversation_for_prompt(turns: List[Turn]) -> str:
    """Formats the conversation history into a string for the LLM prompt."""
    return json.dumps([{"role": t.role, "content": t.content} for t in turns], indent=2)


# NEW: Helper function for robust, code-based deduplication.
def normalize_for_comparison(text: str) -> str:
    """Normalizes text for basic semantic comparison."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_memories_from_turn(conversation_history: List[Turn], existing_memories: List[Memory]) -> List[Dict[str, Any]]:
    """
    Calls the LLM API to extract facts from the latest turn of a conversation,
    avoiding facts that already exist.

    Args:
        conversation_history: A list of turn objects representing the conversation so far.
        existing_memories: A list of memory objects already extracted for this conversation.

    Returns:
        A list of fact dictionaries, or an empty list if none are found or an error occurs.
    """
    if not conversation_history:
        return []

    if existing_memories:
        memories_str = json.dumps([{"content": mem.content} for mem in existing_memories], indent=2)
    else:
        memories_str = "[]"

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(existing_memories=memories_str)
    conversation_str = format_conversation_for_prompt(conversation_history)

    print(f"\n---> Analyzing conversation turn {len(conversation_history)}...")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONVERSATION:\n{conversation_str}\n\nYOUR RESPONSE:"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        response_content = response.choices[0].message.content
        print(f"<--- LLM Response: {response_content}")

        response_data = json.loads(response_content)
        extracted_facts = response_data.get("facts", [])
        return extracted_facts

    except (json.JSONDecodeError, openai.APIError, Exception) as e:
        print(f"[ERROR] Could not process turn. Reason: {e}")
        return []


def main():
    memory_store = JSONStore[Memory](
        file_path="memory_store.json", model_class=Memory, id_attribute="memory_id"
    )
    conv_store = JSONStore[Conversation](
        file_path="conversation_store.json", model_class=Conversation, id_attribute="conversation_id"
    )

    print("Starting memory extraction process...")
    all_mem_ids = [mem.memory_id for mem in memory_store.get_all()]
    next_memory_id = max(all_mem_ids) + 1 if all_mem_ids else 1

    for conv_id in conv_store.list_ids():
        print(f"\n=========================================")
        print(f"Processing Conversation ID: {conv_id}")
        print(f"=========================================")

        conv = conv_store.read(conv_id)
        if not conv:
            continue

        memories_for_this_conv = [mem for mem in memory_store.get_all() if mem.conversation_id == conv_id]
        memories_for_this_conv.sort(key=lambda m: m.timestamp) # Ensure chronological order

        turn_history = []
        for turn in conv.turns:
            turn_history.append(turn)

            facts = extract_memories_from_turn(turn_history, memories_for_this_conv)

            for fact in facts:
                is_redundant = False
                normalized_new_fact = normalize_for_comparison(fact['content'])

                for existing_mem in memories_for_this_conv:
                    normalized_existing = normalize_for_comparison(existing_mem.content)
                    if normalized_new_fact == normalized_existing:
                        is_redundant = True
                        print(f"--- Skipping redundant memory: '{fact['content']}' (similar to mem_id {existing_mem.memory_id})")
                        break

                if is_redundant:
                    continue

                previous_memory_id = memories_for_this_conv[-1].memory_id if memories_for_this_conv else None

                memory = Memory(
                    memory_id=next_memory_id,
                    content=fact['content'],
                    conversation_id=conv.conversation_id,
                    turn_id=turn.turn_id,
                    confidence=fact['confidence'],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    previous_memory_id=previous_memory_id
                )

                memory_store.write(memory)
                memories_for_this_conv.append(memory)
                print(f"+++ Stored new memory: {memory.content}")
                next_memory_id += 1

    print("\n-----------------------------------------")
    print("Memory extraction process finished.")


if __name__ == "__main__":
    main()
