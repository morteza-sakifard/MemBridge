import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv

from models import Memory, Turn
from store import JSONConversationStore as ConversationStore
from store import JSONMemoryStore as MemoryStore

load_dotenv()

try:
    client = openai.OpenAI(
        api_key=os.environ["LITELLM_API_KEY"],
        base_url=os.environ["LITELLM_API_BASE"],
    )
except KeyError:
    raise ConnectionError("API key or base URL not found. Please check your config.env file.")

MODEL_NAME = "gemini-2.5-pro"

SYSTEM_PROMPT = """
You are an expert AI system designed to extract key facts from conversations. Your task is to identify new pieces of information or updates to existing information and structure them as memories.

You MUST follow these rules:
1.  Analyze the conversation provided, paying close attention to the most recent turn.
2.  Extract facts that represent concrete, declarative statements about the user or the world.
3.  For each extracted fact, provide a confidence score from 0.0 to 1.0.
4.  If a new fact corrects or updates a previous fact, you MUST include the `previous_value`.
5.  Your response MUST be a valid JSON object containing a single key "facts" which is a list of extracted fact objects.
6.  If no new or updated facts are found in the last turn, return an empty list: `{"facts": []}`.

--- EXAMPLE ---
CONVERSATION:
[
  {"role": "user", "content": "My name is Alice and I work at OpenAI"},
  {"role": "assistant", "content": "Nice to meet you, Alice!"},
  {"role": "user", "content": "Actually, I just switched jobs to Anthropic"}
]

YOUR RESPONSE:
{
  "facts": [
    {
      "content": "User works at Anthropic.",
      "confidence": 0.98,
      "previous_value": "User works at OpenAI."
    }
  ]
}
"""


def format_conversation_for_prompt(turns: List[Turn]) -> str:
    """Formats the conversation history into a string for the LLM prompt."""
    return json.dumps([{"role": t.role, "content": t.content} for t in turns], indent=2)


def extract_memories_from_turn(conversation_history: List[Turn]) -> List[Dict[str, Any]]:
    """
    Calls the LLM API to extract facts from the latest turn of a conversation.

    Args:
        conversation_history: A list of turn objects representing the conversation so far.

    Returns:
        A list of fact dictionaries, or an empty list if none are found or an error occurs.
    """
    if not conversation_history:
        return []

    conversation_str = format_conversation_for_prompt(conversation_history)

    print(f"\n---> Analyzing conversation turn {len(conversation_history)}...")
    print(f"Context: {conversation_str}")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
    memory_store = MemoryStore()
    conv_store = ConversationStore()

    print("Starting memory extraction process...")

    # Process each conversation
    for conv_id in conv_store.list_ids():
        print(f"\n=========================================")
        print(f"Processing Conversation ID: {conv_id}")
        print(f"=========================================")

        conv = conv_store.read(conv_id)

        turn_history = []
        for turn in conv.turns:
            turn_history.append(turn)

            extracted_facts = extract_memories_from_turn(turn_history)
            for fact in extracted_facts:
                previous_memory_id = None
                previous_value_text = fact.get("previous_value")

                if previous_value_text:
                    # Search for the memory that this fact is updating
                    all_memories = memory_store.get_all_memories()
                    matching_memories = [
                        mem for mem in all_memories
                        if mem.content == previous_value_text and mem.conversation_id == conv_id
                    ]
                    if matching_memories:
                        matching_memories.sort(key=lambda m: m.timestamp, reverse=True)
                        previous_memory_id = matching_memories[0].memory_id
                        print(f"--- Linked memory update. Previous memory '{previous_memory_id}' found.")
                    else:
                         print(f"--- [Warning] LLM provided 'previous_value', but no matching memory was found for content: '{previous_value_text}'")


                memory = Memory(
                    memory_id=f"mem_{uuid.uuid4()}",
                    content=fact['content'],
                    conversation_id=conv.conversation_id,
                    turn_id=turn.turn_id,
                    confidence=fact['confidence'],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    previous_memory_id=previous_memory_id
                )

                memory_store.write(memory)
                print(f"+++ Stored new memory: {memory.content}")

    print("-----------------------------------------")


if __name__ == "__main__":
    main()