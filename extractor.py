import os
import json
import uuid
from datetime import datetime, timezone
from typing import List

import openai
from dotenv import load_dotenv
from models import Conversation, Memory, Fact
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
INPUT_DATA_PATH = "synthetic_data.json"

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


def format_conversation_for_prompt(turns: List[dict]) -> str:
    """Formats the conversation history into a string for the LLM prompt."""
    return json.dumps([{"role": t.role, "content": t.content} for t in turns], indent=2)


def extract_memories_from_turn(conversation_history: List[dict]) -> List[Fact]:
    """
    Calls the LLM API to extract facts from the latest turn of a conversation.

    Args:
        conversation_history: A list of turn objects representing the conversation so far.

    Returns:
        A list of Fact objects, or an empty list if none are found or an error occurs.
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

        # Safely parse the JSON response from the model
        response_data = json.loads(response_content)
        raw_facts = response_data.get("facts", [])

        # Validate each fact using the Pydantic model
        validated_facts = [Fact(**fact) for fact in raw_facts]
        return validated_facts

    except (json.JSONDecodeError, openai.APIError, Exception) as e:
        print(f"[ERROR] Could not process turn. Reason: {e}")
        return []


def main():
    store = MemoryStore()

    print("Starting memory extraction process...")

    # Load conversations from the input file
    try:
        with open(INPUT_DATA_PATH, 'r') as f:
            conversations_data = json.load(f)
        conversations = [Conversation(**conv) for conv in conversations_data]
        print(f"Loaded {len(conversations)} conversations from '{INPUT_DATA_PATH}'.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[FATAL] Could not load or parse input data file. Error: {e}")
        return

    # Process each conversation
    for conv in conversations:
        print(f"\n=========================================")
        print(f"Processing Conversation ID: {conv.conversation_id}")
        print(f"=========================================")

        turn_history = []
        for i, turn in enumerate(conv.turns, 1):
            turn_history.append(turn)

            # Extract facts from the current state of the conversation
            extracted_facts = extract_memories_from_turn(turn_history)

            for fact in extracted_facts:
                turn_identifier = f"conv_{conv.conversation_id}_turn_{i}"
                memory = Memory(
                    fact_id=f"mem_{uuid.uuid4()}",
                    content=fact.content,
                    extracted_from=turn_identifier,
                    confidence=fact.confidence,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    previous_value=fact.previous_value
                )

                store.write(memory)
                print(f"+++ Stored new memory: {memory.content}")

    print("-----------------------------------------")


if __name__ == "__main__":
    main()
