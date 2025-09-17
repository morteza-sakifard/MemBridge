import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated as an API.*")

import json
import os
import re
from datetime import datetime, timezone
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv

from embedding import get_embedding
from models import Conversation, Memory, Turn
from store import JSONStore
from vector_store import VectorStore

load_dotenv()

try:
    client = openai.OpenAI(
        api_key=os.environ["LITELLM_API_KEY"],
        base_url=os.environ["LITELLM_API_BASE"],
    )
except KeyError:
    raise ConnectionError("API key or base URL not found. Please check your .env file.")

MODEL_NAME = "gemini-2.5-pro"

MEMORY_DB_PATH = "memory_store"
COLLECTION_NAME = "memories"

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


def normalize_for_comparison(text: str) -> str:
    """Normalizes text for basic semantic comparison."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_memories_from_turn(conversation_history: List[Turn], existing_memories: List[Memory]) -> List[
    Dict[str, Any]]:
    """
    Calls the LLM API to extract facts from the latest turn of a conversation,
    avoiding facts that already exist.
    """
    if not conversation_history:
        return []

    memories_str = json.dumps([{"content": mem.content} for mem in existing_memories],
                              indent=2) if existing_memories else "[]"
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
        return response_data.get("facts", [])
    except (json.JSONDecodeError, openai.APIError, Exception) as e:
        print(f"[ERROR] Could not process turn. Reason: {e}")
        return []


def main():
    conv_store = JSONStore[Conversation](
        file_path="conversation_store.json", model_class=Conversation, id_attribute="conversation_id"
    )

    try:
        vector_store = VectorStore(
            db_path=MEMORY_DB_PATH,
            collection_name=COLLECTION_NAME
        )
    except Exception as e:
        print(f"Halting execution due to vector store initialization error: {e}")
        return

    print("Starting memory extraction process...")
    all_memories = vector_store.get_all()
    all_mem_ids = [mem.memory_id for mem in all_memories]
    next_memory_id = max(all_mem_ids) + 1 if all_mem_ids else 1

    for conv_id in conv_store.list_ids():
        print(f"\n=========================================")
        print(f"Processing Conversation ID: {conv_id}")
        print(f"=========================================")

        conv = conv_store.read(conv_id)
        if not conv:
            continue

        memories_for_this_conv = [mem for mem in all_memories if mem.conversation_id == conv_id]
        memories_for_this_conv.sort(key=lambda m: m.timestamp)

        turn_history = []
        for turn in conv.turns:
            turn_history.append(turn)
            facts = extract_memories_from_turn(turn_history, memories_for_this_conv)

            for fact in facts:
                is_redundant = any(
                    normalize_for_comparison(fact['content']) == normalize_for_comparison(mem.content)
                    for mem in memories_for_this_conv
                )
                if is_redundant:
                    print(f"--- Skipping redundant memory (code-based check): '{fact['content']}'")
                    continue

                previous_memory_id = memories_for_this_conv[-1].memory_id if memories_for_this_conv else None

                print(f"--- Generating embedding for: '{fact['content']}'")
                embedding_vector = get_embedding(fact['content'], client=client)

                if not embedding_vector:
                    print(f"[WARNING] Skipping memory due to embedding failure: '{fact['content']}'")
                    continue

                memory = Memory(
                    memory_id=next_memory_id,
                    content=fact['content'],
                    conversation_id=conv.conversation_id,
                    turn_id=turn.turn_id,
                    confidence=fact['confidence'],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    previous_memory_id=previous_memory_id,
                    vector=embedding_vector
                )

                vector_store.insert(memory=memory)

                memories_for_this_conv.append(memory)
                next_memory_id += 1

    print("\n-----------------------------------------")
    print("Memory extraction process finished.")


if __name__ == "__main__":
    main()