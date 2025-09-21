import json
import os
from typing import Dict, List, Optional

import openai
from dotenv import load_dotenv
from pydantic import ValidationError

from models import Conversation, Memory, Evaluation, EvaluationResult
from store import JSONStore

load_dotenv()

try:
    client = openai.OpenAI(
        api_key=os.environ["LITELLM_API_KEY"],
        base_url=os.environ["LITELLM_API_BASE"],
    )
except KeyError:
    raise ConnectionError("API key or base URL not found. Please check your .env file.")

MODEL_NAME = "gemini-2.5-pro"
EVALUATION_OUTPUT_FILE = "evaluation_results.json"

JUDGE_PROMPT_TEMPLATE = """
You are an AI quality assurance expert. Your task is to evaluate the quality of a single 'memory' extracted from a conversation.

Evaluate the memory based on the following criteria:
1.  **Correctness**: Is the memory factually correct based *only* on the provided conversation history?
2.  **Relevance**: Is the fact a meaningful and important piece of information to remember about the user or the world? Trivial or conversational fluff should be considered irrelevant.
3.  **Atomicity**: Does the memory represent a single, distinct fact? (e.g., "User likes green and drives an SUV" is not atomic and should be split).

--- CONTEXT ---

**Full Conversation History:**
{conversation_history}

**Extracted Memory (from Turn {turn_id}):**
"{memory_content}"

--- TASK ---

Provide your evaluation in a valid JSON object. The JSON object must contain a single key "evaluation" with the following structure:
- "is_correct": boolean
- "is_relevant": boolean
- "is_atomic": boolean
- "score": integer (1-5, where 5 is excellent)
- "justification": string (a brief, one-sentence explanation for your score)

**Example Response:**
{{
  "evaluation": {{
    "is_correct": true,
    "is_relevant": true,
    "is_atomic": true,
    "score": 5,
    "justification": "The memory accurately captures a key user preference stated directly in the conversation."
  }}
}}

YOUR RESPONSE:
"""


def format_conversation_for_judge(turns: List[Dict]) -> str:
    """Formats the conversation history into a readable string for the LLM prompt."""
    return json.dumps(turns, indent=2)


def get_llm_evaluation(conversation: Conversation, memory: Memory) -> Optional[Evaluation]:
    """
    Calls the LLM API to get a quality evaluation for a single memory.

    Args:
        conversation: The full conversation object.
        memory: The memory object to be evaluated.

    Returns:
        An Evaluation object or None if an error occurs.
    """
    conv_history_json = [{"role": t.role, "content": t.content} for t in conversation.turns]
    conversation_str = format_conversation_for_judge(conv_history_json)

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        conversation_history=conversation_str,
        turn_id=memory.turn_id,
        memory_content=memory.content
    )

    print(f"---> Judging Memory ID {memory.memory_id}: '{memory.content}'")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        response_content = response.choices[0].message.content
        response_data = json.loads(response_content)

        evaluation_data = response_data.get("evaluation")
        if not evaluation_data:
            print(f"[ERROR] LLM response for memory {memory.memory_id} is missing 'evaluation' key.")
            return None

        evaluation = Evaluation(**evaluation_data)
        print(f"<--- Score: {evaluation.score}/5. Justification: {evaluation.justification}")
        return evaluation

    except (json.JSONDecodeError, openai.APIError, ValidationError, Exception) as e:
        print(f"[ERROR] Could not get or parse evaluation for memory {memory.memory_id}. Reason: {e}")
        return None


def main():
    """
    Main function to run the automated evaluation process.
    """
    print("Starting automatic memory extraction evaluation...")

    conv_store = JSONStore[Conversation](
        file_path="conversation_store.json", model_class=Conversation, id_attribute="conversation_id"
    )
    memory_store = JSONStore[Memory](
        file_path="memory_store.json", model_class=Memory, id_attribute="memory_id"
    )

    conversations = {conv.conversation_id: conv for conv in conv_store.get_all()}
    memories = memory_store.get_all()
    memories.sort(key=lambda m: m.memory_id)  # Process in order

    all_results: List[EvaluationResult] = []

    for memory in memories:
        conversation = conversations.get(memory.conversation_id)
        if not conversation:
            print(f"[WARNING] No matching conversation found for Memory ID {memory.memory_id}. Skipping.")
            continue

        evaluation = get_llm_evaluation(conversation, memory)

        if evaluation:
            result = EvaluationResult(
                memory_id=memory.memory_id,
                conversation_id=memory.conversation_id,
                turn_id=memory.turn_id,
                memory_content=memory.content,
                evaluation=evaluation
            )
            all_results.append(result)

        print("-" * 20)

    if all_results:
        with open(EVALUATION_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            results_dict = [res.model_dump() for res in all_results]
            json.dump(results_dict, f, indent=2)
        print(f"\nEvaluation complete. Results saved to '{EVALUATION_OUTPUT_FILE}'")
    else:
        print("\nEvaluation finished, but no results were generated.")


if __name__ == "__main__":
    main()
