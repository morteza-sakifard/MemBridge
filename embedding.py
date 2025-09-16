from typing import List, Optional

import openai

EMBEDDING_MODEL = "gemini-embedding"
EMBEDDING_DIMENSION = 768


def get_embedding(text: str, client: openai.OpenAI) -> Optional[List[float]]:
    """
    Generates an embedding for the given text using an OpenAI-compatible API.

    Args:
        text: The text content to embed.
        client: An initialized OpenAI client instance.

    Returns:
        A list of floats representing the embedding, or None if an error occurs.
    """
    if not text or not isinstance(text, str):
        print("[ERROR] Embedding input must be a non-empty string.")
        return None

    try:
        text = text.replace("\n", " ")

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text],
            extra_body={"drop_params": True}
        )

        embedding = response.data[0].embedding
        return embedding

    except (openai.APIError, Exception) as e:
        print(f"[ERROR] Failed to generate embedding for text: '{text}'. Reason: {e}")
        return None
