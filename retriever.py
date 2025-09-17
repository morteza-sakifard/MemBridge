import os
from typing import List, NamedTuple

import openai
from dotenv import load_dotenv

from embedding import get_embedding
from models import Memory
from vector_store import VectorStore

load_dotenv()

MEMORY_DB_PATH = "memory_store"
COLLECTION_NAME = "memories"


class RetrievalResult(NamedTuple):
    """A structured result for a single retrieved memory."""
    memory: Memory
    score: float


class Retriever:
    """
    Handles embedding-based retrieval of memories from a vector store.
    """

    def __init__(self, vector_store: VectorStore, openai_client: openai.OpenAI):
        """
        Initializes the Retriever.

        Args:
            vector_store: An initialized instance of VectorStore.
            openai_client: An initialized OpenAI client for generating embeddings.
        """
        self.vector_store = vector_store
        self.client = openai_client
        print("Retriever initialized.")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Embeds a query and retrieves the top-k most relevant memories.

        Args:
            query: The user's query string.
            top_k: The number of memories to retrieve.

        Returns:
            A list of RetrievalResult tuples, sorted by relevance (score).
            Each tuple contains the Memory object and its score.
            Returns an empty list if no relevant memories are found or an error occurs.
        """
        print(f"\n---> Retrieving top {top_k} memories for query: '{query}'")

        query_embedding = get_embedding(query, client=self.client)
        if not query_embedding:
            print("[ERROR] Could not generate embedding for the query. Aborting retrieval.")
            return []

        search_results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k
        )

        if not search_results or not search_results.get('ids') or not search_results['ids'][0]:
            print("<--- No relevant memories found.")
            return []

        retrieved_memories: List[RetrievalResult] = []

        ids = search_results['ids'][0]
        distances = search_results['distances'][0]
        metadatas = search_results['metadatas'][0]
        embeddings = search_results['embeddings'][0]

        for i in range(len(ids)):
            try:
                memory_data = {**metadatas[i], "vector": embeddings[i]}
                memory = Memory(**memory_data)

                result = RetrievalResult(memory=memory, score=1-distances[i])
                retrieved_memories.append(result)

            except Exception as e:
                print(f"[WARNING] Could not reconstruct memory for id {ids[i]}. Error: {e}")

        print(f"<--- Found {len(retrieved_memories)} relevant memories.")
        return retrieved_memories


def main():
    """Demonstrates the Retriever functionality."""
    try:
        client = openai.OpenAI(
            api_key=os.environ["LITELLM_API_KEY"],
            base_url=os.environ["LITELLM_API_BASE"],
        )
    except KeyError:
        print("[ERROR] API key or base URL not found. Please check your .env file.")
        return

    try:
        vector_store = VectorStore(
            db_path=MEMORY_DB_PATH,
            collection_name=COLLECTION_NAME
        )
    except Exception as e:
        print(f"Halting execution due to vector store initialization error: {e}")
        return

    retriever = Retriever(vector_store=vector_store, openai_client=client)

    queries = [
        "What is the user's favourite food?",
        "What does Alice like?",
        "Tell me about the user's favourite color."
    ]

    for q in queries:
        results = retriever.retrieve(query=q, top_k=3)

        if results:
            print("-" * 25)
            for result in results:
                print(f"  Score (1-distance): {result.score:.4f}")
                print(f"  Memory ID: {result.memory.memory_id}")
                print(f"  Content: '{result.memory.content}'")
                print(f"  Timestamp: {result.memory.timestamp}")
                print("-" * 15)
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
