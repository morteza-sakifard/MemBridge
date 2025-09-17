import chromadb
from typing import List, Optional, Dict

from models import Memory


class VectorStore:
    """
    A client for interacting with a local ChromaDB vector database,
    acting as the primary store for Memory objects.
    """

    def __init__(self, db_path: str, collection_name: str):
        """
        Initializes the ChromaDB client and gets or creates a collection.

        Args:
            db_path: The directory path for the persistent database.
            collection_name: The name of the collection to use.
        """
        self.db_path = db_path
        self.collection_name = collection_name

        print(f"Initializing ChromaDB with local path: '{self.db_path}'")
        self.client = chromadb.PersistentClient(path=self.db_path)

        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print(f"ChromaDB collection '{self.collection_name}' is ready.")

    def insert(self, memory: Memory):
        """
        Inserts a full memory object into the collection.

        Args:
            memory: The Memory Pydantic model instance.
        """
        if memory.vector is None:
            print(f"[ERROR] Cannot insert memory_id {memory.memory_id} without a vector.")
            return

        memory_id_str = str(memory.memory_id)

        metadata = {
            k: v for k, v in memory.model_dump(exclude={'vector'}).items() if v is not None
        }

        try:
            self.collection.add(
                ids=[memory_id_str],
                embeddings=[memory.vector],
                metadatas=[metadata]
            )
            print(f"Successfully inserted memory_id '{memory.memory_id}' into ChromaDB.")
        except Exception as e:
            print(f"[ERROR] Failed to insert data into ChromaDB for memory_id {memory.memory_id}: {e}")

    def get(self, memory_id: int) -> Optional[Memory]:
        """
        Retrieves a single memory from ChromaDB by its ID.

        Args:
            memory_id: The ID of the memory to retrieve.

        Returns:
            The reconstructed Memory Pydantic model instance, or None if not found.
        """
        result = self.collection.get(ids=[str(memory_id)], include=["metadatas", "embeddings"])

        if not result or not result['ids']:
            return None

        metadata = result['metadatas'][0]
        vector = result['embeddings'][0]

        memory_data = {**metadata, "vector": vector}
        return Memory(**memory_data)

    def get_all(self) -> List[Memory]:
        """
        Retrieves all memories from the collection.

        Note: This can be memory-intensive for very large collections.
        For production, consider pagination or streaming.
        """
        count = self.collection.count()
        if count == 0:
            return []

        result = self.collection.get(limit=count, include=["metadatas", "embeddings"])

        memories: List[Memory] = []
        if result['metadatas'] and result['embeddings']:
            for metadata, vector in zip(result['metadatas'], result['embeddings']):
                try:
                    memory_data = {**metadata, "vector": vector}
                    memories.append(Memory(**memory_data))
                except Exception as e:
                    print(f"[WARNING] Could not reconstruct memory from metadata: {metadata}. Error: {e}")
        return memories

    def search(self, query_vector: List[float], top_k: int) -> Optional[Dict]:
        """
        Performs a similarity search in the collection.

        Args:
            query_vector: The embedding vector of the query.
            top_k: The number of top results to return.

        Returns:
            A dictionary containing the search results from ChromaDB,
            including ids, distances, metadatas, and embeddings.
            Returns None if no results are found or an error occurs.
        """
        if not query_vector:
            print("[WARNING] Query vector is empty. Cannot perform search.")
            return None
        try:
            num_items = self.collection.count()
            if top_k > num_items:
                print(f"[INFO] Requested top_k={top_k} is larger than collection size={num_items}. "
                      f"Returning {num_items} items instead.")
                top_k = num_items

            if top_k == 0:
                return None

            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=["metadatas", "distances", "embeddings"]
            )
            return results
        except Exception as e:
            print(f"[ERROR] Failed to perform search in ChromaDB: {e}")
            return None
