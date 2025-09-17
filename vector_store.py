import chromadb

from models import Memory


class VectorStore:
    """
    A client for interacting with a local ChromaDB vector database.
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
            print(f"Successfully inserted vector for memory_id '{memory.memory_id}' into ChromaDB.")
        except Exception as e:
            print(f"[ERROR] Failed to insert data into ChromaDB for memory_id {memory.memory_id}: {e}")
