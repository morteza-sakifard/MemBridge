from pymilvus import MilvusClient

from models import Memory


class VectorStore:
    """
    A client for interacting with a local Milvus Lite vector database.
    """

    def __init__(self, db_file: str, collection_name: str, vector_dim: int):
        self.db_file = db_file
        self.collection_name = collection_name
        self.vector_dim = vector_dim

        print(f"Initializing Milvus Lite with local file: '{self.db_file}'")
        self.client = MilvusClient(uri=self.db_file)

        self._setup_collection()

    def _setup_collection(self):
        """Creates the collection if it doesn't exist and ensures an index is present."""
        if not self.client.has_collection(collection_name=self.collection_name):
            print(f"Collection '{self.collection_name}' not found. Creating new collection...")
            self._create_collection()
        else:
            print(f"Found existing collection '{self.collection_name}'.")

        self._create_index_if_not_exists()

    def _create_collection(self):
        """Defines and creates the Milvus collection using the MilvusClient API."""
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.vector_dim,
            primary_field_name="memory_id",
            id_type="int",
            vector_field_name="vector",
            metric_type="L2",
            enable_dynamic_field=True
        )
        print("Collection schema created with dynamic field enabled.")

    def _create_index_if_not_exists(self):
        """Creates a vector index on the collection if one does not already exist."""
        indexes = self.client.list_indexes(collection_name=self.collection_name)
        if not indexes:
            print("No index found. Creating IVF_FLAT index on 'vector' field...")
            index_params = self.client.prepare_index_params(
                field_name="vector",
                index_type="IVF_FLAT",
                metric_type="L2",
                params={"nlist": 128}
            )
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            print("Index created successfully.")
        else:
            print(f"Found existing index: {indexes}")

    def insert(self, memory: Memory):
        """
        Inserts a full memory object into the collection.

        Args:
            memory: The Memory Pydantic model instance.
        """
        data_to_insert = memory.model_dump()

        if data_to_insert.get("vector") is None:
            print(f"[ERROR] Cannot insert memory_id {memory.memory_id} without a vector.")
            return

        try:
            self.client.insert(
                collection_name=self.collection_name,
                data=[data_to_insert]
            )
            print(f"Successfully inserted full memory object for memory_id '{memory.memory_id}' into Milvus Lite.")
        except Exception as e:
            print(f"[ERROR] Failed to insert data into Milvus Lite for memory_id {memory.memory_id}: {e}")
