import json
import os
from typing import List, Dict, Any, Optional

from models import Memory


class JSONMemoryStore:
    def __init__(self, file_path: str = "memory_store.json"):
        self.file_path = file_path
        self._memories: Dict[str, Memory] = self._load()

    def _load(self) -> Dict[str, Memory]:
        """Loads memories from the JSON file if it exists."""
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            # The data is stored as a list of dicts, convert it to a dict of Pydantic models
            return {mem['fact_id']: Memory(**mem) for mem in data}
        except (json.JSONDecodeError, IOError):
            return {}

    def _save(self):
        """Saves the current state of memories to the JSON file."""
        with open(self.file_path, 'w') as f:
            # Convert Pydantic models to dicts before saving
            json.dump([mem.model_dump() for mem in self._memories.values()], f, indent=2)

    def write(self, memory: Memory):
        """
        Adds a new memory to the store.

        Args:
            memory: The Memory object to add.
        """
        if memory.fact_id in self._memories:
            print(f"[Warning] Memory with fact_id {memory.fact_id} already exists. Use update instead.")
            return
        self._memories[memory.fact_id] = memory
        self._save()
        print(f"Memory '{memory.fact_id}' written to JSON store.")

    def read(self, fact_id: str) -> Optional[Memory]:
        """
        Reads a single memory by its ID.

        Args:
            fact_id: The ID of the fact to retrieve.

        Returns:
            The Memory object if found, otherwise None.
        """
        return self._memories.get(fact_id)

    def update(self, fact_id: str, updates: Dict[str, Any]):
        """
        Updates an existing memory with new data.

        Args:
            fact_id: The ID of the memory to update.
            updates: A dictionary of fields to update.
        """
        if fact_id not in self._memories:
            print(f"[Error] Memory with fact_id {fact_id} not found.")
            return

        existing_memory = self._memories[fact_id]
        updated_memory = existing_memory.model_copy(update=updates)
        self._memories[fact_id] = updated_memory

        self._save()
        print(f"Memory '{fact_id}' updated in JSON store.")

    def get_all_memories(self) -> List[Memory]:
        """Returns all memories currently in the store."""
        return list(self._memories.values())
