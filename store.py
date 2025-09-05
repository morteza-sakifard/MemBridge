import json
import os
from typing import List, Dict, Any, Optional

from pydantic import ValidationError

from models import Memory, Conversation


class JSONMemoryStore:
    """A simple memory store that uses a local JSON file for persistence."""

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


class JSONConversationStore:
    """A simple store for conversations using a key-value format in a JSON file."""

    def __init__(self, file_path: str = "conversation_store.json"):
        """
        Initializes the store.

        Args:
            file_path: The path to the JSON file.
        """
        self.file_path = file_path
        # The internal representation remains a dictionary for efficient O(1) lookups.
        self._conversations: Dict[str, Conversation] = self._load()
        print(f"Store initialized. Loaded {len(self._conversations)} conversations from '{self.file_path}'.")

    def _load(self) -> Dict[str, Conversation]:
        """
        Loads conversations from a JSON file formatted as a list of objects.

        Returns:
            A dictionary mapping conversation_id to Conversation objects.
        """
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # The file is expected to contain a list of conversation dicts
                data_list = json.load(f)
                if not isinstance(data_list, list):
                    print(f"Warning: Data in '{self.file_path}' is not a list. Starting fresh.")
                    return {}

            # Convert the list into a dictionary for fast lookups, validating each item
            conversations = {}
            for conv_data in data_list:
                conversation = Conversation(**conv_data)
                conversations[conversation.conversation_id] = conversation
            return conversations

        except (json.JSONDecodeError, IOError, ValidationError) as e:
            print(f"Error loading or parsing '{self.file_path}': {e}. Starting with an empty store.")
            return {}

    def _save(self):
        """
        Saves all conversations to the JSON file as a list of objects.
        """
        with open(self.file_path, 'w', encoding='utf-8') as f:
            # Convert the dictionary of conversations back into a list for storage.
            conversation_list = [conv.model_dump() for conv in self._conversations.values()]
            json.dump(conversation_list, f, indent=2)

    def write(self, conversation: Conversation):
        """
        Adds a new conversation or overwrites an existing one by its ID.

        Args:
            conversation: The Conversation object to add or update.
        """
        if not isinstance(conversation, Conversation):
            raise TypeError("Can only write Conversation objects to the store.")

        self._conversations[conversation.conversation_id] = conversation
        self._save()
        print(f"Conversation '{conversation.conversation_id}' written to store.")

    def read(self, conversation_id: str) -> Optional[Conversation]:
        """
        Reads a single conversation by its ID.

        Args:
            conversation_id: The ID of the conversation to retrieve.

        Returns:
            The Conversation object if found, otherwise None.
        """
        return self._conversations.get(conversation_id)

    def list_ids(self) -> List[str]:
        """Returns a list of all conversation IDs."""
        return list(self._conversations.keys())
