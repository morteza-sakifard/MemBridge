import json
import os
from typing import List, Dict, Any, Optional, Type, TypeVar, Generic, Set

from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)


class JSONStore(Generic[T]):
    """A generic, simple store for Pydantic models using a local JSON file."""

    def __init__(self, file_path: str, model_class: Type[T], id_attribute: str, exclude_on_save: Optional[Set[str]] = None):
        """
        Initializes the generic JSON store.

        Args:
            file_path: The path to the JSON file.
            model_class: The Pydantic model class to store (e.g., Memory).
            id_attribute: The name of the ID attribute on the model (e.g., "memory_id").
            exclude_on_save: A set of field names to exclude when saving to JSON.
        """
        self.file_path = file_path
        self.model_class = model_class
        self.id_attribute = id_attribute
        self.exclude_on_save = exclude_on_save or set()
        self._data: Dict[int, T] = self._load()
        print(
            f"Store for '{self.model_class.__name__}' initialized. Loaded {len(self._data)} items from '{self.file_path}'.")

    def _load(self) -> Dict[int, T]:
        """Loads items from the JSON file if it exists."""
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            if not isinstance(data_list, list):
                print(f"[Warning] Data in '{self.file_path}' is not a list. Starting fresh.")
                return {}

            items: Dict[int, T] = {}
            for item_data in data_list:
                try:
                    item = self.model_class(**item_data)
                    item_id = getattr(item, self.id_attribute)
                    items[item_id] = item
                except (ValidationError, AttributeError) as e:
                    print(f"[Warning] Skipping invalid item data: {item_data}. Error: {e}")
            return items
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading or parsing '{self.file_path}': {e}. Starting with an empty store.")
            return {}

    def _save(self):
        """Saves the current state of items to the JSON file."""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump([item.model_dump(exclude=self.exclude_on_save) for item in self._data.values()], f, indent=2)

    def write(self, item: T):
        """
        Adds a new item or overwrites an existing one by its ID.

        Args:
            item: The Pydantic model instance to add or update.
        """
        if not isinstance(item, self.model_class):
            raise TypeError(f"Can only write '{self.model_class.__name__}' objects to this store.")

        item_id = getattr(item, self.id_attribute)
        self._data[item_id] = item
        self._save()
        print(f"{self.model_class.__name__} '{item_id}' written to store.")

    def read(self, item_id: int) -> Optional[T]:
        """
        Reads a single item by its ID.

        Args:
            item_id: The ID of the item to retrieve.

        Returns:
            The Pydantic model instance if found, otherwise None.
        """
        return self._data.get(item_id)

    def update(self, item_id: int, updates: Dict[str, Any]):
        """
        Updates an existing item with new data.

        Args:
            item_id: The ID of the item to update.
            updates: A dictionary of fields to update.
        """
        if item_id not in self._data:
            print(f"[Error] {self.model_class.__name__} with ID {item_id} not found.")
            return

        existing_item = self._data[item_id]
        updated_item = existing_item.model_copy(update=updates)
        self._data[item_id] = updated_item

        self._save()
        print(f"{self.model_class.__name__} '{item_id}' updated in JSON store.")

    def get_all(self) -> List[T]:
        """Returns all items currently in the store."""
        return list(self._data.values())

    def list_ids(self) -> List[int]:
        """Returns a list of all item IDs."""
        return list(self._data.keys())
