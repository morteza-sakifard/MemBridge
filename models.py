from typing import List, Optional

from pydantic import BaseModel


class Turn(BaseModel):
    """Represents a single turn in a conversation."""
    turn_id: int
    role: str
    content: str

class Conversation(BaseModel):
    """Represents a full conversation with multiple turns and ground truth."""
    conversation_id: str
    turns: List[Turn]
    ground_truth: dict

class Memory(BaseModel):
    """Represents the final, structured memory object to be stored."""
    memory_id: str
    content: str
    conversation_id: str
    turn_id: int
    confidence: float
    timestamp: str
    previous_memory_id: Optional[str] = None