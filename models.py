from pydantic import BaseModel, Field
from typing import List, Optional

class Turn(BaseModel):
    """Represents a single turn in a conversation."""
    role: str
    content: str

class Conversation(BaseModel):
    """Represents a full conversation with multiple turns and ground truth."""
    conversation_id: str
    turns: List[Turn]
    ground_truth: dict

class Fact(BaseModel):
    """
    Represents a raw fact as extracted by the LLM.
    """
    content: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    previous_value: Optional[str] = None

class Memory(BaseModel):
    """Represents the final, structured memory object to be stored."""
    fact_id: str
    content: str
    extracted_from: str # e.g., "conv_001_turn_3"
    confidence: float
    timestamp: str
    previous_value: Optional[str] = None
