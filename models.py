from typing import List, Optional

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """Represents a single turn in a conversation."""
    turn_id: int
    role: str
    content: str


class Conversation(BaseModel):
    """Represents a full conversation with multiple turns and ground truth."""
    conversation_id: int
    turns: List[Turn]


class Memory(BaseModel):
    """Represents the final, structured memory object to be stored."""
    memory_id: int
    content: str
    conversation_id: int
    turn_id: int
    confidence: float
    timestamp: str
    previous_memory_id: Optional[int] = None
    vector: Optional[List[float]] = None

class Evaluation(BaseModel):
    """Holds the LLM judge's assessment of a single memory."""
    is_correct: bool = Field(..., description="Is the memory a factually correct statement based on the conversation?")
    is_relevant: bool = Field(..., description="Is the memory a meaningful piece of information worth storing?")
    is_atomic: bool = Field(..., description="Does the memory represent a single, distinct fact?")
    score: int = Field(..., ge=1, le=5, description="An overall quality score from 1 (poor) to 5 (excellent).")
    justification: str = Field(..., description="A brief explanation for the score and assessment.")


class EvaluationResult(BaseModel):
    """Wraps a memory and its corresponding evaluation."""
    memory_id: int
    conversation_id: int
    turn_id: int
    memory_content: str
    evaluation: Evaluation