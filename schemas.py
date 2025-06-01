from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class AnalyzeMessageBase(BaseModel):
    sender_id: str
    receiver_id: str
    subject: str
    content: str
    embedding_vector: Optional[List[float]] = None
    preference_vector: Optional[List[float]] = None
    category: Optional[str] = None
    cochat_id: Optional[str] = None


class AnalyzeResponse(BaseModel):
    content: str
    clarified: str
    summary: str
    category: str
    embedding_vector: List[float]
    recommended: bool
    

class EmbeddingRequest(BaseModel):
    text: str


class PreferenceUpdateByMessageRequest(BaseModel):
    cochat_id: str
    message_id: int