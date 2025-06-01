# api/endpoints.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from services.pipeline import process_message_pipeline, create_embedding, update_preference_by_message

router = APIRouter()

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

class PreferenceRequest(BaseModel):
    text: str

class PreferenceUpdateByMessageRequest(BaseModel):
    cochat_id: str
    message_id: int

@router.post("/analyze_and_filter", response_model=AnalyzeResponse)
async def analyze_message(message: AnalyzeMessageBase):
    return process_message_pipeline(message)

@router.post("/preference/create")
def create_embedding_endpoint(req: PreferenceRequest):
    return create_embedding(req)

@router.post("/preference/update_by_message_id")
def update_preference_endpoint(req: PreferenceUpdateByMessageRequest):
    return update_preference_by_message(req)