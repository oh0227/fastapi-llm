from fastapi import FastAPI, HTTPException
import uvicorn
import nest_asyncio
import requests
from schemas import AnalyzeResponse, AnalyzeMessageBase, EmbeddingRequest, PreferenceUpdateByMessageRequest
from services.pipeline import process_message_pipeline, bert
from services.db import fetch_message_embedding, fetch_user_preference_vector
import os

API_SERVER_URL = os.getenv("API_SERVER_URL")

# --- FastAPI 초기화 ---
app = FastAPI()
nest_asyncio.apply()


# --- API 엔드포인트 ---
@app.post("/analyze_and_filter", response_model=AnalyzeResponse)
async def analyze_message(message: AnalyzeMessageBase):
    try:
        # process_message_pipeline이 직접 AnalyzeResponse 반환하도록 수정
        return process_message_pipeline(message)
    except Exception as e:
        return {
            "error": str(e),
            "result": {
                "content": message.content,
                "clarified": message.content,
                "summary": "요약 실패",
                "category": "others",
                "embedding_vector": [],
                "recommended": False
            }
        }

# --- 사용자 벡터 업데이트 ---
@app.post("/preference/create")
def create_embedding(req: EmbeddingRequest):
    try:
        print(req.text)
        embedding = bert.encode(req.text, normalize_embeddings=True)
        return {
            "embedding": embedding.tolist(),
            "dim": len(embedding)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@app.post("/preference/update_by_message_id")
def update_preference_by_message(req: PreferenceUpdateByMessageRequest):
    message_vec = fetch_message_embedding(req.message_id)
    if not message_vec:
        raise HTTPException(status_code=404, detail="Message embedding not found")

    user_vec = fetch_user_preference_vector(req.cochat_id)
    if not user_vec:
        return {
            "status": "no existing vector, returning message vector",
            "updated_vector": message_vec,
            "vector_length": len(message_vec)
        }

    updated = [(u + m) / 2 for u, m in zip(user_vec, message_vec)]
    return {
        "status": "vector calculated (not saved)",
        "updated_vector": updated,
        "vector_length": len(updated)
    }