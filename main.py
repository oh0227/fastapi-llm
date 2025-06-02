from fastapi import FastAPI, HTTPException
import uvicorn
import nest_asyncio
from schemas import (
    AnalyzeResponse,
    AnalyzeMessageBase,
    EmbeddingRequest,
    PreferenceUpdateByMessageRequest
)
from services.pipeline import (
    process_message_pipeline,
    create_embedding,
    update_preference_by_message
)

import os

app = FastAPI()
nest_asyncio.apply()

@app.post("/analyze_and_filter", response_model=AnalyzeResponse)
async def analyze_message(message: AnalyzeMessageBase):
    try:
        return process_message_pipeline(message)
    except Exception as e:
        # 기본 실패 응답 구성
        return AnalyzeResponse(
            content=message.content,
            clarified=message.content,
            summary="요약 실패",
            category="others",
            embedding_vector=[],
            recommended=False
        )

@app.post("/preference/create")
def create_embedding_endpoint(req: EmbeddingRequest):
    try:
        return create_embedding(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.post("/preference/update_by_message_id")
def update_preference_endpoint(req: PreferenceUpdateByMessageRequest):
    try:
        return update_preference_by_message(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)