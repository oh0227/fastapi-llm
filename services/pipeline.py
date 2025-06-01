# --- Imports ---
import os
import re
import json
import pickle
import psycopg2
from typing import List
from fastapi import HTTPException
from sklearn.metrics.pairwise import cosine_similarity

from config import GPT_MODEL, OPENAI_API_KEY, DB_CONFIG, FAISS_INDEX_PATH, BLOCKED_DOMAINS
from .db import fetch_user_preference_vector, fetch_message_embedding
from .utils import is_recommended
from schemas import AnalyzeResponse

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from duckduckgo_search import DDGS
from newspaper import Article

# --- Model Initialization ---
client = OpenAI(api_key=OPENAI_API_KEY, timeout=15)
bert = SentenceTransformer("all-MiniLM-L6-v2")

# --- FAISS Initialization ---
def init_vector_store():
    store_path = os.path.join(FAISS_INDEX_PATH, "faiss_store.pkl")
    if os.path.exists(store_path):
        with open(store_path, "rb") as f:
            return pickle.load(f)

    dummy_doc = Document(page_content="dummy", metadata={"source": "init"})
    return FAISS.from_documents([dummy_doc], bert)

vector_store = init_vector_store()

def add_docs_to_vectorstore(texts: List[str], source_name: str):
    global vector_store
    new_docs = [Document(page_content=t, metadata={"source": source_name}) for t in texts]
    new_docs = filter_complex_metadata(new_docs)

    if len(vector_store.docstore._dict) == 1 and "init" in str(vector_store.docstore._dict):
        vector_store = FAISS.from_documents(new_docs, bert)
    else:
        vector_store.add_documents(new_docs)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    with open(os.path.join(FAISS_INDEX_PATH, "faiss_store.pkl"), "wb") as f:
        pickle.dump(vector_store, f)

# --- LLM Clarification ---
def clarify_with_llm(message: str) -> dict:
    prompt = f"""
다음 메시지를 더 명확하게 풀어쓰고, 과거 문맥 또는 외부 정보가 필요한지 판단해줘.
형식:
{{
  "clarified": "...",
  "needs_user_context": true/false,
  "db_keywords": [...],
  "needs_external_info": true/false,
  "web_keywords": [...]
}}
메시지: "{message}"
"""
    try:
        res = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(res.choices[0].message.content.strip())
    except Exception as e:
        print("❗ clarify_with_llm 실패:", e)
        return {
            "clarified": message,
            "needs_user_context": False,
            "db_keywords": [],
            "needs_external_info": False,
            "web_keywords": []
        }

def fetch_user_messages_by_keywords(cochat_id: str, keywords: List[str]) -> List[str]:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        keyword_clause = " OR ".join([f"content ILIKE '%{kw}%'" for kw in keywords])
        query = f"""
            SELECT content FROM messages
            JOIN users ON messages.user_id = users.id
            WHERE users.cochat_id = %s AND ({keyword_clause})
            LIMIT 20;
        """
        cur.execute(query, (cochat_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [row[0] for row in rows]
    except Exception as e:
        print("❗ DB 검색 실패:", e)
        return []

def search_user_db_context(content: str, user_contexts: List[str]) -> List[str]:
    if not user_contexts:
        return []
    query_vec = bert.encode(content)
    context_vecs = [bert.encode(txt) for txt in user_contexts]
    sims = cosine_similarity([query_vec], context_vecs)[0]
    top_k = sorted(range(len(sims)), key=lambda i: -sims[i])[:3]
    return [user_contexts[i] for i in top_k]

def clarify_with_rag(message: str, context_list: List[str]) -> str:
    context = "\n\n".join(context_list)
    prompt = f"과거 메시지를 참고해서 아래 문장을 더 명확하게 풀어 써주세요:\n\"{message}\"\n\n과거 메시지:\n{context}"
    try:
        res = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("❗ clarify_with_rag 실패:", e)
        return message

def search_web_pages(query: str, max_results: int = 1) -> List[str]:
    with DDGS() as ddgs:
        return list({r["href"] for r in ddgs.text(query, max_results=max_results)})

def extract_text_from_url(url: str) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception as e:
        print(f"❗ URL 본문 추출 실패: {url}", e)
        return ""

def summarize_and_classify(message: str, context: str) -> dict:
    user_prompt = f"""
메시지: {message}
문맥: {context}
JSON:
{{
  "summary": "...",
  "category": "..."
}}
"""
    system_prompt = """
너는 한국어 메신저 어시스턴트야.
카테고리는: "deadline", "payment", "public", "office", "others"
JSON 형식만 반환해:
{
  "summary": "...",
  "category": "..."
}
"""
    try:
        res = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        result = res.choices[0].message.content.strip()
        try:
            return json.loads(result)
        except:
            summary = re.search(r'"?summary"?\s*[:：]\s*"?([^"\n]+)"?', result)
            category = re.search(r'"?category"?\s*[:：]\s*"?([^"\n]+)"?', result)
            return {
                "summary": summary.group(1) if summary else "요약 실패",
                "category": category.group(1) if category else "others"
            }
    except Exception as e:
        print("❗ 요약/분류 실패:", e)
        return {"summary": "요약 실패", "category": "others"}

def embed_keywords_as_vector(keywords: List[str]) -> List[float]:
    sentence = " ".join(keywords)
    embedding = bert.encode(sentence, normalize_embeddings=True)
    return embedding.tolist()

def process_message_pipeline(message) -> AnalyzeResponse:
    try:
        original = message.content
        gpt_result = clarify_with_llm(original)
        clarified = gpt_result["clarified"]
        user_keywords = gpt_result.get("db_keywords", [])
        web_keywords = gpt_result.get("web_keywords", [])
        needs_user_context = gpt_result.get("needs_user_context", False)
        needs_external_info = gpt_result.get("needs_external_info", False)

        if needs_user_context and user_keywords:
            user_msgs = fetch_user_messages_by_keywords(message.cochat_id, user_keywords)
            context = search_user_db_context(clarified, user_msgs)
            clarified = clarify_with_rag(clarified, context)

        if needs_external_info and web_keywords:
            for kw in web_keywords:
                for url in search_web_pages(kw):
                    if any(b in url for b in BLOCKED_DOMAINS):
                        continue
                    text = extract_text_from_url(url)
                    if text:
                        add_docs_to_vectorstore([text], source_name=url)
                        break

        try:
            retrieved = vector_store.similarity_search(clarified, k=3)
            context_text = " ".join([r.page_content for r in retrieved])
        except Exception as e:
            print("❗ similarity_search 실패:", e)
            context_text = ""

        metadata = summarize_and_classify(clarified, context_text)
        message_vector = bert.encode(clarified).tolist()
        user_vector = fetch_user_preference_vector(message.cochat_id)
        recommended = is_recommended(user_vector, message_vector)

        if metadata.get("summary") == "요약 실패" and recommended:
            short = clarified[:50] + "..." if len(clarified) > 50 else clarified
            fallback = {
                "deadline": f"기한 관련 메시지: {short}",
                "payment": f"결제 관련 내용입니다: {short}",
                "public": f"공지 관련 내용입니다: {short}",
                "office": f"사무실 관련 내용입니다: {short}",
                "others": f"기타 메시지입니다: {short}"
            }.get(metadata.get("category", "others"), short)
            metadata["summary"] = fallback

        return AnalyzeResponse(
            content=original,
            clarified=clarified,
            summary=metadata.get("summary", "요약 실패"),
            category=metadata.get("category", "others"),
            embedding_vector=message_vector,
            recommended=recommended
        )

    except Exception as e:
        print("❗ 전체 파이프라인 실패:", e)
        raise e

def create_embedding(req):
    embedding = bert.encode(req.text, normalize_embeddings=True)
    return {"embedding": embedding.tolist(), "dim": len(embedding)}

def update_preference_by_message(req):
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