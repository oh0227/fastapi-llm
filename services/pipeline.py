import os
import re
import json
import requests
from typing import List
from fastapi import HTTPException
from sklearn.metrics.pairwise import cosine_similarity

from config import GPT_MODEL, OPENAI_API_KEY, DB_CONFIG, BLOCKED_DOMAINS
from .db import fetch_user_preference_vector, fetch_message_embedding
from .utils import is_recommended
from schemas import AnalyzeResponse
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

def openai_chat_completion(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            timeout=15
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❗ openai_chat_completion 실패:", e)
        raise e

def openai_embedding(text: str) -> List[float]:
    try:
        print("🔹 openai_embedding 호출: 입력 길이 =", len(text))
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        vector = response.data[0].embedding
        short_vector = vector[:10] + (["..."] if len(vector) > 10 else [])
        print("🔹 임베딩 결과 (요약):", short_vector)
        return vector
    except Exception as e:
        print("❗ openai_embedding 실패:", e)
        raise e


def extract_json_block(text: str) -> str:
    """
    LLM 응답에서 JSON 블록만 추출
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        raise ValueError("⚠️ JSON 블록을 찾을 수 없습니다. 전체 응답: " + text[:200])

def clarify_with_llm(message: str) -> dict:
    print("🔹 clarify_with_llm 시작:", message[:100])

    prompt = f"""
다음 메시지를 읽고, 줄임말·신조어·인터넷 은어·외래어·약어 등 벡터화 시 의미 손실이 생길 수 있는 단어들을 정확한 의미로 자연스럽게 바꿔줘.
예시:
- 플젝 → 프로젝트
- 덕질 → 특정 분야(아이돌, 애니메이션 등)에 깊이 빠져드는 활동
- 인티 → 인터넷 커뮤니티 사이트
- 플리 → 재생 목록 (playlist)
- 뇌정지 → 너무 놀라거나 당황한 상태
(위 예시처럼 바꾸지 않으면 임베딩 시 의미가 정확히 전달되지 않아)
- 모르는 단어/신조어/줄임말은 절대 추측해서 해석하지 마. 의미가 불명확한 경우는 반드시 "needs_user_context" 또는 "needs_external_info" 항목을 true로 설정해. 
- 특히, 감다뒤, 감겨따, 제미있따, 싱기방기 등 일반 사전에 없는 단어는 네가 임의로 뜻을 추정하면 위험해. 반드시 DB 검색용 키워드로 넘기고, 뜻 모르면 "모름"으로 둬.

또한, 아래 항목을 판단해줘:
1. 문맥(이전 대화)이 필요한지
2. 외부 정보(웹 검색 등)가 필요한지
3. 나중 검색용으로 중요해 보이는 키워드 3개 (DB 검색용 vs 웹 검색용은 구분해서)

아래 형식으로 응답해줘:
{{
  "clarified": "...",
  "needs_user_context": true/false,
  "db_keywords": [...],       # 문맥 필요로 DB 검색에 쓰일 키워드
  "needs_external_info": true/false,
  "web_keywords": [...]       # 외부 검색용 키워드 (지명, 행사명, 신조어 등)
}}

메시지: "{message}"
"""

    try:
        raw_response = openai_chat_completion(prompt)
        print("🔹 LLM 응답 원문:", raw_response[:300])
        json_text = extract_json_block(raw_response)
        parsed = json.loads(json_text)

        return {
            "clarified": parsed.get("clarified", message),
            "needs_user_context": parsed.get("needs_user_context", False),
            "db_keywords": parsed.get("db_keywords", []),
            "needs_external_info": parsed.get("needs_external_info", False),
            "web_keywords": parsed.get("web_keywords", [])
        }

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
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        keyword_clause = " OR ".join(["content ILIKE %s" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]
        query = f"""
            SELECT content FROM messages
            JOIN users ON messages.user_id = users.cochat_id
            WHERE users.cochat_id = %s AND ({keyword_clause})
            LIMIT 20;
        """
        cur.execute(query, (cochat_id, *params))
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
    query_vec = openai_embedding(content)
    context_vecs = [openai_embedding(txt) for txt in user_contexts]
    sims = cosine_similarity([query_vec], context_vecs)[0]
    top_k = sorted(range(len(sims)), key=lambda i: -sims[i])[:3]
    return [user_contexts[i] for i in top_k]

def clarify_with_rag(message: str, context_list: List[str]) -> str:
    context = "\n\n".join(context_list)
    prompt = f"과거 메시지를 참고해서 아래 문장을 더 명확하게 풀어 써주세요:\n\"{message}\"\n\n과거 메시지:\n{context}"
    try:
        return openai_chat_completion(prompt)
    except Exception as e:
        print("❗ clarify_with_rag 실패:", e)
        return message
from urllib.parse import urlparse, parse_qs, unquote

def search_web_pages(query: str, max_results: int = 1) -> List[str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://duckduckgo.com/html/?q={query}"
        r = requests.get(url, headers=headers, timeout=5)

        raw_links = re.findall(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"', r.text)
        cleaned_links = []

        for raw_link in raw_links:
            # 예: //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com
            if "uddg=" in raw_link:
                parsed_url = urlparse(raw_link)
                query_params = parse_qs(parsed_url.query)
                uddg = query_params.get("uddg")
                if uddg:
                    real_url = unquote(uddg[0])
                    if not any(blocked in real_url for blocked in BLOCKED_DOMAINS):
                        cleaned_links.append(real_url)
            else:
                # 그냥 일반적인 링크일 경우 (드물게 발생)
                if raw_link.startswith("//"):
                    real_url = "https:" + raw_link
                else:
                    real_url = raw_link
                if not any(blocked in real_url for blocked in BLOCKED_DOMAINS):
                    cleaned_links.append(real_url)

        return cleaned_links[:max_results]

    except Exception as e:
        print("❗ search_web_pages 실패:", e)
        return []


def extract_text_from_url(url: str) -> str:
    try:
        # 🔹 URL 보정: 스킴이 없으면 https:// 추가
        parsed = urlparse(url)
        if not parsed.scheme:
            if url.startswith("//"):
                url = "https:" + url
            else:
                url = "https://" + url

        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()  # 🔹 HTTP 오류 발생 시 예외 처리

        soup = BeautifulSoup(r.content, "lxml")

        # 🔹 p 태그만 추출 (길이 있는 것만)
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)

        if not text:
            print(f"⚠️ 본문이 비어 있습니다: {url}")
        return text.strip()

    except requests.exceptions.RequestException as req_err:
        print(f"❗ [요청 에러] URL 본문 추출 실패: {url}\n{req_err}")
    except Exception as e:
        print(f"❗ [기타 에러] URL 본문 추출 실패: {url}\n{e}")

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

def process_message_pipeline(message) -> AnalyzeResponse:
    try:
        original = message.content
        print(f"🔹 Pipeline 시작 - 원본 메시지 길이: {len(original)}")

        # 1. 메시지 명확화
        print("🔹 clarify_with_llm 호출")
        gpt_result = clarify_with_llm(original)
        print("🔹 clarify 결과:", gpt_result)

        clarified = gpt_result["clarified"]
        user_keywords = gpt_result.get("db_keywords", [])
        web_keywords = gpt_result.get("web_keywords", [])
        needs_user_context = gpt_result.get("needs_user_context", False)
        needs_external_info = gpt_result.get("needs_external_info", False)

        # 2. 사용자 문맥 처리
        if needs_user_context and user_keywords and message.cochat_id:
            print(f"🔹 사용자 문맥 필요 - 키워드: {user_keywords}")
            user_msgs = fetch_user_messages_by_keywords(message.cochat_id, user_keywords)
            print(f"🔹 사용자 메시지 수: {len(user_msgs)}")
            context = search_user_db_context(clarified, user_msgs)
            print(f"🔹 유사도 높은 문맥 수: {len(context)}")

            # 보정 전후 메시지 비교 및 벡터 변화 로그
            before_rag = clarified
            clarified = clarify_with_rag(before_rag, context)
            print("🔹 재명확화 완료")
            print(f"🔹 clarify_with_rag 변경 전: {before_rag}")
            print(f"🔹 clarify_with_rag 변경 후: {clarified}")

            before_vec = openai_embedding(before_rag)
            after_vec = openai_embedding(clarified)
            print("🔹 clarify_with_rag 전 벡터 (앞 10개):", before_vec[:10])
            print("🔹 clarify_with_rag 후 벡터 (앞 10개):", after_vec[:10])

        # 3. 외부 정보 검색
        if needs_external_info and web_keywords:
            print(f"🔹 외부 정보 필요 - 키워드: {web_keywords}")
            for kw in web_keywords:
                urls = search_web_pages(kw)
                print(f"🔹 검색된 URL들: {urls}")
                for url in urls:
                    text = extract_text_from_url(url)
                    if text:
                        print(f"🔹 URL에서 본문 추출 성공: {url}")
                        break

        # 4. 요약 및 분류
        context_text = ""  # 현재 context는 사용하지 않음
        print("🔹 summarize_and_classify 호출")
        metadata = summarize_and_classify(clarified, context_text)
        print("🔹 요약/분류 결과:", metadata)

        # 5. 메시지 임베딩
        print("🔹 openai_embedding 호출 (clarified 메시지)")
        message_vector = openai_embedding(clarified)
        print("🔹 메시지 벡터 (앞 10개):", message_vector[:10], "..." if len(message_vector) > 10 else "")

        # 6. 사용자 벡터와 추천 여부
        user_vector = fetch_user_preference_vector(message.cochat_id) if message.cochat_id else []
        print("🔹 사용자 벡터 (앞 10개):", user_vector[:10], "..." if len(user_vector) > 10 else "")
        recommended = is_recommended(user_vector, message_vector)
        print("🔹 추천 여부:", recommended)

        # 7. 요약 실패 시 fallback 처리
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
            print("🔹 요약 실패 fallback 사용:", fallback)

        print("✅ Pipeline 완료")
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
    embedding = openai_embedding(req.text)
    return {"embedding": embedding, "dim": len(embedding)}

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