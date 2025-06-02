import os
from sklearn.metrics.pairwise import cosine_similarity

def is_recommended(preference_vector, message_vector, threshold=None):
    if not preference_vector:
        return True

    if threshold is None:
        # 환경변수에서 가져오되, 없으면 기본값 0.6 사용
        threshold = float(os.getenv("RECOMMENDATION_THRESHOLD", 0.6))

    sim = cosine_similarity([preference_vector], [message_vector])[0][0]
    print(f"메시지 유사도: {sim}")
    return sim >= threshold