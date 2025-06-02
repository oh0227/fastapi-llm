from sklearn.metrics.pairwise import cosine_similarity

def is_recommended(preference_vector, message_vector, threshold=0.8):
    if not preference_vector:
        return True
    sim = cosine_similarity([preference_vector], [message_vector])[0][0]
    print(f"메시지 유사도: {sim}")
    return sim >= threshold