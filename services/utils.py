# services/utils.py
import os, re, json, pickle
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import filter_complex_metadata
from config import BLOCKED_DOMAINS, FAISS_INDEX_PATH

def is_recommended(preference_vector, message_vector, threshold=0.8):
    if not preference_vector:
        return True
    sim = cosine_similarity([preference_vector], [message_vector])[0][0]
    return sim >= threshold

def load_vector_store(path, embedding_model):
    if os.path.exists(os.path.join(path, "faiss_store.pkl")):
        with open(os.path.join(path, "faiss_store.pkl"), "rb") as f:
            return pickle.load(f)
    dummy_doc = Document(page_content="dummy", metadata={"source": "init"})
    return FAISS.from_documents([dummy_doc], embedding_model)