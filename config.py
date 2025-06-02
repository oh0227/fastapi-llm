# config.py
import os

GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-32k")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
BLOCKED_DOMAINS = ['namu.wiki', 'collinsdictionary.com', 'reverso.net']

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432))
}