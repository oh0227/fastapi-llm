# services/db.py
import psycopg2
import json
from config import DB_CONFIG

def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)

def fetch_user_preference_vector(cochat_id: str):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT preference_vector FROM users WHERE cochat_id = %s", (cochat_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else []

def fetch_message_embedding(message_id: int):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT embedding_vector FROM messages WHERE id = %s", (message_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else []

def fetch_user_preference_categories(cochat_id: str):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT preference_raw FROM users WHERE cochat_id = %s", (cochat_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    try:
        return json.loads(row[0]) if row and row[0] else []
    except Exception:
        return []