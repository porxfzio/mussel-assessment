"""
services/database.py
---------------------
Local PostgreSQL database — works completely offline.
Replaces Supabase for local development and demo.

Table uses TEXT columns for features (not JSONB) to avoid
driver parsing issues across different psycopg2 versions.
"""

import os
import uuid
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ── Database connection ───────────────────────────────────────────────────────
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:mussel123@localhost:5432/mussel_assessment"
)

engine = create_engine(DB_URL, pool_pre_ping=True)

# ── Image storage folder ──────────────────────────────────────────────────────
IMAGE_FOLDER = os.path.join(
    os.path.dirname(__file__), '..', 'uploaded_images'
)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  IMAGE STORAGE
# ═════════════════════════════════════════════════════════════════════════════

def upload_image(session_id: str, label: str, data: bytes) -> str:
    """
    Saves uploaded image to local folder.
    Returns the file path string.
    """
    folder = os.path.join(IMAGE_FOLDER, session_id)
    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, f"{label}.jpg")
    with open(filepath, "wb") as f:
        f.write(data)

    print(f"[database] Image saved: {filepath}")
    return filepath


# ═════════════════════════════════════════════════════════════════════════════
#  SESSION CRUD
# ═════════════════════════════════════════════════════════════════════════════

def create_session() -> str:
    """
    Inserts a new session row and returns the UUID.
    Called by routers/session.py when the frontend
    registers a new session ID.
    """
    session_id = str(uuid.uuid4())

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO sessions (id, stage)
            VALUES (:id, 1)
        """), {"id": session_id})
        conn.commit()

    print(f"[database] Session created: {session_id}")
    return session_id


def get_session(session_id: str) -> dict | None:
    """
    Fetches a session by ID.
    Returns a dict with all fields, or None if not found.
    Features stored as JSON strings are parsed back to dicts.
    """
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM sessions WHERE id = :id"),
            {"id": session_id}
        )
        row = result.mappings().first()

    if row is None:
        print(f"[database] Session not found: {session_id}")
        return None

    row_dict = dict(row)
    print(f"[database] Session fetched: {session_id}  stage={row_dict.get('stage')}")

    # Parse JSON string fields back to Python dicts
    for key in ["initial_features", "final_features"]:
        val = row_dict.get(key)
        if val is None:
            continue
        try:
            if isinstance(val, str):
                row_dict[key] = json.loads(val)
            # psycopg2 with JSONB may already return a dict — leave as-is
        except (json.JSONDecodeError, TypeError):
            print(f"[database] WARNING: could not parse {key}")
            row_dict[key] = None

    return row_dict


def update_session(session_id: str, fields: dict):
    """
    Updates an existing session row.
    Dict values are serialized to JSON strings before saving.
    """
    if not fields:
        return

    serialized = {}
    for key, value in fields.items():
        if isinstance(value, dict):
            serialized[key] = json.dumps(value)
        elif value is None:
            serialized[key] = None
        else:
            serialized[key] = value

    # Build dynamic SET clause
    set_clause = ", ".join(f"{k} = :{k}" for k in serialized)
    serialized["id"] = session_id

    print(f"[database] Updating session {session_id} — fields: {list(fields.keys())}")

    with engine.connect() as conn:
        conn.execute(
            text(f"UPDATE sessions SET {set_clause} WHERE id = :id"),
            serialized
        )
        conn.commit()

    print(f"[database] Session updated: {session_id}")


def ensure_session_exists(session_id: str):
    """
    Creates the session row if it doesn't exist yet.
    Useful because the frontend generates the UUID locally
    and may not always call /session/create first.
    """
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT id FROM sessions WHERE id = :id"),
            {"id": session_id}
        )
        exists = result.fetchone() is not None

        if not exists:
            conn.execute(
                text("INSERT INTO sessions (id, stage) VALUES (:id, 1)"),
                {"id": session_id}
            )
            conn.commit()
            print(f"[database] Session auto-created: {session_id}")