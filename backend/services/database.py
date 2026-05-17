"""
services/database.py
---------------------
Supabase database — works both locally and on Google Colab.
Uses Supabase for session storage and image storage.
"""

import os
import uuid
import json
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# ── Supabase connection ───────────────────────────────────────────────────────
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")

_client = None

def get_client():
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _client

BUCKET = "mussel-images"


# ═════════════════════════════════════════════════════════════════════════════
#  IMAGE STORAGE
# ═════════════════════════════════════════════════════════════════════════════

def upload_image(session_id: str, label: str, data: bytes) -> str:
    """
    Uploads image to Supabase Storage.
    Returns the storage path string.
    """
    path = f"{session_id}/{label}.jpg"

    try:
        get_client().storage.from_(BUCKET).upload(
            path,
            data,
            {"content-type": "image/jpeg", "upsert": "true"}
        )
        print(f"[database] Image saved to Supabase: {path}")
    except Exception as e:
        print(f"[database] WARNING: Image upload failed: {e}")

    return path


# ═════════════════════════════════════════════════════════════════════════════
#  SESSION CRUD
# ═════════════════════════════════════════════════════════════════════════════

def create_session() -> str:
    """
    Inserts a new session row and returns the UUID.
    """
    session_id = str(uuid.uuid4())

    get_client().table("sessions").insert({
        "id":    session_id,
        "stage": 1,
    }).execute()

    print(f"[database] Session created: {session_id}")
    return session_id


def get_session(session_id: str) -> dict | None:
    """
    Fetches a session by ID.
    Returns a dict with all fields, or None if not found.
    """
    try:
        res = (
            get_client()
            .table("sessions")
            .select("*")
            .eq("id", session_id)
            .single()
            .execute()
        )
        row = res.data
    except Exception:
        print(f"[database] Session not found: {session_id}")
        return None

    if row is None:
        return None

    print(f"[database] Session fetched: {session_id}  stage={row.get('stage')}")

    # Parse JSON string fields back to Python dicts if needed
    for key in ["initial_features", "final_features"]:
        val = row.get(key)
        if val is None:
            continue
        try:
            if isinstance(val, str):
                row[key] = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            print(f"[database] WARNING: could not parse {key}")
            row[key] = None

    return row


def update_session(session_id: str, fields: dict):
    """
    Updates an existing session row.
    """
    if not fields:
        return

    # Supabase handles dicts natively as JSONB
    # but stringify just in case
    serialized = {}
    for key, value in fields.items():
        if isinstance(value, dict):
            serialized[key] = value  # Supabase JSONB accepts dicts directly
        elif value is None:
            serialized[key] = None
        else:
            serialized[key] = value

    print(f"[database] Updating session {session_id} — fields: {list(fields.keys())}")

    get_client().table("sessions").update(serialized).eq("id", session_id).execute()

    print(f"[database] Session updated: {session_id}")


def ensure_session_exists(session_id: str):
    """
    Creates the session row if it doesn't exist yet.
    """
    try:
        res = (
            get_client()
            .table("sessions")
            .select("id")
            .eq("id", session_id)
            .single()
            .execute()
        )
        exists = res.data is not None
    except Exception:
        exists = False

    if not exists:
        get_client().table("sessions").insert({
            "id":    session_id,
            "stage": 1,
        }).execute()
        print(f"[database] Session auto-created: {session_id}")