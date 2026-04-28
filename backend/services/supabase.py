# ── supabase.py ──────────────────────────────────────────────────────

import os
import logging
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_client = None

def get_client():
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        _client = create_client(url, key)
    return _client


BUCKET = "mussel-images"


def upload_image(session_id: str, label: str, data: bytes) -> str:
    """Upload raw bytes to Supabase Storage and return the storage path."""
    path = f"{session_id}/{label}.jpg"
    try:
        get_client().storage.from_(BUCKET).upload(
            path,
            data,
            {"content-type": "image/jpeg", "upsert": "true"},
        )
    except Exception as e:
        logger.error("Storage upload failed for %s: %s", path, e)
        raise
    return path


def get_session(session_id: str) -> dict | None:
    """Return session row or None if it doesn't exist."""
    try:
        res = (
            get_client()
            .table("sessions")
            .select("*")
            .eq("id", session_id)
            .single()
            .execute()
        )
        return res.data
    except Exception:
        # .single() raises if no row found
        return None


def update_session(session_id: str, fields: dict):
    """Upsert a session row — creates it on first call, updates on subsequent calls."""
    get_client().table("sessions").upsert(
        {"id": session_id, **fields},
        on_conflict="id"
    ).execute()