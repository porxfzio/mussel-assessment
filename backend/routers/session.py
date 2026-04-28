# routers/session.py

from fastapi import APIRouter, HTTPException
from services.supabase import get_session

router = APIRouter()


@router.get("/{session_id}")
async def fetch_session(session_id: str):
    """Check if a session exists (used for resuming interrupted assessments)."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session