# services/supabase.py
# Now acts as a thin wrapper that calls the local PostgreSQL database.
# This way predict.py doesn't need to change at all.

from services.database import upload_image, get_session, update_session

# Re-export everything so predict.py imports still work
__all__ = ["upload_image", "get_session", "update_session"]