from fastapi import APIRouter
from services.database import create_session

router = APIRouter()

@router.post("/create")
def create(payload: dict):
    session_id = create_session()
    return {"id": session_id}