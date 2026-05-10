from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import session, predict
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Mussel Assessment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session.router, prefix="/session", tags=["session"])
app.include_router(predict.router, prefix="/predict", tags=["predict"])