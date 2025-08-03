import logging

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.api_v1.routers import api_router
from app.core.config import settings
from app.services.milvus import get_collection

logger = logging.getLogger("face_auth_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Face Auth API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
def on_startup():
    # предварительно инициализируем Milvus (соединение + коллекцию)
    try:
        get_collection()
        logger.info("Milvus collection ready")
    except Exception:
        logger.exception("Failed to initialize Milvus collection")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS or [],  # не ставить ["*"] по-умолчанию
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)
