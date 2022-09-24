from fastapi import APIRouter

from service.api.api_v1.endpoints import index, compare_faces

api_router = APIRouter()
api_router.include_router(index.router, prefix="", tags=["index"])
api_router.include_router(compare_faces.router, prefix="/compare-faces", tags=["compare_faces"])
