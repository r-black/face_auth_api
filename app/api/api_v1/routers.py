from fastapi import APIRouter

from app.api.api_v1.endpoints import index, compare_faces, verify_user, debug


api_router = APIRouter()
api_router.include_router(index.router, prefix="", tags=["index"])
api_router.include_router(compare_faces.router, prefix="/verify-identity", tags=["compare_faces"])
api_router.include_router(verify_user.router, prefix="/verify-user", tags=["verify_user"])
api_router.include_router(debug.router, prefix="/debug", tags=["debug"])
