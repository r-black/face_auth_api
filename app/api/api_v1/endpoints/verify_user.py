import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from starlette.concurrency import run_in_threadpool

from app.core.utils import allowed_file
from app.schemas.authentication import AuthenticationWithScore
from app.core.config import settings
from app.services.face_analysis import image_to_array, get_embedding
from app.services.milvus import search_user_history, save_user_embedding


router = APIRouter()
logger = logging.getLogger(__name__)


# Заглушка: замени на свою реализацию auth
def get_current_user():
    class User:
        id = 7857  # пример
    return User()


@router.post("/verify", response_model=AuthenticationWithScore)
async def verify_existing_user(
    selfie_image: UploadFile = File(...),
    current_user=Depends(get_current_user),
) -> AuthenticationWithScore:
    if not allowed_file(selfie_image.filename):
        raise HTTPException(status_code=415, detail="Unsupported file type")

    max_bytes = 5 * 1024 * 1024
    contents = await selfie_image.read()
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")
    from io import BytesIO
    selfie_image.file = BytesIO(contents)

    try:
        selfie_arr = await run_in_threadpool(image_to_array, selfie_image.file)
        try:
            selfie_emb = await run_in_threadpool(get_embedding, selfie_arr)
        except LookupError:
            return AuthenticationWithScore(
                is_authenticated=False,
                similarity=None,
                threshold=settings.FACE_COMPARE_THRESHOLD,
                detail="No face detected in selfie image",
            )

        # Пытаемся найти историю
        try:
            results = search_user_history(selfie_emb, user_id=current_user.id, top_k=3)
            print(results, current_user.id)
        except Exception:
            # проблема с history collection — fallback
            return AuthenticationWithScore(
                is_authenticated=False,
                similarity=None,
                threshold=settings.FACE_COMPARE_THRESHOLD,
                detail="History unavailable; please reverify with full identity flow",
            )

        if not results or not results[0]:
            return AuthenticationWithScore(
                is_authenticated=False,
                similarity=None,
                threshold=settings.FACE_COMPARE_THRESHOLD,
                detail="No prior embeddings found for user; please reverify with full flow",
            )

        best_score = results[0][0].score
        is_auth = best_score >= settings.FACE_COMPARE_THRESHOLD

        if is_auth:
            # сохраняем новое селфи как обновление истории
            save_user_embedding(current_user.id, selfie_emb, source="reauth")

        detail = None if is_auth else "Similarity below threshold"
        return AuthenticationWithScore(
            is_authenticated=is_auth,
            similarity=float(best_score),
            threshold=settings.FACE_COMPARE_THRESHOLD,
            detail=detail,
        )
    except Exception:
        logger.exception("verify_existing_user failed")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        try:
            selfie_image.file.close()
        except Exception:
            pass
