import logging
import random

from fastapi import APIRouter, File, UploadFile, HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.utils import allowed_file, cosine_similarity
from app.schemas.authentication import AuthenticationWithScore
from app.core.config import settings
from app.services.face_analysis import image_to_array, get_embedding
from app.services.milvus import save_user_embedding

router = APIRouter()
logger = logging.getLogger(__name__)

class NoFaceFound(Exception):
    pass

@router.post("/verify_identity", response_model=AuthenticationWithScore)
async def verify_identity(
    passport_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...),
) -> AuthenticationWithScore:
    if not (allowed_file(passport_image.filename) and allowed_file(selfie_image.filename)):
        raise HTTPException(status_code=415, detail="Unsupported file type")

    # Опционально: ограничение размера
    max_bytes = 5 * 1024 * 1024  # 5MB
    for upload in (passport_image, selfie_image):
        contents = await upload.read()
        if len(contents) > max_bytes:
            raise HTTPException(status_code=413, detail="File too large")
        # нужно вернуть pointer назад, сделаем временно через BytesIO
        from io import BytesIO
        upload.file = BytesIO(contents)

    try:
        # CPU-bound / блокирующие преобразования выполняем в threadpool
        passport_arr = await run_in_threadpool(image_to_array, passport_image.file)
        selfie_arr = await run_in_threadpool(image_to_array, selfie_image.file)

        try:
            passport_emb = await run_in_threadpool(get_embedding, passport_arr)
        except LookupError:
            return AuthenticationWithScore(
                is_authenticated=False,
                similarity=None,
                threshold=settings.FACE_COMPARE_THRESHOLD,
                detail="No face detected in passport image",
            )
        try:
            selfie_emb = await run_in_threadpool(get_embedding, selfie_arr)
        except LookupError:
            return AuthenticationWithScore(
                is_authenticated=False,
                similarity=None,
                threshold=settings.FACE_COMPARE_THRESHOLD,
                detail="No face detected in selfie image",
            )

        similarity = cosine_similarity(passport_emb, selfie_emb)
        is_auth = similarity >= settings.FACE_COMPARE_THRESHOLD

        if is_auth:
            import uuid
            user_id = random.randint(1000, 9999)
            logger.info(f"User {user_id} authenticated")
            save_user_embedding(user_id, selfie_emb, source="signup")

        # Пример асинхронной записи в Milvus: можно вынести в background task, если нужно сохранять
        # тут просто возвращаем результат
        return AuthenticationWithScore(
            is_authenticated=is_auth,
            similarity=similarity,
            threshold=settings.FACE_COMPARE_THRESHOLD,
            detail=None if is_auth else "Similarity below threshold",
        )
    except Exception as e:
        logger.exception("Error in verify_identity")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # UploadFile .file может быть заменён выше, но попытка закрыть безопасна
        try:
            passport_image.file.close()
            selfie_image.file.close()
        except Exception:
            pass
