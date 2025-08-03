from typing import Any
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.core.utils import allowed_file
from app.schemas.authentication import AuthenticationWithScore
from app.services.face_analysis import image_to_array, get_embedding
from app.services.milvus_store import connect, ensure_collection
from app.core.config import settings

router = APIRouter()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@router.post("/verify_identity", response_model=AuthenticationWithScore)
async def verify_identity(
    passport_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...),
) -> Any:
    if not (allowed_file(passport_image.filename) and allowed_file(selfie_image.filename)):
        raise HTTPException(status_code=415, detail="Unsupported file type")

    try:
        # Загрузка и приведение к массиву
        passport_arr = image_to_array(passport_image.file)
        selfie_arr = image_to_array(selfie_image.file)

        # Получаем эмбеддинги лиц; порядок: сначала паспорт, потом селфи
        try:
            passport_emb = get_embedding(passport_arr)
        except LookupError:
            return AuthenticationWithScore(
                is_authenticated=False,
                similarity=None,
                threshold=settings.FACE_COMPARE_THRESHOLD,
                detail="No face detected in passport image",
            )
        try:
            selfie_emb = get_embedding(selfie_arr)
        except LookupError:
            return AuthenticationWithScore(
                is_authenticated=False,
                similarity=None,
                threshold=settings.FACE_COMPARE_THRESHOLD,
                detail="No face detected in selfie image",
            )

        similarity = cosine_similarity(passport_emb, selfie_emb)
        is_auth = similarity >= settings.FACE_COMPARE_THRESHOLD

        # Пример: можно логировать/сохранять в Milvus (например, паспортное embedding как reference)
        connect()
        collection = ensure_collection()
        # (опционально) сохранить паспортное embedding для дальнейших сравнений
        # collection.insert([[], [passport_emb.tolist()]])

        return AuthenticationWithScore(
            is_authenticated=is_auth,
            similarity=similarity,
            threshold=settings.FACE_COMPARE_THRESHOLD,
            detail=None if is_auth else "Similarity below threshold",
        )

    except Exception as e:
        print("verify_identity error:", e)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        try:
            passport_image.file.close()
            selfie_image.file.close()
        except Exception:
            pass
