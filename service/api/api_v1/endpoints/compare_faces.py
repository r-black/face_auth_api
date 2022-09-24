from typing import Any

from fastapi import APIRouter, UploadFile, File, HTTPException
import face_recognition

from service.core.utils import allowed_file
from service.schemas.authentication import Authentication

router = APIRouter()


@router.post("/compare_faces", status_code=200, response_model=Authentication)
async def compare_faces(
    known_image: UploadFile = File(...),
    unknown_image: UploadFile = File(...),
) -> Any:
    if allowed_file(known_image.filename) and allowed_file(unknown_image.filename):
        try:
            known = face_recognition.load_image_file(known_image.file)
            unknown = face_recognition.load_image_file(unknown_image.file)

            known_image_encoding = face_recognition.face_encodings(known)[0]
            unknown_encoding = face_recognition.face_encodings(unknown)[0]

            result = face_recognition.compare_faces([known_image_encoding], unknown_encoding, tolerance=0.43)
            if len(result):
                return Authentication(is_authenticated=result[0])
            else:
                return Authentication()
        except Exception as e:
            print(e)
            raise HTTPException(
                status_code=415,
                detail="There was an error uploading the files",
            )
        finally:
            known_image.file.close()
            unknown_image.file.close()
    else:
        raise HTTPException(
            status_code=415,
            detail="These types of images are not supported",
        )
