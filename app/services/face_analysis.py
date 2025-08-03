from typing import Optional
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis


# ленивый singleton
_face_analyzer: Optional[FaceAnalysis] = None


def get_face_analyzer() -> FaceAnalysis:
    global _face_analyzer
    if _face_analyzer is None:
        _face_analyzer = FaceAnalysis(name="buffalo_l")  # можно заменить модель
        _face_analyzer.prepare(ctx_id=0)
    return _face_analyzer


def image_to_array(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    return np.asarray(image)


def get_embedding(img_arr: np.ndarray) -> np.ndarray:
    fa = get_face_analyzer()
    faces = fa.get(img_arr)
    if not faces:
        raise LookupError("No face detected")
    # самый уверенный
    face = max(faces, key=lambda f: getattr(f, "det_score", 0))
    emb = face.embedding
    # нормализуем (на всякий случай)
    norm = np.linalg.norm(emb)
    if norm == 0:
        raise ValueError("Zero embedding")
    return emb / norm
