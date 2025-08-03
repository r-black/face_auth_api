import numpy as np

from app.core.constants import ALLOWED_EXTENSIONS


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # безопасно, если норма — 0
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    if norma == 0 or normb == 0:
        return 0.0
    return float(np.dot(a, b) / (norma * normb))
