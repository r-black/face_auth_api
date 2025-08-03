from typing import List
from urllib.parse import urlparse
import json

from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )

    API_V1_STR: str = "/api/v1"
    SERVER_NAME: str = "localhost"
    PROJECT_NAME: str = "Face Auth API"

    FACE_COMPARE_THRESHOLD: float = 0.35

    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: str = "19530"
    MILVUS_COLLECTION: str = "face_embeddings"
    MILVUS_DIM: int = 512

    BACKEND_CORS_ORIGINS: List[str] = Field(default_factory=list)

    @staticmethod
    def _parse_origins(raw: object) -> List[str]:
        if isinstance(raw, str):
            raw = raw.strip()
            if raw.startswith("[") and raw.endswith("]"):
                try:
                    parsed = json.loads(raw)
                    if not isinstance(parsed, list):
                        raise ValueError("BACKEND_CORS_ORIGINS JSON must be a list")
                    origins = [str(o) for o in parsed]
                except Exception as e:
                    raise ValueError(f"cannot parse BACKEND_CORS_ORIGINS as JSON: {e}")
            else:
                origins = [piece.strip() for piece in raw.split(",") if piece.strip()]
        elif isinstance(raw, list):
            origins = [str(o) for o in raw]
        else:
            raise ValueError(f"invalid type for BACKEND_CORS_ORIGINS: {type(raw)}")
        return origins

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def _validate_and_normalize_origins(cls, v):
        origins = cls._parse_origins(v)
        normalized: List[str] = []
        for origin in origins:
            parsed = urlparse(origin)
            if not parsed.scheme:
                origin = f"http://{origin}"
            normalized.append(origin)
        return normalized


settings = Settings()
