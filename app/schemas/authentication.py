from pydantic import BaseModel


class AuthenticationWithScore(BaseModel):
    is_authenticated: bool = False
    similarity: float | None = None
    threshold: float
    detail: str | None = None