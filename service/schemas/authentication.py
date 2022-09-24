from pydantic import BaseModel


class Authentication(BaseModel):
    is_authenticated: bool = False
