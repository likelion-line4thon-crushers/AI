from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "Boini AI Report"
    API_PREFIX: str = "/report"
    REDIS_URL: str = "redis://redis:6379/0"
    ALLOW_ORIGINS: str = (
        "http://localhost:8080, "
        "http://localhost:5173, "
        "https://boini.shop, "
        "https://api.boini.shop"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.ALLOW_ORIGINS.split(",") if o.strip()]

settings = Settings()