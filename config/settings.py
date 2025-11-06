from pydantic_settings import BaseSettings
from typing import List, Optional

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

    # ===== LLM 요약 설정 =====
    OPENAI_API_KEY: Optional[str] = None  # [추가] 없으면 요약 기능 비활성
    OPENAI_MODEL: str = "gpt-4o-mini"     # [추가] 기본 요약 모델
    SUMMARY_MAX_LINES: int = 3            # [추가] 요약 줄 수 (기본 3줄)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.ALLOW_ORIGINS.split(",") if o.strip()]

settings = Settings()