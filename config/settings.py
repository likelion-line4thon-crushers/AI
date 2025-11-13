from pydantic_settings import BaseSettings
from typing import List, Optional
from urllib.parse import urlparse, urlunparse, quote

class Settings(BaseSettings):
    APP_NAME: str = "Boini AI Report"
    API_PREFIX: str = "/report"
    REDIS_URL: str = "redis://redis:6379/0"
    ALLOW_ORIGINS: str = (
        "http://localhost:8080, "
        "http://localhost:5173, "
        "https://boini.shop, "
        "https://api.boini.shop, "
        "https://boini.kr"
    )

    # ===== LLM 요약 설정 =====
    OPENAI_API_KEY: Optional[str] = None  # [추가] 없으면 요약 기능 비활성
    OPENAI_MODEL: str = "gpt-4o-mini"     # [추가] 기본 요약 모델
    SUMMARY_MAX_LINES: int = 3            # [추가] 요약 줄 수 (기본 3줄)

    DB_URL: Optional[str] = None           # 예) jdbc:mysql://host:3306/boini  또는  mysql://host:3306/boini
    DB_USERNAME: Optional[str] = None      # 예) root
    DB_PASSWORD: Optional[str] = None      # 예) secret

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.ALLOW_ORIGINS.split(",") if o.strip()]

    # ===== 비동기 SQLAlchemy용 DSN 생성 =====
    def database_url_async(self) -> str:
        """
        DB_URL / DB_USERNAME / DB_PASSWORD 를 조합해
        mysql+aiomysql DSN을 자동 생성한다.
        예시:
            jdbc:mysql://mysql-container:3306/boini
        → mysql+aiomysql://root:secret@mysql-container:3306/boini?charset=utf8mb4
        """
        if not self.DB_URL:
            raise ValueError("환경변수 DB_URL이 필요합니다.")

        raw = self.DB_URL.strip()
        if raw.startswith("jdbc:"):
            raw = raw[len("jdbc:"):]

        parsed = urlparse(raw)
        if parsed.scheme != "mysql":
            raise ValueError(f"지원하지 않는 DB 스킴입니다: {parsed.scheme}")

        username = self.DB_USERNAME or ""
        password = self.DB_PASSWORD or ""
        netloc = parsed.netloc

        # 자격증명(user/pass) 추가
        if "@" not in netloc:
            if username or password:
                cred = quote(username)
                if password:
                    cred += f":{quote(password)}"
                netloc = f"{cred}@{netloc}"

        # 비동기 드라이버 스킴 적용
        scheme_async = "mysql+aiomysql"

        # charset=utf8mb4 보장
        query = parsed.query
        if "charset=" not in query:
            query = f"{query}&charset=utf8mb4" if query else "charset=utf8mb4"

        final = parsed._replace(
            scheme=scheme_async,
            netloc=netloc,
            query=query,
        )
        return urlunparse(final)

settings = Settings()