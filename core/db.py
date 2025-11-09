from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator
from config.settings import settings

# 비동기 엔진 생성
engine = create_async_engine(
    settings.database_url_async(),
    echo=False,  # SQL 로그 보려면 True로 변경
    future=True,
)

# 세션 팩토리
async_session_factory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# FastAPI 의존성으로 쓸 수 있는 세션 함수
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session  # ← 이게 핵심! 반환(return) 아님. 컨텍스트/팩토리 아님.