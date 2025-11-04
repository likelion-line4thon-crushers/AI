# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config.settings import settings
from core.redis import get_redis, close_redis
from routers.report import router as report_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    redis = await get_redis()
    try:
        await redis.ping()
        print("[startup] ✅ Redis connected successfully")
    except Exception as e:
        print(f"[startup] ❌ Redis connection failed: {e}")
        raise

    yield  # 👉 여기까지 실행되면 앱이 '정상 구동 중'

    # Shutdown
    await close_redis()
    print("[shutdown] 🧹 Redis connection closed")

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(report_router)

# 헬스체크
@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": settings.APP_NAME}
