from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from exception.errors import AppException, ErrorResponse, ReportErrorCode
from redis.exceptions import RedisError
from fastapi.responses import JSONResponse

from config.settings import settings
from core.redis import get_redis, close_redis
from core.db import engine
from models.max_slide_report import Base
import models.training_data  # Base.metadata에 TrainingData 테이블 등록
from routers.max_slide_report import router as report_router
from routers.top_question_report import router as topq_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    redis = await get_redis()
    try:
        await redis.ping()
        print("[startup] Redis 연결 성공")
    except Exception as e:
        print(f"[startup] Redis 연결 실패: {e}")
        raise

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("[startup] DB 테이블 생성 완료")
    except Exception as e:
        print(f"[startup] DB 테이블 생성 실패: {e}")
        raise

    yield  # 여기까지 실행되면 앱이 '정상 구동 중'

    # Shutdown
    await close_redis()
    print("[shutdown] 🧹 Redis connection closed")

app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
    root_path="/ai",
)

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
app.include_router(topq_router)

# errors.py에 정의된 타입을 사용
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    err = exc.error
    body = ErrorResponse(
        code=err.code,
        message=err.message,
        detail=exc.detail,
        path=str(request.url.path),
    )
    return JSONResponse(status_code=err.http_status, content=body.model_dump())

@app.exception_handler(RedisError)
async def redis_exception_handler(request: Request, exc: RedisError):
    err = ReportErrorCode.REDIS_ERROR
    body = ErrorResponse(
        code=err.code,
        message=err.message,
        detail=str(exc),
        path=str(request.url.path),
    )
    return JSONResponse(status_code=err.http_status, content=body.model_dump())

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled exception", exc_info=exc)
    err = ReportErrorCode.UNKNOWN
    body = ErrorResponse(
        code=err.code,
        message=err.message,
        detail=str(exc),
        path=str(request.url.path),
    )
    return JSONResponse(status_code=err.http_status, content=body.model_dump())

# 헬스체크
@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": settings.APP_NAME}
