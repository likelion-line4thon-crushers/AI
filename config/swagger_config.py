from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
import os

# JWT Bearer 토큰 스키마
security = HTTPBearer()

def custom_openapi(app: FastAPI):
    if app.openapi_schema:
        return app.openapi_schema

    context_path = os.getenv("CONTEXT_PATH", "") or "/"

    openapi_schema = get_openapi(
        title="Boini API 명세서",
        version="1.0",
        description="Boini Swagger API Documentation",
        routes=app.routes,
        servers=[
            {
                "url": context_path,
                "description": "Boini Server"
            }
        ]
    )

    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    # JWT Bearer 인증 스키마 추가
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }

    # 전역 보안 요구사항 설정
    openapi_schema["security"] = [{"bearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

def setup_swagger(app: FastAPI):
    app.openapi = lambda: custom_openapi(app)

def create_app() -> FastAPI:
    app = FastAPI(
        title="Boini API",
        description="Boini Swagger API Documentation",
        version="1.0",
        docs_url="/swagger-ui",
        redoc_url=None
    )

    # Swagger 설정 적용
    setup_swagger(app)

    return app

# JWT 토큰 의존성
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # 여기에 JWT 토큰 검증 로직 추가
    return {"user_id": "example", "token": token}