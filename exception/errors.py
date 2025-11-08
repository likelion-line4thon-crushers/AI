from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel
from fastapi import status

class ReportErrorCode(Enum):
    TOO_MANY_REQUESTS = ("Q001", status.HTTP_429_TOO_MANY_REQUESTS, "요청이 너무 많습니다. 잠시 후 다시 시도해주세요.")
    INVALID_ROOM_ID    = ("Q002", status.HTTP_400_BAD_REQUEST,        "유효하지 않은 방 ID입니다.")
    REDIS_ERROR        = ("Q003", status.HTTP_503_SERVICE_UNAVAILABLE, "데이터 저장소(Redis) 통신 중 오류가 발생했습니다.")
    STREAM_ERROR       = ("Q004", status.HTTP_500_INTERNAL_SERVER_ERROR,"이벤트 스트림 처리 중 오류가 발생했습니다.")
    NO_QUESTIONS       = ("Q005", status.HTTP_404_NOT_FOUND,           "질문 데이터가 없습니다.")
    UNKNOWN            = ("Q999", status.HTTP_500_INTERNAL_SERVER_ERROR,"알 수 없는 오류가 발생했습니다.")

    # top3_service.py 관련 예외 코드
    MODEL_LOAD_ERROR   = ("Q006", status.HTTP_500_INTERNAL_SERVER_ERROR, "임베딩 모델 로드 중 오류가 발생했습니다.")
    EMBED_ERROR        = ("Q007", status.HTTP_500_INTERNAL_SERVER_ERROR, "문장 임베딩 처리 중 오류가 발생했습니다.")
    CALC_ERROR         = ("Q008", status.HTTP_500_INTERNAL_SERVER_ERROR, "유사도 계산 중 오류가 발생했습니다.")
    PREPROCESS_ERROR   = ("Q009", status.HTTP_500_INTERNAL_SERVER_ERROR, "질문 전처리 중 오류가 발생했습니다.")


    @property
    def code(self) -> str:
        return self.value[0]

    @property
    def http_status(self) -> int:
        return self.value[1]

    @property
    def message(self) -> str:
        return self.value[2]


# 2) 에러 응답 바디 (표준화)
class ErrorResponse(BaseModel):
    code: str
    message: str
    detail: Optional[Any] = None
    path: Optional[str] = None


# 3) 애플리케이션 예외
class AppException(Exception):
    def __init__(self, error: ReportErrorCode, detail: Any = None, path: Optional[str] = None):
        self.error = error
        self.detail = detail
        self.path = path
