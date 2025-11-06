# models/common.py
from typing import Generic, Optional, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class BaseResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str = "요청이 성공적으로 처리되었습니다."
    data: Optional[T] = None

def success(data: T, message: str = "요청이 성공적으로 처리되었습니다.") -> "BaseResponse[T]":
    return BaseResponse[T](success=True, message=message, data=data)
