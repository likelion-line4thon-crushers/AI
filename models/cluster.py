from pydantic import BaseModel
from typing import List


class QuestionInput(BaseModel):
    # Spring Boot가 증분 클러스터링 엔드포인트로 전송하는 새 질문 데이터
    id: str        # Spring Boot가 생성한 질문 고유 ID
    content: str   # 질문 내용
    slide: int     # 질문이 올라온 슬라이드 번호
    ts: int        # 질문 생성 타임스탬프 (epoch ms)


class ClusterItem(BaseModel):
    # 클러스터 1개의 결과 (프론트엔드에 전달되는 단위)
    representative: str      # 대표 질문 문구 (UI 접힌 상태에서 노출)
    count: int               # 해당 클러스터에 묶인 질문 수
    questionIds: List[str]   # 묶인 질문 ID 목록
    slides: List[int]        # 해당 질문들이 등장한 슬라이드 번호 목록 (중복 제거)
    samples: List[str]       # 묶인 질문 샘플 최대 3개 (펼쳤을 때 표시용)


class ClusterReportResponse(BaseModel):
    # 증분 클러스터링 엔드포인트의 응답 (data 필드에 담겨 반환됨)
    roomId: str           # 방 ID
    totalQuestions: int   # 방 전체 누적 질문 수 (Spring Boot 카운터 기준)
    uniqueGroups: int     # 현재 클러스터 개수
    clusters: List[ClusterItem]   # 클러스터 목록 (count 내림차순 정렬)
