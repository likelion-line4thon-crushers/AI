"""
군집화 회색지대(cosine 0.50~0.62) 전용 LLM 판정기 (Phase 2).

두 질문이 '본질적으로 같은 것을 묻는지' gpt-4o 로 판정한다.
외부 API 의존이므로 방어적으로 동작한다:
  - 타임아웃(settings.JUDGE_TIMEOUT_SECONDS) 초과, 네트워크/HTTP 오류,
    비정상/파싱불가 응답 → 모두 None 을 반환한다.
  - 호출부(add_question_to_clusters)는 None/False 를 '합류 안 함(신규)'로 처리하므로,
    실패 시 baseline(신규) 동작이 유지되고 서비스는 죽지 않는다.

반환값:
  True  = 같은 질문 → 합류
  False = 다른 질문 → 신규
  None  = 판정 실패(타임아웃/오류/파싱실패) → 호출부에서 신규로 폴백
"""
import asyncio
import json
import logging
import time
from typing import Optional

from openai import AsyncOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "너는 발표 청중 질문을 군집화하는 판정기야. 두 질문이 '본질적으로 같은 것을 묻는' "
    "질문이면 same=true, 주제나 의도가 다르면 same=false. 반드시 JSON만 출력해."
)

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


async def judge_same(question_a: str, question_b: str) -> Optional[bool]:
    """두 질문이 같은 것을 묻는지 판정. 실패 시 None (호출부에서 신규로 폴백)."""
    if not settings.OPENAI_API_KEY:
        logger.warning("[LLM판정] OPENAI_API_KEY 없음 → 폴백(None)")
        return None

    started = time.perf_counter()
    try:
        client = _get_client()
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.OPENAI_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content":
                        f'질문1: "{question_a}"\n질문2: "{question_b}"\n'
                        '두 질문이 같은 것을 묻고 있나요? '
                        '{"same": true} 또는 {"same": false} 형식 JSON으로만 답하세요.'},
                ],
                temperature=0,
                max_tokens=20,
                response_format={"type": "json_object"},
            ),
            timeout=settings.JUDGE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        elapsed = (time.perf_counter() - started) * 1000
        logger.warning(f"[LLM판정] 타임아웃 {settings.JUDGE_TIMEOUT_SECONDS}s 초과 ({elapsed:.0f}ms) → 폴백(신규)")
        return None
    except Exception as e:
        elapsed = (time.perf_counter() - started) * 1000
        logger.warning(f"[LLM판정] 호출 오류 ({elapsed:.0f}ms): {e!r} → 폴백(신규)")
        return None

    elapsed = (time.perf_counter() - started) * 1000
    try:
        content = resp.choices[0].message.content
        value = json.loads(content).get("same")
    except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"[LLM판정] 응답 파싱 실패 ({elapsed:.0f}ms): {e!r} → 폴백(신규)")
        return None

    if not isinstance(value, bool):
        logger.warning(f"[LLM판정] 비정상 응답 same={value!r} ({elapsed:.0f}ms) → 폴백(신규)")
        return None

    logger.info(
        f"[LLM판정] same={value} ({elapsed:.0f}ms) "
        f"A={question_a[:30]!r} B={question_b[:30]!r}"
    )
    return value
