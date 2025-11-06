import logging
from typing import List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

def _build_prompt(lines: List[str], max_lines: int = 3) -> str:
    joined = "\n".join(f"- {s}" for s in lines if s)
    return (
        "다음은 발표 중 청중 질문 목록입니다.\n"
        "겹치는 내용은 묶어 핵심만 한국어로 요약해 주세요.\n"
        f"출력은 최대 {max_lines}줄, 문장형으로 간결하게.\n\n"
        f"{joined}\n\n"
        f"[출력 형식]\n"
        f"1) ...\n2) ...\n3) ..."
    )

async def summarize_kor(questions: List[str], max_lines: int = 3) -> Optional[str]:
    if not questions:
        return None
    if not settings.OPENAI_API_KEY:
        logger.warning("[요약] OPENAI_API_KEY가 없어 요약을 생략합니다.")
        return None

    try:
        # OpenAI SDK v1.x
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        prompt = _build_prompt(questions, max_lines=max_lines)
        resp = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "넌 발표 보조 요약가야. 한국어로 명확하고 간결하게 적어."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=240,
        )
        text = resp.choices[0].message.content.strip()
        # 안전 가드: 3줄 초과 시 상위 3줄만
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return "\n".join(lines[:max_lines]) if lines else None
    except Exception as e:
        logger.error(f"[요약] OpenAI 호출 중 오류: {e}")
        return None
