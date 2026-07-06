"""
services/llm_judge.judge_same 단위 테스트.
AsyncOpenAI 클라이언트를 완전히 mock 해서 실제 API 호출 없이 방어 로직을 검증한다.
"""
import asyncio
import types

import pytest

import services.llm_judge as J


def _resp(content):
    """OpenAI 응답 객체 흉내 (resp.choices[0].message.content)."""
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


class _FakeCompletions:
    def __init__(self, behavior):
        self._behavior = behavior

    async def create(self, **kwargs):
        return await self._behavior(**kwargs)


class _FakeClient:
    def __init__(self, behavior):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(behavior))


def _install(monkeypatch, behavior, api_key="test-key", timeout=3.0):
    monkeypatch.setattr(J.settings, "OPENAI_API_KEY", api_key)
    monkeypatch.setattr(J.settings, "JUDGE_TIMEOUT_SECONDS", timeout)
    monkeypatch.setattr(J, "_get_client", lambda: _FakeClient(behavior))


async def test_returns_true_when_same(monkeypatch):
    async def behavior(**kwargs):
        return _resp('{"same": true}')
    _install(monkeypatch, behavior)

    assert await J.judge_same("a", "b") is True


async def test_returns_false_when_different(monkeypatch):
    async def behavior(**kwargs):
        return _resp('{"same": false}')
    _install(monkeypatch, behavior)

    assert await J.judge_same("a", "b") is False


async def test_none_when_no_api_key(monkeypatch):
    """API 키가 없으면 호출 없이 None."""
    called = []

    async def behavior(**kwargs):
        called.append(1)
        return _resp('{"same": true}')
    _install(monkeypatch, behavior, api_key=None)

    assert await J.judge_same("a", "b") is None
    assert called == []          # 클라이언트 호출조차 안 함


async def test_none_on_timeout(monkeypatch):
    """타임아웃 초과 시 None(폴백)."""
    async def slow(**kwargs):
        await asyncio.sleep(0.2)
        return _resp('{"same": true}')
    _install(monkeypatch, slow, timeout=0.01)

    assert await J.judge_same("a", "b") is None


async def test_none_on_api_error(monkeypatch):
    """API 호출 예외 시 None(폴백)."""
    async def boom(**kwargs):
        raise RuntimeError("connection reset")
    _install(monkeypatch, boom)

    assert await J.judge_same("a", "b") is None


async def test_none_on_non_json_content(monkeypatch):
    """JSON 이 아닌 응답이면 None(폴백)."""
    async def behavior(**kwargs):
        return _resp("음, 그건 좀 애매하네요")
    _install(monkeypatch, behavior)

    assert await J.judge_same("a", "b") is None


async def test_none_on_missing_same_key(monkeypatch):
    """JSON 이지만 same 키가 없으면 None(폴백)."""
    async def behavior(**kwargs):
        return _resp('{"answer": "yes"}')
    _install(monkeypatch, behavior)

    assert await J.judge_same("a", "b") is None


async def test_none_on_non_bool_same(monkeypatch):
    """same 값이 bool 이 아니면 None(폴백)."""
    async def behavior(**kwargs):
        return _resp('{"same": "yes"}')
    _install(monkeypatch, behavior)

    assert await J.judge_same("a", "b") is None
