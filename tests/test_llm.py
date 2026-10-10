"""Tests for llm.py: cache key is stable and sensitive; the cache prevents a second API call."""

from rag import llm
from rag.config import Settings

MSG = [{"role": "user", "content": "hi"}]


def make_settings(tmp_path) -> Settings:
    return Settings(
        groq_api_key="x",
        groq_base_url="u",
        gen_model="g",
        judge_model="j",
        embedding_model="e",
        cache_dir=tmp_path,
    )


def test_cache_key_is_deterministic_and_sensitive():
    base = llm.cache_key("m", MSG, 0.0)
    assert base == llm.cache_key("m", MSG, 0.0)
    assert base != llm.cache_key("other", MSG, 0.0)
    assert base != llm.cache_key("m", [{"role": "user", "content": "hello"}], 0.0)
    assert base != llm.cache_key("m", MSG, 0.7)


def test_second_call_is_served_from_cache(tmp_path, monkeypatch):
    calls = []

    def fake(model, messages, temperature, settings):
        calls.append(1)
        return "answer"

    monkeypatch.setattr(llm, "_call_groq", fake)
    s = make_settings(tmp_path)
    assert llm.chat("m", MSG, settings=s) == "answer"
    assert llm.chat("m", MSG, settings=s) == "answer"
    assert len(calls) == 1  # the second call never reached the API
