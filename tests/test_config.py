"""Tests for config.py: missing key fails loudly, key never appears in repr."""

import pytest

from rag import config


def test_missing_key_raises(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)  # ignore the real .env
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        config.load_settings()


def test_key_not_in_repr(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GROQ_API_KEY", "secret-test-value")
    settings = config.load_settings()
    assert "secret-test-value" not in repr(settings)
    assert settings.groq_api_key == "secret-test-value"
