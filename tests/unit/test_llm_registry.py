import configparser

import pytest

from tinytroupe.llm.providers.cloudflare import CloudflareProvider
from tinytroupe.llm.providers.gemini import GeminiProvider
from tinytroupe.llm.providers.openai_compatible import OpenAICompatibleProvider
from tinytroupe.llm.registry import get_provider


def test_registry_selects_openai_compatible_from_llm_provider(monkeypatch):
    cfg = configparser.ConfigParser()
    cfg["LLM"] = {"PROVIDER": "openai_compatible"}
    cfg["OpenAI"] = {"MODEL": "gpt-5-mini"}
    cfg["OpenAICompatible"] = {
        "BASE_URL": "https://example.test/v1",
        "API_KEY_ENV": "TEST_OPENAI_KEY",
    }

    monkeypatch.setenv("TEST_OPENAI_KEY", "dummy")
    monkeypatch.setattr("tinytroupe.llm.registry.utils.read_config_file", lambda: cfg)

    provider = get_provider()
    assert isinstance(provider, OpenAICompatibleProvider)


def test_registry_selects_gemini(monkeypatch):
    cfg = configparser.ConfigParser()
    cfg["LLM"] = {"PROVIDER": "gemini"}
    cfg["OpenAI"] = {"MODEL": "gpt-5-mini"}
    cfg["Gemini"] = {
        "API_KEY_ENV": "TEST_GEMINI_KEY",
        "MODEL": "gemini-1.5-pro",
    }

    monkeypatch.setenv("TEST_GEMINI_KEY", "dummy")
    monkeypatch.setattr("tinytroupe.llm.registry.utils.read_config_file", lambda: cfg)

    provider = get_provider()
    assert isinstance(provider, GeminiProvider)


def test_registry_selects_cloudflare(monkeypatch):
    cfg = configparser.ConfigParser()
    cfg["LLM"] = {"PROVIDER": "cloudflare"}
    cfg["OpenAI"] = {"MODEL": "gpt-5-mini"}
    cfg["Cloudflare"] = {
        "ACCOUNT_ID_ENV": "TEST_CF_ACCOUNT",
        "API_TOKEN_ENV": "TEST_CF_TOKEN",
        "MODEL": "@cf/meta/llama-3.1-8b-instruct",
    }

    monkeypatch.setenv("TEST_CF_ACCOUNT", "dummy")
    monkeypatch.setenv("TEST_CF_TOKEN", "dummy")
    monkeypatch.setattr("tinytroupe.llm.registry.utils.read_config_file", lambda: cfg)

    provider = get_provider()
    assert isinstance(provider, CloudflareProvider)


def test_registry_maps_legacy_api_type_to_openai_compatible(monkeypatch):
    cfg = configparser.ConfigParser()
    cfg["OpenAI"] = {
        "API_TYPE": "ollama",
        "MODEL": "llama3",
        "BASE_URL": "http://localhost:11434/v1",
    }
    cfg["OpenAICompatible"] = {
        "API_KEY_ENV": "TEST_LEGACY_KEY",
    }

    monkeypatch.setenv("TEST_LEGACY_KEY", "dummy")
    monkeypatch.setattr("tinytroupe.llm.registry.utils.read_config_file", lambda: cfg)

    provider = get_provider()
    assert isinstance(provider, OpenAICompatibleProvider)


def test_registry_rejects_unsupported_provider(monkeypatch):
    cfg = configparser.ConfigParser()
    cfg["LLM"] = {"PROVIDER": "unsupported"}
    cfg["OpenAI"] = {"API_TYPE": "openai", "MODEL": "gpt-5-mini"}

    monkeypatch.setattr("tinytroupe.llm.registry.utils.read_config_file", lambda: cfg)

    with pytest.raises(ValueError, match="Unsupported provider"):
        get_provider()
