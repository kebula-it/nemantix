import pytest

from nemantix.llm.abstract_proxy import AbstractLLMProxy, LLMProxyException
from nemantix.llm.credentials import Credentials
from nemantix.llm.factory import LLMProxyFactory


def test_factory_requires_credentials_manager():
    # Ensure reset happened from auto-use fixture
    with pytest.raises(LLMProxyException):
        LLMProxyFactory.create_llm_proxy("openai", "gpt-4o")


def test_factory_creates_openai_and_google(monkeypatch):
    # Provide credentials via env
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("GOOGLE_API_KEY", "gk-env")
    AbstractLLMProxy.set_credentials_manager(
        Credentials.load_from_file(file_path="nonexistent.json")
    )

    openai = LLMProxyFactory.create_llm_proxy("openai", "gpt-4o", temperature=0.1)
    google = LLMProxyFactory.create_llm_proxy("google", "gemini-pro", top_p=0.5)

    assert openai.supports_tool_use() is True
    assert google.supports_tool_use() is True


def test_factory_unsupported_vendor(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    AbstractLLMProxy.set_credentials_manager(
        Credentials.load_from_file(file_path="nonexistent.json")
    )
    with pytest.raises(LLMProxyException):
        LLMProxyFactory.create_llm_proxy("x-ai", "grok")
