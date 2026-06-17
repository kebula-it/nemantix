from unittest.mock import patch

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
    AbstractLLMProxy.set_credentials_manager(Credentials())

    openai = LLMProxyFactory.create_llm_proxy("openai", "gpt-4o", temperature=0.1)
    google = LLMProxyFactory.create_llm_proxy("google", "gemini-pro", top_p=0.5)

    assert openai.supports_tool_use() is True
    assert google.supports_tool_use() is True


def test_factory_unsupported_vendor(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    AbstractLLMProxy.set_credentials_manager(Credentials())

    with pytest.raises(LLMProxyException, match="Unsupported LLM vendor"):
        LLMProxyFactory.create_llm_proxy("x-ai", "grok")


def test_factory_vendor_routing_aliases(monkeypatch):
    """Test that the factory correctly routes supported vendor aliases."""
    monkeypatch.setenv("DUMMY_KEY", "env")
    AbstractLLMProxy.set_credentials_manager(Credentials())

    # Test OpenRouter aliases that work
    # (Note: "OpenRouter" casing fails in factory.py due to vendor.lower())
    with patch("nemantix.llm.factory.OpenRouterLLMProxy") as mock_or:
        for alias in ["open-router", "open_router"]:
            LLMProxyFactory.create_llm_proxy(alias, "model-or")
        assert mock_or.call_count == 2

    # Test Llama.cpp Remote aliases
    with patch(
        "nemantix.experimental.llama_cpp_remote_proxy.LlamaCppRemoteLLMProxy"
    ) as mock_lcpp:
        for alias in ["llama.cpp", "llama-cpp", "llama-cpp-remote"]:
            LLMProxyFactory.create_llm_proxy(alias, "model-lcpp")
        assert mock_lcpp.call_count == 3


def test_factory_creates_remaining_vendors(monkeypatch):
    """Test routing for Azure, Anthropic, Ollama, and Local proxies."""
    monkeypatch.setenv("DUMMY_KEY", "sk-env")
    AbstractLLMProxy.set_credentials_manager(Credentials())

    with patch("nemantix.llm.factory.AzureOpenAILLMProxy") as mock_az:
        LLMProxyFactory.create_llm_proxy("azure", "gpt-4o", grammar_path="g.lark")
        mock_az.assert_called_once_with("gpt-4o", grammar_path="g.lark")

    with patch("nemantix.llm.factory.AnthropicLLMProxy") as mock_ant:
        LLMProxyFactory.create_llm_proxy("anthropic", "claude-3")
        mock_ant.assert_called_once_with("claude-3")

    with patch("nemantix.llm.llama_proxy.LlamaProxy") as mock_ollama:
        LLMProxyFactory.create_llm_proxy("ollama", "llama3")
        mock_ollama.assert_called_once_with("llama3")

    with patch("nemantix.llm.local_proxy.LocalLLMProxy") as mock_local:
        LLMProxyFactory.create_llm_proxy("local", "model.gguf")
        mock_local.assert_called_once_with("model.gguf")


def test_factory_grammar_path_resolution(monkeypatch):
    """Test that the factory properly injects the grammar path when not provided."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    AbstractLLMProxy.set_credentials_manager(Credentials())

    with (
        patch("nemantix.llm.factory.OpenAILLMProxy") as mock_openai,
        patch("nemantix.llm.factory.get_grammar_path", return_value="fake.lark"),
    ):
        # Default fallback resolves via get_grammar_path()
        LLMProxyFactory.create_llm_proxy("openai", "gpt-4o")
        mock_openai.assert_called_once_with("gpt-4o", grammar_path="fake.lark")

        # Explicit override bypasses get_grammar_path()
        mock_openai.reset_mock()
        LLMProxyFactory.create_llm_proxy("openai", "gpt-4o", grammar_path="custom.lark")
        mock_openai.assert_called_once_with("gpt-4o", grammar_path="custom.lark")
