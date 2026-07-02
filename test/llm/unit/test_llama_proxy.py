import pytest

from nemantix.core import Toolset, tool
from nemantix.llm.abstract_proxy import (
    AbstractLLMProxy,
    LLMProxyException,
    LLMResponse,
    LLMUsage,
)
from nemantix.llm.credentials import Credentials
from nemantix.llm.llama_proxy import LlamaProxy


@pytest.fixture
def llama_llm_proxy(mock_openai_client, monkeypatch):
    # Patch the OpenAI class where the base OpenAICompatibleProxy instantiates it
    monkeypatch.setattr("nemantix.llm.openai_proxy.OpenAI", mock_openai_client)
    monkeypatch.setenv("OLLAMA_API_KEY", "mock-api-key")
    AbstractLLMProxy.set_credentials_manager(Credentials())
    return LlamaProxy("llama-3.1-8b", host="http://localhost:11434")


def test_llama_init_and_invoke_stream_bind_unbind(llama_llm_proxy):
    proxy = llama_llm_proxy

    # Unbound invoke
    out = proxy.invoke("Hello")
    assert isinstance(out, LLMResponse)
    assert "Mock response to: Hello" in out.text
    assert out.tool_calls == []
    assert isinstance(out.usage, LLMUsage)
    assert out.usage.input_tokens >= 0

    # Bind tools and invoke
    class DummyToolset(Toolset):
        @tool
        def get_current_weather(self, location: str):
            """A dummy tool for weather."""
            pass

    proxy2 = proxy.bind_tools(DummyToolset, ["get_current_weather"])
    out2 = proxy2.invoke("What's the weather?")
    assert "get_current_weather" in [tc["name"] for tc in out2.tool_calls]
    assert out2.tool_calls[0]["args"] == {"location": "Boston"}

    # Streaming
    chunks = list(proxy.stream("abc"))
    assert "".join(chunks) == "Mock stream response."

    # Unbind tools
    proxy3 = proxy.unbind_tools()
    out3 = proxy3.invoke("What's the weather?")
    assert "Mock response to: What's the weather?" in out3.text
    assert out3.tool_calls == []


def test_llama_grammar_based_raises_exception(llama_llm_proxy):
    # Ollama proxy explicitly forbids grammar invocations
    with pytest.raises(
        LLMProxyException, match="Grammar-based invocation is not natively supported"
    ):
        llama_llm_proxy.invoke_grammar_based("Hello")


def test_llama_errors_surface(monkeypatch):
    def bad_ctor(**kwargs):
        raise RuntimeError("ollama connection refused")

    monkeypatch.setattr("nemantix.llm.openai_proxy.OpenAI", bad_ctor)
    monkeypatch.setenv("OLLAMA_API_KEY", "mock-key")
    AbstractLLMProxy.set_credentials_manager(Credentials())

    with pytest.raises(
        LLMProxyException,
        match="Failed to initialize compatible client: ollama connection refused",
    ):
        LlamaProxy("llama-3.1-8b")
