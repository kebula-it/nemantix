import pytest

from nemantix.core import Toolset, tool
from nemantix.llm.abstract_proxy import (
    AbstractLLMProxy,
    LLMProxyException,
    LLMResponse,
    LLMUsage,
)
from nemantix.llm.credentials import Credentials
from nemantix.llm.open_router_proxy import OpenRouterLLMProxy


@pytest.fixture
def open_router_llm_proxy(mock_openai_client, monkeypatch):
    # Patch the OpenAI class where the base OpenAICompatibleProxy instantiates it
    monkeypatch.setattr("nemantix.llm.openai_proxy.OpenAI", mock_openai_client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-api-key")
    AbstractLLMProxy.set_credentials_manager(
        Credentials.load_from_file(file_path="nonexistent.json")
    )
    return OpenRouterLLMProxy(
        model_name="anthropic/claude-3.5-sonnet",
        site_url="https://test.com",
        app_name="TestApp",
    )


def test_open_router_init_and_invoke_stream_bind_unbind(open_router_llm_proxy):
    proxy = open_router_llm_proxy

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


def test_open_router_errors_surface(monkeypatch):
    def bad_ctor(**kwargs):
        raise RuntimeError("openrouter timeout")

    monkeypatch.setattr("nemantix.llm.openai_proxy.OpenAI", bad_ctor)
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key")
    AbstractLLMProxy.set_credentials_manager(
        Credentials.load_from_file(file_path="nonexistent.json")
    )

    with pytest.raises(
        LLMProxyException,
        match="Failed to initialize compatible client: openrouter timeout",
    ):
        OpenRouterLLMProxy("anthropic/claude-3.5-sonnet")
