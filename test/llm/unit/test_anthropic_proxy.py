import pytest
from nemantix.llm.anthropic_proxy import AnthropicLLMProxy
from nemantix.llm.abstract_proxy import (
    AbstractLLMProxy,
    LLMProxyException,
    LLMResponse,
    LLMUsage,
)
from nemantix.llm.credentials import Credentials
from nemantix.core import Toolset, tool


def test_anthropic_init_and_invoke_stream_bind_unbind(anthropic_llm_proxy):
    """
    This test accepts the `anthropic_llm_proxy` fixture,
    which ensures the client is mocked before the proxy is created.
    """
    proxy = anthropic_llm_proxy  # Use the proxy instance from the fixture

    # unbound invoke
    out = proxy.invoke("Hello")
    assert isinstance(out, LLMResponse)
    assert "Mock response to: Hello" in out.text
    assert out.tool_calls == []
    assert isinstance(out.usage, LLMUsage)
    assert out.usage.input_tokens >= 0
    assert out.usage.output_tokens >= 0

    # bind tools and invoke
    class DummyToolset(Toolset):
        @tool
        def get_current_weather(self, location: str):
            """A dummy tool for weather."""
            pass

        @tool
        def get_stock_price(self, ticker: str):
            """Another dummy tool."""
            pass

    proxy2 = proxy.bind_tools(DummyToolset, ["get_current_weather", "get_stock_price"])
    out2 = proxy2.invoke("What's the weather? And stock?")
    assert "get_current_weather" in [tc["name"] for tc in out2.tool_calls]
    assert out2.tool_calls[0]["args"] == {"location": "Boston"}

    # streaming yields characters
    chunks = list(proxy.stream("abc"))
    assert "".join(chunks) == "Mock stream response."

    # unbind
    proxy3 = proxy.unbind_tools()
    out3 = proxy3.invoke("What's the weather? And stock?")
    # After unbinding, the mock should revert to a simple text response
    assert "Mock response to: What's the weather? And stock?" in out3.text
    assert out3.tool_calls == []


def test_anthropic_errors_surface(monkeypatch):
    # noinspection PyUnusedLocal
    def bad_ctor(**kwargs):
        raise RuntimeError("boom")

    # Patch the resolved symbol within the module where it's used
    monkeypatch.setattr("nemantix.llm.anthropic_proxy.anthropic.Anthropic", bad_ctor)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    AbstractLLMProxy.set_credentials_manager(
        Credentials.load_from_file(file_path="nonexistent.json")
    )

    with pytest.raises(
        LLMProxyException, match="Failed to initialize Anthropic client: boom"
    ):
        AnthropicLLMProxy("claude-sonnet-4-5-20250929")
