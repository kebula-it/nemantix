from typing import List

import pytest
from pydantic import BaseModel

from nemantix.core import Toolset, tool
from nemantix.llm.abstract_proxy import (
    AbstractLLMProxy,
    LLMProxyException,
    LLMResponse,
    LLMUsage,
)
from nemantix.llm.anthropic_proxy import AnthropicLLMProxy
from nemantix.llm.credentials import Credentials


class LocationData(BaseModel):
    city: str
    zip_code: int


def test_anthropic_init_and_invoke_stream_bind_unbind(anthropic_llm_proxy):
    """
    This test accepts the `anthropic_llm_proxy` fixture,
    which ensures the client is mocked before the proxy is created.
    """
    proxy = anthropic_llm_proxy

    # unbound invoke
    out = proxy.invoke("Hello")
    assert isinstance(out, LLMResponse)
    assert "Mock response to: Hello" in out.text
    assert out.tool_calls == []
    assert isinstance(out.usage, LLMUsage)

    # bind tools and invoke
    class DummyToolset(Toolset):
        @tool
        def get_current_weather(self, location: str):
            """A dummy tool for weather."""
            pass

        @tool
        def get_complex_weather(self, loc_data: LocationData, tags: List[str]):
            """A tool with complex schemas."""
            pass

    proxy2 = proxy.bind_tools(
        DummyToolset, ["get_current_weather", "get_complex_weather"]
    )

    bound_tools = proxy2._bound_tools
    complex_tool = next(t for t in bound_tools if t["name"] == "get_complex_weather")
    props = complex_tool["input_schema"]["properties"]

    # Verify Pydantic model mapping
    assert props["loc_data"]["type"] == "object"
    assert "city" in props["loc_data"]["properties"]
    assert "zip_code" in props["loc_data"]["properties"]

    # Verify List mapping
    assert props["tags"]["type"] == "array"
    assert props["tags"]["items"]["type"] == "string"
    # ------------------------------------------------

    out2 = proxy2.invoke("What's the weather? And stock?")
    assert "get_current_weather" in [tc["name"] for tc in out2.tool_calls]
    assert out2.tool_calls[0]["args"] == {"location": "Boston"}

    # streaming yields characters
    chunks = list(proxy.stream("abc"))
    assert "".join(chunks) == "Mock stream response."

    # unbind
    proxy3 = proxy.unbind_tools()
    out3 = proxy3.invoke("What's the weather? And stock?")
    assert "Mock response to: What's the weather? And stock?" in out3.text
    assert out3.tool_calls == []


def test_anthropic_errors_surface(monkeypatch):
    def bad_ctor(**__):
        raise RuntimeError("boom")

    # Patch the resolved symbol within the module where it's used
    monkeypatch.setattr("nemantix.llm.anthropic_proxy.anthropic.Anthropic", bad_ctor)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    AbstractLLMProxy.set_credentials_manager(Credentials())

    with pytest.raises(
        LLMProxyException, match="Failed to initialize Anthropic client: boom"
    ):
        AnthropicLLMProxy("claude-sonnet-4-5-20250929")
