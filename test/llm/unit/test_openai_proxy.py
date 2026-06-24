from typing import List

import pytest
from pydantic import BaseModel

from nemantix.core import Toolset, tool
from nemantix.llm.abstract_proxy import (
    AbstractLLMProxy,
    LLMProxyException,
    LLMResponse,
    StructuredLLMResponse,
)
from nemantix.llm.credentials import Credentials
from nemantix.llm.openai_proxy import OpenAILLMProxy


class LocationData(BaseModel):
    city: str
    zip_code: int


def test_openai_init_and_invoke_stream_bind_unbind(openai_llm_proxy):
    proxy = openai_llm_proxy

    # unbound invoke
    out = proxy.invoke("Hello")
    assert isinstance(out, LLMResponse)

    # bind complex tools
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
    complex_tool = next(
        t for t in bound_tools if t["function"]["name"] == "get_complex_weather"
    )
    props = complex_tool["function"]["parameters"]["properties"]

    # Verify Pydantic model mapping
    assert props["loc_data"]["type"] == "object"
    assert "city" in props["loc_data"]["properties"]

    # Verify List mapping
    assert props["tags"]["type"] == "array"
    assert props["tags"]["items"]["type"] == "string"
    # ------------------------------------------------

    out2 = proxy2.invoke("What's the weather? And stock?")
    assert out2.tool_calls[0]["args"] == {"location": "Boston"}

    chunks = list(proxy.stream("abc"))
    assert "".join(chunks) == "Mock stream response."

    proxy3 = proxy.unbind_tools()
    out3 = proxy3.invoke("What's the weather? And stock?")
    assert out3.tool_calls == []


def test_invoke_structured_accepts_list_prompt(openai_llm_proxy):
    class Reply(BaseModel):
        result: str = ""

    messages = [{"role": "user", "content": "hello"}]
    result = openai_llm_proxy.invoke_structured(messages, schema=Reply)
    assert isinstance(result, StructuredLLMResponse)


def test_openai_errors_surface(monkeypatch):
    def bad_ctor(**__):
        raise RuntimeError("boom")

    # Patch the resolved symbol *within the module where it's used*
    monkeypatch.setattr("nemantix.llm.openai_proxy.OpenAI", bad_ctor)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    AbstractLLMProxy.set_credentials_manager(Credentials())

    with pytest.raises(
        LLMProxyException, match="Failed to initialize compatible client: boom"
    ):
        OpenAILLMProxy("gpt-4o")
