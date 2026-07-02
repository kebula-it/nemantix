from typing import List

import pytest
from google.genai import types
from pydantic import BaseModel

from nemantix.core import Toolset, tool
from nemantix.llm.abstract_proxy import (
    AbstractLLMProxy,
    Credentials,
    LLMProxyException,
    LLMResponse,
)
from nemantix.llm.google_proxy import GoogleLLMProxy


class LocationData(BaseModel):
    city: str
    zip_code: int


def test_google_init_and_invoke_stream_bind_unbind(google_llm_proxy):
    proxy = google_llm_proxy

    out = proxy.invoke("Ciao")
    assert isinstance(out, LLMResponse)

    class DummyToolset(Toolset):
        @tool
        def get_current_weather(self, location: str):
            """Get weather tool."""
            pass

        @tool
        def get_complex_weather(self, loc_data: LocationData, tags: List[str]):
            """A tool with complex schemas."""
            pass

    proxy2 = proxy.bind_tools(
        DummyToolset, ["get_current_weather", "get_complex_weather"]
    )

    tools = proxy2._generation_config.tools[0].function_declarations
    complex_tool = next(t for t in tools if t.name == "get_complex_weather")
    props = complex_tool.parameters.properties

    # Verify Pydantic model mapping
    assert props["loc_data"].type == types.Type.OBJECT
    assert "zip_code" in props["loc_data"].properties

    # Verify List mapping
    assert props["tags"].type == types.Type.ARRAY
    assert props["tags"].items.type == types.Type.STRING
    # ------------------------------------------------

    out2 = proxy2.invoke("Che tempo fa?")
    assert out2.tool_calls

    chunks = list(proxy.stream("abc"))
    assert "".join(chunks) == "ok1ok2"

    proxy3 = proxy.unbind_tools()
    out3 = proxy3.invoke("Che tempo fa?")
    assert out3.tool_calls == []


def test_google_errors_surface(monkeypatch):
    def bad_ctor(**__):
        raise RuntimeError("kaput")

    # Patch the constructor to simulate an error
    monkeypatch.setattr("nemantix.llm.google_proxy.genai.Client", bad_ctor)

    monkeypatch.setenv("GOOGLE_API_KEY", "gk-env")
    AbstractLLMProxy.set_credentials_manager(Credentials())

    with pytest.raises(LLMProxyException):
        GoogleLLMProxy("gemini-pro")
