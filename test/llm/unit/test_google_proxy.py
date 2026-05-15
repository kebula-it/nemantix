import pytest
from nemantix.llm.google_proxy import GoogleLLMProxy
from nemantix.llm.abstract_proxy import LLMProxyException, LLMResponse, LLMUsage
from nemantix.core import Toolset, tool


def test_google_init_and_invoke_stream_bind_unbind(google_llm_proxy):
    proxy = google_llm_proxy

    # Test invoking the proxy
    out = proxy.invoke("Ciao")
    assert isinstance(out, LLMResponse)
    assert "Ciao" in out.text
    assert out.tool_calls == []
    assert isinstance(out.usage, LLMUsage)
    assert out.usage.input_tokens >= 0
    assert out.usage.output_tokens >= 0

    # Simulate binding tools
    class DummyToolset(Toolset):
        @tool
        def get_current_weather(self, location: str):
            """Get weather tool."""
            pass

    proxy2 = proxy.bind_tools(DummyToolset, ["get_current_weather"])
    out2 = proxy2.invoke("Che tempo fa?")
    assert out2.tool_calls

    # Test streaming
    chunks = list(proxy.stream("abc"))
    assert "".join(chunks) == "ok1ok2"

    # Unbind tools and test again
    proxy3 = proxy.unbind_tools()
    out3 = proxy3.invoke("Che tempo fa?")
    assert out3.tool_calls == []


def test_google_errors_surface(monkeypatch):
    # noinspection PyUnusedLocal
    def bad_ctor(**kwargs):
        raise RuntimeError("kaput")

    # Patch the constructor to simulate an error
    monkeypatch.setattr("nemantix.llm.google_proxy.genai.Client", bad_ctor)

    monkeypatch.setenv("GOOGLE_API_KEY", "gk-env")

    with pytest.raises(LLMProxyException):
        GoogleLLMProxy("gemini-pro")
