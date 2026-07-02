from unittest.mock import MagicMock

import pytest

from nemantix.core import Toolset
from nemantix.llm.abstract_proxy import LLMProxyException, LLMResponse
from nemantix.llm.local_proxy import LocalLLMProxy


class _DummyToolset(Toolset):
    pass


def _make_local_proxy() -> LocalLLMProxy:
    proxy = object.__new__(LocalLLMProxy)
    proxy.llm = MagicMock(return_value="Generated text")
    proxy.params = None
    proxy.temperature = 0.8
    proxy.top_p = 0.95
    proxy.model_name = "test-model"
    proxy.role_start_token = ""
    proxy.role_stop_token = ""
    return proxy


def test_local_invoke_returns_llm_response():
    proxy = _make_local_proxy()
    result = proxy.invoke("hello")
    assert isinstance(result, LLMResponse)
    assert result.text == "Generated text"
    assert result.tool_calls == []


def test_local_invoke_with_list_returns_llm_response():
    proxy = _make_local_proxy()
    messages = [{"role": "user", "content": [{"text": "hello"}]}]
    result = proxy.invoke(messages)
    assert isinstance(result, LLMResponse)


def test_local_invoke_grammar_based_raises():
    proxy = _make_local_proxy()
    with pytest.raises(LLMProxyException, match="Not supported"):
        proxy.invoke_grammar_based("hello")


def test_local_bind_tools_raises():
    proxy = _make_local_proxy()
    with pytest.raises(LLMProxyException, match="Not supported"):
        proxy.bind_tools(_DummyToolset, [])


def test_local_unbind_tools_raises():
    proxy = _make_local_proxy()
    with pytest.raises(LLMProxyException, match="Not supported"):
        proxy.unbind_tools()
