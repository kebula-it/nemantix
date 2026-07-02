from typing import Any, Iterator, Union
from unittest.mock import patch

import pytest

from nemantix.llm import LLMResponse
from nemantix.llm.abstract_proxy import AbstractLLMProxy
from nemantix.llm.config import LLMProxyConfig


class DummyLLMProxy(AbstractLLMProxy):
    def invoke_grammar_based(
        self, prompt: Union[str, list], **kwargs: Any
    ) -> LLMResponse:
        pass

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        pass

    def supports_tool_use(self) -> bool:
        pass

    def unbind_tools(self) -> "AbstractLLMProxy":
        pass

    def __init__(self, name="dummy"):
        self.name = name

    def get_name(self):
        return self.name

    def invoke(self, *args, **kwargs):
        pass

    def invoke_structured(self, *args, **kwargs):
        pass

    def messages_from(self, *args, **kwargs):
        pass

    def bind_tools(self, toolset_class, tool_names):
        pass


@pytest.fixture
def mock_credentials():
    with patch("nemantix.llm.config.Credentials") as mock:
        yield mock


@pytest.fixture
def mock_set_credentials_manager():
    with patch.object(AbstractLLMProxy, "set_credentials_manager") as mock:
        yield mock


@pytest.fixture
def mock_factory():
    with patch("nemantix.llm.config.LLMProxyFactory") as mock:
        mock.create_llm_proxy.return_value = DummyLLMProxy("factory_created")
        yield mock


def test_init_sets_defaults_and_loads_credentials(
    mock_credentials, mock_set_credentials_manager
):
    """Test that __init__ sets the default spec correctly and loads credentials."""
    config = LLMProxyConfig()

    # Verify Credentials() was instantiated directly
    mock_credentials.assert_called_once()

    # Verify the instantiated credentials manager was passed to the proxy
    mock_set_credentials_manager.assert_called_once_with(mock_credentials.return_value)

    assert config.default_spec == {"vendor": "openai", "model": "gpt-5-mini"}


def test_get_instantiates_from_dict_spec(mock_credentials, mock_factory):
    """Test that providing a dictionary spec correctly delegates to the factory."""
    custom_spec = {
        "vendor": "anthropic",
        "model": "claude-3",
        "kwargs": {"temperature": 0.5},
    }
    config = LLMProxyConfig(internal=custom_spec)

    proxy = config.get("internal")

    mock_factory.create_llm_proxy.assert_called_once_with(
        vendor="anthropic", model_name="claude-3", temperature=0.5
    )
    assert proxy == mock_factory.create_llm_proxy.return_value


def test_get_returns_pre_instantiated_proxy(mock_credentials, mock_factory):
    """Test that providing an already instantiated proxy bypasses the factory."""
    my_proxy = DummyLLMProxy("pre_instantiated")
    config = LLMProxyConfig(external=my_proxy)

    proxy = config.get("external")

    # Ensure factory was never called
    mock_factory.create_llm_proxy.assert_not_called()
    assert proxy is my_proxy


def test_get_unknown_key_falls_back_to_default(mock_credentials, mock_factory):
    """Test that requesting an unknown proxy key falls back to the default spec."""
    config = LLMProxyConfig()

    proxy = config.get("unknown_proxy")

    # Factory should be called with the default spec parameters
    mock_factory.create_llm_proxy.assert_called_once_with(
        vendor="openai", model_name="gpt-5-mini"
    )

    # Both the unknown key and 'default' should point to the exact same cached proxy instance
    assert proxy == mock_factory.create_llm_proxy.return_value
    assert config._proxies["unknown_proxy"] is config._proxies["default"]


def test_getattr_delegates_to_get(mock_credentials, mock_factory):
    """Test that the __getattr__ magic method functions as an alias to get()."""
    my_proxy = DummyLLMProxy("getattr_proxy")
    config = LLMProxyConfig(coding=my_proxy)

    # Accessing config.coding should trigger __getattr__ and return the proxy
    assert config.coding is my_proxy


def test_get_caches_instantiated_proxies(mock_credentials, mock_factory):
    """Test that subsequent calls to get() for the same key do not call the factory twice."""
    config = LLMProxyConfig()

    proxy_first_call = config.get("summary")
    proxy_second_call = config.get("summary")

    assert proxy_first_call is proxy_second_call
    mock_factory.create_llm_proxy.assert_called_once()
