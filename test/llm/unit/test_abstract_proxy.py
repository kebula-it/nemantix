from typing import Any, Iterator, List, Type

import pytest
from pydantic import BaseModel

from nemantix.llm.abstract_proxy import (
    AbstractLLMProxy,
    LLMProxyException,
    LLMResponse,
    LLMUsage,
    StructuredLLMResponse,
)
from nemantix.llm.credentials import Credentials


class DummyProxy(AbstractLLMProxy):
    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            text=prompt, tool_calls=[], usage=LLMUsage(input_tokens=0, output_tokens=0)
        )

    def get_name(self) -> str:
        return "Dummy"

    def invoke_structured(
        self, prompt: str, schema: Type[BaseModel]
    ) -> StructuredLLMResponse:
        raise NotImplementedError()

    def invoke_grammar_based(self, prompt: str, **kwargs: Any) -> LLMResponse:
        raise NotImplementedError()

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        for ch in prompt:
            yield ch

    def supports_tool_use(self) -> bool:
        return False

    def bind_tools(self, tools: List[Any]) -> "DummyProxy":
        return self

    def unbind_tools(self) -> "DummyProxy":
        return self

    def messages_from(
        self, prompts_with_roles: list[dict[str, str] | tuple[str, str]]
    ) -> list[dict]:
        return prompts_with_roles


def test_get_api_key_from_kwargs_required():
    AbstractLLMProxy.set_credentials_manager(Credentials())
    p = DummyProxy()
    key = p._get_api_key("openai_api_key", api_key="sk-kwargs")
    assert key == "sk-kwargs"


def test_get_api_key_from_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-var")
    AbstractLLMProxy.set_credentials_manager(Credentials())
    p = DummyProxy()
    key = p._get_api_key("openai_api_key")
    assert key == "sk-env-var"


def test_get_api_key_required_missing_raises(monkeypatch):
    # Prevent load_dotenv from reading the real .env file
    monkeypatch.setattr(
        "nemantix.llm.credentials.load_dotenv", lambda *args, **kwargs: None
    )

    # Clear the environment variable
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    AbstractLLMProxy.set_credentials_manager(Credentials())
    p = DummyProxy()

    with pytest.raises(LLMProxyException):
        p._get_api_key("openai_api_key", required=True)


def test_get_api_key_not_required_returns_none(monkeypatch):
    # Prevent load_dotenv from reading the real .env file
    monkeypatch.setattr(
        "nemantix.llm.credentials.load_dotenv", lambda *args, **kwargs: None
    )

    # Clear the environment variable
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    AbstractLLMProxy.set_credentials_manager(Credentials())
    p = DummyProxy()

    assert p._get_api_key("openai_api_key", required=False) is None
