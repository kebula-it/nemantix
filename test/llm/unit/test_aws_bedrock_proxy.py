from typing import List

import pytest
from pydantic import BaseModel

from nemantix.core import Toolset, tool
from nemantix.llm.abstract_proxy import (
    AbstractLLMProxy,
    LLMProxyException,
    LLMResponse,
    LLMUsage,
    StructuredLLMResponse,
)
from nemantix.llm.aws_bedrock_proxy import AWSBedrockLLMProxy
from nemantix.llm.credentials import Credentials


class CityData(BaseModel):
    city: str
    zip_code: int


# ---------------------------------------------------------------------------
# Plain invoke / stream
# ---------------------------------------------------------------------------


def test_bedrock_invoke_plain(bedrock_llm_proxy):
    out = bedrock_llm_proxy.invoke("Hello")

    assert isinstance(out, LLMResponse)
    assert "Mock response to: Hello" in out.text
    assert out.tool_calls == []
    assert isinstance(out.usage, LLMUsage)
    assert out.usage.input_tokens == 10
    assert out.usage.output_tokens == 5


def test_bedrock_invoke_with_message_list(bedrock_llm_proxy):
    messages = [{"role": "user", "content": "Hello from list"}]
    out = bedrock_llm_proxy.invoke(messages)

    assert isinstance(out, LLMResponse)
    assert "Mock response to: Hello from list" in out.text


def test_bedrock_stream(bedrock_llm_proxy):
    chunks = list(bedrock_llm_proxy.stream("Hello"))
    assert "".join(chunks) == "Mock stream response."


# ---------------------------------------------------------------------------
# Tool binding
# ---------------------------------------------------------------------------


def test_bedrock_bind_tools_schema(bedrock_llm_proxy):
    class DummyToolset(Toolset):
        @tool
        def get_current_weather(self, location: str):
            """A dummy weather tool."""
            pass

        @tool
        def get_complex_weather(self, city_data: CityData, tags: List[str]):
            """A tool with complex schemas."""
            pass

    proxy = bedrock_llm_proxy.bind_tools(
        DummyToolset, ["get_current_weather", "get_complex_weather"]
    )

    tool_names = [t["toolSpec"]["name"] for t in proxy._bound_tools]
    assert "get_current_weather" in tool_names
    assert "get_complex_weather" in tool_names

    complex_spec = next(
        t["toolSpec"]
        for t in proxy._bound_tools
        if t["toolSpec"]["name"] == "get_complex_weather"
    )
    props = complex_spec["inputSchema"]["json"]["properties"]

    assert props["city_data"]["type"] == "object"
    assert "city" in props["city_data"]["properties"]
    assert "zip_code" in props["city_data"]["properties"]
    assert props["tags"]["type"] == "array"
    assert props["tags"]["items"]["type"] == "string"


def test_bedrock_invoke_with_tools(bedrock_llm_proxy):
    class DummyToolset(Toolset):
        @tool
        def get_current_weather(self, location: str):
            """A dummy weather tool."""
            pass

    proxy = bedrock_llm_proxy.bind_tools(DummyToolset, ["get_current_weather"])
    out = proxy.invoke("What's the weather?")

    assert out.tool_calls[0]["name"] == "get_current_weather"
    assert out.tool_calls[0]["args"] == {"location": "Boston"}
    assert out.tool_calls[0]["id"] == "id_123"


def test_bedrock_unbind_tools(bedrock_llm_proxy):
    class DummyToolset(Toolset):
        @tool
        def get_current_weather(self, location: str):
            """A dummy weather tool."""
            pass

    bedrock_llm_proxy.bind_tools(DummyToolset, ["get_current_weather"])
    assert len(bedrock_llm_proxy._bound_tools) == 1

    bedrock_llm_proxy.unbind_tools()
    assert bedrock_llm_proxy._bound_tools == []

    out = bedrock_llm_proxy.invoke("What's the weather?")
    assert out.tool_calls == []


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


def test_bedrock_invoke_structured(bedrock_llm_proxy):
    result = bedrock_llm_proxy.invoke_structured("Get city data.", CityData)

    assert isinstance(result, StructuredLLMResponse)
    assert isinstance(result.result, CityData)
    assert result.result.city == "Boston"
    assert result.result.zip_code == 2101
    assert isinstance(result.usage, LLMUsage)


# ---------------------------------------------------------------------------
# Grammar-based (falls back to invoke)
# ---------------------------------------------------------------------------


def test_bedrock_invoke_grammar_based_falls_back(bedrock_llm_proxy):
    out = bedrock_llm_proxy.invoke_grammar_based("Hello")

    assert isinstance(out, LLMResponse)
    assert "Mock response to: Hello" in out.text


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def test_bedrock_supports_tool_use(bedrock_llm_proxy):
    assert bedrock_llm_proxy.supports_tool_use() is True


def test_bedrock_get_name(bedrock_llm_proxy):
    name = bedrock_llm_proxy.get_name()
    assert "AWS Bedrock" in name
    assert "us.anthropic.claude-3-5-sonnet-20241022-v2:0" in name


def test_bedrock_messages_from(bedrock_llm_proxy):
    messages = bedrock_llm_proxy.messages_from(
        [
            ("user", "Hello"),
            {"role": "assistant", "prompt": "Hi there!"},
        ]
    )

    assert messages[0] == {"role": "user", "content": [{"text": "Hello"}]}
    assert messages[1] == {"role": "assistant", "content": [{"text": "Hi there!"}]}


# ---------------------------------------------------------------------------
# _normalize_messages (static)
# ---------------------------------------------------------------------------


def test_normalize_messages_string_prompt():
    msgs, system = AWSBedrockLLMProxy._normalize_messages("Hello")

    assert system is None
    assert msgs == [{"role": "user", "content": [{"text": "Hello"}]}]


def test_normalize_messages_system_role():
    msgs, system = AWSBedrockLLMProxy._normalize_messages(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
    )

    assert system == [{"text": "You are a helpful assistant."}]
    assert msgs == [{"role": "user", "content": [{"text": "Hello"}]}]


def test_normalize_messages_list_content():
    msgs, system = AWSBedrockLLMProxy._normalize_messages(
        [{"role": "user", "content": [{"text": "block one"}, {"text": "block two"}]}]
    )

    assert system is None
    assert msgs[0]["content"] == [{"text": "block one"}, {"text": "block two"}]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_bedrock_init_error(monkeypatch):
    def bad_client(*_args, **_kwargs):
        raise RuntimeError("no credentials")

    monkeypatch.setattr("nemantix.llm.aws_bedrock_proxy.boto3.client", bad_client)
    AbstractLLMProxy.set_credentials_manager(Credentials())

    with pytest.raises(
        LLMProxyException, match="Failed to initialize AWS Bedrock client"
    ):
        AWSBedrockLLMProxy("any-model")


def test_bedrock_client_error_on_invoke(bedrock_llm_proxy):
    from botocore.exceptions import ClientError

    def raise_client_error(**_kwargs):
        raise ClientError(
            {"Error": {"Code": "ValidationException", "Message": "bad request"}},
            "Converse",
        )

    bedrock_llm_proxy._client.converse = raise_client_error

    with pytest.raises(LLMProxyException, match="AWS Bedrock API Error"):
        bedrock_llm_proxy.invoke("Hello")


def test_bedrock_generic_error_on_invoke(bedrock_llm_proxy):
    def raise_generic(**_kwargs):
        raise RuntimeError("unexpected")

    bedrock_llm_proxy._client.converse = raise_generic

    with pytest.raises(LLMProxyException, match="Error invoking AWS Bedrock LLM"):
        bedrock_llm_proxy.invoke("Hello")


def test_bedrock_stream_error(bedrock_llm_proxy):
    def raise_on_stream(**_kwargs):
        raise RuntimeError("stream broken")

    bedrock_llm_proxy._client.converse_stream = raise_on_stream

    with pytest.raises(LLMProxyException, match="Error streaming from AWS Bedrock LLM"):
        list(bedrock_llm_proxy.stream("Hello"))
