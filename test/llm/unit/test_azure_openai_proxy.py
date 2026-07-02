from typing import List
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from nemantix.core import Toolset, tool
from nemantix.llm.abstract_proxy import (
    AbstractLLMProxy,
    LLMProxyException,
    LLMResponse,
    StructuredLLMResponse,
)
from nemantix.llm.azure_openai_proxy import AzureOpenAILLMProxy
from nemantix.llm.credentials import Credentials

# =============================================================================
# Mocks & Fixtures
# =============================================================================


@pytest.fixture
def mock_azure_openai_client(monkeypatch):
    """
    A robust mock for AzureOpenAI that simulates standard responses,
    streaming chunks, and tool calling sequences.
    """

    class MockMessage:
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

        def model_dump(self, **kwargs):
            return {
                "role": "assistant",
                "content": self.content,
                "tool_calls": self.tool_calls,
            }

        # Support dict() casting used in azure_openai_proxy.py
        def keys(self):
            return ["content", "tool_calls"]

        def __getitem__(self, key):
            return getattr(self, key)

    class MockDelta:
        def __init__(self, content):
            self.content = content

    class MockChoice:
        def __init__(self, message=None, delta=None):
            self.message = message
            self.delta = delta

    class MockChunk:
        """Simulates an OpenAI ChatCompletionChunk"""

        def __init__(self, choices):
            self.choices = choices

    class MockUsage:
        def __init__(self):
            self.prompt_tokens = 10
            self.completion_tokens = 20
            self.prompt_tokens_details = MagicMock(cached_tokens=0)

    class MockStreamContextManager:
        """A simple context manager to mock the streaming response."""

        def __init__(self, events):
            self.events = events

        def __enter__(self):
            return self.events

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockCompletions:
        def create(self, **kwargs):
            messages = kwargs.get("messages", [])

            # Streaming mode (Yields a MockChunk with choices)
            if kwargs.get("stream"):
                chunk = MockChunk(
                    choices=[MockChoice(delta=MockDelta("Mock stream response."))]
                )
                return MockStreamContextManager([chunk])

            # Structured output mode
            if kwargs.get("response_format"):
                return MagicMock(
                    choices=[MockChoice(message=MockMessage('{"result": "mocked"}'))],
                    usage=MockUsage(),
                )

            # Tool use mode
            if kwargs.get("tools"):
                if any(m.get("role") == "tool" for m in messages):
                    return MagicMock(
                        choices=[
                            MockChoice(
                                message=MockMessage("Weather is sunny in Boston.")
                            )
                        ],
                        usage=MockUsage(),
                    )

                tc = MagicMock()
                tc.type = "function"
                tc.function.name = "get_current_weather"
                tc.function.arguments = '{"location": "Boston"}'
                tc.id = "call_123"
                return MagicMock(
                    choices=[MockChoice(message=MockMessage(None, tool_calls=[tc]))],
                    usage=MockUsage(),
                )

            # Standard text mode
            msg_text = str(messages[-1]["content"]) if messages else ""
            return MagicMock(
                choices=[
                    MockChoice(message=MockMessage(f"Mock response to: {msg_text}"))
                ],
                usage=MockUsage(),
            )

    class MockChat:
        def __init__(self):
            self.completions = MockCompletions()

    class MockAzureOpenAI:
        def __init__(self, **kwargs):
            self.chat = MockChat()

    monkeypatch.setattr("nemantix.llm.azure_openai_proxy.AzureOpenAI", MockAzureOpenAI)
    return MockAzureOpenAI


@pytest.fixture
def azure_proxy(mock_azure_openai_client, monkeypatch):
    """
    Initializes the AzureOpenAILLMProxy using environment variables.
    """
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-azure-key")
    AbstractLLMProxy.set_credentials_manager(Credentials())

    return AzureOpenAILLMProxy(
        deployment_name="gpt-4o",
        api_version="2024-02-15-preview",
        azure_endpoint="https://fake-endpoint.openai.azure.com/",
    )


class LocationData(BaseModel):
    city: str
    zip_code: int


# =============================================================================
# Tests
# =============================================================================


def test_azure_openai_init_and_invoke_stream_bind_unbind(azure_proxy):
    """
    This test accepts the `azure_openai_llm_proxy` fixture (from conftest.py).
    """
    proxy = azure_proxy

    # unbound invoke
    out = proxy.invoke("Hello")
    assert isinstance(out, LLMResponse)

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

    # Bind both simple and complex tools
    proxy2 = proxy.bind_tools(
        DummyToolset, ["get_current_weather", "get_complex_weather"]
    )

    # --- NEW ASSERTIONS TO TEST COMPLEX SCHEMA GENERATION ---
    bound_tools = proxy2._bound_tools

    # Isolate the complex tool payload
    complex_tool = next(
        t for t in bound_tools if t["function"]["name"] == "get_complex_weather"
    )
    props = complex_tool["function"]["parameters"]["properties"]

    # 1. Verify Pydantic model mapping (loc_data)
    assert props["loc_data"]["type"] == "object"
    assert "city" in props["loc_data"]["properties"]
    assert props["loc_data"]["properties"]["city"]["type"] == "string"
    assert "zip_code" in props["loc_data"]["properties"]
    assert props["loc_data"]["properties"]["zip_code"]["type"] == "integer"

    # 2. Verify List mapping (tags)
    assert props["tags"]["type"] == "array"
    assert props["tags"]["items"]["type"] == "string"
    # --------------------------------------------------------

    # Test standard invocation with the mocked client
    out2 = proxy2.invoke("What's the weather? And stock?")
    assert "get_current_weather" in [tc["name"] for tc in out2.tool_calls]
    assert out2.tool_calls[0]["args"] == {"location": "Boston"}

    # streaming yields characters
    chunks = list(proxy.stream("abc"))
    assert "".join(chunks) == "Mock stream response."

    # unbind
    proxy3 = proxy.unbind_tools()
    out3 = proxy3.invoke("What's the weather? And stock?")
    assert out3.tool_calls == []


def test_azure_openai_errors_surface(monkeypatch):
    """Ensure instantiation errors are safely wrapped in LLMProxyException."""

    def bad_ctor(**kwargs):
        raise RuntimeError("boom")

    # Patch the resolved symbol within the azure_openai_proxy module
    monkeypatch.setattr("nemantix.llm.azure_openai_proxy.AzureOpenAI", bad_ctor)

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-azure-key")
    AbstractLLMProxy.set_credentials_manager(Credentials())

    with pytest.raises(
        LLMProxyException, match="Failed to initialize Azure OpenAI client: boom"
    ):
        AzureOpenAILLMProxy(
            deployment_name="gpt-4o",
            api_version="2024-02-15-preview",
            azure_endpoint="https://fake.endpoint",
        )


def test_azure_stream_accepts_list_prompt(azure_proxy):
    messages = [{"role": "user", "content": "abc"}]
    chunks = list(azure_proxy.stream(messages))
    assert "".join(chunks) == "Mock stream response."


def test_azure_invoke_structured_accepts_list_prompt(azure_proxy):
    from pydantic import BaseModel

    class Reply(BaseModel):
        result: str = ""

    messages = [{"role": "user", "content": "hello"}]
    result = azure_proxy.invoke_structured(messages, schema=Reply)
    assert isinstance(result, StructuredLLMResponse)


def test_azure_openai_grammar_based_unsupported_model(azure_proxy):
    """
    Ensure invoke_grammar_based rejects deployments that do not begin with 'gpt-5'.
    """
    with pytest.raises(
        NotImplementedError,
        match="invoke_grammar_based is not supported on this deployment",
    ):
        azure_proxy.invoke_grammar_based("Hello")
