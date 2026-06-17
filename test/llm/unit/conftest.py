from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Iterator, List

import pytest
from google import genai

from nemantix.llm import AbstractLLMProxy, Credentials
from nemantix.llm.anthropic_proxy import AnthropicLLMProxy
from nemantix.llm.google_proxy import GoogleLLMProxy
from nemantix.llm.openai_proxy import OpenAILLMProxy


class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=False):
        d = {"content": self.content, "tool_calls": self.tool_calls}
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


@pytest.fixture
def mock_google_client():
    @dataclass
    class MockUsageMetadata:
        prompt_token_count: int = 10
        candidates_token_count: int = 5
        cached_content_token_count: int = 0

    @dataclass
    class MockResponse:
        text: str
        parts: Any
        usage_metadata: MockUsageMetadata = None

        def __post_init__(self):
            if self.usage_metadata is None:
                self.usage_metadata = MockUsageMetadata()

    class MockModels:
        @staticmethod
        def generate_content(*args, **kwargs):
            contents = kwargs.get("contents")

            # --- Safely extract plain text from the normalized messages list ---
            prompt_text = ""
            if isinstance(contents, list):
                for msg in contents:
                    for part in msg.get("parts", []):
                        prompt_text += part.get("text", "")
            elif isinstance(contents, str):
                prompt_text = contents
            # -------------------------------------------------------------------

            if prompt_text:
                config = kwargs.get("config", {})
                tools = getattr(config, "tools", [])

                # Check the extracted string instead of the raw list
                if "tempo" in prompt_text and tools:
                    return MockResponse(
                        "",
                        [
                            Namespace(
                                function_call=Namespace(
                                    name="get_current_weather", args=[]
                                )
                            )
                        ],
                    )
                return MockResponse(prompt_text, [])
            return MockResponse("", [])

        @staticmethod
        def generate_content_stream(*args, **kwargs):
            return iter([Namespace(text="ok1"), Namespace(text="ok2")])

    # Create a MagicMock for the genai.Client
    class MockClient:
        models = MockModels

        def __init__(self, *args, **kwargs):
            pass

    # Return the mocked client
    return MockClient


@pytest.fixture
def google_llm_proxy(mock_google_client, monkeypatch):
    # Mock the actual GoogleLLMProxy's _client to use the mock_google_client
    monkeypatch.setattr(genai, "Client", mock_google_client)

    # Optionally, mock the API key if needed
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-api-key")
    AbstractLLMProxy.set_credentials_manager(Credentials())

    # Return the proxy instance
    return GoogleLLMProxy(
        "gemini-2.5-flash", temperature=0.2, max_output_tokens=50, top_k=10, top_p=0.9
    )


@pytest.fixture
def mock_openai_client():
    """A more accurate mock for the openai.OpenAI client."""

    @dataclass
    class MockStreamChoice:
        delta: Namespace

    @dataclass
    class MockStreamEvent:
        choices: List[MockStreamChoice]

    @dataclass
    class MockUsage:
        prompt_tokens: int = 10
        completion_tokens: int = 5
        prompt_tokens_details: Any = None

    @dataclass
    class MockChoice:
        message: Namespace

    @dataclass
    class MockResponse:
        choices: List[MockChoice]
        usage: MockUsage = None

        def __post_init__(self):
            if self.usage is None:
                self.usage = MockUsage()

    def mock_create(**kwargs):
        """This function mimics the client.chat.completions.create method."""
        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            # Simulate a streaming response
            def stream_generator() -> Iterator[MockStreamEvent]:
                yield MockStreamEvent(
                    choices=[MockStreamChoice(delta=Namespace(content="Mock "))]
                )
                yield MockStreamEvent(
                    choices=[MockStreamChoice(delta=Namespace(content="stream "))]
                )
                yield MockStreamEvent(
                    choices=[MockStreamChoice(delta=Namespace(content="response."))]
                )

            # The real SDK returns a context manager for streaming
            class MockStreamContextManager:
                def __enter__(self):
                    return stream_generator()

                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass

            return MockStreamContextManager()

        # Simulate a non-streaming (invoke) response
        prompt = kwargs.get("messages", [{}])[0].get("content", "")
        tools = kwargs.get("tools")

        # Check for tool call trigger
        if tools and ("weather" in prompt.lower() or "stock" in prompt.lower()):
            tool_call = Namespace(
                id="call_123",
                type="function",
                function=Namespace(
                    name="get_current_weather", arguments='{"location": "Boston"}'
                ),
            )
            message = MockMessage(content=None, tool_calls=[tool_call])
        else:
            # Regular text response
            message = MockMessage(content=f"Mock response to: {prompt}", tool_calls=[])

        return MockResponse(choices=[MockChoice(message=message)])

    # noinspection PyUnusedLocal
    class MockOpenAI:
        def __init__(self, **kwargs):
            # The chat.completions.create path is what we need to mock
            self.chat = Namespace(completions=Namespace(create=mock_create))

    return MockOpenAI


@pytest.fixture
def openai_llm_proxy(mock_openai_client, monkeypatch):
    # Patch the OpenAI class that OpenAILLMProxy imports and uses
    monkeypatch.setattr("nemantix.llm.openai_proxy.OpenAI", mock_openai_client)
    monkeypatch.setenv("OPENAI_API_KEY", "mock-api-key")
    AbstractLLMProxy.set_credentials_manager(Credentials())
    return OpenAILLMProxy("gpt-4o-mini", temperature=0.2, max_output_tokens=42)


@pytest.fixture
def mock_anthropic_client():
    """A mock for the anthropic.Anthropic client."""

    @dataclass
    class MockTextBlock:
        text: str
        type: str = "text"

    @dataclass
    class MockToolUseBlock:
        name: str
        input: dict
        type: str = "tool_use"

    @dataclass
    class MockUsage:
        input_tokens: int = 10
        output_tokens: int = 5

    @dataclass
    class MockResponse:
        content: list
        usage: MockUsage = None

        def __post_init__(self):
            if self.usage is None:
                self.usage = MockUsage()

    class MockStreamContextManager:
        def __init__(self):
            # Simulate Anthropic's text_stream generator
            self.text_stream = iter(["Mock ", "stream ", "response."])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockMessages:
        def create(self, **kwargs):
            messages = kwargs.get("messages", [])
            prompt = messages[0].get("content", "") if messages else ""
            tools = kwargs.get("tools")

            # Anthropic tool choice checking (for invoke_structured)
            tool_choice = kwargs.get("tool_choice")
            if tool_choice and tool_choice.get("type") == "tool":
                forced_tool = tool_choice.get("name")
                block = MockToolUseBlock(
                    name=forced_tool, input={"mocked_field": "mocked_value"}
                )
                return MockResponse(content=[block])

            # Standard tool call trigger
            if tools and ("weather" in prompt.lower() or "stock" in prompt.lower()):
                block = MockToolUseBlock(
                    name="get_current_weather", input={"location": "Boston"}
                )
                return MockResponse(content=[block])
            else:
                # Regular text response
                block = MockTextBlock(text=f"Mock response to: {prompt}")
                return MockResponse(content=[block])

        def stream(self, **kwargs):
            return MockStreamContextManager()

    # noinspection PyUnusedLocal
    class MockAnthropic:
        def __init__(self, **kwargs):
            self.messages = MockMessages()

    return MockAnthropic


@pytest.fixture
def anthropic_llm_proxy(mock_anthropic_client, monkeypatch):
    # Patch the Anthropic class that AnthropicLLMProxy imports and uses
    monkeypatch.setattr(
        "nemantix.llm.anthropic_proxy.anthropic.Anthropic", mock_anthropic_client
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-api-key")
    AbstractLLMProxy.set_credentials_manager(Credentials())
    return AnthropicLLMProxy(
        "claude-3-5-sonnet-20241022", temperature=0.2, max_output_tokens=42
    )


@pytest.fixture(autouse=True)
def clear_env_and_state(monkeypatch):
    # Reset credentials manager between tests
    import nemantix.llm.abstract_proxy as ap

    ap.AbstractLLMProxy._credentials_manager = None
    yield
