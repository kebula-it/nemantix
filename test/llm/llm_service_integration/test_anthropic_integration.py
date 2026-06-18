import os
import pytest

from nemantix.llm import AbstractLLMProxy
from nemantix.llm import Credentials
from nemantix.llm.anthropic_proxy import AnthropicLLMProxy
from nemantix.llm.abstract_proxy import LLMResponse, LLMUsage
from nemantix.core import tool, Toolset


# Define a real toolset for the integration test
class WeatherToolset(Toolset):
    @tool
    def get_current_weather(self, city: str) -> str:
        """
        Returns the current weather given the provided city.

        :param city: the city whose weather will be returned.
        """
        if city == "London":
            return "Rainy"
        return "sunny"


@pytest.fixture(scope="module")
def live_anthropic_proxy():
    """Fixture to provide a live, configured AnthropicLLMProxy instance."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    AbstractLLMProxy.set_credentials_manager(Credentials())
    proxy = AnthropicLLMProxy(model, max_output_tokens=128)
    return proxy


@pytest.mark.integration
def test_live_text_generation(live_anthropic_proxy):
    """
    Tests basic text generation with the live API.
    """
    prompt = "Reply with just 'ok'."
    response = live_anthropic_proxy.invoke(prompt)

    assert isinstance(response, LLMResponse)

    text_response = response.text
    assert isinstance(text_response, str)
    assert len(text_response) > 0
    assert "ok" in text_response.lower()

    assert not response.tool_calls

    assert isinstance(response.usage, LLMUsage)
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


@pytest.mark.integration
def test_live_function_call(live_anthropic_proxy):
    """
    Tests the full function calling flow with the live API.
    """
    # Bind the toolset using the new signature
    live_anthropic_proxy.bind_tools(WeatherToolset, ["get_current_weather"])

    prompt = "What's the weather in London right now?"
    response = live_anthropic_proxy.invoke(prompt)

    # Assert that the model correctly identified the tool to call
    assert isinstance(response, LLMResponse)

    tool_calls = response.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert "name" in tool_call
    assert "args" in tool_call
    assert tool_call["name"] == "get_current_weather"
    assert isinstance(tool_call["args"], dict)
    assert "city" in tool_call["args"]
    assert tool_call["args"]["city"] == "London"

    # Clean up tools for other tests
    live_anthropic_proxy.unbind_tools()


@pytest.mark.integration
def test_live_streaming_generation(live_anthropic_proxy):
    """
    Tests the full function calling flow with the live API.
    """
    prompt = "Reply with 'I am a Large Language model.'."

    text_response = ""
    for chunk in live_anthropic_proxy.stream(prompt):
        assert isinstance(chunk, str)
        text_response += chunk

    assert len(text_response) > 0
    assert "I am a Large Language model" in text_response
