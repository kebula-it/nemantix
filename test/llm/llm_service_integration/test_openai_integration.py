import os
import pytest

from conftest import CREDENTIALS_PATH
from nemantix.llm import AbstractLLMProxy
from nemantix.llm import Credentials
from nemantix.llm.openai_proxy import OpenAILLMProxy
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
def live_openai_proxy():
    """Fixture to provide a live, configured OpenAILLMProxy instance."""
    if not os.path.exists(CREDENTIALS_PATH):
        pytest.skip("Credentials file 'credentials.json' not found")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    AbstractLLMProxy.set_credentials_manager(
        Credentials.load_from_file(file_path=str(CREDENTIALS_PATH))
    )
    proxy = OpenAILLMProxy(model, max_output_tokens=128)
    return proxy


@pytest.mark.integration
def test_live_text_generation(live_openai_proxy):
    """
    Tests basic text generation with the live API.
    """
    prompt = "Reply with just 'ok'."
    response = live_openai_proxy.invoke(prompt)

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
def test_live_function_call(live_openai_proxy):
    """
    Tests the full function calling flow with the live API.
    """
    # Bind the toolset using the new signature
    live_openai_proxy.bind_tools(WeatherToolset, ["get_current_weather"])

    prompt = "What's the weather in London right now?"
    response = live_openai_proxy.invoke(prompt)

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
    live_openai_proxy.unbind_tools()


@pytest.mark.integration
def test_live_streaming_generation(live_openai_proxy):
    """
    Tests streaming generation with the live API.
    """
    prompt = "Reply with 'I am a Large Language model.'."

    text_response = ""
    for chunk in live_openai_proxy.stream(prompt):
        assert isinstance(chunk, str)
        text_response += chunk

    assert len(text_response) > 0
    assert "I am a Large Language model" in text_response
