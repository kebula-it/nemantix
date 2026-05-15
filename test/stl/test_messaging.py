import os
import pytest
from nemantix.stl.messaging.base import MessagingToolset
from nemantix.core import Toolset

# Define paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming bot_config.json is in the same directory as the test file,
# or adjust this path relative to BASE_DIR if it sits elsewhere (e.g., "../src/bot_config.json")
BOT_CONFIG_PATH = os.path.join(BASE_DIR, "bot_config.json")


@pytest.fixture
def bot_config():
    """
    Checks for the configuration file. If it doesn't exist, skips the test.
    Returns the path to the configuration file.
    """
    if not os.path.exists(BOT_CONFIG_PATH):
        pytest.skip(f"File {BOT_CONFIG_PATH} missing. Skipping live messaging tests.")
    return BOT_CONFIG_PATH


class TestMessagingToolsetInit:
    def test_init_tool(self, bot_config):
        """Test initialization of tool."""
        ts = Toolset.get_tool(
            tool_name="MessagingToolset.send_message", instance_args=(bot_config,)
        )
        assert callable(ts)


@pytest.mark.external
class TestMessagingSendMessage:
    def test_send_message(self, bot_config):
        ts = Toolset.get_tool(
            tool_name="MessagingToolset.send_message", instance_args=(bot_config,)
        )
        result = ts(chat_id="916483163", text="Hello World!")
        assert "Successfully sent to ID" in str(result)

    def test_send_message_error(self, bot_config):
        ts = Toolset.get_tool(
            tool_name="MessagingToolset.send_message", instance_args=(bot_config,)
        )
        result = ts(chat_id="", text="Hello World!")
        assert "Error sending to" in str(result)


@pytest.mark.external
class TestMessagingChat:
    def test_get_chat(self, bot_config):
        ts = Toolset.get_tool(
            tool_name="MessagingToolset.get_chat_id", instance_args=(bot_config,)
        )
        result = ts()
        assert "Error fetching updates:" not in str(result)
