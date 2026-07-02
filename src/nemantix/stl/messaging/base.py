import os
import json
import requests
from typing import Optional
from nemantix.core import tool, Toolset


class MessagingToolset(Toolset):
    """
    A toolset for interacting with Telegram Bots.
    Allows sending messages to specific users via their unique Chat ID.
    """

    def __init__(
        self, config_path: Optional[str] = None, bot_token: Optional[str] = None
    ):
        """
        Initializes the MessagingToolset. It can load the token directly or from a JSON file.

        Args:
            config_path (str, optional): The file path to a JSON configuration file containing the 'bot_token'.
            bot_token (str, optional): The direct Telegram bot token. If config_path is provided, this is ignored.

        Example calls:
            # From JSON:
            MessagingToolset(config_path="config.json")

            # Direct token:
            MessagingToolset(bot_token="YOUR_TOKEN_HERE")
        """
        super().__init__()

        # If a JSON config file path is provided, extract the bot_token from it
        if config_path:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                bot_token = config.get("bot_token", bot_token)

        # Ensure we actually have a token before proceeding
        if not bot_token:
            raise ValueError(
                "A bot_token must be provided either directly or via a valid config_path JSON."
            )

        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    @tool
    def send_message(self, chat_id: str, text: str) -> str:
        """
        Send a text message to a specific Telegram account.

        Args:
            chat_id (str): The unique numeric ID of the destination account (e.g., "987654321").
            text (str): The content of the message.

        Returns:
            str: Confirmation message.

        Example call:
            send_message(
                chat_id="123456789",
                text="Hello! This is a message for you."
            )
        """
        url = f"{self.base_url}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}

        try:
            response = requests.post(url, json=payload, timeout=10)
            data = response.json()

            if data.get("ok"):
                return f"Successfully sent to ID {chat_id}."
            else:
                return f"Error sending to {chat_id}: {data.get('description')}"

        except Exception as e:
            return f"Network error: {str(e)}"

    @tool
    def get_chat_id(self) -> str:
        """
        Check recent messages to find the Chat ID of users who messaged the bot.
        Use this to find the 'username' needed to send messages to them.

        Returns:
            str: A list of users and their Chat IDs.

        Example call:
            get_chat_id()
        """
        url = f"{self.base_url}/getUpdates"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()

            if not data.get("ok"):
                return f"Error: {data.get('description')}"

            updates = data["result"]
            if not updates:
                return "No new messages found. Send a message to the bot first."

            results = []
            seen_ids = set()

            for u in reversed(updates):
                if "message" in u:
                    cid = u["message"]["chat"]["id"]
                    if cid not in seen_ids:
                        sender = u["message"]["from"].get("username", "Unknown")
                        results.append(f"User: @{sender} -> Chat ID: {cid}")
                        seen_ids.add(cid)

            return "\n".join(results)
        except Exception as e:
            return f"Error fetching updates: {str(e)}"
