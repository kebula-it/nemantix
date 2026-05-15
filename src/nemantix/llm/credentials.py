# llm/credentials.py

import os
import json

from typing import Dict, Any, Optional
from nemantix.common import get_package_logger

logger = get_package_logger(__name__)


class Credentials:
    """
    A class to manage LLM API credentials, loading them from a file
    and falling back to environment variables if necessary.
    """

    def __init__(self, file_path: str = "credentials.json"):
        """
        Initializes the Credentials manager.

        Args:
            file_path: The path to the JSON credentials file.
        """
        self._credentials_data: Dict[str, Any] = {}
        self._file_path = file_path
        self._load_from_file()

    def _load_from_file(self):
        """
        Attempts to load credentials from the specified JSON file.
        Updates the internal dictionary.
        """
        try:
            with open(self._file_path, "r") as f:
                file_credentials = json.load(f)
            # Convert keys to lowercase for consistency
            self._credentials_data = {k.lower(): v for k, v in file_credentials.items()}
            logger.debug(f"Credentials loaded successfully from '{self._file_path}'.")

        except FileNotFoundError:
            logger.info(
                f"Info: Credentials file '{self._file_path}' not found. "
                f"Will attempt to load from environment variables when requested."
            )

        except json.JSONDecodeError:
            logger.info(
                f"Warning: Could not decode JSON from '{self._file_path}'. "
                f"Please check the file format. Will attempt to load from environment "
                f"variables when requested."
            )

        except Exception as e:
            logger.error(
                f"An unexpected error occurred while reading '{self._file_path}': {e}. "
                f"Will attempt to load from environment variables when requested.",
                exc_info=True,
            )

    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Retrieves an API key. It first checks the loaded file credentials,
        then falls back to environment variables, and stores it in the internal
        dictionary if found from environment variables.

        Args:
            key_name: The name of the API key to retrieve (e.g., "openai_api_key", "google_api_key").

        Returns:
            The API key string if found, otherwise None.
        """
        # 1. Check if already in the loaded data (from file or previously from env)
        api_key = self._credentials_data.get(key_name)
        if api_key:
            return api_key

        # 2. If not found, try environment variable
        env_var_name = (
            key_name.upper()
        )  # Convert to uppercase for common env var naming
        api_key = os.getenv(env_var_name)
        if api_key:
            self._credentials_data[key_name] = api_key  # Store for future access
            # print(f"'{key_name}' loaded from environment variable '{env_var_name}'.")
            return api_key
        else:
            logger.warning(
                f"Warning: '{key_name}' not found in credentials file or environment "
                f"variable '{env_var_name}'."
            )
            return None

    @staticmethod
    def load_from_file(file_path: str = "credentials.json") -> "Credentials":
        """
        Static method to create and return an instance of the Credentials class,
        loading credentials from the specified file.
        """
        return Credentials(file_path=file_path)
