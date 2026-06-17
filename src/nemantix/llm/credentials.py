import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from nemantix.common import get_package_logger

logger = get_package_logger(__name__)


class Credentials:
    """
    A class to manage LLM API credentials, loading them from environment variables.
    """

    def __init__(self):
        """
        Initializes the Credentials manager and loads variables from .env if present.
        """
        self._credentials_data: Dict[str, Any] = {}
        load_dotenv()

    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Retrieves an API key from environment variables and caches it.

        Args:
            key_name: The name of the API key to retrieve.

        Returns:
            The API key string if found, otherwise None.
        """
        if key_name in self._credentials_data:
            return self._credentials_data[key_name]

        env_var_name = key_name.upper()
        api_key = os.getenv(env_var_name)

        if api_key:
            self._credentials_data[key_name] = api_key
            return api_key

        logger.warning(
            f"Warning: '{key_name}' not found in environment variable '{env_var_name}' or .env file."
        )
        return None
