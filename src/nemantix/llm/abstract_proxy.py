import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Type, Union

from pydantic import BaseModel

from nemantix.llm.credentials import Credentials

if TYPE_CHECKING:
    from nemantix.core.tools import Toolset


@dataclass
class LLMUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class LLMResponse:
    text: str
    tool_calls: List[Dict[str, Any]]
    usage: LLMUsage
    proxy: "AbstractLLMProxy"


@dataclass
class StructuredLLMResponse:
    result: BaseModel
    usage: LLMUsage
    proxy: "AbstractLLMProxy"


class LLMProxyException(Exception):
    """Custom exception for LLM proxy-related errors."""

    pass


class AbstractLLMProxy(abc.ABC):
    """
    Abstract base class for LLM proxies.
    Defines a common interface for interacting with different LLM vendors.
    Manages a static Credentials manager for all proxy instances.
    """

    _credentials_manager: Optional[Credentials] = None

    @abc.abstractmethod
    def get_name(self) -> str:
        """Returns the LMM proxy name"""
        pass

    @staticmethod
    def set_credentials_manager(manager: Credentials):
        """
        Sets the global Credentials manager for all LLM proxies.
        This should be called once at the application's startup.
        """
        AbstractLLMProxy._credentials_manager = manager

    @classmethod
    def is_credential_manager_set(cls) -> bool:
        return cls._credentials_manager is not None

    @staticmethod
    def _get_api_key(
        key_name: str, required: bool = True, **kwargs: Any
    ) -> Optional[str]:
        """
        Retrieves the API key from kwargs, then from the Credentials manager.

        It first checks `kwargs` for a generic "api_key" parameter or a
        vendor-specific key parameter matching `key_name`. If not found,
        it attempts to retrieve it from the globally set credentials manager.

        Args:
            key_name (str): The specific name of the API key to retrieve
                (e.g., "openai_api_key", "google_api_key").
            required (bool): Whether to raise an exception if the key is not found.
                Defaults to True.
            **kwargs (Any): Additional keyword arguments that might contain the API key.

        Returns:
            Optional[str]: The retrieved API key as a string, or None if the key
                is not found and `required` is False.

        Raises:
            LLMProxyException: If the credentials manager has not been initialized.
            LLMProxyException: If `required` is True and the API key is not found
                in kwargs, the credentials file, or environment variables.
        """
        if AbstractLLMProxy._credentials_manager is None:
            raise LLMProxyException(
                "Credentials manager not set. Call AbstractLLMProxy.set_credentials_manager() first."
            )

        # 1) explicit kwargs
        api_key = kwargs.pop("api_key", None)
        vendor_key_param = kwargs.pop(key_name, None)
        if not api_key:
            api_key = vendor_key_param

        # 2) credentials manager (file/env)
        if not api_key:
            api_key = AbstractLLMProxy._credentials_manager.get_api_key(key_name)

        if required and not api_key:
            raise LLMProxyException(
                f"{key_name.replace('_', ' ').title()} is required but not found in parameters, credentials file, or environment variables."
            )
        return api_key

    @abc.abstractmethod
    def invoke(self, prompt: str | list, **kwargs: Any) -> LLMResponse:
        """
        Invokes the LLM with a given prompt, automatically using tools if they have
        been previously bound to the proxy.

        Args:
            prompt: The input prompt string or a list of message dicts for multi-turn conversations.
            **kwargs: Vendor-model specific parameters.

        Returns:
            LLMResponse with text, tool_calls, and usage fields.
        """
        pass

    @abc.abstractmethod
    def invoke_structured(
        self, prompt: str | list, schema: Type[BaseModel], **kwargs
    ) -> StructuredLLMResponse:
        """
        Invokes the LLM with structured output using vendor-specific implementations.

        Each vendor should implement this using their best approach:
        - OpenAI: response_format with json_schema
        - Google: response_schema with response_mime_type
        - Others: may fall back to function calling or prompt engineering

        Args:
            prompt: The input prompt string.
            schema: The Pydantic model schema for structured output.

        Returns:
            StructuredLLMResponse with result (validated Pydantic model) and usage fields.
        """
        pass

    @abc.abstractmethod
    def invoke_grammar_based(
        self, prompt: Union[str, list], **kwargs: Any
    ) -> LLMResponse:
        """
        Invokes the LLM with a grammar-based tool to process the input prompt using a custom grammar.

        Args:
            prompt: The input prompt string or a list of message dicts for multi-turn conversations.
            **kwargs (Any): Additional vendor-specific parameters or overrides to be passed to the LLM.

        Returns:
            LLMResponse with text, tool_calls, and usage fields.

        Raises:
            LLMProxyException: If there is an error invoking the LLM or any other issue during the request.
        """
        pass

    @abc.abstractmethod
    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """
        Streams the LLM's response token by token.
        This method is primarily for text-only streaming.

        Args:
            prompt: The input prompt string.
            **kwargs: Vendor-model specific parameters.

        Returns:
            An iterator that yields response chunks as strings.
        """
        pass

    @abc.abstractmethod
    def supports_tool_use(self) -> bool:
        """
        Returns True if the LLM can make use of tools (function calling), False otherwise.

        Returns:
            bool: True if tool use is supported, False otherwise.
        """
        pass

    @abc.abstractmethod
    def bind_tools(
        self, toolset_class: Type["Toolset"], tool_names: List[str]
    ) -> "AbstractLLMProxy":
        """
        Binds a list of tools from a Toolset to the LLM. This modifies the proxy instance
        to include these tools for all subsequent invocations.

        Args:
            toolset_class: The Toolset subclass containing the tool definitions.
            tool_names: A list of tool names (strings) to bind.

        Returns:
            AbstractLLMProxy: The proxy instance with tools bound.
        """
        pass

    @abc.abstractmethod
    def unbind_tools(self) -> "AbstractLLMProxy":
        """
        Reverts the LLM proxy to its initial state without any bound tools.

        Returns:
            AbstractLLMProxy: The proxy instance with tools unbound.
        """
        pass

    @abc.abstractmethod
    def messages_from(
        self, prompts_with_roles: list[dict[str, str] | tuple[str, str]]
    ) -> list[dict]:
        """Formats the prompts (and roles) according to the specific message format"""
        pass
