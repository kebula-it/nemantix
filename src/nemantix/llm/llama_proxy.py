from typing import Any, Optional, Union

from nemantix.common.logger import get_package_logger
from nemantix.llm.abstract_proxy import LLMProxyException, LLMResponse
from nemantix.llm.openai_proxy import OpenAICompatibleProxy

logger = get_package_logger(__name__)


class LlamaProxy(OpenAICompatibleProxy):
    """
    LLM proxy for models served via Ollama (local inference).
    Leverages Ollama's native OpenAI-compatible /v1 endpoint.
    """

    def __init__(
        self,
        model_name: str,
        # Ollama native OpenAI-compatible endpoint
        host: Optional[str] = "http://localhost:11434",
        **kwargs: Any,
    ):
        # Ensure the host is properly formatted with the OpenAI /v1 compatibility suffix
        base_url = (host or '').rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"

        # Initialize the base OpenAI proxy with Ollama's local URL
        super().__init__(
            model_name=model_name,
            api_key_name="ollama_api_key",  # defaults to "no-key-required" if not set
            base_url=base_url,
            **kwargs,
        )

    def get_name(self) -> str:
        return f"Ollama ({self.model_name})"

    # ----------------------------- Interface Overrides -----------------------------

    def invoke_grammar_based(
        self, prompt: Union[str, list], **kwargs: Any
    ) -> LLMResponse:
        """
        Ollama does not support custom grammar injection via the OpenAI compatibility layer.
        """
        raise LLMProxyException(
            "Grammar-based invocation is not natively supported by Ollama's OpenAI endpoint."
        )
