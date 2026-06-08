from typing import Any, Dict, List, Type, Union

from pydantic import BaseModel

from nemantix.common.logger import get_package_logger
from nemantix.llm.abstract_proxy import (
    LLMProxyException,
    LLMResponse,
    StructuredLLMResponse,
)
from nemantix.llm.openai_proxy import OpenAICompatibleProxy

logger = get_package_logger(__name__)


class LlamaCppRemoteLLMProxy(OpenAICompatibleProxy):
    """
    LLM proxy for a llama.cpp server instance utilizing the official OpenAI Python SDK.
    Connects seamlessly to a remote or local llama-server REST endpoint.
    """

    def __init__(
        self,
        model_name: str | None = "auto",
        base_url: str = "http://localhost:8080/v1",
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name or "auto",
            api_key_name="llamacpp_api_key",
            base_url=base_url,
            **kwargs,
        )

        # Auto-fetch model name if not provided
        if self.model_name == "auto":
            try:
                available_models = self._client.models.list()
                if available_models.data:
                    self.model_name = available_models.data[0].id
                else:
                    self.model_name = "default-model"

                logger.info(f'Using model "{self.model_name}"')

            except Exception as err:
                logger.warning(f"Could not auto-fetch model name: {err}")
                self.model_name = "default-model"

    def get_name(self) -> str:
        return f"Llama.cpp ({self.model_name})"

    def invoke(
        self, prompt: Union[str, list], tool_choice="auto", **kwargs: Any
    ) -> LLMResponse:
        messages = self._flatten_messages(prompt)
        return super().invoke(prompt=messages, tool_choice=tool_choice, **kwargs)

    def invoke_structured(
        self, prompt: Union[str, list], schema: Type[BaseModel], **kwargs
    ) -> StructuredLLMResponse:
        messages = self._flatten_messages(prompt)
        return super().invoke_structured(prompt=messages, schema=schema, **kwargs)

    def invoke_grammar_based(
        self, prompt: Union[str, list], **kwargs: Any
    ) -> LLMResponse:
        raise LLMProxyException("LARL Grammar invocation is not natively "
                                "supported by llama.cpp's server endpoint.")

    def _flatten_messages(self, prompt: Union[str, list]) -> List[Dict[str, Any]]:
        """
        Extracts and flattens rich content arrays into plain strings.
        llama-server throws 'unsupported content[].type' if it sees complex objects.
        """
        if isinstance(prompt, str):
            return self.messages_from([("user", prompt)])

        safe_messages = []
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Flatten array structures (e.g., OpenAI multimodal content blocks)
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        texts.append(item)
                content = "\n".join(texts)

            safe_messages.append({"role": role, "content": str(content)})

        return safe_messages
