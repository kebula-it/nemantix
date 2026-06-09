import inspect
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Type

from pydantic import BaseModel

from .abstract_proxy import (
    AbstractLLMProxy,
    LLMProxyException,
    LLMResponse,
    LLMUsage,
    StructuredLLMResponse,
)

if TYPE_CHECKING:
    from ..core.tools import Toolset

try:
    import anthropic
except ImportError as e:
    raise ImportError(
        "Please install the official Anthropic SDK: `pip install anthropic>=0.18.0`"
    ) from e


class AnthropicLLMProxy(AbstractLLMProxy):
    """
    LLM proxy for Anthropic models (e.g., Claude Sonnet, Opus) using the official SDK.
    Keeps the same public interface as AbstractLLMProxy.
    """

    def __init__(
            self,
            model_name: str,
            temperature: Optional[float] = None,
            max_output_tokens: Optional[int] = None,
            base_url: Optional[str] = None,
            **kwargs: Any,
    ):
        self.model_name = model_name
        self._bound_tools: List[Dict[str, Any]] = []

        api_key = self._get_api_key("anthropic_api_key", **kwargs)

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        try:
            self._client = anthropic.Anthropic(**client_kwargs)
        except Exception as e:
            raise LLMProxyException(
                f"Failed to initialize Anthropic client: {e}"
            ) from e

        # Anthropic *requires* max_tokens for its Messages API.
        # We default to 4096 if not provided.
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens or 4096

    # ----------------------------- helpers -----------------------------

    @staticmethod
    def _build_usage(u) -> LLMUsage:
        return LLMUsage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )

    @staticmethod
    def _map_parameter_type(parameter: inspect.Parameter) -> str:
        ann_type = parameter.annotation
        if ann_type is str:
            return "string"
        if ann_type is int:
            return "integer"
        if ann_type is bool:
            return "boolean"
        if ann_type is float:
            return "number"
        return "string"

    # ----------------------------- interface -----------------------------

    def get_name(self) -> str:
        return f'Anthropic {self.model_name}'

    def invoke(self, prompt: str | list, **kwargs: Any) -> LLMResponse:
        try:
            # Handle prompt as string or list of messages
            messages = (
                [{"role": "user", "content": prompt}]
                if isinstance(prompt, str)
                else prompt
            )

            req: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self._max_output_tokens,
            }
            if self._temperature is not None:
                req["temperature"] = self._temperature
            if self._bound_tools:
                req["tools"] = self._bound_tools

            req.update(kwargs)

            response = self._client.messages.create(**req)

            text = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    text += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        {
                            "name": block.name,
                            "args": block.input,  # Anthropic returns this as a dict automatically
                        }
                    )

            return LLMResponse(text=text.strip(), tool_calls=tool_calls,
                               usage=self._build_usage(response.usage), proxy=self)
        except Exception as e:
            raise LLMProxyException(f"Error invoking Anthropic LLM: {e}") from e

    def invoke_structured(self, prompt: str, schema: Type[BaseModel], **kwargs) -> StructuredLLMResponse:
        """
        Uses Anthropic's tool choice forcing to guarantee the output matches
        the provided Pydantic schema.
        """
        tool_name = schema.__name__
        tool_schema = {
            "name": tool_name,
            "description": f"Output schema for {tool_name}",
            "input_schema": schema.model_json_schema(),
        }

        try:
            req: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self._max_output_tokens,
                "tools": [tool_schema],
                "tool_choice": {
                    "type": "tool",
                    "name": tool_name,
                },  # Force the model to use this tool
            }
            if self._temperature is not None:
                req["temperature"] = self._temperature

            response = self._client.messages.create(**req)

            # Find the forced tool use block
            for block in response.content:
                if block.type == "tool_use" and block.name == tool_name:
                    return StructuredLLMResponse(
                        result=schema.model_validate(block.input),
                        usage=self._build_usage(response.usage),
                        proxy=self,
                    )

            # Fallback if the API somehow didn't return a tool block (highly unlikely with tool_choice forced)
            raise LLMProxyException(
                "Anthropic API did not return the requested structured tool block."
            )

        except Exception as e:
            raise LLMProxyException(
                f"Error invoking Anthropic LLM (structured): {e}"
            ) from e

    def invoke_grammar_based(self, prompt: str | list, **kwargs: Any) -> LLMResponse:
        """
        This method is not implemented for standard Anthropic models natively
        through the public API.
        """
        raise NotImplementedError(
            "invoke_grammar_based is not natively implemented for Anthropic models."
        )

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        try:
            messages = [{"role": "user", "content": prompt}]
            req: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self._max_output_tokens,
            }
            if self._temperature is not None:
                req["temperature"] = self._temperature
            if self._bound_tools:
                req["tools"] = self._bound_tools

            req.update(kwargs)

            with self._client.messages.stream(**req) as stream:
                for text_chunk in stream.text_stream:
                    if text_chunk:
                        yield text_chunk

        except Exception as e:
            raise LLMProxyException(f"Error streaming from Anthropic LLM: {e}") from e

    def supports_tool_use(self) -> bool:
        return True

    def bind_tools(
            self, toolset_class: Type["Toolset"], tool_names: List[str]
    ) -> "AnthropicLLMProxy":
        try:
            bound_tools = []
            for name in tool_names:
                full_name = f"{toolset_class.__name__}.{name}"
                if full_name not in toolset_class.REGISTRY:
                    continue
                info = toolset_class.REGISTRY[full_name]

                properties = {}
                required = []
                for p_name, param in info["parameters"].items():
                    properties[p_name] = {
                        "type": self._map_parameter_type(param),
                        "description": p_name,
                    }
                    if param.default == inspect.Parameter.empty:
                        required.append(p_name)

                bound_tools.append(
                    {
                        "name": name,
                        "description": (info["docstring"] or "").strip(),
                        "input_schema": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    }
                )

            self._bound_tools = bound_tools
            return self
        except Exception as e:
            raise LLMProxyException(
                f"Failed to bind tools to Anthropic LLM: {e}"
            ) from e

    def unbind_tools(self) -> "AnthropicLLMProxy":
        self._bound_tools = []
        return self

    def messages_from(self, prompts_with_roles: list[dict[str, str] | tuple[str, str]]) -> list[dict]:
        raise NotImplementedError
