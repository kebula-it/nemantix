import inspect
import json
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Type, Union

from pydantic import BaseModel

from nemantix.llm.abstract_proxy import (
    AbstractLLMProxy,
    LLMProxyException,
    LLMResponse,
    LLMUsage,
    StructuredLLMResponse,
)

if TYPE_CHECKING:
    from nemantix.core.tools import Toolset

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "Please install the official OpenAI SDK to use OpenRouter: `pip install openai>=1.0.0`"
    ) from e


class OpenRouterLLMProxy(AbstractLLMProxy):
    """
    LLM proxy for OpenRouter models using the official OpenAI SDK.
    """

    def __init__(
        self,
        model_name: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        site_url: str = "https://https://github.com/kebula-it/nemantix",
        app_name: str = "Nemantix",
        grammar_path: Optional[str] = None,
        **kwargs: Any,
    ):
        self._model_name = model_name
        self._bound_tools: List[Dict[str, Any]] = []
        self._toolset_class: Type["Toolset"] | None = None

        # Fetch OpenRouter key via the credentials manager
        api_key = self._get_api_key("openrouter_api_key", **kwargs)

        # OpenRouter highly recommends setting HTTP-Referer and X-Title for rankings
        default_headers = {
            "HTTP-Referer": site_url,
            "X-Title": app_name,
        }

        try:
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers=default_headers,
            )
        except Exception as e:
            raise LLMProxyException(
                f"Failed to initialize OpenRouter client: {e}"
            ) from e

        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

        self._grammar = None
        if grammar_path is not None:
            with open(grammar_path, "r") as f:
                self._grammar = f.read()

    def get_name(self) -> str:
        return f"OpenRouter ({self._model_name})"

    # ----------------------------- Helpers -----------------------------

    @staticmethod
    def _build_usage(usage_obj) -> LLMUsage:
        if not usage_obj:
            return LLMUsage(input_tokens=0, output_tokens=0)

        cached = getattr(usage_obj, "prompt_tokens_details", None)
        cache_read = getattr(cached, "cached_tokens", 0) if cached else 0

        return LLMUsage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            cache_read_tokens=cache_read,
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

    @staticmethod
    def _extract_tool_calls(message: Any) -> List[Dict[str, Any]]:
        out = []
        tool_calls = getattr(message, "tool_calls", []) or []
        for tc in tool_calls:
            if tc.type == "function":
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                out.append({"name": name, "args": args})
        return out

    # ----------------------------- Interface -----------------------------

    def invoke(
        self, prompt: Union[str, list], tool_choice="auto", **kwargs: Any
    ) -> LLMResponse:
        messages = (
            self.messages_from([("user", prompt)])
            if isinstance(prompt, str)
            else prompt
        )

        req: Dict[str, Any] = {"model": self._model_name, "messages": messages}

        if self._bound_tools:
            req["tools"] = self._bound_tools
            req["tool_choice"] = str(tool_choice)
        if self._temperature is not None:
            req["temperature"] = self._temperature
        if self._max_output_tokens is not None:
            req["max_tokens"] = self._max_output_tokens

        req.update(kwargs)

        try:
            resp = self._client.chat.completions.create(**req)
            msg = resp.choices[0].message

            tool_calls_extracted = self._extract_tool_calls(msg)
            return LLMResponse(
                text=msg.content or "",
                tool_calls=tool_calls_extracted,
                usage=self._build_usage(resp.usage),
            )
        except Exception as e:
            raise LLMProxyException(f"Error invoking OpenRouter LLM: {e}") from e

    def invoke_structured(
        self, prompt: Union[str, list], schema: Type[BaseModel], **kwargs
    ) -> StructuredLLMResponse:
        messages = (
            self.messages_from([("user", prompt)])
            if isinstance(prompt, str)
            else prompt
        )

        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": False,
                "schema": schema.model_json_schema(),
            },
        }

        req: Dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "response_format": json_schema,
        }

        if self._temperature is not None:
            req["temperature"] = self._temperature
        if self._max_output_tokens is not None:
            req["max_tokens"] = self._max_output_tokens
        req.update(kwargs)

        try:
            resp = self._client.chat.completions.create(**req)
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            return StructuredLLMResponse(
                result=schema.model_validate(data), usage=self._build_usage(resp.usage)
            )
        except Exception as e:
            raise LLMProxyException(
                f"Error invoking OpenRouter LLM (structured): {e}"
            ) from e

    def invoke_grammar_based(
        self, prompt: Union[str, list], **kwargs: Any
    ) -> LLMResponse:
        if not self._grammar:
            raise LLMProxyException(
                "No grammar path was provided during initialization."
            )

        req: Dict[str, Any] = {"model": self._model_name, "input": prompt}
        if self._temperature is not None:
            req["temperature"] = self._temperature
        if self._max_output_tokens is not None:
            req["max_output_tokens"] = self._max_output_tokens

        req["tools"] = [
            {
                "type": "custom",
                "name": "nxs_grammar",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": self._grammar,
                },
            }
        ]
        req.update(kwargs)

        try:
            resp = self._client.responses.create(**req)
            msg = resp.output[1].content[0].text
            return LLMResponse(
                text=msg, tool_calls=[], usage=self._build_usage(resp.usage)
            )
        except Exception as e:
            raise LLMProxyException(
                f"Error invoking OpenRouter grammar endpoint: {e}"
            ) from e

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        req: Dict[str, Any] = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        if self._temperature is not None:
            req["temperature"] = self._temperature
        if self._max_output_tokens is not None:
            req["max_tokens"] = self._max_output_tokens
        req.update(kwargs)

        try:
            with self._client.chat.completions.create(**req) as stream:
                for event in stream:
                    chunk = getattr(event.choices[0].delta, "content", None)
                    if chunk:
                        yield chunk
        except Exception as e:
            raise LLMProxyException(f"Error streaming from OpenRouter: {e}") from e

    def supports_tool_use(self) -> bool:
        return True

    def bind_tools(
        self, toolset_class: Type["Toolset"], tool_names: List[str] = None
    ) -> "OpenRouterLLMProxy":
        bound_tools = []
        if tool_names is None:
            tool_names = toolset_class.get_tool_names()

        for name in tool_names:
            full_name = f"{toolset_class.__name__}.{name}"
            if full_name not in toolset_class.REGISTRY:
                continue

            info = toolset_class.REGISTRY[full_name]
            self._toolset_class = toolset_class

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
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": (info["docstring"] or "").strip(),
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                }
            )

        self._bound_tools = bound_tools
        return self

    def unbind_tools(self) -> "OpenRouterLLMProxy":
        self._bound_tools = []
        return self

    def messages_from(
        self, prompts_with_roles: list[dict[str, str] | tuple[str, str]]
    ) -> list[dict]:
        messages = []

        for value in prompts_with_roles:
            if isinstance(value, dict):
                prompt = value["prompt"]
                role = value["role"]
            else:
                role, prompt = value[0], value[1]

            messages.append({"role": role, "content": prompt})

        return messages
