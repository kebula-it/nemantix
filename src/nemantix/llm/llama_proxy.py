import inspect
import json
from typing import Any, Iterator, Dict, List, Optional, Type, Union, TYPE_CHECKING

from pydantic import BaseModel

from nemantix.common.logger import get_package_logger
from nemantix.llm.abstract_proxy import (AbstractLLMProxy, LLMProxyException,
                                         LLMResponse, LLMUsage, StructuredLLMResponse)

if TYPE_CHECKING:
    from nemantix.core.tools import Toolset

try:
    from ollama import chat
except ImportError as e:
    raise ImportError("Please install the Ollama Python library: `pip install ollama`") from e

logger = get_package_logger(__name__)

ToolSpec = Dict[str, Any]


class LlamaProxy(AbstractLLMProxy):
    """LLM proxy for models served via Ollama (local inference)."""

    def __init__(
            self,
            model_name: str,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            host: Optional[str] = None,
            **__,
    ):
        self.model_name = model_name
        self._bound_tools: List[ToolSpec] = []
        self._toolset_class: Optional[Type["Toolset"]] = None
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._host = host

    # ----------------------------- helpers -----------------------------

    def _build_options(self) -> Dict[str, Any]:
        opts: Dict[str, Any] = {}
        if self._temperature is not None:
            opts["temperature"] = self._temperature
        if self._max_tokens is not None:
            opts["num_predict"] = self._max_tokens
        return opts

    def _build_usage(self, response) -> LLMUsage:
        return LLMUsage(
            input_tokens=getattr(response, "prompt_eval_count", 0) or 0,
            output_tokens=getattr(response, "eval_count", 0) or 0,
        )

    @staticmethod
    def _map_parameter_type(parameter: inspect.Parameter) -> str:
        ann = parameter.annotation
        if ann is str:
            return "string"
        if ann is int:
            return "integer"
        if ann is bool:
            return "boolean"
        if ann is float:
            return "number"
        return "string"

    def _call_tools(self, response, messages: list) -> Any:
        """Execute tool calls returned by the model and return the follow-up response."""
        tool_calls = response.message.tool_calls or []
        messages.append(response.message)

        for tc in tool_calls:
            fn_name = tc.function.name
            args = tc.function.arguments or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            tool_name = f"{self._toolset_class.__name__}.{fn_name}"
            tool_instance = self._toolset_class.get_tool(tool_name)

            logger.debug(f'Calling tool "{fn_name}"..')
            try:
                result = tool_instance(**args)
                result_str = json.dumps(result) if not isinstance(result, str) else result
            except Exception as exc:
                result_str = f"Error executing tool: {exc}"
                logger.warning(result_str)

            logger.debug(f'Tool "{fn_name}" returned: "{result_str}"')
            messages.append({"role": "tool", "content": result_str})

        kwargs: Dict[str, Any] = {"model": self.model_name, "messages": messages}
        opts = self._build_options()
        if opts:
            kwargs["options"] = opts
        if self._host:
            kwargs["host"] = self._host

        return chat(**kwargs)

    # ----------------------------- interface -----------------------------

    def get_name(self) -> str:
        return f"Ollama {self.model_name}"

    def invoke(self, prompt: Union[str, list], **kwargs: Any) -> LLMResponse:
        try:
            messages = (
                [{"role": "user", "content": prompt}]
                if isinstance(prompt, str)
                else prompt
            )

            req: Dict[str, Any] = {"model": self.model_name, "messages": messages}
            opts = self._build_options()
            if opts:
                req["options"] = opts
            if self._bound_tools:
                req["tools"] = self._bound_tools
            if self._host:
                req["host"] = self._host
            req.update(kwargs)

            response = chat(**req)

            tool_calls_raw = response.message.tool_calls or []
            tool_calls = [
                {"name": tc.function.name, "args": tc.function.arguments or {}}
                for tc in tool_calls_raw
            ]

            if tool_calls_raw and self._toolset_class is not None:
                response = self._call_tools(response, messages)

            text = response.message.content or ""
            return LLMResponse(text=text, tool_calls=tool_calls, usage=self._build_usage(response))

        except Exception as exc:
            raise LLMProxyException(f"Error invoking Ollama model: {exc}") from exc

    def invoke_structured(self, prompt: Union[str, list], schema: Type[BaseModel], **kwargs) -> StructuredLLMResponse:
        try:
            messages = (
                [{"role": "user", "content": prompt}]
                if isinstance(prompt, str)
                else prompt
            )

            req: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "format": schema.model_json_schema(),
            }
            opts = self._build_options()
            if opts:
                req["options"] = opts
            if self._host:
                req["host"] = self._host
            req.update(kwargs)

            response = chat(**req)
            content = response.message.content or "{}"
            data = json.loads(content)
            return StructuredLLMResponse(
                result=schema.model_validate(data),
                usage=self._build_usage(response),
            )

        except Exception as exc:
            raise LLMProxyException(f"Error invoking Ollama model (structured): {exc}") from exc

    def invoke_grammar_based(self, prompt: Union[str, list], **kwargs: Any) -> LLMResponse:
        raise LLMProxyException("Grammar-based invocation is not supported by Ollama.")

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        try:
            req: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
            opts = self._build_options()
            if opts:
                req["options"] = opts
            if self._host:
                req["host"] = self._host
            req.update(kwargs)

            for chunk in chat(**req):
                text = chunk.message.content
                if text:
                    yield text

        except Exception as exc:
            raise LLMProxyException(f"Error streaming from Ollama model: {exc}") from exc

    def supports_tool_use(self) -> bool:
        return True

    def bind_tools(self, toolset_class: Type["Toolset"], tool_names: List[str] = None) -> "LlamaProxy":
        try:
            if tool_names is None or not isinstance(tool_names, (list, tuple)):
                tool_names = toolset_class.get_tool_names()

            self._toolset_class = toolset_class
            bound_tools: List[ToolSpec] = []

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

                bound_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": (info["docstring"] or "").strip(),
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                })

            self._bound_tools = bound_tools
            return self

        except Exception as exc:
            raise LLMProxyException(f"Failed to bind tools to Ollama model: {exc}") from exc

    def unbind_tools(self) -> "LlamaProxy":
        self._bound_tools = []
        self._toolset_class = None
        return self

    def messages_from(self, prompts_with_roles: list[dict[str, str] | tuple[str, str]]) -> list[dict]:
        messages = []
        for value in prompts_with_roles:
            if isinstance(value, dict):
                assert "prompt" in value and "role" in value
                role, prompt = value["role"], value["prompt"]
            else:
                assert isinstance(value, (list, tuple)) and len(value) == 2
                role, prompt = value[0], value[1]
            messages.append({"role": role, "content": prompt})
        return messages
