import inspect
import json
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Type

from pydantic import BaseModel

from nemantix.common import get_package_logger
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
    # OpenAI Python SDK v1.x
    from openai import OpenAI
except Exception as e:
    raise ImportError(
        "Please install the official OpenAI SDK: `pip install openai>=1.0.0`"
    ) from e

logger = get_package_logger(__name__)
ToolSpec = Dict[
    str, Any
]  # expected to be {"type":"function","function":{"name":..., "parameters": {...}}}



class OpenAICompatibleProxy(AbstractLLMProxy):
    """Base class for OpenAI-compatible API LLM proxies"""

    def __init__(
        self,
        model_name: str,
        api_key_name: str,  # Injected by the subclass
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        grammar_path: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,  # For custom headers/settings
        **kwargs: Any,
    ):
        self.model_name = model_name
        self._bound_tools: List[ToolSpec] = []
        self._toolset_class: Type["Toolset"] | None = None
        self._reasoning_effort = reasoning_effort

        api_key = self._get_api_key(api_key_name, required=False, **kwargs)

        final_client_kwargs: Dict[str, Any] = {"api_key": api_key or "no-key-required"}
        if base_url:
            final_client_kwargs["base_url"] = base_url
        if client_kwargs:
            final_client_kwargs.update(client_kwargs)

        try:
            self._client = OpenAI(**final_client_kwargs)
        except Exception as e:
            raise LLMProxyException(
                f"Failed to initialize compatible client: {e}"
            ) from e

        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

        self._grammar = None
        if grammar_path is not None:
            with open(grammar_path, "r") as f:
                self._grammar = f.read()

    # ----------------------------- helpers -----------------------------
    @staticmethod
    def _build_usage(u) -> LLMUsage:
        cached = 0
        if u and u.prompt_tokens_details:
            cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
        return LLMUsage(
            input_tokens=u.prompt_tokens if u else 0,
            output_tokens=u.completion_tokens if u else 0,
            cache_read_tokens=cached,
        )

    @staticmethod
    def _map_parameter_type(parameter: inspect.Parameter) -> str:
        # TODO: proper type mapping, including complex Pydantic parameters
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
    def _extract_tool_calls(choice_msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Converts OpenAI tool_calls into [{"name":..., "args":{...}}, ...].
        """
        out: List[Dict[str, Any]] = []
        for tc in choice_msg.get("tool_calls", []) or []:
            if tc and tc.type == "function":
                fn = tc.function
                name = fn.name
                args_str = fn.arguments
                try:
                    import json

                    args = json.loads(args_str)
                except JSONDecodeError:
                    args = {"_raw": args_str}
                out.append({"name": name, "args": args})
        return out

    # ----------------------------- interface -----------------------------

    def get_name(self) -> str:
        raise NotImplementedError

    def invoke(self, prompt: str | list, tool_choice='auto', **kwargs: Any) -> LLMResponse:
        try:
            # if 1 message -> convert to user message. If list of messages->pass it (for context)
            message = (
                [{"role": "user", "content": prompt}]
                if isinstance(prompt, str)
                else prompt)

            req: Dict[str, Any] = {
                "model": self.model_name,
                "messages": message,
            }
            if self._bound_tools:
                req["tools"] = self._bound_tools
                req["tool_choice"] = str(tool_choice)

            if self._temperature is not None:
                req["temperature"] = self._temperature

            if self._max_output_tokens is not None:
                req["max_completion_tokens"] = self._max_output_tokens

            if self._reasoning_effort is not None:
                req["reasoning_effort"] = self._reasoning_effort

            # allow caller overrides
            req.update(kwargs)

            resp = self._client.chat.completions.create(**req)
            msg = resp.choices[0].message

            tool_calls = self._extract_tool_calls(
                msg.__dict__ if hasattr(msg, "__dict__") else dict(msg))

            if msg.tool_calls:
                msg = self._call_tools(msg, message, prompt, request=req)

            text = msg.content or ""
            return LLMResponse(text=text, tool_calls=tool_calls, proxy=self,
                               usage=self._build_usage(resp.usage))

        except Exception as e:
            raise LLMProxyException(f"Error invoking OpenAI LLM: {e}") from e

    def invoke_structured(self, prompt: str, schema: Type[BaseModel], tool_choice='auto') -> StructuredLLMResponse:
        """
        Uses OpenAI Structured Outputs (response_format: json_schema) to force the
        model to return JSON that conforms to the provided Pydantic schema.

        Requires a model that supports Structured Outputs in Chat Completions
        """
        # Build the json_schema block once from your Pydantic model
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": False,  # enforce exact adherence, set to False if there are invalid schema errors
                "schema": schema.model_json_schema(),  # your Pydantic schema
            },
        }

        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else prompt)

        try:
            req = {
                "model": self.model_name,
                "messages": messages,
                "response_format": json_schema,
            }

            if self._bound_tools:
                req["tools"] = self._bound_tools
                req["tool_choice"] = str(tool_choice)

            if self._temperature is not None:
                req["temperature"] = self._temperature
            if self._max_output_tokens is not None:
                req["max_completion_tokens"] = self._max_output_tokens
            if self._reasoning_effort is not None:
                req["reasoning_effort"] = self._reasoning_effort

            resp = self._client.chat.completions.create(**req)
            msg = resp.choices[0].message

            if msg.tool_calls:
                msg = self._call_tools(msg, messages, prompt, request=req)

            content = msg.content or "{}"
            data = json.loads(content)  # guaranteed valid JSON with structured outputs

            return StructuredLLMResponse(result=schema.model_validate(data), proxy=self,
                                         usage=self._build_usage(resp.usage))

        except Exception as e:
            raise LLMProxyException(
                f"Error invoking OpenAI LLM (structured): {e}"
            ) from e

    def invoke_grammar_based(self, prompt: str | list, **kwargs: Any) -> LLMResponse:
        """
        Invokes the LLM with a grammar-based tool to process the input prompt using a custom grammar file.

        This method reads a grammar definition from a file and sends it along with the prompt to the model,
        which processes it using the specified grammar. It is designed for cases where the input requires
        specialized grammar rules to be applied during the model's response generation.

        Args:
            prompt (str): The input string containing the prompt to be processed.
            **kwargs (Any): Additional vendor-specific parameters or overrides to be passed to the LLM.

        Returns:
            LLMResponse with text, tool_calls, and usage fields.

        Raises:
            LLMProxyException: If there is an error invoking the OpenAI LLM or any other issue during the request.
        """

        try:
            req: Dict[str, Any] = {
                "model": self.model_name,
                "input": prompt,
            }
            if self._temperature is not None:
                req["temperature"] = self._temperature
            if self._max_output_tokens is not None:
                req["max_output_tokens"] = self._max_output_tokens
            if self._reasoning_effort is not None:
                req["reasoning"] = {"effort": self._reasoning_effort}

            req["tools"] = [
                {
                    "type": "custom",
                    "name": "nxs_grammar",
                    "description": "",
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": self._grammar,
                    },
                },
            ]

            req.update(kwargs)

            resp = self._client.responses.create(**req)

            msg = None
            try:
                if not getattr(resp, 'output', None):
                    raise ValueError("Output array is empty or missing.")

                for item in reversed(resp.output):

                    if hasattr(item, 'content') and item.content:
                        msg = item.content[0].text
                        break

                    elif type(item).__name__ == "ResponseCustomToolCall":
                        tool_args = getattr(item, 'input', None) or getattr(item, 'args', None) or getattr(item,
                                                                                                           'arguments',
                                                                                                           None)

                        if isinstance(tool_args, dict):
                            msg = next(iter(tool_args.values()), str(tool_args))
                        elif isinstance(tool_args, str):
                            msg = tool_args

                        if msg:
                            break

                if msg is None:
                    raise ValueError("Invalid output structure or missing fields.")

            except Exception as parse_err:
                raise LLMProxyException(
                    f"Error parsing the answer from model {self.model_name}. Details: {parse_err}\nAnswer raw: {resp}"
                ) from parse_err

            return LLMResponse(
                text=msg,
                tool_calls=[],
                usage=LLMUsage(
                    input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(resp.usage, "output_tokens", 0) or 0),
                proxy=self,
            )
        except Exception as e:
            raise LLMProxyException(f"Error invoking OpenAI LLM: {e}") from e

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        try:
            req: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
            if self._bound_tools:
                req["tools"] = self._bound_tools
            if self._temperature is not None:
                req["temperature"] = self._temperature
            if self._max_output_tokens is not None:
                req["max_tokens"] = self._max_output_tokens
            if self._reasoning_effort is not None:
                req["reasoning_effort"] = self._reasoning_effort

            req.update(kwargs)

            with self._client.chat.completions.create(**req) as stream:
                for event in stream:
                    try:
                        delta = event.choices[0].delta  # SDK v1.x
                        chunk = getattr(delta, "content", None)
                        if chunk:
                            yield chunk
                    except Exception:
                        # tolerate any non-content events
                        continue
        except Exception as e:
            raise LLMProxyException(f"Error streaming from OpenAI LLM: {e}") from e

    def supports_tool_use(self) -> bool:
        return True

    def bind_tools(self, toolset_class: Type["Toolset"],
                   tool_names: List[str] = None) -> "OpenAICompatibleProxy":
        try:
            bound_tools = []

            if tool_names is None or not isinstance(tool_names, (list, tuple)):
                tool_names = toolset_class.get_tool_names()

            for name in tool_names:
                full_name = f"{toolset_class.__name__}.{name}"
                if full_name not in toolset_class.REGISTRY:
                    continue

                info = toolset_class.REGISTRY[full_name]
                # self._tools[full_name] = info
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
                    })

            self._bound_tools = bound_tools
            return self
        except Exception as e:
            raise LLMProxyException(f"Failed to bind tools to OpenAI LLM: {e}") from e

    def unbind_tools(self) -> "OpenAICompatibleProxy":
        self._bound_tools = []
        return self

    def messages_from(self, prompts_with_roles: list[dict[str, str] | tuple[str, str]]) -> list[dict]:
        messages = []

        for value in prompts_with_roles:
            if isinstance(value, dict):
                assert 'prompt' in value
                assert 'role' in value
                prompt = value["prompt"]
                role = value["role"]
            else:
                assert isinstance(value, (list, tuple)) and len(value) == 2
                prompt = value[1]
                role = value[0]

            message = {"role": role, "content": prompt}
            messages.append(message)

        return messages

    def _call_tools(self, msg, messages, prompt, request):
        if not isinstance(messages, list):
            messages = [prompt]

        messages.append(msg.model_dump(exclude_none=True))

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            args_str = tool_call.function.arguments

            tool_name = f'{self._toolset_class.__name__}.{fn_name}'
            tool_instance = self._toolset_class.get_tool(tool_name)

            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                logger.warning(f'Failed to JSON-decode tool arguments: "{args_str}"')
                args = {}

            logger.debug(f'Calling tool "{fn_name}"..')
            try:
                result = tool_instance(**args)
                # Ensure result is a string (JSON stringify dicts/lists)
                tool_result_str = json.dumps(result) if not isinstance(result, str) else result
            except Exception as e:
                # Pass execution errors back so the LLM knows it failed
                tool_result_str = f"Error executing tool: {e}"
                logger.warning(tool_result_str)

            logger.debug(f'Call of tool "{fn_name}" ended with result:\n"{tool_result_str}"')
            messages.append(dict(role='tool', tool_call_id=tool_call.id,
                                 name=fn_name, content=tool_result_str))

        # Second LLM call: Send the tool results back to get the final answer
        request["messages"] = messages
        resp_final = self._client.chat.completions.create(**request)
        msg = resp_final.choices[0].message
        return msg


class OpenAILLMProxy(OpenAICompatibleProxy):
    """
    LLM proxy for OpenAI models using the official SDK (no LangChain).
    Keeps the same public interface as AbstractLLMProxy.
    """

    def __init__(self, model_name: str, **kwargs):
        # Extract OpenAI-specific kwargs
        client_kwargs = {}
        for k in ("organization", "project"):
            if k in kwargs and kwargs[k] is not None:
                client_kwargs[k] = kwargs.pop(k)

        super().__init__(model_name=model_name, api_key_name='openai_api_key',
                         client_kwargs=client_kwargs, **kwargs)

    def get_name(self) -> str:
        return f'OpenAI {self.model_name}'
