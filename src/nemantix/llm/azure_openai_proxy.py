# llm/azure_openai_proxy.py (Azure OpenAI via official SDK)
import inspect
import json
import os
from typing import Any, Iterator, Dict, Optional, List, Type, TYPE_CHECKING

import httpx
from pydantic import BaseModel

from nemantix.common import get_package_logger

logger = get_package_logger(__name__)

from nemantix.llm.abstract_proxy import (AbstractLLMProxy, LLMProxyException,
                                         LLMResponse, LLMUsage, StructuredLLMResponse)

try:
    # OpenAI Python SDK v1.x
    from openai import AzureOpenAI
except Exception as e:
    raise ImportError(
        "Please install the official OpenAI SDK: `pip install openai>=1.0.0`"
    ) from e

if TYPE_CHECKING:
    from nemantix.core.tools import Toolset

ToolSpec = Dict[str, Any]


class AzureOpenAILLMProxy(AbstractLLMProxy):
    """
    LLM proxy for Azure OpenAI deployments using the official SDK (no LangChain).
    Mirrors the public interface of AbstractLLMProxy / OpenAILLMProxy.
    """

    def __init__(
            self,
            deployment_name: str,
            api_version: str,
            azure_endpoint: str,
            temperature: Optional[float] = None,
            max_output_tokens: Optional[int] = None,
            grammar_path: Optional[str] = None,
            reasoning_effort="high",
            **kwargs: Any,
    ):
        self._deployment_name = deployment_name
        self._bound_tools: List[ToolSpec] = []
        self._toolset_class: Type["Toolset"] = None
        self._grammar: Optional[str] = None
        self._reasoning_effort = reasoning_effort

        api_key = self._get_api_key("azure_openai_api_key", **kwargs)

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "api_version": api_version,
            "azure_endpoint": azure_endpoint,
        }
        # allow overriding endpoint via base_url for compatibility
        if "base_url" in kwargs and kwargs["base_url"] is not None:
            client_kwargs["azure_endpoint"] = kwargs["base_url"]

        disable_ssl_verification = kwargs.pop("disable_ssl_verification", False)
        http_headers = kwargs.pop("http_headers", {})
        hostname = os.environ.get("AZURE_OPENAI_HOSTNAME")
        if hostname:
            http_headers["Host"] = hostname

        httpx_client = httpx.Client(
            verify=not disable_ssl_verification,
            headers=http_headers
        )
        client_kwargs["http_client"] = httpx_client

        for k in ("azure_ad_token", "azure_ad_token_provider", "organization", "project", "azure_deployment"):
            if k in kwargs and kwargs[k] is not None:
                client_kwargs[k] = kwargs[k]

        try:
            self._client = AzureOpenAI(**client_kwargs)
        except Exception as e:
            raise LLMProxyException(f"Failed to initialize Azure OpenAI client: {e}") from e

        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

        if grammar_path is not None:
            with open(grammar_path, "r") as f:
                self._grammar = f.read()

    # ----------------------------- helpers -----------------------------
    def _build_usage(self, u) -> 'LLMUsage':
        cached = 0
        if u and getattr(u, "prompt_tokens_details", None):
            cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
        return LLMUsage(
            input_tokens=u.prompt_tokens if u else 0,
            output_tokens=u.completion_tokens if u else 0,
            cache_read_tokens=cached,
        )

    @staticmethod
    def _map_parameter_type(parameter: inspect.Parameter):
        # TODO: proper type mapping, including complex Pydantic parameters
        ann_type = parameter.annotation
        if ann_type == str:
            return "string"
        if ann_type == int:
            return "integer"
        if ann_type == bool:
            return "boolean"
        if ann_type == float:
            return "number"
        return "string"

    @staticmethod
    def _extract_tool_calls(choice_msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Converts OpenAI tool_calls into [{"name":..., "args":{...}}, ...].
        """
        out: List[Dict[str, Any]] = []
        for tc in choice_msg.get("tool_calls", []) or []:
            if tc and getattr(tc, "type", None) == "function":
                fn = tc.function
                name = fn.name
                args_str = fn.arguments
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {"_raw": args_str}
                out.append({"name": name, "args": args})
        return out

    # ----------------------------- interface -----------------------------
    def get_name(self) -> str:
        return f'Azure OpenAI {self._deployment_name}'

    def invoke(self, prompt: str | list, tool_choice='auto', **kwargs: Any) -> 'LLMResponse':
        try:
            # if 1 message -> convert to user message. If list of messages->pass it (for context)            message = [{"role": "user", "content": prompt}] if type(prompt) is str else prompt
            message = (
                [{"role": "user", "content": prompt}]
                if isinstance(prompt, str)
                else prompt)

            req: Dict[str, Any] = {
                "model": self._deployment_name,
                "messages": message,
            }
            # allow per-call override of deployment name if provided
            if "azure_deployment" in kwargs and kwargs.get("azure_deployment"):
                req["model"] = kwargs.pop("azure_deployment")

            if self._bound_tools:
                req["tools"] = self._bound_tools
                req["tool_choice"] = str(tool_choice)

            if self._temperature is not None:
                req["temperature"] = self._temperature

            if self._max_output_tokens is not None:
                req["max_completion_tokens"] = self._max_output_tokens

            if self._reasoning_effort is not None:
                req["reasoning_effort"] = self._reasoning_effort

            req.update(kwargs)

            resp = self._client.chat.completions.create(**req)
            msg = resp.choices[0].message

            tool_calls = self._extract_tool_calls(msg.__dict__ if hasattr(msg, "__dict__") else dict(msg))

            if msg.tool_calls:
                msg = self._call_tools(msg, message, prompt, request=req)

            text = msg.content or ""

            return LLMResponse(text=text, tool_calls=tool_calls, usage=self._build_usage(resp.usage))
        except Exception as e:
            raise LLMProxyException(f"Error invoking Azure OpenAI LLM: {e} {e.__cause__}") from e

    def invoke_structured(self, prompt: str, schema: Type[BaseModel], tool_choice='auto', **kwargs: Any) -> 'StructuredLLMResponse':
        """
        Uses Azure OpenAI Structured Outputs (json_schema) to enforce responses
        matching the provided Pydantic schema. Requires a deployment that
        supports structured outputs.
        """
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": False,
                "schema": schema.model_json_schema(),
            },
        }

        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else prompt)

        try:
            req = {
                "model": kwargs.pop("azure_deployment", self._deployment_name),
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
            data = json.loads(content)

            return StructuredLLMResponse(result=schema.model_validate(data), usage=self._build_usage(resp.usage))
        except Exception as e:
            raise LLMProxyException(f"Error invoking Azure OpenAI LLM (structured): {e}") from e

    def invoke_grammar_based(self, prompt: str | list, **kwargs: Any) -> 'LLMResponse':
        # Azure grammar/tools: support only GPT-5+ families (e.g., gpt-5*).
        model_name = kwargs.get("azure_deployment", self._deployment_name).lower()
        supported_prefixes = ("gpt-5", )
        if not any(model_name.startswith(pref) for pref in supported_prefixes):
            raise NotImplementedError(
                "invoke_grammar_based is not supported on this deployment; use GPT-5 models or newer."
            )

        if not self._grammar:
            raise LLMProxyException("Grammar not loaded: provide grammar_path during initialization")

        try:
            req: Dict[str, Any] = {
                "model": kwargs.pop("azure_deployment", self._deployment_name),
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
                    f"Error parsing the answer from model {model_name}. Details: {parse_err}\nRaw answer: {resp}"
                ) from parse_err

            return LLMResponse(
                text=msg,
                tool_calls=[],
                usage=LLMUsage(
                    input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(resp.usage, "output_tokens", 0) or 0))

        except Exception as e:
            raise LLMProxyException(f"Error invoking Azure OpenAI LLM: {e}") from e

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        try:
            req: Dict[str, Any] = {
                "model": kwargs.pop("azure_deployment", self._deployment_name),
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
            if self._bound_tools:
                req["tools"] = self._bound_tools
            if self._temperature is not None:
                req["temperature"] = self._temperature
            if self._max_output_tokens is not None:
                req["max_completion_tokens"] = self._max_output_tokens
            if self._reasoning_effort is not None:
                req["reasoning_effort"] = self._reasoning_effort

            req.update(kwargs)

            with self._client.chat.completions.create(**req) as stream:
                for event in stream:
                    try:
                        delta = event.choices[0].delta
                        chunk = getattr(delta, "content", None)
                        if chunk:
                            yield chunk
                    except Exception:
                        continue
        except Exception as e:
            raise LLMProxyException(f"Error streaming from Azure OpenAI LLM: {e}") from e

    def supports_tool_use(self) -> bool:
        return True

    def bind_tools(self, toolset_class: Type['Toolset'], tool_names: List[str] = None) -> "AzureOpenAILLMProxy":
        try:
            bound_tools = []

            if tool_names is None or not isinstance(tool_names, (list, tuple)):
                tool_names = toolset_class.get_tool_names()

            for name in tool_names:
                full_name = f"{toolset_class.__name__}.{name}"
                if full_name not in toolset_class.REGISTRY:
                    continue
                info = toolset_class.REGISTRY[full_name]
                self._toolset_class = toolset_class

                properties = {}
                required = []
                for p_name, param in info['parameters'].items():
                    properties[p_name] = {
                        "type": self._map_parameter_type(param),
                        "description": p_name
                    }
                    if param.default == inspect.Parameter.empty:
                        required.append(p_name)

                bound_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": (info['docstring'] or "").strip(),
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                })

            self._bound_tools = bound_tools
            return self
        except Exception as e:
            raise LLMProxyException(f"Failed to bind tools to OpenAI LLM: {e}") from e

    def unbind_tools(self) -> "AzureOpenAILLMProxy":
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
        msg_final = resp_final.choices[0].message
        return msg_final