import inspect
import json
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Type

from google import genai
from google.genai import types
from pydantic import BaseModel

from nemantix.common.logger import get_package_logger
from nemantix.llm.abstract_proxy import (
    AbstractLLMProxy,
    LLMProxyException,
    LLMResponse,
    LLMUsage,
    StructuredLLMResponse,
)

if TYPE_CHECKING:
    from nemantix.core.tools import Toolset


logger = get_package_logger(__name__)


class GoogleLLMProxy(AbstractLLMProxy):
    """
    A proxy for interacting with Google's Gemini models using the google-genai SDK.

    This class handles client initialization, tool binding, and invocation logic,
    including parsing text and function call responses. It also provides robust
    error handling by mapping Google API errors to the custom LLMProxyException.
    """

    def __init__(
        self,
        model_name: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        grammar_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initializes the Google Generative AI LLM proxy.

        Args:
            model_name: The name of the Google model (e.g., "gemini-pro", "gemini-1.5-flash").
            temperature: Controls randomness (0.0 to 1.0).
            max_output_tokens: Maximum number of tokens to generate.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            **kwargs: Additional parameters to pass to ChatGoogleGenerativeAI constructor not explicitly listed.
                      These could include `timeout`, `convert_system_message_to_human`, etc.
        """
        self.model_name = model_name
        self._google_api_key = self._get_api_key("google_api_key", **kwargs)

        try:
            self._client = genai.Client(api_key=self._google_api_key)
        except Exception as e:
            raise LLMProxyException(f"Failed to initialize genai.Client: {e}") from e

        self._generation_config: types.GenerateContentConfig = (
            types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_k=top_k,
                top_p=top_p,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            )
        )

        if grammar_path is not None:
            with open(grammar_path, "r") as f:
                grammar = f.read()
            self._grammar = grammar

    @staticmethod
    def _build_usage(meta) -> LLMUsage:
        return LLMUsage(
            input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            cache_read_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
        )

    def get_name(self) -> str:
        return f"Google {self.model_name}"

    def invoke(self, prompt: str | list, **kwargs) -> LLMResponse:
        try:
            messages, system_instruction = self._normalize_messages(prompt)
            config = self._build_config(system_instruction=system_instruction)

            response = self._client.models.generate_content(
                model=self.model_name, contents=messages, config=config
            )

            tool_calls_list = []
            if response.parts:
                for part in response.parts:
                    if part.function_call:
                        tool_calls_list.append(
                            {
                                "name": part.function_call.name,
                                "args": dict(part.function_call.args),
                            }
                        )

            return LLMResponse(
                text=response.text or "",
                tool_calls=tool_calls_list,
                usage=self._build_usage(response.usage_metadata),
                proxy=self,
            )
        except Exception as e:
            raise LLMProxyException(
                f"An unexpected error occurred during invocation: {e}"
            ) from e

    def invoke_structured(
        self, prompt: str | list, schema: Type[BaseModel], **kwargs
    ) -> StructuredLLMResponse:
        json_schema = schema.model_json_schema()

        try:
            messages, system_instruction = self._normalize_messages(prompt)

            # Apply structured output overrides to the base config
            config = self._build_config(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_json_schema=json_schema,
            )

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=config,
            )

            content = response.text or "{}"
            data = json.loads(content)

            return StructuredLLMResponse(
                result=schema.model_validate(data),
                usage=self._build_usage(response.usage_metadata),
                proxy=self,
            )
        except Exception as e:
            raise LLMProxyException(
                f"Error invoking Google LLM (structured): {e}"
            ) from e

    def invoke_grammar_based(self, prompt: str | list, **kwargs: Any) -> LLMResponse:
        """
        This method is not implemented for Google Gemini models.
        It raises an exception to indicate the lack of implementation.

        Args:
            prompt (str): The input string to be processed by the model.
            **kwargs (Any): Additional parameters to be passed to the model.

        Raises:
            NotImplementedError: If called, since grammar-based invocation is not supported for Google Gemini models.
        """
        logger.warning(
            "Grammar invoke is not natively supported, using invoke() instead."
        )
        return self.invoke(prompt, **kwargs)

    def stream(self, prompt: str | list, **kwargs: Any) -> Iterator[str]:
        try:
            messages, system_instruction = self._normalize_messages(prompt)
            config = self._build_config(system_instruction=system_instruction)

            for chunk in self._client.models.generate_content_stream(
                model=self.model_name, contents=messages, config=config
            ):
                yield chunk.text
        except Exception as e:
            raise LLMProxyException(
                f"An unexpected error occurred during invocation: {e}"
            ) from e

    def supports_tool_use(self) -> bool:
        return True

    @staticmethod
    def _map_parameter_type(parameter: inspect.Parameter) -> "types.Type":
        # TODO: proper type mapping, including complex Pydantic parameters
        ann_type = parameter.annotation
        if ann_type is str:
            return types.Type.STRING
        if ann_type is int:
            return types.Type.INTEGER
        if ann_type is bool:
            return types.Type.BOOLEAN
        if ann_type is float:
            return types.Type.NUMBER
        return types.Type.STRING

    def bind_tools(
        self, toolset_class: Type["Toolset"], tool_names: List[str]
    ) -> "GoogleLLMProxy":
        if not tool_names:
            self._generation_config.tools = None
            return self

        try:
            function_declarations = []
            for name in tool_names:
                full_name = f"{toolset_class.__name__}.{name}"
                if full_name not in toolset_class.REGISTRY:
                    continue

                info = toolset_class.REGISTRY[full_name]

                properties = {}
                required = []
                for p_name, param in info["parameters"].items():
                    properties[p_name] = types.Schema(
                        type=self._map_parameter_type(param), description=p_name
                    )
                    if param.default == inspect.Parameter.empty:
                        required.append(p_name)

                function_declarations.append(
                    types.FunctionDeclaration(
                        name=name,
                        description=(info["docstring"] or "").strip()
                        or "No description provided.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties=properties,
                            required=required,
                        ),
                    )
                )

            if function_declarations:
                tools = types.Tool(function_declarations=function_declarations)
                self._generation_config.tools = [tools]
            else:
                self._generation_config.tools = None

            return self
        except Exception as e:
            raise LLMProxyException(f"Failed to bind tools to Google LLM: {e}") from e

    def unbind_tools(self) -> "GoogleLLMProxy":
        """
        Unbinds all the tools from this proxy.
        """
        self._generation_config.tools = None
        return self

    def messages_from(
        self, prompts_with_roles: list[dict[str, str] | tuple[str, str]]
    ) -> list[dict]:
        raise NotImplementedError

    @staticmethod
    def _normalize_messages(prompt: str | list) -> tuple[list[dict], Optional[str]]:
        raw_messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        )

        def extract_text(content) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, dict):
                # Look for standard text fields
                return str(
                    content.get(
                        "text", content.get("content", content.get("value", ""))
                    )
                )
            if isinstance(content, list):
                # Join text from multiple nested blocks
                text_parts = []
                for block in content:
                    if isinstance(block, str):
                        text_parts.append(block)
                    elif isinstance(block, dict):
                        text_parts.append(
                            str(
                                block.get(
                                    "text", block.get("content", block.get("value", ""))
                                )
                            )
                        )
                return "\n".join(text_parts)
            return str(content)

        messages = []
        system_instruction = None

        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "user")
            content = msg.get("content", "")

            extracted_text = extract_text(content)

            if role == "system":
                # Google Gemini expects system_instruction explicitly in the config
                if system_instruction is None:
                    system_instruction = extracted_text
                else:
                    system_instruction += "\n" + extracted_text
            else:
                # Map standard roles to Gemini roles
                gemini_role = "model" if role == "assistant" else "user"
                messages.append(
                    {"role": gemini_role, "parts": [{"text": extracted_text}]}
                )

        return messages, system_instruction

    def _build_config(
        self, system_instruction: Optional[str] = None, **overrides
    ) -> types.GenerateContentConfig:
        """Safely merges base config with dynamic system instructions and overrides."""
        config_kwargs = {
            "temperature": self._generation_config.temperature,
            "max_output_tokens": self._generation_config.max_output_tokens,
            "top_k": self._generation_config.top_k,
            "top_p": self._generation_config.top_p,
            "automatic_function_calling": self._generation_config.automatic_function_calling,
            "tools": self._generation_config.tools,
        }

        if system_instruction is not None:
            config_kwargs["system_instruction"] = system_instruction

        config_kwargs.update(overrides)
        return types.GenerateContentConfig(**config_kwargs)
