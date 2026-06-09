import inspect
import json
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Type

from google import genai
from google.genai import types
from pydantic import BaseModel

from .abstract_proxy import (
    AbstractLLMProxy,
    LLMProxyException,
    LLMResponse,
    LLMUsage,
    StructuredLLMResponse,
)

if TYPE_CHECKING:
    from nemantix.core.tools import Toolset


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
        return f'Google {self.model_name}'

    def invoke(self, prompt: str | list, **kwargs) -> LLMResponse:
        """
        Invokes the Gemini model with a given prompt.

        This method sends the prompt to the model and processes the response.
        It can handle two types of responses:
        1. A standard text response.
        2. A function call request from the model.

        It also wraps API calls in a try-except block to catch potential
        Google API errors and re-raise them as a custom LLMProxyException.

        Args:
            prompt: The user prompt to send to the model.

        Returns:
            A dictionary with the following keys:
            - text: A string containing the model's text response.
            - tool_calls: A dictionary representing a function call, with 'name' and 'args' keys.

        Raises:
            LLMProxyException: If the model is not bound, or if an API error occurs.
        """
        try:
            response = self._client.models.generate_content(
                model=self.model_name, contents=prompt, config=self._generation_config
            )

            tool_calls_list = []
            for part in response.parts:
                if part.function_call:
                    tool_calls_list.append(
                        {
                            "name": part.function_call.name,
                            "args": dict(part.function_call.args),
                        }
                    )

            # If no function call is found, return the consolidated text response.
            return LLMResponse(text=response.text or '', tool_calls=tool_calls_list,
                               usage=self._build_usage(response.usage_metadata),
                               proxy=self)
        except Exception as e:
            # Catch any other unexpected errors during the process.
            raise LLMProxyException(
                f"An unexpected error occurred during invocation: {e}"
            ) from e

    def invoke_structured(self, prompt: str, schema: Type[BaseModel], **kwargs) -> StructuredLLMResponse:
        """
        Uses Structured Outputs (response_format: json_schema) to force the
        model to return JSON that conforms to the provided Pydantic schema.

        Requires a model that supports Structured Outputs in Chat Completions.
        """
        json_schema = schema.model_json_schema()

        try:
            full_config = types.GenerateContentConfig(
                temperature=self._generation_config.temperature,
                max_output_tokens=self._generation_config.max_output_tokens,
                top_k=self._generation_config.top_k,
                top_p=self._generation_config.top_p,
                automatic_function_calling=self._generation_config.automatic_function_calling,
                response_mime_type="application/json",
                response_json_schema=json_schema,
            )

            # Send the request to the model
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=full_config,  # Pass the full configuration
            )

            content = response.text or "{}"
            data = json.loads(
                content
            )  # Ensured that the JSON is valid with structured outputs

            # Return the validated data with the Pydantic model
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
        raise NotImplementedError(
            "invoke_grammar_based is not implemented for Google Gemini models."
        )

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        try:
            for chunk in self._client.models.generate_content_stream(
                    model=self.model_name, contents=prompt, config=self._generation_config
            ):
                yield chunk.text
        except Exception as e:
            # Catch any other unexpected errors during the process.
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

    def messages_from(self, prompts_with_roles: list[dict[str, str] | tuple[str, str]]) -> list[dict]:
        raise NotImplementedError
