import inspect
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Type

from pydantic import BaseModel, TypeAdapter

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

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError as e:
    raise ImportError(
        "Please install the AWS SDK for Python: `pip install boto3>=1.34.0`"
    ) from e

logger = get_package_logger(__name__)


class AWSBedrockLLMProxy(AbstractLLMProxy):
    """
    LLM proxy for Amazon Bedrock models using the boto3 Converse API.
    Provides a unified interface across all supported Bedrock models (Claude, Llama, Mistral, etc.).
    """

    def __init__(
        self,
        model_name: str,
        region_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self._bound_tools: List[Dict[str, Any]] = []

        # Attempt to retrieve credentials from the credentials manager/kwargs if explicitly provided.
        # Otherwise, boto3 defaults to its own credential chain (e.g., ~/.aws/credentials, IAM roles).
        aws_access_key_id = self._get_api_key(
            "aws_access_key_id", required=False, **kwargs
        )
        aws_secret_access_key = self._get_api_key(
            "aws_secret_access_key", required=False, **kwargs
        )
        aws_session_token = self._get_api_key(
            "aws_session_token", required=False, **kwargs
        )

        # Region can be configured through kwargs or fallback to boto3 defaults (like AWS_REGION)
        self._region_name = region_name or kwargs.get("aws_region")

        client_kwargs: Dict[str, Any] = {}
        if self._region_name:
            client_kwargs["region_name"] = self._region_name
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            client_kwargs["aws_session_token"] = aws_session_token

        try:
            self._client = boto3.client("bedrock-runtime", **client_kwargs)
        except Exception as e:
            raise LLMProxyException(
                f"Failed to initialize AWS Bedrock client: {e}"
            ) from e

        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    # ----------------------------- helpers -----------------------------

    @staticmethod
    def _build_usage(usage_data: dict) -> LLMUsage:
        return LLMUsage(
            input_tokens=usage_data.get("inputTokens", 0),
            output_tokens=usage_data.get("outputTokens", 0),
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )

    @staticmethod
    def _build_parameter_schema(parameter: inspect.Parameter) -> Dict[str, Any]:
        ann_type = parameter.annotation
        if ann_type is inspect.Parameter.empty:
            return {"type": "string"}

        try:
            schema = TypeAdapter(ann_type).json_schema()

            schema.pop("title", None)
            if "items" in schema and isinstance(schema["items"], dict):
                schema["items"].pop("title", None)

            return schema
        except Exception:
            return {"type": "string"}

    @staticmethod
    def _normalize_messages(
        prompt: str | list,
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        Normalizes prompts into the Bedrock Converse format.
        Returns a tuple of (messages, system_prompts).
        """
        raw_messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        )

        messages = []
        system_prompts = []

        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role")
            content = msg.get("content", "")

            # Bedrock expects content as a list of dicts (e.g., [{"text": "..."}])
            normalized_content = []
            if isinstance(content, str):
                normalized_content.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        normalized_content.append({"text": block})
                    elif isinstance(block, dict):
                        # Attempt to map standard attributes to Bedrock structure
                        if "text" in block:
                            normalized_content.append({"text": block["text"]})
                        elif "toolUse" in block or "toolResult" in block:
                            normalized_content.append(block)
                        else:
                            normalized_content.append({"text": str(block)})
            elif isinstance(content, dict):
                if "text" in content:
                    normalized_content.append({"text": content["text"]})
                elif "toolUse" in content or "toolResult" in content:
                    normalized_content.append(content)
                else:
                    normalized_content.append({"text": str(content)})

            if role == "system":
                system_prompts.extend(normalized_content)
            elif role in ("user", "assistant"):
                messages.append({"role": role, "content": normalized_content})

        return messages, system_prompts if system_prompts else None

    # ----------------------------- interface -----------------------------

    def get_name(self) -> str:
        return f"AWS Bedrock {self.model_name}"

    def invoke(self, prompt: str | list, **kwargs: Any) -> LLMResponse:
        try:
            messages, system_prompts = self._normalize_messages(prompt)

            req: Dict[str, Any] = {
                "modelId": self.model_name,
                "messages": messages,
            }
            if system_prompts:
                req["system"] = system_prompts

            # Build Inference Configuration
            inference_config = {}
            if self._max_output_tokens is not None:
                inference_config["maxTokens"] = self._max_output_tokens
            if self._temperature is not None:
                inference_config["temperature"] = self._temperature

            if inference_config:
                req["inferenceConfig"] = inference_config

            if self._bound_tools:
                req["toolConfig"] = {"tools": self._bound_tools}

            req.update(kwargs)

            response = self._client.converse(**req)
            output_message = response.get("output", {}).get("message", {})

            text = ""
            tool_calls = []

            for block in output_message.get("content", []):
                if "text" in block:
                    text += block["text"]
                elif "toolUse" in block:
                    tool_use = block["toolUse"]
                    tool_calls.append(
                        {
                            "name": tool_use.get("name"),
                            "args": tool_use.get("input"),
                            "id": tool_use.get(
                                "toolUseId"
                            ),  # Keep ID if required by external loop
                        }
                    )

            return LLMResponse(
                text=text.strip(),
                tool_calls=tool_calls,
                usage=self._build_usage(response.get("usage", {})),
                proxy=self,
            )

        except ClientError as e:
            raise LLMProxyException(f"AWS Bedrock API Error: {e}") from e
        except Exception as e:
            raise LLMProxyException(f"Error invoking AWS Bedrock LLM: {e}") from e

    def invoke_structured(
        self, prompt: str | list, schema: Type[BaseModel], **kwargs
    ) -> StructuredLLMResponse:
        """
        Uses AWS Bedrock's tool choice forcing to guarantee the output matches
        the provided Pydantic schema.
        """
        tool_name = schema.__name__
        tool_schema = {
            "toolSpec": {
                "name": tool_name,
                "description": f"Output schema for {tool_name}",
                "inputSchema": {"json": schema.model_json_schema()},
            }
        }

        try:
            messages, system_prompts = self._normalize_messages(prompt)

            req: Dict[str, Any] = {
                "modelId": self.model_name,
                "messages": messages,
                "toolConfig": {
                    "tools": [tool_schema],
                    "toolChoice": {"tool": {"name": tool_name}},
                },
            }
            if system_prompts:
                req["system"] = system_prompts

            inference_config = {}
            if self._max_output_tokens is not None:
                inference_config["maxTokens"] = self._max_output_tokens
            if self._temperature is not None:
                inference_config["temperature"] = self._temperature

            if inference_config:
                req["inferenceConfig"] = inference_config

            response = self._client.converse(**req)
            output_message = response.get("output", {}).get("message", {})

            for block in output_message.get("content", []):
                if "toolUse" in block and block["toolUse"]["name"] == tool_name:
                    return StructuredLLMResponse(
                        result=schema.model_validate(block["toolUse"]["input"]),
                        usage=self._build_usage(response.get("usage", {})),
                        proxy=self,
                    )

            raise LLMProxyException(
                "AWS Bedrock API did not return the requested structured tool block."
            )

        except Exception as e:
            raise LLMProxyException(
                f"Error invoking AWS Bedrock LLM (structured): {e}"
            ) from e

    def invoke_grammar_based(self, prompt: str | list, **kwargs: Any) -> LLMResponse:
        logger.warning(
            "Grammar invoke is not natively supported in AWS Bedrock, using standard invoke() instead."
        )
        return self.invoke(prompt, **kwargs)

    def stream(self, prompt: str | list, **kwargs: Any) -> Iterator[str]:
        try:
            messages, system_prompts = self._normalize_messages(prompt)

            req: Dict[str, Any] = {
                "modelId": self.model_name,
                "messages": messages,
            }
            if system_prompts:
                req["system"] = system_prompts

            inference_config = {}
            if self._max_output_tokens is not None:
                inference_config["maxTokens"] = self._max_output_tokens
            if self._temperature is not None:
                inference_config["temperature"] = self._temperature

            if inference_config:
                req["inferenceConfig"] = inference_config

            if self._bound_tools:
                req["toolConfig"] = {"tools": self._bound_tools}

            req.update(kwargs)

            response = self._client.converse_stream(**req)

            for event in response.get("stream", []):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        yield delta["text"]

        except Exception as e:
            raise LLMProxyException(f"Error streaming from AWS Bedrock LLM: {e}") from e

    def supports_tool_use(self) -> bool:
        return True

    def bind_tools(
        self, toolset_class: Type["Toolset"], tool_names: List[str]
    ) -> "AWSBedrockLLMProxy":
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
                    param_schema = self._build_parameter_schema(param)
                    param_schema["description"] = p_name
                    properties[p_name] = param_schema

                    if param.default == inspect.Parameter.empty:
                        required.append(p_name)

                bound_tools.append(
                    {
                        "toolSpec": {
                            "name": name,
                            "description": (info["docstring"] or "").strip(),
                            "inputSchema": {
                                "json": {
                                    "type": "object",
                                    "properties": properties,
                                    "required": required,
                                }
                            },
                        }
                    }
                )

            self._bound_tools = bound_tools
            return self
        except Exception as e:
            raise LLMProxyException(
                f"Failed to bind tools to AWS Bedrock LLM: {e}"
            ) from e

    def unbind_tools(self) -> "AWSBedrockLLMProxy":
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
                prompt = value[1]
                role = value[0]

            # Enforce Bedrock format directly
            messages.append({"role": role, "content": [{"text": prompt}]})

        return messages
