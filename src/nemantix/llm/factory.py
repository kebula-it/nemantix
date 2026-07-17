from pathlib import Path
from typing import Any

from nemantix.llm.abstract_proxy import AbstractLLMProxy, LLMProxyException
from nemantix.llm.anthropic_proxy import AnthropicLLMProxy
from nemantix.llm.azure_openai_proxy import AzureOpenAILLMProxy
from nemantix.llm.google_proxy import GoogleLLMProxy
from nemantix.llm.open_router_proxy import OpenRouterLLMProxy
from nemantix.llm.openai_proxy import OpenAILLMProxy


def get_grammar_path():
    return Path(__file__).parent.parent / "core/nxs_v2_grammar.lark"


class LLMProxyFactory:
    """
    Factory class to create instances of LLM proxies based on the vendor.
    """

    @staticmethod
    def create_llm_proxy(
        vendor: str, model_name: str, grammar_path=None, **kwargs: Any
    ) -> AbstractLLMProxy:
        """
        Creates and returns an instance of an LLM proxy.

        Args:
            vendor: The LLM vendor (e.g., "openai", "google").
            model_name: The specific model name for the vendor.
            **kwargs: Vendor-model specific parameters to pass to the LLM constructor.

        Returns:
            An instance of a class inheriting from AbstractLLMProxy.

        Raises:
            LLMProxyException: If the specified vendor is not supported or credentials manager is not set.
            :param vendor: LLM vendor.
            :param model_name: LLM model name.
            :param grammar_path: path to the lark grammar file.
        """
        if not AbstractLLMProxy.is_credential_manager_set():
            raise LLMProxyException(
                "Credentials manager not set. Call AbstractLLMProxy.set_credentials_manager() first."
            )

        vendor = vendor.lower()
        if vendor == "openai":
            if grammar_path is None:
                grammar_path = get_grammar_path()

            return OpenAILLMProxy(model_name, grammar_path=grammar_path, **kwargs)

        elif vendor == "azure":
            if grammar_path is None:
                grammar_path = get_grammar_path()

            return AzureOpenAILLMProxy(model_name, grammar_path=grammar_path, **kwargs)

        elif vendor == "google":
            return GoogleLLMProxy(model_name, **kwargs)

        elif vendor == "anthropic":
            return AnthropicLLMProxy(model_name, **kwargs)

        elif vendor in ["OpenRouter", "open-router", "open_router"]:
            return OpenRouterLLMProxy(model_name, grammar_path=grammar_path, **kwargs)

        elif vendor in ["llama.cpp", "llama-cpp", "llama-cpp-remote"]:
            from nemantix.experimental.llama_cpp_remote_proxy import (
                LlamaCppRemoteLLMProxy,
            )

            return LlamaCppRemoteLLMProxy(model_name, **kwargs)

        elif vendor == "ollama":
            from nemantix.llm.llama_proxy import LlamaProxy

            return LlamaProxy(model_name, **kwargs)

        elif vendor == "local":
            from nemantix.llm.local_proxy import LocalLLMProxy

            return LocalLLMProxy(model_name, **kwargs)

        elif vendor in ["bedrock", "aws-bedrock", "aws_bedrock"]:
            from nemantix.llm.aws_bedrock_proxy import AWSBedrockLLMProxy

            return AWSBedrockLLMProxy(model_name, **kwargs)

        else:
            raise LLMProxyException(f"Unsupported LLM vendor: {vendor}")
