from nemantix.llm.openai_proxy import OpenAICompatibleProxy


class OpenRouterLLMProxy(OpenAICompatibleProxy):
    """
    LLM proxy for OpenRouter models using the official OpenAI SDK.
    """

    def __init__(
        self,
        model_name: str,
        site_url: str = "https://github.com/kebula-it/nemantix",
        app_name: str = "Nemantix",
        **kwargs,
    ):
        # OpenRouter highly recommends setting these headers
        default_headers = {
            "HTTP-Referer": site_url,
            "X-Title": app_name,
        }

        super().__init__(
            model_name=model_name,
            api_key_name="openrouter_api_key",
            base_url="https://openrouter.ai/api/v1",
            client_kwargs={"default_headers": default_headers},
            **kwargs,
        )
