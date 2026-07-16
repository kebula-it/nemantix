## LLM Proxy Reference

An LLM Proxy is a wrapper for an LLM model (e.g., GPT-5), which is used internally
by the `Coder` and/or explicitely in a Nemantix script (e.g., when doing `do llm using ["prompt..."]`.)

### LLM Proxy Instantiation

Proxies are found within the `nemantix.llm` package and are instantiated by
`LLMProxyFactory` by specifying the `vendor`, `model` name, and proxy-specific `**kwargs`.

Actually, the `Expertise` provides a shortcut instantiation method:

```python
from nemantix.core import Expertise

llm_proxy = Expertise.get_default_llm(vendor='openai',
                                      model='gpt-5-mini')
```

> NOTE: LLM proxies require API_KEYs to be defined in a `.env` file 
> (stored in the project root directory) or directly in your system's environment variables. Each vendor
> has its own API_KEY.

### Available Vendors

* **OpenAI** (`vendor='openai'`)
* **Azure OpenAI** (`vendor='azure'`): for accessing OpenAI models hosted on Microsoft Azure cloud.
* **Google** (`vendor='google'`)
* **Anthropic** (`vendor='anthropic'`)
* **AWS Bedrock** (`vendor='bedrock'`): access Amazon Bedrock models (Claude, Llama, Mistral, etc.)
  via the boto3 Converse API. Requires `boto3>=1.34.0`.
  Authentication follows the standard boto3 credential chain (IAM role, `~/.aws/credentials`, env vars)
  or accepts explicit `aws_access_key_id`, `aws_secret_access_key`, and `aws_session_token` kwargs.
  The `region_name` (or `aws_region`) kwarg selects the AWS region.

  ```python
  llm_proxy = Expertise.get_default_llm(
      vendor='bedrock',
      model='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
      region_name='us-east-1',
  )
  ```

* **OpenRouter** (`vendor='open_router'`): the proxy uses OpenAI-compatible APIs.
* **Local LLM** (`vendor='local'`): use for self-hosted LLM on local machine,
  it uses `llama.cpp` backend for both CPU and GPU inference.

```python
class LocalLLMProxy(AbstractLLMProxy):

    def __init__(self, weights_path: PathLike, context_size=16_384,
                 temperature=1.0, top_p=0.95, max_tokens=2048,
                 device='cpu', dtype='bfloat16', role_start_token='',
                 role_stop_token='', **__):
```

+ \[experimental\] **Remote LLM** (`vendor='llama-cpp'`): use for self-hosted LLM on a *remote* machine
  running a `llama.cpp` server; supports both CPU and GPU inference on GGUF models.
  The proxy uses OpenAI-compatible APIs.

```python
class LlamaCppRemoteLLMProxy(AbstractLLMProxy):

    def __init__(self, model_name: str | None = "auto",
                 base_url: str = "http://localhost:8080/v1",
                 temperature: Optional[float] = None,
                 max_output_tokens: Optional[int] = None,
                 grammar_path: Optional[str] = None,
                 **kwargs: Any):
```