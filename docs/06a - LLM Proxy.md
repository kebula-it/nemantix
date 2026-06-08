## LLM Proxy Reference

An LLM Proxy is a wrapper for an LLM model (e.g., GPT-5), which is used internally 
by the `Coder` and/or explicitely in a Nemantix script (e.g., when doing `do llm using ["prompt..."]`.) 

### LLM Proxy Instantiation
Proxies are found within the `nemantix.llm` package and are instantiated by 
`LLMProxyFactory` by specifying the `vendor`, `model` name, and proxy-specific `**kwargs`. 

Actually, the `Expertise` provides a shortcut instantiation method:
```python
from nemantix.core import Expertise

llm_proxy = Expertise.get_default_llm('credentials-path/',
                                      vendor='openai',
                                      model='gpt-5-mini')
```

> NOTE: LLM proxies require API_KEYs to be defined in a `credentials.json` file
or in a `.env` file. Each vendor has its own API_KEY.

### Available Vendors

* **OpenAI** (`vendor='openai'`)
* **Azure OpenAI** (`vendor='azure'`): for accessing OpenAI models hosted on Microsoft Azure cloud.
* **Google** (`vendor='google'`)
* **Antrophic** (`vendor='antrophic'`)
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