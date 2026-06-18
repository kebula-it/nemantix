# LLM Configuration & Custom Providers

By default, the code in this tutorial is configured to use OpenAI (specifically the GPT-5 model).

However, Nemantix is completely model-agnostic. If you want to use a different provider (such as Anthropic Claude,
Google Gemini, or a local open-source model), you don't need to change any of your agent's `.nxs` logic. You simply need
to instantiate a custom LLM Proxy and pass it down to your application.

## The Inheritance Rule

In Nemantix, the LLM configuration follows a strict inheritance pattern:

If an LLM proxy is **not explicitly provided** to an Agent, the Agent will automatically use the same LLM proxy that was
provided to its Expertise.

This means you usually only need to configure the LLM once and attach it to your Expertise.

## How to Change the LLM Provider

To switch providers, you must use the `LLMProxyFactory` to create an `AbstractLLMProxy` and inject it into your Python
setup. The sensitive API key remains in your `.env`, but the structural configuration happens in your code.

Here is how to update your `main.py` to use a different provider:

```python
from pathlib import Path
from nemantix.core import Expertise, Agent
from nemantix.security import Verifier

# 1. Import the LLMProxyFactory, AbstractLLMProxy and Credentials
from nemantix.llm.factory import LLMProxyFactory
from nemantix.llm.abstract_proxy import AbstractLLMProxy
from nemantix.llm.credentials import Credentials


def main() -> None:
    current_folder = Path.cwd()
    verifier = Verifier(current_folder / 'keys/publickey.crt')

    # 2. Set a Credentials Manager
    credentials = Credentials()
    AbstractLLMProxy.set_credentials_manager(credentials)

    # 3. Create your custom LLM Proxy
    # Supported vendors usually include "openai", "anthropic", "google"
    custom_llm = LLMProxyFactory.create_llm_proxy(
        vendor="anthropic",  # Change this to your preferred vendor
        model_name="claude-..."  # Specify the exact model name string
    )

    # 4. Pass the Proxy to the Expertise
    exp = Expertise.from_local_scripts(
        paths=[current_folder / 'nxs/your_script.nxs'],
        verifier=verifier,
        llm=custom_llm  # Injects the model configuration
    )

    # 5. Create the Agent
    # Because we don't explicitly pass an llm here, 
    # the Agent automatically inherits 'custom_llm' from 'exp'!
    agent = Agent(expertise=exp, build_on_start=True)

    # Run your agent
    err, out = agent.run(user_request="Hello, Agent!")
    print(out)


if __name__ == '__main__':
    main()
```

## List of Supported Providers

When calling `LLMProxyFactory.create(vendor="...", model_name="...")`, you must use the correct vendor ID string. Here
is the list of fully supported vendors in Nemantix, along with the corresponding key you must use in your
`.env`.

1. **OpenAI**
    - Vendor id: "openai"
    - Environment variable: openai_api_key

2. **Anthropic (Claude)**
    - Vendor id: "anthropic"
    - Environment variable: ANTHROPIC_API_KEY

3. **Google (Gemini)**
    - Vendor id: "google"
    - Environment variable: GOOGLE_API_KEY

4. **Azure OpenAI**
    - Vendor id: "azure"
    - Environment variable: AZURE_OPENAI_API_KEY

5. **OpenRouter**
    - Vendor id: "openrouter", "open-router", "open_router"
    - Environment variable: OPENROUTER_API_KEY

6. **Ollama**
    - Vendor id: "ollama"
    - Environment variable: OLLAMA_API_KEY

7. **Local models**
    - Vendor id: "local"

8. **Llama.cpp Remote (experimental)**
    - Vendor id: "llama.cpp", "llama-cpp", "llama-cpp-remote"
    - Environment variable: LLAMACPP_API_KEY

--- 

[Go back to the Setup Page](./environment-setup.md)