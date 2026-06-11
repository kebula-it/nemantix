# LLM Configuration & Custom Providers

By default, the code in this tutorial is configured to use OpenAI (specifically the gpt-5 model).

However, Nemantix is completely model-agnostic. If you want to use a different provider (such as Anthropic Claude, Google Gemini, or a local open-source model), you don't need to change any of your agent's `.nxs` logic. You simply need to instantiate a custom LLM Proxy and pass it down to your application.

## The Inheritance Rule
In Nemantix, the LLM configuration follows a strict inheritance pattern:

If an LLM proxy is **not explicitly provided** to an Agent, the Agent will automatically use the same LLM proxy that was provided to its Expertise.

This means you usually only need to configure the LLM once and attach it to your Expertise.

## How to Change the LLM Provider
To switch providers, you must use the `LLMProxyFactory` to create an `AbstractLLMProxy` and inject it into your Python setup. The sensitive API key remains in your `credentials.json`, but the structural configuration happens in your code.

Here is how to update your `main.py` to use a different provider:

```python
from pathlib import Path
from nemantix.core import Expertise, Agent
from nemantix.security import Verifier

# 1. Import the LLMProxyFactory
from nemantix.llm.factory import LLMProxyFactory

def main():
    current_folder = Path.cwd()
    verifier = Verifier(current_folder / 'keys/publickey.crt')
    credentials_path = current_folder / 'credentials.json'
    
    # 2. Create your custom LLM Proxy
    # Supported vendors usually include "openai", "anthropic", "google"
    custom_llm = LLMProxyFactory.create_llm_proxy(
        vendor="anthropic",                  # Change this to your preferred vendor
        model_name="claude-..."              # Specify the exact model name string
    )

    # 3. Pass the Proxy to the Expertise
    exp = Expertise.from_local_scripts(
        paths=[current_folder / 'nxs/your_script.nxs'],
        verifier=verifier,
        credentials_path=credentials_path, # Injects the API key from JSON
        llm=custom_llm                     # Injects the model configuration
    )

    # 4. Create the Agent
    # Because we don't explicitly pass an llm_proxy here, 
    # the Agent automatically inherits 'custom_llm' from 'exp'!
    agent = Agent(expertise=exp, build_on_start=True)
    
    # Run your agent
    err, out = agent.run(user_request="Hello, Agent!")
    print(out)

if __name__ == '__main__':
    main()
```

## List of Supported Providers

When calling `LLMProxyFactory.create(vendor="...", model_name="...")`, you must use the correct vendor ID string. Here is the list of fully supported vendors in Nemantix, along with the corresponding key you must use in your `credentials.json`.

1. **OpenAI**
    - Vendor ID: "openai"
    - Credentials Key: "openai_api_key"

2. **Anthropic (Claude)**
    - Vendor ID: "anthropic"
    -  Credentials Key: "anthropic_api_key"

3. **Google (Gemini)**
    - Vendor ID: "google"
    - Credentials Key: "google_api_key"

4. **Azure OpenAI**
    - Vendor ID: "azure"
    - Credentials Key: "azure_openai_api_key"

5. **OpenRouter**
    - Vendor ID: "OpenRouter", "open-router", "open_router"
    - Credentials Key: "openrouter_api_key"

6. **Ollama**
    - Vendor ID: "ollama"
    - Credentials Key: "ollama_api_key"

7. **Local Models**
    - Vendor ID: "local"

8. **Llama.Cpp Remote (Experimental)**
    - Vendor ID: "llama.cpp", "llama-cpp", "llama-cpp-remote"
    - Credentials Key: "llamacpp_api_key"

--- 

[Go back to the Setup Page](./environment-setup.md)