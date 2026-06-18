# Setup

Before start with any tutorial we need to setup the environment for the execution of nemantix.

## Step 1: Initialize the Project and Virtual Environment

First, create a main directory for your workspaces and set up a Python Virtual Environment (.venv).
This isolates your project dependencies from your global Python installation.

Open your terminal and run:

### For macOS and Linux:

**1. Create a project folder and navigate into it**

```bash
mkdir nemantix-tutorial
cd nemantix-tutorial
```

**2. Create the virtual environment**

```bash
python -m venv .venv
```

**3. Activate the virtual environment**

```bash
source .venv/bin/activate
```

### For Windows:

**1. Create a project folder and navigate into it**

```dos
mkdir nemantix-tutorial
cd nemantix-tutorial
```

**2. Create the virtual environment**

```dos
python -m venv .venv
```

**3. Activate the virtual environment**

```dos
.venv\Scripts\activate
```

## Step 2: Install Python Packages

**1. Create `requirements.txt`:**

- Create a new file named `requirements.txt` in your project's root directory
- Add the dependency by copying and pasting this exactly:

```plaintext
nemantix[all]
```

**2. Install the packages:**

```bash
pip install -r requirements.txt
```

## Step 3: Setup SSL Keys for Nemantix Verifier

Nemantix uses a Verifier module that requires a public key/certificate (`publickey.crt`)
to ensure the integrity and security of the `.nxs` scripts being executed.

For local development and testing, you can generate a self-signed certificate using OpenSSL.

Run the following commands in your terminal:

```bash
# 1. Create a folder for the keys
mkdir keys

# 2. Generate a standard RSA private key
openssl genrsa -out keys/privatekey.pem 2048

# 3. Extract the public key from it
openssl rsa -in keys/privatekey.pem -pubout -out keys/publickey.crt
```

## Step 4: Configure LLM Credentials via .env

Nemantix needs to know which Large Language Model (LLM) provider you are using (e.g., OpenAI, Anthropic) and your API
key. We securely manage these secrets exclusively through environment variables.

Create a file named `.env` in your project root and add the key that matches the provider you plan to use. Use the
variable names below for each provider:

### OpenAI

```env
OPENAI_API_KEY=sk-OPENAI_KEY_HERE
```

### Anthropic (Claude)

```env
ANTHROPIC_API_KEY=anthropic-KEY_HERE
```

### Azure OpenAI

```env
AZURE_OPENAI_API_KEY=AZURE_KEY_HERE
```

### OpenRouter

```env
OPENROUTER_API_KEY=openrouter-KEY_HERE
```

### Ollama

```env
OLLAMA_API_KEY=OLLAMA_KEY_HERE
```

### Llama.Cpp Remote (Experimental)

```env
LLAMACPP_API_KEY=LLAMACPP_KEY_HERE
```

### Google (Gemini / Google Cloud)

```env
GOOGLE_API_KEY=GOOGLE_API_KEY_HERE
```

### ⚠️ Important: Choosing Your LLM Provider (LLM Proxy)

While `.env` securely stores your API keys, the actual selection of the provider and model (e.g., switching
from the default OpenAI to Anthropic Claude or a local model) is done in your Python code using the `LLMProxyFactory`.

For a detailed guide on how to instantiate the proxy and pass it to your Agent and Expertise,
please read the [LLM Proxy Setup Guide](./llm-setup.md).

## Step 5: Folder Structure Check

Your folder architecture should now look exactly like this:

```
nemantix-tutorial/
│
├── .venv/                   # Python virtual environment
├── keys/
│   ├── publickey.crt        # The SSL cert you generated
│   └── privatekey.pem       # (Optional) Generated alongside the crt
│
├── .env                     # Your LLM API keys
├── main.py                  # (Optional) Only needed if configuring a custom LLM
└── requirements.txt         # Package list
```

--- 

[Go back to the Tutorials Page](./README.md)