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

# 2. Generate a local self-signed certificate and private key (Valid for 365 days)
# Note: Git Bash or WSL is recommended if you are on Windows.
openssl req -x509 -newkey rsa:2048 -keyout keys/privatekey.pem -out keys/publickey.crt -days 365 -nodes -subj "/CN=localhost"
```


## Step 4: Configure LLM Credentials
Nemantix needs to know which Large Language Model (LLM) provider you are using (e.g., OpenAI, Anthropic) and your API key.
In the Python code, we linked this to a `credentials.json` file.

Create a file named `credentials.json` in your project root with exactly one top-level
key that matches the provider you plan to use. Use the field names below for each provider.

- OpenAI

```json
{
  "openai_api_key": "sk-OPENAI_KEY_HERE"
}
```

- Anthropic (Claude)

```json
{
  "anthropic_api_key": "anthropic-KEY_HERE"
}
```

- Azure OpenAI

```json
{
  "azure_openai_api_key": "AZURE_KEY_HERE"
}
```

- OpenRouter
```json
{
  "openrouter_api_key": "openrouter-KEY_HERE"
}
```

- Ollama
```json
{
  "ollama_api_key": "OLLAMA_KEY_HERE"
}
```


- Google (Gemini / Google Cloud)
```json
{
  "google_api_key": "GOOGLE_API_KEY_HERE"
}
```

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
├── credentials.json         # Your LLM API keys
└── requirements.txt         # Package list
```


--- 

[Go back to the Tutorials Page](./index.md)