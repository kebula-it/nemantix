# 🌤️ Tool-Augmented AI Agent with Nemantix + Open-Meteo

## Introduction

The goal of this project is to build a Tool-Augmented AI Agent.

Large Language Models (LLMs) are powerful at understanding and generating text, but they don’t have direct access to real-world data like weather APIs, databases, or live systems.

To solve this, we combine:

- Nemantix → to orchestrate agent behavior using NXS scripts
- Python tools → to connect the agent to external APIs
- Open-Meteo API → to fetch real-time weather data

This creates a system where:

- The LLM decides what to do,
- Python executes real actions,
- Nemantix coordinates how everything connects.


## System Overview

Our agent follows this pipeline:

```text
User Request → Nemantix Deliberate → LLM extraction → Tool execution → Weather API → Response formatting
```

Example: `"What's the weather in Milan?"`

Becomes:
- Extract city → "Milan"
- Call tool → fetch coordinates + weather
- Return structured result
- Format final response

## Update Dependencies

Before writing any code, we need to make sure our environment includes all required packages.

In the `requirements.txt` file you created during setup, add **requests**:

```txt
nemantix[all]
requests
```

Why do we need `requests`?

We use it to:
- Make HTTP calls to the Open-Meteo API
- Fetch real-time weather data
- Communicate with external services from Python

Without it, our tool layer would not be able to retrieve any live data.

Once added, install everything (if you haven’t already):

```bash
pip install -r requirements.txt
```

## Create main.py

Now we move to the core of the project.

Create a new file in the root of your project: `main.py`

This file is the entry point of the entire application.

It contains:
- The tool definitions (Python logic)
- The Nemantix agent setup
- The expertise loading (NXS scripts)
- The interactive CLI loop

In short:
>
> `main.py` is where everything comes together and the system becomes executable.
>

## Nemantix Side Setup (NXS Files)

Alongside the Python code, Nemantix requires a separate folder for agent logic.

Inside your project root, create a folder called:

```
nxs/
```

This folder will contain all Nemantix scripts that define how the agent behaves.

### Create the NXS file

Inside the nxs folder, create a file called:

```
meteo-deliberate.nxs
```

So your structure should look like this:
```
project/
├── main.py
├── requirements.txt
├── credentials.json
├── keys/
└── nxs/
    └── meteo-deliberate.nxs
```

## What is a Toolset?

Before continuing, we need to introduce an important concept used in Nemantix: **Toolsets**.

A Toolset is a Python class that groups together one or more functions (called tools) that an AI agent is allowed to use.

Each tool inside a Toolset:
- Performs a real action (like calling an API or querying a database)
- Is exposed to the agent through the @tool decorator
- Can be called from inside an NXS script

In simple terms: A Toolset is the bridge between the AI agent and real-world actions.

For example, in this project, we will create a Toolset that:
- Calls the Open-Meteo API
- Retrieves weather data for a given city

## Imports

```python
import requests
from pathlib import Path

from nemantix.core import Expertise, Agent
from nemantix.core.tools import Toolset, tool
from nemantix.security import Verifier
```

## Tool Layer

We define a toolset that connects the agent to real-world weather data.

```python
class OpenMeteoTools(Toolset):

    @tool
    def get_weather_by_city(self, city_name: str) -> dict:
```

This tool does 2 things:

- Converts city → coordinates
- Fetches weather using Open-Meteo API

### Step 1: Geocoding (City → Coordinates)

Open-Meteo requires latitude/longitude.

```python
geocode_url = "https://geocoding-api.open-meteo.com/v1/search"

geo_params = {
    "name": city_name,
    "count": 1,
    "language": "it",
    "format": "json"
}

geo_response = requests.get(geocode_url, params=geo_params)
geo_data = geo_response.json()
```

If no city is found:

```python
if not geo_data.get("results"):
    return {
        "status": "error",
        "error": f"City not found: {city_name}"
    }
```

### Step 2: Weather API Call

```python
location = geo_data["results"][0]

lat = location["latitude"]
lon = location["longitude"]
resolved_city = location["name"]

weather_url = "https://api.open-meteo.com/v1/forecast"

weather_params = {
    "latitude": lat,
    "longitude": lon,
    "current_weather": "true"
}

weather_response = requests.get(weather_url, params=weather_params)
weather_data = weather_response.json()
```

### Step 3: Return structured result

```python
if "current_weather" in weather_data:
    current = weather_data["current_weather"]

    return {
        "status": "success",
        "city": resolved_city,
        "temperature_celsius": current["temperature"],
        "windspeed_kmh": current["windspeed"],
        "time": current["time"]
    }
else:
    return {
        "status": "error",
        "error": "No data available for that locality."
    }
```

## Nemantix Layer (`meteo-deliberate.nxs`)

This is where the AI behavior is defined.

### Tool Import

```nxs
from toolset OpenMeteoTools use get_weather_by_city
```

### Deliberate: Weather Intent
```
deliberate CheckWeatherDeliberate when >> user asks for weather <<:
```

This tells Nemantix:

`“Run this block when the user is asking about weather”`

### Step 1: Extract City Name (LLM call)

```nxs
do llm using [
  [prompt] = "Extract only the city name from: [user_request]"
] producing [[city_name]]
```

This step converts messy text into structured data.

Example:

`"What’s the weather in Rome tomorrow?" → Rome`


### Step 2: Call Python Tool
```nxs
do tool OpenMeteoTools.get_weather_by_city
   using [[city_name] = [city_name]]
   producing [[weather_data]]
```

Now the agent leaves the LLM and executes real code.

### Step 3: Format Response

```nxs
if [[weather_data:status]] == "success":
    [[result] =
        "Weather in " | [city_name] |
        ": " | to_str([weather_data:temperature_celsius]) |
        " C, wind " | to_str([weather_data:windspeed_kmh]) |
        " km/h"
    ]
else:
    [[result] = "Error: " | [weather_data:error]]
```
### Step 4: Return result

```
return [result]
```

## Python Runtime Loop (`main.py`)

```python
def main():
    current_folder = Path.cwd()

    exp = Expertise.from_local_scripts(
        paths=[current_folder / "nxs/meteo-deliberate.nxs"],
        verifier=Verifier(current_folder / "keys/publickey.crt"),
        credentials_path=current_folder / "credentials.json",
    )

    agent = Agent(expertise=exp, build_on_start=True)

    while True:
        prompt = input("City (:exit to quit): ")

        if prompt == ":exit":
            break

        err, out = agent.run(
            user_request=f"Fetch weather for {prompt}"
        )

        if err:
            print("Error:", err)
        else:
            print(out)
```

### Step 1. Loading the Project Environment

The first thing `main()` does is prepare the environment:

- It locates the current project folder
- It loads security keys (used by Nemantix to verify scripts)
- It loads LLM credentials (to access the model)

This step ensures the system is **secure and properly configured** before the agent runs.

### Step 2. Creating the Verifier

Nemantix uses a **Verifier** to ensure that NXS scripts have not been modified or corrupted.

In practice, this means:

> Only trusted agent logic is executed.

So `main()` loads a public key and passes it to the system to validate all NXS files before execution.

---

### Step 3. Loading the Expertise (The Agent Brain)

The most important step is loading the **Expertise**.

An *Expertise* in Nemantix is:

> A packaged set of NXS scripts that define what an agent can do and how it behaves.

In this project:

- `meteo-deliberate.nxs` contains the weather agent logic

So `main()` tells Nemantix:

> “Load this file and use it as the brain of the agent.”

At this point:
- The agent knows **when to respond**
- The agent knows **what tools exist**
- The agent knows **how to structure reasoning**

But it still cannot run — it needs an execution wrapper.


### Step 4. Creating the Agent

Once the Expertise is loaded, `main()` creates an **Agent**.

The Agent is the runtime object that:

- Receives user input
- Passes it to the NXS logic
- Executes tool calls (Python functions)
- Returns the final response

So you can think of it as:

> The Agent is the “engine that runs the brain”.

Without it:
- The NXS file is just a definition
- Nothing actually executes

---

## Step 5. Starting the Interactive Loop

After setup, `main()` enters an infinite loop:

```text
User → input → Agent → response → print → repeat
```


---

# 🧾 Wrap-Up

At this point, we have everything needed to understand the structure of a **tool-augmented AI agent built with Nemantix**.

Even though the system looks simple from the outside, it is actually composed of multiple well-separated layers that each have a very specific responsibility.

---

## 🧠 What We Built

We created a system where:

- The **user** asks a natural language question (e.g. “What’s the weather in Rome?”)
- The **LLM** extracts structured information (like the city name)
- A **Nemantix NXS script** decides what to do with that information
- A **Python tool** retrieves real data from an external API
- The system returns a clean, human-readable response


From here, the system can be extended in any direction: more agents, richer tools, or more complex deliberates.

## Full Source Code

You can find the complete project [**Here**](./code/)