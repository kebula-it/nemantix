# 🧭 Nemantix Tutorials

Welcome to the **Nemantix tutorial collection**.

This tutorials are designed to help you learn how to build **tool-augmented AI agents** using the Nemantix framework, starting from simple examples and progressively moving toward more advanced multi-agent systems.

Each tutorial is self-contained and focuses on a specific concept such as:
- Tool usage
- Agent orchestration
- NXS behavioral scripting
- API integration
- Multi-agent architectures


# 📚 What You’ll Learn

By following these tutorials, you will understand how to:

- Build AI agents that can use external tools
- Connect LLM reasoning with real-world APIs
- Structure agent behavior using `.nxs` scripts
- Separate logic (NXS) from execution (Python)

# 🚀 Getting Started (Setup Guide)

Before starting any tutorial, make sure to complete the setup process.

👉 Follow the official setup guide here:

[**📦 Setup Guide**](./environment-setup.md)

This guide will walk you through:
- Creating a Python environment
- Installing dependencies
- Generating security keys for Nemantix
- Configuring LLM credentials
- Preparing your project structure

# 🧠 Nemantix Documentation

To understand the framework in depth, refer to the official documentation:

[**📖 Nemantix Docs**](../docs/)

It includes:
- Core concepts (Agent, Expertise, Toolset)
- NXS syntax reference
- Security model (Verifier system)
- Tool integration patterns


# 📂 Available Tutorials

Each tutorial lives in its own folder and builds a complete working system.

## 🌤️ 1. Tool-Augmented Weather Agent

Learn how to build your first AI agent that:
- Understands natural language
- Extracts structured data (city names)
- Calls a Python tool
- Fetches real-time weather from Open-Meteo
- Returns a formatted response

👉 Folder: [weather-agent-tutorial](./weather-agent-tutorial/tutorial.md)


## 📝 2. Multi-Agent Todo System

Learn how to build a **multi-agent architecture** where:
- One agent reads data (Reader Agent)
- Another modifies data (Writer Agent)
- Responsibilities are separated across expertises
- Agents collaborate through structured toolsets

👉 Folder: [multi-agent-todo-tutorial](./multi-agent-todo-tutorial/tutorial.md)


# 🧩 Project Philosophy

Nemantix is built around a simple idea:

> AI systems should not rely only on prompting — they should be structured, verifiable, and tool-driven.

This means:
- Logic lives in **NXS scripts**
- Execution lives in **Python**
- Intelligence comes from **LLMs**
- Capability comes from **tools**

# 🔥 Recommended Learning Path

If you're new, follow this order:

1. 📘 Weather Agent Tutorial (start here)
2. 🧠 Multi-Agent Todo System
3. 🧩 Build your own custom tool-augmented agent

# 🤝 Contributing

If you'd like to extend these tutorials:
- Add new agent examples
- Improve explanations
- Build advanced multi-agent systems

Pull requests are welcome.
