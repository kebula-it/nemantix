# AGENTS.md
Telegraph style. Root rules only. Read scoped `AGENTS.md` before subtree work. 

## Context
* Nemantix, a platform for Intentware Engineering. 
* Nemantix shifts AI from prompt-based generation to Semantic Agency. 
* The system relies on a custom language called NXS, which compiles to NXC (executable specifications) and is cryptographically signed in to NXV (verifiable scripts). 

## Goals
* Write safe, deterministic, and verifiable Python code for the core platform, or syntactically correct NXS scripts for the agentic workflows.
* Preserve code correctness before quantity.
* When generating NXS scripts pay attention to the guidelines.

## Tech Stack
- **Language**: Python 3.12+
- **Custom Language**: NXS (Nemantix Intentional Language)
- **Key Libraries**: Pydantic (schema definitions), SQLAlchemy (database exploration), SymPy/MoviePy (toolset operations), cryptography (ECDSA signatures).

## Layout
- `src/nemantix/core/`: Contains the fundamental platform components (Agent, Executor, Expertise, Coder, Runtime).
- `src/nemantix/stl/`: Contains the Nemantix Standard Toolset Library (e.g., `WebSearchToolset`, `LocalFileSystemToolset`).
- `src/nemantix/security/`: Cryptographic signing and verification components.
- `src/nemantix/hub/`: Observability tools (Debugger `ndb`, Profiler, Tracer, and Observer).
- `src/nemantix/knowledge_base`: Powers ingestion and Graph-RAG.
- `docs/`: Full documentation of Nemantix.
- `tests/`: unit and integration tests.

## Commands
- **Run Tests**: `pytest tests/`
- **Type Checking**: `mypy src/`

## Quality Rules
- **Python**: 
  - Strictly use type hints for all function arguments and return types.
  - Use Google-style docstrings.
  - When creating custom toolsets in `src/nemantix/stl/`, inherit from `nemantix.core.runtime.Toolset` and wrap exposed methods with the `@tool` decorator.
- **NXS**: 
  - Define `deliberate` blocks to encapsulate specific workflows.
  - Ensure variable assignments use double brackets `[[var] = [other_var]]` (external brackets represent an expression, internal brackets represent variables).
  - Always explicitly declare `@completion: drafted->frozen` (or similar qualifiers) on plans or actions to inform the Coder.
  - Use `frames` for structured schema extraction (e.g., `do llm using ["..."] producing [[content]] as {Ticket}`).

## Boundaries and Hard Rules
- **Never** modify `.nxv` files manually. All signatures must be generated programmatically using `nemantix.security.Signer`.
- **Never** confuse NXS prompt syntax (`>> prompt <<`) with standard string definitions.
- Do not bypass the `Executor` or `Interpreter` when testing action generation. Tool invocations must go through the standard NXS syntax: `do tool ... using ... producing ...`.
- Restrict file operations to `LocalFileSystemToolset` with a sandboxed root directory during tests; do not use raw `os` or `shutil` commands for agent actions.