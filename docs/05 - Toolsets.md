# Understanding Toolsets in Nemantix

To understand how agents interact with their environment, it helps to first understand the concept of **tools**. A tool is a specific, actionable function that an agent can execute to perform a single task—such as sending an email, reading a file, or querying a database.

A **toolset**, then, is a logical collection of these tools grouped together. Toolsets provide ready-to-use tools for integrating with external APIs, internal systems, compute and storage resources, as well as automation and connectors.

In Nemantix, the **Standard Toolset Library (STL)** is the built-in collection of these toolsets, and it is exactly what makes an Agent capable of acting in the world out-of-the-box.

> A special family of **builtin toolsets** (common string, collection, and number operations) is always available in every script without any import. See [Builtin Toolsets](<05b - Builtin Toolsets.md>).

By declaring toolsets within an NXS `deliberate` block, you define exactly which actions the agent is permitted to use. This approach provides several key benefits:

* It constrains actions to known integration points.
* It enables safer execution without implicit external calls.
* It supports widespread reuse across different plans.

During execution, the Runtime invokes these tools, and the Executor factors their availability into its decision-making process.

---

## Building a Custom Toolset

While Nemantix comes with a standard library, you can easily build custom toolsets to interact with your proprietary APIs or specific internal systems.

To create a new toolset, you must follow these three steps:

1. Inherit from the base `Toolset` class.
2. Define an `__init__` method to handle any required setup or configuration, such as loading credentials or setting directory paths.
3. Apply the `@tool` decorator to any specific method that you want to expose to the Nemantix framework or its agents.

### Example: A Simple Text Processing Toolset

Here is an example of creating a custom file named `text_processor.py`:

```python
from nemantix.core.tools import tool, Toolset

class TextProcessorToolset(Toolset):
    """
    A custom toolset for processing and analyzing text.
    """

    def __init__(self, default_language: str = "en"):
        """
        Initialize the toolset with shared configuration.

        Args:
            default_language (str): The language to use for text operations.
        """
        super().__init__()
        self.language = default_language

    @tool
    def count_words(self, text: str) -> str:
        """
        Counts the number of words in a given string.

        Args:
            text (str): The input text.

        Returns:
            str: A formatted string containing the word count.
        """
        word_count = len(text.split())
        return f"The text contains {word_count} words (Language: {self.language})."

    @tool
    def to_uppercase(self, text: str) -> str:
        """
        Converts a given string to uppercase.

        Args:
            text (str): The input text.

        Returns:
            str: The uppercase version of the text.
        """
        return text.upper()

```

### Toolset generation
The coding capabilities of Nemantix allow to generate a fully functioning toolset
starting from a textual description (the micro-prompt) within NXS scripts.
For example, starting from an `.nxs` script:
```
toolset PizzaToolset:
>>> This toolset must bake pizza <<<
__toolset
```
the coding phase produces an `.nxc` script that may look like the following:

```
toolset PizzaToolset:
  >>> This toolset must bake pizza
    class PizzaToolset(Toolset):
        """
        Toolset for baking pizzas.

        This class provides a single tool method to process pizza baking requests.
        It encapsulates the functionality needed to accept a pizza description and
        return the result of baking that pizza.
        """

        def __init__(self) -> None:
            """
            Initialize the Pippo toolset.

            No external configuration is required for this toolset.
            """
            # No configurable defaults were provided; initialize internal state if needed.
            self.ready: bool = True

        @tool
        def pizza(self, order: str) -> str:
            """
            Bake a pizza based on the provided order string.

            Args:
                order (str): A string describing the pizza to bake (e.g., type, toppings).

            Returns:
                str: A message describing the completed pizza baking result.
            """
            if not self.ready:
                return "Oven not ready."
            # Simulate baking process (simple synchronous representation).
            baked_pizza = f"Baked pizza: {order}"
            return baked_pizza
  <<<
__toolset
```

In general, the toolset coding is driven by both the micro-prompt description and usages of
the methods annotated with `@tool`.

---
## Using a toolset in NXS

In NXS, you bring toolsets into your script by declaring them at the top level or within a `deliberate` to define which external capabilities your actions are allowed to use. You import specific tools using the `from toolset [Name] use [tool_name]` syntax, or you can use an asterisk (`*`) to import all targets. If your toolset requires initialization, you can pass arguments using the `with` keyword and assign it a custom alias using the `as` keyword. Once imported, you execute these tools inside an action's `body:` block using the `do` statement, passing inputs via the `using` keyword and capturing the results via the `producing` keyword.

For a deeper dive into the exact syntax, advanced expressions, and block formatting, please refer to [`03 - NXS language.md`](./03%20-%20NXS%20language.md) file, specifically the [**NXS Syntax**](./03%20-%20NXS%20language.md#requires) and [**Tool / action / deliberate calls (`do`)**](./03%20-%20NXS%20language.md#dostatement) sections.

---

## Dynamically Loading Tools: The `get_tool` Method

The `Toolset.get_tool()` method is the standard mechanism in Nemantix for dynamically loading and utilizing a tool. It automatically handles the instantiation of the parent Toolset class and returns the specific tool function you requested so that it is ready to be executed.

When calling `get_tool()`, you must provide two primary arguments:

1. **`tool_name`**: A string that combines the Toolset class name and the tool's method name, separated by a dot (e.g., `"WebSearchToolset.search_web"`).
2. **`instance_args`**: A tuple containing the exact arguments required by the Toolset's `__init__` method to properly set up its internal state.

### Code Example

Below is an example utilizing a hypothetical `WebSearchToolset` that requires a `region` string for initialization.

```python
from nemantix.core.tools import Toolset

# 1. Fetch the tool dynamically
search_tool = Toolset.get_tool(
    tool_name="WebSearchToolset.search_web",

    # instance_args must match the __init__ parameters of WebSearchToolset.
    # IMPORTANT: Even if there is only one argument, it MUST be a tuple
    # (notice the trailing comma).
    instance_args=("us-en",)
)

# 2. Execute the tool
# Now you pass the arguments required by the actual @tool method
results = search_tool(query="Latest advancements in AI", max_results=3)

# 3. View the output
for result in results:
    print(f"Title: {result['title']}")
    print(f"Link: {result['link']}\n")

```

---

## The Nemantix Standard Toolset Library (NSTL)

The Nemantix Standard Toolset Library (NSTL) is located in the `nemantix.stl` module. It provides a comprehensive suite of pre-built toolsets that are ready to be imported directly into your NXS scripts.

By declaring these pre-existing toolsets in your `deliberate` block (e.g., `from toolset WebSearchToolset use search_web`), you efficiently constrain your actions to known integrations, enable safer agent execution, and promote highly reusable behaviors across your agent's overall plans.

### Available Toolsets Overview

Below is a summary of the toolsets available out-of-the-box, organized by macro-category.

#### File Operations

Toolsets focused on safely reading, writing, moving, and managing files across different environments.

| Toolset                       | Description                                                        |
|-------------------------------|--------------------------------------------------------------------|
| **`LocalFileSystemToolset`**  | Safe file system operations within a securely sandboxed directory. |
| **`RemoteFileSystemToolset`** | Interacting with remote file servers via FTP, FTPS, or SFTP.       |

---

#### Communication & Messaging

Toolsets built for asynchronous and synchronous communication with users or other systems.

| Toolset                | Description                                             |
|------------------------|---------------------------------------------------------|
| **`EmailToolset`**     | Sending and reading emails via SMTP and IMAP protocols. |
| **`MessagingToolset`** | Interacting with Telegram Bots to send direct messages. |

---

#### Web & Networking

Toolsets designed for interacting with external web services, APIs, and the broader internet.

| Toolset                | Description                                                               |
|------------------------|---------------------------------------------------------------------------|
| **`RequestsToolset`**  | Performing stateless HTTP requests with explicit authentication handling. |
| **`WebSearchToolset`** | Searching the live web and news without requiring an API key.             |

---

#### Data & Computation

Toolsets meant for deep analytical work, complex calculations, and structured data retrieval.

| Toolset                  | Description                                                                  |
|--------------------------|------------------------------------------------------------------------------|
| **`MathSolverToolset`**  | Advanced symbolic mathematical calculations using the SymPy library.         |
| **`SqlExplorerToolset`** | Schema inspection and query execution on SQL databases utilizing SQLAlchemy. |

---

#### Media Processing

Toolsets dedicated to editing, converting, and analyzing audio, video, and image files.

| Toolset                     | Description                                    |
|-----------------------------|------------------------------------------------|
| **`AudioProcessorToolset`** | Audio processing using MoviePy (FFMPEG-based). |
| **`MediaToolset`**          | Media processing for both images and videos.   |

---

> **Note**: For a deep dive into the specific classes, methods, parameters, and return types for the Base `Toolset` class and all NSTL components, please refer to the external file: [`05a - Toolset API Reference.md`](./05a%20-%20Toolset%20API%20Reference.md).

---

Next: [Agents](./06%20-%20Agents.md)
