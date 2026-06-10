# Multi Agentic Todo Manager

In this tutorial, we’ll build a small but fully functional multi-agent TODO system using Nemantix.

The goal is not just to create another CLI application, but to understand a different way of structuring software: one where behavior is delegated to specialized agents, and application logic is defined declaratively through NXS expertises instead of being hardcoded in Python.

# `main.py`

Create a `main.py` file in the root of your project.
This file is where the entire Python layer of the application will live.

It acts as the entry point and the main orchestration layer of the system.

What goes inside `main.py`?

In this file we define and connect all the core runtime components, including:

- The Verifier (for validating NXS scripts)
- The credentials configuration (for LLM access)
- The Toolset (e.g. TodoManagerToolset)
- The Expertise objects (Reader and Writer)
- nThe Agent instances
- The CLI loop that routes user input

In other words, all Python-side orchestration code lives here.


# The Imports — Understanding Dependencies

```python
import sqlite3
from pathlib import Path
from nemantix.core.tools import Toolset, tool
from nemantix.core import Expertise, Agent
from nemantix.security import Verifier
```

What Each Import Does?
|Import	                        | Purpose
|-------------------------------|----------------------------------------
| `sqlite3`	                    | Python's built-in SQLite database library for creating, reading, and managing the TODO database
| `pathlib.Path`                | Object-oriented path handling (safer and more portable than string paths)
| `nemantix.core.tools.Toolset` | Base class for creating reusable tool collections that agents can call
| `nemantix.core.tools.tool`	| Decorator that transforms Python methods into agent-callable operations
| `nemantix.core.Expertise`	    | Orchestrates NXS expertise; loads agent behavior definitions from scripts
| `nemantix.core.Agent`	        | The agent itself—coordinates reasoning, execution, memory, and tool calling
| `nemantix.security.Verifier`  |Cryptographically verifies that NXS scripts are authentic and unmodified

## Understanding `Toolsets` and the `@tool` Decorator

What is a Toolset?
A `Toolset` is a Python class that encapsulates a collection of tools (methods) that agents can call.
It's the bridge between your business logic and agent execution.

```python
class TodoManagerToolset(Toolset):
    def __init__(self, db_uri: str = "todos.db"):
        super().__init__()  # Call parent Toolset constructor
        self._db = sqlite3.connect(db_uri)
        self._init_db()
```

Why Create a Toolset?
Without a Toolset, database logic would be scattered everywhere or embedded directly in agent prompts. With a Toolset, you get:

| Benefit	             | Explanation
|------------------------|----------------------------
| Separation of Concerns | Business logic (databases) stays separate from agent orchestration
| Reusability	         | Multiple agents can use the same toolset without code duplication
| Clarity	             | Each tool has a single, well-defined responsibility
| Type Safety	         | Function signatures define exact inputs and outputs
| Verifiability	         | Tool calls are traceable, auditable, and inspectable

What Does the `@tool` Decorator Do?

```python
@tool
def create_todo(self, text: str) -> bool:
    # This method becomes an agent-callable operation
    pass
```    

The `@tool` decorator transforms a regular Python method into an agent-callable operation by:

- Making it discoverable — The Nemantix framework scans the toolset and identifies all @tool methods
- Creating a callable interface — Agents can reference and invoke this tool by name
- Result: When an agent receives a request like "Create a new todo with text 'Buy groceries'", it can:
  - Recognize that the `create_todo` tool exists
  - Extract the parameter "Buy groceries" from the natural-language request
  - Call the tool with the correct parameter
  - Receive and return the result to the user
  
# The TodoManagerToolset

## Setting Up the Database

```python
class TodoManagerToolset(Toolset):

    def __init__(self, db_uri: str = "todos.db"):
        super().__init__()
        self._db = sqlite3.connect(db_uri)
        self._init_db()
```

What happens:

- `super().init()` — Calls the Toolset parent class constructor to initialize the tool framework
- `sqlite3.connect(db_uri)` — Opens or creates an SQLite database at the specified path (default: "todos.db")
- `self._init_db()` — Creates the Todo table if it doesn't already exist

## Resource Cleanup

```python
def __del__(self):
    self.close()

def close(self):
    self._db.close()
```

Why this matters:

The destructor (`__del__`) ensures that when the toolset is garbage collected, the database connection is properly closed
The `close()` method can be called explicitly if needed.

## Database Initialization

```python
def _init_db(self):
    cu = self._db.cursor()
    
    cu.execute('''
    CREATE TABLE IF NOT EXISTS Todo (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        is_completed INT DEFAULT 0
    )
    ''')
    
    self._db.commit()
    cu.close()
```

## Core Tool 1: `create_todo` — Add a New Task

```python
@tool
def create_todo(self, text: str) -> bool:
    cu = self._db.cursor()
    
    try:
        cu.execute('INSERT INTO Todo (text) VALUES (?)', (text,))
        self._db.commit()
        return True
    except sqlite3.Error as e:
        print(f"Todo Creation error: {e}")
        return False
    finally:
        cu.close()
```

## Core Tool 2: `find_todo` — Retrieve a Single Task

```python
@tool
def find_todo(self, todo_id: int) -> dict:
    cu = self._db.cursor()
    
    cu.execute(
        'SELECT id, text, is_completed ROM Todo WHERE id = ?',
        (todo_id,)
    )
    
    row = cu.fetchone()
    cu.close()
    
    if not row:
        return {
            "status": "error",
            "error": f"No Todo found with ID {todo_id}"
        }
    
    return {
        "status": "success",
        "todo": {
            'id': row[0],
            'text': row[1],
            'is_completed': bool(row[2]),
        }
    }
```

## Core Tool 3: `list_todos` — Retrieve All Tasks

```python
@tool
def list_todos(self) -> dict:
    cu = self._db.cursor()
    cu.execute('SELECT id, text, is_completed FROM Todo')
    rows = cu.fetchall()
    cu.close()
    
    todos = [
        {
            'id': row[0],
            'text': row[1],
            'is_completed': bool(row[2]),
        }
        for row in rows
    ]
    
    return {"status": "success", "todos": todos}
```

## Core Tool 4: `delete_todo` — Remove a Task

```python
@tool
def delete_todo(self, todo_id: int) -> bool:
    cu = self._db.cursor()
    cu.execute('DELETE FROM Todo WHERE id = ?', (todo_id,))
    rows_affected = cu.rowcount
    self._db.commit()
    cu.close()
    
    return rows_affected > 0
```

## Core Tool 5: `complete_todo` — Mark a Task as Done

```python
@tool
def complete_todo(self, todo_id: int) -> bool:
    cu = self._db.cursor()
    cu.execute('UPDATE Todo SET is_completed = 1 WHERE id = ?', (todo_id,))
    rows_affected = cu.rowcount
    self._db.commit()
    cu.close()
    
    return rows_affected > 0
```

## Core Tool 5: `flush_todos` — Remove completed todos

```python
@tool
def flush_todos(self) -> dict:
    cu = self._db.cursor()
    cu.execute('DELETE FROM Todo WHERE is_completed = 1')
    deleted_count = cu.rowcount
    self._db.commit()
    cu.close()
        
    return {"status": "success", "deleted_count": deleted_count}
```



# `main()` - The Entry Point

Now it's time to build the heart of our application: the `main()` function.

This is where we'll:

- Load our agent expertise
- Create the agents
- Start the command-line interface
- Delegate tasks to the appropriate agent

Let's begin with the function declaration:

```python
def main() -> None:
    # the rest of the code
```

## Creating the Agent Expertise

Before we can create an agent, we need to define its **Expertise**.

In Nemantix, an `Expertise` represents the collection of NXS scripts that describe what an agent is capable of doing. You can think of it as the agent's "skill set" or "knowledge domain".

Our TODO application uses two different expertises:

* **Reader Expertise** → can read and search todos
* **Writer Expertise** → can create, update and delete todos

This separation follows a common multi-agent pattern where each agent has a clear responsibility.

### Loading the Required Resources

First, let's prepare a few paths that will be used by both expertises:

```python
current_folder = Path.cwd()

verifier = Verifier(current_folder / 'keys/publickey.crt')
credentials = current_folder / 'credentials.json'
```

**What are these objects?**
- `current_folder`: Gets the current working directory so we can build paths relative to our project.

- `Verifier`

```python
verifier = Verifier(current_folder / 'keys/publickey.crt')
```

Nemantix supports verification of NXS scripts through public-key cryptography.

The verifier checks that the scripts being loaded are trusted and have not been modified unexpectedly.

- `credentials.json`

```python
credentials = current_folder / 'credentials.json'
```

This file contains the credentials used to access the LLM provider.
By default, Nemantix uses GPT models, but you can configure different providers when building your expertise.

---

## Creating the Reader Expertise

Now we can load the NXS script that contains all the logic for reading todos.

```python
reader_exp = Expertise.from_local_scripts(
    paths=[current_folder / 'nxs/reader.nxs'],
    verifier=verifier,
    credentials_path=credentials,
)
```

Let's examine the parameters:

- `paths`: A list of NXS files to load.

In this case we only load one script:
```text
nxs/
└── reader.nxs
```

This script contains all the intents, actions and plans related to reading data from the todo list.

- `verifier`: Used to validate the authenticity of the NXS script before it is executed.
- `credentials_path`: Allows the agent to access the configured language model.

---

## Creating the Writer Expertise

The writer expertise is created in exactly the same way:

```python
writer_exp = Expertise.from_local_scripts(
    paths=[current_folder / 'nxs/writer.nxs'],
    verifier=verifier,
    credentials_path=credentials,
)
```

This expertise loads a different script:

```text
nxs/
└── writer.nxs
```

Unlike the reader, this script contains operations that modify data, such as:

* Creating todos
* Completing todos
* Deleting todos
* Flushing completed tasks

At this point we have defined the capabilities of our two future agents,
but we haven't actually created any agents yet.

In the next section we'll instantiate the agents using these expertises.

## Creating the Agents

At this point we have loaded our two expertises:

* `reader_exp`
* `writer_exp`

However, an expertise only defines capabilities.
To execute those capabilities we need to create actual agents.

An agent is the runtime component responsible for interpreting user requests,
selecting the appropriate workflow, and executing the actions defined by its expertise.

Let's instantiate our agents:

```python
reader_agent = Agent(
    expertise=reader_exp,
    build_on_start=True
)

writer_agent = Agent(
    expertise=writer_exp,
    build_on_start=True
)
```

The relationship between the two concepts can be summarized as follows:

```text
Expertise  ──►  Agent
Knowledge      Runtime
Definition     Execution
```

The `reader_agent` is built using the Reader Expertise and will
therefore only have access to the workflows defined in `reader.nxs`.

Similarly, the `writer_agent` is built from the Writer Expertise
and will execute the workflows contained in `writer.nxs`.

This separation allows each agent to focus on a specific domain of responsibility:

```text
Reader Agent
├── List Todos
└── Find Todo

Writer Agent
├── Create Todo
├── Complete Todo
├── Delete Todo
└── Flush Completed Todos
```

## Routing User Requests

With both agents instantiated, the rest of the application is
responsible for interacting with the user and delegating
requests to the appropriate agent.

The application enters an infinite loop:

```python
while True:
```

At each iteration, a menu is displayed showing the available operations:

```python
print(
    """
    === Agentic TODO Manager ===
    new      - Creates a new todo
    list     - List all todos
    find     - Find a todo by id
    delete   - Delete todo by id
    complete - Mark a todo as completed
    flush    - Remove all completed todos
    exit     - Exit
    """
)
```

The user's choice is then read and normalized:

```python
command = input(": ").strip().lower()
```

This ensures commands are handled consistently regardless of
capitalization or surrounding whitespace.

### Delegating Commands

The CLI itself contains no business logic. Its only responsibility is
translating user actions into natural-language requests and forwarding
them to the correct agent.

Read operations are delegated to the Reader Agent:

```python
err, out = reader_agent.run(
    user_request="List all my todos"
)
```

```python
err, out = reader_agent.run(
    user_request=f"Find the todo with id: {todo_id}"
)
```

Write operations are delegated to the Writer Agent:

```python
err, out = writer_agent.run(
    user_request=f"Create a new todo with this description: {task}"
)
```

```python
err, out = writer_agent.run(
    user_request=f"Delete the todo with id: {todo_id}"
)
```

```python
err, out = writer_agent.run(
    user_request=f"Mark the todo with id {todo_id} as completed"
)
```

```python
err, out = writer_agent.run(
    user_request="Execute flush removing all completed todos"
)
```

This design keeps the Python layer intentionally thin:
rather than implementing TODO operations directly,
it delegates responsibility to the agents and their underlying NXS workflows.

### Handling Responses

Every call to `run()` returns a tuple containing an error and a result:

```python
err, out = agent.run(...)
```

The application simply checks whether an error occurred
and prints the appropriate message:

```python
if err:
    print(f"\n[AGENT ERROR]: {err}")
else:
    print(f"\n[AGENT RESPONSE]: {out}")
```

### Exiting the Application

The loop continues until the user enters:

```plaintext
exit
```

which triggers:

```python
if command == 'exit':
    break
```

Finally, the application is started through the standard Python entry point:

```python
if __name__ == '__main__':
    main()
```

At this stage, the entire TODO manager is operational.
The Python code acts as an orchestration layer, while the actual
behavior of the application is defined by the Reader and Writer
expertises loaded from the NXS scripts.

# Creating the Reader Expertise Script

Before implementing the Reader Expertise, we need to create the NXS script that will define its behavior.

Inside your project directory, create a folder named `nxs`:

```text
project/
├── main.py
├── credentials.json
├── keys/
│   └── publickey.crt
└── nxs/
```

Inside the `nxs` folder, create a file called `reader.nxs`:

```text
project/
├── main.py
├── credentials.json
├── keys/
│   └── publickey.crt
└── nxs/
    └── reader.nxs
```

This is the file that will be loaded by the Reader Expertise:

```python
reader_exp = Expertise.from_local_scripts(
    paths=[current_folder / 'nxs/reader.nxs'],
    verifier=verifier,
    credentials_path=credentials,
)
```

Everything the Reader Agent knows how to do will be defined inside this script.

In our case, the Reader Expertise will be responsible for:

- Finding a specific TODO item
- Listing all TODO items
- Formatting TODO objects into a human-readable format

Open `reader.nxs` and add the following code:

```nxs
from toolset TodoManagerToolset use find_todo, list_todos

action ConvertTodoToStringAction >> Converts a todo into a string representation <<:
  ...
__action

deliberate FindTodoDeliberate when >> the user wants to search, find or read a single specific todo <<:
  ...
__deliberate

deliberate ListTodosDeliberate when >> the user wants to see the list of all todos, show them or list them <<:
  ...
__deliberate
```

In the next sections, we'll break down each part of this script and understand how actions, tools, and deliberates work together to implement the Reader Agent's behavior.


## Implementing the Reader Expertise

Now that we've created the `reader.nxs` file, it's time to implement the expertise that powers our Reader Agent.

The purpose of this expertise is simple: provide the agent with the ability to retrieve and display TODO items.

The complete script is composed of three parts:

1. Tool imports
2. A reusable action
3. Two deliberates

Let's build them one by one.

## Importing the Required Tools

The first thing our expertise needs is access to the TODO operations exposed by our toolset.

At the top of `reader.nxs`, add:

```nxs
from toolset TodoManagerToolset use find_todo, list_todos
```

This statement imports two tools:

| Tool         | Purpose                             |
| ------------ | ----------------------------------- |
| `find_todo`  | Retrieves a specific todo by its ID |
| `list_todos` | Retrieves all todos                 |

Tools are the bridge between an agent and the outside world.

While the agent can reason about requests, it needs tools whenever it wants to interact with external systems such as databases, APIs, files, or services.

In our case, these tools provide access to the TODO storage layer.


## Creating a Reusable Formatting Action

Before implementing the workflows themselves, we'll create a reusable action responsible for formatting TODO items.

Add the following action:

```nxs
action ConvertTodoToStringAction >> Converts a todo into a string representation <<:
```

Actions are reusable blocks of logic.

If you're familiar with traditional programming, you can think of them as helper functions.

Instead of repeating formatting code throughout multiple workflows, we centralize it in a single action.

### Defining the Inputs

The action receives a todo object:

```nxs
in:
  todo (required) >> The todo data <<
__in
```

The object is expected to contain:

- `id`
- `text`
- `is_completed`

### Defining the Output

The action returns a single string:

```nxs
out:
  formatted_todo >> The formatted string representation of the todo <<
__out
```

### Building the Output

Inside the body, we start by constructing the base representation:

```nxs
[[formatted_output] = "Todo #"
                    | to_str([todo:id])
                    | " - "
                    | [todo:text]]
```

For a todo such as:

```json
{
  "id": 3,
  "text": "Write documentation"
}
```

the result becomes:

```text
Todo #3 - Write documentation
```

Next we append the completion status:

```nxs
if [[todo:is_completed]]:
   [[formatted_output] = [formatted_output] | " \[Completed\]"]
else:
   [[formatted_output] = [formatted_output] | " \[Pending\]"]
__if
```

Depending on the value of `is_completed`, the output will include either:

```text
[Completed]
```

or

```text
[Pending]
```

finally we return the final result:

```nxs
return [formatted_output]
```

This action will be reused by all the workflows in our expertise.

## Creating the Find Todo Workflow

The first capability of our Reader Agent is finding a specific TODO item.

To implement it, add the following deliberate:

```nxs
deliberate FindTodoDeliberate when >> the user wants to search, find or read a single specific todo <<:
```

A deliberate defines a workflow that the agent may execute when the corresponding intent is detected.

The `when` clause acts as a semantic description that helps Nemantix determine when the workflow should be considered.

Requests such as:

```text
Find todo 5
```

```text
Show task 10
```

```text
Get todo number 3
```

should all activate this deliberate.

### Extracting the Todo Identifier

The workflow begins by extracting the ID from the user's request.

```nxs
do llm using [
   [prompt] = "Extract only the numeric ID of the todo from this request. No other words. Request: [user_request]"
] producing [[todo_id]]
```

Instead of implementing custom parsing logic, we leverage the language model.

For example:

```text
Show me todo number 12
```

becomes:

```text
12
```

### Retrieving the Todo

Once we have the identifier, we invoke the tool:

```nxs
do tool TodoManagerToolset.find_todo
   using [[todo_id] = to_num([todo_id])]
   producing [[db_result]]
```

The tool performs the actual lookup and returns the result.

### Formatting the Response

If the operation succeeds:

```nxs
if [[db_result:status] == "success"]:
```

we convert the todo into a readable string:

```nxs
do action ConvertTodoToStringAction
```

and return the result.

If the todo doesn't exist, we simply return the error message provided by the tool.

This keeps the workflow simple and focused on orchestration rather than implementation details.

## Creating the List Todos Workflow

The second capability of the Reader Agent is listing all available TODO items.

Add the following deliberate:

```nxs
deliberate ListTodosDeliberate when >> the user wants to see the list of all todos, show them or list them <<:
```

This workflow will be triggered by requests such as:

```text
List my todos
```

```text
Show all tasks
```

```text
What are my todos?
```

### Retrieving All Todos

The first step is calling the appropriate tool:

```nxs
do tool TodoManagerToolset.list_todos
   producing [[db_result]]
```

If the operation succeeds, we initialize the response:

```nxs
[[result] = "Here are your Todos:\n"]
```

### Iterating Through the Collection

Next, we iterate through every returned todo:

```nxs
repeat each [db_result:todos] as [index], [todo]:
```

For each item we reuse the formatting action:

```nxs
do action ConvertTodoToStringAction
   using [[todo] = [todo]]
   producing [[todo_str]]
```

The resulting string is then appended to the final response.

Because all formatting is centralized in a single action, any future change only needs to be implemented once.

# Implementing the Writer Expertise

With the Reader Expertise completed, it's time to implement the second half of our application: the Writer Expertise.

While the Reader Agent is responsible for retrieving information, the Writer Agent handles every operation that modifies the TODO list.

Create a new file called `writer.nxs` inside the `nxs` directory:

```text
project/
├── main.py
├── nxs/
│   ├── reader.nxs
│   └── writer.nxs
```

This file will contain all workflows responsible for:

- Creating todos
- Deleting todos
- Completing todos
- Removing completed todos

Unlike the Reader Expertise, which contains a reusable action for formatting data, the Writer Expertise focuses entirely on orchestration and tool execution.


## Importing the Required Tools

Let's start by importing the tools that perform the actual modifications.

```nxs
from toolset TodoManagerToolset
     use create_todo,
         delete_todo,
         complete_todo,
         flush_todos
```

These tools represent the write operations available to the agent.

| Tool            | Purpose                     |
| --------------- | --------------------------- |
| `create_todo`   | Creates a new todo          |
| `delete_todo`   | Deletes a todo by ID        |
| `complete_todo` | Marks a todo as completed   |
| `flush_todos`   | Removes all completed todos |

The Writer Agent will use these tools whenever it needs to update the underlying TODO storage.


## Creating Todos

The first workflow handles task creation.

```nxs
deliberate CreateTodoDeliberate when >> the user wants to create, add, or insert a new todo <<:
```

The `when` clause describes the situations in which Nemantix should consider this deliberate.

Requests such as:

```text
Create a todo to buy milk
```

```text
Add a task to finish the documentation
```

```text
Insert a new todo called Prepare presentation
```

should all activate this workflow.

### Extracting the Task Description

Before creating a todo, we need to determine what the user wants to add.

The workflow asks the language model to extract the task description:

```nxs
do llm using [
   [prompt] = "Extract the task description from: [user_request]. Return only the string description, no quotes or other words."
] producing [[task]]
```

For example:

```text
Create a todo to finish the Nemantix tutorial
```

becomes:

```text
finish the Nemantix tutorial
```

This allows the workflow to work with natural language without requiring rigid command formats.


### Creating the Todo

Once the description has been extracted, we invoke the tool:

```nxs
do tool TodoManagerToolset.create_todo
   using [[text] = [task]]
   producing [[success]]
```

The extracted text is passed directly to the storage layer.

If the operation succeeds:

```nxs
if [[success]]:
```

the workflow returns:

```text
New Todo successfully created: finish the Nemantix tutorial
```

Otherwise an error message is generated.


## Deleting Todos

The next workflow handles task removal.

```nxs
deliberate DeleteTodoDeliberate when >> the user wants to delete, remove, or eliminate a specific todo <<:
```

This deliberate should be selected for requests such as:

```text
Delete todo 4
```

```text
Remove task number 8
```

```text
Eliminate todo 15
```

### Extracting the Identifier

The workflow first extracts the target ID:

```nxs
do llm using [
   [prompt] = "Extract only the numeric ID to be deleted from the request (or context): [user_request]. Return only the number."
] producing [[todo_id]]
```

For example:

```text
Delete todo number 12
```

becomes:

```text
12
```


### Executing the Deletion

The extracted identifier is then passed to the tool:

```nxs
do tool TodoManagerToolset.delete_todo
   using [[todo_id] = to_num([todo_id])]
   producing [[success]]
```

If the deletion succeeds, the workflow returns a confirmation message.

Otherwise, an error message informs the user that the operation could not be completed.


## Completing Todos

The third workflow allows users to mark tasks as completed.

```nxs
deliberate CompleteTodoDeliberate when >> the user wants to complete, mark as done, or close a todo <<:
```

Examples include:

```text
Complete todo 3
```

```text
Mark task 7 as done
```

```text
Close todo number 11
```

### Identifying the Todo

As with deletion, the workflow first extracts the numeric identifier:

```nxs
do llm using [
   [prompt] = "Extract only the numeric ID to be completed from the request: [user_request]. Return only the number."
] producing [[todo_id]]
```

---

## Updating the Todo

The identifier is then passed to:

```nxs
do tool TodoManagerToolset.complete_todo
   using [[todo_id] = to_num([todo_id])]
   producing [[success]]
```

If the operation succeeds, the workflow responds with:

```text
Great! Todo 3 marked as completed.
```

Otherwise, it reports the failure.

The structure is very similar to the deletion workflow, demonstrating how Nemantix plans can reuse the same orchestration pattern while targeting different tools.

---

## Flushing Completed Todos

The final workflow removes every completed task from the system.

```nxs
deliberate FlushTodosDeliberate when >> the user wants to clean, flush, or remove all already completed todos <<:
```

This deliberate is designed for requests such as:

```text
Flush completed todos
```

```text
Remove all completed tasks
```

```text
Clean up completed todos
```

### Executing the Flush Operation

Unlike the previous workflows, no identifier or task description needs to be extracted.

The workflow directly invokes the tool:

```nxs
do tool TodoManagerToolset.flush_todos
   producing [[db_result]]
```

The tool returns additional information about the operation.

For example:

```json
{
  "status": "success",
  "deleted_count": 5
}
```

### Building the Response

If the operation succeeds:

```nxs
if [[db_result:status] == "success"]:
```

the workflow generates a message such as:

```text
Flush completed. 5 completed todos have been removed.
```

using the value returned by the tool.

If the operation fails, an error message is returned instead.

# Wrap-Up: What We Built

At this point, we’ve completed a full end-to-end Nemantix application using a multi-agent architecture.

What started as a simple CLI TODO manager has evolved into a structured system where behavior is no longer hardcoded in Python, but distributed across agent expertises defined in NXS.

This completes the full implementation of the Nemantix multi-agent TODO system.

You now have a working reference architecture combining:

- Python orchestration layer
- Multiple specialized agents
- Declarative NXS workflows
- Tool-based execution layer

From here, the system can be extended in any direction: more agents, richer tools, or more complex deliberates.

## Full Source Code

You can find the complete project [**Here**](./code/)
