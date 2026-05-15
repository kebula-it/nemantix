# EventHub

The EventHub is responsible to propagate events to subscribed components
like the Debugger, Profiler, etc., to enable full observability of the 
execution of Nemantix Scripts.

Main observability components:
* **Debugger**: tracks the executed statements providing pdb-like commands; 
* **Profiler**: records the duration of calls;
* **Tracer**: traces the agent execution to explain its behavior.
* **Observer**: monitors hardware- and agent-related metrics.

Observability components can be passed directly to `Expertise` via `observers=`;
the hub is created and wired internally:
```python
from nemantix.core import Expertise
from nemantix.hub import Debugger, Profiler, Tracer

Expertise.from_local_scripts(paths=['...'], verifier=...,
                             observers=[Debugger(), Profiler(), Tracer()])
```

## Debugger
The debugger tracks the execution of a Nemantix script line-by-line. 
When either a `@breakpoint` annotation or a raised error is encountered,
the interactive debugging interface (`ndb`) is enabled.

### Breakpoints

Use the `@breakpoint` intentable annotation on any statement to pause execution there:

```nxs
action my_action:
    body:
        @breakpoint
        [[x] = 42]
        ...
    __
__action
```

Conditional breakpoints are also supported — the debugger only activates when
the condition is truthy:

```nxs
@breakpoint: [[x] > 10]
[[x] = [x] + 1]
```

### ndb

`ndb` is the Nemantix interactive debugger. When it activates, it prints the
current location and statement, then waits for commands:

```
> ticket.nxs(22) [preprocess_ticket]
-> [[structured_ticket] = ("code": ..., "lang": [lang], "content": [content])]
Commands:
    q/quit, c/continue, n/next, s/step, r/return,
    p/print [var], h/help, e/eval [expr],
    l/list [line [end]]
(ndb): 
```

Supported commands (short form in parentheses, optional argument in square brackets):

- **quit** (`q`): exits and disables the debugger for the current execution.
- **continue** (`c`): resumes execution until the next breakpoint or error.
- **next** (`n`): steps to the next statement, skipping into called actions (*step over*).
- **step** (`s`): steps into the next statement, descending into called actions (*step into*).
- **return** (`r`): continues until the current action returns (*step out*).
- **print** (`p`) `[var]`: prints all variables in the operational memory. If `var` is given, prints only that variable.
- **eval** (`e`) `[expr]`: evaluates a Nemantix expression or `do` statement inline. The result is printed and any side-effects (variable assignments) are applied to the running context.
- **list** (`l`) `[line [end]]`: lists the source lines around the current position. Optionally pass a center `line`, or a `line` and `end` to show a specific range.
- **help** (`h`): prints the command reference.

> Pressing Enter without typing a command repeats the last command.

Example session:

```
> ticket.nxs(22) [preprocess_ticket]
-> [[structured_ticket] = (...)]
(ndb): p
Operational Memory:
  text = "Ticket-ABC fix the infra"
  prefix_size = 8
  lang = "en"
  content = "fix the infra"
(ndb): e [[x] = 99]
99
(ndb): p x
x = 99
(ndb): l
  17    do llm using [...] producing [[lang]]
  19    do llm using [...] producing [[content]]
  21  
  22 -> [[structured_ticket] = (...)]
  24    return [structured_ticket]
(ndb): n
(ndb): q
closing ndb
```

### Enabling the Debugger

```python
from nemantix.core import Expertise
from nemantix.hub import Debugger

exp = Expertise.from_local_scripts(paths=['...'], verifier=...,
                                   observers=[Debugger()])
```

## Profiler
The profiler tracks **coding** (of toolsets, actions, and deliberates) and **execution events** (like action, tool, and builtin calls)
to measure their respective elapsed time, while maintaining the call stack.

The profiler offers a basic `print()` that shows the execution statistics:
```
======================================================================
PROFILER REPORT
======================================================================
Coding:
▶ SummarizeSupportTicket [deliberate]
  [Total: 88.29s | Attempts: 2]
▶ GenerateTicket [deliberate]
  [Total: 57.28s | Attempts: 1]
----------------------------------------------------------------------
Execution:
▶ generate_ticket [action]
   [Total: 29424.24ms | Self:    1.06ms]
  ├─ preprocess_ticket [action]
     [Total: 29423.18ms | Self:    1.04ms]
    ├─ size [builtin]
       [Total:    0.03ms | Self:    0.03ms]
    ├─ llm [builtin]
       [Total: 9658.74ms | Self: 9658.74ms]
    ├─ llm [builtin]
       [Total: 19763.32ms | Self: 19763.32ms]
    ├─ substring [builtin]
       [Total:    0.05ms | Self:    0.05ms]
Total execution time:   29.42s
======================================================================
```

### Selective profiling with `@profile`

By default, the profiler collects data for every call. Pass `profile_mode='annotated'` to
restrict the report to code explicitly marked with the `@profile` annotation:

```python
profiler = Profiler(profile_mode='annotated')
```

`@profile` is supported on **deliberates**, **plans**, and **actions** — the elements that
emit `CALL_ENTER`/`CALL_EXIT` events and therefore appear in the execution tree.

Mark an individual **action** to profile just that call subtree:

```nxs
@profile
action preprocess_ticket:
    body:
        ...
    __
__action
```

Mark a **plan** to profile the plan's execution (excluding the deliberate's own setup overhead):

```nxs
deliberate GenerateTicket:
    @profile
    plan:
        ...
```

Mark a **deliberate** to profile both the deliberate node and its full plan execution as one tree:

```nxs
@profile
deliberate GenerateTicket:
    ...
```

#### Conditional profiling

`@profile` accepts an expression. The annotation is only activated when the expression
evaluates to truthy, enabling environment- or data-driven profiling:

```nxs
@profile: [[debug_mode]]
action preprocess_ticket:
    ...
```

```nxs
@profile: [[ticket_size] > 1000]
deliberate GenerateTicket:
    ...
```

#### Output

In annotated mode the report only shows the marked subtrees. Nodes carrying the `@profile`
annotation are labelled `[@profile]` in the output:

```
======================================================================
PROFILER REPORT [annotated mode]
======================================================================
...
Execution:
▶ GenerateTicket [deliberate] [@profile]
   [Total: 29424.24ms | Self:    1.06ms]
  ├─ preprocess_ticket [action]
     [Total: 29423.18ms | Self:    1.04ms]
    ├─ llm [builtin]
       [Total: 9658.74ms | Self: 9658.74ms]
    ├─ llm [builtin]
       [Total: 19763.32ms | Self: 19763.32ms]

Total execution time:   29.42s
======================================================================
```

## Tracer
The tracer provides an **interactive, visual timeline** of the agent execution.
It extends the Profiler and organizes recorded events into two top-level sections:

- **Coding** – deliberate and action compilation steps, with attempt counts.
- **Execution** – the runtime call tree (actions, tools, builtins, …).

Each section can be expanded level-by-level to drill down into nested calls.

### Interactive viewer
Calling `tracer.print()` launches the interactive session. Each node is rendered
as a proportional time bar using `█` (active) and `░` (idle):

```
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
TRACER > Execution GenerateTicket
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
     0.00ms                                                                                             15797.97ms
     ├───────────────────────────────────────────────────────────────────────────────────────────────────────────┤

[0]  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     ComposeRawTicket [action]  0.00ms → 0.24ms  (0.24ms)  ▶ 1 call

[1]  ░███████████████████████████████████████████████████████████████████████████████████████████████████████████░
     preprocess_ticket [action]  0.27ms → 15797.71ms  (15797.44ms)  ▶ 4 calls

[2]  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
     AssembleTicket [action]  15797.75ms → 15797.97ms  (0.22ms)  ▶ 1 call

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  [idx] expand │ b: back │ f <ms> <ms> / fc: time filter │ ft <tag> / fct: type filter │ fca: clear all │ q: quit
  > 
```

Supported commands (short forms shown in parentheses):

- **[idx]**: expand node `idx`, navigating into its children.
- **b** (back, ..): go back one level to the parent. At the top level it is a no-op.
- **f \<start_ms\> \<end_ms\>**: apply a **time filter** – only nodes whose interval overlaps `[start_ms, end_ms]` are shown.
- **fc** (clear time): remove the active time filter.
- **ft \<tag\>**: apply a **type filter** – only nodes whose `type` tag matches `tag` are shown (e.g. `ft builtin`, `ft action`).
- **fct** (clear type): remove the active type filter.
- **fca** (clear all): remove all active filters.
- **q** (quit, exit): close the tracer.

Active filters are displayed below the breadcrumb path so it is always clear what is being shown:
```
  [Active Filters: Time: 1000.00ms - 50000.00ms, Type: 'builtin']
```

To enable the tracer:

```python
from nemantix.core import Expertise
from nemantix.hub import Tracer

exp = Expertise.from_local_scripts(paths=['...'], verifier=...,
                                   observers=[Tracer()])
```

## Observer
The `Observer` monitors both hardware and agent metrics over a running session:
+ **Hardware metrics**:
  + Execution platform: local or remote machine;
  + Average CPU utilization in %;
  + RAM usage in MBytes;
  + Total number of I/O operations;
  + Total volume in KBytes of sent and received network packets.
+ **Agent metrics**:
  + Number of requests submitted to the agent;
  + LLM invocations: grouped by proxy, distinguishing between internal and 
  explicit usages (e.g., `do llm using [...]`).
  + Runtime `nxc` codings.
  + Tool calls frequency;
  + Error count.

Moreover, the observer collects the system logs for later storage and analysis.

### Monitoring an Agent Running Session
The monitoring of hardware-related metrics can be enabled using the `with` 
Python syntax on an `Agent` instance:
```python
from nemantix.core import Agent, Expertise
from nemantix.hub import Observer

observer = Observer()
expertise = Expertise.from_local_scripts(..., observers=[observer])
agent = Agent(expertise, ...)

# starts the running session monitoring 
with agent:
    _, out = agent.run(user_request='...')
    
    # you can monitor multiple requests in the same session
    _, out = agent.run(user_request='another request...')

# prints the monitoring report
observer.print()
```

### Monitoring Report
An example observer report (obtained by calling `observer.print()`) can be like:
```
==================================================
📊 OBSERVER REPORT
==================================================
💻 HARDWARE METRICS (Session Window)
  ├─ Environment:   Local Machine (Ubuntu 24.04.4 LTS)
  ├─ CPU Avg Usage: 4.3%
  ├─ RAM End State: 206.33 MB
  ├─ Disk I/O:      554 Reads | 42 Writes
  └─ Network:       177.09 KB Down | 202.57 KB Up
  
🤖 AGENT METRICS
  ├─ User Requests: 1
  ├─ LLM Invocations: 6
    ─ OpenAI gpt-5-mini: 6 calls (internal: 2)
  ├─ Runtime codings: 0
  └─ Errors Encountered: 0
 
🛠️  TOOL UTILIZATION
  └─ No tools utilized.
==================================================
```
### Collect Logs in a Database
If the observer is instantiated with a database connector (`DBConnector`), it forwards the 
application logs to the provided database:

```python
from nemantix.hub import Observer, ObserverLogModel
from nemantix.common.connectors import DBConnector


# in-mem SQLite example
sql_connector = DBConnector.sqlite_in_mem()

# OR postgres DB:
db_url = 'postgresql+psycopg://user:password@localhost:5432/nemantix'
postgres_connector = DBConnector(database_url=db_url)

observer = Observer(connector=sql_connector)

# execute agent
with agent:
    ...

# print the recorded logs from the DB
with observer.connector.get_session() as session:
    logs = session.query(ObserverLogModel).all()

    for log in logs:
        print(f"[{log.timestamp}] {log.message}")
```

---

## Usage example
Complete example of debugging, profiling, tracing and observing:
```python
from nemantix.core import Agent, Expertise
from nemantix.security.verifier import DebugVerifier
from nemantix.hub import Debugger, Profiler, Tracer, Observer

profiler = Profiler()
tracer = Tracer()
observer = Observer()

exp = Expertise.from_local_scripts(paths=['examples/ticket.nxs'],
                                   verifier=DebugVerifier(),
                                   observers=[Debugger(), profiler, 
                                              tracer, observer],
                                   credentials_path='credentials.json')

agent = Agent(expertise=exp, build_on_start=True)

with agent:
  err, out = agent.run(user_request='Generate the ticket for issue '
                                    '<fix necessary for infrastructure orchestration code>')
# reports
profiler.print()
tracer.print()
observer.print()
```

Next: [Knowledge Base](08%20-%20Knowledge%20Base.md)
