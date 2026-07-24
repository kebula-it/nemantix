# Builtin Toolsets

Large Language Models are heavily trained on Python, so when the **Coder** translates an
`.nxs` specification into executable `.nxc` code it naturally reaches for Python idioms such as
`text.split(" ")`, `" ".join(parts)`, `s.upper()`, or `sorted(items)`. NXS has none of these:
there is no `.method()` syntax on values, and the language ships only a small set of builtins.

**Builtin toolsets** close this gap. They are ordinary [toolsets](<05 - Toolsets.md>) that provide
the common string, collection, and number operations the Coder keeps reaching for — but with two
special properties:

1. **Always available.** They are auto-imported into every script. You never write a
   `from toolset ... use *` for them.
2. **Advertised to the Coder.** Their tools (names, signatures, and docstrings) are injected into
   every coding prompt, so the LLM knows they exist and how to call them.

They are still *tools*, so they are invoked with the ordinary `do` form and **cannot** be called
inline inside an expression (unlike language builtins such as `size(...)` or `substring(...)`).

## Available toolsets

| Toolset             | Tools                                                                                                               |
|---------------------|---------------------------------------------------------------------------------------------------------------------|
| `StringToolset`     | `split`, `join`, `upper`, `lower`, `strip`, `replace`, `starts_with`, `ends_with`, `find`, `pad`                    |
| `CollectionToolset` | `keys`, `values`, `append`, `contains`, `index`, `sort`, `reverse`, `slice`, `range`, `unique`, `merge`, `is_empty` |
| `NumberToolset`     | `abs`, `round`, `floor`, `ceil`, `min`, `max`, `sum`                                                                |
| `JsonToolset`       | `loads`, `dumps`                                                                                                    |
| `RegexToolset`      | `regex_search`, `regex_findall`, `regex_sub`, `regex_split`                                                         |

The flat tool names are unique across all three toolsets, so you can call them either by their bare
name (`do split ...`) or fully qualified (`do tool StringToolset.split ...`).

A tool that returns a list produces an NXS **struct**: index it with `[words:0]` and iterate it with
the `each` loop. NXS structs passed *into* these tools arrive already unboxed (a positional struct
as a list, a named struct as a dict).

## Calling convention

Use the `do` form, passing arguments in a single `using [...]` bracket. Multiple named arguments are
comma-separated:

```
do split using [[text] = [sentence], [sep] = " "] producing [[words]]
```

This is **not** valid — builtin toolsets cannot be called inline in an expression:

```
# INVALID: split is a tool, not a language builtin
[[words] = split([sentence], " ")]
```

### Python idiom → NXS

| Python                | NXS                                                                                            |
|-----------------------|------------------------------------------------------------------------------------------------|
| `text.split(sep)`     | `do split using [[text] = [text], [sep] = [sep]] producing [[parts]]`                          |
| `sep.join(parts)`     | `do join using [[parts] = [parts], [sep] = [sep]] producing [[line]]`                          |
| `text.upper()`        | `do upper using [[text] = [text]] producing [[u]]`                                             |
| `text.strip()`        | `do strip using [[text] = [text]] producing [[clean]]`                                         |
| `text.replace(a, b)`  | `do replace using [[text] = [text], [old] = [a], [new] = [b]] producing [[r]]`                 |
| `sorted(items)`       | `do sort using [[items] = [items]] producing [[ordered]]`                                      |
| `x in items`          | `do contains using [[container] = [items], [item] = [x]] producing [[found]]`                  |
| `sum(values)`         | `do sum using [[values] = [values]] producing [[total]]`                                       |
| `json.loads(text)`    | `do tool JsonToolset.loads using [[text] = [text]] producing [[data]]`                         |
| `json.dumps(value)`   | `do tool JsonToolset.dumps using [[value] = [value]] producing [[body]]`                       |
| `re.findall(p, text)` | `do regex_findall using [[text] = [text], [pattern] = [p]] producing [[matches]]`              |
| `re.sub(p, r, text)`  | `do regex_sub using [[text] = [text], [pattern] = [p], [replacement] = [r]] producing [[out]]` |
| `items[0]`            | `[items:0]` (accessor, not a tool)                                                             |
| `len(x)`              | `size([x])` (language builtin)                                                                 |

## Minimal working example

This script imports nothing, yet freely uses string, collection, and number operations:

```
deliberate builtin_toolsets_example when >> Show the always-available builtin toolsets <<:

  @completion: frozen
  plan:
      in:
        sentence (default ["the quick brown fox"])
      __in

      out:
        report
      __out

      body:
          # split into words (the result is a struct) and upper-case the first one
          do split using [[text] = [sentence], [sep] = " "] producing [[words]]
          [[first] = [words:0]]
          do upper using [[text] = [first]] producing [[first_upper]]

          # collection + number ops
          [[count] = size([words])]
          do sort using [[items] = [words]] producing [[sorted_words]]
          [[count_squared] = [count] ^ 2]    # power is an operator, not a tool

          # build the final report
          do join using [[parts] = [sorted_words], [sep] = ", "] producing [[joined]]
          [[report] = "first word: " | [first_upper]
              | " | words: " | to_str([count])
              | " | squared: " | to_str([count_squared])
              | " | sorted: " | [joined]]

          return [report]
      __body
  __plan
__deliberate
```

A runnable copy lives at `examples/builtin_toolsets.nxs`.

## Precedence

Builtin toolsets are seeded as the baseline before a script's own imports. The dispatch order for an
unqualified `do <name>` is **actions → tools → builtins**, so a user-defined `action` named like a
builtin tool takes precedence for that bare name. Explicitly importing a builtin toolset (for
example `from toolset StringToolset use *`, with a subset of tools, or under an alias) is a tolerated
no-op and never raises.

## Adding a builtin toolset

Builtin toolsets live in the light-weight package `nemantix.core.toolsets` (kept separate from
`nemantix.stl`, which pulls in optional heavy dependencies). To add one:

1. Write a normal `Toolset` subclass with `@tool` methods (see [Toolsets](<05 - Toolsets.md>) and the
   [Toolset API Reference](<05a - Toolset API Reference.md>)). Give each tool a clear docstring that
   includes an NXS `do`-form example — this text is shown to the Coder.
2. Keep the flat tool names unique across all builtin toolsets, and prefer polymorphic tools over
   per-type duplicates.
3. Add the class to `BUILTIN_TOOLSETS` in `nemantix/core/toolsets/__init__.py`.

The interpreter auto-seeds every class in `BUILTIN_TOOLSETS`, and the Coder automatically advertises
their tools.
