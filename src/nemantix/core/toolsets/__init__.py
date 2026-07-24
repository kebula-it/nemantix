"""Builtin toolsets.

These toolsets are always available in every NXS script: they are auto-imported
by the interpreter, so a script never needs a ``from toolset ... use *`` for
them. Their tools are invoked with the ordinary ``do`` form (they are *not*
callable inline in expressions like the language builtins).

They deliberately live outside ``nemantix.stl`` (which pulls in optional heavy
dependencies) and depend only on ``nemantix.core``, so they work on a base
installation.

``BUILTIN_TOOLSETS`` is the canonical list consumed both by the interpreter
(to auto-seed ``context.tools``) and by the coder (to advertise the tools to the
LLM in the coding prompts).
"""

from nemantix.core.toolsets.collection_toolset import (
    CollectionToolset as CollectionToolset,
)
from nemantix.core.toolsets.json_toolset import JsonToolset as JsonToolset
from nemantix.core.toolsets.number_toolset import NumberToolset as NumberToolset
from nemantix.core.toolsets.regex_toolset import RegexToolset as RegexToolset
from nemantix.core.toolsets.string_toolset import StringToolset as StringToolset

BUILTIN_TOOLSETS = [
    StringToolset,
    CollectionToolset,
    NumberToolset,
    JsonToolset,
    RegexToolset,
]
