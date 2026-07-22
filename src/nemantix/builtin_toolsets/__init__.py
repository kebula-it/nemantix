"""Builtin toolsets.

These toolsets are always available in every NXS script: they are auto-imported
by the interpreter, so a script never needs a ``from toolset ... use *`` for
them. Their tools are invoked with the ordinary ``do`` form (they are *not*
callable inline in expressions like the language builtins).

They deliberately live outside ``nemantix.stl`` (which pulls in optional heavy
dependencies) and depend only on ``nemantix.core``, so they work on a base
install.

``BUILTIN_TOOLSETS`` is the canonical list consumed both by the interpreter
(to auto-seed ``context.tools``) and by the coder (to advertise the tools to the
LLM in the coding prompts).
"""

from nemantix.builtin_toolsets.collection_ops import (
    CollectionToolset as CollectionToolset,
)
from nemantix.builtin_toolsets.json_ops import JsonToolset as JsonToolset
from nemantix.builtin_toolsets.number_ops import NumberToolset as NumberToolset
from nemantix.builtin_toolsets.regex_ops import RegexToolset as RegexToolset
from nemantix.builtin_toolsets.string_ops import StringToolset as StringToolset

BUILTIN_TOOLSETS = [
    StringToolset,
    CollectionToolset,
    NumberToolset,
    JsonToolset,
    RegexToolset,
]
