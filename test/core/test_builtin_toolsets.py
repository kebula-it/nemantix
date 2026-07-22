"""Unit tests for the always-available builtin toolsets.

These exercise the tool implementations directly (as plain Python callables),
independently of the interpreter. Interpreter-level auto-availability and
idempotent-import behavior is covered in ``test_interpreter.py``.
"""

import pytest

from nemantix.builtin_toolsets import BUILTIN_TOOLSETS
from nemantix.core.tools import Toolset


def _call(tool_name, *args, **kwargs):
    return Toolset.get_tool(tool_name)(*args, **kwargs)


# =============================================================================
# Registration
# =============================================================================


def test_builtin_toolsets_registered():
    for toolset_cls in BUILTIN_TOOLSETS:
        assert toolset_cls.get_tool_names(), f"{toolset_cls.__name__} exposes no tools"


def test_builtin_tool_names_are_unique():
    """The flat (bare) tool names must not collide: they share one namespace in
    ``context.tools``."""
    seen = {}
    collisions = []
    for toolset_cls in BUILTIN_TOOLSETS:
        for name in toolset_cls.get_tool_names():
            if name in seen:
                collisions.append((name, seen[name], toolset_cls.__name__))
            seen[name] = toolset_cls.__name__

    assert not collisions, f"colliding builtin tool names: {collisions}"


def test_builtin_tools_have_docstrings():
    for toolset_cls in BUILTIN_TOOLSETS:
        for name, doc in toolset_cls.get_tool_descriptions().items():
            assert doc and doc.strip(), (
                f"{toolset_cls.__name__}.{name} has no docstring"
            )


# =============================================================================
# StringToolset
# =============================================================================


@pytest.mark.parametrize(
    "tool_name,args,kwargs,expected",
    [
        ("StringToolset.split", ("a,b,c",), {"sep": ","}, ["a", "b", "c"]),
        ("StringToolset.split", ("a b c",), {}, ["a", "b", "c"]),
        ("StringToolset.join", (["a", "b"],), {"sep": "-"}, "a-b"),
        ("StringToolset.join", ([1, 2, 3],), {"sep": ","}, "1,2,3"),
        ("StringToolset.join", ({"x": "a", "y": "b"},), {"sep": "/"}, "a/b"),
        ("StringToolset.upper", ("aB",), {}, "AB"),
        ("StringToolset.lower", ("aB",), {}, "ab"),
        ("StringToolset.strip", ("  hi  ",), {}, "hi"),
        ("StringToolset.strip", ("xxhixx",), {"chars": "x"}, "hi"),
        ("StringToolset.replace", ("a.b.c",), {"old": ".", "new": "/"}, "a/b/c"),
        ("StringToolset.starts_with", ("https://x",), {"prefix": "https"}, True),
        ("StringToolset.ends_with", ("f.nxs",), {"suffix": ".nxs"}, True),
        ("StringToolset.find", ("a=b",), {"sub": "="}, 1),
        ("StringToolset.find", ("abc",), {"sub": "z"}, -1),
        ("StringToolset.pad", ("7",), {"width": 3, "fill": "0"}, "700"),
    ],
)
def test_string_ops(tool_name, args, kwargs, expected):
    assert _call(tool_name, *args, **kwargs) == expected


# =============================================================================
# CollectionToolset
# =============================================================================


@pytest.mark.parametrize(
    "tool_name,args,kwargs,expected",
    [
        ("CollectionToolset.keys", ({"a": 1, "b": 2},), {}, ["a", "b"]),
        ("CollectionToolset.keys", ([10, 20, 30],), {}, [0, 1, 2]),
        ("CollectionToolset.values", ({"a": 1, "b": 2},), {}, [1, 2]),
        ("CollectionToolset.values", ([1, 2],), {}, [1, 2]),
        ("CollectionToolset.append", ([1, 2],), {"item": 3}, [1, 2, 3]),
        ("CollectionToolset.contains", ([1, 2, 3],), {"item": 2}, True),
        ("CollectionToolset.contains", ("abc",), {"item": "b"}, True),
        ("CollectionToolset.contains", ({"k": 1},), {"item": "k"}, True),
        ("CollectionToolset.contains", ([1],), {"item": 9}, False),
        ("CollectionToolset.index", (["a", "b"],), {"item": "b"}, 1),
        ("CollectionToolset.index", ([1],), {"item": 9}, -1),
        ("CollectionToolset.sort", ([3, 1, 2],), {}, [1, 2, 3]),
        ("CollectionToolset.sort", ([1, 3, 2],), {"descending": True}, [3, 2, 1]),
        ("CollectionToolset.reverse", ([1, 2, 3],), {}, [3, 2, 1]),
        ("CollectionToolset.slice", ([1, 2, 3, 4],), {"start": 1, "end": 3}, [2, 3]),
        ("CollectionToolset.slice", ("abcd",), {"start": 0, "end": 2}, "ab"),
        ("CollectionToolset.range", (0, 3), {}, [0, 1, 2]),
        ("CollectionToolset.range", (1, 6), {"step": 2}, [1, 3, 5]),
        ("CollectionToolset.unique", ([1, 1, 2, 3, 2],), {}, [1, 2, 3]),
        ("CollectionToolset.merge", ([1, 2], [3]), {}, [1, 2, 3]),
        ("CollectionToolset.merge", ({"a": 1}, {"b": 2, "a": 9}), {}, {"a": 9, "b": 2}),
        ("CollectionToolset.is_empty", ("",), {}, True),
        ("CollectionToolset.is_empty", ([],), {}, True),
        ("CollectionToolset.is_empty", (None,), {}, True),
        ("CollectionToolset.is_empty", ([1],), {}, False),
        ("CollectionToolset.is_empty", (0,), {}, False),
    ],
)
def test_collection_ops(tool_name, args, kwargs, expected):
    assert _call(tool_name, *args, **kwargs) == expected


def test_append_does_not_mutate_input():
    original = [1, 2]
    result = _call("CollectionToolset.append", original, item=3)
    assert result == [1, 2, 3]
    assert original == [1, 2]


# =============================================================================
# NumberToolset
# =============================================================================


@pytest.mark.parametrize(
    "tool_name,args,kwargs,expected",
    [
        ("NumberToolset.abs", (-5,), {}, 5),
        ("NumberToolset.round", (3.14159,), {"ndigits": 2}, 3.14),
        ("NumberToolset.floor", (2.7,), {}, 2),
        ("NumberToolset.ceil", (2.1,), {}, 3),
        ("NumberToolset.min", ([3, 1, 2],), {}, 1),
        ("NumberToolset.max", ([3, 1, 2],), {}, 3),
        ("NumberToolset.sum", ([1, 2, 3],), {}, 6),
        ("NumberToolset.sum", ([],), {}, 0),
    ],
)
def test_number_ops(tool_name, args, kwargs, expected):
    assert _call(tool_name, *args, **kwargs) == expected


# =============================================================================
# JsonToolset
# =============================================================================


def test_json_loads_object_and_array():
    assert _call("JsonToolset.loads", '{"a": 1, "b": [2, 3]}') == {
        "a": 1,
        "b": [2, 3],
    }
    assert _call("JsonToolset.loads", "[1, 2, 3]") == [1, 2, 3]


def test_json_dumps_compact_and_pretty():
    assert _call("JsonToolset.dumps", {"a": 1, "b": [2, 3]}) == '{"a": 1, "b": [2, 3]}'
    pretty = _call("JsonToolset.dumps", {"a": 1}, pretty=True)
    assert "\n" in pretty and '"a": 1' in pretty


def test_json_dumps_serializes_structs():
    from nemantix.core.runtime import Struct

    positional = Struct()
    positional.set(1)
    positional.set(2)
    assert _call("JsonToolset.dumps", positional) == "[1, 2]"

    named = Struct()
    named.set("bar", key="foo")
    assert _call("JsonToolset.dumps", named) == '{"foo": "bar"}'


def test_json_roundtrip():
    original = {"name": "Alice", "tags": ["x", "y"], "age": 30}
    text = _call("JsonToolset.dumps", original)
    assert _call("JsonToolset.loads", text) == original


# =============================================================================
# RegexToolset
# =============================================================================


@pytest.mark.parametrize(
    "tool_name,args,kwargs,expected",
    [
        ("RegexToolset.regex_search", ("abc123",), {"pattern": r"\d+"}, True),
        ("RegexToolset.regex_search", ("abc",), {"pattern": r"\d+"}, False),
        (
            "RegexToolset.regex_findall",
            ("a1b22c333",),
            {"pattern": r"\d+"},
            ["1", "22", "333"],
        ),
        (
            "RegexToolset.regex_sub",
            ("a  b   c",),
            {"pattern": r"\s+", "replacement": " "},
            "a b c",
        ),
        (
            "RegexToolset.regex_split",
            ("a, b,c",),
            {"pattern": r",\s*"},
            ["a", "b", "c"],
        ),
    ],
)
def test_regex_ops(tool_name, args, kwargs, expected):
    assert _call(tool_name, *args, **kwargs) == expected
