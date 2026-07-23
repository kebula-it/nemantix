import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from nemantix.core import node as nmx_nodes
from nemantix.core.runtime import (
    Builtin,
    DocRef,
    ExternalVariables,
    Frame,
    Metadata,
    Opaque,
    OperationalEnv,
    Secret,
    Struct,
    compute_similarity,
    get_globals,
)

# =============================================================================
# OperationalEnv & Metadata Tests
# =============================================================================


def test_operational_env():
    env = OperationalEnv()
    env.set("x", 10)
    assert env.get("x") == 10
    assert env.get("y") is None
    assert (
        env.get("y", 5) is None
    )  # Note: OperationalEnv ignores the default parameter in its get signature


def test_metadata():
    meta = Metadata()
    meta.set("label", "value")
    assert meta.get("label") == "value"
    assert meta.get("unknown", "default_val") == "default_val"


# =============================================================================
# Struct Tests
# =============================================================================


def test_struct_set_and_get():
    s = Struct()
    s.set("apple", key=None)  # Auto-assigns int key 0
    s.set("banana", key="fruit")  # Assigns int key 1 AND string key "fruit"

    assert s.get(0) == "apple"
    assert s.get(1) == "banana"
    assert s.get("fruit") == "banana"

    # Negative indexing
    assert s.get(-1) == "banana"
    assert s.get(-2) == "apple"
    assert s.get(-3) is None


def test_struct_update_field():
    s = Struct()
    s.set("old_val", key="my_key")
    s.update_field("my_key", "new_val")
    assert s.get("my_key") == "new_val"
    assert s.get(0) == "new_val"  # The underlying int index should also be updated


def test_struct_properties():
    s = Struct()
    s.set(10)
    s.set(20, key="b")

    assert len(s) == 2
    assert not s.can_be_seen_as_list()  # Has a string key

    s2 = Struct()
    s2.set(1)
    assert s2.can_be_seen_as_list()


def test_struct_to_args_and_kwargs():
    s = Struct()
    s.set("pos1")
    s.set("kw1", key="name")

    args, kwargs = s.to_args_and_kwargs()
    assert args == ["pos1"]
    assert kwargs == {"name": "kw1"}


def test_struct_union_and_append():
    s1 = Struct()
    s1.set(1, key="a")

    s2 = Struct()
    s2.set(2, key="b")

    s_union = s1.union(s2)
    assert s_union.get("a") == 1
    assert s_union.get("b") == 2

    s1.append(3)
    assert s1.get(1) == 3  # the next integer index


def test_struct_contains_opaques_and_unbox():
    o = Opaque({"real": "data"})
    s = Struct()
    s.set(o, key="opaque_val")

    assert s.contains_opaques() is True

    unboxed_s = Opaque.unbox_in(s)
    assert isinstance(unboxed_s, Struct)
    assert unboxed_s.get("opaque_val") == {"real": "data"}


# =============================================================================
# ExternalVariables & Secret Tests
# =============================================================================


def test_external_variables():
    ext_vars = ExternalVariables(api_key="123", secrets={"db_pass": "abc"})

    assert ext_vars.get("api_key") == "123"

    # Secrets should be boxed
    secret_val = ext_vars.get("db_pass")
    assert isinstance(secret_val, Secret)
    assert secret_val.unbox() == "abc"
    assert secret_val.name == "db_pass"

    # Read-only checks (should not modify the struct)
    ext_vars.set("new", key="test")
    assert ext_vars.get("test") is None


def test_external_variables_get_names():
    names = ExternalVariables.get_names(
        {"api_key": "123", "secrets": {"db_pass": "abc"}}
    )
    assert "api_key" in names
    assert "db_pass" in names


def test_external_variables_update_is_noop():
    """`.update()` must not mutate the read-only structure."""
    ext_vars = ExternalVariables(api_key="123")
    ext_vars.update({"new_key": "should_not_appear"})
    assert ext_vars.get("new_key") is None


def test_external_variables_update_field_is_noop():
    """`.update_field()` routes through the read-only `.set()` and must not mutate."""
    ext_vars = ExternalVariables(api_key="123")
    ext_vars.update_field("api_key", "changed")
    assert ext_vars.get("api_key") == "123"


def test_external_variables_read_only_shape_helpers():
    """Shape-related helpers are hardcoded to hide the underlying content."""
    ext_vars = ExternalVariables(api_key="123", secrets={"db_pass": "abc"})

    assert ext_vars.as_flat_list() == []
    assert ext_vars.to_args_and_kwargs() == ([], {})
    assert ext_vars.can_be_seen_as_list() is False


def test_external_variables_secrets_non_dict_is_wrapped():
    """A non-dict `secrets` value is coerced into a single secret named 'secrets'."""
    ext_vars = ExternalVariables(secrets="topsecret")

    secret_val = ext_vars.get("secrets")
    assert isinstance(secret_val, Secret)
    assert secret_val.unbox() == "topsecret"


def test_external_variables_secrets_collision_overwrites_plain_value():
    """A key present both as a plain kwarg and inside `secrets` is overwritten
    by the boxed secret (with a warning logged)."""
    ext_vars = ExternalVariables(
        db_pass="plain_value", secrets={"db_pass": "secret_value"}
    )

    secret_val = ext_vars.get("db_pass")
    assert isinstance(secret_val, Secret)
    assert secret_val.unbox() == "secret_value"


def test_external_variables_repr():
    ext_vars = ExternalVariables(api_key="123")
    assert repr(ext_vars) == f"ExternalVariables(num_vars={len(ext_vars)})"


# =============================================================================
# Frame Tests
# =============================================================================


@pytest.fixture
def sample_frame():
    f = Frame("USER")
    f.add_slot("name", cardinality="1", types=[{"name": nmx_nodes.SlotTypesEnum.TEXT}])
    f.add_slot("age", cardinality="0..1", types=[{"name": nmx_nodes.SlotTypesEnum.INT}])
    return f


def test_frame_apply_prefix_exact_match(sample_frame):
    s = Struct()
    s.set("John", key="name")
    s.set(30, key="age")

    result = sample_frame.apply_prefix(s)
    assert result is not None
    assert result.get("name") == "John"
    assert result.get("age") == 30


def test_frame_apply_prefix_invalid_type(sample_frame):
    s = Struct()
    s.set("John", key="name")
    s.set("thirty", key="age")  # Invalid type for INT

    result = sample_frame.apply_prefix(s)
    assert result is None


def test_frame_apply_prefix_extra_fields(sample_frame):
    s = Struct()
    s.set("John", key="name")
    s.set(30, key="age")
    s.set("extra", key="unknown")

    # In apply_prefix, extra fields are ignored but if the parsed valid slots
    # don't equal the frame's total slots, it might fail or pass depending on exact missing slots.
    # In this logic, it counts validated slots. If valid == total_slots, it passes.
    result = sample_frame.apply_prefix(s)
    assert result is not None
    assert result.get("unknown") is None


def test_frame_apply_postfix_with_defaults(sample_frame):
    s = Struct()
    s.set("John", key="name")
    # "age" is missing

    result = sample_frame.apply_postfix(s)
    assert result is not None
    assert result.get("name") == "John"
    assert result.get("age") == 0  # Default for INT


def test_frame_nested_frames():
    root = Frame("ROOT")
    child = Frame("CHILD")
    child.add_slot(
        "val", cardinality="1", types=[{"name": nmx_nodes.SlotTypesEnum.INT}]
    )

    root.add_frame(child)
    root.add_slot(
        "nested",
        cardinality="1",
        types=[{"type": nmx_nodes.SlotTypesEnum.FRAME, "name": "CHILD"}],
    )

    s_child = Struct()
    s_child.set(99, key="val")
    s_root = Struct()
    s_root.set(s_child, key="nested")

    result = root.apply_prefix(s_root)
    assert result is not None
    assert isinstance(result.get("nested"), Struct)
    assert result.get("nested").get("val") == 99


def test_struct_from_python_dict():
    s = Struct.from_python({"name": "John", "age": 30})
    assert isinstance(s, Struct)
    assert s.get("name") == "John"
    assert s.get("age") == 30


def test_struct_from_python_list():
    s = Struct.from_python([10, 20, 30])
    assert isinstance(s, Struct)
    assert s.can_be_seen_as_list()
    assert s.get(0) == 10
    assert s.get(2) == 30


def test_struct_from_python_nested():
    s = Struct.from_python({"user": {"name": "A"}, "tags": [1, 2]})
    assert isinstance(s.get("user"), Struct)
    assert s.get("user").get("name") == "A"
    assert isinstance(s.get("tags"), Struct)
    assert s.get("tags").get(1) == 2


def test_struct_from_python_scalar_passthrough():
    assert Struct.from_python(42) == 42
    assert Struct.from_python("hi") == "hi"


def test_struct_from_python_is_total_no_conversion_failure():
    """from_python never raises: an unsupported object passes through unchanged
    (including when nested). Rejection of non-structures is enforced one layer up,
    in the interpreter's _coerce_to_struct, not here."""
    sentinel = object()
    assert Struct.from_python(sentinel) is sentinel

    nested = Struct.from_python({"x": sentinel, "n": [sentinel]})
    assert isinstance(nested, Struct)
    assert nested.get("x") is sentinel
    assert nested.get("n").get(0) is sentinel


def test_struct_from_python_numeric_string_keys_are_named_fields():
    """Numeric-looking JSON keys stay named string fields (no int() mangling),
    mirroring how a struct literal ("0": ..., "01": ...) is built."""
    s = Struct.from_python({"0": 10, "01": "val", "5": 5})

    # preserved verbatim as named fields (in particular "01" is NOT collapsed to 1)
    assert s.get("0") == 10
    assert s.get("01") == "val"
    assert s.get("5") == 5
    assert not s.can_be_seen_as_list()  # they are named fields, not positional


def test_frame_apply_prefix_on_from_python_nested():
    """A nested frame validates a Struct built from a plain (JSON-like) dict."""
    root = Frame("ROOT")
    child = Frame("CHILD")
    child.add_slot(
        "val", cardinality="1", types=[{"name": nmx_nodes.SlotTypesEnum.INT}]
    )
    root.add_frame(child)
    root.add_slot(
        "nested",
        cardinality="1",
        types=[{"type": nmx_nodes.SlotTypesEnum.FRAME, "name": "CHILD"}],
    )

    s = Struct.from_python({"nested": {"val": 99}})
    result = root.apply_prefix(s)
    assert result is not None
    assert isinstance(result.get("nested"), Struct)
    assert result.get("nested").get("val") == 99


# =============================================================================
# DocRef Tests
# =============================================================================


def test_docref_union():
    d1 = DocRef(node_id="1", score=0.9, breadcrumbs="a", content="Hello")
    d2 = DocRef(node_id="2", score=0.8, breadcrumbs="b", content="World")

    union_d = d1.union(d2)
    assert union_d.get("content") == "Hello World"


def test_docref_equality():
    d1 = DocRef(node_id="1", score=0.9, breadcrumbs="a", content="Hello")
    d2 = DocRef(node_id="1", score=0.1, breadcrumbs="b", content="Diff")
    d3 = DocRef(node_id="2", score=0.9, breadcrumbs="a", content="Hello")
    d4 = DocRef(node_id="3", score=0.9, breadcrumbs="a", content="Hello world")

    assert d1 == d2  # Same node_id
    assert d1 == d3  # different id but same content
    assert d1 != d4  # different id and content


# =============================================================================
# Builtin Tests
# =============================================================================


def test_builtin_coalesce():
    assert Builtin.coalesce(None, None, 5, 10) == 5
    assert Builtin.coalesce(None, a=None, b="text") == "text"


def test_builtin_type():
    assert Builtin.type(None) == "none"
    assert Builtin.type(10) == "num"
    assert Builtin.type(10.5) == "num"
    assert Builtin.type("hello") == "str"
    assert Builtin.type(True) == "bool"
    assert Builtin.type(Struct()) == "struct"
    assert Builtin.type(DocRef("1", 1.0, "", "")) == "doc"
    assert Builtin.type(Opaque({})) == "opaque"


def test_builtin_size():
    assert Builtin.size() == 0
    assert Builtin.size("abc") == 3

    s = Struct()
    s.set(1)
    s.set(2)
    assert Builtin.size(s) == 2

    assert Builtin.size(1, 2, 3) == 3


def test_builtin_size_opaque():
    """An Opaque wrapping None has size 0; any other wrapped value has size 1."""
    assert Builtin.size(Opaque(None)) == 0
    assert Builtin.size(Opaque("hidden")) == 1
    assert (
        Builtin.size(Opaque(0)) == 1
    )  # a falsy-but-not-None payload still counts as 1


def test_builtin_size_docref():
    """A DocRef always has size 1, regardless of its content.

    Regression test: DocRef subclasses Struct, so the DocRef-specific branch
    must be checked before the Struct branch, or it becomes unreachable.
    """
    doc = DocRef(node_id="1", score=0.9, breadcrumbs="a", content="Hello")
    assert Builtin.size(doc) == 1


def test_builtin_substring():
    assert Builtin.substring("nemantix", 0, 3) == "nem"
    assert Builtin.substring("nemantix", 3) == "antix"
    assert Builtin.substring(None) == ""


def test_builtin_to_num():
    assert Builtin.to_num(42) == 42
    assert Builtin.to_num("42") == 42
    assert Builtin.to_num("-42.5") == -42.5
    assert Builtin.to_num("true") == 1
    assert Builtin.to_num("invalid") == 0
    assert Builtin.to_num(True) == 1


def test_builtin_to_bool():
    assert Builtin.to_bool(True) is True
    assert Builtin.to_bool(1) is True
    assert Builtin.to_bool(0) is False
    assert Builtin.to_bool("true") is True
    assert Builtin.to_bool("false") is False
    assert Builtin.to_bool("None") is False


def test_builtin_to_str():
    assert Builtin.to_str(True) == "true"
    assert Builtin.to_str(12.5) == "12.5"
    assert Builtin.to_str(None) == "none"


def test_builtin_bool_implicit():
    """Test the implicit (soft) `bool` builtin, distinct from `to_bool`."""
    assert Builtin.bool(None) is None

    # Struct: truthy iff non-empty
    empty_struct = Struct()
    non_empty_struct = Struct()
    non_empty_struct.set(1)
    assert Builtin.bool(empty_struct) is False
    assert Builtin.bool(non_empty_struct) is True

    # DocRef: truthy iff node_id is non-empty.
    # Regression test: DocRef subclasses Struct, so the DocRef-specific branch
    # must be checked before the Struct branch, or it becomes unreachable.
    doc = DocRef(node_id="1", score=0.9, breadcrumbs="a", content="Hello")
    empty_doc = DocRef(node_id="", score=0.0, breadcrumbs="", content="")
    assert Builtin.bool(doc) is True
    assert Builtin.bool(empty_doc) is False

    # Opaque: truthy iff identifier is non-negative (memory addresses always are;
    # force a negative one to exercise the False branch)
    opaque = Opaque("hidden")
    assert Builtin.bool(opaque) is True
    opaque.identifier = -1
    assert Builtin.bool(opaque) is False

    # Fallthrough delegates to `to_bool`
    assert Builtin.bool(True) is True
    assert Builtin.bool("false") is False


def test_builtin_num_implicit():
    """Test the implicit (soft) `num` builtin, distinct from `to_num`."""
    assert Builtin.num(None) is None

    # Collections are never implicitly numeric
    assert Builtin.num((1, 2)) is None
    assert Builtin.num([1, 2]) is None
    assert Builtin.num({"a": 1}) is None
    assert Builtin.num({1, 2}) is None
    assert Builtin.num(Struct()) is None
    assert (
        Builtin.num(DocRef(node_id="1", score=0.9, breadcrumbs="a", content="Hi"))
        is None
    )
    assert Builtin.num(Opaque(42)) is None

    # Fallthrough delegates to `to_num`
    assert Builtin.num("42") == 42
    assert Builtin.num(True) == 1


def test_builtin_str_implicit():
    """Test the implicit (soft) `str` builtin, distinct from `to_str`."""
    assert Builtin.str(None) is None

    # Fallthrough delegates to `to_str`
    assert Builtin.str(True) == "true"
    assert Builtin.str(12.5) == "12.5"


def test_builtin_math():
    assert Builtin.sin(0) == 0.0
    assert Builtin.cos(0) == 1.0
    assert Builtin.sqrt(4) == 2.0


def test_builtin_to_str_scalar():
    """Test standard scalar conversions."""
    assert Builtin.to_str(True) == "true"
    assert Builtin.to_str(False) == "false"
    assert Builtin.to_str(12) == "12"
    assert Builtin.to_str(12.5) == "12.5"
    assert Builtin.to_str(None) == "none"
    assert Builtin.to_str("hello") == "hello"


def test_builtin_to_str_struct_positional():
    """
    Test that a Struct with positional fields is stringified correctly,
    proving the hack is no longer needed.
    """
    s = Struct()
    s.set(1)
    s.set("apple")
    s.set(False)

    result = Builtin.to_str(s)
    # The output should be the literal representation of the Struct
    assert result == "Struct(1, apple, False)"


def test_builtin_to_str_struct_nominal():
    """Test that a Struct with nominal (keyword) fields is stringified correctly."""
    s = Struct()
    s.set(10, key="x")
    s.set(20, key="y")

    result = Builtin.to_str(s)
    assert result == "Struct(x: 10, y: 20)"


def test_builtin_to_str_struct_mixed():
    """Test that a Struct with mixed positional and nominal fields stringifies correctly."""
    s = Struct()
    s.set("first")
    s.set(99, key="code")
    s.set("last")

    result = Builtin.to_str(s)
    assert result == "Struct(first, code: 99, last)"


# =============================================================================
# Helper/Utility Tests
# =============================================================================


def test_compute_similarity():
    embedder = MagicMock()
    embedder.embed.side_effect = lambda x: np.array([1, 0] if x == "A" else [0, 1])

    # Orthogonal vectors should have 0 similarity
    sim = compute_similarity(embedder, "A", "B")
    assert sim == 0.0

    # Identical vectors should have 1 similarity
    sim_identical = compute_similarity(embedder, "A", "A")
    assert sim_identical == 1.0

    # Using pre-computed embeddings
    emb_a = np.array([0.5, 0.5])
    emb_b = np.array([0.5, 0.5])
    sim_precomputed = compute_similarity(None, None, None, a_emb=emb_a, b_emb=emb_b)
    assert math.isclose(sim_precomputed, 0.5)


def test_get_globals():
    globs = get_globals()
    assert "Toolset" in globs
    assert "Opaque" in globs
    assert "__builtins__" in globs
    assert "print" in globs["__builtins__"]
