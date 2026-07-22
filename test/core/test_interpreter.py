from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from pydantic import BaseModel

from nemantix.core import exceptions as nmx_ex
from nemantix.core import node as nmx_nodes
from nemantix.core import runtime as nmx_runtime
from nemantix.core.expertise import JsonParsingMode
from nemantix.core.interpreter import Interpreter
from nemantix.core.node import (
    BinaryOperationEnum,
    FileMeta,
    NodeMeta,
    SimilarityEnum,
    SimilarityQualifierEnum,
    SlotTypesEnum,
    UnaryOperationEnum,
    VariableTypeEnum,
)
from nemantix.core.parser import AsFrame
from nemantix.core.script import Script
from nemantix.core.tools import Toolset

HERE = Path(__file__).parent


# =============================================================================
# Minimal Expertise stub (real object, no MagicMock)
# =============================================================================


class DummyExpertise:
    def __init__(self):
        self.script_by_loc = {}
        self.event_hub = None
        self.deliberate_to_script_loc = {}
        self.requires_map = {}

    def get_required_scripts(self, script):
        loc = script.get_location()
        req_locs = self.requires_map.get(loc, [])
        return [self.script_by_loc[r_loc] for r_loc in req_locs]


# =============================================================================
# Robust node builders (work across minor signature differences)
# =============================================================================


def _pick_enum(
    enum_cls, preferred_names: list[str], exclude_names: set[str] | None = None
):
    exclude_names = exclude_names or set()
    for n in preferred_names:
        if hasattr(enum_cls, n):
            return getattr(enum_cls, n)
    for m in enum_cls:
        if m.name not in exclude_names:
            return m
    return list(enum_cls)[0]


# Choose a string-like VariableTypeEnum safely (the enum name can differ between versions)
_STRING_TYPE = _pick_enum(
    VariableTypeEnum,
    preferred_names=["STRING", "TEXT", "STR"],
    exclude_names={"NONE", "INT", "FLOAT", "BOOL", "FSTRING", "LIST"},
)


def make_meta():
    # New interpreter doesn’t require file_meta for these unit tests.
    file_meta = FileMeta((1, 2), (1, 2), HERE / "test_scripts/test_syntax.nxs")
    return {"file_meta": file_meta, "node_meta": NodeMeta([], "", file_meta)}


def make_node(cls, **attrs):
    """
    Try to instantiate a node with matching kwargs; if signature mismatch,
    fallback to __new__ + setattr. This keeps tests resilient to small API changes.
    """
    try:
        sig = inspect.signature(cls)
        filtered = {k: v for k, v in attrs.items() if k in sig.parameters}
        return cls(**filtered)
    except Exception:
        obj = cls.__new__(cls)
        for k, v in attrs.items():
            setattr(obj, k, v)
        if not hasattr(obj, "meta"):
            obj.meta = make_meta()
        return obj


def make_value(val, type_enum: VariableTypeEnum | None = None):
    if val is None:
        inferred = getattr(VariableTypeEnum, "NONE", list(VariableTypeEnum)[0])
        return make_node(
            nmx_nodes.SingleValue, value=None, inferred_type=inferred, meta=make_meta()
        )

    if type_enum is None:
        if isinstance(val, bool):
            type_enum = VariableTypeEnum.BOOL
        elif isinstance(val, int):
            type_enum = VariableTypeEnum.INT
        elif isinstance(val, float):
            type_enum = VariableTypeEnum.FLOAT
        elif isinstance(val, str):
            type_enum = _STRING_TYPE
        else:
            raise RuntimeError(f"Unsupported literal type: {type(val)}")

    return make_node(
        nmx_nodes.SingleValue, value=val, inferred_type=type_enum, meta=make_meta()
    )


def make_var(name: str, path=None):
    # Variable(path=[]) is used for nested struct navigation
    return make_node(
        nmx_nodes.Variable, name=name, path=path or [], prompt=None, meta=make_meta()
    )


def make_binary_op(op, left, right):
    return make_node(
        nmx_nodes.BinaryOperation,
        operation=op,
        first=make_value(left)
        if not isinstance(left, (nmx_nodes.Variable, nmx_nodes.SingleValue))
        else left,
        second=make_value(right)
        if not isinstance(right, (nmx_nodes.Variable, nmx_nodes.SingleValue))
        else right,
        meta=make_meta(),
    )


# =============================================================================
# Fixtures
# =============================================================================


class DummyEmbedder:
    def __init__(self):
        self._map = {}

    def set_embedding(self, text: str, vec):
        self._map[text] = np.array(vec, dtype=float)

    def embed(self, text: str):
        return np.array(self._map.get(text, [0.0]), dtype=float)


class DummyLLM:
    def __init__(self):
        self.calls = []

    def get_name(self) -> str:
        return "dummy-llm"

    def messages_from(self, prompts_with_roles: list):
        return prompts_with_roles

    def invoke(self, prompt: str, **kwargs):
        self.calls.append((prompt, None))
        return SimpleNamespace(
            text="dummy text",
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            proxy=self,
        )

    def invoke_structured(self, prompt: str, schema: type[BaseModel]):
        self.calls.append((prompt, schema))

        class _R:
            def __init__(self, data):
                self._data = data

            def model_dump(self):
                return dict(self._data)

        # Look at the schema to decide what dummy data to return.
        # If it looks like a boolean check (e.g., has an 'is_included' or 'result' field of type bool), return True.
        dummy_data = {}
        for field_name, field_info in schema.model_fields.items():
            if (
                field_name == "holds"
                or field_info.annotation is bool
                or field_name == "score"
                or field_info.annotation is float
            ):
                dummy_data = Interpreter.SimilaritySchema(holds=True, score=1.0)
                break

        if not dummy_data:
            dummy_data = {"name": "Luigi", "age": 42}
            result = _R(dummy_data)
        else:
            result = dummy_data

        return SimpleNamespace(
            result=result,
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            proxy=self,
        )


@pytest.fixture
def interpreter_instance(dummy_llm_proxy_config_class):
    exp = DummyExpertise()
    emb = DummyEmbedder()
    llm = DummyLLM()
    llm_proxies = dummy_llm_proxy_config_class(dummy_llm=llm)
    return Interpreter(expertise=exp, llm=llm, embedder=emb, proxy_config=llm_proxies)


# =============================================================================
# Tests
# =============================================================================


def test_initialization(interpreter_instance):
    assert interpreter_instance.context.env is not None
    assert interpreter_instance.context.frames is not None
    assert interpreter_instance.context.tools is not None
    # Special vars are registered at init
    assert interpreter_instance.context.env.get("PI") is not None
    assert interpreter_instance.context.env.get("STATE") is not None


# --- Assignments ---


def test_simple_assignment(interpreter_instance):
    """x = 10"""
    assignment = make_node(
        nmx_nodes.Assignment,
        var=make_var("x"),
        value=make_value(10),
        meta=make_meta(),
    )
    interpreter_instance.interpret_statement(assignment)
    assert interpreter_instance.context.env.get("x") == 10


def test_nested_path_assignment(interpreter_instance):
    """
    user.profile.age = 30
    """
    interpreter_instance.context.env.set(var_name="user", value=nmx_runtime.Struct())

    var_node = make_var(
        "user",
        path=[
            make_value("profile", _STRING_TYPE),
            make_value("age", _STRING_TYPE),
        ],
    )
    assignment = make_node(
        nmx_nodes.Assignment,
        var=var_node,
        value=make_value(30),
        meta=make_meta(),
    )
    interpreter_instance.interpret_statement(assignment)

    user = interpreter_instance.context.env.get("user")
    assert isinstance(user, nmx_runtime.Struct)
    profile = user.get("profile")
    assert isinstance(profile, nmx_runtime.Struct)
    assert profile.get("age") == 30


def _struct_with_mixed_keys():
    """( "0": 10, "01": "val", field: 123, "pos" )"""
    x = nmx_runtime.Struct()
    x.set(10, key="0")
    x.set("val", key="01")
    x.set(123, key="field")
    x.set("pos")
    return x


def test_field_accessor_string_reaches_named_field(interpreter_instance):
    interpreter_instance.context.env.set(var_name="x", value=_struct_with_mixed_keys())
    var = make_var("x", path=[make_value("01", _STRING_TYPE)])
    assert interpreter_instance.unbox_value(var) == "val"


def test_field_accessor_variable_resolves(interpreter_instance):
    interpreter_instance.context.env.set(var_name="x", value=_struct_with_mixed_keys())
    interpreter_instance.context.env.set(var_name="index", value="01")
    var = make_var("x", path=[make_var("index")])
    assert interpreter_instance.unbox_value(var) == "val"


def test_field_accessor_int_index(interpreter_instance):
    interpreter_instance.context.env.set(var_name="x", value=_struct_with_mixed_keys())
    var = make_var("x", path=[make_value(0)])
    assert interpreter_instance.unbox_value(var) == 10


def test_struct_accessor_rejected_on_read(interpreter_instance):
    """[x:(1, 2)] — a struct/collection cannot be used as a field index."""
    interpreter_instance.context.env.set(var_name="x", value=_struct_with_mixed_keys())
    coll = make_node(
        nmx_nodes.Collection,
        value=[make_value(1), make_value(2)],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )
    var = make_var("x", path=[coll])
    with pytest.raises(
        nmx_ex.NemantixRuntimeException,
        match=r"field accessor must be an integer index or a string field name",
    ):
        interpreter_instance.unbox_value(var)


def test_struct_valued_variable_accessor_rejected(interpreter_instance):
    """[x:[y]] where [y] holds a struct — caught only at runtime."""
    interpreter_instance.context.env.set(var_name="x", value=_struct_with_mixed_keys())
    interpreter_instance.context.env.set(var_name="y", value=nmx_runtime.Struct())
    var = make_var("x", path=[make_var("y")])
    with pytest.raises(nmx_ex.NemantixRuntimeException, match=r"Cannot index"):
        interpreter_instance.unbox_value(var)


def test_struct_accessor_rejected_on_assignment(interpreter_instance):
    """[x:(1, 2)] = v — the guard also applies on the assignment path."""
    interpreter_instance.context.env.set(var_name="x", value=_struct_with_mixed_keys())
    coll = make_node(
        nmx_nodes.Collection,
        value=[make_value(1), make_value(2)],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )
    assignment = make_node(
        nmx_nodes.Assignment,
        var=make_var("x", path=[coll]),
        value=make_value(1),
        meta=make_meta(),
    )
    with pytest.raises(
        nmx_ex.NemantixRuntimeException,
        match=r"field accessor must be an integer index or a string field name",
    ):
        interpreter_instance.interpret_statement(assignment)


def test_assignment_to_special_var_raises(interpreter_instance):
    assignment = make_node(
        nmx_nodes.Assignment,
        var=make_var("PI"),
        value=make_value(123),
        meta=make_meta(),
    )
    with pytest.raises(
        nmx_ex.NemantixRuntimeException, match=r"Cannot assign special variable"
    ):
        interpreter_instance.interpret_statement(assignment)


# --- Binary Operations: Arithmetic ---


@pytest.mark.parametrize(
    "op, left, right, expected",
    [
        (BinaryOperationEnum.ADD, 5, 3, 8),
        (BinaryOperationEnum.SUB, 10, 4, 6),
        (BinaryOperationEnum.MUL, 6, 7, 42),
        (BinaryOperationEnum.DIV, 20, 5, 4.0),
        (BinaryOperationEnum.MOD, 10, 3, 1),
        (BinaryOperationEnum.POW, 2, 3, 8),
        (BinaryOperationEnum.ADD, 2.5, 3.5, 6.0),
        (BinaryOperationEnum.MUL, 3, 1.5, 4.5),
        (BinaryOperationEnum.ADD, -5, -3, -8),
    ],
)
def test_arithmetic_operations(interpreter_instance, op, left, right, expected):
    expr = make_binary_op(op, left, right)
    assert interpreter_instance.interpret_expression(expr) == expected


def test_division_by_zero(interpreter_instance):
    expr = make_binary_op(BinaryOperationEnum.DIV, 10, 0)
    assert interpreter_instance.interpret_expression(expr) is None


# --- Binary Operations: Logical ---


@pytest.mark.parametrize(
    "op, left, right, expected",
    [
        (BinaryOperationEnum.LOGICAL_AND, True, True, True),
        (BinaryOperationEnum.LOGICAL_AND, True, False, False),
        (BinaryOperationEnum.LOGICAL_OR, False, True, True),
        (BinaryOperationEnum.LOGICAL_XOR, True, True, False),
    ],
)
def test_logical_operations(interpreter_instance, op, left, right, expected):
    expr = make_binary_op(op, left, right)
    assert interpreter_instance.interpret_expression(expr) == expected


# --- Binary Operations: Comparison ---


@pytest.mark.parametrize(
    "op, left, right, expected",
    [
        (BinaryOperationEnum.EQ, 5, 5, True),
        (BinaryOperationEnum.EQ, 5, 6, False),
        (BinaryOperationEnum.NE, 5, 6, True),
        (BinaryOperationEnum.GT, 10, 5, True),
        (BinaryOperationEnum.LT, 5, 10, True),
        (BinaryOperationEnum.GTE, 5, 5, True),
        (BinaryOperationEnum.LTE, 6, 5, False),
        (BinaryOperationEnum.EQ, "apple", "apple", True),
        (BinaryOperationEnum.LT, "apple", "banana", True),
    ],
)
def test_comparison_operations(interpreter_instance, op, left, right, expected):
    expr = make_binary_op(op, left, right)
    assert interpreter_instance.interpret_expression(expr) == expected


# --- Binary Operations: String/Misc ---


def test_string_concat(interpreter_instance):
    expr = make_binary_op(BinaryOperationEnum.CONCAT, "Hello", "World")
    assert interpreter_instance.interpret_expression(expr) == "HelloWorld"


def test_string_concat_fail(interpreter_instance):
    with pytest.raises(nmx_ex.NemantixOperationException):
        expr = make_binary_op(BinaryOperationEnum.CONCAT, "Hello", 12)
        interpreter_instance.interpret_expression(expr)


def test_logical_op(interpreter_instance):
    expr = make_binary_op(BinaryOperationEnum.LOGICAL_OR, True, False)
    assert interpreter_instance.interpret_expression(expr) is True


def test_logical_op_fail(interpreter_instance):
    with pytest.raises(nmx_ex.NemantixOperationException):
        expr = make_binary_op(BinaryOperationEnum.LOGICAL_OR, True, "None")
        interpreter_instance.interpret_expression(expr)


def test_equality_op_none(interpreter_instance):
    expr = make_binary_op(BinaryOperationEnum.EQ, None, None)
    assert interpreter_instance.interpret_expression(expr) is True


def test_equality_op(interpreter_instance):
    expr = make_binary_op(BinaryOperationEnum.EQ, 1, 2)
    assert interpreter_instance.interpret_expression(expr) is False


def test_equality_op_fail(interpreter_instance):
    with pytest.raises(nmx_ex.NemantixOperationException):
        expr = make_binary_op(BinaryOperationEnum.EQ, True, "None")
        interpreter_instance.interpret_expression(expr)


def test_comparison_op(interpreter_instance):
    expr = make_binary_op(BinaryOperationEnum.LT, 1, 2)
    assert interpreter_instance.interpret_expression(expr) is True


def test_comparison_op_none(interpreter_instance):
    with pytest.raises(nmx_ex.NemantixOperationException):
        expr = make_binary_op(BinaryOperationEnum.LTE, None, None)
        interpreter_instance.interpret_expression(expr)


def test_comparison_op_fail(interpreter_instance):
    with pytest.raises(nmx_ex.NemantixOperationException):
        expr = make_binary_op(BinaryOperationEnum.GT, None, "None")
        interpreter_instance.interpret_expression(expr)


def test_arithmetic_op(interpreter_instance):
    expr = make_binary_op(BinaryOperationEnum.ADD, 1, 2)
    assert interpreter_instance.interpret_expression(expr) == 3


def test_arithmetic_op_fail_bool(interpreter_instance):
    with pytest.raises(nmx_ex.NemantixOperationException):
        expr = make_binary_op(BinaryOperationEnum.SUB, 1, True)
        interpreter_instance.interpret_expression(expr)


def test_arithmetic_op_fail(interpreter_instance):
    with pytest.raises(nmx_ex.NemantixOperationException):
        expr = make_binary_op(BinaryOperationEnum.DIV, 1, "True")
        interpreter_instance.interpret_expression(expr)


def test_fallback_operator(interpreter_instance):
    expr1 = make_binary_op(BinaryOperationEnum.FALLBACK, 10, 20)
    assert interpreter_instance.interpret_expression(expr1) == 10

    expr2 = make_binary_op(BinaryOperationEnum.FALLBACK, make_value(None), 20)
    expr2.first = make_value(None)
    assert interpreter_instance.interpret_expression(expr2) == 20


# --- Unary Operations ---


def test_unary_math(interpreter_instance):
    neg_expr = make_node(
        nmx_nodes.UnaryOperation,
        operation=UnaryOperationEnum.NEG,
        operand=make_value(5),
        meta=make_meta(),
    )
    assert interpreter_instance.interpret_expression(neg_expr) == -5

    pos_expr = make_node(
        nmx_nodes.UnaryOperation,
        operation=UnaryOperationEnum.POS,
        operand=make_value(5),
        meta=make_meta(),
    )
    assert interpreter_instance.interpret_expression(pos_expr) == 5


def test_unary_none(interpreter_instance):
    with pytest.raises(nmx_ex.NemantixOperationException):
        neg_expr = make_node(
            nmx_nodes.UnaryOperation,
            operation=UnaryOperationEnum.NEG,
            operand=make_value(None),
            meta=make_meta(),
        )
        interpreter_instance.interpret_expression(neg_expr)

    pos_expr = make_node(
        nmx_nodes.UnaryOperation,
        operation=UnaryOperationEnum.POS,
        operand=make_value(5),
        meta=make_meta(),
    )
    assert interpreter_instance.interpret_expression(pos_expr) == 5


def test_unary_not(interpreter_instance):
    not_expr = make_node(
        nmx_nodes.UnaryOperation,
        operation=UnaryOperationEnum.NOT,
        operand=make_value(True, VariableTypeEnum.BOOL),
        meta=make_meta(),
    )
    assert interpreter_instance.interpret_expression(not_expr) is False


def test_unary_not_fail(interpreter_instance):
    with pytest.raises(nmx_ex.NemantixOperationException):
        neg_expr = make_node(
            nmx_nodes.UnaryOperation,
            operation=UnaryOperationEnum.NOT,
            operand=make_value("String", VariableTypeEnum.STRING),
            meta=make_meta(),
        )
        interpreter_instance.interpret_expression(neg_expr)


def test_unary_neg_bool_fail(interpreter_instance):
    with pytest.raises(nmx_ex.NemantixOperationException):
        neg_expr = make_node(
            nmx_nodes.UnaryOperation,
            operation=UnaryOperationEnum.NEG,
            operand=make_value(True, VariableTypeEnum.BOOL),
            meta=make_meta(),
        )
        interpreter_instance.interpret_expression(neg_expr)


def test_unary_neg_fail(interpreter_instance):
    with pytest.raises(nmx_ex.NemantixOperationException):
        neg_expr = make_node(
            nmx_nodes.UnaryOperation,
            operation=UnaryOperationEnum.NEG,
            operand=make_value("string", VariableTypeEnum.BOOL),
            meta=make_meta(),
        )
        interpreter_instance.interpret_expression(neg_expr)


def test_unary_pos_fail(interpreter_instance):
    with pytest.raises(nmx_ex.NemantixOperationException):
        neg_expr = make_node(
            nmx_nodes.UnaryOperation,
            operation=UnaryOperationEnum.POS,
            operand=make_value("string", VariableTypeEnum.STRING),
            meta=make_meta(),
        )
        interpreter_instance.interpret_expression(neg_expr)


def test_unary_pos_bool_fail(interpreter_instance):
    with pytest.raises(nmx_ex.NemantixOperationException):
        neg_expr = make_node(
            nmx_nodes.UnaryOperation,
            operation=UnaryOperationEnum.POS,
            operand=make_value(False, VariableTypeEnum.BOOL),
            meta=make_meta(),
        )
        interpreter_instance.interpret_expression(neg_expr)


# --- Control Flow: If/Elif/Else ---


def test_if_elif_else(interpreter_instance):
    """
    if false: x=1
    elif true: x=2
    else: x=3
    """
    x_var = make_var("x")

    if_block = make_node(
        nmx_nodes.IfBlock,
        condition=make_value(False, VariableTypeEnum.BOOL),
        children=[
            make_node(
                nmx_nodes.Assignment, var=x_var, value=make_value(1), meta=make_meta()
            )
        ],
        body=[
            make_node(
                nmx_nodes.Assignment, var=x_var, value=make_value(1), meta=make_meta()
            )
        ],  # compat
        meta=make_meta(),
    )
    elif_block = make_node(
        nmx_nodes.ElifBlock,
        condition=make_value(True, VariableTypeEnum.BOOL),
        children=[
            make_node(
                nmx_nodes.Assignment, var=x_var, value=make_value(2), meta=make_meta()
            )
        ],
        body=[
            make_node(
                nmx_nodes.Assignment, var=x_var, value=make_value(2), meta=make_meta()
            )
        ],
        meta=make_meta(),
    )
    else_block = make_node(
        nmx_nodes.ElseBlock,
        children=[
            make_node(
                nmx_nodes.Assignment, var=x_var, value=make_value(3), meta=make_meta()
            )
        ],
        body=[
            make_node(
                nmx_nodes.Assignment, var=x_var, value=make_value(3), meta=make_meta()
            )
        ],
        meta=make_meta(),
    )

    cond_block = make_node(nmx_nodes.ConditionBlock, meta=make_meta())
    # New interpreter iterates conditional.children
    cond_block.children = [if_block, elif_block, else_block]

    interpreter_instance.interpret_statement(cond_block)
    assert interpreter_instance.context.env.get("x") == 2


# --- Loops ---


def test_repeat_while(interpreter_instance):
    """
    x = 0
    while x < 3:
        x = x + 1
    """
    interpreter_instance.context.env.set(var_name="x", value=0)

    condition = make_node(
        nmx_nodes.BinaryOperation,
        operation=BinaryOperationEnum.LT,
        first=make_var("x"),
        second=make_value(3),
        meta=make_meta(),
    )
    body_stmt = make_node(
        nmx_nodes.Assignment,
        var=make_var("x"),
        value=make_node(
            nmx_nodes.BinaryOperation,
            operation=BinaryOperationEnum.ADD,
            first=make_var("x"),
            second=make_value(1),
            meta=make_meta(),
        ),
        meta=make_meta(),
    )

    loop = make_node(
        nmx_nodes.RepeatWhileBlock,
        condition=condition,
        max=make_value(100),
        max_it=make_value(100),  # compat
        meta=make_meta(),
    )
    loop.children = [body_stmt]

    interpreter_instance.interpret_statement(loop)
    assert interpreter_instance.context.env.get("x") == 3


def test_repeat_until(interpreter_instance):
    """
    x = 0
    until x == 3:
        x = x + 1
    """
    interpreter_instance.context.env.set(var_name="x", value=0)

    condition = make_node(
        nmx_nodes.BinaryOperation,
        operation=BinaryOperationEnum.EQ,
        first=make_var("x"),
        second=make_value(3),
        meta=make_meta(),
    )
    body_stmt = make_node(
        nmx_nodes.Assignment,
        var=make_var("x"),
        value=make_node(
            nmx_nodes.BinaryOperation,
            operation=BinaryOperationEnum.ADD,
            first=make_var("x"),
            second=make_value(1),
            meta=make_meta(),
        ),
        meta=make_meta(),
    )

    loop = make_node(
        nmx_nodes.RepeatUntilBlock,
        condition=condition,
        max=100,
        max_it=100,
        meta=make_meta(),
    )
    loop.children = [body_stmt]

    interpreter_instance.interpret_statement(loop)
    assert interpreter_instance.context.env.get("x") == 3


def test_repeat_each(interpreter_instance):
    """
    my_list = [10, 20, 30]
    sum = 0
    repeat each idx, val in my_list:
       sum = sum + val
    """
    interpreter_instance.context.env.set(var_name="my_list", value=[10, 20, 30])
    interpreter_instance.context.env.set(var_name="sum", value=0)

    each_expr = make_var("my_list")

    body_stmt = make_node(
        nmx_nodes.Assignment,
        var=make_var("sum"),
        value=make_node(
            nmx_nodes.BinaryOperation,
            operation=BinaryOperationEnum.ADD,
            first=make_var("sum"),
            second=make_var("val"),
            meta=make_meta(),
        ),
        meta=make_meta(),
    )

    loop = make_node(
        nmx_nodes.RepeatEachBlock,
        each=each_expr,
        as_vars=["idx", "val"],
        meta=make_meta(),
    )
    loop.children = [body_stmt]

    interpreter_instance.interpret_statement(loop)
    assert interpreter_instance.context.env.get("sum") == 60


def test_repeat_times(interpreter_instance):
    """
    x = 0
    repeat 5 times:
        x = x + 1
    """
    interpreter_instance.context.env.set(var_name="x", value=0)

    body_stmt = make_node(
        nmx_nodes.Assignment,
        var=make_var("x"),
        value=make_node(
            nmx_nodes.BinaryOperation,
            operation=BinaryOperationEnum.ADD,
            first=make_var("x"),
            second=make_value(1),
            meta=make_meta(),
        ),
        meta=make_meta(),
    )

    loop = make_node(
        nmx_nodes.RepeatTimesBlock, as_vars=None, times=5, meta=make_meta()
    )
    loop.children = [body_stmt]

    interpreter_instance.interpret_statement(loop)
    assert interpreter_instance.context.env.get("x") == 5


def test_loop_break(interpreter_instance):
    """
    x = 0
    repeat 10 times:
        x = x + 1
        if x == 3:
            break
    """
    interpreter_instance.context.env.set(var_name="x", value=0)

    inc_stmt = make_node(
        nmx_nodes.Assignment,
        var=make_var("x"),
        value=make_node(
            nmx_nodes.BinaryOperation,
            operation=BinaryOperationEnum.ADD,
            first=make_var("x"),
            second=make_value(1),
            meta=make_meta(),
        ),
        meta=make_meta(),
    )

    break_stmt = make_node(nmx_nodes.Break, meta=make_meta())
    if_block = make_node(
        nmx_nodes.IfBlock,
        condition=make_node(
            nmx_nodes.BinaryOperation,
            operation=BinaryOperationEnum.EQ,
            first=make_var("x"),
            second=make_value(3),
            meta=make_meta(),
        ),
        children=[break_stmt],
        body=[break_stmt],
        meta=make_meta(),
    )
    cond_block = make_node(nmx_nodes.ConditionBlock, meta=make_meta())
    cond_block.children = [if_block]

    loop = make_node(nmx_nodes.RepeatTimesBlock, as_vars=[], times=10, meta=make_meta())
    loop.children = [inc_stmt, cond_block]

    interpreter_instance.interpret_statement(loop)
    assert interpreter_instance.context.env.get("x") == 3


def test_loop_continue(interpreter_instance):
    """
    sum = 0
    repeat each idx, val in [1,2,3,4]:
        if val == 2: continue
        sum = sum + val
    Expected: 8
    """
    interpreter_instance.context.env.set(var_name="list", value=[1, 2, 3, 4])
    interpreter_instance.context.env.set(var_name="sum", value=0)

    continue_stmt = make_node(nmx_nodes.Continue, meta=make_meta())
    if_block = make_node(
        nmx_nodes.IfBlock,
        condition=make_node(
            nmx_nodes.BinaryOperation,
            operation=BinaryOperationEnum.EQ,
            first=make_var("val"),
            second=make_value(2),
            meta=make_meta(),
        ),
        children=[continue_stmt],
        body=[continue_stmt],
        meta=make_meta(),
    )
    cond_block = make_node(nmx_nodes.ConditionBlock, meta=make_meta())
    cond_block.children = [if_block]

    sum_stmt = make_node(
        nmx_nodes.Assignment,
        var=make_var("sum"),
        value=make_node(
            nmx_nodes.BinaryOperation,
            operation=BinaryOperationEnum.ADD,
            first=make_var("sum"),
            second=make_var("val"),
            meta=make_meta(),
        ),
        meta=make_meta(),
    )

    loop = make_node(
        nmx_nodes.RepeatEachBlock,
        each=make_var("list"),
        as_vars=["idx", "val"],
        meta=make_meta(),
    )
    loop.children = [cond_block, sum_stmt]

    interpreter_instance.interpret_statement(loop)
    assert interpreter_instance.context.env.get("sum") == 8


# --- Similarity ---


def test_similarity_expression(interpreter_instance):
    """
    "A" ~ "B" with CLOSE qualifier; embeddings are set to yield similarity 1.0
    """
    interpreter_instance.embedder.set_embedding("A", [1.0])
    interpreter_instance.embedder.set_embedding("B", [1.0])

    sim_op = make_node(
        nmx_nodes.SimilarityOperation,
        operation=SimilarityEnum.SIM,
        qualifier=SimilarityQualifierEnum.CLOSE,
        first=make_value("A", _STRING_TYPE),
        second=make_value("B", _STRING_TYPE),
        meta=make_meta(),
    )
    assert interpreter_instance.interpret_expression(sim_op) is True


@pytest.mark.parametrize(
    "qualifier, sim_score, expected",
    [
        (SimilarityQualifierEnum.CLOSE, 0.9, True),
        (SimilarityQualifierEnum.CLOSE, 0.8, False),
        (SimilarityQualifierEnum.STRICT, 0.95, True),
        (SimilarityQualifierEnum.STRICT, 0.90, False),
        (SimilarityQualifierEnum.LOOSE, 0.65, True),
        (SimilarityQualifierEnum.LOOSE, 0.5, False),
        (SimilarityQualifierEnum.FAR, 0.3, True),
        (SimilarityQualifierEnum.FAR, 0.5, False),
        (SimilarityQualifierEnum.ABOUT, 0.76, True),
        (SimilarityQualifierEnum.ABOUT, 0.70, False),
    ],
)
def test_similarity_qualifiers_static(
    interpreter_instance, qualifier, sim_score, expected
):
    # Directly test the qualifier function (more stable than embedding dot-products)
    assert Interpreter._apply_similarity_qualifier(sim_score, qualifier) is expected


def test_similarity_numeric_qualifier(interpreter_instance):
    assert Interpreter._apply_similarity_qualifier(0.75, 0.8) is False
    assert Interpreter._apply_similarity_qualifier(0.85, 0.8) is True


def test_similarity_struct_filtering(interpreter_instance):
    """
    data (Struct) ~ "fruit" filters values by similarity.
    """
    # Build struct: {"relevant_key":"apple", "irrelevant_key":"bicycle"}
    data = nmx_runtime.Struct()
    data.set(key="relevant_key", value="apple")
    data.set(key="irrelevant_key", value="bicycle")
    interpreter_instance.context.env.set(var_name="data", value=data)

    # embeddings: query "fruit"=[1.0], apple=[0.9] (pass CLOSE), bicycle=[0.1] (fail CLOSE)
    interpreter_instance.embedder.set_embedding("fruit", [1.0])
    interpreter_instance.embedder.set_embedding("apple", [0.9])
    interpreter_instance.embedder.set_embedding("bicycle", [0.1])

    sim_op = make_node(
        nmx_nodes.SimilarityOperation,
        operation=SimilarityEnum.SIM,
        qualifier=SimilarityQualifierEnum.CLOSE,
        first=make_var("data"),
        second=make_value("fruit", _STRING_TYPE),
        meta=make_meta(),
    )

    result = interpreter_instance.interpret_expression(sim_op)
    assert isinstance(result, nmx_runtime.Struct)
    assert "relevant_key" in result
    assert result.get("relevant_key") == "apple"
    assert "irrelevant_key" not in result


# --- Semantic inclusion ---


def test_right_inclusion_expression(interpreter_instance):
    """
    "A" ~> "B" with CLOSE qualifier; embeddings are set to yield similarity 1.0
    """
    sim_op = make_node(
        nmx_nodes.SimilarityOperation,
        operation=SimilarityEnum.SIM_RIGHT,
        qualifier=SimilarityQualifierEnum.CLOSE,
        first=make_value("A", _STRING_TYPE),
        second=make_value("B", _STRING_TYPE),
        meta=make_meta(),
    )
    result = interpreter_instance.interpret_expression(sim_op) is True
    assert len(interpreter_instance.llm.calls) > 0
    assert result is True


def test_left_inclusion_expression(interpreter_instance):
    """
    "A" <~ "B" with CLOSE qualifier; embeddings are set to yield similarity 1.0
    """
    sim_op = make_node(
        nmx_nodes.SimilarityOperation,
        operation=SimilarityEnum.SIM_LEFT,
        qualifier=SimilarityQualifierEnum.CLOSE,
        first=make_value("A", _STRING_TYPE),
        second=make_value("B", _STRING_TYPE),
        meta=make_meta(),
    )
    result = interpreter_instance.interpret_expression(sim_op) is True
    assert len(interpreter_instance.llm.calls) > 0
    assert result is True


@pytest.mark.parametrize(
    "qualifier, sim_score, expected",
    [
        (SimilarityQualifierEnum.CLOSE, 0.9, True),
        (SimilarityQualifierEnum.CLOSE, 0.8, False),
        (SimilarityQualifierEnum.STRICT, 0.95, True),
        (SimilarityQualifierEnum.STRICT, 0.90, False),
        (SimilarityQualifierEnum.LOOSE, 0.75, True),
        (SimilarityQualifierEnum.LOOSE, 0.5, False),
        (SimilarityQualifierEnum.FAR, 0.3, True),
        (SimilarityQualifierEnum.FAR, 0.5, False),
        (SimilarityQualifierEnum.ABOUT, 0.86, True),
        (SimilarityQualifierEnum.ABOUT, 0.70, False),
    ],
)
def test_semantic_qualifiers_static(
    interpreter_instance, qualifier, sim_score, expected
):
    # Directly test the qualifier function (more stable than embedding dot-products)
    assert Interpreter._apply_semantic_qualifier(sim_score, qualifier) is expected


def test_semantic_numeric_qualifier(interpreter_instance):
    assert Interpreter._apply_semantic_qualifier(0.75, 0.8) is False
    assert Interpreter._apply_semantic_qualifier(0.85, 0.8) is True


# --- Tool Call Errors ---


def test_do_statement_unknown_tool(interpreter_instance):
    do_stmt = make_node(
        nmx_nodes.DoStatement,
        name="unknown_tool",
        callable_type=getattr(nmx_nodes.CallableTypeEnum, "TOOL"),
        using=None,
        prompt=None,
        producing=None,
        producing_schema=None,
        meta=make_meta(),
    )

    with pytest.raises(nmx_ex.NemantixRuntimeException, match=r"No tool named"):
        interpreter_instance.interpret_do_statement(do=do_stmt)


# =============================================================================
# Action Input Validation Tests (new _set_action_inputs behavior)
# =============================================================================


@dataclass
class DummyActionInput:
    name: str
    required: bool = True
    default: object = None  # Expression or None
    prompt: object = None  # MicroPrompt or None


@dataclass
class DummyAction:
    name: str
    input: list[DummyActionInput]
    output: list[object]
    children: list[object]
    meta: dict


def test_action_inputs_valid_kwargs_out_of_order(interpreter_instance):
    args = nmx_runtime.Struct()
    args.set(key="bar", value=20)
    args.set(key="foo", value=10)
    args = nmx_runtime.Struct.unbox_in(args)

    action = DummyAction(
        name="my_action",
        input=[DummyActionInput("foo", True), DummyActionInput("bar", True)],
        output=[],
        children=[],
        meta=make_meta(),
    )

    interpreter_instance._set_block_inputs(action, args)
    assert interpreter_instance.context.env.get("foo") == 10
    assert interpreter_instance.context.env.get("bar") == 20


def test_action_inputs_valid_positional(interpreter_instance):
    action = DummyAction(
        name="my_action",
        input=[DummyActionInput("foo", True), DummyActionInput("bar", True)],
        output=[],
        children=[],
        meta=make_meta(),
    )

    interpreter_instance._set_block_inputs(action, [10, 20])
    assert interpreter_instance.context.env.get("foo") == 10
    assert interpreter_instance.context.env.get("bar") == 20


def test_action_inputs_extra_kwarg_raises_error(interpreter_instance):
    args = nmx_runtime.Struct()
    args.set(key="foo", value=10)
    args.set(key="baz", value=20)
    args = args.unbox_in(args)

    action = DummyAction(
        name="my_action",
        input=[DummyActionInput("foo", True)],
        output=[],
        children=[],
        meta=make_meta(),
    )

    with pytest.raises(
        nmx_ex.NemantixRuntimeException, match=r"unexpected keyword arguments: baz"
    ):
        interpreter_instance._set_block_inputs(action, args)


def test_action_inputs_too_many_positional_raises_error(interpreter_instance):
    action = DummyAction(
        name="my_action",
        input=[DummyActionInput("foo", True)],
        output=[],
        children=[],
        meta=make_meta(),
    )

    with pytest.raises(
        nmx_ex.NemantixRuntimeException, match=r"expects at most 1 positional arguments"
    ):
        interpreter_instance._set_block_inputs(action, [10, 20])


def test_action_inputs_missing_required_raises_error(interpreter_instance):
    args = nmx_runtime.Struct()
    args.set(key="foo", value=10)

    action = DummyAction(
        name="my_action",
        input=[DummyActionInput("foo", True), DummyActionInput("bar", True)],
        output=[],
        children=[],
        meta=make_meta(),
    )

    with pytest.raises(
        nmx_ex.NemantixRuntimeException, match=r'Missing required argument "bar"'
    ):
        interpreter_instance._set_block_inputs(action, args)


def test_action_inputs_uses_defaults_for_missing_optional(interpreter_instance):
    args = nmx_runtime.Struct()
    args.set(key="foo", value=10)
    args = args.unbox_in(args)

    # default is an Expression, so we pass a SingleValue(99)
    action = DummyAction(
        name="my_action",
        input=[
            DummyActionInput("foo", True),
            DummyActionInput("bar", False, default=make_value(99)),
        ],
        output=[],
        children=[],
        meta=make_meta(),
    )

    interpreter_instance._set_block_inputs(action, args)
    assert interpreter_instance.context.env.get("foo") == 10
    assert interpreter_instance.context.env.get("bar") == 99


# =============================================================================
# Do LLM with producing_schema tests (new: schema via frame path navigator)
# =============================================================================


def test_do_llm_structured_output(interpreter_instance):
    # Put a runtime Frame in frames memory
    person = nmx_runtime.Frame("PERSON")
    person.add_slot("name", cardinality="1", types=[{"name": SlotTypesEnum.TEXT}])
    person.add_slot("age", cardinality="0..1", types=[{"name": SlotTypesEnum.INT}])

    interpreter_instance.context.frames["PERSON"] = person

    do_stmt = make_node(
        nmx_nodes.DoStatement,
        name="llm",
        callable_type=None,
        using=make_value("Extract Luigi who is 42", _STRING_TYPE),
        prompt=None,
        producing=make_var("output_var"),
        producing_schema="PERSON",
        meta=make_meta(),
    )

    interpreter_instance.interpret_do_statement(do_stmt)

    # Ensure our dummy llm was called
    assert len(interpreter_instance.llm.calls) == 1
    called_prompt, called_schema = interpreter_instance.llm.calls[0]
    assert called_prompt == "Extract Luigi who is 42"
    assert isinstance(called_schema, type)  # pydantic model class

    # Result stored in env as packed return value (dict -> Struct)
    out = interpreter_instance.context.env.get("output_var")
    assert isinstance(out, nmx_runtime.Struct)
    assert out.get("name") == "Luigi"
    assert out.get("age") == 42


def test_do_llm_structured_missing_frame(interpreter_instance):
    do_stmt = make_node(
        nmx_nodes.DoStatement,
        name="llm",
        callable_type=None,
        using=make_value("hi", _STRING_TYPE),
        prompt=None,
        producing=make_var("out"),
        producing_schema="GHOST_FRAME",
        meta=make_meta(),
    )

    with pytest.raises(
        nmx_ex.NemantixRuntimeException,
        match=r"Undefined root frame referenced: GHOST_FRAME",
    ):
        interpreter_instance.interpret_do_statement(do_stmt)


def test_do_llm_unsupported_generative_schema(interpreter_instance):
    # producing_schema not str -> unsupported
    do_stmt = make_node(
        nmx_nodes.DoStatement,
        name="llm",
        callable_type=None,
        using=make_value("hi", _STRING_TYPE),
        prompt=None,
        producing=make_var("out"),
        producing_schema=make_node(
            nmx_nodes.MicroPrompt, prompt="make it up", meta=make_meta()
        ),
        meta=make_meta(),
    )

    with pytest.raises(
        nmx_ex.NemantixRuntimeException,
        match=r"Generative schema blocks are not yet supported",
    ):
        interpreter_instance.interpret_do_statement(do_stmt)


def test_do_tool_structured_output_applies_frame(interpreter_instance):
    """
    Tests that removing the 'fn_name == "llm"' check allows standard tools
    to have their outputs validated and type-cast by a producing_schema.
    """
    # frame with an optional 'age' field
    person = nmx_runtime.Frame("PERSON")
    person.add_slot("name", cardinality="1", types=[{"name": SlotTypesEnum.TEXT}])
    person.add_slot("age", cardinality="0..1", types=[{"name": SlotTypesEnum.INT}])
    interpreter_instance.context.frames["PERSON"] = person

    # A dummy tool that returns a raw dict (missing the 'age' field)
    interpreter_instance.context.tools["get_person"] = lambda: {"name": "Mario"}

    do_stmt = make_node(
        nmx_nodes.DoStatement,
        name="get_person",
        callable_type=nmx_nodes.CallableTypeEnum.TOOL,
        using=None,
        prompt=None,
        producing=make_var("output_var"),
        producing_schema="PERSON",
        meta=make_meta(),
    )

    interpreter_instance.interpret_do_statement(do_stmt)

    # Verify the output was cast through the frame
    out = interpreter_instance.context.env.get("output_var")

    assert isinstance(out, nmx_runtime.Struct)
    assert out.get("name") == "Mario"
    # Proof that the frame was applied! The tool didn't return 'age',
    # but apply_postfix safely defaulted it to 0.
    assert out.get("age") == 0


# =============================================================================
# Action implicit/explicit return tests (works with DummyAction + real statements)
# =============================================================================


@dataclass
class DummyActionOutput:
    name: str


def test_action_implicit_return_single_out(interpreter_instance):
    assignment = make_node(
        nmx_nodes.Assignment,
        var=make_var("my_result"),
        value=make_value(42),
        meta=make_meta(),
    )

    action = DummyAction(
        name="test_implicit_return",
        input=[],
        output=[DummyActionOutput("my_result")],
        children=[assignment],
        meta=make_meta(),
    )

    result = interpreter_instance.interpret_block(action)
    assert result == 42


def test_action_implicit_return_multiple_out(interpreter_instance):
    assign1 = make_node(
        nmx_nodes.Assignment,
        var=make_var("res1"),
        value=make_value(10),
        meta=make_meta(),
    )
    assign2 = make_node(
        nmx_nodes.Assignment,
        var=make_var("res2"),
        value=make_value(20),
        meta=make_meta(),
    )

    action = DummyAction(
        name="test_implicit_multiple",
        input=[],
        output=[DummyActionOutput("res1"), DummyActionOutput("res2")],
        children=[assign1, assign2],
        meta=make_meta(),
    )

    result = interpreter_instance.interpret_block(action)
    assert result == [10, 20]


def test_action_explicit_return_overrides_implicit(interpreter_instance):
    assign = make_node(
        nmx_nodes.Assignment,
        var=make_var("my_result"),
        value=make_value(42),
        meta=make_meta(),
    )
    explicit_return = make_node(
        nmx_nodes.Return, val=[make_value(99)], meta=make_meta()
    )

    action = DummyAction(
        name="test_explicit_override",
        input=[],
        output=[DummyActionOutput("my_result")],
        children=[assign, explicit_return],
        meta=make_meta(),
    )

    result = interpreter_instance.interpret_block(action)
    assert result == 99


# =============================================================================
# New feature coverage: _frame_to_pydantic_schema nested frame paths
# =============================================================================


def test_frame_to_pydantic_schema_nested_frame(interpreter_instance):
    root = nmx_runtime.Frame("ROOT")
    inner = nmx_runtime.Frame("INNER")
    inner.add_slot("code", cardinality="1", types=[{"name": SlotTypesEnum.TEXT}])
    root.add_frame(inner)
    root.add_slot(
        "child", cardinality="1", types=[{"type": SlotTypesEnum.FRAME, "name": "INNER"}]
    )

    interpreter_instance.context.frames["ROOT"] = root

    schema = interpreter_instance._frame_to_pydantic_schema("ROOT")
    assert "child" in schema.model_fields
    # nested is a BaseModel subclass
    nested_type = schema.model_fields["child"].annotation
    assert isinstance(nested_type, type) and issubclass(nested_type, BaseModel)
    assert "code" in nested_type.model_fields


# -----------------------------------------------------------------------------
# Schemed collection Tests
# -----------------------------------------------------------------------------


def test_eval_schemed_collection_pre_apply_as_frame(interpreter_instance):
    """
    Tests evaluation using an AsFrame dataframe, resolving from the global context,
    and applying a PRE (prefix) schema.
    """
    # 1. Setup frame in context
    frame = nmx_runtime.Frame("MY_FRAME")
    frame.apply_prefix = MagicMock(return_value="pre_applied_struct")
    interpreter_instance.context.frames["MY_FRAME"] = frame

    # 2. Setup SchemedCollection using AsFrame
    as_frame = AsFrame(value="my_frame", meta=make_meta())
    inner_collection = make_node(
        nmx_nodes.Collection,
        value=[make_value(42)],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    schemed_col = make_node(
        nmx_nodes.SchemedCollection,
        value=inner_collection,
        dataframe=as_frame,
        apply_type=nmx_nodes.FrameApplyEnum.PRE,
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    # 3. Execute
    result = interpreter_instance.eval_schemed_collection(schemed_col)

    # 4. Assert
    frame.apply_prefix.assert_called_once()
    assert result == "pre_applied_struct"


def test_eval_schemed_collection_post_apply_collection_df(interpreter_instance):
    """
    Tests evaluation when the dataframe is a Collection (value becomes the frame name),
    resolving from the global context, and applying a POST (suffix) schema.
    """
    # 1. Setup frame in context
    frame = nmx_runtime.Frame("OTHER_FRAME")
    frame.apply_postfix = MagicMock(return_value="post_applied_struct")
    interpreter_instance.context.frames["OTHER_FRAME"] = frame

    # 2. Setup SchemedCollection where dataframe is a Collection
    inner_collection = make_node(
        nmx_nodes.Collection,
        value=[make_value("data")],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    schemed_col = make_node(
        nmx_nodes.SchemedCollection,
        value="other_frame",  # In this branch, value contains the frame name string
        dataframe=inner_collection,
        apply_type=nmx_nodes.FrameApplyEnum.POST,
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    # 3. Execute
    result = interpreter_instance.eval_schemed_collection(schemed_col)

    # 4. Assert
    frame.apply_postfix.assert_called_once()
    assert result == "post_applied_struct"


def test_eval_schemed_collection_with_enclosing_frame(interpreter_instance):
    """
    Tests evaluation where the frame is resolved via a provided enclosing_frame.
    """
    # 1. Setup enclosing frame with a sub-frame
    enclosing_frame = nmx_runtime.Frame("PARENT")
    sub_frame = nmx_runtime.Frame("SUB")
    sub_frame.apply_prefix = MagicMock(return_value="sub_pre_applied")
    enclosing_frame.frames["SUB"] = sub_frame

    # 2. Setup SchemedCollection
    as_frame = AsFrame(value="sub", meta=make_meta())
    inner_collection = make_node(
        nmx_nodes.Collection,
        value=[make_value(1)],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    schemed_col = make_node(
        nmx_nodes.SchemedCollection,
        value=inner_collection,
        dataframe=as_frame,
        apply_type=nmx_nodes.FrameApplyEnum.PRE,
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    # 3. Execute with enclosing_frame passed
    result = interpreter_instance.eval_schemed_collection(
        schemed_col, enclosing_frame=enclosing_frame
    )

    # 4. Assert
    sub_frame.apply_prefix.assert_called_once()
    assert result == "sub_pre_applied"


def test_eval_schemed_collection_missing_frame_raises(interpreter_instance):
    """
    Tests that a NemantixRuntimeException is raised if the target global frame is undefined.
    """
    as_frame = AsFrame(value="missing_frame", meta=make_meta())
    inner_collection = make_node(
        nmx_nodes.Collection,
        value=[],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    schemed_col = make_node(
        nmx_nodes.SchemedCollection,
        value=inner_collection,
        dataframe=as_frame,
        apply_type=nmx_nodes.FrameApplyEnum.PRE,
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    with pytest.raises(
        nmx_ex.NemantixRuntimeException, match=r'Undefined frame "MISSING_FRAME"'
    ):
        interpreter_instance.eval_schemed_collection(schemed_col)


def test_eval_schemed_collection_missing_subframe_raises(interpreter_instance):
    """
    Tests that a NemantixRuntimeException is raised if the target sub-frame is missing
    from the enclosing_frame.
    """
    enclosing_frame = nmx_runtime.Frame("PARENT")

    as_frame = AsFrame(value="missing_sub", meta=make_meta())
    inner_collection = make_node(
        nmx_nodes.Collection,
        value=[],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    schemed_col = make_node(
        nmx_nodes.SchemedCollection,
        value=inner_collection,
        dataframe=as_frame,
        apply_type=nmx_nodes.FrameApplyEnum.PRE,
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    with pytest.raises(
        nmx_ex.NemantixRuntimeException,
        match=r'Undefined frame "MISSING_SUB" in frame "PARENT"',
    ):
        interpreter_instance.eval_schemed_collection(
            schemed_col, enclosing_frame=enclosing_frame
        )


# =============================================================================
# Frame application on already-defined operands (variables / JSON strings)
# =============================================================================


def _person_frame() -> nmx_runtime.Frame:
    f = nmx_runtime.Frame("PERSON")
    f.add_slot("name", cardinality="1", types=[{"name": nmx_nodes.SlotTypesEnum.TEXT}])
    f.add_slot("age", cardinality="0..1", types=[{"name": nmx_nodes.SlotTypesEnum.INT}])
    return f


def _schemed_var(var_name, apply_type, path=None):
    return make_node(
        nmx_nodes.SchemedCollection,
        value=make_var(var_name, path=path),
        dataframe=AsFrame(value="person", meta=make_meta()),
        apply_type=apply_type,
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )


def test_eval_schemed_collection_on_variable_postfix(interpreter_instance):
    """[my_struct]{Person} — loose application on a variable holding a Struct."""
    interpreter_instance.context.frames["PERSON"] = _person_frame()
    s = nmx_runtime.Struct()
    s.set("John", key="name")
    interpreter_instance.context.env.set(var_name="my_struct", value=s)

    result = interpreter_instance.eval_schemed_collection(
        _schemed_var("my_struct", nmx_nodes.FrameApplyEnum.POST)
    )
    assert isinstance(result, nmx_runtime.Struct)
    assert result.get("name") == "John"
    assert result.get("age") == 0  # default filled by postfix


def test_eval_schemed_collection_on_variable_prefix_strict(interpreter_instance):
    """{Person}[my_struct] — strict application returns None when slots missing."""
    interpreter_instance.context.frames["PERSON"] = _person_frame()
    complete = nmx_runtime.Struct()
    complete.set("John", key="name")
    complete.set(30, key="age")
    interpreter_instance.context.env.set(var_name="ok", value=complete)

    result = interpreter_instance.eval_schemed_collection(
        _schemed_var("ok", nmx_nodes.FrameApplyEnum.PRE)
    )
    assert isinstance(result, nmx_runtime.Struct)
    assert result.get("age") == 30

    partial = nmx_runtime.Struct()
    partial.set("John", key="name")  # missing "age"
    interpreter_instance.context.env.set(var_name="partial", value=partial)
    assert (
        interpreter_instance.eval_schemed_collection(
            _schemed_var("partial", nmx_nodes.FrameApplyEnum.PRE)
        )
        is None
    )


def test_eval_schemed_collection_on_path_access(interpreter_instance):
    """[outer:inner]{Person} — operand is a nested struct via path access."""
    interpreter_instance.context.frames["PERSON"] = _person_frame()
    inner = nmx_runtime.Struct()
    inner.set("Ann", key="name")
    outer = nmx_runtime.Struct()
    outer.set(inner, key="inner")
    interpreter_instance.context.env.set(var_name="outer", value=outer)

    result = interpreter_instance.eval_schemed_collection(
        _schemed_var("outer", nmx_nodes.FrameApplyEnum.POST, path=[make_value("inner")])
    )
    assert isinstance(result, nmx_runtime.Struct)
    assert result.get("name") == "Ann"


def test_eval_schemed_collection_on_json_string(interpreter_instance):
    """[json]{Person} — operand is a string holding valid JSON."""
    interpreter_instance.context.frames["PERSON"] = _person_frame()
    interpreter_instance.context.env.set(
        var_name="json", value='{"name": "Zoe", "age": 9}'
    )

    result = interpreter_instance.eval_schemed_collection(
        _schemed_var("json", nmx_nodes.FrameApplyEnum.POST)
    )
    assert isinstance(result, nmx_runtime.Struct)
    assert result.get("name") == "Zoe"
    assert result.get("age") == 9


def test_eval_schemed_collection_quasi_json_repaired_by_llm(interpreter_instance):
    """In LENIENT mode, invalid JSON is repaired via an LLM call, then applied."""
    interpreter_instance.expertise.json_parsing = JsonParsingMode.LENIENT
    interpreter_instance.context.frames["PERSON"] = _person_frame()
    # trailing comma -> json.loads fails, triggering repair
    interpreter_instance.context.env.set(
        var_name="bad", value='{"name": "Ivo", "age": 7,}'
    )

    interpreter_instance.proxies.external.invoke = MagicMock(
        return_value=SimpleNamespace(
            text='{"name": "Ivo", "age": 7}',
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            proxy=interpreter_instance.llm,  # DummyLLM: has get_name() for event emission
        )
    )

    result = interpreter_instance.eval_schemed_collection(
        _schemed_var("bad", nmx_nodes.FrameApplyEnum.POST)
    )
    interpreter_instance.proxies.external.invoke.assert_called_once()
    assert isinstance(result, nmx_runtime.Struct)
    assert result.get("name") == "Ivo"
    assert result.get("age") == 7


def test_eval_schemed_collection_quasi_json_strict_raises(interpreter_instance):
    """In STRICT mode (default), invalid JSON raises and never calls the LLM."""
    interpreter_instance.expertise.json_parsing = JsonParsingMode.STRICT
    interpreter_instance.context.frames["PERSON"] = _person_frame()
    interpreter_instance.context.env.set(
        var_name="bad",
        value='{"name": "Ivo", "age": 7,}',  # trailing comma
    )
    interpreter_instance.proxies.internal.invoke = MagicMock()

    with pytest.raises(
        nmx_ex.NemantixRuntimeException, match=r"Invalid JSON for frame application"
    ):
        interpreter_instance.eval_schemed_collection(
            _schemed_var("bad", nmx_nodes.FrameApplyEnum.POST)
        )
    interpreter_instance.proxies.internal.invoke.assert_not_called()


def test_parse_json_emits_success_event(interpreter_instance):
    """Valid JSON emits a frame_apply success signal."""
    interpreter_instance._emit_json_parse = MagicMock()
    interpreter_instance.context.frames["PERSON"] = _person_frame()
    interpreter_instance.context.env.set(
        var_name="j", value='{"name": "Zoe", "age": 9}'
    )

    interpreter_instance.eval_schemed_collection(
        _schemed_var("j", nmx_nodes.FrameApplyEnum.POST)
    )

    interpreter_instance._emit_json_parse.assert_called_once()
    call = interpreter_instance._emit_json_parse.call_args
    assert call.args[1] is True  # success
    assert call.args[2] == "frame_apply"  # source
    assert call.kwargs["repaired"] is False


def test_parse_json_emits_failure_event_strict(interpreter_instance):
    """Invalid JSON under strict emits a failure signal (mode='strict')."""
    interpreter_instance.expertise.json_parsing = JsonParsingMode.STRICT
    interpreter_instance._emit_json_parse = MagicMock()
    interpreter_instance.context.frames["PERSON"] = _person_frame()
    interpreter_instance.context.env.set(var_name="bad", value='{"name": "x",}')

    with pytest.raises(nmx_ex.NemantixRuntimeException):
        interpreter_instance.eval_schemed_collection(
            _schemed_var("bad", nmx_nodes.FrameApplyEnum.POST)
        )

    call = interpreter_instance._emit_json_parse.call_args
    assert call.args[1] is False  # success
    assert call.kwargs["mode"] == "strict"


def test_parse_json_emits_repaired_success_event_lenient(interpreter_instance):
    """Lenient repair emits a success signal with repaired=True and the LLM name."""
    interpreter_instance.expertise.json_parsing = JsonParsingMode.LENIENT
    interpreter_instance.context.frames["PERSON"] = _person_frame()
    interpreter_instance.context.env.set(
        var_name="bad", value='{"name": "Ivo", "age": 7,}'
    )
    interpreter_instance.proxies.external.invoke = MagicMock(
        return_value=SimpleNamespace(
            text='{"name": "Ivo", "age": 7}',
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            proxy=interpreter_instance.llm,
        )
    )
    interpreter_instance._emit_json_parse = MagicMock()

    interpreter_instance.eval_schemed_collection(
        _schemed_var("bad", nmx_nodes.FrameApplyEnum.POST)
    )

    call = interpreter_instance._emit_json_parse.call_args
    assert call.args[1] is True  # success
    assert call.kwargs["repaired"] is True
    assert call.kwargs["name"] is not None


def test_eval_schemed_collection_scalar_json_raises(interpreter_instance):
    """A JSON scalar (not object/array) cannot be a struct -> runtime error."""
    interpreter_instance.context.frames["PERSON"] = _person_frame()
    interpreter_instance.context.env.set(var_name="scalar", value="42")

    with pytest.raises(
        nmx_ex.NemantixRuntimeException, match=r"JSON must be an object or array"
    ):
        interpreter_instance.eval_schemed_collection(
            _schemed_var("scalar", nmx_nodes.FrameApplyEnum.POST)
        )


@pytest.mark.parametrize("bad_value", [42, 3.14, True, None])
def test_eval_schemed_collection_on_non_struct_value_raises(
    interpreter_instance, bad_value
):
    """A variable holding a non-structure, non-string value (int/float/bool/none)
    cannot have a frame applied -> runtime error (distinct from the JSON-scalar path)."""
    interpreter_instance.context.frames["PERSON"] = _person_frame()
    interpreter_instance.context.env.set(var_name="x", value=bad_value)

    # matches the bare "...structure." message, NOT the JSON "(...object or array)" one
    with pytest.raises(
        nmx_ex.NemantixRuntimeException,
        match=r"can only be applied to a structure\.",
    ):
        interpreter_instance.eval_schemed_collection(
            _schemed_var("x", nmx_nodes.FrameApplyEnum.POST)
        )


def test_eval_schemed_collection_lenient_repair_failure_raises(interpreter_instance):
    """LENIENT mode: when the LLM 'repair' is still invalid JSON, raise 'repair failed'
    and emit a JSON_PARSE failure with repaired=True."""
    interpreter_instance.expertise.json_parsing = JsonParsingMode.LENIENT
    interpreter_instance.context.frames["PERSON"] = _person_frame()
    interpreter_instance.context.env.set(
        var_name="bad",
        value='{"name": "x",}',  # trailing comma -> triggers repair
    )
    # the "repaired" text the LLM returns is itself still not valid JSON
    interpreter_instance.proxies.external.invoke = MagicMock(
        return_value=SimpleNamespace(
            text="still {not} json",
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            proxy=interpreter_instance.llm,
        )
    )
    interpreter_instance._emit_json_parse = MagicMock()

    with pytest.raises(nmx_ex.NemantixRuntimeException, match=r"repair failed"):
        interpreter_instance.eval_schemed_collection(
            _schemed_var("bad", nmx_nodes.FrameApplyEnum.POST)
        )

    interpreter_instance.proxies.external.invoke.assert_called_once()  # repair attempted
    call = interpreter_instance._emit_json_parse.call_args
    assert call.args[1] is False  # success
    assert call.kwargs["repaired"] is True


def _person_with_date_frame() -> nmx_runtime.Frame:
    """PERSON with a nested DATE frame typing its `date` slot.

    frame Person:
        slot name    as TEXT
        slot surname as TEXT
        slot date    as DATE
        frame Date:
            slot day   as INT
            slot month as INT
            slot year  as INT
    """
    person = nmx_runtime.Frame("PERSON")
    date = nmx_runtime.Frame("DATE")
    for part in ("day", "month", "year"):
        date.add_slot(
            part, cardinality="1", types=[{"name": nmx_nodes.SlotTypesEnum.INT}]
        )
    person.add_frame(date)
    person.add_slot(
        "name", cardinality="1", types=[{"name": nmx_nodes.SlotTypesEnum.TEXT}]
    )
    person.add_slot(
        "surname", cardinality="1", types=[{"name": nmx_nodes.SlotTypesEnum.TEXT}]
    )
    person.add_slot(
        "date",
        cardinality="1",
        types=[{"type": nmx_nodes.SlotTypesEnum.FRAME, "name": "DATE"}],
    )
    return person


def test_eval_schemed_collection_nested_frame_on_variable(interpreter_instance):
    """{Person}[var] where var holds a struct with a nested DATE struct."""
    interpreter_instance.context.frames["PERSON"] = _person_with_date_frame()

    date = nmx_runtime.Struct()
    date.set(7, key="day")
    date.set(3, key="month")
    date.set(1998, key="year")
    person = nmx_runtime.Struct()
    person.set("Ada", key="name")
    person.set("Lovelace", key="surname")
    person.set(date, key="date")
    interpreter_instance.context.env.set(var_name="p", value=person)

    result = interpreter_instance.eval_schemed_collection(
        _schemed_var("p", nmx_nodes.FrameApplyEnum.PRE)  # prefix = strict
    )
    assert isinstance(result, nmx_runtime.Struct)
    assert result.get("name") == "Ada"
    inner = result.get("date")
    assert isinstance(inner, nmx_runtime.Struct)
    assert inner.get("day") == 7
    assert inner.get("month") == 3
    assert inner.get("year") == 1998


def test_eval_schemed_collection_nested_frame_on_json(interpreter_instance):
    """[json]{Person} where the JSON string carries a nested DATE object."""
    interpreter_instance.context.frames["PERSON"] = _person_with_date_frame()
    interpreter_instance.context.env.set(
        var_name="payload",
        value=(
            '{"name": "Ada", "surname": "Lovelace", '
            '"date": {"day": 7, "month": 3, "year": 1998}}'
        ),
    )

    result = interpreter_instance.eval_schemed_collection(
        _schemed_var("payload", nmx_nodes.FrameApplyEnum.PRE)  # strict
    )
    assert isinstance(result, nmx_runtime.Struct)
    assert result.get("surname") == "Lovelace"
    inner = result.get("date")
    assert isinstance(inner, nmx_runtime.Struct)
    assert inner.get("day") == 7
    assert inner.get("month") == 3
    assert inner.get("year") == 1998


# =============================================================================
# interpret_imports — toolset resolution error propagation
# =============================================================================


def _make_import(name: str, elements) -> nmx_nodes.ImportToolsetStatement:
    return nmx_nodes.ImportToolsetStatement(
        name=name,
        elements=elements,
        args=None,
        alias=None,
        meta=make_meta(),
    )


def test_interpret_imports_wildcard_unknown_toolset_raises_with_original_message(
    interpreter_instance,
):
    """from _GhostToolset import * — load() failure must surface as RuntimeException."""
    stmt = _make_import("_GhostToolset", "*")
    with pytest.raises(nmx_ex.NemantixRuntimeException, match=r"_GhostToolset"):
        interpreter_instance.interpret_imports([stmt])


def test_interpret_imports_direct_unknown_toolset_raises_with_original_message(
    interpreter_instance,
):
    """from _GhostToolset import some_tool — load() failure must surface as RuntimeException."""
    stmt = _make_import("_GhostToolset", ["some_tool"])
    with pytest.raises(nmx_ex.NemantixRuntimeException, match=r"_GhostToolset"):
        interpreter_instance.interpret_imports([stmt])


# =============================================================================
# Tests for _set_block_inputs
# =============================================================================


def test_set_block_inputs_mixed_args_correct_order(interpreter_instance):
    """Test valid mixed arguments (positional followed by keyword)."""
    action = DummyAction(
        name="mixed_action",
        input=[
            DummyActionInput("pos1", True),
            DummyActionInput("kw1", True),
            DummyActionInput("kw2", False, default=make_value("default_val")),
        ],
        output=[],
        children=[],
        meta=make_meta(),
    )
    # Provided: 1 positional, 1 keyword, 1 omitted (should use default)
    provided = ([100], {"kw1": 200})

    interpreter_instance._set_block_inputs(action, provided)
    assert interpreter_instance.context.env.get("pos1") == 100
    assert interpreter_instance.context.env.get("kw1") == 200
    assert interpreter_instance.context.env.get("kw2") == "default_val"


def test_set_block_inputs_collision_raises(interpreter_instance):
    """Test that providing the same argument positionally and nominally raises an error."""
    action = DummyAction(
        name="collision_action",
        input=[DummyActionInput("foo", True), DummyActionInput("bar", True)],
        output=[],
        children=[],
        meta=make_meta(),
    )

    # User tries: `do collision_action using [10, foo=20]`
    provided = ([10], {"foo": 20})

    with pytest.raises(
        nmx_ex.NemantixRuntimeException, match=r'got multiple values for argument "foo"'
    ):
        interpreter_instance._set_block_inputs(action, provided)


def test_set_block_inputs_skips_unnamed_microprompt(interpreter_instance):
    """Test that unnamed inputs (microprompts) are safely skipped and don't consume positional args."""
    action = DummyAction(
        name="microprompt_action",
        input=[
            DummyActionInput(
                "",
                True,
                prompt=nmx_nodes.MicroPrompt(prompt="Do a thing", meta=make_meta()),
            ),
            DummyActionInput("real_arg", True),
        ],
        output=[],
        children=[],
        meta=make_meta(),
    )

    # Even though there are 2 inputs in the block, only 1 is named.
    # Passing 1 positional arg should map to 'real_arg'.
    interpreter_instance._set_block_inputs(action, [42])
    assert interpreter_instance.context.env.get("real_arg") == 42


def test_set_block_inputs_too_many_positionals_with_microprompts(interpreter_instance):
    """Test that max positional bounds check correctly ignores unnamed microprompt inputs."""
    action = DummyAction(
        name="microprompt_action_err",
        input=[
            DummyActionInput(
                "",
                True,
                prompt=nmx_nodes.MicroPrompt(prompt="Do a thing", meta=make_meta()),
            ),
            DummyActionInput("real_arg", True),
        ],
        output=[],
        children=[],
        meta=make_meta(),
    )

    with pytest.raises(
        nmx_ex.NemantixRuntimeException, match=r"expects at most 1 positional arguments"
    ):
        interpreter_instance._set_block_inputs(action, [42, 99])


# =============================================================================
# Tests for interpret_expression
# =============================================================================


def test_interpret_expression_list_unwrap(interpreter_instance):
    """Test that outer list wrappers are stripped (simulating parser artifacts)."""
    inner_val = make_value(100)
    # The parser sometimes wraps nodes in a list: [SingleValue(100)]
    result = interpreter_instance.interpret_expression([inner_val])
    assert result == 100


def test_interpret_expression_collection_to_struct(interpreter_instance):
    """Test that a Collection of values evaluates to a Struct."""
    collection = make_node(
        nmx_nodes.Collection,
        value=[
            make_value(1),
            make_value(2),
            {"c": make_value(3)},
        ],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )
    result = interpreter_instance.interpret_expression(collection)

    assert isinstance(result, nmx_runtime.Struct)
    assert result.get(0) == 1
    assert result.get(1) == 2
    assert result.get("c") == 3


def test_interpret_expression_builtin_function(interpreter_instance):
    """Test builtin function execution within an expression."""
    builtin_expr = make_node(
        nmx_nodes.BuiltinFunction,
        function=nmx_nodes.BuiltinFunctionEnum.SIZE,
        args=[make_value("hello world", _STRING_TYPE)],
        meta=make_meta(),
    )
    result = interpreter_instance.interpret_expression(builtin_expr)
    assert result == 11  # len("hello world")


def test_interpret_expression_meta_expression(interpreter_instance):
    """Test retrieval of intentables via MetaExpression."""
    # Setup intentable in metadata memory
    intentable = nmx_runtime.Metadata()
    intentable["goal"] = "Test Goal"
    interpreter_instance.metadata["my_intent"] = intentable

    meta_expr = make_node(
        nmx_nodes.MetaExpression, quals=["my_intent", "goal"], meta=make_meta()
    )
    result = interpreter_instance.interpret_expression(meta_expr)
    assert result == "Test Goal"


def test_interpret_expression_meta_expression_missing_raises(interpreter_instance):
    """Test missing intentable label raises exception."""
    meta_expr = make_node(
        nmx_nodes.MetaExpression, quals=["ghost_intent"], meta=make_meta()
    )
    with pytest.raises(
        nmx_ex.NemantixRuntimeException, match=r'Intentable "ghost_intent" not defined!'
    ):
        interpreter_instance.interpret_expression(meta_expr)


# =============================================================================
# Tests for interpret_do_statement
# =============================================================================


def test_do_statement_action_call(interpreter_instance):
    """Test execution of a global Action call."""
    # Mock an action closure in context
    called_args = None

    def mock_closure(args, callee=None):
        nonlocal called_args
        called_args = args
        return "action_success"

    interpreter_instance.context.actions["my_global_action"] = {
        "closure": mock_closure,
        "is_global": True,
        "imported_by": set(),
    }

    # Create do statement: do action my_global_action using [42] producing [out_var]
    using_col = make_node(
        nmx_nodes.Collection,
        value=[make_value(42)],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )
    do_stmt = make_node(
        nmx_nodes.DoStatement,
        name="my_global_action",
        callable_type=nmx_nodes.CallableTypeEnum.ACTION,
        using=using_col,
        producing=make_var("out_var"),
        prompt=None,
        producing_schema=None,
        meta=make_meta(),
    )

    # Use make_node to safely instantiate Deliberate regardless of __init__ signature
    dummy_deliberate = make_node(
        nmx_nodes.Deliberate, name="dummy_delib", meta=make_meta()
    )
    interpreter_instance._set_global_deliberate(dummy_deliberate)

    interpreter_instance.interpret_do_statement(do_stmt)

    # Assert closure was called with positional 42 and empty kwargs
    assert called_args == ([42], {})
    # Assert environment was updated with output
    assert interpreter_instance.context.env.get("out_var") == "action_success"


def test_do_statement_private_action_cross_call_raises(interpreter_instance):
    """Test calling a private action from a deliberate that didn't import/own it."""
    interpreter_instance.context.actions["private_act"] = {
        "closure": lambda args, callee: None,
        "is_global": False,
        "imported_by": {"other_deliberate"},
    }

    do_stmt = make_node(
        nmx_nodes.DoStatement,
        name="private_act",
        callable_type=nmx_nodes.CallableTypeEnum.ACTION,
        using=None,
        producing=None,
        prompt=None,
        producing_schema=None,
        meta=make_meta(),
    )

    my_deliberate = make_node(nmx_nodes.Deliberate, name="my_delib", meta=make_meta())
    interpreter_instance._set_global_deliberate(my_deliberate)

    with pytest.raises(
        nmx_ex.NemantixRuntimeException,
        match=r'Private action "private_act" cannot be called from deliberate "my_delib"',
    ):
        interpreter_instance.interpret_do_statement(do_stmt)


def test_do_statement_tool_call_unboxes_opaques(interpreter_instance):
    """Test Tool calls properly unbox Opaque objects and Structs before passing to Python functions."""
    called_kwargs = None

    def mock_tool(**kwargs):
        nonlocal called_kwargs
        called_kwargs = kwargs
        return "tool_success"

    interpreter_instance.context.tools["my_tool"] = mock_tool

    # Create an Opaque object and put it directly in the operational environment
    opaque_obj = nmx_runtime.Opaque(obj="hidden_secret_string")
    interpreter_instance.context.env.set(var_name="secret_var", value=opaque_obj)

    # Assignment: [data] = secret_var
    using_assign = make_node(
        nmx_nodes.Assignment,
        var=make_var("data"),
        value=make_var("secret_var"),  # Use a Variable node to look it up
        meta=make_meta(),
    )

    do_stmt = make_node(
        nmx_nodes.DoStatement,
        name="my_tool",
        callable_type=nmx_nodes.CallableTypeEnum.TOOL,
        using=using_assign,
        producing=None,
        prompt=None,
        producing_schema=None,
        meta=make_meta(),
    )

    interpreter_instance.interpret_do_statement(do_stmt)

    # Assert the tool received the UNBOXED string, not the Opaque wrapper
    assert called_kwargs == {"data": "hidden_secret_string"}


def test_do_statement_builtin_print(interpreter_instance, capsys):
    """Test execution of builtin print."""
    using_col = make_node(
        nmx_nodes.Collection,
        value=[make_value("Hello Builtin")],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )
    do_stmt = make_node(
        nmx_nodes.DoStatement,
        name="print",
        callable_type=None,
        using=using_col,
        producing=None,
        prompt=None,
        producing_schema=None,
        meta=make_meta(),
    )

    interpreter_instance.interpret_do_statement(do_stmt)
    captured = capsys.readouterr()
    assert "Hello Builtin\n" in captured.out


class DummyKnowledgeBase:
    """Minimal duck-typed KB stub for the retrieve/expand/extend/generalize builtins."""

    def __init__(self):
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(("retrieve", kwargs))
        return []

    def expand(self, **kwargs):
        self.calls.append(("expand", kwargs))
        return {"node_id": kwargs["node_id"], "content": "c", "breadcrumbs": ""}

    def extend(self, **kwargs):
        self.calls.append(("extend", kwargs))
        return {"previous_sibling": None, "next_sibling": None}

    def generalize(self, **kwargs):
        self.calls.append(("generalize", kwargs))
        return {"node_id": kwargs["node_id"], "content": "c", "breadcrumbs": ""}


@pytest.mark.parametrize(
    "builtin_name, kwarg_name, kwarg_value",
    [
        ("retrieve", "query", "search terms"),
        ("expand", "node_id", "node_1"),
        ("extend", "node_id", "node_1"),
        ("generalize", "node_id", "node_1"),
    ],
)
def test_do_statement_kb_builtin_keyword_only_arg_no_index_error(
    interpreter_instance, builtin_name, kwarg_name, kwarg_value
):
    """Regression test: passing the KB builtin's argument only as a keyword must not raise IndexError.

    Reproduces a bug where `kwargs_.get(key, args_[0])` eagerly evaluated `args_[0]`
    even when `key` was already present in kwargs_, crashing when the argument arrived
    purely as a keyword (e.g. `do retrieve using [ [query] = [q] ] producing [ [hits] ]`).
    """
    interpreter_instance.knowledge_base = DummyKnowledgeBase()

    using_assign = make_node(
        nmx_nodes.Assignment,
        var=make_var(kwarg_name),
        value=make_value(kwarg_value),
        meta=make_meta(),
    )

    do_stmt = make_node(
        nmx_nodes.DoStatement,
        name=builtin_name,
        callable_type=None,
        using=using_assign,
        producing=make_var("out"),
        prompt=None,
        producing_schema=None,
        meta=make_meta(),
    )

    interpreter_instance.interpret_do_statement(do_stmt)

    call_name, call_kwargs = interpreter_instance.knowledge_base.calls[0]
    assert call_name == builtin_name
    assert call_kwargs[kwarg_name] == kwarg_value


def test_do_statement_tool_exception_wraps_in_nemantix_exception(interpreter_instance):
    """Test that a crashing Python tool is caught and wrapped in a NemantixRuntimeException."""

    def crashing_tool():
        raise ValueError("Critical Python Error")

    interpreter_instance.context.tools["crash_tool"] = crashing_tool

    do_stmt = make_node(
        nmx_nodes.DoStatement,
        name="crash_tool",
        callable_type=nmx_nodes.CallableTypeEnum.TOOL,
        using=None,
        producing=None,
        prompt=None,
        producing_schema=None,
        meta=make_meta(),
    )

    with pytest.raises(
        nmx_ex.NemantixRuntimeException,
        match=r'Exception in execution of "crashing_tool". Error: Critical Python Error',
    ):
        interpreter_instance.interpret_do_statement(do_stmt)


def test_do_statement_multiple_outputs_with_producing_schema(interpreter_instance):
    """
    Test that mapping multiple producing variables through a producing_schema
    correctly triggers the LLM schema mapping logic.
    """
    # 1. Setup frame in context
    person_frame = nmx_runtime.Frame("PERSON")
    person_frame.add_slot("name", cardinality="1", types=[{"name": SlotTypesEnum.TEXT}])
    person_frame.add_slot("age", cardinality="1", types=[{"name": SlotTypesEnum.INT}])
    interpreter_instance.context.frames["PERSON"] = person_frame

    # 2. Set up a tool that returns a raw list
    interpreter_instance.context.tools["get_data"] = lambda: ["Luigi", 42]

    # 3. Setup LLM to mock the schema mapping dict response
    # The prompt asks to map producing variable names to schema slot names.
    interpreter_instance.llm.invoke = MagicMock(
        return_value=SimpleNamespace(
            text="{'var_n': 'name', 'var_a': 'age'}",
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            proxy=interpreter_instance.llm,
        )
    )

    # 4. AST Nodes
    producing_col = make_node(
        nmx_nodes.Collection,
        value=[make_var("var_n"), make_var("var_a")],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    do_stmt = make_node(
        nmx_nodes.DoStatement,
        name="get_data",
        callable_type=nmx_nodes.CallableTypeEnum.TOOL,
        using=None,
        prompt=None,
        producing=producing_col,
        producing_schema="PERSON",
        meta=make_meta(),
    )

    # Execute
    interpreter_instance.interpret_do_statement(do_stmt)

    # Assert variables were set in the environment correctly based on LLM mapping + Frame casting
    assert interpreter_instance.context.env.get("var_n") == "Luigi"
    assert interpreter_instance.context.env.get("var_a") == 42


# =============================================================================
# SchemedCollection & AsFrame Integration Tests
# =============================================================================


def test_interpret_expression_schemed_collection_as_frame(interpreter_instance):
    """
    Tests that a SchemedCollection using an AsFrame dataframe successfully
    routes through interpret_expression and applies real Frame typing/defaults.
    """
    # 1. Set up a real frame
    point_frame = nmx_runtime.Frame("POINT")
    point_frame.add_slot("x", cardinality="1", types=[{"name": SlotTypesEnum.INT}])
    point_frame.add_slot("y", cardinality="1", types=[{"name": SlotTypesEnum.INT}])
    interpreter_instance.context.frames["POINT"] = point_frame

    # 2. Build AsFrame and SchemedCollection
    # AST equivalent of: {POINT}[prefix]: [x: 10, y: 20]
    as_frame = AsFrame(value="point", meta=make_meta())

    inner_dict = {"x": make_value(10), "y": make_value(20)}
    collection = make_node(
        nmx_nodes.Collection,
        value=[inner_dict],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    schemed_col = make_node(
        nmx_nodes.SchemedCollection,
        value=collection,
        dataframe=as_frame,
        apply_type=nmx_nodes.FrameApplyEnum.PRE,
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    # 3. Interpret via the main expression evaluator
    result = interpreter_instance.interpret_expression(schemed_col)

    # 4. Assert it returns a Struct correctly cast by the real Frame
    assert isinstance(result, nmx_runtime.Struct)
    assert result.get("x") == 10
    assert result.get("y") == 20


def test_eval_collection_with_nested_schemed_collection(interpreter_instance):
    """
    Tests the recursive nature of eval_collection by nesting a
    SchemedCollection inside a standard Collection.
    """
    # 1. Set up a real frame
    point_frame = nmx_runtime.Frame("POINT")
    point_frame.add_slot("x", cardinality="1", types=[{"name": SlotTypesEnum.INT}])
    interpreter_instance.context.frames["POINT"] = point_frame

    # 2. Build nested SchemedCollection
    # AST equivalent of: [ "some_string", {POINT}[prefix]: [x: 42] ]
    as_frame = AsFrame(value="point", meta=make_meta())

    inner_col = make_node(
        nmx_nodes.Collection,
        value=[{"x": make_value(42)}],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )
    schemed_col = make_node(
        nmx_nodes.SchemedCollection,
        value=inner_col,
        dataframe=as_frame,
        apply_type=nmx_nodes.FrameApplyEnum.PRE,
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    outer_col = make_node(
        nmx_nodes.Collection,
        value=[make_value("some_string", _STRING_TYPE), schemed_col],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    # 3. Execute
    result = interpreter_instance.eval_collection(outer_col)

    # 4. Assert outer collection and nested SchemedCollection Struct
    assert isinstance(result, nmx_runtime.Struct)
    assert result.get(0) == "some_string"

    nested_struct = result.get(1)
    assert isinstance(nested_struct, nmx_runtime.Struct)
    assert nested_struct.get("x") == 42


# =============================================================================
# Tests for _discover_toolsets_and_imports (Fail-Fast & Collision Logic)
# =============================================================================


def test_discover_toolsets_synthesizes_import_for_no_arg_toolset(interpreter_instance):
    """Test that a toolset requiring no args gets an automatic import synthesized."""

    class NoArgTool(Toolset):
        def __init__(self):
            super().__init__()

    Toolset._classes["NoArgTool"] = NoArgTool

    mock_script = MagicMock()
    mock_script.get_location.return_value = "main.nxs"
    mock_decl = make_node(
        nmx_nodes.PythonToolDeclaration, name="NoArgTool", prompt=None, meta=make_meta()
    )

    mock_script.toolsets_decl = [mock_decl]
    mock_script.toolset_imports = {}

    interpreter_instance.interpret_tool_declaration = MagicMock()
    interpreter_instance._discover_toolsets_and_imports(mock_script)

    assert "NoArgTool" in interpreter_instance.context.toolsets


def test_discover_toolsets_fail_fast_on_required_args(interpreter_instance):
    """Test that a toolset requiring explicit args halts execution if not explicitly imported."""

    class RequiredArgTool(Toolset):
        def __init__(self, api_key):
            super().__init__()
            self.api_key = api_key

    Toolset._classes["RequiredArgTool"] = RequiredArgTool

    mock_script = MagicMock()
    mock_script.get_location.return_value = "main.nxs"
    mock_decl = make_node(
        nmx_nodes.PythonToolDeclaration,
        name="RequiredArgTool",
        prompt=None,
        meta=make_meta(),
    )

    mock_script.toolsets_decl = [mock_decl]
    mock_script.toolset_imports = {}

    interpreter_instance.interpret_tool_declaration = MagicMock()

    with pytest.raises(
        nmx_ex.NemantixRuntimeException,
        match=r"Declared toolset 'RequiredArgTool' requires initialization arguments",
    ):
        interpreter_instance._discover_toolsets_and_imports(mock_script)


def test_discover_toolsets_synthesizes_import_with_defaults_and_kwargs(
    interpreter_instance,
):
    """Test that a toolset with defaults or *args/**kwargs is deemed safe and doesn't fail-fast."""

    class ForgivingTool(Toolset):
        def __init__(self, default_val=10, *args, **kwargs):
            super().__init__()

    Toolset._classes["ForgivingTool"] = ForgivingTool

    mock_script = MagicMock()
    mock_script.get_location.return_value = "main.nxs"
    mock_decl = make_node(
        nmx_nodes.PythonToolDeclaration,
        name="ForgivingTool",
        prompt=None,
        meta=make_meta(),
    )

    mock_script.toolsets_decl = [mock_decl]
    mock_script.toolset_imports = {}

    interpreter_instance.interpret_tool_declaration = MagicMock()
    interpreter_instance._discover_toolsets_and_imports(mock_script)

    assert "ForgivingTool" in interpreter_instance.context.toolsets


def test_discover_toolsets_warns_on_global_collision(interpreter_instance):
    """Test that redeclaring an existing global toolset emits a warning."""

    class CollisionTool(Toolset):
        pass

    Toolset._classes["CollisionTool"] = CollisionTool

    mock_script = MagicMock()
    mock_script.get_location.return_value = "main.nxs"
    mock_decl = make_node(
        nmx_nodes.PythonToolDeclaration,
        name="CollisionTool",
        prompt=None,
        meta=make_meta(),
    )

    mock_script.toolsets_decl = [mock_decl]
    mock_script.toolset_imports = {}

    interpreter_instance.interpret_tool_declaration = MagicMock()

    with patch("nemantix.core.interpreter.logger.warning") as mock_warning:
        interpreter_instance._discover_toolsets_and_imports(mock_script)

        warning_emitted = any(
            "Global toolset collision detected: 'CollisionTool'" in call_args[0][0]
            for call_args in mock_warning.call_args_list
        )
        assert warning_emitted is True, "Expected a global collision warning."

    assert "CollisionTool" in interpreter_instance.context.toolsets
    assert (
        interpreter_instance.context.toolsets_locations["CollisionTool"] == "main.nxs"
    )


def test_discover_toolsets_warns_on_same_script_collision(interpreter_instance):
    """Test that declaring the same toolset twice in the same script emits a warning, but proceeds."""

    mock_script = MagicMock()
    mock_script.get_location.return_value = "main.nxs"

    mock_decl_1 = make_node(
        nmx_nodes.PythonToolDeclaration,
        name="RepeatTool",
        prompt=None,
        meta=make_meta(),
    )
    mock_decl_2 = make_node(
        nmx_nodes.PythonToolDeclaration,
        name="RepeatTool",
        prompt=None,
        meta=make_meta(),
    )

    mock_script.toolsets_decl = [mock_decl_1, mock_decl_2]
    mock_script.toolset_imports = {}

    # SIMULATE THE COMPILER: When the interpreter processes the declaration,
    # it normally executes Python code that places the class in Toolset._classes
    def mock_compile_toolset(decl):
        class RepeatToolClass(Toolset):
            pass

        Toolset._classes[decl.name] = RepeatToolClass

    interpreter_instance.interpret_tool_declaration = MagicMock(
        side_effect=mock_compile_toolset
    )

    with patch("nemantix.core.interpreter.logger.warning") as mock_warning:
        interpreter_instance._discover_toolsets_and_imports(mock_script)

        warning_emitted = any(
            "Toolset 'RepeatTool' is declared multiple times in 'main.nxs'"
            in call_args[0][0]
            for call_args in mock_warning.call_args_list
        )
        assert warning_emitted is True, "Expected a same-script collision warning."

    assert "RepeatTool" in interpreter_instance.context.toolsets
    assert interpreter_instance.context.toolsets_locations["RepeatTool"] == "main.nxs"

    # Cleanup global state
    Toolset._classes.pop("RepeatTool", None)


def test_discover_toolsets_raises_on_cross_script_collision(interpreter_instance):
    """Test that declaring a toolset already declared in an imported script raises a hard error."""

    # Simulate that 'shared.nxs' was already processed and added to the context
    interpreter_instance.context.toolsets.add("SharedTool")
    interpreter_instance.context.toolsets_locations["SharedTool"] = "shared.nxs"

    mock_script = MagicMock()
    mock_script.get_location.return_value = "main.nxs"

    mock_decl = make_node(
        nmx_nodes.PythonToolDeclaration,
        name="SharedTool",
        prompt=None,
        meta=make_meta(),
    )

    mock_script.toolsets_decl = [mock_decl]
    mock_script.toolset_imports = {}

    interpreter_instance.interpret_tool_declaration = MagicMock()

    with pytest.raises(
        nmx_ex.NemantixRuntimeException,
        match=r"Cross-script collision: Toolset 'SharedTool' was already declared in 'shared.nxs'",
    ):
        interpreter_instance._discover_toolsets_and_imports(mock_script)


# =============================================================================
# Tests for _parse_do_using (Strict Argument Passing Rules)
# =============================================================================


def test_parse_do_using_naked_variable_no_unpacking(interpreter_instance):
    """
    Rule: A single variable (even holding a Struct) passed without brackets
    is treated as exactly ONE positional argument. No implicit unpacking.
    Syntax: do action using my_struct
    """
    my_struct = nmx_runtime.Struct()
    my_struct.set(key="a", value=1)
    interpreter_instance.context.env.set("my_struct", my_struct)

    do_stmt = make_node(
        nmx_nodes.DoStatement, using=make_var("my_struct"), meta=make_meta()
    )

    args, kwargs = interpreter_instance._extract_args_and_kwargs(do_stmt.using, do_stmt)

    assert len(args) == 1
    assert isinstance(args[0], nmx_runtime.Struct)
    assert args[0].get("a") == 1
    assert len(kwargs) == 0


def test_parse_do_using_list_wrapper_stripped(interpreter_instance):
    """
    Rule: Outer Python lists wrapping the AST node (a parser quirk) must be stripped.
    """
    collection = make_node(
        nmx_nodes.Collection,
        value=[make_value(42)],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    # Notice `using` is a Python list containing the Collection
    do_stmt = make_node(nmx_nodes.DoStatement, using=[collection], meta=make_meta())

    args, kwargs = interpreter_instance._extract_args_and_kwargs(do_stmt.using, do_stmt)
    assert args == [42]
    assert len(kwargs) == 0


def test_parse_do_using_explicit_positional_collection(interpreter_instance):
    """
    Rule: [...] brackets define an explicit argument list.
    Syntax: do action using [1, 2, 3]
    """
    collection = make_node(
        nmx_nodes.Collection,
        value=[make_value(1), make_value(2), make_value(3)],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    do_stmt = make_node(nmx_nodes.DoStatement, using=collection, meta=make_meta())

    args, kwargs = interpreter_instance._extract_args_and_kwargs(do_stmt.using, do_stmt)
    assert args == [1, 2, 3]
    assert len(kwargs) == 0


def test_parse_do_using_explicit_nominal_assignments(interpreter_instance):
    """
    Rule: Assignment nodes inside the collection become kwargs.
    Syntax: do action using [[x] = 1, [y] = 2]
    """
    assign_x = make_node(
        nmx_nodes.Assignment, var=make_var("x"), value=make_value(1), meta=make_meta()
    )
    assign_y = make_node(
        nmx_nodes.Assignment, var=make_var("y"), value=make_value(2), meta=make_meta()
    )

    collection = make_node(
        nmx_nodes.Collection,
        value=[assign_x, assign_y],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    do_stmt = make_node(nmx_nodes.DoStatement, using=collection, meta=make_meta())

    args, kwargs = interpreter_instance._extract_args_and_kwargs(do_stmt.using, do_stmt)
    assert args == []
    assert kwargs == {"x": 1, "y": 2}


def test_parse_do_using_mixed_valid_order(interpreter_instance):
    """
    Rule: Positional arguments before keyword arguments is perfectly valid.
    Syntax: do action using [1, [x] = 2]
    """
    assign_x = make_node(
        nmx_nodes.Assignment, var=make_var("x"), value=make_value(2), meta=make_meta()
    )

    collection = make_node(
        nmx_nodes.Collection,
        value=[make_value(1), assign_x],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    do_stmt = make_node(nmx_nodes.DoStatement, using=collection, meta=make_meta())

    args, kwargs = interpreter_instance._extract_args_and_kwargs(do_stmt.using, do_stmt)
    assert args == [1]
    assert kwargs == {"x": 2}


def test_parse_do_using_mixed_invalid_order_raises(interpreter_instance):
    """
    Rule: Positional arguments following keyword arguments triggers a strict Pythonic error.
    Syntax: do action using [[x] = 1, 2]
    """
    assign_x = make_node(
        nmx_nodes.Assignment, var=make_var("x"), value=make_value(1), meta=make_meta()
    )

    collection = make_node(
        nmx_nodes.Collection,
        value=[assign_x, make_value(2)],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    do_stmt = make_node(nmx_nodes.DoStatement, using=collection, meta=make_meta())

    with pytest.raises(
        nmx_ex.NemantixRuntimeException,
        match=r"Positional argument follows nominal argument",
    ):
        interpreter_instance._extract_args_and_kwargs(do_stmt.using, do_stmt)


def test_parse_do_using_bare_dicts_mapped_to_kwargs(interpreter_instance):
    """
    Rule: Bare Python dictionaries within the collection (representing `a: "a"`)
    are properly mapped to kwargs to prevent AttributeError during AST meta lookup.
    Syntax: do action using [a: "a", b: "b"]
    """
    # The parser emits nominal struct fields as bare dicts
    dict_item = {"a": make_value("val_a"), "b": make_value("val_b")}

    collection = make_node(
        nmx_nodes.Collection,
        value=[dict_item],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    do_stmt = make_node(nmx_nodes.DoStatement, using=collection, meta=make_meta())

    args, kwargs = interpreter_instance._extract_args_and_kwargs(do_stmt.using, do_stmt)
    assert args == []
    assert kwargs == {"a": "val_a", "b": "val_b"}


def test_parse_do_using_bare_dicts_followed_by_positional_raises(interpreter_instance):
    """
    Rule: Even bare dictionary struct fields trigger the strict ordering exception
    if followed by a positional argument.
    Syntax: do action using [a: "a", 2]
    """
    dict_item = {"a": make_value("val_a")}

    collection = make_node(
        nmx_nodes.Collection,
        value=[dict_item, make_value(2)],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    do_stmt = make_node(nmx_nodes.DoStatement, using=collection, meta=make_meta())

    with pytest.raises(
        nmx_ex.NemantixRuntimeException,
        match=r"Positional argument follows nominal argument",
    ):
        interpreter_instance._extract_args_and_kwargs(do_stmt.using, do_stmt)


# =============================================================================
# Tests for Builtin Function Execution in interpret_expression
# =============================================================================


def test_builtin_strict_no_implicit_unpacking(interpreter_instance):
    """
    Rule: A Struct passed via a variable must be passed as exactly ONE
    positional argument, without implicit unboxing.
    AST equivalent:
        [my_struct] = (10, 20)
        size([my_struct])
    """
    mock_fn = MagicMock(return_value="mock_success")

    with patch.dict(
        "nemantix.core.interpreter.BUILTIN_FUNCTIONS",
        {nmx_nodes.BuiltinFunctionEnum.SIZE: mock_fn},
    ):
        # 1. Set the struct in the environment
        my_struct = nmx_runtime.Struct()
        my_struct.set(10)
        my_struct.set(20)
        interpreter_instance.context.env.set("my_struct", my_struct)

        # 2. Pass the variable to the builtin
        builtin_expr = make_node(
            nmx_nodes.BuiltinFunction,
            function=nmx_nodes.BuiltinFunctionEnum.SIZE,
            args=[make_var("my_struct")],
            meta=make_meta(),
        )

        interpreter_instance.interpret_expression(builtin_expr)

        # Verify strict *args passing
        assert mock_fn.call_count == 1
        call_args, call_kwargs = mock_fn.call_args

        # It must receive exactly ONE argument (the intact Struct)
        assert len(call_args) == 1
        assert len(call_kwargs) == 0
        passed_struct = call_args[0]
        assert isinstance(passed_struct, nmx_runtime.Struct)
        assert passed_struct.get(0) == 10
        assert passed_struct.get(1) == 20


def test_builtin_zero_arguments(interpreter_instance):
    """Test that builtins calling with zero arguments (e.g. `size()`) pass safely."""
    mock_fn = MagicMock(return_value="mock_success")

    with patch.dict(
        "nemantix.core.interpreter.BUILTIN_FUNCTIONS",
        {nmx_nodes.BuiltinFunctionEnum.SIZE: mock_fn},
    ):
        builtin_expr = make_node(
            nmx_nodes.BuiltinFunction,
            function=nmx_nodes.BuiltinFunctionEnum.SIZE,
            args=[],
            meta=make_meta(),
        )

        interpreter_instance.interpret_expression(builtin_expr)

        assert mock_fn.call_count == 1
        call_args = mock_fn.call_args[0]
        assert len(call_args) == 0


def test_builtin_multiple_arguments(interpreter_instance):
    """Test that standard multi-argument functions receive all positional args untouched."""
    mock_fn = MagicMock(return_value="mock_success")

    with patch.dict(
        "nemantix.core.interpreter.BUILTIN_FUNCTIONS",
        {nmx_nodes.BuiltinFunctionEnum.SUBSTRING: mock_fn},
    ):
        # Mirror the parser: encapsulate arguments in a Collection
        collection = make_node(
            nmx_nodes.Collection,
            value=[make_value("hello", _STRING_TYPE), make_value(0), make_value(2)],
            inferred_type=VariableTypeEnum.LIST,
            meta=make_meta(),
        )
        builtin_expr = make_node(
            nmx_nodes.BuiltinFunction,
            function=nmx_nodes.BuiltinFunctionEnum.SUBSTRING,
            args=[collection],
            meta=make_meta(),
        )

        interpreter_instance.interpret_expression(builtin_expr)

        assert mock_fn.call_count == 1
        call_args = mock_fn.call_args[0]
        assert len(call_args) == 3
        assert call_args == ("hello", 0, 2)


def test_builtin_llm_prompt_extraction_string(interpreter_instance):
    """
    Test that the LLM telemetry logic correctly extracts the prompt
    string when the builtin is LLM and the first arg is a string.
    """
    with patch.object(interpreter_instance, "_emit_call_enter") as mock_emit:
        builtin_expr = make_node(
            nmx_nodes.BuiltinFunction,
            function=nmx_nodes.BuiltinFunctionEnum.LLM,
            args=[make_value("Summarize this text", _STRING_TYPE)],
            meta=make_meta(),
        )

        # The llm proxy is already dummied out in our fixture to not crash
        interpreter_instance.interpret_expression(builtin_expr)

        mock_emit.assert_called_once()
        _, kwargs = mock_emit.call_args

        # Ensure the string was mapped to the telemetry payload
        assert kwargs["callable_type"] == "builtin"
        assert kwargs["callable_name"] == "llm"
        assert kwargs["callable_prompt"] == "Summarize this text"


def test_builtin_llm_prompt_extraction_non_string_fallback(interpreter_instance):
    """
    Test that if the LLM receives a complex payload (like a Struct) instead of a string,
    the telemetry prompt payload safely falls back to an empty string.
    """
    with patch.object(interpreter_instance, "_emit_call_enter") as mock_emit:
        # Pass a Struct to the LLM
        collection = make_node(
            nmx_nodes.Collection,
            value=[make_value(42)],
            inferred_type=VariableTypeEnum.LIST,
            meta=make_meta(),
        )

        builtin_expr = make_node(
            nmx_nodes.BuiltinFunction,
            function=nmx_nodes.BuiltinFunctionEnum.LLM,
            args=[collection],
            meta=make_meta(),
        )

        # Mock ask_llm so it doesn't crash internally trying to parse a Struct as a string prompt
        with patch(
            "nemantix.core.runtime.Builtin.ask_llm",
            return_value=MagicMock(text="dummy"),
        ):
            interpreter_instance.interpret_expression(builtin_expr)

        mock_emit.assert_called_once()
        _, kwargs = mock_emit.call_args

        # Ensure fallback to empty string triggered
        assert kwargs["callable_prompt"] == ""


def test_builtin_exception_wrapped_in_runtime_exception(interpreter_instance):
    """
    Test that raw Python Exceptions thrown by builtins are securely caught,
    wrapped in a NemantixRuntimeException, and format the error string correctly.
    """

    def crashing_builtin(*args):
        raise ValueError("Critical Math Failure")

    with patch.dict(
        "nemantix.core.interpreter.BUILTIN_FUNCTIONS",
        {nmx_nodes.BuiltinFunctionEnum.SQRT: crashing_builtin},
    ):
        builtin_expr = make_node(
            nmx_nodes.BuiltinFunction,
            function=nmx_nodes.BuiltinFunctionEnum.SQRT,
            args=[make_value(-1)],
            meta=make_meta(),
        )

        with pytest.raises(nmx_ex.NemantixRuntimeException) as exc_info:
            interpreter_instance.interpret_expression(builtin_expr)

        err_msg = str(exc_info.value)
        assert "Critical Math Failure" in err_msg
        assert 'error in builtin function call "SQRT"' in err_msg
        assert "[-1]" in err_msg  # Ensures args are printed in the error


# =============================================================================
# Tests for Builtin TO_STR (via Interpreter Execution Pipeline)
# =============================================================================


def test_interpret_builtin_to_str_scalar(interpreter_instance):
    """
    Test that a scalar passed to the builtin is evaluated and stringified correctly.
    AST equivalent: to_str(True)
    """
    builtin_expr = make_node(
        nmx_nodes.BuiltinFunction,
        function=nmx_nodes.BuiltinFunctionEnum.TO_STR,
        args=[make_value(True)],
        meta=make_meta(),
    )

    result = interpreter_instance.interpret_expression(builtin_expr)
    assert result == "true"


def test_interpret_builtin_to_str_single_struct_positional(interpreter_instance):
    """
    Test that a Struct passed via a variable is NOT implicitly
    unpacked by `function(*args)`. The builtin must receive the intact Struct.
    AST equivalent:
        [my_struct] = (1, 2)
        to_str([my_struct])
    """
    # 1. Set the struct in the environment
    my_struct = nmx_runtime.Struct()
    my_struct.set(1)
    my_struct.set(2)
    interpreter_instance.context.env.set("my_struct", my_struct)

    # 2. Pass the variable to the builtin
    builtin_expr = make_node(
        nmx_nodes.BuiltinFunction,
        function=nmx_nodes.BuiltinFunctionEnum.TO_STR,
        args=[make_var("my_struct")],
        meta=make_meta(),
    )

    result = interpreter_instance.interpret_expression(builtin_expr)
    assert result == "Struct(1, 2)"


def test_interpret_builtin_to_str_single_struct_mixed(interpreter_instance):
    """
    Test that a Struct with mixed fields passed via a variable is securely stringified.
    AST equivalent:
        [my_struct] = (10, name: "Luigi")
        to_str([my_struct])
    """
    my_struct = nmx_runtime.Struct()
    my_struct.set(10)
    my_struct.set("Luigi", key="name")
    interpreter_instance.context.env.set("my_struct", my_struct)

    builtin_expr = make_node(
        nmx_nodes.BuiltinFunction,
        function=nmx_nodes.BuiltinFunctionEnum.TO_STR,
        args=[make_var("my_struct")],
        meta=make_meta(),
    )

    result = interpreter_instance.interpret_expression(builtin_expr)
    assert result == "Struct(10, name: Luigi)"


def test_interpret_builtin_multiple_args_drops_extras_safely(interpreter_instance):
    """
    Because we removed the `_` hack, if a user maliciously calls to_str with
    multiple arguments, `x` captures the first, and `*_, **__` absorbs the rest.
    AST equivalent: to_str(1, 2, 3)
    """
    builtin_expr = make_node(
        nmx_nodes.BuiltinFunction,
        function=nmx_nodes.BuiltinFunctionEnum.TO_STR,
        args=[make_value(1), make_value(2), make_value(3)],
        meta=make_meta(),
    )

    # The new to_str(x=None, *_, **__) assigns x=1. The 2 and 3 are absorbed and ignored.
    result = interpreter_instance.interpret_expression(builtin_expr)
    assert result == "1"


# =============================================================================
# Tests for Builtin Functions (via Interpreter Execution Pipeline)
# =============================================================================


def _eval_builtin(interpreter, func_enum, args_ast):
    """
    Helper to cleanly build and evaluate a builtin function AST node.
    Mirrors the real parser by wrapping the argument list in a Collection.
    """
    collection = make_node(
        nmx_nodes.Collection,
        value=args_ast,
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )

    builtin_expr = make_node(
        nmx_nodes.BuiltinFunction,
        function=func_enum,
        args=[collection],
        meta=make_meta(),
    )
    return interpreter.interpret_expression(builtin_expr)


def test_interpret_builtin_coalesce(interpreter_instance):
    """Test coalesce returns the first non-null argument."""
    res = _eval_builtin(
        interpreter_instance,
        nmx_nodes.BuiltinFunctionEnum.COALESCE,
        [make_value(None), make_value(None), make_value(42), make_value(99)],
    )
    assert res == 42


def test_interpret_builtin_exists(interpreter_instance):
    """Test exists correctly identifies null vs non-null."""
    assert (
        _eval_builtin(
            interpreter_instance,
            nmx_nodes.BuiltinFunctionEnum.EXISTS,
            [make_value(None)],
        )
        is False
    )
    assert (
        _eval_builtin(
            interpreter_instance, nmx_nodes.BuiltinFunctionEnum.EXISTS, [make_value(0)]
        )
        is True
    )

    # Intact Struct evaluation via variable
    my_struct = nmx_runtime.Struct()
    interpreter_instance.context.env.set("empty_struct", my_struct)

    assert (
        _eval_builtin(
            interpreter_instance,
            nmx_nodes.BuiltinFunctionEnum.EXISTS,
            [make_var("empty_struct")],
        )
        is True
    )


def test_interpret_builtin_type(interpreter_instance):
    """Test type builtin cleanly identifies standard and complex types."""
    assert (
        _eval_builtin(
            interpreter_instance, nmx_nodes.BuiltinFunctionEnum.TYPE, [make_value(None)]
        )
        == "none"
    )
    assert (
        _eval_builtin(
            interpreter_instance, nmx_nodes.BuiltinFunctionEnum.TYPE, [make_value(42.5)]
        )
        == "num"
    )
    assert (
        _eval_builtin(
            interpreter_instance,
            nmx_nodes.BuiltinFunctionEnum.TYPE,
            [make_value("hello", _STRING_TYPE)],
        )
        == "str"
    )

    # Intact Struct evaluation via variable
    my_struct = nmx_runtime.Struct()
    my_struct.set(1)
    my_struct.set(2)
    interpreter_instance.context.env.set("my_struct", my_struct)

    assert (
        _eval_builtin(
            interpreter_instance,
            nmx_nodes.BuiltinFunctionEnum.TYPE,
            [make_var("my_struct")],
        )
        == "struct"
    )


def test_interpret_builtin_size(interpreter_instance):
    """Test size builtin with single scalar, single struct, and multiple arguments."""
    # Single String
    assert (
        _eval_builtin(
            interpreter_instance,
            nmx_nodes.BuiltinFunctionEnum.SIZE,
            [make_value("abc", _STRING_TYPE)],
        )
        == 3
    )

    # Single Struct via variable
    my_struct = nmx_runtime.Struct()
    my_struct.set(1)
    my_struct.set(2)
    interpreter_instance.context.env.set("my_struct", my_struct)

    assert (
        _eval_builtin(
            interpreter_instance,
            nmx_nodes.BuiltinFunctionEnum.SIZE,
            [make_var("my_struct")],
        )
        == 2
    )

    # Multiple Positional Arguments (Passed as a Collection representing the argument list)
    assert (
        _eval_builtin(
            interpreter_instance,
            nmx_nodes.BuiltinFunctionEnum.SIZE,
            [make_value(10), make_value(20), make_value(30)],
        )
        == 3
    )


def test_interpret_builtin_substring(interpreter_instance):
    """Test substring handles multiple positional arguments securely."""
    res = _eval_builtin(
        interpreter_instance,
        nmx_nodes.BuiltinFunctionEnum.SUBSTRING,
        [make_value("nemantix", _STRING_TYPE), make_value(3), make_value(8)],
    )
    assert res == "antix"


def test_interpret_builtin_to_num_and_to_bool(interpreter_instance):
    """Test explicit cast builtins."""
    # to_num
    assert (
        _eval_builtin(
            interpreter_instance,
            nmx_nodes.BuiltinFunctionEnum.TO_NUM,
            [make_value("42.5", _STRING_TYPE)],
        )
        == 42.5
    )
    assert (
        _eval_builtin(
            interpreter_instance,
            nmx_nodes.BuiltinFunctionEnum.TO_NUM,
            [make_value(True)],
        )
        == 1
    )

    # to_bool
    assert (
        _eval_builtin(
            interpreter_instance, nmx_nodes.BuiltinFunctionEnum.TO_BOOL, [make_value(1)]
        )
        is True
    )
    assert (
        _eval_builtin(
            interpreter_instance,
            nmx_nodes.BuiltinFunctionEnum.TO_BOOL,
            [make_value("false", _STRING_TYPE)],
        )
        is False
    )


def test_interpret_builtin_soft_conversions(interpreter_instance):
    """Test implicit (soft) cast builtins, focusing on their Struct handling."""
    my_struct = nmx_runtime.Struct()
    my_struct.set(1)
    interpreter_instance.context.env.set("my_struct", my_struct)

    # num() safely fails on a Struct and returns None
    assert (
        _eval_builtin(
            interpreter_instance,
            nmx_nodes.BuiltinFunctionEnum.NUM,
            [make_var("my_struct")],
        )
        is None
    )

    # bool() on a Struct evaluates its length
    assert (
        _eval_builtin(
            interpreter_instance,
            nmx_nodes.BuiltinFunctionEnum.BOOL,
            [make_var("my_struct")],
        )
        is True
    )

    empty_struct = nmx_runtime.Struct()
    interpreter_instance.context.env.set("empty_struct", empty_struct)

    assert (
        _eval_builtin(
            interpreter_instance,
            nmx_nodes.BuiltinFunctionEnum.BOOL,
            [make_var("empty_struct")],
        )
        is False
    )


def test_interpret_builtin_math(interpreter_instance):
    """Test math builtins pass parameters through correctly."""
    assert (
        _eval_builtin(
            interpreter_instance, nmx_nodes.BuiltinFunctionEnum.SQRT, [make_value(4)]
        )
        == 2.0
    )
    assert (
        _eval_builtin(
            interpreter_instance, nmx_nodes.BuiltinFunctionEnum.SIN, [make_value(0)]
        )
        == 0.0
    )
    assert (
        _eval_builtin(
            interpreter_instance, nmx_nodes.BuiltinFunctionEnum.COS, [make_value(0)]
        )
        == 1.0
    )


# =============================================================================
# Tests for Unified _extract_args_and_kwargs
# =============================================================================


def test_extract_args_and_kwargs_positional_only(interpreter_instance):
    """Test extracting purely positional arguments."""
    collection = make_node(
        nmx_nodes.Collection,
        value=[make_value(10), make_value(20)],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )
    args, kwargs = interpreter_instance._extract_args_and_kwargs(
        [collection], statement=None
    )
    assert args == [10, 20]
    assert kwargs == {}


def test_extract_args_and_kwargs_nominal_only(interpreter_instance):
    """Test extracting purely nominal arguments (both Assignment and bare dicts)."""
    assign_node = make_node(
        nmx_nodes.Assignment, var=make_var("x"), value=make_value(1), meta=make_meta()
    )
    dict_node = {"y": make_value(2), "z": make_value(3)}

    collection = make_node(
        nmx_nodes.Collection,
        value=[assign_node, dict_node],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )
    args, kwargs = interpreter_instance._extract_args_and_kwargs(
        [collection], statement=None
    )
    assert args == []
    assert kwargs == {"x": 1, "y": 2, "z": 3}


def test_extract_args_and_kwargs_mixed_valid_order(interpreter_instance):
    """Test extracting mixed arguments in the correct order (positional first)."""
    assign_node = make_node(
        nmx_nodes.Assignment, var=make_var("x"), value=make_value(99), meta=make_meta()
    )

    collection = make_node(
        nmx_nodes.Collection,
        value=[make_value(42), assign_node],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )
    args, kwargs = interpreter_instance._extract_args_and_kwargs(
        [collection], statement=None
    )
    assert args == [42]
    assert kwargs == {"x": 99}


def test_extract_args_and_kwargs_mixed_invalid_order_raises(interpreter_instance):
    """Test that a positional argument following a keyword argument raises a strict error."""
    assign_node = make_node(
        nmx_nodes.Assignment, var=make_var("x"), value=make_value(99), meta=make_meta()
    )

    collection = make_node(
        nmx_nodes.Collection,
        value=[assign_node, make_value(42)],
        inferred_type=VariableTypeEnum.LIST,
        meta=make_meta(),
    )
    with pytest.raises(
        nmx_ex.NemantixRuntimeException,
        match=r"Positional argument follows nominal argument",
    ):
        interpreter_instance._extract_args_and_kwargs([collection], statement=None)


# =============================================================================
# Tests for Builtin Execution with kwargs
# =============================================================================


def test_interpret_builtin_with_kwargs(interpreter_instance):
    """
    Test that inline builtins properly receive kwargs extracted from their args list.
    AST equivalent: substring("nemantix", end=5)
    """
    mock_fn = MagicMock(return_value="mock_success")

    with patch.dict(
        "nemantix.core.interpreter.BUILTIN_FUNCTIONS",
        {nmx_nodes.BuiltinFunctionEnum.SUBSTRING: mock_fn},
    ):
        assign_end = make_node(
            nmx_nodes.Assignment,
            var=make_var("end"),
            value=make_value(5),
            meta=make_meta(),
        )
        collection = make_node(
            nmx_nodes.Collection,
            value=[make_value("nemantix", _STRING_TYPE), assign_end],
            inferred_type=VariableTypeEnum.LIST,
            meta=make_meta(),
        )

        builtin_expr = make_node(
            nmx_nodes.BuiltinFunction,
            function=nmx_nodes.BuiltinFunctionEnum.SUBSTRING,
            args=[collection],
            meta=make_meta(),
        )

        interpreter_instance.interpret_expression(builtin_expr)
        mock_fn.assert_called_once_with("nemantix", end=5)


def test_interpret_builtin_single_struct_no_implicit_unpacking(interpreter_instance):
    """
    Test that resolving a variable to a Struct securely passes it to a standard
    multi-argument builtin without unpacking it.
    AST equivalent:
        [my_struct] = ("nemantix", end: 5)
        substring([my_struct])
    """
    mock_fn = MagicMock(return_value="mock_success")

    with patch.dict(
        "nemantix.core.interpreter.BUILTIN_FUNCTIONS",
        {nmx_nodes.BuiltinFunctionEnum.SUBSTRING: mock_fn},
    ):
        my_struct = nmx_runtime.Struct()
        my_struct.set("nemantix")
        my_struct.set(5, key="end")
        interpreter_instance.context.env.set("my_struct", my_struct)

        builtin_expr = make_node(
            nmx_nodes.BuiltinFunction,
            function=nmx_nodes.BuiltinFunctionEnum.SUBSTRING,
            args=[make_var("my_struct")],  # Passed as ONE single argument
            meta=make_meta(),
        )

        interpreter_instance.interpret_expression(builtin_expr)

        assert mock_fn.call_count == 1
        call_args, call_kwargs = mock_fn.call_args

        assert len(call_args) == 1
        assert len(call_kwargs) == 0

        passed_struct = call_args[0]
        assert isinstance(passed_struct, nmx_runtime.Struct)
        assert passed_struct.get(0) == "nemantix"
        assert passed_struct.get("end") == 5


def test_interpret_builtin_invalid_argument_order_raises(interpreter_instance):
    """
    Test that an invalid inline builtin call halts execution natively via the extractor.
    AST equivalent: substring(start=3, "nemantix")
    """
    mock_fn = MagicMock(return_value="mock_success")

    with patch.dict(
        "nemantix.core.interpreter.BUILTIN_FUNCTIONS",
        {nmx_nodes.BuiltinFunctionEnum.SUBSTRING: mock_fn},
    ):
        assign_start = make_node(
            nmx_nodes.Assignment,
            var=make_var("start"),
            value=make_value(3),
            meta=make_meta(),
        )
        collection = make_node(
            nmx_nodes.Collection,
            value=[assign_start, make_value("nemantix", _STRING_TYPE)],
            inferred_type=VariableTypeEnum.LIST,
            meta=make_meta(),
        )

        builtin_expr = make_node(
            nmx_nodes.BuiltinFunction,
            function=nmx_nodes.BuiltinFunctionEnum.SUBSTRING,
            args=[collection],
            meta=make_meta(),
        )

        with pytest.raises(
            nmx_ex.NemantixRuntimeException,
            match=r"Positional argument follows nominal argument",
        ):
            interpreter_instance.interpret_expression(builtin_expr)


def test_interpret_builtin_type_and_size_with_struct_vars(interpreter_instance):
    """Test `type` and `size` builtins when evaluating a Struct variable."""
    my_struct = nmx_runtime.Struct()
    my_struct.set(1)
    my_struct.set(2)
    interpreter_instance.context.env.set("my_struct", my_struct)

    # type([my_struct])
    type_expr = make_node(
        nmx_nodes.BuiltinFunction,
        function=nmx_nodes.BuiltinFunctionEnum.TYPE,
        args=[make_var("my_struct")],
        meta=make_meta(),
    )
    assert interpreter_instance.interpret_expression(type_expr) == "struct"

    # size([my_struct])
    size_expr = make_node(
        nmx_nodes.BuiltinFunction,
        function=nmx_nodes.BuiltinFunctionEnum.SIZE,
        args=[make_var("my_struct")],
        meta=make_meta(),
    )
    assert interpreter_instance.interpret_expression(size_expr) == 2


def test_interpret_builtin_soft_conversions_with_struct_vars(interpreter_instance):
    """Test implicit (soft) cast builtins against Struct variables."""
    my_struct = nmx_runtime.Struct()
    my_struct.set(1)
    interpreter_instance.context.env.set("my_struct", my_struct)

    # num() safely fails on a Struct and returns None
    num_expr = make_node(
        nmx_nodes.BuiltinFunction,
        function=nmx_nodes.BuiltinFunctionEnum.NUM,
        args=[make_var("my_struct")],
        meta=make_meta(),
    )
    assert interpreter_instance.interpret_expression(num_expr) is None

    # bool() on a Struct evaluates its length
    bool_expr = make_node(
        nmx_nodes.BuiltinFunction,
        function=nmx_nodes.BuiltinFunctionEnum.BOOL,
        args=[make_var("my_struct")],
        meta=make_meta(),
    )
    assert interpreter_instance.interpret_expression(bool_expr) is True


# =============================================================================
# Tests for update_context
# =============================================================================


def test_update_context_with_script(interpreter_instance):
    """
    Test that updating the context with a Script registers the script location
    and discovers its global actions.
    """
    # 1. Setup mock script with a global action
    mock_script = MagicMock()
    mock_script.get_location.return_value = "dynamic_script.nxs"

    dummy_action = DummyAction(
        name="dynamic_global_action", input=[], output=[], children=[], meta=make_meta()
    )
    mock_script.actions = {"dynamic_global_action": dummy_action}

    # 2. Call update_context
    interpreter_instance.update_context(script=mock_script)

    # 3. Assertions
    assert "dynamic_global_action" in interpreter_instance.context.actions
    assert (
        interpreter_instance.context.actions["dynamic_global_action"]["is_global"]
        is True
    )
    assert (
        interpreter_instance.context.actions["dynamic_global_action"]["action"]
        == dummy_action
    )


def test_update_context_with_deliberate(interpreter_instance):
    """
    Test that updating the context with a Deliberate registers the deliberate name
    and discovers its private (generated) actions.
    """
    # 1. Setup deliberate with a private action
    dummy_action = DummyAction(
        name="dynamic_private_action",
        input=[],
        output=[],
        children=[],
        meta=make_meta(),
    )

    mock_deliberate = make_node(
        nmx_nodes.Deliberate,
        name="dynamic_deliberate",
        generated_actions=[dummy_action],
        meta=make_meta(),
    )

    # 2. Call update_context
    interpreter_instance.update_context(deliberate=mock_deliberate)

    # 3. Assertions
    assert "dynamic_deliberate" in interpreter_instance.context._seen_deliberates

    # Private actions are registered under their short name and fully qualified name
    assert "dynamic_private_action" in interpreter_instance.context.actions
    assert (
        "dynamic_deliberate.dynamic_private_action"
        in interpreter_instance.context.actions
    )

    assert (
        interpreter_instance.context.actions["dynamic_private_action"]["is_global"]
        is False
    )
    assert (
        "dynamic_deliberate"
        in interpreter_instance.context.actions["dynamic_private_action"]["imported_by"]
    )


def test_update_context_with_both_script_and_deliberate(interpreter_instance):
    """
    Test that update_context correctly handles being passed both a Script and a Deliberate simultaneously.
    """
    mock_script = MagicMock()
    mock_script.get_location.return_value = "combo_script.nxs"
    mock_script.actions = {}

    mock_deliberate = make_node(
        nmx_nodes.Deliberate,
        name="combo_deliberate",
        generated_actions=[],
        meta=make_meta(),
    )

    interpreter_instance.update_context(script=mock_script, deliberate=mock_deliberate)

    assert "combo_deliberate" in interpreter_instance.context._seen_deliberates


def test_update_context_none_args(interpreter_instance):
    """
    Test that calling update_context with no arguments safely returns without modifying state.
    """
    initial_script_count = len(interpreter_instance.context._seen_scripts)
    initial_delib_count = len(interpreter_instance.context._seen_deliberates)

    interpreter_instance.update_context()

    assert len(interpreter_instance.context._seen_scripts) == initial_script_count
    assert len(interpreter_instance.context._seen_deliberates) == initial_delib_count


def test_update_context_forces_overwrite(interpreter_instance):
    """
    Test that update_context forces an overwrite of existing actions because it
    passes `should_update=True` to the discovery methods.
    """
    # 1. Pre-populate the context with a stale action
    interpreter_instance.context.actions["updatable_action"] = {
        "closure": None,
        "is_global": False,
        "action": "stale_reference",
    }

    # 2. Setup a script with a fresh action carrying the exact same name
    mock_script = MagicMock()
    mock_script.get_location.return_value = "refresh.nxs"

    fresh_action = DummyAction(
        name="updatable_action", input=[], output=[], children=[], meta=make_meta()
    )
    mock_script.actions = {"updatable_action": fresh_action}

    # 3. Update the context
    interpreter_instance.update_context(script=mock_script)

    # 4. Assert the action was overwritten with the fresh attributes
    updated_action_info = interpreter_instance.context.actions["updatable_action"]
    assert updated_action_info["is_global"] is True
    assert updated_action_info["action"] == fresh_action
    assert updated_action_info["closure"] is not None


# =============================================================================
# Tests for _build_context (Caching & Discovery)
# =============================================================================


def test_build_context_fresh_script_and_deliberate(interpreter_instance):
    """
    Test that a completely unseen script and deliberate trigger all the discovery methods.
    """
    mock_script = MagicMock(spec=Script)
    mock_script.get_location.return_value = "main.nxs"

    mock_deliberate = make_node(
        nmx_nodes.Deliberate, name="fresh_deliberate", meta=make_meta()
    )

    # Mock the discovery methods to verify they are called
    interpreter_instance._discover_actions = MagicMock()
    interpreter_instance._discover_frames = MagicMock()
    interpreter_instance._discover_toolsets_and_imports = MagicMock()
    interpreter_instance._discover_deliberate_actions = MagicMock()

    # Empty requires_map to isolate the test to just the main script
    interpreter_instance.expertise.requires_map = {"main.nxs": []}

    interpreter_instance._build_context(mock_script, mock_deliberate)

    # 1. Assert global scope tracking was updated
    assert interpreter_instance._get_global_script() == mock_script
    assert interpreter_instance._get_global_deliberate() == mock_deliberate

    # 2. Assert discovery methods were called exactly once for the script/deliberate
    interpreter_instance._discover_actions.assert_called_once_with(mock_script)
    interpreter_instance._discover_frames.assert_called_once_with(mock_script)
    interpreter_instance._discover_toolsets_and_imports.assert_called_once_with(
        mock_script
    )
    interpreter_instance._discover_deliberate_actions.assert_called_once_with(
        mock_deliberate
    )

    # 3. Assert they are now tracked in the context
    assert "main.nxs" in interpreter_instance.context._seen_scripts
    assert "fresh_deliberate" in interpreter_instance.context._seen_deliberates


def test_build_context_already_seen_skips_discovery(interpreter_instance):
    """
    Test that if a script and deliberate are already in the context,
    the interpreter skips the expensive AST discovery phase entirely.
    """
    mock_script = MagicMock(spec=Script)
    mock_script.get_location.return_value = "cached.nxs"

    mock_deliberate = make_node(
        nmx_nodes.Deliberate, name="cached_deliberate", meta=make_meta()
    )

    # Pre-seed the context tracking sets
    interpreter_instance.context.add_script(mock_script)
    interpreter_instance.context.add_deliberate(mock_deliberate)

    # Mock the discovery methods
    interpreter_instance._discover_actions = MagicMock()
    interpreter_instance._discover_frames = MagicMock()
    interpreter_instance._discover_toolsets_and_imports = MagicMock()
    interpreter_instance._discover_deliberate_actions = MagicMock()

    interpreter_instance.expertise.requires_map = {"cached.nxs": []}

    # Execute
    interpreter_instance._build_context(mock_script, mock_deliberate)

    # Assert that NO discovery methods were called
    interpreter_instance._discover_actions.assert_not_called()
    interpreter_instance._discover_frames.assert_not_called()
    interpreter_instance._discover_toolsets_and_imports.assert_not_called()
    interpreter_instance._discover_deliberate_actions.assert_not_called()

    # The globals should still be updated for execution context though
    assert interpreter_instance._get_global_script() == mock_script
    assert interpreter_instance._get_global_deliberate() == mock_deliberate


def test_build_context_with_required_scripts_partial_cache(interpreter_instance):
    """
    Test that _build_context properly resolves the `requires_map`, skipping scripts
    it has already seen, but discovering the ones it hasn't.
    """
    mock_main = MagicMock(spec=Script)
    mock_main.get_location.return_value = "main.nxs"

    mock_req1 = MagicMock(spec=Script)
    mock_req1.get_location.return_value = "lib_cached.nxs"

    mock_req2 = MagicMock(spec=Script)
    mock_req2.get_location.return_value = "lib_fresh.nxs"

    mock_deliberate = make_node(
        nmx_nodes.Deliberate,
        name="main_deliberate",
        generated_actions=[],
        meta=make_meta(),
    )

    # Setup expertise script routing and requirements
    interpreter_instance.expertise.script_by_loc = {
        "main.nxs": mock_main,
        "lib_cached.nxs": mock_req1,
        "lib_fresh.nxs": mock_req2,
    }
    interpreter_instance.expertise.requires_map = {
        "main.nxs": ["lib_cached.nxs", "lib_fresh.nxs"]
    }

    # Mark `lib_cached.nxs` as already seen
    interpreter_instance.context.add_script(mock_req1)

    # Mock discovery
    interpreter_instance._discover_actions = MagicMock()
    interpreter_instance._discover_frames = MagicMock()
    interpreter_instance._discover_toolsets_and_imports = MagicMock()

    # Execute
    interpreter_instance._build_context(mock_main, mock_deliberate)

    # Extract all the arguments passed to _discover_actions
    action_calls = interpreter_instance._discover_actions.call_args_list

    # Assert `lib_fresh` (required script, called with kwargs)
    # and `main` (main script, called positionally) were discovered
    assert call(script=mock_req2) in action_calls
    assert call(mock_main) in action_calls

    # Assert `lib_cached` was SKIPPED
    assert call(script=mock_req1) not in action_calls

    # Ensure all scripts ended up in the seen list
    assert "main.nxs" in interpreter_instance.context._seen_scripts
    assert "lib_fresh.nxs" in interpreter_instance.context._seen_scripts
    assert "lib_cached.nxs" in interpreter_instance.context._seen_scripts


# =============================================================================
# access_expr — dynamic struct field access via [struct:([field])]
# =============================================================================


def test_unbox_variable_access_expr_named_field(interpreter_instance):
    """
    Dynamic field access: [struct:([field])] where field = "end".
    AST equivalent:
        [struct] = ("ciao", end: 4)
        [field]  = "end"
        result   = [struct:([field])]   → should be 4
    """
    struct = nmx_runtime.Struct()
    struct.set("ciao")
    struct.set(4, key="end")
    interpreter_instance.context.env.set("struct", struct)
    interpreter_instance.context.env.set("field", "end")

    # Build [struct:([field])] — a Variable whose path contains another Variable
    field_var = make_var("field")
    struct_var = make_var("struct", path=[field_var])

    result = interpreter_instance.unbox_value(struct_var)
    assert result == 4


def test_unbox_variable_access_expr_index_field(interpreter_instance):
    """
    Dynamic index access: [struct:([idx])] where idx = 0.
    """
    struct = nmx_runtime.Struct()
    struct.set("ciao")
    struct.set(4, key="end")
    interpreter_instance.context.env.set("struct", struct)
    interpreter_instance.context.env.set("idx", 0)

    idx_var = make_var("idx")
    struct_var = make_var("struct", path=[idx_var])

    result = interpreter_instance.unbox_value(struct_var)
    assert result == "ciao"
