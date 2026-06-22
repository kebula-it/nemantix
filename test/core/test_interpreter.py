from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import BaseModel

from nemantix.core import exceptions as nmx_ex
from nemantix.core import node as nmx_nodes
from nemantix.core import runtime as nmx_runtime
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

HERE = Path(__file__).parent


# =============================================================================
# Minimal Expertise stub (real object, no MagicMock)
# =============================================================================


class DummyExpertise:
    def __init__(self):
        self.script_by_loc = {}
        self.event_hub = None
        self.deliberate_to_script_loc = {}


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
    dummy_delib = make_node(nmx_nodes.Deliberate, name="dummy_delib", meta=make_meta())
    interpreter_instance._set_global_deliberate(dummy_delib)

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

    my_delib = make_node(nmx_nodes.Deliberate, name="my_delib", meta=make_meta())
    interpreter_instance._set_global_deliberate(my_delib)

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

    # 2. Setup a tool that returns a raw list
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
