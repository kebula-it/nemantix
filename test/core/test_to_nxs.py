from pathlib import Path

import pytest

from nemantix.core import node
from nemantix.core.parser import ParserLark

SCRIPTS_DIR = Path(__file__).parent / "test_scripts"
EXAMPLE_FILES = sorted(SCRIPTS_DIR.glob("*.nxs"))


def _meta():
    return {"file_meta": None, "node_meta": None}


def round_trip(source: str) -> str:
    """Parse source, serialize back to NXS, re-parse — returns re-serialized form."""
    stmts = ParserLark().parse(source, "<test>")
    nxs = "\n".join(stmt.to_nxs() for stmt in stmts)
    # must not raise
    ParserLark().parse(nxs, "<round-trip>")
    return nxs


def _parse_toplevel(nxs: str) -> None:
    ParserLark().parse(nxs, "<test-toplevel>")


def _parse_as_body(nxs: str) -> None:
    src = f"action _T >> t <<:\nbody:\n{nxs}\n__body\n__action"
    ParserLark().parse(src, "<test-body>")


def _parse_as_in(nxs: str) -> None:
    src = f"action _T >> t <<:\nin:\n{nxs}\n__in\nbody:\n__body\n__action"
    ParserLark().parse(src, "<test-in>")


def _parse_as_out(nxs: str) -> None:
    src = f"action _T >> t <<:\nout:\n{nxs}\n__out\nbody:\n__body\n__action"
    ParserLark().parse(src, "<test-out>")


def _parse_as_slot(nxs: str) -> None:
    src = f"frame _F:\n{nxs}\n__frame"
    ParserLark().parse(src, "<test-slot>")


def _parse_as_plan(nxs: str) -> None:
    src = f"deliberate _D when >> w <<:\n{nxs}\n__deliberate"
    ParserLark().parse(src, "<test-plan>")


# =============================================================================
# Step 1 — MicroPrompt
# =============================================================================


def test_microprompt_single_line():
    mp = node.MicroPrompt(prompt="a prompt", meta=_meta())
    assert mp.to_nxs() == ">> a prompt <<"


def test_microprompt_multiline():
    mp = node.MicroPrompt(prompt="multi\nline", meta=_meta())
    assert mp.to_nxs() == ">>> multi\nline <<<"


# =============================================================================
# Step 2 — Collection and SchemedCollection
# =============================================================================


def test_schemed_collection_prefix():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    sc = node.SchemedCollection(
        value=[x],
        inferred_type=node.VariableTypeEnum.LIST,
        dataframe="PersonFrame",
        apply_type=node.FrameApplyEnum.PRE,
        meta=_meta(),
    )
    result = sc.to_nxs()
    assert result.startswith("{PersonFrame}") and "[x]" in result


def test_schemed_collection_suffix():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    sc = node.SchemedCollection(
        value=[x],
        inferred_type=node.VariableTypeEnum.LIST,
        dataframe="PersonFrame",
        apply_type=node.FrameApplyEnum.POST,
        meta=_meta(),
    )
    result = sc.to_nxs()
    assert result.endswith("{PersonFrame}") and "[x]" in result


def test_schemed_collection_variable_operand_suffix():
    x = node.Variable(name="my_struct", prompt=None, path=None, meta=_meta())
    sc = node.SchemedCollection(
        value=x,  # operand is a bare variable, not a collection literal
        inferred_type=node.VariableTypeEnum.LIST,
        dataframe="Person",
        apply_type=node.FrameApplyEnum.POST,
        meta=_meta(),
    )
    # variable operand: no spurious wrapping parens
    assert sc.to_nxs() == "[my_struct] {Person}"


def test_schemed_collection_variable_operand_prefix():
    x = node.Variable(name="my_struct", prompt=None, path=None, meta=_meta())
    sc = node.SchemedCollection(
        value=x,
        inferred_type=node.VariableTypeEnum.LIST,
        dataframe="Person",
        apply_type=node.FrameApplyEnum.PRE,
        meta=_meta(),
    )
    assert sc.to_nxs() == "{Person} [my_struct]"


def test_collection_list():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    y = node.Variable(name="y", prompt=None, path=None, meta=_meta())
    col = node.Collection(
        value=[x, y], inferred_type=node.VariableTypeEnum.LIST, meta=_meta()
    )
    result = col.to_nxs()
    assert "[x]" in result and "[y]" in result


def test_collection_nested():
    inner = node.Collection(
        value=[
            node.SingleValue(
                value=1, inferred_type=node.VariableTypeEnum.INT, meta=_meta()
            )
        ],
        inferred_type=node.VariableTypeEnum.LIST,
        meta=_meta(),
    )
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    outer = node.Collection(
        value=[x, inner], inferred_type=node.VariableTypeEnum.LIST, meta=_meta()
    )
    result = outer.to_nxs()
    assert "[x]" in result and "1" in result


def test_collection_dict():
    val = node.SingleValue(
        value=1, inferred_type=node.VariableTypeEnum.INT, meta=_meta()
    )
    col = node.Collection(
        value={"count": val}, inferred_type=node.VariableTypeEnum.DICT, meta=_meta()
    )
    result = col.to_nxs()
    assert "count" in result and "1" in result
    _parse_as_body(f"[ [z] = {result} ]")


# =============================================================================
# Step 3 — SimilarityOperation
# =============================================================================


def test_similarity_operation_simple():
    a = node.Variable(name="a", prompt=None, path=None, meta=_meta())
    b = node.Variable(name="b", prompt=None, path=None, meta=_meta())
    # silence PyCharm false positive
    # noinspection PyTypeChecker
    op = node.SimilarityOperation(
        operation=node.SimilarityEnum.SIM,
        qualifier=None,
        first=a,
        second=b,
        meta=_meta(),
    )
    result = op.to_nxs()
    assert "[a]" in result and "[b]" in result and "~" in result


def test_similarity_operation_with_qualifier():
    a = node.Variable(name="a", prompt=None, path=None, meta=_meta())
    b = node.Variable(name="b", prompt=None, path=None, meta=_meta())
    op = node.SimilarityOperation(
        operation=node.SimilarityEnum.SIM_QUAL,
        qualifier=(node.SimilarityQualifierEnum.CLOSE, None),
        first=a,
        second=b,
        meta=_meta(),
    )
    result = op.to_nxs()
    assert "close" in result


def test_similarity_operation_number_qualifier():
    a = node.Variable(name="a", prompt=None, path=None, meta=_meta())
    b = node.Variable(name="b", prompt=None, path=None, meta=_meta())
    threshold = node.SingleValue(
        value=0.82, inferred_type=node.VariableTypeEnum.FLOAT, meta=_meta()
    )
    op = node.SimilarityOperation(
        operation=node.SimilarityEnum.SIM_QUAL,
        qualifier=(node.SimilarityQualifierEnum.NUMBER, threshold),
        first=a,
        second=b,
        meta=_meta(),
    )
    result = op.to_nxs()
    assert "0.82" in result
    _parse_as_body(f"[ {result} ]")


# =============================================================================
# Step 4 — MetaExpression
# =============================================================================


def test_meta_expression():
    me = node.MetaExpression(quals=["some_var", "intent", "completion"], meta=_meta())
    assert me.to_nxs() == "{{some_var@intent.completion}}"


# =============================================================================
# Step 5 — SingleValue, Return, Break, Continue
# =============================================================================


def test_break():
    result = node.Break(meta=_meta()).to_nxs()
    assert result == "break"
    _parse_as_body(result)


def test_continue():
    result = node.Continue(meta=_meta()).to_nxs()
    assert result == "continue"
    _parse_as_body(result)


def test_return_no_value():
    result = node.Return(val=[], meta=_meta()).to_nxs()
    assert result == "return"
    _parse_as_body(result)


def test_return_with_values():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    y = node.Variable(name="y", prompt=None, path=None, meta=_meta())
    result = node.Return(val=[x, y], meta=_meta()).to_nxs()
    assert result == "return [x], [y]"
    _parse_as_body(result)


@pytest.mark.parametrize(
    "value, inferred_type, expected",
    [
        (True, node.VariableTypeEnum.BOOL, "true"),
        (False, node.VariableTypeEnum.BOOL, "false"),
        (3.14, node.VariableTypeEnum.FLOAT, "3.14"),
        (None, node.VariableTypeEnum.NONE, "none"),
    ],
    ids=["bool_true", "bool_false", "float", "none"],
)
def test_single_value_types(value, inferred_type, expected):
    sv = node.SingleValue(value=value, inferred_type=inferred_type, meta=_meta())
    result = sv.to_nxs()
    assert result == expected
    _parse_as_body(f"[ [x] = {result} ]")


# =============================================================================
# Step 6 — PythonToolDeclaration
# =============================================================================


def test_python_tool_declaration():
    prompt = node.MicroPrompt(prompt="do something", meta=_meta())
    decl = node.PythonToolDeclaration(name="MyTool", prompt=prompt, meta=_meta())
    result = decl.to_nxs()
    assert (
        result.startswith("toolset MyTool:")
        and ">> do something <<" in result
        and result.endswith("__toolset")
    )
    _parse_toplevel(result)


# =============================================================================
# Step 7 — ImportStatement and ImportToolsetStatement
# =============================================================================


def test_import_toolset_no_alias_no_args():
    stmt = node.ImportToolsetStatement(
        name="my_toolset", elements=["MyTool"], args=None, alias=None, meta=_meta()
    )
    result = stmt.to_nxs()
    assert result == "from toolset my_toolset use MyTool"
    _parse_toplevel(result)


def test_import_toolset_with_alias():
    stmt = node.ImportToolsetStatement(
        name="my_toolset", elements=["MyTool"], args=None, alias="mt", meta=_meta()
    )
    result = stmt.to_nxs()
    assert result == "from toolset my_toolset as mt use MyTool"
    _parse_toplevel(result)


def test_import_toolset_with_args():
    args = node.Variable(name="cfg", prompt=None, path=None, meta=_meta())
    stmt = node.ImportToolsetStatement(
        name="my_toolset", elements=["MyTool"], args=args, alias="mt", meta=_meta()
    )
    result = stmt.to_nxs()
    assert "as mt" in result and "with [[cfg]]" in result and "use MyTool" in result
    _parse_toplevel(result)


def test_import_toolset_block_form_round_trip():
    src = "from toolset my_toolset use:\n    MyTool\n__use"
    result = round_trip(src)
    assert "from toolset my_toolset use MyTool" in result


def test_import_toolset_block_form_with_alias():
    src = "from toolset my_toolset as mt use:\n    MyTool\n__use"
    result = round_trip(src)
    assert "from toolset my_toolset as mt use MyTool" in result


def test_import_toolset_block_form_with_alias_and_args():
    src = "from toolset my_toolset as mt with [[cfg]] use:\n    MyTool\n__use"
    result = round_trip(src)
    assert "as mt" in result and "with [[cfg]]" in result and "use MyTool" in result


def test_import_toolset_multi_tool():
    stmt = node.ImportToolsetStatement(
        name="my_toolset",
        elements=["Tool1", "Tool2"],
        args=None,
        alias=None,
        meta=_meta(),
    )
    result = stmt.to_nxs()
    assert result == "from toolset my_toolset use Tool1, Tool2"
    _parse_toplevel(result)


def test_import_toolset_wildcard():
    stmt = node.ImportToolsetStatement(
        name="my_toolset", elements=["*"], args=None, alias=None, meta=_meta()
    )
    result = stmt.to_nxs()
    assert result == "from toolset my_toolset use *"
    _parse_toplevel(result)


# =============================================================================
# Step 8 — ActionInput and ActionOutput
# =============================================================================


def test_action_input_required():
    prompt = node.MicroPrompt(prompt="the user name", meta=_meta())
    ai = node.ActionInput(
        name="username", required=True, default=None, prompt=prompt, meta=_meta()
    )
    result = ai.to_nxs()
    assert result == "username (required) >> the user name <<"
    _parse_as_in(result)


def test_action_input_default():
    default_val = node.SingleValue(
        value="guest", inferred_type=node.VariableTypeEnum.STRING, meta=_meta()
    )
    prompt = node.MicroPrompt(prompt="user name", meta=_meta())
    ai = node.ActionInput(
        name="username",
        required=False,
        default=default_val,
        prompt=prompt,
        meta=_meta(),
    )
    result = ai.to_nxs()
    assert "username" in result and "default" in result and "guest" in result
    _parse_as_in(result)


def test_action_output():
    prompt = node.MicroPrompt(prompt="greeting message", meta=_meta())
    ao = node.ActionOutput(name="message", prompt=prompt, meta=_meta())
    result = ao.to_nxs()
    assert result == "message >> greeting message <<"
    _parse_as_out(result)


def test_action_input_unnamed_mod():
    prompt = node.MicroPrompt(prompt="any name", meta=_meta())
    ai = node.ActionInput(
        name="", required=True, default=None, prompt=prompt, meta=_meta()
    )
    result = ai.to_nxs()
    assert "(required)" in result and "any name" in result
    _parse_as_in(result)


def test_action_input_unnamed():
    prompt = node.MicroPrompt(prompt="any value", meta=_meta())
    ai = node.ActionInput(
        name="", required=False, default=None, prompt=prompt, meta=_meta()
    )
    result = ai.to_nxs()
    assert "any value" in result
    _parse_as_in(result)


def test_action_output_unnamed():
    prompt = node.MicroPrompt(prompt="the result", meta=_meta())
    ao = node.ActionOutput(name="", prompt=prompt, meta=_meta())
    result = ao.to_nxs()
    assert "the result" in result
    _parse_as_out(result)


# =============================================================================
# Step 9 — IfBlock, ElifBlock, ElseBlock, ConditionBlock
# =============================================================================


def _simple_cond():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    one = node.SingleValue(
        value=1, inferred_type=node.VariableTypeEnum.INT, meta=_meta()
    )
    return node.BinaryOperation(
        operation=node.BinaryOperationEnum.EQ, first=x, second=one, meta=_meta()
    )


def test_if_block():
    ib = node.IfBlock(
        condition=_simple_cond(), body=[node.Break(meta=_meta())], meta=_meta()
    )
    result = ib.to_nxs()
    assert result.startswith("if ") and "break" in result and "__if" not in result


def test_elif_block():
    eb = node.ElifBlock(
        condition=_simple_cond(), body=[node.Continue(meta=_meta())], meta=_meta()
    )
    result = eb.to_nxs()
    assert (
        result.startswith("elif ") and "continue" in result and "__elif" not in result
    )


def test_else_block():
    eb = node.ElseBlock(body=[node.Break(meta=_meta())], meta=_meta())
    result = eb.to_nxs()
    assert result.startswith("else:") and "break" in result and "__else" not in result


def test_condition_block():
    if_block = node.IfBlock(
        condition=_simple_cond(), body=[node.Break(meta=_meta())], meta=_meta()
    )
    else_block = node.ElseBlock(body=[node.Continue(meta=_meta())], meta=_meta())
    cb = node.ConditionBlock(
        if_block=if_block, elif_list=[], else_block=else_block, meta=_meta()
    )
    result = cb.to_nxs()
    assert "if " in result and "else:" in result and result.endswith("__if")
    _parse_as_body(result)


def test_condition_block_with_elif():
    if_block = node.IfBlock(
        condition=_simple_cond(), body=[node.Break(meta=_meta())], meta=_meta()
    )
    elif_block = node.ElifBlock(
        condition=_simple_cond(), body=[node.Continue(meta=_meta())], meta=_meta()
    )
    else_block = node.ElseBlock(body=[node.Return(val=[], meta=_meta())], meta=_meta())
    cb = node.ConditionBlock(
        if_block=if_block, elif_list=[elif_block], else_block=else_block, meta=_meta()
    )
    result = cb.to_nxs()
    assert "if " in result and "elif " in result and "else:" in result
    _parse_as_body(result)


# =============================================================================
# Step 10 — RepeatEachBlock, RepeatTimesBlock, RepeatWhileBlock, RepeatUntilBlock
# =============================================================================


def test_repeat_each_block():
    items_var = node.Variable(name="items", prompt=None, path=None, meta=_meta())
    rb = node.RepeatEachBlock(each=items_var, as_vars=["i", "item"], meta=_meta())
    rb.children = [node.Break(meta=_meta())]
    result = rb.to_nxs()
    assert (
        result.startswith("repeat each")
        and "as [i], [item]" in result
        and "__repeat" in result
    )
    _parse_as_body(result)


def test_repeat_times_block():
    rb = node.RepeatTimesBlock(times=3, as_vars=None, meta=_meta())
    rb.children = [node.Break(meta=_meta())]
    result = rb.to_nxs()
    assert result.startswith("repeat 3 times") and "__repeat" in result
    _parse_as_body(result)


def test_repeat_while_block():
    rb = node.RepeatWhileBlock(condition=_simple_cond(), max_it=None, meta=_meta())
    rb.children = [node.Break(meta=_meta())]
    result = rb.to_nxs()
    assert result.startswith("repeat while") and "__repeat" in result
    _parse_as_body(result)


def test_repeat_until_block():
    rb = node.RepeatUntilBlock(condition=_simple_cond(), max_it=5, meta=_meta())
    rb.children = [node.Break(meta=_meta())]
    result = rb.to_nxs()
    assert (
        result.startswith("repeat until") and "max 5" in result and "__repeat" in result
    )
    _parse_as_body(result)


def test_repeat_generic_round_trip():
    src = (
        "action _T >> t <<:\n"
        "body:\n"
        "    repeat >> do something <<:\n"
        "        break\n"
        "    __repeat\n"
        "__body\n"
        "__action"
    )
    result = round_trip(src)
    assert "repeat" in result and "break" in result


# =============================================================================
# Step 11 — Slot and Frame
# =============================================================================


def test_slot_with_type_and_prompt():
    prompt = node.MicroPrompt(prompt="full name", meta=_meta())
    slot = node.Slot(
        name="name",
        types={node.SlotTypesEnum.TEXT: None},
        card=None,
        prompt=prompt,
        meta=_meta(),
    )
    result = slot.to_nxs()
    assert result == "slot name as TEXT >> full name <<"
    _parse_as_slot(result)


def test_slot_with_cardinality():
    slot = node.Slot(
        name="tags",
        types={node.SlotTypesEnum.TEXT: None},
        card="*",
        prompt=None,
        meta=_meta(),
    )
    result = slot.to_nxs()
    assert "slot tags" in result and "TEXT" in result and "[*]" in result
    _parse_as_slot(result)


def test_frame():
    prompt = node.MicroPrompt(prompt="full name", meta=_meta())
    slot = node.Slot(
        name="name",
        types={node.SlotTypesEnum.TEXT: None},
        card=None,
        prompt=prompt,
        meta=_meta(),
    )
    frame = node.Frame(name="Person", meta=_meta())
    frame.children = [slot]
    result = frame.to_nxs()
    assert (
        result.startswith("frame Person:")
        and "slot name" in result
        and result.endswith("__frame")
    )
    _parse_toplevel(result)


def test_frame_with_inline_prompt():
    prompt_child = node.MicroPrompt(prompt="personal identity data", meta=_meta())
    slot = node.Slot(
        name="name",
        types={node.SlotTypesEnum.TEXT: None},
        card=None,
        prompt=None,
        meta=_meta(),
    )
    frame = node.Frame(name="Person", meta=_meta())
    frame.children = [prompt_child, slot]
    result = frame.to_nxs()
    assert ">> personal identity data <<" in result
    assert "slot name" in result
    _parse_toplevel(result)


def test_frame_with_multiline_prompt():
    prompt_child = node.MicroPrompt(prompt="line one\nline two", meta=_meta())
    frame = node.Frame(name="Doc", meta=_meta())
    frame.children = [prompt_child]
    result = frame.to_nxs()
    assert ">>> line one\nline two <<<" in result
    _parse_toplevel(result)


def test_slot_no_type():
    slot = node.Slot(name="x", types=None, card=None, prompt=None, meta=_meta())
    result = slot.to_nxs()
    assert result == "slot x"
    _parse_as_slot(result)


def test_slot_int_type():
    slot = node.Slot(
        name="count",
        types={node.SlotTypesEnum.INT: None},
        card=None,
        prompt=None,
        meta=_meta(),
    )
    result = slot.to_nxs()
    assert result == "slot count as INT"
    _parse_as_slot(result)


def test_slot_bool_type():
    slot = node.Slot(
        name="active",
        types={node.SlotTypesEnum.BOOL: None},
        card=None,
        prompt=None,
        meta=_meta(),
    )
    result = slot.to_nxs()
    assert result == "slot active as BOOL"
    _parse_as_slot(result)


def test_slot_float_type():
    slot = node.Slot(
        name="score",
        types={node.SlotTypesEnum.FLOAT: None},
        card=None,
        prompt=None,
        meta=_meta(),
    )
    result = slot.to_nxs()
    assert result == "slot score as FLOAT"
    _parse_as_slot(result)


def test_slot_enum_type():
    slot = node.Slot(
        name="color",
        types={node.SlotTypesEnum.ENUM: ["red", "blue"]},
        card=None,
        prompt=None,
        meta=_meta(),
    )
    result = slot.to_nxs()
    assert 'ENUM("red", "blue")' in result
    _parse_as_slot(result)


def test_slot_frame_type():
    slot = node.Slot(
        name="person",
        types={node.SlotTypesEnum.FRAME: "PersonFrame"},
        card=None,
        prompt=None,
        meta=_meta(),
    )
    result = slot.to_nxs()
    assert "PersonFrame" in result and "slot person" in result
    # PersonFrame must be declared before it can be referenced in a slot
    src = f"frame PersonFrame:\n__frame\nframe _F:\n{result}\n__frame"
    ParserLark().parse(src, "<test-slot-frame>")


def test_frame_nested():
    inner = node.Frame(name="Inner", meta=_meta())
    inner.children = [
        node.Slot(
            name="x",
            types={node.SlotTypesEnum.INT: None},
            card=None,
            prompt=None,
            meta=_meta(),
        )
    ]
    outer = node.Frame(name="Outer", meta=_meta())
    outer.children = [inner]
    result = outer.to_nxs()
    assert "frame Outer:" in result and "frame Inner:" in result
    _parse_toplevel(result)


# =============================================================================
# Step 12 — ActionBlock
# =============================================================================


def test_action_block_round_trip():
    src = (
        "action Greet >> greet the user <<:\n"
        "    in:\n"
        "        name (required) >> the user name <<\n"
        "    __in\n"
        "    out:\n"
        "        message >> greeting message <<\n"
        "    __out\n"
        "    body:\n"
        "        break\n"
        "    __body\n"
        "__action"
    )
    result = round_trip(src)
    assert "action Greet" in result and "__action" in result


def test_action_block_minimal_round_trip():
    src = "action Foo >> do foo <<:\n    body:\n    __body\n__action"
    result = round_trip(src)
    assert "action Foo" in result


# =============================================================================
# Step 13 — PlanBlock
# =============================================================================


def test_plan_block():
    pb = node.PlanBlock(action_inputs=[], action_outputs=[], body=[], meta=_meta())
    result = pb.to_nxs()
    assert (
        result.startswith("plan:") and "body:" in result and result.endswith("__plan")
    )
    _parse_as_plan(result)


# =============================================================================
# Step 14 — Deliberate
# =============================================================================


def test_deliberate_round_trip():
    src = (
        "deliberate Foo when >> run when needed <<:\n"
        "    mandate:\n"
        "    >> do things <<\n"
        "    __mandate\n"
        "    plan:\n"
        "        body:\n"
        "        __body\n"
        "    __plan\n"
        "__deliberate"
    )
    result = round_trip(src)
    assert "deliberate Foo" in result and "__deliberate" in result


# =============================================================================
# Step 15 — Require
# =============================================================================


def test_require():
    r = node.Require(file_path="path/to/script.nxs", meta=_meta())
    result = r.to_nxs()
    assert result == "require path/to/script.nxs"
    _parse_toplevel(result)


def test_require_nxc():
    r = node.Require(file_path="path/to/script.nxc", meta=_meta())
    result = r.to_nxs()
    assert result == "require path/to/script.nxc"
    _parse_toplevel(result)


# =============================================================================
# Step 16 — Variable with struct path
# =============================================================================


def test_variable_with_path():
    path = [
        node.SingleValue(
            value="field", inferred_type=node.VariableTypeEnum.STRING, meta=_meta()
        ),
        node.SingleValue(
            value="sub_field", inferred_type=node.VariableTypeEnum.STRING, meta=_meta()
        ),
    ]
    v = node.Variable(name="struct", prompt=None, path=path, meta=_meta())
    result = v.to_nxs()
    assert result == "[struct:field:sub_field]"
    _parse_as_body(f"[ [x] = {result} ]")


def test_variable_with_prompt():
    prompt = node.MicroPrompt(prompt="explain x", meta=_meta())
    v = node.Variable(name="x", prompt=prompt, path=None, meta=_meta())
    result = v.to_nxs()
    assert result == "[x >> explain x <<]"
    _parse_as_body(f"[ [z] = {result} ]")


def test_variable_access_index():
    path = [
        node.SingleValue(
            value="0", inferred_type=node.VariableTypeEnum.INT, meta=_meta()
        )
    ]
    v = node.Variable(name="items", prompt=None, path=path, meta=_meta())
    result = v.to_nxs()
    assert result == "[items:0]"
    _parse_as_body(f"[ [x] = {result} ]")


def test_variable_access_expr():
    idx = node.Variable(name="i", prompt=None, path=None, meta=_meta())
    v = node.Variable(name="items", prompt=None, path=[idx], meta=_meta())
    result = v.to_nxs()
    assert result == "[items:([i])]"
    _parse_as_body(f"[ [x] = {result} ]")


# =============================================================================
# Step 17 — Assignment
# =============================================================================


def test_assignment_single_value():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    one = node.SingleValue(
        value=1, inferred_type=node.VariableTypeEnum.INT, meta=_meta()
    )
    a = node.Assignment(var=x, value=one, meta=_meta())
    result = a.to_nxs()
    assert result == "[x] = 1"
    _parse_as_body(f"[ {result} ]")


def test_assignment_multi_value():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    y = node.Variable(name="y", prompt=None, path=None, meta=_meta())
    z = node.Variable(name="z", prompt=None, path=None, meta=_meta())
    a = node.Assignment(var=x, value=[y, z], meta=_meta())
    result = a.to_nxs()
    assert result == "[x] = ([y], [z])"
    _parse_as_body(f"[ {result} ]")


# =============================================================================
# Step 18 — Unary and Binary operations
# =============================================================================


def test_unary_operation_neg():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    op = node.UnaryOperation(
        operation=node.UnaryOperationEnum.NEG, operand=x, meta=_meta()
    )
    result = op.to_nxs()
    assert result == "-[x]"
    _parse_as_body(f"[ {result} ]")


def test_unary_operation_not():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    op = node.UnaryOperation(
        operation=node.UnaryOperationEnum.NOT, operand=x, meta=_meta()
    )
    result = op.to_nxs()
    assert result == "![x]"
    _parse_as_body(f"[ {result} ]")


def test_binary_operation_add():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    one = node.SingleValue(
        value=1, inferred_type=node.VariableTypeEnum.INT, meta=_meta()
    )
    op = node.BinaryOperation(
        operation=node.BinaryOperationEnum.ADD, first=x, second=one, meta=_meta()
    )
    result = op.to_nxs()
    assert result == "[x] + 1"
    _parse_as_body(f"[ {result} ]")


def test_binary_operation_concat():
    a = node.SingleValue(
        value="hello", inferred_type=node.VariableTypeEnum.STRING, meta=_meta()
    )
    b = node.SingleValue(
        value="world", inferred_type=node.VariableTypeEnum.STRING, meta=_meta()
    )
    op = node.BinaryOperation(
        operation=node.BinaryOperationEnum.CONCAT, first=a, second=b, meta=_meta()
    )
    result = op.to_nxs()
    assert result == '"hello" | "world"'
    _parse_as_body(f"[ {result} ]")


def test_binary_operation_fallback():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    default = node.SingleValue(
        value="default", inferred_type=node.VariableTypeEnum.STRING, meta=_meta()
    )
    op = node.BinaryOperation(
        operation=node.BinaryOperationEnum.FALLBACK,
        first=x,
        second=default,
        meta=_meta(),
    )
    result = op.to_nxs()
    assert result == '[x] ?? "default"'
    _parse_as_body(f"[ {result} ]")


def test_binary_operation_sub():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    y = node.Variable(name="y", prompt=None, path=None, meta=_meta())
    op = node.BinaryOperation(
        operation=node.BinaryOperationEnum.SUB, first=x, second=y, meta=_meta()
    )
    result = op.to_nxs()
    assert result == "[x] - [y]"
    _parse_as_body(f"[ {result} ]")


def test_binary_operation_mul():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    y = node.Variable(name="y", prompt=None, path=None, meta=_meta())
    op = node.BinaryOperation(
        operation=node.BinaryOperationEnum.MUL, first=x, second=y, meta=_meta()
    )
    result = op.to_nxs()
    assert result == "[x] * [y]"
    _parse_as_body(f"[ {result} ]")


def test_binary_operation_eq():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    y = node.Variable(name="y", prompt=None, path=None, meta=_meta())
    op = node.BinaryOperation(
        operation=node.BinaryOperationEnum.EQ, first=x, second=y, meta=_meta()
    )
    result = op.to_nxs()
    assert result == "[x] == [y]"
    _parse_as_body(f"[ {result} ]")


def test_binary_operation_lt():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    y = node.Variable(name="y", prompt=None, path=None, meta=_meta())
    op = node.BinaryOperation(
        operation=node.BinaryOperationEnum.LT, first=x, second=y, meta=_meta()
    )
    result = op.to_nxs()
    assert result == "[x] < [y]"
    _parse_as_body(f"[ {result} ]")


def test_binary_operation_gt():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    y = node.Variable(name="y", prompt=None, path=None, meta=_meta())
    op = node.BinaryOperation(
        operation=node.BinaryOperationEnum.GT, first=x, second=y, meta=_meta()
    )
    result = op.to_nxs()
    assert result == "[x] > [y]"
    _parse_as_body(f"[ {result} ]")


# =============================================================================
# Step 19 — Semantic inclusion (~> and <~)
# =============================================================================


@pytest.mark.parametrize(
    "operation, qualifier, expected_substrings",
    [
        (node.SimilarityEnum.SIM_RIGHT, None, ["~>", "[a]", "[b]"]),
        (node.SimilarityEnum.SIM_LEFT, None, ["<~", "[a]", "[b]"]),
        (
            node.SimilarityEnum.SIM_QUAL_RIGHT,
            (node.SimilarityQualifierEnum.CLOSE, None),
            ["close", "~>"],
        ),
        (
            node.SimilarityEnum.SIM_QUAL_LEFT,
            (node.SimilarityQualifierEnum.CLOSE, None),
            ["close", "<~"],
        ),
    ],
    ids=["right", "left", "right_qualified", "left_qualified"],
)
def test_similarity_inclusion(operation, qualifier, expected_substrings):
    a = node.Variable(name="a", prompt=None, path=None, meta=_meta())
    b = node.Variable(name="b", prompt=None, path=None, meta=_meta())
    # silence PyCharm false positive
    # noinspection PyTypeChecker
    op = node.SimilarityOperation(
        operation=operation, qualifier=qualifier, first=a, second=b, meta=_meta()
    )
    result = op.to_nxs()
    for s in expected_substrings:
        assert s in result
    _parse_as_body(f"[ {result} ]")


# =============================================================================
# Step 20 — BuiltinFunction
# =============================================================================


def test_builtin_function_list_args():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    bf = node.BuiltinFunction(
        function=node.BuiltinFunctionEnum.TO_STR,
        args=[x],
        meta=_meta(),
    )
    result = bf.to_nxs()
    assert result == "to_str([x])"
    _parse_as_body(f"[ {result} ]")


def test_builtin_function_single_arg_normalised():
    x = node.Variable(name="x", prompt=None, path=None, meta=_meta())
    bf = node.BuiltinFunction(
        function=node.BuiltinFunctionEnum.TO_STR,
        args=x,
        meta=_meta(),
    )
    assert isinstance(bf.args, list) and len(bf.args) == 1
    assert bf.to_nxs() == "to_str([x])"


# =============================================================================
# Step 21 — DoStatement
# =============================================================================


def test_do_statement_inline():
    text = node.Variable(name="text", prompt=None, path=None, meta=_meta())
    inp = node.Variable(name="input", prompt=None, path=None, meta=_meta())
    using = node.Assignment(var=text, value=inp, meta=_meta())
    stmt = node.DoStatement(
        name="trim",
        callable_type="tool",
        using=using,
        prompt=node.MicroPrompt(prompt="trim it", meta=_meta()),
        producing=None,
        producing_schema=None,
        meta=_meta(),
    )
    result = stmt.to_nxs()
    assert "do tool trim" in result and "using" in result and ">> trim it <<" in result
    _parse_as_body(result)


def test_do_statement_multiline():
    text = node.Variable(name="text", prompt=None, path=None, meta=_meta())
    inp = node.Variable(name="input", prompt=None, path=None, meta=_meta())
    using = node.Assignment(var=text, value=inp, meta=_meta())
    file_meta = node.FileMeta(line=(1, 4), column=(0, 0))
    stmt = node.DoStatement(
        name="trim",
        callable_type="tool",
        using=using,
        prompt=node.MicroPrompt(prompt="trim it", meta=_meta()),
        producing=None,
        producing_schema=None,
        meta={"file_meta": file_meta, "node_meta": None},
    )
    result = stmt.to_nxs()
    assert result.startswith("do tool trim:") and "__do" in result and "\n" in result
    _parse_as_body(result)


def test_do_statement_with_producing_and_schema():
    text = node.Variable(name="text", prompt=None, path=None, meta=_meta())
    inp = node.Variable(name="input", prompt=None, path=None, meta=_meta())
    using = node.Assignment(var=text, value=inp, meta=_meta())
    out = node.Variable(name="result", prompt=None, path=None, meta=_meta())
    stmt = node.DoStatement(
        name="extract",
        callable_type="action",
        using=using,
        prompt=None,
        producing=out,
        producing_schema="PersonFrame",
        meta=_meta(),
    )
    result = stmt.to_nxs()
    assert "producing" in result and "{PersonFrame}" in result
    _parse_as_body(result)


def test_do_statement_builtin():
    stmt = node.DoStatement(
        name="print",
        callable_type=None,
        using=node.Assignment(
            var=node.Variable(name="text", prompt=None, path=None, meta=_meta()),
            value=node.SingleValue(
                value="hello", inferred_type=node.VariableTypeEnum.STRING, meta=_meta()
            ),
            meta=_meta(),
        ),
        prompt=None,
        producing=None,
        producing_schema=None,
        meta=_meta(),
    )
    result = stmt.to_nxs()
    assert result.startswith("do print") and "using" in result
    _parse_as_body(result)


# =============================================================================
# Step 22 — Deliberate with generated_actions
# =============================================================================


def test_deliberate_with_generated_actions():
    action = node.ActionBlock(
        name="SubAction",
        prompt=node.MicroPrompt(prompt="do sub thing", meta=_meta()),
        action_inputs=[],
        action_outputs=[],
        body=None,
        meta=_meta(),
    )
    plan = node.PlanBlock(action_inputs=[], action_outputs=[], body=[], meta=_meta())
    d = node.Deliberate(
        name="Foo",
        when=node.MicroPrompt(prompt="run when needed", meta=_meta()),
        mandate=node.MicroPrompt(prompt="follow mandate", meta=_meta()),
        plan=plan,
        meta=_meta(),
        generated_actions=[action],
    )
    result = d.to_nxs()
    assert "action SubAction" in result and "__deliberate" in result
    ParserLark().parse(result, "<test-deliberate-generated>")


# =============================================================================
# Step 23 — Round-trip integration over all .nxs test scripts
# =============================================================================


@pytest.mark.parametrize("filepath", EXAMPLE_FILES, ids=lambda p: p.name)
def test_round_trip_script(filepath):
    source = filepath.read_text(encoding="utf-8")
    if not source.strip():
        return
    stmts = ParserLark().parse(source, str(filepath))
    nxs = "\n".join(stmt.to_nxs() for stmt in stmts)
    ParserLark().parse(nxs, "<round-trip>")
