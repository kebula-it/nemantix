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
# Step 5 — SchemedCollection
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


# =============================================================================
# Step 4 — SimilarityOperation
# =============================================================================


def test_similarity_operation_simple():
    a = node.Variable(name="a", prompt=None, path=None, meta=_meta())
    b = node.Variable(name="b", prompt=None, path=None, meta=_meta())
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


# =============================================================================
# Step 3 — MetaExpression
# =============================================================================


def test_meta_expression():
    me = node.MetaExpression(quals=["some_var", "intent", "completion"], meta=_meta())
    assert me.to_nxs() == "{{some_var@intent.completion}}"


# =============================================================================
# Step 2 — Return, Break, Continue
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


# =============================================================================
# Step 6 — PythonToolDeclaration
# =============================================================================


def test_python_tool_declaration():
    prompt = node.MicroPrompt(prompt="do something", meta=_meta())
    decl = node.PythonToolDeclaration(name="MyTool", prompt=prompt, meta=_meta())
    result = decl.to_nxs()
    assert result.startswith("toolset MyTool:") and ">> do something <<" in result and result.endswith("__toolset")
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


# =============================================================================
# Step 11 — Slot and Frame
# =============================================================================


def test_slot_with_type_and_prompt():
    prompt = node.MicroPrompt(prompt="full name", meta=_meta())
    slot = node.Slot(
        name="name",
        types=[node.SlotTypesEnum.TEXT],
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
        types=[node.SlotTypesEnum.TEXT],
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
        types=[node.SlotTypesEnum.TEXT],
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
        "    guidelines:\n"
        "    >> do things <<\n"
        "    __guidelines\n"
        "    plan:\n"
        "        body:\n"
        "        __body\n"
        "    __plan\n"
        "__deliberate"
    )
    result = round_trip(src)
    assert "deliberate Foo" in result and "__deliberate" in result


# =============================================================================
# Step 15 — Round-trip integration over all .nxs test scripts
# =============================================================================


@pytest.mark.parametrize("filepath", EXAMPLE_FILES, ids=lambda p: p.name)
def test_round_trip_script(filepath):
    source = filepath.read_text(encoding="utf-8")
    if not source.strip():
        return
    stmts = ParserLark().parse(source, str(filepath))
    nxs = "\n".join(stmt.to_nxs() for stmt in stmts)
    ParserLark().parse(nxs, "<round-trip>")
