import pytest
from lark import UnexpectedToken, UnexpectedCharacters

from nemantix.core.node import (
    Deliberate,
    PlanBlock,
    ActionBlock,
    ActionInput,
    ActionOutput,
    DoStatement,
    RepeatTimesBlock,
    RepeatEachBlock,
    RepeatWhileBlock,
    Variable,
    SingleValue,
    VariableTypeEnum,
    Frame,
    Slot,
    PythonToolDeclaration,
    ImportToolsetStatement,
    SimilarityOperation,
    SimilarityEnum,
    SimilarityQualifierEnum,
    NodeMeta,
    BinaryOperation,
    SlotTypesEnum,
    Assignment,
    BinaryOperationEnum,
    CallableTypeEnum,
    MicroPrompt,
    SchemedCollection,
    FrameApplyEnum,
)
from nemantix.core.parser import ParserLark


# Helper class to parse strings directly without file I/O
class StringParser(ParserLark):
    def parse_string(self, text):
        """Parses a string directly into an AST, bypassing file resolution."""
        # Use the internal Lark instance to parse string to tree
        tree = self._lark.parse(text)
        # Transform the tree to AST nodes
        # passing 'None' as file path since we are parsing raw strings
        return self._transformer.transform_with_file_info(tree, file=None)


@pytest.fixture(scope="module")
def parser(grammar_path):
    """
    Initializes the parser with the NXS v2 grammar.
    Uses the fixture from conftest.py to locate the grammar file.
    """
    return StringParser(grammar=grammar_path)


def test_deliberate_structure(parser):
    """
    Test the top-level 'deliberate' structure, including 'when' condition,
    imports, mandate, and the plan block.
    """
    code = """
    from toolset CAD use align_tool, measure_tool

    action my_action >> my beautiful action <<:
        body:
            >> do something
        __
    __

    deliberate alignment when >> I need to align the plans <<:
        mandate:
            >> ensure precision
        __

        plan:
            body:
                >> do something
            __
        __
    __
    """
    nodes = parser.parse_string(code)
    assert len(nodes) == 3
    imports, action, deliberate = nodes

    # Node Type
    assert isinstance(deliberate, Deliberate)
    assert deliberate.name == "alignment"

    # Condition
    assert deliberate.when.prompt == "I need to align the plans"
    assert isinstance(deliberate.when.meta, dict)

    # Imports
    assert isinstance(imports, ImportToolsetStatement)
    assert len(imports.elements) == 2

    assert imports.name == "CAD"
    assert imports.elements == [
        "align_tool",
        "measure_tool",
    ]  # Check specific list content
    assert imports.alias is None  # Ensure optional fields are None

    # Mandate
    assert deliberate.mandate.prompt == "ensure precision"

    # Plan
    plan = deliberate.get_plan()
    assert isinstance(plan, PlanBlock)
    assert (
        plan.qualifier is None
    )  # Default is None (or Undefined based on implementation)
    assert len(plan.children) == 1

    # Action inside plan
    assert isinstance(action, ActionBlock)
    assert action.name == "my_action"
    assert action.prompt.prompt == "my beautiful action"
    assert len(action.children) == 1


def test_guidelines_deprecated_alias(parser):
    """guidelines: keyword still parses but emits DeprecationWarning; result lands in .mandate."""
    import warnings

    code = """
    deliberate alignment when >> I need to align <<:
        guidelines:
            >> ensure precision
        __

        plan:
            body:
                >> do something
            __
        __
    __
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nodes = parser.parse_string(code)

    deliberate = next(n for n in nodes if isinstance(n, Deliberate))
    assert deliberate.mandate.prompt == "ensure precision"

    deprecation_messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert any("guidelines" in m and "mandate" in m for m in deprecation_messages)


def test_action_inputs_outputs(parser):
    """
    Test detailed action input/output definitions, including:
    - Named inputs
    - Modifiers (required, optional, default)
    - Prompts
    """
    code = """
    action data_process >> process my data <<:
        in:
            doc (required) >> the document to process
            threshold (default [0.5])
            opt_param (optional)
            (required) >> unnamed input
        __
        out: result >> the processed outcome

        body:
            >> process
        __
    __

    deliberate test when >> test <<:
    plan:
        in:
            doc (required) >> the document to process
            threshold (default [0.5])
            opt_param (optional)
            (required) >> unnamed input
        __
        out: result >> the processed outcome

        body:
            do action using [[doc], [threshold], [opt_param]]
        __
    __
    __
    """
    nodes = parser.parse_string(code)
    action = nodes[0]

    # Inputs
    assert len(action.input) == 4

    # Named + Required + Prompt
    inp0 = action.input[0]
    assert isinstance(inp0, ActionInput)
    assert inp0.name == "doc"
    assert inp0.required is True
    assert inp0.prompt.prompt == "the document to process"
    assert inp0.default is None

    # Named + Default
    inp1 = action.input[1]
    assert inp1.name == "threshold"
    assert inp1.required is False
    assert isinstance(inp1.default, SingleValue)
    assert inp1.default.value == 0.5
    assert inp1.default.inferred_type == VariableTypeEnum.FLOAT
    assert inp1.prompt is None

    # Named + Optional
    inp2 = action.input[2]
    assert inp2.name == "opt_param"
    assert inp2.required is False
    assert inp2.default is None

    # Unnamed + Required
    inp3 = action.input[3]
    assert inp3.name == ""  # Unnamed
    assert inp3.required is True
    assert inp3.prompt.prompt == "unnamed input"

    # Outputs
    assert isinstance(action.output, list)
    assert len(action.output) == 1

    out_res = action.output[0]
    assert isinstance(out_res, ActionOutput)
    assert out_res.name == "result"
    assert out_res.prompt.prompt == "the processed outcome"

    # Body
    assert len(action.children) == 1
    assert action.children[0].prompt == "process"


def test_frame_and_slots(parser):
    """
    Test Frame definitions with comprehensive coverage of slot attributes.
    """
    code = """
    frame AddressFrame:
        slot city as TEXT
    __

    frame Person:
        slot name as TEXT [1] >> The full name <<
        slot age as INT [0..1]
        slot role as ENUM("User", "Admin")
        slot main_address as AddressFrame
    __
    """
    nodes = parser.parse_string(code)
    assert len(nodes) == 2

    # --- Frame 1: AddressFrame ---
    addr_frame = nodes[0]
    assert isinstance(addr_frame, Frame)
    assert addr_frame.name == "AddressFrame"
    assert len(addr_frame.children) == 1

    # Slot: city (Basic Text)
    city_slot = addr_frame.children[0]
    assert isinstance(city_slot, Slot)
    assert city_slot.name == "city"
    assert len(city_slot.types) == 1
    assert SlotTypesEnum.TEXT in city_slot.types
    assert city_slot.cardinality is None
    assert city_slot.prompt is None

    # --- Frame 2: Person ---
    person_frame = nodes[1]
    assert isinstance(person_frame, Frame)
    assert person_frame.name == "Person"
    assert len(person_frame.children) == 4

    # Slot 1: name (Text + Card + Prompt)
    name_slot = person_frame.children[0]
    assert name_slot.name == "name"
    assert SlotTypesEnum.TEXT in name_slot.types
    assert name_slot.cardinality == "1"
    assert name_slot.prompt is not None
    assert name_slot.prompt.prompt == "The full name"

    # Slot 2: age (Int + Range Card)
    age_slot = person_frame.children[1]
    assert age_slot.name == "age"
    assert SlotTypesEnum.INT in age_slot.types
    assert age_slot.cardinality == "0..1"

    # Slot 3: role (Enum definition)
    role_slot = person_frame.children[2]
    assert role_slot.name == "role"
    enum_vals = role_slot.types.get(SlotTypesEnum.ENUM)
    assert isinstance(enum_vals, list)
    assert "User" in enum_vals
    assert "Admin" in enum_vals

    # Slot 4: main_address (Custom Frame Reference)
    addr_slot = person_frame.children[3]
    assert addr_slot.name == "main_address"
    assert addr_slot.types.get(SlotTypesEnum.FRAME) == "AddressFrame"


def test_intentables_label_and_meta(parser):
    """
    Test the 'Intentable' syntax: {label} @meta:value.
    Verifies that NodeMeta is correctly attached to the AST node.
    """
    code = """
    {act_1}
    @desc: "An action"
    action my_action >> action description <<:
        body:
            >> ...
        __
    __

    {my_delib}
    @intent.goal: "Test Metadata"
    @priority: 1
    deliberate tagged_deliberate when >> trigger <<:
        plan:
            body:
                do action my_action
            __
        __
    __
    """
    nodes = parser.parse_string(code)
    action, delib = nodes

    # Check Deliberate Meta
    assert delib.meta["node_meta"] is not None
    nm = delib.meta["node_meta"]
    assert isinstance(nm, NodeMeta)
    assert nm.label == "my_delib"
    assert len(nm.annotations) == 2

    # Check Annotation contents
    ann_dict = {a.name: a.value for a in nm.annotations}
    assert "intent.goal" in ann_dict
    assert isinstance(ann_dict["intent.goal"], SingleValue)
    assert ann_dict["intent.goal"].value == "Test Metadata"

    assert "priority" in ann_dict
    assert isinstance(ann_dict["priority"], SingleValue)
    assert ann_dict["priority"].value == 1

    # Check Action Meta
    nm_act = action.meta["node_meta"]
    assert nm_act.label == "act_1"
    assert len(nm_act.annotations) == 1
    assert nm_act.annotations[0].name == "desc"
    assert nm_act.annotations[0].value.value == "An action"


def test_similarity_operations(parser):
    """
    Test the semantic similarity operators (~, ~>, <~) and their qualifiers.
    """
    code = """
    action check >> compare animals <<:
        body:
            if ["cat" ~ "animal"]:
                >> basic sim
            elif ["cat" ~close~ "feline"]:
                >> qualified sim
            elif ["cat" ~> "mammal"]:
                >> inclusion
            elif ["mammal" <~ "cat"]:
                >> reverse inclusion
            __
        __
    __

    deliberate sim when >> x <<:
            plan:
                body:
                    do action check
                __
            __
        __
    """
    nodes = parser.parse_string(code)
    action_body = nodes[0].children
    cond_block = action_body[0]  # ConditionBlock

    # Get if statements
    if_stmt = cond_block.children[0]
    op1 = if_stmt.condition
    assert isinstance(op1, SimilarityOperation)
    assert op1.operation == SimilarityEnum.SIM
    assert op1.qualifier is None
    assert isinstance(op1.first, SingleValue)
    assert op1.first.value == "cat"
    assert isinstance(op1.second, SingleValue)
    assert op1.second.value == "animal"

    elif_stmts = cond_block.children[1:]

    # ~close~
    op2 = elif_stmts[0].condition
    assert isinstance(op2, SimilarityOperation)
    assert op2.operation == SimilarityEnum.SIM_QUAL
    assert op2.qualifier[0] == SimilarityQualifierEnum.CLOSE
    assert isinstance(op2.second, SingleValue)
    assert op2.second.value == "feline"

    # ~>
    op3 = elif_stmts[1].condition
    assert isinstance(op3, SimilarityOperation)
    assert op3.operation == SimilarityEnum.SIM_RIGHT
    assert isinstance(op3.second, SingleValue)
    assert op3.second.value == "mammal"

    # <~
    op4 = elif_stmts[2].condition
    assert isinstance(op4, SimilarityOperation)
    assert op4.operation == SimilarityEnum.SIM_LEFT
    assert isinstance(op4.first, SingleValue)
    assert isinstance(op4.second, SingleValue)
    assert op4.first.value == "mammal"
    assert op4.second.value == "cat"


def test_expressions_and_variables(parser):
    """
    Test complex expressions including variable access, paths, and arithmetic.
    """
    from nemantix.core.node import Collection  # Ensure Collection is imported

    code = """
    action calc >> expression testing <<:
        body:
            # Variable access
            [[x] = [y]]
            # Path access
            [[val] = [user:profile:age]]
            # List/Structure
            [[list] = (1, 2, key: "value")]
            # Arithmetic
            [[res] = ([a] + 5) * 2]
        __
    __

    deliberate expr_test when >> x <<:
        plan:
            body:
                do action calc
            __
        __
    __
    """
    nodes = parser.parse_string(code)
    body = nodes[0].children

    # [x] = [y]
    assign1 = body[0]
    assert isinstance(assign1, Assignment)
    assert assign1.var.name == "x"
    assert isinstance(assign1.value, Variable)
    assert assign1.value.name == "y"

    # [val] = [user:profile:age]
    assign_path = body[1]
    assert assign_path.var.name == "val"
    var_source = assign_path.value
    assert isinstance(var_source, Variable)
    assert var_source.name == "user"

    # Verify path components are parsed as SingleValues
    assert len(var_source.path) == 2
    assert isinstance(var_source.path[0], SingleValue)
    assert var_source.path[0].value == "profile"
    assert var_source.path[1].value == "age"

    # List/Structure: [[list] = (1, 2, key: "value")]
    assign_list = body[2]
    collection = assign_list.value

    # Verify Collection wrapper
    assert isinstance(collection, Collection)
    assert collection.inferred_type == VariableTypeEnum.LIST
    assert len(collection.value) == 3

    # Item 1: Integer 1
    item1 = collection.value[0]
    assert isinstance(item1, SingleValue)
    assert item1.value == 1
    assert item1.inferred_type == VariableTypeEnum.INT

    # Item 2: Integer 2
    item2 = collection.value[1]
    assert isinstance(item2, SingleValue)
    assert item2.value == 2
    assert item2.inferred_type == VariableTypeEnum.INT

    # Item 3: Key-Value pair {key: "value"}
    # The parser transforms `key: value` into a python dict containing AST nodes
    item3 = collection.value[2]
    assert isinstance(item3, dict)
    assert "key" in item3
    kv_val = item3["key"]
    assert isinstance(kv_val, SingleValue)
    assert kv_val.value == "value"
    assert kv_val.inferred_type == VariableTypeEnum.STRING

    # Arithmetic precedence check
    assign_math = body[3]
    # ([a] + 5) * 2  -> BinaryOp(MUL, BinaryOp(ADD, a, 5), 2)
    outer_op = assign_math.value
    assert isinstance(outer_op, BinaryOperation)
    assert outer_op.operation == BinaryOperationEnum.MUL

    # Check right operand (2)
    assert isinstance(outer_op.second, SingleValue)
    assert outer_op.second.value == 2

    # Check left operand (Group -> Add)
    # Note: group_expr usually returns the inner item directly in AST if purely grouping
    inner_op = outer_op.first
    assert isinstance(inner_op, BinaryOperation)
    assert inner_op.operation == BinaryOperationEnum.ADD

    assert isinstance(inner_op.first, Variable)
    assert inner_op.first.name == "a"
    assert isinstance(inner_op.second, SingleValue)
    assert inner_op.second.value == 5


def test_toolset_definition(parser):
    """
    Test defining a Python toolset block.
    """
    code = """
    toolset MyTools:
        >>>
        class MyTools(Toolset):
            def run(self): pass
        <<<
    __
    """
    nodes = parser.parse_string(code)
    assert len(nodes) == 1
    ts = nodes[0]
    assert isinstance(ts, PythonToolDeclaration)
    assert ts.name == "MyTools"
    # Ensure the python code prompt is captured
    assert "class MyTools" in ts.prompt.prompt
    assert "def run(self)" in ts.prompt.prompt


def test_loops(parser):
    """
    Test different loop constructs: repeat times, repeat each, repeat while.
    """
    code = """
    action looping >> many loops << :
            body:
                repeat 5 times:
                    >> work
                __

                repeat each [items] as [idx], [val]:
                    >> work
                __

                repeat while [[x] < 10]:
                    >> work
                __
            __
        __

    deliberate loops when >> x <<:
        plan:
            body:
                do action looping
            __
        __
    __
    """
    nodes = parser.parse_string(code)
    body = nodes[0].children

    # Repeat Times
    loop_times = body[0]
    assert isinstance(loop_times, RepeatTimesBlock)
    assert loop_times.times == 5
    assert len(loop_times.children) == 1
    assert loop_times.children[0].prompt == "work"

    # Repeat Each
    loop_each = body[1]
    assert isinstance(loop_each, RepeatEachBlock)
    # Check iterable
    assert isinstance(loop_each.each, Variable)
    assert loop_each.each.name == "items"
    # Check bindings
    assert len(loop_each.as_vars) == 2
    assert "idx" in loop_each.as_vars
    assert "val" in loop_each.as_vars

    # Repeat While
    loop_while = body[2]
    assert isinstance(loop_while, RepeatWhileBlock)
    assert isinstance(loop_while.condition, BinaryOperation)
    assert loop_while.condition.operation == BinaryOperationEnum.LT
    assert isinstance(loop_while.condition.first, Variable)
    assert isinstance(loop_while.condition.second, SingleValue)
    assert loop_while.condition.first.name == "x"
    assert loop_while.condition.second.value == 10


def test_do_statement_calls(parser):
    """
    Test 'do' statements for tool/action calls, including 'using' and 'producing'.
    """
    code = """
    action call >> let's test the do << :
        body:
            do tool my_tool using [[arg] = 1] producing [[res]] >> Call tool

            do action other.action:
                using [[a] = [x]]
                producing [[y]]
            __
        __
    __
    """
    nodes = parser.parse_string(code)
    body = nodes[0].children

    # First DO: do tool my_tool ...
    stmt1 = body[0]
    assert isinstance(stmt1, DoStatement)
    assert stmt1.callable_type == CallableTypeEnum.TOOL
    assert stmt1.name == "my_tool"

    # Check Using: [arg] = 1
    assert isinstance(stmt1.using, Assignment)
    assert stmt1.using.var.name == "arg"
    assert isinstance(stmt1.using.value, SingleValue)
    assert stmt1.using.value.value == 1

    # Check Producing: [res]
    # If producing is just a variable `[[res]]`, expression parsing returns a Variable or List
    # In 'expression' rule: `[[res]]` -> List([res]) or Variable([res]) depending on grammar?
    # Actually `[[res]]` parses as `Variable` if it is simple.
    # But `expression` usually wraps things. Let's check the type.
    assert stmt1.producing is not None
    # Check prompt
    assert stmt1.prompt.prompt.strip() == "Call tool"

    # Second DO: do action other.action ...
    stmt2 = body[1]
    assert isinstance(stmt2, DoStatement)
    assert stmt2.callable_type == CallableTypeEnum.ACTION
    assert stmt2.name == "other.action"

    # Check Using: [a] = [x]
    assert isinstance(stmt2.using, Assignment)
    assert stmt2.using.var.name == "a"
    assert isinstance(stmt2.using.value, Variable)
    assert stmt2.using.value.name == "x"


def test_do_statement_with_producing_schema(parser):
    """
    Test that a 'do llm' statement correctly extracts the producing schema
    when provided as a Frame identifier.
    """
    code = """
    action test_action >> action <<:
        body:
            do llm using ["extract data"] producing [[result]] as {PERSON}
        __
    __
    """
    nodes = parser.parse_string(code)
    body = nodes[0].children

    do_stmt = body[0]
    assert isinstance(do_stmt, DoStatement)
    assert do_stmt.name == "llm"

    # Check producing variable
    assert isinstance(do_stmt.producing, Variable)
    assert do_stmt.producing.name == "result"

    # Check producing schema (Frame Name)
    assert do_stmt.producing_schema == "PERSON"


def test_frame_apply_on_variable(parser):
    """Frame application on already-defined variables, both postfix and prefix."""
    code = """
    action test_action >> action <<:
        body:
            [[loose] = [my_struct]{Person}]
            [[strict] = {Person}[my_struct]]
            [[nested] = [my_struct:field]{Person}]
        __
    __
    """
    body = parser.parse_string(code)[0].children

    loose = body[0].value
    assert isinstance(loose, SchemedCollection)
    assert loose.apply_type == FrameApplyEnum.POST
    assert isinstance(loose.value, Variable)
    assert loose.value.name == "my_struct"

    strict = body[1].value
    assert isinstance(strict, SchemedCollection)
    assert strict.apply_type == FrameApplyEnum.PRE
    assert isinstance(strict.value, Variable)

    nested = body[2].value
    assert isinstance(nested, SchemedCollection)
    assert isinstance(nested.value, Variable)
    assert nested.value.name == "my_struct"
    assert len(nested.value.path) == 1


def test_do_statement_with_generative_schema(parser):
    """
    Test that a 'do llm' statement correctly extracts a MicroPrompt
    when the schema is provided generatively.
    """
    code = """
    action test_action >> action <<:
        body:
            do llm using ["extract data"] producing [[result]] as >>> any valid json <<<
        __
    __
    """
    nodes = parser.parse_string(code)
    body = nodes[0].children

    do_stmt = body[0]
    assert isinstance(do_stmt, DoStatement)
    assert do_stmt.name == "llm"

    # Check producing schema (Generative MicroPrompt)
    assert isinstance(do_stmt.producing_schema, MicroPrompt)
    assert do_stmt.producing_schema.prompt == "any valid json"


def test_syntax_error_handling(parser):
    """
    Test that invalid syntax raises an appropriate exception.
    """
    # Missing colon after deliberate condition
    code = """
    deliberate broken when >> test <<
        plan:
            action x >> example action <<: __
        __
    __
    """
    # noinspection PyTypeChecker
    with pytest.raises((UnexpectedToken, UnexpectedCharacters)):
        parser.parse_string(code)


def test_missing_end_block(parser):
    """
    Test detection of unclosed blocks.
    """
    code = """
    deliberate unclosed when >> test <<:
        plan:
            action x >> example action <<:
                body:
                    >> do something
            # Missing __ or __action or __body
    """
    # noinspection PyTypeChecker
    with pytest.raises((UnexpectedToken, UnexpectedCharacters)):
        parser.parse_string(code)


class TestWhitespaceBoundaries:
    """
    Keyword and end-marker boundary enforcement.

    Fix 1: alphabetic keyword terminals require a word boundary (\\b) after the
    keyword, so glued forms like 'actionfoo' are rejected.

    Fix 2: end-marker terminals require end-of-line after '__*', so concatenated
    markers ('____') and inline statements ('__[[expr]]') are rejected.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _action(body: str) -> str:
        return f"action a >>desc<<:\nbody:\n{body}\n__\n__\n"

    # ------------------------------------------------------------------
    # Fix 1 — keyword gluing: must raise
    # ------------------------------------------------------------------
    def test_action_keyword_glued_to_name(self, parser):
        with pytest.raises((UnexpectedToken, UnexpectedCharacters)):
            parser.parse_string("actionfoo >>x<<:\nbody:\n__\n__\n")

    def test_action_keyword_glued_to_keyword_as_name(self, parser):
        # 'actionaction' — second 'action' was silently used as the name
        with pytest.raises((UnexpectedToken, UnexpectedCharacters)):
            parser.parse_string("actionaction >>x<<:\nbody:\n__\n__\n")

    def test_frame_keyword_glued_to_name(self, parser):
        with pytest.raises((UnexpectedToken, UnexpectedCharacters)):
            parser.parse_string("framefoo:\n__\n")

    def test_deliberate_keyword_glued_to_name(self, parser):
        with pytest.raises((UnexpectedToken, UnexpectedCharacters)):
            parser.parse_string("deliberatefoo when >>c<<:\n__\n")

    def test_do_keyword_glued_to_callee(self, parser):
        with pytest.raises((UnexpectedToken, UnexpectedCharacters)):
            parser.parse_string(self._action("dofoo\n"))

    def test_do_tool_keywords_glued(self, parser):
        with pytest.raises((UnexpectedToken, UnexpectedCharacters)):
            parser.parse_string(self._action("dotool foo\n"))

    def test_repeat_keyword_glued_to_each(self, parser):
        with pytest.raises((UnexpectedToken, UnexpectedCharacters)):
            parser.parse_string(self._action("repeateach [x]:\n__\n"))

    # ------------------------------------------------------------------
    # Fix 2 — end-marker issues: must raise
    # ------------------------------------------------------------------
    def test_double_end_block_concatenated(self, parser):
        # '____' on one line must not silently close two blocks
        with pytest.raises((UnexpectedToken, UnexpectedCharacters)):
            parser.parse_string(self._action("if [true]:\n____\n__\n"))

    def test_statement_after_end_block_same_line(self, parser):
        with pytest.raises((UnexpectedToken, UnexpectedCharacters)):
            parser.parse_string(self._action("__[[var]=3]\n"))

    def test_named_end_marker_with_statement_same_line(self, parser):
        with pytest.raises((UnexpectedToken, UnexpectedCharacters)):
            parser.parse_string(self._action("if [true]:\n__if[[v]=3]\n__\n"))

    # ------------------------------------------------------------------
    # Regression — correct forms must still parse
    # ------------------------------------------------------------------
    def test_spaced_action_parses(self, parser):
        result = parser.parse_string("action foo >>x<<:\nbody:\n__\n__\n")
        assert result

    def test_spaced_frame_parses(self, parser):
        result = parser.parse_string("frame foo:\n__\n")
        assert result

    def test_end_block_with_inline_comment(self, parser):
        # __ followed by an inline comment must still be a valid end-marker
        result = parser.parse_string(
            "action a >>desc<<:\nbody:\n__  # end body\n__  # end action\n"
        )
        assert result

    def test_named_end_marker_on_own_line(self, parser):
        # __if on its own line is valid; helper then adds __ (body) + __ (action)
        result = parser.parse_string(self._action("if [true]:\n__if\n"))
        assert result

    def test_nested_if_separate_end_markers(self, parser):
        # two nested ifs each closed on their own line; helper adds body + action closers
        result = parser.parse_string(
            self._action("if [true]:\nif [false]:\n__\n__\n")
        )
        assert result
