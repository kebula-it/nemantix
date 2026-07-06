"""TDD suite for M1: Symbol Table — core static analysis of NXS/NXC/NXV ASTs."""

from __future__ import annotations

from nemantix.core.node import (
    ActionBlock,
    ActionInput,
    Assignment,
    BinaryOperation,
    BinaryOperationEnum,
    BuiltinFunction,
    BuiltinFunctionEnum,
    ConditionBlock,
    Deliberate,
    DoStatement,
    FileMeta,
    Frame,
    IfBlock,
    ImportToolsetStatement,
    MicroPrompt,
    NodeMeta,
    PlanBlock,
    PythonToolDeclaration,
    RepeatEachBlock,
    RepeatTimesBlock,
    Slot,
    Variable,
)
from nemantix.core.symbol_table import (
    Scope,
    Symbol,
    SymbolKind,
    SymbolTableBuilder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fm(line: int, col: int = 0, end_col: int | None = None) -> FileMeta:
    ec = end_col if end_col is not None else col + 1
    return FileMeta(line=(line, line), column=(col, ec))


def _meta(line: int, col: int = 0) -> dict:
    nm = NodeMeta(annotations=[], label=None, file_meta=_fm(line, col))
    return {"node_meta": nm}


def _mp(line: int = 1) -> MicroPrompt:
    return MicroPrompt(prompt="test", meta=_meta(line))


def _var(name: str, line: int, col: int = 0) -> Variable:
    return Variable(name=name, prompt=None, path=None, meta=_meta(line, col))


def _assign(name: str, line: int) -> Assignment:
    return Assignment(var=_var(name, line), value=None, meta=_meta(line))


def _deliberate(name: str, line: int) -> Deliberate:
    plan = PlanBlock(action_inputs=[], action_outputs=[], body=None, meta=_meta(line))
    return Deliberate(
        name=name, when=_mp(line), mandate=_mp(line), plan=plan, meta=_meta(line)
    )


def _action(name: str, line: int) -> ActionBlock:
    return ActionBlock(
        name=name,
        prompt=_mp(line),
        action_inputs=[],
        action_outputs=[],
        body=None,
        meta=_meta(line),
    )


def _frame(name: str, line: int) -> Frame:
    return Frame(name=name, meta=_meta(line))


def _slot(name: str, line: int) -> Slot:
    return Slot(name=name, types=None, card=None, prompt=None, meta=_meta(line))


def _do(name: str, line: int, using=None) -> DoStatement:
    return DoStatement(
        name=name,
        callable_type=None,
        using=using,
        prompt=None,
        producing=None,
        producing_schema=None,
        meta=_meta(line),
    )


# ---------------------------------------------------------------------------
# Group 1: Scope
# ---------------------------------------------------------------------------


class TestScope:
    def test_define_and_local_lookup(self):
        scope = Scope()
        sym = Symbol(name="x", kind=SymbolKind.VARIABLE, defined_at=_fm(1))
        scope.define(sym)
        assert scope.local("x") is sym

    def test_lookup_finds_in_local(self):
        scope = Scope()
        sym = Symbol(name="x", kind=SymbolKind.VARIABLE, defined_at=_fm(1))
        scope.define(sym)
        assert scope.lookup("x") is sym

    def test_lookup_walks_parent(self):
        parent = Scope()
        child = Scope(parent=parent)
        sym = Symbol(name="x", kind=SymbolKind.VARIABLE, defined_at=_fm(1))
        parent.define(sym)
        assert child.lookup("x") is sym

    def test_lookup_returns_none_for_undefined(self):
        scope = Scope()
        assert scope.lookup("missing") is None

    def test_local_does_not_walk_parent(self):
        parent = Scope()
        child = Scope(parent=parent)
        sym = Symbol(name="x", kind=SymbolKind.VARIABLE, defined_at=_fm(1))
        parent.define(sym)
        assert child.local("x") is None

    def test_child_shadows_parent(self):
        parent = Scope()
        child = Scope(parent=parent)
        parent_sym = Symbol(name="x", kind=SymbolKind.VARIABLE, defined_at=_fm(1))
        child_sym = Symbol(name="x", kind=SymbolKind.VARIABLE, defined_at=_fm(2))
        parent.define(parent_sym)
        child.define(child_sym)
        assert child.lookup("x") is child_sym

    def test_symbols_property_returns_copy(self):
        scope = Scope()
        sym = Symbol(name="x", kind=SymbolKind.VARIABLE, defined_at=_fm(1))
        scope.define(sym)
        result = scope.symbols
        assert "x" in result
        result["y"] = sym
        assert "y" not in scope.symbols


# ---------------------------------------------------------------------------
# Group 2: Variable definitions
# ---------------------------------------------------------------------------


class TestVariableDefinitions:
    def test_assignment_creates_variable_symbol(self):
        table = SymbolTableBuilder().build([_assign("x", line=3)])
        sym = table.global_scope.lookup("x")
        assert sym is not None
        assert sym.kind == SymbolKind.VARIABLE

    def test_assignment_symbol_has_correct_line(self):
        table = SymbolTableBuilder().build([_assign("x", line=7)])
        sym = table.global_scope.lookup("x")
        assert sym.defined_at.line[0] == 7

    def test_multiple_assignments_all_defined(self):
        table = SymbolTableBuilder().build(
            [
                _assign("a", 1),
                _assign("b", 2),
                _assign("c", 3),
            ]
        )
        for name in ("a", "b", "c"):
            assert table.global_scope.lookup(name) is not None

    def test_repeat_each_as_vars_are_variables(self):
        node = RepeatEachBlock(
            each=_assign("items", 1),
            as_vars=["item", "idx"],
            meta=_meta(2),
        )
        table = SymbolTableBuilder().build([node])
        assert table.global_scope.lookup("item").kind == SymbolKind.VARIABLE
        assert table.global_scope.lookup("idx").kind == SymbolKind.VARIABLE

    def test_repeat_times_as_vars_are_variables(self):
        node = RepeatTimesBlock(times=3, as_vars=["i"], meta=_meta(5))
        table = SymbolTableBuilder().build([node])
        assert table.global_scope.lookup("i").kind == SymbolKind.VARIABLE


# ---------------------------------------------------------------------------
# Group 3: Top-level declarations
# ---------------------------------------------------------------------------


class TestDeclarations:
    def test_deliberate_creates_deliberate_symbol(self):
        table = SymbolTableBuilder().build([_deliberate("my_deliberate", line=1)])
        sym = table.global_scope.lookup("my_deliberate")
        assert sym is not None
        assert sym.kind == SymbolKind.DELIBERATE

    def test_frame_creates_frame_symbol(self):
        table = SymbolTableBuilder().build([_frame("MyFrame", line=2)])
        sym = table.global_scope.lookup("MyFrame")
        assert sym is not None
        assert sym.kind == SymbolKind.FRAME

    def test_slot_inside_frame_creates_slot_symbol(self):
        f = _frame("MyFrame", line=2)
        f.children.append(_slot("age", line=3))
        table = SymbolTableBuilder().build([f])
        # slots live in the frame's subscope — findable by position, not from global
        sym = table.at(line=3, col=0)
        assert sym is not None
        assert sym.kind == SymbolKind.SLOT

    def test_import_toolset_creates_tool_symbol_with_alias(self):
        node = ImportToolsetStatement(
            name="filesystem", elements=[], args=None, alias="fs", meta=_meta(1)
        )
        table = SymbolTableBuilder().build([node])
        assert table.global_scope.lookup("fs").kind == SymbolKind.TOOL

    def test_import_toolset_without_alias_uses_name(self):
        node = ImportToolsetStatement(
            name="filesystem", elements=[], args=None, alias=None, meta=_meta(1)
        )
        table = SymbolTableBuilder().build([node])
        assert table.global_scope.lookup("filesystem").kind == SymbolKind.TOOL

    def test_python_tool_declaration_creates_tool_symbol(self):
        node = PythonToolDeclaration(name="my_tool", prompt=_mp(1), meta=_meta(1))
        table = SymbolTableBuilder().build([node])
        assert table.global_scope.lookup("my_tool").kind == SymbolKind.TOOL

    def test_action_inside_deliberate_creates_action_symbol(self):
        d = _deliberate("my_deliberate", line=1)
        d.generated_actions.append(_action("do_something", line=5))
        table = SymbolTableBuilder().build([d])
        # action is defined inside the scope of the deliberate — not in global
        deliberate_sym = table.global_scope.lookup("my_deliberate")
        assert deliberate_sym is not None
        assert table.global_scope.lookup("do_something") is None


# ---------------------------------------------------------------------------
# Group 4: References
# ---------------------------------------------------------------------------


class TestReferences:
    def test_variable_usage_adds_reference(self):
        stmts = [
            _assign("x", line=1),
            Assignment(
                var=_var("y", 2),
                value=_var("x", 3),
                meta=_meta(2),
            ),
        ]
        table = SymbolTableBuilder().build(stmts)
        sym = table.global_scope.lookup("x")
        assert len(sym.references) == 1
        assert sym.references[0].line[0] == 3

    def test_undefined_variable_usage_has_no_reference(self):
        stmts = [
            Assignment(var=_var("y", 1), value=_var("unknown", 1), meta=_meta(1)),
        ]
        table = SymbolTableBuilder().build(stmts)
        assert table.global_scope.lookup("unknown") is None

    def test_do_statement_adds_reference_to_deliberate(self):
        stmts = [
            _deliberate("my_deliberate", line=1),
            _do("my_deliberate", line=5),
        ]
        table = SymbolTableBuilder().build(stmts)
        sym = table.global_scope.lookup("my_deliberate")
        assert len(sym.references) == 1
        assert sym.references[0].line[0] == 5


# ---------------------------------------------------------------------------
# Group 5: SymbolTable.at()
# ---------------------------------------------------------------------------


class TestAt:
    def test_at_definition_line_returns_symbol(self):
        table = SymbolTableBuilder().build([_assign("x", line=4)])
        sym = table.at(line=4, col=0)
        assert sym is not None
        assert sym.name == "x"

    def test_at_unknown_position_returns_none(self):
        table = SymbolTableBuilder().build([_assign("x", line=4)])
        assert table.at(line=99, col=0) is None

    def test_at_reference_position_returns_symbol(self):
        stmts = [
            _assign("x", line=1),
            Assignment(var=_var("y", 2), value=_var("x", 3, col=5), meta=_meta(2)),
        ]
        table = SymbolTableBuilder().build(stmts)
        sym = table.at(line=3, col=5)
        assert sym is not None
        assert sym.name == "x"


# ---------------------------------------------------------------------------
# Group 6: SymbolTable.lookup()
# ---------------------------------------------------------------------------


class TestLookup:
    def test_lookup_finds_symbol_in_scope(self):
        table = SymbolTableBuilder().build([_assign("x", line=1)])
        sym = table.lookup("x", table.global_scope)
        assert sym is not None

    def test_lookup_returns_none_for_missing(self):
        table = SymbolTableBuilder().build([])
        assert table.lookup("missing", table.global_scope) is None


# ---------------------------------------------------------------------------
# Group 7: Expression traversal
# ---------------------------------------------------------------------------


def _bin_op(left, right, line: int) -> BinaryOperation:
    return BinaryOperation(
        operation=BinaryOperationEnum.CONCAT,
        first=left,
        second=right,
        meta=_meta(line),
    )


def _builtin_call(arg, line: int) -> BuiltinFunction:
    return BuiltinFunction(
        function=BuiltinFunctionEnum.TO_STR,
        args=[arg],
        meta=_meta(line),
    )


def _if_cond(condition, line: int) -> ConditionBlock:
    if_block = IfBlock(condition=condition, body=[], meta=_meta(line))
    return ConditionBlock(
        if_block=if_block, elif_list=None, else_block=None, meta=_meta(line)
    )


class TestExpressionTraversal:
    def test_builtin_do_name_not_unresolved(self):
        builder = SymbolTableBuilder()
        builder.build([_do("print", line=1)])
        names = [n for n, _ in builder.unresolved]
        assert "print" not in names

    def test_defined_var_in_do_using_is_resolved(self):
        builder = SymbolTableBuilder()
        builder.build([_assign("x", line=1), _do("print", line=2, using=_var("x", 2))])
        assert builder.unresolved == []

    def test_undefined_var_in_do_using_is_unresolved(self):
        builder = SymbolTableBuilder()
        builder.build([_do("print", line=1, using=_var("ghost", 1))])
        names = [n for n, _ in builder.unresolved]
        assert "ghost" in names

    def test_defined_var_in_binary_op_rhs_is_resolved(self):
        builder = SymbolTableBuilder()
        stmts = [
            _assign("x", line=1),
            Assignment(
                var=_var("y", 2),
                value=_bin_op(_var("x", 2), _var("x", 2), line=2),
                meta=_meta(2),
            ),
        ]
        builder.build(stmts)
        assert builder.unresolved == []

    def test_undefined_vars_in_binary_op_are_unresolved(self):
        builder = SymbolTableBuilder()
        op = _bin_op(_var("ghost1", 1), _var("ghost2", 1), line=1)
        builder.build([Assignment(var=_var("y", 1), value=op, meta=_meta(1))])
        names = [n for n, _ in builder.unresolved]
        assert "ghost1" in names
        assert "ghost2" in names

    def test_defined_var_in_builtin_arg_is_resolved(self):
        builder = SymbolTableBuilder()
        stmts = [
            _assign("x", line=1),
            _do("print", line=2, using=_builtin_call(_var("x", 2), line=2)),
        ]
        builder.build(stmts)
        assert builder.unresolved == []

    def test_undefined_var_in_builtin_arg_is_unresolved(self):
        builder = SymbolTableBuilder()
        builder.build(
            [_do("print", line=1, using=_builtin_call(_var("ghost", 1), line=1))]
        )
        names = [n for n, _ in builder.unresolved]
        assert "ghost" in names

    def test_defined_var_in_if_condition_is_resolved(self):
        builder = SymbolTableBuilder()
        builder.build([_assign("x", line=1), _if_cond(_var("x", 2), line=2)])
        assert builder.unresolved == []

    def test_undefined_var_in_if_condition_is_unresolved(self):
        builder = SymbolTableBuilder()
        builder.build([_if_cond(_var("ghost", 1), line=1)])
        names = [n for n, _ in builder.unresolved]
        assert "ghost" in names

    def test_producing_var_is_defined_not_referenced(self):
        do_stmt = _do("print", line=1, using=None)
        do_stmt.producing = _var("result", 2)
        builder = SymbolTableBuilder()
        builder.build([do_stmt])
        assert all(n != "result" for n, _ in builder.unresolved)
        sym = builder._global.lookup("result")
        assert sym is not None

    def test_undefined_action_call_goes_to_unresolved_calls(self):
        builder = SymbolTableBuilder()
        builder.build([_do("ghost_action", line=1)])
        names = [n for n, *_ in builder.unresolved_calls]
        assert "ghost_action" in names
        assert all(n != "ghost_action" for n, _ in builder.unresolved)

    def test_defined_action_call_not_in_unresolved_calls(self):
        action = _action("my_action", line=1)
        do_stmt = _do("my_action", line=2)
        builder = SymbolTableBuilder()
        builder.build([action, do_stmt])
        assert builder.unresolved_calls == []

    def test_undefined_frame_schema_goes_to_unresolved_schemas(self):
        do_stmt = _do("print", line=1)
        do_stmt.producing_schema = "GhostFrame"
        builder = SymbolTableBuilder()
        builder.build([do_stmt])
        names = [n for n, _ in builder.unresolved_schemas]
        assert "GhostFrame" in names

    def test_defined_frame_schema_not_in_unresolved_schemas(self):
        frame = _frame("MyFrame", line=1)
        do_stmt = _do("print", line=2)
        do_stmt.producing_schema = "MyFrame"
        builder = SymbolTableBuilder()
        builder.build([frame, do_stmt])
        assert builder.unresolved_schemas == []


# ---------------------------------------------------------------------------
# Group 9: Deliberate — private actions and plan inputs
# ---------------------------------------------------------------------------


class TestDeliberateScope:
    def test_private_action_in_generated_actions_not_unresolved(self):
        deliberate = _deliberate("my_d", 1)
        action = _action("pow3", 2)
        deliberate.generated_actions.append(action)
        do = _do("pow3", 3)
        deliberate.children.append(do)
        builder = SymbolTableBuilder()
        builder.build([deliberate])
        call_names = [n for n, *_ in builder.unresolved_calls]
        assert "pow3" not in call_names

    def test_private_action_not_visible_outside_deliberate(self):
        deliberate = _deliberate("my_d", 1)
        deliberate.generated_actions.append(_action("pow3", 2))
        do_outside = _do("pow3", 4)
        builder = SymbolTableBuilder()
        builder.build([deliberate, do_outside])
        call_names = [n for n, *_ in builder.unresolved_calls]
        assert "pow3" in call_names

    def test_plan_inputs_visible_inside_deliberate(self):
        inp = ActionInput(
            name="x", required=False, default=None, prompt=None, meta=_meta(2)
        )
        plan = PlanBlock(
            action_inputs=[inp], action_outputs=[], body=None, meta=_meta(1)
        )
        deliberate = Deliberate(
            name="my_d", when=_mp(1), mandate=_mp(1), plan=plan, meta=_meta(1)
        )
        # Variable reference to "x" inside the deliberate's body
        ref_do = _do("print", 3)
        ref_do.using = _var("x", 3)
        deliberate.children.append(ref_do)
        builder = SymbolTableBuilder()
        builder.build([deliberate])
        unresolved_names = [n for n, _ in builder.unresolved]
        assert "x" not in unresolved_names

    def test_plan_inputs_not_visible_outside_deliberate(self):
        inp = ActionInput(
            name="secret", required=False, default=None, prompt=None, meta=_meta(2)
        )
        plan = PlanBlock(
            action_inputs=[inp], action_outputs=[], body=None, meta=_meta(1)
        )
        deliberate = Deliberate(
            name="my_d", when=_mp(1), mandate=_mp(1), plan=plan, meta=_meta(1)
        )
        ref_outside = _do("print", 5)
        ref_outside.using = _var("secret", 5)
        builder = SymbolTableBuilder()
        builder.build([deliberate, ref_outside])
        unresolved_names = [n for n, _ in builder.unresolved]
        assert "secret" in unresolved_names


# ---------------------------------------------------------------------------
# Group 10: Repeat blocks — None as_vars regression
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Group 10b: SymbolTable.at() — multiline FileMeta
# ---------------------------------------------------------------------------


class TestAtMultiline:
    def _multiline_fm(
        self, line_start: int, line_end: int, col_start: int = 1, col_end: int = 9
    ) -> dict:
        """FileMeta spanning multiple lines (mimics action block encoding)."""
        from nemantix.core.node import FileMeta, NodeMeta

        fm = FileMeta(line=(line_start, line_end), column=(col_start, col_end))
        return {"node_meta": NodeMeta(annotations=[], label=None, file_meta=fm)}

    def test_at_start_line_any_col_from_col_start(self):
        """Clicking at col > col_end on the first line of a multiline block still finds the symbol."""
        from nemantix.core.node import ActionBlock, MicroPrompt

        block = ActionBlock(
            name="my_action",
            prompt=MicroPrompt(prompt="p", meta=self._multiline_fm(1, 1)),
            action_inputs=[],
            action_outputs=[],
            body=None,
            meta=self._multiline_fm(2, 10, col_start=1, col_end=9),
        )
        table = SymbolTableBuilder().build([block])
        # col=7 and col=13 are both past col_end=9 but still on the start line
        assert table.at(line=2, col=7) is not None
        assert table.at(line=2, col=7).name == "my_action"
        assert table.at(line=2, col=13) is not None
        assert table.at(line=2, col=13).name == "my_action"

    def test_at_middle_line_any_col_finds_symbol(self):
        """Clicking on a middle line (not start/end) finds the multiline symbol at any column."""
        from nemantix.core.node import ActionBlock, MicroPrompt

        block = ActionBlock(
            name="my_action",
            prompt=MicroPrompt(prompt="p", meta=self._multiline_fm(1, 1)),
            action_inputs=[],
            action_outputs=[],
            body=None,
            meta=self._multiline_fm(2, 10, col_start=1, col_end=9),
        )
        table = SymbolTableBuilder().build([block])
        assert table.at(line=5, col=50) is not None
        assert table.at(line=5, col=50).name == "my_action"

    def test_at_narrower_symbol_wins_over_multiline(self):
        """A single-line symbol inside a multiline block takes priority over the block."""
        from nemantix.core.node import ActionBlock, ActionInput, MicroPrompt

        inp = ActionInput(
            name="x",
            required=True,
            default=None,
            prompt=None,
            meta={"file_meta": FileMeta(line=(5, 5), column=(5, 6))},
        )
        block = ActionBlock(
            name="my_action",
            prompt=MicroPrompt(prompt="p", meta=self._multiline_fm(1, 1)),
            action_inputs=[inp],
            action_outputs=[],
            body=None,
            meta=self._multiline_fm(2, 10, col_start=1, col_end=9),
        )
        table = SymbolTableBuilder().build([block])
        sym = table.at(line=5, col=5)
        assert sym is not None
        assert sym.name == "x"


class TestRepeatBlocks:
    def test_repeat_times_without_as_does_not_crash(self):
        block = RepeatTimesBlock(times=3, as_vars=None, meta=_meta(1))
        builder = SymbolTableBuilder()
        builder.build([block])  # must not raise TypeError

    def test_repeat_each_without_as_does_not_crash(self):
        block = RepeatEachBlock(each=_var("items", 1), as_vars=None, meta=_meta(1))
        builder = SymbolTableBuilder()
        builder.build([block])  # must not raise TypeError

    def test_repeat_times_with_as_defines_variable(self):
        block = RepeatTimesBlock(times=3, as_vars=["i"], meta=_meta(1))
        builder = SymbolTableBuilder()
        table = builder.build([block])
        assert table.global_scope.lookup("i") is not None


# ---------------------------------------------------------------------------
# Group: forward-reference resolution (second pass in build)
# ---------------------------------------------------------------------------


class TestForwardReferenceResolution:
    """Action/variable used before it is defined must appear in sym.references."""

    def _action(self, name: str, line: int) -> ActionBlock:
        return ActionBlock(
            name=name,
            prompt=MicroPrompt(prompt="p", meta=_meta(line)),
            action_inputs=[],
            action_outputs=[],
            body=None,
            meta=_meta(line),
        )

    def test_do_before_action_definition_resolves_reference(self):
        """do my_action (line 1) then action my_action (line 2): must record 1 ref."""
        call = _do("my_action", line=1)
        defn = self._action("my_action", line=2)
        builder = SymbolTableBuilder()
        table = builder.build([call, defn])
        sym = table.global_scope.lookup("my_action")
        assert sym is not None
        assert len(sym.references) == 1

    def test_do_before_action_is_in_index(self):
        """The resolved forward reference must be findable via at()."""
        call = _do("my_action", line=1)
        defn = self._action("my_action", line=2)
        table = SymbolTableBuilder().build([call, defn])
        found = table.at(1, 1)
        assert found is not None
        assert found.name == "my_action"

    def test_unresolved_calls_cleared_for_resolved_symbol(self):
        """After resolution, unresolved_calls must not contain the resolved name."""
        call = _do("my_action", line=1)
        defn = self._action("my_action", line=2)
        builder = SymbolTableBuilder()
        builder.build([call, defn])
        names = [n for n, *_ in builder.unresolved_calls]
        assert "my_action" not in names

    def test_truly_undefined_symbol_stays_unresolved(self):
        """Do-statement referencing a non-existent action must remain in unresolved_calls."""
        call = _do("ghost_action", line=1)
        builder = SymbolTableBuilder()
        builder.build([call])
        names = [n for n, *_ in builder.unresolved_calls]
        assert "ghost_action" in names
