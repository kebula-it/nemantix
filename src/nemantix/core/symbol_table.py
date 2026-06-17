"""Static analysis of NXS/NXC/NXV ASTs: Symbol, Scope, SymbolTable, SymbolTableBuilder."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol

from nemantix.core.node import (
    ActionBlock,
    Assignment,
    BinaryOperation,
    BlockStatement,
    BuiltinFunction,
    BuiltinFunctionEnum,
    CallableTypeEnum,
    Collection,
    ConditionBlock,
    Deliberate,
    DoStatement,
    ElifBlock,
    ElseBlock,
    FileMeta,
    Frame,
    IfBlock,
    ImportToolsetStatement,
    NodeMeta,
    PlanBlock,
    PythonToolDeclaration,
    RepeatEachBlock,
    RepeatTimesBlock,
    RepeatUntilBlock,
    RepeatWhileBlock,
    Return,
    SimilarityOperation,
    Slot,
    Statement,
    UnaryOperation,
    Variable,
)


class SymbolKind(Enum):
    VARIABLE = auto()
    DELIBERATE = auto()
    ACTION = auto()
    FRAME = auto()
    SLOT = auto()
    TOOL = auto()
    IMPORT = auto()
    BUILTIN = auto()


@dataclass
class Symbol:
    name: str
    kind: SymbolKind
    defined_at: FileMeta
    references: list[FileMeta] = field(default_factory=list)
    signature: str | None = None
    description: str | None = None


class Scope:
    def __init__(self, parent: Scope | None = None) -> None:
        self.parent = parent
        self._symbols: dict[str, Symbol] = {}

    def define(self, symbol: Symbol) -> None:
        self._symbols[symbol.name] = symbol

    def lookup(self, name: str) -> Symbol | None:
        sym = self._symbols.get(name)
        if sym is not None:
            return sym
        return self.parent.lookup(name) if self.parent else None

    def local(self, name: str) -> Symbol | None:
        return self._symbols.get(name)

    @property
    def symbols(self) -> dict[str, Symbol]:
        return dict(self._symbols)


class SymbolTable:
    def __init__(
        self, global_scope: Scope, index: list[tuple[FileMeta, Symbol]]
    ) -> None:
        self._global = global_scope
        self._index = index

    def lookup(self, name: str, scope: Scope) -> Symbol | None:
        return scope.lookup(name)

    def at(self, line: int, col: int) -> Symbol | None:
        best: Symbol | None = None
        best_span = float("inf")
        for fm, sym in self._index:
            if not (fm.line[0] <= line <= fm.line[1]):
                continue
            if fm.line[0] == fm.line[1]:
                # single-line: strict column check
                if not (fm.column[0] <= col <= fm.column[1]):
                    continue
            else:
                # multi-line: column[0] is start-col on first line,
                # column[1] is end-col on last line; middle lines are unrestricted
                if line == fm.line[0] and col < fm.column[0]:
                    continue
                if line == fm.line[1] and col > fm.column[1]:
                    continue
            span = (fm.line[1] - fm.line[0]) * 100_000 + (fm.column[1] - fm.column[0])
            if span < best_span:
                best_span = span
                best = sym
        return best

    @property
    def global_scope(self) -> Scope:
        return self._global


class _HasMeta(Protocol):
    meta: dict


def _build_action_sig(inputs: list, outputs: list) -> str | None:
    sections: list[str] = []
    if inputs:
        inp_lines: list[str] = []
        for inp in inputs:
            name = getattr(inp, "name", None) or "?"
            required = getattr(inp, "required", True)
            default = getattr(inp, "default", None)
            req_str = "required" if required else "optional"
            if not required and default is not None:
                req_str += f", default: {getattr(default, 'value', str(default))}"
            p = mp.prompt if (mp := getattr(inp, "prompt", None)) else None
            inp_lines.append(f"- **{name}** ({req_str})" + (f": {p}" if p else ""))
        sections.append("Inputs:\n" + "\n".join(inp_lines))
    if outputs:
        out_lines: list[str] = []
        for out in outputs:
            name = getattr(out, "name", None) or "?"
            p = mp.prompt if (mp := getattr(out, "prompt", None)) else None
            out_lines.append(f"- **{name}**" + (f": {p}" if p else ""))
        sections.append("Outputs:\n" + "\n".join(out_lines))
    return "\n\n".join(sections) if sections else None


def _file_meta(node: _HasMeta) -> FileMeta | None:
    nm = node.meta.get("node_meta")
    if isinstance(nm, NodeMeta):
        return nm.file_meta
    fm = node.meta.get("file_meta")
    if isinstance(fm, FileMeta):
        return fm
    return None


def _completion_initial(node: _HasMeta) -> str | None:
    """Returns the initial completion state from @completion (e.g. '_', 'frozen')."""
    nm = node.meta.get("node_meta")
    if not isinstance(nm, NodeMeta):
        return None
    for ann in nm.annotations:
        if ann.name == "completion":
            return str(ann.value).split("->")[0].strip()
    return None


_BUILTIN_DOCS: dict[str, tuple[str | None, str | None]] = {
    "exists": (
        "exists(value) → bool",
        "Returns `true` if *value* is not `null`. Extra arguments are ignored.",
    ),
    "coalesce": (
        "coalesce(a, b, ...) → any",
        "Returns the first non-`null` argument. Accepts any number of positional or keyword arguments.",
    ),
    "print": (
        "print(*args, **kwargs)",
        "Prints one or more values to standard output, separated by spaces. `null` values are displayed as `<NONE>`.",
    ),
    "type": (
        "type(value) → str",
        "Returns the runtime type name of *value*: `none`, `num`, `str`, `bool`, `struct`, `doc`, or `opaque`.",
    ),
    "llm": (
        "llm(prompt) → str",
        "Sends *prompt* to the configured LLM and returns the response.",
    ),
    "size": (
        "size(value, ...) → num",
        "With one argument: returns the length of a string or struct, or `0` for other types. "
        "With multiple arguments: returns the count of arguments.",
    ),
    "substring": (
        "substring(s, start=0, end=null) → str",
        "Returns the slice `s[start:end]`. *end* defaults to the length of *s*. "
        "Non-numeric *start*/*end* values default to `0` and `len(s)` respectively.",
    ),
    "bool": (
        "bool(value) → bool | null",
        "Soft boolean conversion. Returns `null` if *value* is `null`. "
        "Structs: `true` if non-empty. Otherwise delegates to `to_bool`.",
    ),
    "num": (
        "num(value) → num | null",
        "Soft numeric conversion. Returns `null` if *value* is `null` or a collection type. "
        "Otherwise delegates to `to_num`.",
    ),
    "str": (
        "str(value) → str | null",
        "Soft string conversion. Returns `null` if *value* is `null`. Otherwise delegates to `to_str`.",
    ),
    "to_bool": (
        "to_bool(value) → bool",
        "Explicit boolean conversion. Numbers: `false` if `0`. "
        "Strings: parses `'true'`/`'false'`/`'none'`; non-empty strings are `true`. Never returns `null`.",
    ),
    "to_num": (
        "to_num(value) → num",
        "Explicit numeric conversion. Booleans become `1`/`0`. "
        "Strings are parsed as integer or float; unrecognised strings return `0`. Never returns `null`.",
    ),
    "to_str": (
        "to_str(value) → str",
        "Explicit string conversion. Booleans become lowercase `'true'`/`'false'`. `null` becomes `''`. Never returns `null`.",
    ),
    "sin": (
        "sin(x) → num",
        "Returns the sine of *x*, where *x* is in radians.",
    ),
    "cos": (
        "cos(x) → num",
        "Returns the cosine of *x*, where *x* is in radians.",
    ),
    "sqrt": (
        "sqrt(x) → num",
        "Returns the square root of *x*.",
    ),
}


class SymbolTableBuilder:
    def __init__(self) -> None:
        self._global = Scope()
        self._current = self._global
        self._index: list[tuple[FileMeta, Symbol]] = []
        self.unresolved: list[tuple[str, FileMeta]] = []
        self.unresolved_calls: list[tuple[str, FileMeta, bool]] = []
        self.unresolved_schemas: list[tuple[str, FileMeta]] = []
        self.duplicates: list[tuple[str, FileMeta]] = []
        self._lenient = False
        self._seed_builtins()

    def _seed_builtins(self) -> None:
        _sentinel = FileMeta(line=(0, 0), column=(0, 0))
        for enum_val in BuiltinFunctionEnum:
            sig, desc = _BUILTIN_DOCS.get(enum_val.value, (None, None))
            self._global.define(
                Symbol(
                    name=enum_val.value,
                    kind=SymbolKind.BUILTIN,
                    defined_at=_sentinel,
                    signature=sig,
                    description=desc,
                )
            )

    def build(self, statements: list[Statement]) -> SymbolTable:
        for stmt in statements:
            self._visit(stmt)
        self._resolve_forward_refs()
        return SymbolTable(self._global, self._index)

    def _resolve_forward_refs(self) -> None:
        """Second pass: resolve symbols that were used before being defined."""
        remaining: list[tuple[str, FileMeta]] = []
        for name, fm in self.unresolved:
            sym = self._global.lookup(name)
            if sym is not None:
                sym.references.append(fm)
                self._index.append((fm, sym))
            else:
                remaining.append((name, fm))
        self.unresolved = remaining

        remaining_calls: list[tuple[str, FileMeta, bool]] = []
        for name, fm, lenient in self.unresolved_calls:
            sym = self._global.lookup(name)
            if sym is not None:
                sym.references.append(fm)
                self._index.append((fm, sym))
            else:
                remaining_calls.append((name, fm, lenient))
        self.unresolved_calls = remaining_calls

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _define(self, name: str, kind: SymbolKind, fm: FileMeta) -> Symbol:
        if self._current.local(name) is not None:
            self.duplicates.append((name, fm))
        sym = Symbol(name=name, kind=kind, defined_at=fm)
        self._current.define(sym)
        self._index.append((fm, sym))
        return sym

    def _reference(self, name: str, fm: FileMeta) -> None:
        sym = self._current.lookup(name)
        if sym is not None:
            sym.references.append(fm)
            self._index.append((fm, sym))
        else:
            self.unresolved.append((name, fm))

    def _call_reference(self, name: str, fm: FileMeta) -> None:
        sym = self._current.lookup(name)
        if sym is not None:
            sym.references.append(fm)
            self._index.append((fm, sym))
        else:
            self.unresolved_calls.append((name, fm, self._lenient))

    def _define_producing(self, expr: object) -> None:
        if expr is None:
            return
        if isinstance(expr, Variable):
            fm = _file_meta(expr)
            if fm and expr.name:
                existing = self._current.lookup(expr.name)
                if existing is not None and existing.kind == SymbolKind.VARIABLE:
                    existing.references.append(fm)
                    self._index.append((fm, existing))
                else:
                    self._define(expr.name, SymbolKind.VARIABLE, fm)
        elif isinstance(expr, Collection):
            if isinstance(expr.value, list):
                for item in expr.value:
                    self._define_producing(item)

    def _push(self) -> None:
        self._current = Scope(parent=self._current)

    def _pop(self) -> None:
        if self._current.parent:
            self._current = self._current.parent

    # ------------------------------------------------------------------
    # Expression traversal
    # ------------------------------------------------------------------

    def _visit_expression(self, node: object) -> None:
        if node is None:
            return
        if isinstance(node, Variable):
            fm = _file_meta(node)
            if fm and node.name:
                self._reference(node.name, fm)
        elif isinstance(node, BinaryOperation):
            self._visit_expression(node.first)
            self._visit_expression(node.second)
        elif isinstance(node, UnaryOperation):
            self._visit_expression(node.operand)
        elif isinstance(node, BuiltinFunction):
            fm = _file_meta(node)
            if fm:
                sym = self._current.lookup(node.function.value)
                if sym is not None:
                    sym.references.append(fm)
                    self._index.append((fm, sym))
            for arg in node.args:
                self._visit_expression(arg)
        elif isinstance(node, SimilarityOperation):
            self._visit_expression(node.first)
            self._visit_expression(node.second)
        elif isinstance(node, Collection):
            if isinstance(node.value, list):
                for item in node.value:
                    self._visit_expression(item)
        # SingleValue, MetaExpression: no Variable references to recurse into

    # ------------------------------------------------------------------
    # Statement visitor dispatch
    # ------------------------------------------------------------------

    def _visit(self, node: Statement) -> None:
        if isinstance(node, Assignment):
            self._visit_assignment(node)
        elif isinstance(node, Deliberate):
            self._visit_deliberate(node)
        elif isinstance(node, ActionBlock):
            self._visit_action_block(node)
        elif isinstance(node, Frame):
            self._visit_frame(node)
        elif isinstance(node, ImportToolsetStatement):
            self._visit_import_toolset(node)
        elif isinstance(node, PythonToolDeclaration):
            self._visit_python_tool(node)
        elif isinstance(node, RepeatEachBlock):
            self._visit_repeat_each(node)
        elif isinstance(node, RepeatTimesBlock):
            self._visit_repeat_times(node)
        elif isinstance(node, RepeatWhileBlock):
            self._visit_repeat_while(node)
        elif isinstance(node, RepeatUntilBlock):
            self._visit_repeat_until(node)
        elif isinstance(node, DoStatement):
            self._visit_do(node)
        elif isinstance(node, ConditionBlock):
            self._visit_condition_block(node)
        elif isinstance(node, IfBlock):
            self._visit_if_block(node)
        elif isinstance(node, ElifBlock):
            self._visit_elif_block(node)
        elif isinstance(node, ElseBlock):
            self._visit_else_block(node)
        elif isinstance(node, Return):
            self._visit_return(node)
        elif isinstance(node, PlanBlock):
            self._visit_plan_block(node)
        elif isinstance(node, BlockStatement):
            self._visit_block(node)

    def _visit_assignment(self, node: Assignment) -> None:
        fm = _file_meta(node)
        if fm and node.var and node.var.name:
            name = node.var.name
            existing = self._current.lookup(name)
            if existing is not None and existing.kind == SymbolKind.VARIABLE:
                existing.references.append(fm)
                self._index.append((fm, existing))
            else:
                self._define(name, SymbolKind.VARIABLE, fm)
        if node.value is not None:
            self._visit_expression(node.value)

    def _visit_variable_ref(self, node: Variable) -> None:
        fm = _file_meta(node)
        if fm and node.name:
            self._reference(node.name, fm)

    def _visit_deliberate(self, node: Deliberate) -> None:
        fm = _file_meta(node)
        if fm and node.name:
            plan = node.get_plan()
            sig = _build_action_sig(plan.input, plan.output) if plan else None
            sym = self._define(node.name, SymbolKind.DELIBERATE, fm)
            sym.signature = sig
        initial = _completion_initial(node)
        was_lenient = self._lenient
        if initial is not None and initial != "frozen":
            self._lenient = True
        self._push()
        for action in node.generated_actions:
            self._visit(action)
        for child in node.children or []:
            self._visit(child)
        self._pop()
        self._lenient = was_lenient

    def _visit_action_block(self, node: ActionBlock) -> None:
        fm = _file_meta(node)
        if fm and node.name:
            sym = self._define(node.name, SymbolKind.ACTION, fm)
            sym.signature = _build_action_sig(node.input, node.output)
            prompt = getattr(node, "prompt", None)
            sym.description = getattr(prompt, "prompt", None) if prompt else None
        self._push()
        for inp in node.input:
            inp_fm = _file_meta(inp)
            if inp_fm and inp.name:
                self._define(inp.name, SymbolKind.VARIABLE, inp_fm)
        for out in node.output:
            out_fm = _file_meta(out)
            if out_fm and out.name:
                self._define(out.name, SymbolKind.VARIABLE, out_fm)
        for child in node.children or []:
            self._visit(child)
        self._pop()

    def _visit_plan_block(self, node: PlanBlock) -> None:
        initial = _completion_initial(node)
        was_lenient = self._lenient
        if initial is not None and initial != "frozen":
            self._lenient = True
        for inp in node.input:
            inp_fm = _file_meta(inp)
            if inp_fm and inp.name:
                self._define(inp.name, SymbolKind.VARIABLE, inp_fm)
        for out in node.output:
            out_fm = _file_meta(out)
            if out_fm and out.name:
                self._define(out.name, SymbolKind.VARIABLE, out_fm)
        for child in node.children or []:
            self._visit(child)
        self._lenient = was_lenient

    def _visit_frame(self, node: Frame) -> None:
        fm = _file_meta(node)
        if fm and node.name:
            self._define(node.name, SymbolKind.FRAME, fm)
        self._push()
        for child in node.children or []:
            if isinstance(child, Slot):
                self._visit_slot(child)
            else:
                self._visit(child)
        self._pop()

    def _visit_slot(self, node: Slot) -> None:
        fm = _file_meta(node)
        if fm and node.name:
            self._define(node.name, SymbolKind.SLOT, fm)

    def _visit_import_toolset(self, node: ImportToolsetStatement) -> None:
        fm = _file_meta(node)
        if fm:
            name = node.alias if node.alias else node.name
            self._define_or_ref_tool(name, fm)

    def _visit_python_tool(self, node: PythonToolDeclaration) -> None:
        fm = _file_meta(node)
        if fm and node.name:
            self._define_or_ref_tool(node.name, fm)

    def _define_or_ref_tool(self, name: str, fm: FileMeta) -> None:
        existing = self._current.lookup(name)
        if existing is not None and existing.kind == SymbolKind.TOOL:
            existing.references.append(fm)
            self._index.append((fm, existing))
        else:
            self._define(name, SymbolKind.TOOL, fm)

    def _visit_repeat_each(self, node: RepeatEachBlock) -> None:
        fm = _file_meta(node)
        if fm:
            for var_name in node.as_vars or []:
                self._define(var_name, SymbolKind.VARIABLE, fm)
        self._visit_expression(node.each)
        self._push()
        for child in node.children or []:
            self._visit(child)
        self._pop()

    def _visit_repeat_times(self, node: RepeatTimesBlock) -> None:
        fm = _file_meta(node)
        if fm:
            for var_name in node.as_vars or []:
                self._define(var_name, SymbolKind.VARIABLE, fm)
        self._push()
        for child in node.children or []:
            self._visit(child)
        self._pop()

    def _visit_repeat_while(self, node: RepeatWhileBlock) -> None:
        self._visit_expression(node.condition)
        self._push()
        for child in node.children or []:
            self._visit(child)
        self._pop()

    def _visit_repeat_until(self, node: RepeatUntilBlock) -> None:
        self._visit_expression(node.condition)
        self._push()
        for child in node.children or []:
            self._visit(child)
        self._pop()

    def _visit_do(self, node: DoStatement) -> None:
        fm = _file_meta(node)
        if fm and node.name:
            is_tool = node.callable_type == CallableTypeEnum.TOOL or "." in node.name
            lookup_name = node.name.split(".")[0] if is_tool else node.name
            self._call_reference(lookup_name, fm)
        self._visit_expression(node.using)
        self._define_producing(node.producing)
        if isinstance(node.producing_schema, str) and fm:
            sym = self._current.lookup(node.producing_schema)
            if sym is None:
                self.unresolved_schemas.append((node.producing_schema, fm))

    def _visit_condition_block(self, node: ConditionBlock) -> None:
        for child in node.children or []:
            self._visit(child)

    def _visit_if_block(self, node: IfBlock) -> None:
        self._visit_expression(node.condition)
        self._push()
        for child in node.children or []:
            self._visit(child)
        self._pop()

    def _visit_elif_block(self, node: ElifBlock) -> None:
        self._visit_expression(node.condition)
        self._push()
        for child in node.children or []:
            self._visit(child)
        self._pop()

    def _visit_else_block(self, node: ElseBlock) -> None:
        self._push()
        for child in node.children or []:
            self._visit(child)
        self._pop()

    def _visit_return(self, node: Return) -> None:
        for expr in node.val:
            self._visit_expression(expr)

    def _visit_block(self, node: BlockStatement) -> None:
        self._push()
        for child in node.children or []:
            self._visit(child)
        self._pop()
