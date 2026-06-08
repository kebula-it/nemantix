from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from lark import (
    Lark,
    ParseTree,
    Token,
    Transformer,
    Tree,
    UnexpectedCharacters,
    UnexpectedToken,
    v_args,
)
from lark.visitors import Visitor_Recursive

from nemantix.common.logger import get_package_logger
from nemantix.core.custom_types import PathLike
from nemantix.core.exceptions import NemantixParserException
from nemantix.core.node import (
    ActionBlock,
    ActionInput,
    ActionOutput,
    Annotation,
    Assignment,
    BinaryOperation,
    BinaryOperationEnum,
    BlockStatement,
    Break,
    BuiltinFunction,
    BuiltinFunctionEnum,
    CallableTypeEnum,
    Collection,
    ConditionBlock,
    Continue,
    Deliberate,
    DoStatement,
    ElifBlock,
    ElseBlock,
    Expression,
    FileMeta,
    Frame,
    FrameApplyEnum,
    IfBlock,
    ImportToolsetStatement,
    LeafStatement,
    MetaExpression,
    MicroPrompt,
    NodeMeta,
    PlanBlock,
    PythonToolDeclaration,
    RepeatBlock,
    RepeatEachBlock,
    RepeatTimesBlock,
    RepeatUntilBlock,
    RepeatWhileBlock,
    Require,
    Return,
    SchemedCollection,
    SimilarityEnum,
    SimilarityOperation,
    SingleValue,
    Slot,
    SlotTypesEnum,
    Statement,
    UnaryOperation,
    UnaryOperationEnum,
    Value,
    Variable,
    VariableTypeEnum,
    builtin_func_map,
    map_sim_qual_kw,
    slot_types_map,
)

logger = get_package_logger(__name__)

logger.setLevel("DEBUG")
FSTRING_START_EXPR_PATTERN = r"(?<!\\)\["
RESERVED_VAR_NAMES = ['_', '__', 'when', 'from', 'use', 'as', 'with',
                      'include', 'guidelines', 'if', 'elif', 'else',
                      'repeat', 'while', 'until', 'each', 'do',
                      'return', 'break', 'continue', 'using',
                      'producing', 'required', 'optional', 'true',
                      'false', 'none', 'undefined', 'drafted',
                      'frozen', '__deliberate', '__guidelines', '__plan',
                      '__action', '__body', '__do', '__in', '__out', '__repeat',
                      '__if', '__toolset', '__use', '__frame']
IMPORTED_TOOLSETS = {}

@dataclass
class AsFrame:
    """Simulate node to temporarily store file meta of AS clause in producing clause (do)"""
    value: str
    meta: dict


# =============================================================================
# Utilities
# =============================================================================
def get_grammar_path() -> Path:
    """Return the grammar file path shipped with this module."""
    return Path(__file__).parent / "nxs_v2_grammar.lark"


def read_grammar() -> str:
    """Read and return the Lark grammar as a string."""
    return get_grammar_path().read_text(encoding="utf-8")


def _clean_prompt(text: str) -> str:
    """Normalize prompt content by stripping NXS prompt markers."""
    text = text.strip()

    # >>> ... <<< (block)
    if text.startswith(">>>") and text.endswith("<<<"):
        return text[3:-3].strip()

    # >> ... << (inline) OR >> ... (line prompt)
    if text.startswith(">>"):
        if text.endswith("<<"):
            return text[2:-2].strip()
        return text[2:].strip()

    return text


# Lazily-initialized parser for f-string fragments (start_fstr).
_FSTRING_PARSER: Optional[Lark] = None
_FRAME_PARSER: Optional[Lark] = None
_STMT_PARSER: Optional[Lark] = None


def _get_fstring_parser() -> Lark:
    global _FSTRING_PARSER
    if _FSTRING_PARSER is None:
        _FSTRING_PARSER = Lark(read_grammar(), start=["start_fstr"],
                               parser="lalr", propagate_positions=True)

    assert isinstance(_FSTRING_PARSER, Lark)
    return _FSTRING_PARSER


def _get_frame_parser() -> Lark:
    global _FRAME_PARSER
    if _FRAME_PARSER is None:
        _FRAME_PARSER = Lark(read_grammar(), start=["start_frame"],
                             parser="lalr", propagate_positions=True)

    assert isinstance(_FRAME_PARSER, Lark)
    return _FRAME_PARSER


def _get_stmt_parser() -> Lark:
    global _STMT_PARSER
    if _STMT_PARSER is None:
        _STMT_PARSER = Lark(read_grammar(), start=["start_stmt"],
                            parser="lalr", propagate_positions=True)

    assert isinstance(_STMT_PARSER, Lark)
    return _STMT_PARSER


_ESCAPE_MAP: Dict[str, str] = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "\\": "\\",
    '"': '"',
    "'": "'",
    "[": "[",
    "]": "]",
}


def _unescape_string_literal(s: str) -> str:
    """Unescape common backslash escapes used by the DSL."""
    out: List[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c != "\\":  # normal char
            out.append(c)
            i += 1
            continue

        # trailing backslash
        if i == len(s) - 1:
            out.append("\\")
            i += 1
            continue

        nxt = s[i + 1]
        mapped = _ESCAPE_MAP.get(nxt)
        if mapped is not None:
            out.append(mapped)
        else:
            # Unknown escape: preserve verbatim
            out.append("\\")
            out.append(nxt)
        i += 2

    return "".join(out)


def _is_escaped(pos: int, s: str) -> bool:
    """True if s[pos] is escaped by an odd number of backslashes right before it."""
    backslashes = 0
    i = pos - 1
    while i >= 0 and s[i] == "\\":
        backslashes += 1
        i -= 1
    return (backslashes % 2) == 1


def parse_escaped_string(string: str, *, file: Optional[Path] = None) -> List[object]:
    """
    Parse DSL f-strings with [expr] segments into literal chunks + AST nodes.

    Returns a list containing:
      - literal string chunks (unescaped)
      - AST nodes for each parsed [expr]
    """
    counter = 0
    expr_start: Optional[int] = None
    literal_start = 0
    out_list: List[object] = []

    i = 0
    while i < len(string):
        c = string[i]

        if c == "[" and not _is_escaped(i, string):
            if counter == 0:
                # Emit preceding literal
                if literal_start < i:
                    out_list.append(_unescape_string_literal(string[literal_start:i]))
                expr_start = i
            counter += 1

        elif c == "]" and not _is_escaped(i, string):
            counter -= 1
            if counter < 0:
                raise NemantixParserException("Unbalanced ']' in formatted string")

            if counter == 0:
                if expr_start is None:
                    raise NemantixParserException("Internal f-string parse error")

                expr_end = i
                expr_text = string[expr_start: expr_end + 1]

                tree = _get_fstring_parser().parse(expr_text)
                tr = AstTransformer()
                # Propagate file context to inner transformer (for FileMeta)
                tr._current_file = file
                transformed = tr.transform(tree)

                out_list.append(transformed.children[0])

                expr_start = None
                literal_start = i + 1

        i += 1

    if counter != 0:
        raise NemantixParserException("Unbalanced '[' in formatted string")

    if literal_start < len(string):
        out_list.append(_unescape_string_literal(string[literal_start:]))

    return out_list


# =============================================================================
# Transformer: Lark -> AST
# =============================================================================
# noinspection PyPep8Naming,PyMethodMayBeStatic,PyUnusedLocal
class AstTransformer(Transformer):
    """
    Convert the Lark parse tree into AST nodes (see nemantix.core.node).

    Notes:
    - This transformer keeps the AST fairly lightweight.
    - Metadata is stored under node.meta as {"file_meta": FileMeta, "node_meta": NodeMeta|None}.
    """

    def __init__(self):
        super().__init__()
        self._current_file: Optional[Path] = None
        self._frame_names: list[str] = []  # Known frame names referenced by slot types

    def transform_with_file_info(self, tree: Tree, file: PathLike):
        """Transform a parse tree, attaching 'file' to generated FileMeta."""
        self._current_file = file
        return self.transform(tree)

    def _build_file_meta(self, lark_meta) -> FileMeta:
        return FileMeta(
            (lark_meta.line, lark_meta.end_line),
            (lark_meta.column, lark_meta.end_column),
            file=self._current_file)

    # ------------------------------------------------------------------
    # start / toolset / deliberate / include
    # ------------------------------------------------------------------
    def start(self, items):
        # start: toolset* deliberate+
        return [i for i in items if i is not None]

    @v_args(meta=True)
    def require(self, meta, items):
        return Require(items[0].value, meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    # ---- toolset ----
    def toolset_name(self, items):
        return str(items[0])

    def toolset_body(self, items):
        # toolset_body: (prompt_block | prompt_line)
        return items[0]

    @v_args(meta=True)
    def toolset(self, meta, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        name = items[0]
        prompt = items[1]
        return PythonToolDeclaration(
            name=name,
            prompt=prompt,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta},
        )

    # ---- deliberate + import ----
    def deliberate_name(self, items) -> str:
        # deliberate_name: CNAME
        return str(items[0])

    def deliberate_condition(self, items):
        # deliberate_condition: prompt_block
        # prompt_block is already transformed into MicroPrompt
        return items[0]

    def ident_list(self, items):
        # ident_list: CNAME ("," CNAME)*
        return [str(t) for t in items if isinstance(t, Token)]

    def wildcard(self, items):
        return "*"

    def import_inline(self, items):
        return items[0] if items else []

    def import_block(self, items):
        # import_block: ":" import_targets+ (_END_BLOCK | _END_USE)
        targets: List[str] = []
        has_wildcard = False
        for it in items:
            if isinstance(it, Token):
                continue
            if it == "*":
                has_wildcard = True
            elif isinstance(it, list):
                targets.extend(it)
        return "*" if has_wildcard else targets

    def as_alias(self, items):
        return "alias", items[0].value

    def with_args(self, items):
        return "args", items[0]

    @v_args(meta=True)
    def import_toolset(self, meta, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        toolset_name = items[0]
        tools = items[-1]
        alias = None
        args = None

        for it in items:
            if isinstance(it, tuple):
                if it[0] == "alias":
                    alias = it[1]
                elif it[0] == "args":
                    args = it[1]

        # keep track of imported toolsets
        # TODO: check tool name as key
        for tool in tools:
            IMPORTED_TOOLSETS[tool] = toolset_name

        return ImportToolsetStatement(
            name=toolset_name,
            elements=tools,
            alias=alias,
            args=args,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta},
        )

    def import_stmt(self, items):
        return items[0]

    @v_args(meta=True)
    def guidelines(self, meta, items):
        """
        guidelines: _GUIDELINES ":" (prompt_line | prompt_block)+ (_END_BLOCK | _END_GUIDELINES)
        Returns a single MicroPrompt (multi-line concatenation).
        """
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        prompts = [it for it in items if isinstance(it, MicroPrompt)]
        if not prompts:
            return None
        if len(prompts) == 1:
            return prompts[0]
        text = "\n".join(p.prompt for p in prompts)
        return MicroPrompt(text, meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta})

    def plan_type(self, items):
        return items[0].value

    def plan_qualifier(self, items):
        return items

    @v_args(meta=True)
    def plan(self, meta, items):
        """
        action:
          _ACTION (action_name prompt_block? | prompt_block) ":"
          action_ins? action_out? action_body (_END_BLOCK | _END_ACTION)
        -> ActionBlock
        """
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        non_tokens: List[Any] = [i for i in items if not isinstance(i, Token)]
        if not non_tokens:
            return None

        ins: List[ActionInput] = []
        outs: List[ActionOutput] = []
        body: List[Statement] = []

        for obj in non_tokens:
            if isinstance(obj, str):
                continue

            elif isinstance(obj, list) and obj:
                first_elem = obj[0]

                if isinstance(first_elem, ActionInput):
                    ins = obj
                elif isinstance(first_elem, ActionOutput):
                    outs = obj
                elif isinstance(first_elem, Statement):
                    body = obj

        if all([i is None for i in ins]):
            ins = []

        if all([o is None for o in outs]):
            outs = []

        file_meta = FileMeta((meta.line, meta.end_line), (meta.column, meta.end_column), file=self._current_file)
        return PlanBlock(
            action_inputs=ins,
            action_outputs=outs,
            body=body,
            meta={"file_meta": file_meta, "node_meta": node_meta})

    @v_args(meta=True)
    def deliberate(self, meta, items):
        """
        deliberate:
          _DELIBERATE deliberate_name? _WHEN deliberate_condition ":" import* guidelines?
            plan (_END_BLOCK | _END_DELIBERATE)
        """
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None

        name: Optional[str] = None
        condition: Optional[MicroPrompt] = None
        guidelines: Optional[MicroPrompt] = None
        plan_blocks: Optional[PlanBlock] = None
        actions: List[ActionBlock] = []

        for obj in items:
            if isinstance(obj, Token):
                continue
            if isinstance(obj, str) and name is None:
                name = obj
            elif isinstance(obj, MicroPrompt) and condition is None:
                condition = obj
            elif isinstance(obj, MicroPrompt):
                guidelines = obj
            elif isinstance(obj, PlanBlock):
                plan_blocks = obj
            elif isinstance(obj, ActionBlock):
                actions.append(obj)

        if condition is None:
            raise NemantixParserException(f"Missing deliberate condition in: {items!r}")

        assert isinstance(name, str)

        return Deliberate(
            name=name,
            when=condition,
            guidelines=guidelines,
            plan=plan_blocks,
            generated_actions=actions,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta})

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------
    @v_args(meta=True)
    def prompt_stmt(self, meta, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        token: Token = items[0]
        return MicroPrompt(
            _clean_prompt(token.value),
            meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta},
        )

    @v_args(meta=True)
    def prompt_block(self, meta, items):
        token: Token = items[0]
        return MicroPrompt(
            _clean_prompt(token.value),
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    @v_args(meta=True)
    def prompt_line(self, meta, items):
        token: Token = items[0]
        return MicroPrompt(
            _clean_prompt(token.value),
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    @v_args(meta=True)
    def prompt_line_variable(self, meta, items):
        token: Token = items[0]
        return MicroPrompt(
            _clean_prompt(token.value),
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    # ------------------------------------------------------------------
    # Variables / expressions (minimal mapping)
    # ------------------------------------------------------------------
    def var_accessor(self, items):
        return str(items[0])

    @v_args(meta=True)
    def access_index(self, meta, items):
        return SingleValue(
            items[0].value,
            VariableTypeEnum.INT,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    @v_args(meta=True)
    def access_key(self, meta, items):
        return SingleValue(
            items[0].value,
            VariableTypeEnum.STRING,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    def access_expr(self, items):
        return items[0]

    def prompted_variable(self, items):
        # (variable prompt_block?) | prompt_block
        if isinstance(items[0], Variable):
            var = items[0]
            if len(items) > 1 and isinstance(items[1], MicroPrompt):
                var.prompt = items[1]
            return var
        return items[0]

    def group_expr(self, items):
        return items

    @v_args(meta=True)
    def math_func(self, meta, items):
        fn_name = items[0]
        if fn_name not in builtin_func_map:
            raise NemantixParserException(f"Undefined builtin function: \"{fn_name}(...)\"!")
        return BuiltinFunction(
            builtin_func_map[fn_name],
            args=items[1:],
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    @v_args(meta=True)
    def assign_var(self, meta, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        var = items[0]
        value = items[1]
        return Assignment(
            var=var,
            value=value,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta},
        )

    @v_args(meta=True)
    def number(self, meta, items):
        text = str(items[0])
        exp = None
        if "e" in text:
            text, exp = text.split("e")
        if "." in text:
            val = float(text) if not exp else float(text) * (10 ** int(exp))
            return SingleValue(val, VariableTypeEnum.FLOAT,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

        val = int(text) if not exp else int(text) * (10 ** int(exp))
        return SingleValue(val, VariableTypeEnum.INT,
                           meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def boolean(self, meta, items):
        item = items[0]
        if isinstance(item, Token):
            item = item.value == 'true'

        return SingleValue(item, VariableTypeEnum.BOOL,
                           meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    def string(self, items):
        return items[0]

    def ESCAPED_STRING(self, token):
        # token.value includes quotes
        val = token.value[1:-1]
        file_meta = FileMeta((token.line, token.end_line), (token.column, token.end_column), file=self._current_file)

        if re.search(FSTRING_START_EXPR_PATTERN, val):
            try:
                parts = parse_escaped_string(val)
            except Exception as e:
                raise NemantixParserException(f"Error in formatted string parsing at line {token.line}, "
                                              f"column {token.column} \n{token}\n{e}")
            return SingleValue(parts, VariableTypeEnum.FSTRING, meta={"file_meta": file_meta, "node_meta": None})

        return SingleValue(val, VariableTypeEnum.STRING, meta={"file_meta": file_meta, "node_meta": None})

    @v_args(meta=True)
    def variable(self, meta, items):
        prompt = None
        path = []
        name = None

        # Handle nested list structure and extract prompt if present
        if isinstance(items[0], list):
            if len(items) > 1:
                prompt = items[1]
            items = items[0]

        for it in items:
            if isinstance(it, Token):
                name = it.value
            elif isinstance(it, MicroPrompt):
                prompt = it
            elif isinstance(it, SingleValue):
                path.append(it)

        if name in RESERVED_VAR_NAMES:
            raise NemantixParserException(f'Cannot declare variable with reserved name "{name}"!')

        return Variable(
            name=name,
            prompt=prompt,
            path=path,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None}
        )

    def var_path(self, items):
        return items

    # ------------------------------------------------------------------
    # Binary ops
    # ------------------------------------------------------------------
    @staticmethod
    def _get_binary_operands(items):
        it0 = items[0][0] if isinstance(items[0], list) and len(items[0]) == 1 else items[0]
        it1 = items[1][0] if isinstance(items[1], list) and len(items[1]) == 1 else items[1]
        return it0, it1

    @v_args(meta=True)
    def add(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.ADD, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def sub(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.SUB, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def mul(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.MUL, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def div(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.DIV, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def mod(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.MOD, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def pow(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.POW, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def eq(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.EQ, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def neq(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.NE, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def lt(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.LT, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def gt(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.GT, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def lte(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.LTE, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def gte(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.GTE, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def op_or(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.LOGICAL_OR, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def op_xor(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.LOGICAL_XOR, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def op_and(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.LOGICAL_AND, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def op_fallback(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.FALLBACK, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def string_concat(self, meta, items):
        it0, it1 = self._get_binary_operands(items)
        return BinaryOperation(BinaryOperationEnum.CONCAT, it0, it1,
                               meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    # ------------------------------------------------------------------
    # Similarity ops
    # ------------------------------------------------------------------
    def sim_qual_kw(self, items):
        return items[0].type, items[0].value

    def sim_qual(self, items):
        if isinstance(items[0], tuple):
            return items[0]
        return items[0].type, items[0].value

    @v_args(meta=True)
    def sem_sim(self, meta, items):
        return SimilarityOperation(
            SimilarityEnum.SIM,
            qualifier=None,
            first=items[0],
            second=items[1],
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    @v_args(meta=True)
    def sem_sim_qual(self, meta, items):
        qualifier = (map_sim_qual_kw[items[1][0]], items[1][1])
        return SimilarityOperation(
            SimilarityEnum.SIM_QUAL,
            qualifier=qualifier,
            first=items[0],
            second=items[2],
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    @v_args(meta=True)
    def sem_in(self, meta, items):
        return SimilarityOperation(
            SimilarityEnum.SIM_RIGHT,
            qualifier=None,
            first=items[0],
            second=items[1],
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    @v_args(meta=True)
    def sem_in_qual(self, meta, items):
        qualifier = (map_sim_qual_kw[items[1][0]], items[1][1])
        return SimilarityOperation(
            SimilarityEnum.SIM_QUAL_RIGHT,
            qualifier=qualifier,
            first=items[0],
            second=items[2],
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    @v_args(meta=True)
    def sem_in_rev(self, meta, items):
        return SimilarityOperation(
            SimilarityEnum.SIM_LEFT,
            qualifier=None,
            first=items[0],
            second=items[1],
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    @v_args(meta=True)
    def sem_in_rev_qual(self, meta, items):
        qualifier = (map_sim_qual_kw[items[1][0]], items[1][1])
        return SimilarityOperation(
            SimilarityEnum.SIM_QUAL_LEFT,
            qualifier=qualifier,
            first=items[0],
            second=items[2],
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    # ------------------------------------------------------------------
    # Unary ops
    # ------------------------------------------------------------------
    @v_args(meta=True)
    def unary_plus(self, meta, items):
        return UnaryOperation(UnaryOperationEnum.POS, items[0],
                              meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def unary_minus(self, meta, items):
        return UnaryOperation(UnaryOperationEnum.NEG, items[0],
                              meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def op_not(self, meta, items):
        return UnaryOperation(UnaryOperationEnum.NOT, items[0],
                              meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    # ------------------------------------------------------------------
    # Lists / structures
    # ------------------------------------------------------------------
    @v_args(meta=True)
    def list_create(self, meta, items):
        return Collection(items, VariableTypeEnum.LIST,
                          meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    def list_key(self, items):
        return "key", items[0].value

    def list_literal_comma(self, items):
        return items

    def list_item_value(self, items):
        return items[0]

    @v_args(meta=True)
    def list_literal(self, meta, items):
        data = items[0] if len(items) > 0 else []
        return Collection(data, VariableTypeEnum.LIST,
                          meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    def list_item_kv(self, items):
        k, v = None, None
        for it in items:
            if isinstance(it, tuple) and it[0] == "key":
                k = it[1]
            else:
                v = it
        return {k: v}

    def list_literal_single_kv(self, items):
        return items[0]

    @v_args(meta=True)
    def frame_apply(self, meta, items) -> AsFrame:
        apply = AsFrame(value=items[0][0].value, meta={"file_meta": self._build_file_meta(meta), "node_meta": None})
        return apply

    @v_args(meta=True)
    def structure_prefix(self, meta, items):
        frame_name = items[0]
        val = items[1]
        return SchemedCollection(
            value=val,
            inferred_type=VariableTypeEnum.LIST,
            dataframe=frame_name,
            apply_type=FrameApplyEnum.PRE,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    @v_args(meta=True)
    def structure_suffix(self, meta, items):
        frame_name = items[1]
        val = items[0]
        return SchemedCollection(
            value=val,
            inferred_type=VariableTypeEnum.LIST,
            dataframe=frame_name,
            apply_type=FrameApplyEnum.POST,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    def inner_expression(self, items):
        return items[0]

    @v_args(meta=True)
    def expression(self, meta, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        it = items[0]
        if getattr(it, "meta", None):
            if it.meta.get("node_meta") is None:
                it.meta["node_meta"] = node_meta
        else:
            it.meta = {"file_meta": self._build_file_meta(meta), "node_meta": node_meta}
        return it

    def prompted_expression(self, items):
        return items[0]

    def prompted_integer(self, items):
        # Keep behavior as-is (returns int for INT tokens)
        for it in items:
            if isinstance(it, Token) and it.type == "INT":
                return int(it.value)
        return 0

    @v_args(meta=True)
    def none(self, meta, items):
        return SingleValue(None, VariableTypeEnum.NONE,
                           meta={"file_meta": self._build_file_meta(meta),
                                 "node_meta": None})

    # ------------------------------------------------------------------
    # IN / OUT (ActionInput / ActionOutput)
    # ------------------------------------------------------------------

    def required(self, items):
        return {"type": "required"}

    def optional(self, items):
        return {"type": "optional"}

    def default(self, items):
        expr = None
        for it in items:
            if isinstance(it, Expression) or isinstance(it, Value):
                expr = it
        return {"type": "default", "default": expr}

    def in_modifier(self, items):
        for it in items:
            if isinstance(it, dict):
                return it
        return {"type": "required"}

    @v_args(meta=True)
    def input_named(self, meta, items):
        name = None
        modifier = None
        prompt = None

        if items and isinstance(items[0], Tree):
            name = items[0].children[0].value
            rest = items[1:]
        else:
            rest = items

        default_expr, prompt, required = self.get_parameter_modifiers(modifier, rest)

        if name == "__in" or name == "__":
            return None

        return ActionInput(
            name=name or "",
            required=required,
            default=default_expr,
            prompt=prompt,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    def get_parameter_modifiers(self, modifier: dict | None, rest: List) -> tuple[Expression | None, MicroPrompt, bool]:
        prompt = None

        for it in rest:
            if isinstance(it, dict):
                modifier = it
            elif isinstance(it, MicroPrompt):
                prompt = it

        required = True
        default_expr = None

        if modifier:
            if modifier["type"] == "required":
                required = True

            elif modifier["type"] == "optional":
                required = False

            elif modifier["type"] == "default":
                required = False
                default_expr = modifier.get("default")

        return default_expr, prompt, required

    @v_args(meta=True)
    def input_unnamed_mod(self, meta, items):
        modifier = None
        prompt = None
        default_expr, prompt, required = self.get_parameter_modifiers(modifier, items)

        return ActionInput(
            name="",
            required=required,
            default=default_expr,
            prompt=prompt,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    @v_args(meta=True)
    def input_unnamed(self, meta, items):
        prompt = items[0] if items else None
        return ActionInput(
            name="",
            required=True,
            default=None,
            prompt=prompt,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    def in_inline(self, items):
        if not items or items[0] is None:
            return []
        return items

    def in_block(self, items):
        return [it for it in items if isinstance(it, ActionInput)]

    def in_body(self, items):
        if len(items) == 1 and isinstance(items[0], list):
            return items[0]
        return [it for it in items if isinstance(it, ActionInput)]

    def action_ins(self, items):
        return self._process_action_ins_or_outs(items)

    def _transform_action_ins_outs(self, items) -> list[Any]:
        logger.debug(items)
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None

        for it in items:
            logger.debug(f'Item: "{it}"')
            if isinstance(it, list):
                for a in it:
                    if a.meta and a.meta.get("node_meta") is None:
                        a.meta["node_meta"] = node_meta
                return it

        return []

    def _process_action_ins_or_outs(self, items):
        assert isinstance(items, list)

        if len(items) == 1:
            if isinstance(items[0], list) and all([i is None for i in items[0]]):
                items = []

        elif all([i is None for i in items]):
            items = []

        return self._transform_action_ins_outs(items)

    @v_args(meta=True)
    def output_named(self, meta, items):
        var_name = []
        prompt = None

        for it in items:
            if isinstance(it, Tree):
                if it.data == "varname":
                    for it_c in it.children:
                        var_name = it_c.value

            if isinstance(it, MicroPrompt):
                prompt = it

        if var_name == "__out" or var_name == "__":
            return None

        return ActionOutput(name=var_name, prompt=prompt,
                            meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def output_unnamed(self, meta, items):
        return ActionOutput(
            name="",
            prompt=items[0],
            meta={"file_meta": self._build_file_meta(meta), "node_meta": None},
        )

    def out_inline(self, items):
        return [it for it in items if isinstance(it, ActionOutput)]

    def out_block(self, items):
        return [it for it in items if isinstance(it, ActionOutput)]

    def out_body(self, items):
        if len(items) == 1 and isinstance(items[0], list):
            return items[0]
        return [it for it in items if isinstance(it, ActionOutput)]

    def action_out(self, items):
        return self._process_action_ins_or_outs(items)

    # ------------------------------------------------------------------
    # Action / body
    # ------------------------------------------------------------------
    def action_body(self, items):
        return [it for it in items if isinstance(it, Statement)]

    def action_name(self, items):
        return str(items[0])

    @v_args(meta=True)
    def action(self, meta, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        non_tokens: List[Any] = [i for i in items if not isinstance(i, Token)]
        if not non_tokens:
            return None

        name_str: Optional[str] = None
        desc_prompt: Optional[MicroPrompt] = None

        ins: List[ActionInput] = []
        outs: List[ActionOutput] = []
        body: List[Statement] = []

        for obj in non_tokens:
            if isinstance(obj, str):
                name_str = obj
            elif isinstance(obj, MicroPrompt):
                desc_prompt = obj
            elif isinstance(obj, list) and obj:
                first_elem = obj[0]

                if isinstance(first_elem, ActionInput):
                    ins = obj
                elif isinstance(first_elem, ActionOutput):
                    outs = obj
                elif isinstance(first_elem, Statement):
                    body = obj

        if all([i is None for i in ins]):
            ins = []

        if all([o is None for o in outs]):
            outs = []

        file_meta = FileMeta((meta.line, meta.end_line), (meta.column, meta.end_column), file=self._current_file)
        return ActionBlock(
            name=name_str,
            prompt=desc_prompt,
            action_inputs=ins,
            action_outputs=outs,
            body=body,
            meta={"file_meta": file_meta, "node_meta": node_meta},
        )

    # statement alias
    def instruction_block(self, items):
        return items[0]

    def instruction_line(self, items):
        return items[0]

    # ------------------------------------------------------------------
    # Loops
    # ------------------------------------------------------------------
    def loop_body(self, items):
        return [it for it in items if isinstance(it, Statement)]

    def loop_iterable(self, items):
        return items[0]

    def loop_condition(self, items):
        return items[0]

    def idx_var(self, items):
        var: Variable = items[0]
        return var.name or ""

    def item_var(self, items):
        var: Variable = items[0]
        return var.name or ""

    @v_args(meta=True)
    def generic_loop(self, meta, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        non_tokens = [i for i in items if not isinstance(i, Token)]

        block = RepeatBlock(meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta})
        for st in non_tokens[1:]:
            if isinstance(st, Statement):
                block.add_node(st)
        return block

    def loop_each_where(self, items):
        return items

    @v_args(meta=True)
    def loop_each(self, meta, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        filtered: List[Any] = [i for i in items if not isinstance(i, Token)]

        if len(filtered) < 3:
            block = RepeatEachBlock(
                each=filtered[0],
                as_vars=[],
                meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta},
            )
            body = filtered[1]
        else:
            bind = filtered[1].children  # Tree(Token('RULE', 'loop_each_bind'), ['idx', 'var_name', [where_cond]])
            idx_name = bind[0]
            item_name = bind[1]
            block = RepeatEachBlock(
                each=filtered[0],
                as_vars=[idx_name, item_name],
                meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta},
            )
            body = filtered[2]

        for st in body:
            block.add_node(st)
        return block

    def max_iterations(self, items):
        for it in items:
            if isinstance(it, int) or isinstance(it, Expression):
                return "max", it
        return None

    def loop_times_index(self, items):
        return "index", items[0]

    @v_args(meta=True)
    def loop_times(self, meta, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        times: Optional[int] = None
        body: List[Statement] = []
        as_vars = None
        for it in items:
            if isinstance(it, int):
                times = it
            elif isinstance(it, list):
                body = it
            elif isinstance(it, tuple) and it[0] == "index":
                as_vars = it[1]
        block = RepeatTimesBlock(times or 0, as_vars=as_vars,
                                 meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta})
        for st in body:
            block.add_node(st)
        return block

    @v_args(meta=True)
    def loop_conditional(self, meta, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        mode: Optional[str] = None  # "_WHILE" / "_UNTIL"
        cond: Optional[LeafStatement] = None
        max_iter: Optional[int] = None
        body: List[Statement] = []

        for it in items:
            if isinstance(it, Token) and it.type in ("WHILE", "UNTIL"):
                mode = it.type
            elif isinstance(it, tuple) and it[0] == "max":
                max_iter = it[1]
            else:
                if cond is None:
                    cond = it
                else:
                    if isinstance(it, list):
                        body.extend(it)
                    else:
                        body.append(it)

        if mode == "WHILE":
            block = RepeatWhileBlock(condition=cond, max_it=max_iter,
                                     meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta})
        else:
            block = RepeatUntilBlock(condition=cond, max_it=max_iter,
                                     meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta})

        for st in body:
            if isinstance(st, Statement):
                block.add_node(st)
        return block

    # ------------------------------------------------------------------
    # If / Elif / Else
    # ------------------------------------------------------------------
    def condition_expression(self, items):
        return items[0]

    def condition_if_body(self, items):
        return [it for it in items if isinstance(it, Statement)]

    @v_args(meta=True)
    def elif_clause(self, meta, items):
        return ElifBlock(condition=items[0], body=items[1],
                         meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def else_clause(self, meta, items):
        return ElseBlock(body=items[0], meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def if_clause(self, meta, items):
        return IfBlock(condition=items[0], body=items[1],
                       meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def condition_if(self, meta, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        if_block = None
        elif_list = []
        else_block = None

        for it in items:
            if isinstance(it, IfBlock):
                if_block = it
            elif isinstance(it, ElifBlock):
                elif_list.append(it)
            elif isinstance(it, ElseBlock):
                else_block = it

        assert if_block is not None

        return ConditionBlock(
            if_block=if_block,
            elif_list=elif_list,
            else_block=else_block,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta})

    # ------------------------------------------------------------------
    # Function calls: do ...
    # ------------------------------------------------------------------
    def qualified_name(self, items):
        return items

    def argument_list(self, items):
        for it in items:
            if isinstance(it, Expression):
                return it
        return items[0]

    def using_clause(self, items):
        payload = None
        for it in items:
            if isinstance(it, Token):
                continue
            payload = it
            break
        return "using", payload

    def producing_clause(self, items):
        payload = None
        schema = None

        # Filter out the _PRODUCING and _AS tokens
        parsed_items = [it for it in items if not isinstance(it, Token)]

        if len(parsed_items) > 0:
            payload = parsed_items[0]
        if len(parsed_items) > 1:
            schema = parsed_items[1]

        return "producing", payload, schema

    def callable_type(self, items):
        return "callable_type", items[0].value

    def inline_func_call(self, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        callable_type = items.pop(0)[1] if items and isinstance(items[0], tuple) and items[0][
            0] == "callable_type" else None
        name: Optional[str] = None
        using_expr: Optional[Expression] = None
        producing_expr: Optional[Expression] = None
        producing_schema: Optional[str | MicroPrompt] = None
        prompt: Optional[MicroPrompt] = None

        start_item = items[0][0] if isinstance(items[0], list) else items[0]
        end_item = items[-1] if len(items) > 1 else start_item

        # Build file span meta (best-effort, keep existing behavior).
        if isinstance(start_item, Token):
            start_line = start_item.line
            len_callable = 0 if not callable_type else len(callable_type) + 1
            start_column = max(start_item.column - (3 + len_callable), 0)
        else:
            it = start_item[1] if isinstance(start_item, tuple) else start_item
            start_line = it.meta["file_meta"].line[0]
            len_callable = 0 if not callable_type else len(callable_type) + 1
            start_column = max(it.meta["file_meta"].column[0] - (3 + len_callable), 0)

        if isinstance(end_item, Token):
            end_line = end_item.line
            end_column = end_item.column
        else:
            if isinstance(end_item, tuple):
                end_line = 0
                end_column = 0
                
                for it in end_item:
                    if it is None or not hasattr(it, 'meta'):
                        continue

                    end_line = max(end_line, it.meta["file_meta"].line[-1])
                    end_column = max(end_column, it.meta["file_meta"].column[-1])
            else:
                it = end_item
                end_line = it.meta["file_meta"].line[-1]
                end_column = it.meta["file_meta"].column[-1]

        line = (start_line, end_line)
        column = (start_column, end_column)

        for it in items:
            if isinstance(it, list):
                if len(it) == 1:
                    name = it[0].value
                else:
                    name = ".".join([i.value for i in it])

            elif isinstance(it, tuple):
                kind = it[0]
                if kind == "using":
                    using_expr = it[1]

                elif kind == "producing":
                    producing_expr = it[1]

                    if len(it) > 2 and it[2] is not None:
                        if isinstance(it[2], AsFrame):
                            producing_schema = it[2].value
                        else:
                            assert isinstance(it[2], MicroPrompt)
                            producing_schema = it[2]

            elif isinstance(it, MicroPrompt):
                prompt = it

        if isinstance(producing_expr, SingleValue):
            raise NemantixParserException(f'Cannot produce a constant value "{producing_expr.value}" in DO statement!')

        # TODO: handle imports from included files and manage imported deliberates
        # if ( name not in builtin_func_map and
        #     (name not in IMPORTED_TOOLSETS and
        #      (len(name.split("."))>1 and name.split(".")[-1] not in IMPORTED_TOOLSETS))):
        #     raise NemantixParserException(f'Undefined function "{name}" in DO statement!"')

        file_meta = FileMeta(line=line, column=column)
        return DoStatement(
            name=name,
            callable_type=callable_type,
            using=using_expr,
            prompt=prompt,
            producing=producing_expr,
            producing_schema=producing_schema,
            meta={"file_meta": file_meta, "node_meta": node_meta},
        )

    def block_func_call(self, items):
        return self.inline_func_call(items)

    def func_call(self, items):
        return items[0]

    # ------------------------------------------------------------------
    # Slots / Frames
    # ------------------------------------------------------------------
    def slot_name(self, items):
        return {"name": items[0].value}

    def slot_enum(self, items):
        return {"ENUM_TYPE": [it.value for it in items[1:]]}

    def slot_types(self, items):
        types_list = []
        for it in items:
            if isinstance(it, Token):
                it = it.value

            if isinstance(it, str):
                if it in slot_types_map:
                    types_list.append(slot_types_map[it])
                elif it in self._frame_names:
                    types_list.append({SlotTypesEnum.FRAME: it})
                else:
                    raise NemantixParserException(
                        f"Unknown slot type: {it}. Choose one of: {list(slot_types_map.keys())}"
                    )

            if isinstance(it, dict) and "ENUM_TYPE" in it:
                types_list.append({SlotTypesEnum.ENUM: it.get("ENUM_TYPE")})

        return {"types": types_list}

    def slot_card(self, items):
        return {"cardinality": items[0].value}

    def slot_spec(self, items):
        join_dict = {}
        for d in items:
            join_dict = join_dict | d
        return join_dict

    @v_args(meta=True)
    def slot(self, meta, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        name = None
        types = None
        cardinality = None
        prompt = None

        for it in items:
            if isinstance(it, MicroPrompt):
                prompt = it
            else:
                if "name" in it:
                    name = it["name"]
                if "types" in it:
                    types = it["types"] if len(it["types"]) > 0 else None
                if "cardinality" in it:
                    cardinality = it["cardinality"] if len(it["cardinality"]) > 0 else None

        return Slot(
            name=name,
            types=types,
            card=cardinality,
            prompt=prompt,
            meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta},
        )

    def frame_name(self, items):
        return items[0].value

    def frame_body(self, items):
        return items

    @v_args(meta=True)
    def frame(self, meta, items):
        node_meta = items.pop(0) if items and isinstance(items[0], NodeMeta) else None
        frame = Frame(name=items[0], meta={"file_meta": self._build_file_meta(meta), "node_meta": node_meta})
        self._frame_names.append(items[0])
        if len(items) > 1:
            for e in items[1]:
                frame.add_node(e)
        return frame

    # ------------------------------------------------------------------
    # Intentables (label + annotations)
    # ------------------------------------------------------------------
    def label(self, items):
        return items[0].value

    def meta_decl(self, items):
        name = ".".join([n.value for n in items[0]]) if len(items[0]) > 1 else items[0][0].value
        value = items[1] if len(items) > 1 else None

        if isinstance(value, Token):
            value = value.value
        return Annotation(name, value)

    @v_args(meta=True)
    def intentable_prefix(self, meta, items):
        lab = None
        _annotations = []
        for it in items:
            if isinstance(it, str):
                lab = it
            elif isinstance(it, Annotation):
                _annotations.append(it)
        return NodeMeta(annotations=_annotations, label=lab, file_meta=self._build_file_meta(meta))

    def meta_content(self, items):
        name = items[0].value
        ret = [name]
        ret.extend(i.value for i in items[1])
        return ret

    def inner_meta_expr_value(self, items):
        if items[0].meta["node_meta"]:
            raise NemantixParserException("Parsing error on node '" + str(items[
                                                                              0]) + "':\nNested intentables are not allowed. Maybe you used a '{label}' as a value of a '@meta'")
        return items[0]

    @v_args(meta=True)
    def meta_expression(self, meta, items):
        items = items[0] if items and isinstance(items[0], list) else items
        return MetaExpression(quals=items, meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    # --- Flow control ---
    @v_args(meta=True)
    def return_statement(self, meta, items):
        return Return(items, meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def break_statement(self, meta, items):
        return Break(meta={"file_meta": self._build_file_meta(meta), "node_meta": None})

    @v_args(meta=True)
    def continue_statement(self, meta, items):
        return Continue(meta={"file_meta": self._build_file_meta(meta), "node_meta": None})


class FixerTransformer(AstTransformer):
    """Transformer class that attempts to fix minor logical errors"""

    def transform_with_file_info(self, tree: Tree, file: PathLike):
        logger.debug(f'Using the fixer transformer on "{file}"')
        return super().transform_with_file_info(tree, file)

    def inline_func_call(self, items) -> DoStatement:
        do = super().inline_func_call(items)

        # fix for wrong qualifier usage on builtins
        name = do.name

        if do.callable_type == CallableTypeEnum.TOOL and name not in IMPORTED_TOOLSETS:
            if name in BuiltinFunctionEnum:
                logger.warning(f'Changed callable_type from "{do.callable_type}" to "None" '
                               f'to avoid error on builtin call "{name}"!')
                do.callable_type = None

        # fix output variables in do llm
        if name == BuiltinFunctionEnum.LLM.value:
            producing = do.producing
            if isinstance(producing, Collection):
                if len(producing.value) >= 1:
                    logger.warning(f'Changed producing clause in do llm from '
                                   f'"{[v for v in producing.value]}" to "{producing.value[0]}"!')
                    do.producing = producing.value[0]

        # fix wrong producing schema (frame) usage
        if do.callable_type in [CallableTypeEnum.TOOL, CallableTypeEnum.ACTION]:
            if do.producing_schema is not None:
                logger.warning(f'Removed producing schema "{do.producing_schema}" on action or tool!')
                do.producing_schema = None

        return do


class IncludeCollector(Visitor_Recursive):
    """Preprocess tree to extract include statements."""

    def __init__(self):
        super().__init__()
        self._includes = []

    def include(self, tree):
        self._includes.append(tree.children[0].value)

    def extract_includes(self, tree):
        self.visit(tree)
        ret = self._includes
        self._includes = []
        return ret


# =============================================================================
# Facade: ParserLark
# =============================================================================
TOKEN_MAP = {
    "_TOOLSET": "'toolset'",
    "_DELIBERATE": "'deliberate'",
    "_ACTION": "'action'",
    "_PLAN": "'plan'",
    "_FRAME": "'frame'",
    "_INCLUDE": "'include'",
    "_SLOT": "'slot'",
    "_END_BLOCK": "'__'",
    "_END_DELIBERATE": "'__deliberate'",
    "_END_ACTION": "'__action'",
    "_END_PLAN": "'__plan'",
    "_END_BODY": "'__body'",
    "_END_IF": "'__if'",
    "_END_REPEAT": "'__repeat'",
    "_END_TOOLSET": "'__toolset'",
    "_END_USE": "'__use'",
    "_END_FRAME": "'__frame'",
    "_END_IN": "'__in'",
    "_END_OUT": "'__out'",
    "_END_GUIDELINES": "'__guidelines'",
    "_WHEN": "'when'",
    "_FROM": "'from'",
    "_USE": "'use'",
    "_DO": "'do'",
    "_IF": "'if'",
    "_ELSE": "'else'",
    "_ELIF": "'elif'",
    "_RETURN": "'return'",
    "_BREAK": "'break'",
    "_CONTINUE": "'continue'",
    "_AS": "'as'",
    "CNAME": "an identifier (name)",
    "INT": "an integer",
    "ESCAPED_STRING": 'a string (e.g. "text")',
    "NXS_PATH": "a .nxs file path",
    "CARDINALITY": "cardinality (e.g. *, 0..1)",
    "INT_TYPE": "'INT'",
    "FLOAT_TYPE": "'FLOAT'",
    "TEXT_TYPE": "'TEXT'",
    "ENUM_TYPE": "'ENUM'",
    "STRUCT_TYPE": "'STRUCT'",
    "COLON": "':'",
    "PROMPT_INLINE_TEXT": ">> text <<",
    "PROMPT_BLOCK_TEXT": ">>> text <<<",
    "PROMPT_LINE_TEXT": ">> text",
    "PROMPT_LINE_VAR_TEXT": ">> text]",
    "LSQB": "'['",
    "RSQB": "']'",
    "LBRACE": "'{'",
    "RBRACE": "'}'",
    "AT": "'@'",
    "PIPE": "'|'",
    "NEWLINE": "newline",
    "$END": "end of file",
    "ACTION": "'action'",
    "DELIBERATE": "'deliberate'",
    "TOOL": "'tool'",
    "_BODY": "'body'",
    "_EACH": "'each'",
    "_END_DO": "'__do'",
    "_GUIDELINES": "'guidelines'",
    "_IN": "'in'",
    "_MAX": "'max'",
    "_NONE": "'none'",
    "_OPTIONAL": "'optional'",
    "_OUT": "'out'",
    "_PRODUCING": "'producing'",
    "_REPEAT": "'repeat'",
    "_REQUIRE": "'require'",
    "_REQUIRED": "'required'",
    "_TIMES": "'times'",
    "_USING": "'using'",
    "_WHERE": "'where'",
    "_WITH": "'with'",
    "_DEFAULT": "'default'",
    "_SHORT_NONE": "'_'",
    "BOOL_TYPE": "'BOOL'",
    "FLOAT": "a decimal number",
    "TRUE": "'true'",
    "FALSE": "'false'",
    "FROZEN": "'frozen'",
    "DRAFTED": "'drafted'",
    "UNDEFINED": "'undefined'",
    "UNTIL": "'until'",
    "WHILE": "'while'",
    "NXC_PATH": "a .nxc file path",
    "SIM_ABOUT": "'~'",
    "SIM_CLOSE": "'~~'",
    "SIM_FAR": "'~~~'",
    "SIM_LOOSE": "'~~?'",
    "SIM_STRICT": "'=~'",
}


@lru_cache(maxsize=None)
def _get_cached_lark(grammar_path: str) -> Lark:
    grammar_text = Path(grammar_path).read_text(encoding="utf-8")
    return Lark(grammar_text, start="start", parser="lalr", propagate_positions=True)


class ParserLark:
    """
    A facade that wraps Lark and returns a list of top-level AST nodes.

    Supports parsing one or more files and resolves includes recursively.
    """

    def __init__(self, grammar: PathLike | None = None):
        grammar_path = Path(grammar) if grammar is not None else get_grammar_path()
        self._lark = _get_cached_lark(str(grammar_path))
        self._transformer = AstTransformer()
        self._fixer_transformer = FixerTransformer()
        self._include_collector = IncludeCollector()

    def parse(self, content: str, location: PathLike, verbose=False, enable_fixer=False):
        """
        Parse one root location or a list of root locations, using a caller-provided registry.

        Args:
            content: NX_ Script content
            location: NX_ location of Script
            verbose: if True, visits and prints the parse tree
            enable_fixer: if True, uses the fixer transformer

        Includes are resolved as opaque string identifiers and must match keys in script_by_loc
        (after minimal normalization).
        """
        try:
            tree = self._lark.parse(content)
        except (UnexpectedToken, UnexpectedCharacters) as e:
            msg = self._format_error(e, content)
            raise SyntaxError(f"\nIn script: {location}\n{msg}") from e

        # Optional debug
        if verbose:
            self.visit_tree(tree)

        if enable_fixer:
            ast_nodes = self._fixer_transformer.transform_with_file_info(tree, location)
        else:
            ast_nodes = self._transformer.transform_with_file_info(tree, location)

        return ast_nodes

    def parse_lark_tree(self, source: str) -> ParseTree:
        try:
            return self._lark.parse(source)
        except (UnexpectedToken, UnexpectedCharacters) as e:
            msg = self._format_error(e, source)
            raise NemantixParserException(f"{msg}") from e

    @staticmethod
    def _get_context(text: str, line: int, column: int, span: int = 40) -> str:
        """Extract the specific line and point to the error column."""
        lines = text.splitlines()
        if line - 1 >= len(lines):
            return "End of File"

        error_line = lines[line - 1]
        error_line_num_digits = len(str(line))
        pointer = " " * (column - 3 + error_line_num_digits) + "^"
        return f"\n  Line {line}: {error_line}\n           {pointer}"

    def _format_error(self, e: Any, text: str) -> str:
        """Generate a user-friendly syntax error message."""
        msg = ["\nSyntax Error"]

        if isinstance(e, UnexpectedCharacters):
            msg.append(f"Unexpected character found: '{e.char}'")
            msg.append(self._get_context(text, e.line, e.column))

            if ">>" in text.splitlines()[e.line - 1]:
                msg.append("💡 Hint: It looks like you might have an unclosed prompt.")
                msg.append("         Inline prompts must end with '<<'.")
                msg.append("         Block prompts must be '>>> ... <<<'")
            elif e.char == '"':
                msg.append("💡 Hint: You might have an unclosed string literal.")

        elif isinstance(e, UnexpectedToken):
            token = e.token
            msg.append(f"Unexpected token: '{token.value}' (Type: {token.type})")
            msg.append(self._get_context(text, e.line, e.column))

            expected = sorted([TOKEN_MAP.get(t, t) for t in e.expected])

            if ":" in e.expected or "COLON" in e.expected:
                msg.append("💡 Hint: You might be missing a colon ':' at the end of the previous definition.")
            elif any("_END_" in t for t in e.expected):
                msg.append("💡 Hint: A block might not be closed properly.")
                msg.append(f"         Expected one of: {', '.join(expected)}")
                if token.type == "$END":
                    msg.append("         (You reached the end of the file without closing a block)")
            elif "]" in e.expected:
                msg.append("💡 Hint: You might have forgotten to close a variable bracket ']'")
            else:
                formatted_expected = ", ".join(f"'{x}'" if not x.startswith("'") else x for x in expected)
                msg.append(f"Expected one of: {formatted_expected}")

        return "\n".join(msg)

    @staticmethod
    def print_ast(nodes: list[Statement] | Statement, meta: bool = False):
        """
        Pretty-print the AST with indentation.

        - Uses only the first line of __str__ as a header.
        - For BlockStatements, it descends into children.
        - If meta=True, prints FileMeta/NodeMeta when available.
        """

        def _print(obj, level: int = 0):
            if isinstance(obj, list):
                for n in obj:
                    _print(n, level)
                return

            node = obj
            pad = "  " * level

            s = str(node) if node is not None else ""
            lines = s.splitlines() if s else [""]

            meta_line = ""
            if meta and not isinstance(node, Tree) and getattr(node, "meta", None):
                fm = node.meta.get("file_meta")
                nm = node.meta.get("node_meta")
                if fm is not None or nm is not None:
                    meta_line = f"[FileMeta {fm} // NodeMeta:{nm}]"

            if meta_line:
                logger.debug(f"{pad}{meta_line}")

            logger.debug(f"{pad}- {lines[0]}")

            if not isinstance(node, BlockStatement):
                for extra in lines[1:]:
                    logger.debug(f"{pad}    {extra}")

            if isinstance(node, BlockStatement) and getattr(node, "children", None):
                for child in node.children:
                    _print(child, level + 1)

        _print(nodes, level=0)

    def visit_tree(self, node: Tree | Token, indent: int = 0):
        """Debug utility to print the raw Lark tree (not used by default)."""
        if isinstance(node, Tree):
            logger.debug("  " * indent + f"Tree({node.data})")
            for child in node.children:
                self.visit_tree(child, indent + 1)
        elif isinstance(node, Token):
            logger.debug("  " * indent + f"Token({node.type}, {node.value})")
