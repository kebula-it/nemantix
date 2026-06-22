from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from nemantix.core.exceptions import NemantixException


# =============================================================================
# Meta
# =============================================================================
class Annotation:
    """Represents an annotation attached to a node."""

    def __init__(self, name: str, value: Any):
        self.name: str = name
        self.value: Any = value

    def __str__(self):
        return f"@{self.name}: {self.value}"


class Meta:
    """Base class for metadata objects."""

    pass


class FileMeta(Meta):
    """File position info for a parsed element."""

    def __init__(
        self, line: tuple[int, int], column: tuple[int, int], file: Path | None = None
    ):
        self.line: tuple[int, int] = line
        self.column: tuple[int, int] = column
        self.file: Path | None = file

    def __str__(self):
        return (
            f"{self.file} from line:column {self.line[0]}:{self.column[0]} "
            f"to {self.line[1]}:{self.column[1]}"
        )


class NodeMeta(Meta):
    """Metadata attached to a statement node."""

    def __init__(
        self, annotations: list[Annotation], label: str | None, file_meta: FileMeta
    ):
        self.annotations: list[Annotation] = annotations
        self.label: str | None = label
        self.file_meta: FileMeta = file_meta

    def __str__(self):
        label_str = f"label={self.label}" if self.label else ""
        ann_str = [str(a) for a in self.annotations]
        return f"NodeMeta: {label_str}, annotations={ann_str}"


# =============================================================================
# Nodes
# =============================================================================
class Statement:
    """Base class for statements. Can be a leaf or a block (composite)."""

    def __init__(self, meta: dict[str, Meta | None]):
        self.meta = meta

    def to_nxs(self, **kwargs) -> str:
        raise NotImplementedError


class LeafStatement(Statement):
    """Statement that does not contain child statements."""

    def __init__(self, meta: dict[str, Meta | None]):
        super().__init__(meta)


standard_intentables_map = {
    "completion": "intent.completion",
    "breakdown": "intent.breakdown",
    "goal": "intent.goal",
}


class BlockStatement(Statement):
    """Statement that can contain child statements."""

    def __init__(self, meta: dict[str, Meta | None]):
        super().__init__(meta)
        self.children = []

    def add_node(self, node):
        """Add a child node to this block."""
        self.children.append(node)

    def get_annotation_value(self, name: str):
        if self.meta["node_meta"] is None:
            raise NemantixException(
                f"The node \n   '{self}'\nhas no annotation named '{name}'."
            )

        val = None
        node_meta = self.meta["node_meta"]
        assert isinstance(node_meta, NodeMeta)

        for annot in node_meta.annotations:
            if name == annot.name or (
                name in standard_intentables_map
                and standard_intentables_map[name] == annot.name
            ):
                val = annot.value

        if val is None:
            raise NemantixException(
                f"The node \n   '{self}'\nhas no annotation named '{name}'."
            )

        return val

    def get_qualifier(self) -> Any | list:
        if hasattr(self, "qualifier"):
            if self.qualifier is not None:
                return self.qualifier

        try:
            qualifier = self.get_annotation_value("completion")

            # Unwrap SingleValue if present
            if hasattr(qualifier, "value"):
                qualifier = qualifier.value

            # Convert to list if it was parsed as a raw string
            if isinstance(qualifier, str):
                if "->" in qualifier:
                    qualifier = qualifier.split("->")
                else:
                    qualifier = [qualifier]

            if isinstance(qualifier, list | tuple):
                # unwrap SingleValues inside lists
                def _get_str(item):
                    return (
                        str(item.value).strip()
                        if hasattr(item, "value")
                        else str(item).strip()
                    )

                if len(qualifier) == 1:
                    val = _get_str(qualifier[0])
                    qualifier = [plan_qualifier_map[val], plan_qualifier_map[val]]

                elif len(qualifier) == 2:
                    qualifier = [
                        plan_qualifier_map[_get_str(qualifier[0])],
                        plan_qualifier_map[_get_str(qualifier[1])],
                    ]
            else:
                raise NemantixException(
                    f"Received {type(qualifier)} as completion qualifier!"
                )

            return qualifier

        except NemantixException:
            return None

    def is_not_valid_qualifier(self) -> bool:
        if hasattr(self, "qualifier"):
            return (
                self.qualifier
                and not plan_qualifier_ordered_map[self.qualifier[0]]
                <= plan_qualifier_ordered_map[self.qualifier[1]]
            )
        return True

    def get_intent(self):
        intent = None
        for name in ["intent.goal", "goal"]:
            try:
                intent = self.get_annotation_value(name)
            except NemantixException:
                pass

            if intent:
                return intent

    def __str__(self):
        children_str = ", ".join(str(c) for c in self.children)
        return f"{self.__class__.__name__}(children=[{children_str}])"


# =============================================================================
# LeafStatement subclasses
# =============================================================================
class Require(LeafStatement):
    def __init__(self, file_path: str, meta: dict[str, Meta | None]):
        super().__init__(meta)
        self.file_path = file_path

    def __str__(self):
        return f"Require: {self.file_path!r}"

    def to_nxs(self) -> str:
        return f"require {self.file_path}"


class MicroPrompt(LeafStatement):
    def __init__(self, prompt: str, meta: dict[str, Meta | None]):
        super().__init__(meta)
        self.prompt = prompt

    def __str__(self):
        return f"MicroPrompt: {self.prompt!r}"


class PythonToolDeclaration(LeafStatement):
    def __init__(self, name: str, prompt: MicroPrompt, meta: dict[str, Meta | None]):
        super().__init__(meta)
        self.name = name
        self.prompt = prompt

    def __str__(self):
        return f"PythonToolDecl: {self.name} - {self.prompt}"


# =============================================================================
# Expressions
# =============================================================================
class VariableTypeEnum(Enum):
    INT = "int"
    BOOL = "boolean"
    STRING = "string"
    FSTRING = "fstring"
    FLOAT = "float"
    LIST = "list"
    DICT = "dict"
    NONE = "none"


class FrameApplyEnum(Enum):
    PRE = "prefix"
    POST = "suffix"


class Expression(LeafStatement):
    def __init__(self, meta: dict[str, Meta | None]):
        super().__init__(meta)


class MetaExpression(Expression):
    def __init__(self, quals: list[str], meta: dict[str, Meta | None]):
        super().__init__(meta)
        self.quals = quals

    def __str__(self):
        # first element is the one before "@"
        return f"MetaExpression({self.quals[0]}@{'.'.join(self.quals[1:])})"


class ExpressionOperand(Expression):
    """Marker base for operands in expressions."""

    def __init__(self, meta: dict[str, Meta | None]):
        super().__init__(meta)


class Value(ExpressionOperand):
    def __init__(
        self, value: Any, inferred_type: VariableTypeEnum, meta: dict[str, Meta | None]
    ):
        super().__init__(meta)
        self.value = value
        self.inferred_type = inferred_type

    def __str__(self):
        return f"VALUE({self.inferred_type}: {self.value})"

    def to_nxs(self, **kwargs):
        if self.inferred_type in [
            VariableTypeEnum.INT,
            VariableTypeEnum.FLOAT,
            VariableTypeEnum.BOOL,
        ]:
            code = str(self.value).lower()

        elif self.inferred_type in [VariableTypeEnum.STRING, VariableTypeEnum.FSTRING]:
            code = f'"{self.value}"'

        elif self.inferred_type == VariableTypeEnum.LIST:
            content = [v.to_nxs(**kwargs) for v in self.value]
            code = f"[{', '.join(content)}]"

        elif self.inferred_type == VariableTypeEnum.DICT:
            content = [f"{k}: {v.to_nxs(**kwargs)}" for k, v in self.value.items()]
            code = f"[{', '.join(content)}]"
        else:
            code = "none"

        return code


class SingleValue(Value):
    def __init__(
        self, value: Any, inferred_type: VariableTypeEnum, meta: dict[str, Meta | None]
    ):
        super().__init__(value, inferred_type, meta)

    def to_nxs(self, **kwargs):
        if self.inferred_type == VariableTypeEnum.FSTRING:
            content = []
            for v in self.value:
                if isinstance(v, str):
                    content.append(v)

                elif isinstance(v, Variable):
                    content.append(f"[{v.name}]")
                else:
                    content.append(str(v).lower())

            return f'"{"".join(content)}"'

        return super().to_nxs(**kwargs)


class Variable(ExpressionOperand):
    def __init__(
        self,
        name: str | None,
        prompt: MicroPrompt | None,
        path: str | None | list[SingleValue],
        meta: dict[str, Meta | None],
    ):
        super().__init__(meta)
        self.name = name
        self.prompt = prompt
        self.path = path

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return NotImplemented
        return (self.name, self.prompt, self.path) == (
            other.name,
            other.prompt,
            other.path,
        )

    def __str__(self):
        prompt_str = f" prompt={self.prompt.prompt}" if self.prompt is not None else ""
        path_str = (
            f"{[str(p) for p in self.path] if isinstance(self.path, list) else self.path}"
            if self.path
            else ""
        )
        return f"Variable: {self.name}{path_str}{prompt_str}"

    def to_nxs(self, **kwargs) -> str:
        path_str = ""
        if self.path:
            # Handle nested paths like [user:address:city]
            path_str = ":" + ":".join(
                str(p.value) if hasattr(p, "value") else str(p) for p in self.path
            )
        return f"[{self.name}{path_str}]"


class Collection(Value):
    def __init__(
        self,
        value: list[SingleValue] | dict[str, SingleValue],
        inferred_type: VariableTypeEnum,
        meta: dict[str, Meta | None],
    ):
        super().__init__(value, inferred_type, meta)

    def __str__(self):
        val_str = (
            [str(v) for v in self.value]
            if isinstance(self.value, list)
            else str(self.value)
        )
        return f"Collection({self.inferred_type}: {val_str})"

    def to_nxs(self, **kwargs) -> str:
        buffer = []
        strip_parenthesis = [True]

        if isinstance(self.value, dict):
            buffer.append(self._dict_to_nxs(value=self.value, **kwargs))
            strip_parenthesis.append(False)
        else:
            assert isinstance(self.value, (list, tuple))

            for v in self.value:
                if isinstance(v, dict):
                    buffer.append(self._dict_to_nxs(value=v, **kwargs))
                    strip_parenthesis.append(False)

                elif isinstance(v, Variable):
                    buffer.append(f"[{v.name}]")
                    strip_parenthesis.append(True)

                elif isinstance(v, Expression):
                    buffer.append(v.to_nxs(**kwargs))
                    strip_parenthesis.append(True)

                elif hasattr(v, "to_nxs"):
                    buffer.append(v.to_nxs(**kwargs))

                elif hasattr(v, "name"):
                    buffer.append(f"[{v.name}]")
                    strip_parenthesis.append(True)
                else:
                    buffer.append(str(v))
                    strip_parenthesis.append(True)

        if all(strip_parenthesis):
            return ", ".join(buffer)

        return f"({', '.join(buffer)})"

    def _dict_to_nxs(self, value: dict, **kwargs) -> str:
        buf = []
        for k, v in value.items():
            if isinstance(v, list):
                if len(v) == 1:
                    v = v[0]
                else:
                    raise NotImplementedError(f"list: {v}")

            buf.append(f"{k}: {v.to_nxs(**kwargs)}")

        return ", ".join(buf)


class SchemedCollection(Collection):
    def __init__(
        self,
        value: list[SingleValue] | dict[str, SingleValue],
        inferred_type: VariableTypeEnum,
        dataframe: str,
        apply_type: FrameApplyEnum,
        meta: dict[str, Meta | None],
    ):
        super().__init__(value, inferred_type, meta)
        self.dataframe = dataframe
        self.apply_type = apply_type

    def __str__(self):
        pos_str = self.apply_type.value
        val_str = (
            [str(v) for v in self.value]
            if isinstance(self.value, list)
            else str(self.value)
        )

        if hasattr(self.dataframe, 'value'):
            dataframe = self.dataframe.value
        else:
            dataframe = str(self.dataframe)

        return f"SchemedCollection {'{' + dataframe + '}'}[{pos_str}]: {val_str})"


class Assignment(Expression):
    def __init__(
        self, var: Variable, value: Expression | None, meta: dict[str, Meta | None]
    ):
        super().__init__(meta)
        self.var = var
        self.value = value

    def __str__(self):
        return f"ASSIGN: {self.var} = {self.value}"

    def to_nxs(self, **kwargs) -> str:
        if isinstance(self.value, list):
            values = [v.to_nxs(**kwargs) for v in self.value]
            return f"[{self.var.name}] = ({', '.join(values)})"

        if self.value is None:
            return f"[{self.var.name}] = none"

        return f"[{self.var.name}] = {self.value.to_nxs(**kwargs)}"


class BinaryOperationEnum(Enum):
    # Concatenation
    CONCAT = "|"
    # Fallback / Null coalescing
    FALLBACK = "??"
    # Logical
    LOGICAL_OR = "||"
    LOGICAL_XOR = "^^"
    LOGICAL_AND = "&&"
    # Comparisons
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LTE = "<="
    GTE = ">="
    # Arithmetic Sum
    ADD = "+"
    SUB = "-"
    # Arithmetic Product
    MUL = "*"
    DIV = "/"
    MOD = "%"
    # Power
    POW = "^"


class SimilarityEnum(Enum):
    SIM = "~"
    SIM_QUAL = "~op~"
    # Right
    SIM_RIGHT = "~>"
    SIM_QUAL_RIGHT = "~op~>"
    # Left
    SIM_LEFT = "<~"
    SIM_QUAL_LEFT = "<~op~"


class SimilarityQualifierEnum(Enum):
    FAR = "FAR"
    LOOSE = "LOOSE"
    ABOUT = "ABOUT"
    CLOSE = "CLOSE"
    STRICT = "STRICT"
    NUMBER = "NUMBER"


map_sim_qual_kw = {
    "SIM_FAR": SimilarityQualifierEnum.FAR,
    "SIM_LOOSE": SimilarityQualifierEnum.LOOSE,
    "SIM_ABOUT": SimilarityQualifierEnum.ABOUT,
    "SIM_CLOSE": SimilarityQualifierEnum.CLOSE,
    "SIM_STRICT": SimilarityQualifierEnum.STRICT,
    "NUMBER": SimilarityQualifierEnum.NUMBER,
}


class UnaryOperationEnum(Enum):
    POS = "+"
    NEG = "-"
    NOT = "!"


class BinaryOperation(Expression):
    """Binary (or N-ary) operation: math, logic, concat, comparisons, etc."""

    def __init__(
        self,
        operation: BinaryOperationEnum,
        first: Expression,
        second: Expression,
        meta: dict[str, Meta | None],
    ):
        super().__init__(meta)
        self.first = first
        self.operation = operation
        self.second = second

    def __str__(self):
        op = (
            self.operation.value if isinstance(self.operation, Enum) else self.operation
        )
        return f"BinaryOperation: [{self.first}] {op} [{self.second}]"

    def to_nxs(self, **kwargs) -> str:
        return f"{self.first.to_nxs(**kwargs)} {self.operation.value} {self.second.to_nxs(**kwargs)}"


class SimilarityOperation(Expression):
    def __init__(
        self,
        operation: SimilarityEnum,
        qualifier: tuple[SimilarityQualifierEnum, Value]
        | SimilarityQualifierEnum
        | None,
        first: Expression,
        second: Expression,
        meta: dict[str, Meta | None],
    ):
        super().__init__(meta)
        self.first = first
        self.operation = operation
        self.second = second
        self.qualifier = qualifier

    def __str__(self):
        # Keep the current signature, but make the string rendering robust.
        qualifier_str = ""
        q = self.qualifier

        # Common internal representation: (SimilarityQualifierEnum, value|None)
        if (
            isinstance(q, tuple)
            and len(q) == 2
            and isinstance(q[0], SimilarityQualifierEnum)
        ):
            q_enum, q_val = q
            name = q_enum.name
            qualifier_str = f"{name}"
            if q_enum == SimilarityQualifierEnum.NUMBER and q_val is not None:
                qualifier_str += f"={q_val}"
        elif isinstance(q, SimilarityQualifierEnum):
            qualifier_str = q.name
        elif q is not None:
            qualifier_str = str(q)

        op_name = (
            self.operation.name
            if isinstance(self.operation, Enum)
            else str(self.operation)
        )
        return f"SimilarityOperation: [{self.first}] {op_name} {qualifier_str} [{self.second}]"


class BuiltinFunctionEnum(Enum):
    EXISTS = "exists"
    COALESCE = "coalesce"
    PRINT = "print"
    TYPE = "type"
    LLM = "llm"
    SIZE = "size"
    SUBSTRING = "substring"
    RETRIEVE = "retrieve"
    EXPAND = "expand"
    EXTEND = "extend"
    GENERALIZE = "generalize"

    # conversions
    BOOL = "bool"
    NUM = "num"
    STR = "str"
    TO_BOOL = "to_bool"
    TO_NUM = "to_num"
    TO_STR = "to_str"

    # math
    SIN = "sin"
    COS = "cos"
    SQRT = "sqrt"


builtin_func_map = {v.value: v for v in BuiltinFunctionEnum.__members__.values()}


class BuiltinFunction(Expression):
    def __init__(
        self,
        function: BuiltinFunctionEnum,
        args: list[Expression],
        meta: dict[str, Meta | None],
    ):
        super().__init__(meta)
        self.function = function
        self.args = args

    def __str__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"BuiltinFunction({self.function.value}({args_str}))"

    def to_nxs(self, **kwargs) -> str:
        if isinstance(self.args, list):
            args = ", ".join(a.to_nxs(**kwargs) for a in self.args)
        else:
            args = self.args.to_nxs(**kwargs)

        return f"{self.function.name.lower()}({args})"


class UnaryOperation(Expression):
    """Unary '+', '-', '!'."""

    def __init__(
        self,
        operation: UnaryOperationEnum,
        operand: Expression,
        meta: dict[str, Meta | None],
    ):
        super().__init__(meta)
        self.operation = operation
        self.operand = operand

    def __str__(self):
        op = (
            self.operation.value if isinstance(self.operation, Enum) else self.operation
        )
        return f"UnaryOp({op}, {self.operand})"

    def to_nxs(self, **kwargs) -> str:
        return f"{self.operation.value}{self.operand.to_nxs(**kwargs)}"


class CallableTypeEnum(Enum):
    TOOL = "tool"
    ACTION = "action"


callable_type_map = {"tool": CallableTypeEnum.TOOL, "action": CallableTypeEnum.ACTION}


class DoStatement(LeafStatement):
    def __init__(
        self,
        name: str | None,
        callable_type: str | None,
        using: Expression | None,
        prompt: MicroPrompt | None,
        producing: Expression | None,
        producing_schema: str | MicroPrompt | None,
        meta: dict[str, Meta | None],
    ):
        super().__init__(meta)
        self.name = name
        self.callable_type = callable_type_map[callable_type] if callable_type else None
        self.using = using
        self.prompt = prompt
        self.producing = producing
        self.producing_schema = producing_schema

    def __str__(self):
        type_str = f"{self.callable_type} " if self.callable_type is not None else ""
        name_str = self.name if self.name is not None else "_"
        return (
            f"DO: {type_str}{name_str} using:{self.using}, prompt:{self.prompt}, producing:{self.producing}, "
            f"as:{self.producing_schema}"
        )

    def to_nxs(self, **kwargs):
        code = ["do"]
        lines = self.meta["file_meta"].line
        is_multiline = lines[1] - lines[0] > 0

        if self.callable_type is not None:
            code.append(str(self.callable_type.value).lower())

        code.append(str(self.name))

        if is_multiline:
            code = [" ".join(code) + ":"]
            char = "\n"
            space = "  "
        else:
            char = " "
            space = ""

        if self.using is not None:
            code.append(f"{space}using [{self.using.to_nxs(**kwargs)}]")

        if self.producing is not None:
            code.append(f"{space}producing [{self.producing.to_nxs(**kwargs)}]")

        if isinstance(self.producing_schema, str):
            code.append(f"{space}as {{{self.producing_schema}}}")

        if is_multiline:
            if isinstance(self.prompt, MicroPrompt):
                code.append(f"__do >> {self.prompt.prompt} <<")
            else:
                code.append("__do")
        else:
            if isinstance(self.prompt, MicroPrompt):
                code.append(f">> {self.prompt.prompt} <<")

        return char.join(code)


class Return(LeafStatement):
    def __init__(self, val: list[Expression], meta: dict[str, Meta | None]):
        super().__init__(meta)
        self.val = val

    def __str__(self):
        return f"RETURN: {self.val}"


class Break(LeafStatement):
    def __init__(self, meta: dict[str, Meta | None]):
        super().__init__(meta)

    def __str__(self):
        return "BREAK"


class Continue(LeafStatement):
    def __init__(self, meta: dict[str, Meta | None]):
        super().__init__(meta)

    def __str__(self):
        return "CONTINUE"


# =============================================================================
# BlockStatement subclasses
# =============================================================================
class RepeatBlock(BlockStatement):
    def __init__(self, meta: dict[str, Meta | None]):
        super().__init__(meta)


class RepeatEachBlock(RepeatBlock):
    def __init__(
        self, each: LeafStatement, as_vars: list[str], meta: dict[str, Meta | None]
    ):
        super().__init__(meta)
        self.each = each
        self.as_vars = as_vars

    def __str__(self):
        children_str = "\n   - ".join(str(c) for c in self.children)
        return f"RepeatEachBlock(each={self.each}, as_vars={self.as_vars}) \n{children_str}"


class RepeatTimesBlock(RepeatBlock):
    def __init__(self, times: int, as_vars: list[str], meta: dict[str, Meta | None]):
        super().__init__(meta)
        self.times = times
        self.as_vars = as_vars

    def __str__(self):
        children_str = "\n   - ".join(str(c) for c in self.children)
        return f"RepeatTimesBlock(times={self.times} as_vars={self.as_vars})\n{children_str})"


class RepeatWhileBlock(RepeatBlock):
    def __init__(
        self,
        condition: LeafStatement,
        max_it: Expression | int,
        meta: dict[str, Meta | None],
    ):
        super().__init__(meta)
        self.condition = condition
        self.max = max_it

    def __str__(self):
        children_str = "\n   - ".join(str(c) for c in self.children)
        return f"RepeatWhileBlock(condition={self.condition}, max={self.max}) \n{children_str})"


class RepeatUntilBlock(RepeatBlock):
    def __init__(
        self,
        condition: LeafStatement,
        max_it: Expression | int,
        meta: dict[str, Meta | None],
    ):
        super().__init__(meta)
        self.condition = condition
        self.max = max_it

    def __str__(self):
        children_str = ", ".join(str(c) for c in self.children)
        return f"RepeatUntilBlock(condition={self.condition}, max={self.max}) \n{children_str})"


# =============================================================================
# If / Elif / Else
# =============================================================================
class IfBlock(BlockStatement):
    def __init__(
        self,
        condition: LeafStatement | None,
        body: list[Statement] | None,
        meta: dict[str, Meta | None],
    ):
        super().__init__(meta)
        self.condition = condition
        if body:
            self.children = body

    def __str__(self):
        children_str = "\n   - ".join(str(c) for c in self.children)
        return f"If({self.condition}): \n{children_str}"


class ElifBlock(BlockStatement):
    def __init__(
        self,
        condition: LeafStatement | None,
        body: list[Statement] | None,
        meta: dict[str, Meta | None],
    ):
        super().__init__(meta)
        self.condition = condition
        if body:
            self.children = body

    def __str__(self):
        children_str = "\n   - ".join(str(c) for c in self.children)
        return f"Elif({self.condition}): \n{children_str}"


class ElseBlock(BlockStatement):
    def __init__(self, body: list[Statement] | None, meta: dict[str, Meta | None]):
        super().__init__(meta)
        if body:
            self.children = body

    def __str__(self):
        children_str = "\n   - ".join(str(c) for c in self.children)
        return f"Else: \n{children_str}"


class ConditionBlock(BlockStatement):
    def __init__(
        self,
        if_block: IfBlock,
        elif_list: list[ElifBlock] | None,
        else_block: ElseBlock | None,
        meta: dict[str, Meta | None],
    ):
        super().__init__(meta)
        self.children.append(if_block)
        for elif_stmt in elif_list or []:
            self.children.append(elif_stmt)
        if else_block:
            self.children.append(else_block)

    def __str__(self):
        children_str = "\n   - ".join(str(c) for c in self.children)
        return f"ConditionBlock: \n{children_str}"


# =============================================================================
# Action
# =============================================================================
@dataclass
class ActionInput:
    name: str
    required: bool
    default: Expression | None
    prompt: MicroPrompt
    meta: dict[str, Meta | None]

    def __str__(self):
        default_str = f", default={self.default}" if self.default is not None else ""
        prompt_str = f", prompt={self.prompt.prompt}" if self.prompt is not None else ""
        return f"ActionInput(name={self.name}, required={self.required}{default_str}{prompt_str})"


@dataclass
class ActionOutput:
    name: str
    prompt: MicroPrompt
    meta: dict[str, Meta | None]

    def __str__(self):
        prompt_str = f", prompt={self.prompt}" if self.prompt else ""
        return f"ActionOutput(name={self.name}{prompt_str})"


class ActionBlock(BlockStatement):
    def __init__(
        self,
        name: str | None,
        prompt: MicroPrompt,
        action_inputs: list[ActionInput],
        action_outputs: list[ActionOutput],
        body: list[Statement] | None,
        meta: dict[str, Meta],
    ):
        super().__init__(meta)
        self.name = name
        self.prompt = prompt
        self.input = action_inputs
        self.output = action_outputs
        if body:
            self.children = body
        self.qualifier = None
        self.qualifier = self.get_qualifier()
        if (
            self.qualifier
            and not plan_qualifier_ordered_map[self.qualifier[0]]
            <= plan_qualifier_ordered_map[self.qualifier[1]]
        ):
            raise NemantixException(
                "Action completion qualifiers must specify increasing"
                " completion levels (e.g drafted->frozen). "
                f"Completion '{self.qualifier[0].value}->{self.qualifier[1].value}' "
                f"is not allowed."
            )

    def __str__(self):
        children_str = "\n     - ".join(str(c) for c in self.children)
        prompt_str = f">>{self.prompt.prompt}<< " if self.prompt else ""
        qual_str = f",[{self.qualifier}]" if self.qualifier else ""
        return f"Action{qual_str} '{self.name}' {prompt_str}with input={self.input}, output={self.output}\n {children_str}"


class ImportStatement(LeafStatement):
    def __init__(self, name: str, elements: list[str], meta: dict[str, Meta | None]):
        super().__init__(meta)
        self.name = name
        self.elements = elements

    def __str__(self):
        return f"ImportStatement(name={self.name}, elements={self.elements})"


class ImportToolsetStatement(ImportStatement):
    def __init__(
        self,
        name: str,
        elements: list[str],
        args: Expression,
        alias: str,
        meta: dict[str, Meta | None],
    ):
        super().__init__(name, elements, meta)
        self.args = args
        self.alias = alias

    def get_aliased_name(self):
        """Returns a string 'name:alias' if there's an alias, else only 'name'"""
        return f"{self.name}:{self.alias}" if self.alias else f"{self.name}"

    def __str__(self):
        alias_str = f" as {self.alias}" if self.alias else ""
        with_str = f" with {self.args}, " if self.args else ""
        return f"ImportToolsetStatement(name={self.name}{alias_str}, {with_str}elements={self.elements})"


class PlanQualifierEnum(Enum):
    FROZEN = "frozen"
    DRAFTED = "drafted"
    UNDEFINED = "undefined"
    NONE = "none"


plan_qualifier_map = {
    "frozen": PlanQualifierEnum.FROZEN,
    "drafted": PlanQualifierEnum.DRAFTED,
    "undefined": PlanQualifierEnum.UNDEFINED,
    "None": PlanQualifierEnum.NONE,
}

plan_qualifier_ordered_map = {
    PlanQualifierEnum.FROZEN: 2,
    PlanQualifierEnum.DRAFTED: 1,
    PlanQualifierEnum.UNDEFINED: 0,
    PlanQualifierEnum.NONE: -1,
}


class PlanBlock(BlockStatement):
    def __init__(
        self,
        action_inputs: list[ActionInput],
        action_outputs: list[ActionOutput],
        body: list[Statement] | None,
        meta: dict[str, Meta],
    ):
        super().__init__(meta)
        self.input = action_inputs
        self.output = action_outputs

        if body:
            self.children = body

        self.qualifier = None
        self.qualifier = self.get_qualifier()

        if self.is_not_valid_qualifier():
            raise NemantixException(
                "Plan/deliberate completion qualifiers must specify "
                "increasing completion levels (e.g drafted->frozen). "
                f"Completion '{self.qualifier[0].value}->{self.qualifier[1].value}' "
                f"is not allowed."
            )

    def __str__(self):
        children_str = "\n     - ".join(str(c) for c in self.children)
        qual_str = f",[{self.qualifier}]" if self.qualifier else ""
        return f"Plan{qual_str} with input={self.input}, output={self.output}\n {children_str}"


class Deliberate(BlockStatement):
    def __init__(
        self,
        name: str,
        when: MicroPrompt,
        guidelines: MicroPrompt,
        plan: PlanBlock,
        meta: dict[str, Meta | None],
        generated_actions: list[ActionBlock] = None,
    ):
        super().__init__(meta)
        self.when = when
        self.guidelines = guidelines
        self.name = name
        self.generated_actions = generated_actions if generated_actions else []
        self.qualifier = None
        self.qualifier = self.get_qualifier()

        if self.is_not_valid_qualifier():
            raise NemantixException(
                "Plan/deliberate completion qualifiers must specify increasing"
                " completion levels (e.g drafted->frozen). "
                f"Completion '{self.qualifier[0].value}->{self.qualifier[1].value}' "
                f"is not allowed."
            )
        if plan:
            if (
                self.qualifier is not None
                and plan.qualifier is not None
                and (
                    plan.qualifier[0] != self.qualifier[0]
                    or plan.qualifier[1] != self.qualifier[1]
                )
            ):
                raise NemantixException(
                    "Deliberate and its plan completion qualifier must be the same or you can specify only one of them."
                )

            if plan.qualifier is None and self.qualifier is not None:
                plan.qualifier = self.qualifier  # copy qualifier to plan

            if plan.qualifier is not None and self.qualifier is None:
                self.qualifier = plan.qualifier  # copy qualifier from plan

            self.children.append(plan)

    def get_plan(self) -> PlanBlock | None:
        return self.children[0] if len(self.children) > 0 else None

    def add_actions(self, actions=list[ActionBlock] | ActionBlock):
        if isinstance(actions, list):
            self.generated_actions.extend(actions)
        if isinstance(actions, ActionBlock):
            self.generated_actions.append(actions)

    def __str__(self):
        name = f"'{self.name}' " if self.name is not None else ""
        children_str = "\n     - ".join(str(c) for c in self.children)
        actions = (
            "actions:" + str([str(action) for action in self.generated_actions])
            if self.generated_actions is not None
            else ""
        )
        return (
            f"Deliberate: {name}when={self.when}, "
            f"guidelines={self.guidelines} "
            f"{actions}\n"
            f"\n   {children_str}"
        )


# =============================================================================
# Frames and Slots
# =============================================================================
class SlotTypesEnum(Enum):
    INT = auto()
    BOOL = auto()
    FLOAT = auto()
    TEXT = auto()
    ENUM = auto()
    STRUCT = auto()
    FRAME = auto()


slot_types_map = {
    "INT": SlotTypesEnum.INT,
    "BOOL": SlotTypesEnum.BOOL,
    "FLOAT": SlotTypesEnum.FLOAT,
    "TEXT": SlotTypesEnum.TEXT,
    "ENUM": SlotTypesEnum.ENUM,
    "STRUCT": SlotTypesEnum.STRUCT,
    "FRAME": SlotTypesEnum.FRAME,
}


class Slot(LeafStatement):
    def __init__(
        self,
        name: str | None,
        types: list[SlotTypesEnum] | None,
        card: str | None,
        prompt: MicroPrompt | None,
        meta: dict[str, Meta | None],
    ):
        super().__init__(meta)
        self.name = name
        self.types = types
        self.cardinality = card
        self.prompt = prompt

    def __str__(self):
        types_str = (
            f" of type {[t for t in self.types]}" if self.types is not None else ""
        )
        card_str = (
            f" and cardinality {self.cardinality}"
            if self.cardinality is not None
            else ""
        )
        prompt_str = f" prompt={self.prompt}" if self.prompt is not None else ""
        return f"Slot '{self.name}'{types_str}{card_str}{prompt_str}"


class Frame(BlockStatement):
    def __init__(self, name: str | None, meta: dict[str, Meta | None]):
        super().__init__(meta)
        self.name = name

    def __str__(self):
        children_str = "\n  - ".join(str(c) for c in self.children)
        name = f"'{self.name}' " if self.name is not None else ""
        return f"Frame {name}: \n{children_str}"
