from __future__ import annotations

import functools
import math
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Type, Union

import numpy as np
import numpy.typing as npt
from lark import Token
from pydantic import BaseModel, create_model

from nemantix.common import context
from nemantix.common.logger import get_package_logger
from nemantix.core import custom_types as nmx_types
from nemantix.core import exceptions as nmx_ex
from nemantix.core import node as nmx_nodes
from nemantix.core import runtime as nmx_runtime
from nemantix.core.expertise import Expertise
from nemantix.core.node import (
    BinaryOperationEnum,
    BuiltinFunctionEnum,
    Deliberate,
    Expression,
    SimilarityEnum,
    SimilarityQualifierEnum,
    SlotTypesEnum,
    UnaryOperationEnum,
    VariableTypeEnum,
)
from nemantix.core.parser import AsFrame
from nemantix.core.prompt import (
    LEFT_SEM_INCL_PROMPT,
    RIGHT_SEM_INCL_PROMPT,
    SCHEMA_APPLY_PROMPT,
    SEM_INCL_TEMPLATE,
)
from nemantix.core.runtime import Builtin, Struct
from nemantix.core.script import Script
from nemantix.core.tools import Toolset
from nemantix.hub import Event, EventType
from nemantix.llm import (
    AbstractLLMProxy,
    LLMProxyConfig,
    LLMResponse,
    StructuredLLMResponse,
)

if TYPE_CHECKING:
    from nemantix.knowledge_base.core.nemantix_knowledge_base import (
        NemantixKnowledgeBase,
    )


logger = get_package_logger(__name__)

BUILTIN_FUNCTIONS = {
    # misc
    BuiltinFunctionEnum.PRINT: Builtin.print,
    BuiltinFunctionEnum.EXISTS: Builtin.exists,
    BuiltinFunctionEnum.TYPE: Builtin.type,
    BuiltinFunctionEnum.COALESCE: Builtin.coalesce,
    BuiltinFunctionEnum.LLM: Builtin.ask_llm,
    BuiltinFunctionEnum.SIZE: Builtin.size,
    BuiltinFunctionEnum.SUBSTRING: Builtin.substring,
    # knowledge base
    BuiltinFunctionEnum.RETRIEVE: Builtin.retrieve,
    BuiltinFunctionEnum.EXPAND: Builtin.expand,
    BuiltinFunctionEnum.EXTEND: Builtin.extend,
    BuiltinFunctionEnum.GENERALIZE: Builtin.generalize,
    # conversions
    BuiltinFunctionEnum.BOOL: Builtin.bool,
    BuiltinFunctionEnum.NUM: Builtin.num,
    BuiltinFunctionEnum.STR: Builtin.str,
    BuiltinFunctionEnum.TO_BOOL: Builtin.to_bool,
    BuiltinFunctionEnum.TO_NUM: Builtin.to_num,
    BuiltinFunctionEnum.TO_STR: Builtin.to_str,
    # math
    BuiltinFunctionEnum.SIN: Builtin.sin,
    BuiltinFunctionEnum.COS: Builtin.cos,
    BuiltinFunctionEnum.SQRT: Builtin.sqrt,
}


class ReturnType(Enum):
    NONE = (0,)
    RETURN = (1,)
    BREAK = (2,)
    CONTINUE = 3

    def is_return(self) -> bool:
        return self == ReturnType.RETURN

    def is_break(self):
        return self == ReturnType.BREAK

    def is_continue(self):
        return self == ReturnType.CONTINUE

    def __bool__(self):
        return self != ReturnType.NONE


# TODO: handle verifiable micro-prompts
class Interpreter:
    """Nemantix interpreter"""

    _SPECIAL_VARS = set()

    class InterpretationContext:
        def __init__(self):
            self.actions = {}
            self.frames = nmx_runtime.Frames()
            self.schemas: dict[str, type[BaseModel]] = {}
            self.tools = nmx_runtime.Tools()
            self.toolsets: set[str] = set()
            self.env = nmx_runtime.OperationalEnv()

    class SimilaritySchema(BaseModel):
        """LLM response schema used by similarity and semantic inclusion operators"""

        holds: bool
        score: float

    class CallEvent:
        def __init__(
            self,
            interpreter: "Interpreter",
            stmt,
            name: str = None,
            kind: str = None,
            args: list | None = None,
            kwargs: dict | None = None,
        ):
            self.interpreter = interpreter
            self.stmt = stmt
            self.callable_name = name
            self.callable_type = kind
            self.callable_prompt = ""

            if isinstance(args, (list, tuple)):
                if len(args) == 0:
                    prompt = (kwargs or {}).get("prompt", "")
                else:
                    prompt = args[0] or ""

                llm_prompt = prompt if name == "llm" and isinstance(prompt, str) else ""
                self.callable_prompt = llm_prompt

            if isinstance(stmt, nmx_nodes.Deliberate):
                self.callable_name = stmt.name
                self.callable_type = "deliberate"

            elif isinstance(stmt, nmx_nodes.PlanBlock):
                assert isinstance(self.callable_name, str)
                self.callable_type = "plan"
            else:
                assert isinstance(self.callable_name, str)
                assert isinstance(self.callable_type, str)

        def __enter__(self):
            self.interpreter._emit_call_enter(
                stmt=self.stmt,
                callable_name=self.callable_name,
                callable_type=self.callable_type,
                callable_prompt=self.callable_prompt,
            )

        def __exit__(self, *_):
            self.interpreter._emit_call_exit(stmt=self.stmt)

    # TODO: deprecate "llm" argument
    def __init__(
        self,
        expertise: Expertise,
        proxy_config: LLMProxyConfig,
        llm: AbstractLLMProxy | None = None,
        embedder=None,
        knowledge_base=None,
        external_variables: nmx_runtime.ExternalVariables | None = None,
        agent_state: Optional[nmx_runtime.Struct] = None,
    ):
        # operational memory
        self.metadata = nmx_runtime.Metadata()
        self.expertise = expertise
        self.context = self.InterpretationContext()

        # globals: file scope
        self.globals = nmx_runtime.get_globals()
        self.globals["__scope"] = []

        # inner models
        self.embedder = embedder
        self.proxies = proxy_config
        self.llm = llm or self.proxies.internal

        self.knowledge_base = knowledge_base
        self.external_vars = external_variables or nmx_runtime.ExternalVariables()

        if agent_state is None:
            agent_state = nmx_runtime.Struct()

        elif not isinstance(agent_state, nmx_runtime.Struct):
            logger.warning(
                f'Given "agent_state" is not a Struct but a "{type(agent_state)}"! '
                f"Ignoring."
            )
            agent_state = nmx_runtime.Struct()

        assert isinstance(agent_state, nmx_runtime.Struct)
        self.agent_state: nmx_runtime.Struct = agent_state
        self._set_special_variables()

    def interpret_coded_request(
        self,
        script: Script,
        request_deliberate: Deliberate,
        user_inputs: Expression | None = None,
    ):
        self._build_context(script, deliberate=request_deliberate)
        outputs = self.interpret_deliberate(request_deliberate, user_inputs=user_inputs)

        self.context.frames.print()
        self.metadata.print()

        return outputs

    def interpret(
        self, deliberate: Deliberate, user_inputs: Optional[nmx_nodes.Expression] = None
    ):
        deliberate_name = deliberate.name
        script = self.expertise.get_script_from_deliberate(deliberate_name)

        self._build_context(script, deliberate)
        outputs = self.interpret_deliberate(deliberate, user_inputs=user_inputs)

        self.context.frames.print()
        self.metadata.print()

        return outputs

    def interpret_tool_declaration(self, declaration: nmx_nodes.PythonToolDeclaration):
        self.interpret_intentable(metadata=declaration.meta, stmt=declaration)
        code = declaration.prompt.prompt.strip()

        if code.find("class ") == -1:
            err_msg = f"Malformed tool declaration (generation failed completely): {declaration.name}"
            raise self._runtime_exception(err_msg, statement=declaration)

        logger.debug(f'Exec on "\n{code}\n"')

        try:
            exec(code, self.globals)

        except Exception:
            err_msg = f'Invalid code for toolset "{declaration.name}"!'
            raise self._runtime_exception(err_msg, statement=declaration)

    def interpret_intentable(
        self,
        metadata: dict[str, nmx_nodes.Meta | None],
        stmt: nmx_nodes.Statement = None,
        name_override: str | None = None,
    ):
        meta = metadata.get("node_meta", None)
        if meta is None:
            return

        assert isinstance(meta, nmx_nodes.NodeMeta)
        label = meta.label
        intentable = self.metadata.get(label, nmx_runtime.Metadata())

        for annotation in meta.annotations:
            key = annotation.name.lower()
            value = annotation.value

            if key in ["goal", "completion", "audience", "style"]:
                key = f"intent.{key}"

            if isinstance(value, nmx_nodes.MicroPrompt):
                logger.warning(
                    f'Ignoring intentable in micro-prompt "{value.prompt}"..'
                )

            elif isinstance(value, list):
                if key != "intent.completion":
                    err = f'Intentable "{key}" does not support "{value}" as value!'
                    raise self._runtime_exception(err)

                if len(value) == 1:
                    value = value[0]

                elif len(value) == 2:
                    a = value[0] if isinstance(value[0], str) else "_"
                    b = value[1] if isinstance(value[1], str) else "_"
                    value = f"{a} -> {b}"
                else:
                    assert 0 <= len(value) <= 2

            elif isinstance(value, nmx_nodes.SingleValue):
                value = self.unbox_value(value)
            elif isinstance(value, nmx_nodes.Expression):
                value = self.interpret_expression(value)
            else:
                assert value is None or isinstance(value, str)
                value = value

            intentable[key] = value

            if annotation.name == "breakpoint" and stmt is not None:
                if annotation.value is not None and not Builtin.to_bool(value):
                    continue

                self._emit_event(
                    stmt,
                    event_type=EventType.BREAKPOINT,
                    payload=dict(interpreter=self),
                )

            if annotation.name == "profile" and stmt is not None:
                if annotation.value is not None and not Builtin.to_bool(value):
                    continue

                profile_name = (
                    name_override
                    if name_override is not None
                    else getattr(stmt, "name", None)
                )
                self._emit_event(
                    stmt,
                    event_type=EventType.PROFILE_MARK,
                    payload=dict(name=profile_name),
                )

        # TODO: if label is None, intentables are overwritten
        self.metadata[label] = intentable

    def interpret_frame(self, frame: nmx_nodes.Frame):
        self.interpret_intentable(metadata=frame.meta, stmt=frame)

        assert isinstance(frame.name, str)
        if frame.name in self.context.frames:
            err_msg = f'Frame "{frame.name}" already defined!'
            raise self._runtime_exception(err_msg, statement=frame)

        frame_ = nmx_runtime.Frame(name=frame.name)
        for slot in frame.children:
            self.interpret_intentable(metadata=slot.meta, stmt=slot)

            if isinstance(slot, nmx_nodes.Frame):
                inner_frame = self.interpret_frame(frame=slot)
                frame_.add_frame(inner_frame)

            elif isinstance(slot, nmx_nodes.Slot):
                types = []

                for t in slot.types or []:
                    if isinstance(t, dict):
                        frame_type, frame_name = list(t.items())[0]
                        types.append(dict(type=frame_type, name=frame_name))
                    else:
                        assert isinstance(t, nmx_nodes.SlotTypesEnum)
                        types.append(dict(name=t))

                assert isinstance(slot.name, str)
                frame_.add_slot(
                    name=slot.name, cardinality=slot.cardinality, types=types
                )
            else:
                logger.warning(f"Skipping: {slot}")

        return frame_

    def interpret_imports(self, imports: list[nmx_nodes.ImportToolsetStatement]):
        for import_stmt in imports:
            self.interpret_intentable(metadata=import_stmt.meta, stmt=import_stmt)

            tool_class = import_stmt.name
            tool_alias = import_stmt.alias
            arguments = None

            if import_stmt.args is not None:
                arguments = [self.interpret_expression(expression=import_stmt.args)]

                if len(arguments) == 1 and isinstance(arguments[0], nmx_runtime.Struct):
                    arguments, _ = arguments[0].to_args_and_kwargs()

                arguments = nmx_runtime.Opaque.unbox_in(arguments)

            elements = import_stmt.elements
            if elements == "*" or elements == ["*"]:
                toolset_class = None

                for tool_cls in Toolset.get_registered_classes():
                    if tool_cls.__name__ == tool_class:
                        toolset_class = tool_cls
                        break

                if toolset_class is None:
                    try:
                        Toolset.load(tool_class)
                        toolset_class = Toolset._classes.get(tool_class)
                    except nmx_ex.NemantixException as exc:
                        raise self._runtime_exception(str(exc)) from exc

                if toolset_class is None:
                    raise self._runtime_exception(f"No toolset {tool_class}!")

                elements = toolset_class.get_tool_names()

            elif tool_class not in Toolset._classes:
                try:
                    Toolset.load(tool_class)
                except nmx_ex.NemantixException as exc:
                    raise self._runtime_exception(str(exc)) from exc

            for tool in elements:
                if isinstance(tool_alias, str):
                    tool_name = f"{tool_alias}.{tool}"
                    Toolset.register_alias(tool_class, tool_name=tool, alias=tool_alias)
                else:
                    tool_name = f"{tool_class}.{tool}"

                if tool_name in Toolset.REGISTRY:
                    if tool_name not in self.context.tools:
                        self.context.tools[tool_name] = Toolset.get_tool(
                            tool_name,
                            instance_alias=tool_alias,
                            instance_args=arguments,
                        )
                    else:
                        logger.info(f'Tool "{tool_name}" already imported!')

                    if tool not in self.context.tools:
                        self.context.tools[tool] = self.context.tools[tool_name]
                else:
                    self._emit_error(
                        stmt=import_stmt, error=f'Tool "{tool_name}" not defined!'
                    )
                    raise self._runtime_exception(
                        action_or_tool=tool_name,
                        statement=import_stmt,
                        cls=nmx_ex.NemantixImportException,
                        emit=False,
                    )

    def interpret_deliberate(
        self,
        deliberate: nmx_nodes.Deliberate,
        user_inputs: Optional[nmx_nodes.Expression] = None,
    ):
        self._set_global_deliberate(deliberate)
        self._push_scope(scope=deliberate.name)

        with self.CallEvent(self, stmt=deliberate):
            self.interpret_intentable(metadata=deliberate.meta, stmt=deliberate)

            plan = deliberate.get_plan()
            assert plan is not None

            result = self.interpret_plan(
                plan, inputs=self._unpack_user_inputs(user_inputs)
            )

        self._pop_scope()
        return result

    def interpret_plan(self, plan: nmx_nodes.PlanBlock, inputs: Optional[list] = None):
        deliberate = self._get_global_deliberate()
        plan_name = f"{deliberate.name}::plan" if deliberate else "plan"

        with self.CallEvent(self, stmt=plan, name=plan_name):
            self.interpret_intentable(
                metadata=plan.meta, stmt=plan, name_override=plan_name
            )
            result = self.interpret_block(block=plan, args=inputs)

        return result

    def interpret_block(
        self,
        block: nmx_types.PlanOrActionBlock,
        args: Optional[list] = None,
        callee: nmx_nodes.DoStatement | None = None,
    ):
        self.interpret_intentable(metadata=block.meta, stmt=block)
        is_plan = isinstance(block, nmx_nodes.PlanBlock)
        if not is_plan:
            prev_env = self.context.env

        try:
            if not is_plan:
                self.context.env = nmx_runtime.OperationalEnv()
                self._push_scope(scope=block.name or "block")

            return_value = None
            explicit_return = False

            self._set_special_variables()
            self._set_block_inputs(block, args, callee=callee)

            for statement in block.children or []:
                # ignore 'break' and 'continue' at action level (i.e., not nested)
                if isinstance(statement, (nmx_nodes.Break, nmx_nodes.Continue)):
                    logger.warning(f"Ignoring: {statement}")
                    continue

                stmt_value, return_type = self.interpret_statement(statement)

                if return_type.is_return():
                    return_value = stmt_value
                    explicit_return = True
                    break

                if return_type.is_break():
                    break

            # Handle implicit return from 'out' block
            if not explicit_return and block.output:
                extracted_values = []

                for out_arg in block.output:
                    if out_arg.name:
                        # Pull the value from the action's local memory scope
                        extracted_values.append(self.context.env.get(out_arg.name))

                # Bind the extracted values to the return_value
                if len(extracted_values) == 1:
                    return_value = extracted_values[0]
                elif len(extracted_values) > 1:
                    return_value = extracted_values

        finally:
            if not is_plan:
                self.context.env = prev_env
                self._pop_scope()

        return return_value

    def interpret_statement(
        self, statement: nmx_nodes.Statement
    ) -> tuple[Any, ReturnType]:
        return_type = ReturnType.NONE
        return_value = None

        self.interpret_intentable(metadata=statement.meta, stmt=statement)

        if isinstance(statement, nmx_nodes.ConditionBlock):
            return_value, return_type = self.interpret_conditional(
                conditional=statement
            )

            if return_type.is_return():
                return return_value, return_type

            if return_type.is_break() or return_type.is_continue():
                # NOTE: continue statement is ignored at action-level
                return None, return_type

        elif isinstance(statement, nmx_nodes.Assignment):
            self.interpret_assignment(assignment=statement)

        # loops
        elif isinstance(statement, nmx_nodes.RepeatTimesBlock):
            self._push_scope(scope="repeat_times")
            num_iterations = statement.times
            iter_var = statement.as_vars
            should_set_iter_var = True

            if isinstance(iter_var, list) and len(iter_var) > 0:
                iter_var = iter_var[0]
            else:
                if not isinstance(iter_var, str):
                    should_set_iter_var = False

            for iter_num in range(num_iterations):
                if should_set_iter_var:
                    self.context.env.set(iter_var, value=iter_num)

                return_value, return_type = self.eval_loop_block(
                    statements=statement.children
                )

                if return_type.is_return():
                    self._pop_scope()
                    return return_value, return_type

                if return_type.is_break():
                    return_type = ReturnType.NONE
                    break

            self._pop_scope()

        elif isinstance(statement, nmx_nodes.RepeatEachBlock):
            idx_var_name, item_var_name = statement.as_vars

            assert isinstance(statement.each, nmx_nodes.Expression), (
                f"Loop iterable must be an Expression, got {type(statement.each).__name__}, in {statement}"
            )

            self._push_scope(scope="repeat_each")
            iterator = self.interpret_expression(expression=statement.each)
            assert isinstance(iterator, Iterable)

            for i, item in enumerate(iterator):
                self.context.env.set(var_name=idx_var_name, value=i)
                self.context.env.set(var_name=item_var_name, value=item)

                return_value, return_type = self.eval_loop_block(
                    statements=statement.children
                )

                if return_type.is_return():
                    self._pop_scope()
                    return return_value, return_type

                if return_type.is_break():
                    return_type = ReturnType.NONE
                    break

            self._pop_scope()

        elif isinstance(
            statement, (nmx_nodes.RepeatWhileBlock, nmx_nodes.RepeatUntilBlock)
        ):
            num_iterations = 0
            is_repeat_until = isinstance(statement, nmx_nodes.RepeatUntilBlock)
            scope_pushed = False

            if isinstance(statement.max, (int, float)):
                max_iterations = int(statement.max)
            else:
                max_iterations = self.interpret_expression(expression=statement.max)
                max_iterations = Builtin.to_num(max_iterations)

            assert isinstance(statement.condition, nmx_nodes.Expression), (
                f"Loop condition must be an Expression, got {type(statement.condition).__name__}"
            )

            while True:
                condition = self.interpret_expression(expression=statement.condition)

                if is_repeat_until:
                    condition = not condition

                if not condition:
                    break

                if not scope_pushed:
                    if is_repeat_until:
                        self._push_scope("repeat_until")
                    else:
                        self._push_scope("repeat_while")

                    scope_pushed = True

                num_iterations += 1
                return_value, return_type = self.eval_loop_block(
                    statements=statement.children
                )

                if return_type.is_return():
                    self._pop_scope()
                    return return_value, return_type

                if return_type.is_break():
                    return_type = ReturnType.NONE
                    break

                if num_iterations >= max_iterations:
                    break

            self._pop_scope()

        # builtins
        elif isinstance(statement, nmx_nodes.DoStatement):
            self.interpret_do_statement(do=statement)

        # return statements: return, break, continue
        elif isinstance(statement, nmx_nodes.Return):
            self._emit_line(stmt=statement)
            return_value = [
                self.interpret_expression(expression=v) for v in statement.val
            ]

            if len(return_value) == 1:
                return_value = return_value[0]

            return_type = ReturnType.RETURN

        elif isinstance(statement, nmx_nodes.Break):
            self._emit_line(stmt=statement)
            return_type = ReturnType.BREAK

        elif isinstance(statement, nmx_nodes.Continue):
            self._emit_line(stmt=statement)
            return_type = ReturnType.CONTINUE

        return return_value, return_type

    def interpret_do_statement(self, do: nmx_nodes.DoStatement):
        self._emit_line(stmt=do)

        is_action_call = False
        is_tool_call = False
        fn_name = do.name
        callable_type = do.callable_type

        # qualified do statement
        if callable_type == nmx_nodes.CallableTypeEnum.ACTION:
            # action lookup
            action_info = self.context.actions.get(fn_name, {})
            callable_fn = action_info.get("closure", None)
            is_action_call = True

            if callable_fn is None:
                err_msg = f'[do action] No action named "{fn_name}"!'
                raise self._runtime_exception(err_msg, statement=do)

            current_deliberate = self._get_global_deliberate()
            assert current_deliberate is not None

            is_global_action = action_info.get("is_global", False)
            is_imported = current_deliberate.name in action_info.get(
                "imported_by", set()
            )

            if not is_global_action and not is_imported:
                err_msg = (
                    f'Private action "{fn_name}" cannot be called from deliberate '
                    f'"{current_deliberate.name}"!"'
                )
                raise self._runtime_exception(err_msg, statement=do)

        elif callable_type == nmx_nodes.CallableTypeEnum.TOOL:
            # lookup tool
            callable_fn = self.context.tools.get(fn_name)

            if callable_fn is None:
                err_msg = f'[do tool] No tool named "{fn_name}"!'
                raise self._runtime_exception(err_msg, statement=do)

            is_tool_call = True
        else:
            # unqualified do statement
            # lookup action, tool, and builtin
            action_info = self.context.actions.get(fn_name, {})
            callable_fn = action_info.get("closure", None)

            if callable_fn is None:
                callable_fn = self.context.tools.get(fn_name)

                if callable_fn is None:
                    builtin_name = nmx_nodes.builtin_func_map.get(fn_name, None)
                    callable_fn = BUILTIN_FUNCTIONS.get(builtin_name, None)

                    if builtin_name == BuiltinFunctionEnum.LLM:

                        def __llm_call(prompt, *_, **kwargs_):
                            response = Builtin.ask_llm(
                                self.proxies.external, prompt, **kwargs_
                            )
                            self._emit_llm(
                                stmt=do, prompt=prompt, llm_response=response
                            )
                            return response.text

                        callable_fn = __llm_call

                    elif builtin_name == BuiltinFunctionEnum.PRINT:

                        def __print_call(*args_, **kwargs_):
                            Builtin.print(*args_, **kwargs_)
                            parts = ["<NONE>" if a is None else str(a) for a in args_]

                            if kwargs_:
                                parts.append(str(kwargs_))

                            self._emit_event(
                                do,
                                EventType.OUTPUT,
                                payload={"text": " ".join(parts) + "\n"},
                            )

                        callable_fn = __print_call

                    elif builtin_name == BuiltinFunctionEnum.RETRIEVE:

                        def __retrieve(*args_, **kwargs_):
                            self._emit_retrieve(
                                stmt=do,
                                knowledge_base=self.knowledge_base,
                                query=kwargs_.get("query", args_[0]),
                            )

                            return Builtin.retrieve(
                                self.knowledge_base, *args_, **kwargs_
                            )

                        callable_fn = __retrieve

                    elif builtin_name == BuiltinFunctionEnum.EXPAND:

                        def __expand(*args_, **kwargs_):
                            self._emit_expand(
                                stmt=do,
                                knowledge_base=self.knowledge_base,
                                node_id=kwargs_.get("node_id", args_[0]),
                            )

                            return Builtin.expand(
                                self.knowledge_base, *args_, **kwargs_
                            )

                        callable_fn = __expand

                    elif builtin_name == BuiltinFunctionEnum.EXTEND:

                        def __extend(*args_, **kwargs_):
                            self._emit_extend(
                                stmt=do,
                                knowledge_base=self.knowledge_base,
                                node_id=kwargs_.get("node_id", args_[0]),
                            )

                            return Builtin.extend(
                                self.knowledge_base, *args_, **kwargs_
                            )

                        callable_fn = __extend

                    elif builtin_name == BuiltinFunctionEnum.GENERALIZE:

                        def __generalize(*args_, **kwargs_):
                            self._emit_generalize(
                                stmt=do,
                                knowledge_base=self.knowledge_base,
                                node_id=kwargs_.get("node_id", args_[0]),
                            )

                            return Builtin.generalize(
                                self.knowledge_base, *args_, **kwargs_
                            )

                        callable_fn = __generalize

                    if callable_fn is None:
                        err_msg = f'No action, tool, or builtin named "{fn_name}"!'
                        raise self._runtime_exception(err_msg, statement=do)
                else:
                    is_tool_call = True
            else:
                is_action_call = True

        # function call
        if is_action_call:
            kind = "action"
        elif is_tool_call:
            kind = "tool"
        else:
            kind = "builtin"

        outputs = do.producing
        args, kwargs = self._parse_do_using(do=do)

        with self.CallEvent(
            self, stmt=do, name=fn_name, kind=kind, args=args, kwargs=kwargs
        ):
            # LLM with schema call
            if fn_name == "llm" and getattr(do, "producing_schema", None):
                frame_name = do.producing_schema

                if not isinstance(frame_name, str):
                    raise self._runtime_exception(
                        "Generative schema blocks are not yet supported for execution.",
                        statement=do,
                    )
                try:
                    if frame_name not in self.context.schemas:
                        pydantic_schema = self._frame_to_pydantic_schema(
                            frame_name, statement=do
                        )
                        self.context.schemas[frame_name] = pydantic_schema
                    else:
                        pydantic_schema = self.context.schemas[frame_name]

                except ValueError as e:
                    raise self._runtime_exception(str(e), statement=do)

                # Extract prompt string
                if len(args) == 1:
                    if isinstance(args[0], str):
                        prompt_ = args[0]

                    elif isinstance(args[0], Struct):
                        prompt_ = args[0].get("prompt", str(args[0]))
                    else:
                        prompt_ = None
                else:
                    prompt_ = kwargs.get("prompt", None)

                if not isinstance(prompt_, str):
                    err_msg = (
                        f"Provided input prompt is not a string "
                        f'but a "{Builtin.type(prompt_)}"!'
                    )
                    raise self._runtime_exception(err_msg, statement=do)

                structured = self.proxies.external.invoke_structured(
                    prompt_, schema=pydantic_schema
                )
                self._emit_llm(
                    stmt=do,
                    prompt=prompt_,
                    schema=pydantic_schema,
                    llm_response=structured,
                )
                result = structured.result.model_dump()
            else:
                callable_fn = self._wrap_callable_with_try_except(
                    fn=callable_fn, statement=do
                )

                if is_action_call:
                    # Pass arguments as-is (Struct, list, or scalar).
                    # The action's _set_action_inputs will handle unpacking and kwarg validation.
                    result = callable_fn((args, kwargs), callee=do)
                else:
                    # Handle Tools and Builtins (External calls)
                    if is_tool_call:
                        args = nmx_runtime.Opaque.unbox_in_list(args)
                        kwargs = nmx_runtime.Opaque.unbox_in_dict(kwargs)

                        args = nmx_runtime.Struct.unbox_in_list(args)
                        kwargs = nmx_runtime.Struct.unbox_in_dict(kwargs)

                    if len(args) == 0:
                        result = callable_fn(**kwargs)

                    elif len(kwargs) == 0:
                        result = callable_fn(*args)
                    else:
                        result = callable_fn(*args, **kwargs)

            producing_schema = getattr(do, "producing_schema", None)
            schema_applied = False

            if isinstance(producing_schema, str) and outputs is not None:
                result = self._apply_frame_schema(do, result, producing_schema)
                schema_applied = True

            if isinstance(outputs, nmx_nodes.Variable):
                assert isinstance(outputs.name, str)
                packed_value = self.pack_return_value(value=result)
                self.context.env.set(var_name=outputs.name, value=packed_value)

            elif isinstance(outputs, nmx_nodes.Collection):
                packed_value = self.pack_return_value(value=result)
                assert isinstance(packed_value, nmx_runtime.Struct)

                for i, var in enumerate(outputs.value):
                    assert isinstance(var, nmx_nodes.Variable)
                    assert isinstance(var.name, str)
                    val = (
                        packed_value.get(var.name)
                        if schema_applied
                        else packed_value.get(i)
                    )
                    self.context.env.set(var_name=var.name, value=val)

    def interpret_conditional(
        self, conditional: nmx_nodes.ConditionBlock
    ) -> tuple[Any, ReturnType]:
        for block in conditional.children:
            if isinstance(block, (nmx_nodes.IfBlock, nmx_nodes.ElifBlock)):
                assert isinstance(block.condition, nmx_nodes.Expression), (
                    f"Block condition must be an Expression, got {type(block.condition).__name__}"
                )
                self._emit_line(stmt=block.condition)
                condition = self.interpret_expression(block.condition)

                if condition:
                    if isinstance(block, nmx_nodes.IfBlock):
                        self._push_scope(scope="if")
                    else:
                        self._push_scope(scope="elif")

                    for statement in block.children or []:
                        return_value, return_type = self.interpret_statement(statement)

                        if return_type:
                            self._pop_scope()
                            return return_value, return_type

                    # short-circuit
                    self._pop_scope()
                    break
            else:
                assert isinstance(block, nmx_nodes.ElseBlock)

                self._push_scope(scope="else")
                for statement in block.children or []:
                    return_value, return_type = self.interpret_statement(statement)

                    if return_type:
                        self._pop_scope()
                        return return_value, return_type

                self._pop_scope()

        return None, ReturnType.NONE

    def interpret_assignment(self, assignment: nmx_nodes.Assignment):
        self._emit_line(stmt=assignment)

        var_name = assignment.var.name
        var_path = assignment.var.path
        value = self.interpret_expression(expression=assignment.value)
        assert isinstance(var_name, str)

        if var_name in self._SPECIAL_VARS:
            err_msg = f'Cannot assign special variable "{var_name}"!'
            raise self._runtime_exception(err_msg, statement=assignment)

        if isinstance(var_path, list) and len(var_path) > 0:
            # navigate the path
            path = var_name
            struct = self.context.env.get(var_name)

            if not isinstance(struct, nmx_runtime.Struct):
                actual = nmx_runtime.Builtin.type(struct)
                err_msg = (
                    f'Cannot assign to "[{var_name}]" using a path: '
                    f'"[{var_name}]" is {actual}, not a collection.'
                )
                raise self._runtime_exception(err_msg, statement=assignment)

            for field in var_path[:-1]:
                field = self.unbox_value(field)

                if field not in struct:
                    # create the missing field
                    struct.set(value=nmx_runtime.Struct(), key=field)

                struct = struct[field]
                path = f"{path}.{field}"

                if not isinstance(struct, nmx_runtime.Struct):
                    actual = nmx_runtime.Builtin.type(struct)
                    err_msg = (
                        f'Cannot navigate into "[{path}]": '
                        f'"[{path}]" is {actual}, not a collection.'
                    )
                    raise self._runtime_exception(err_msg, statement=assignment)

            field = self.unbox_value(var_path[-1])
            struct.update_field(key=field, value=value)
            logger.debug(f'"{path}.{field}" = {value}')
        else:
            logger.debug(f'"{var_name}" = {value}')
            self.context.env.set(var_name, value=value)

        return value

    def interpret_expression(self, expression: nmx_nodes.Expression | None):
        if expression is None:
            return None

        if isinstance(expression, list):
            assert len(expression) == 1
            expression = expression[0]

        self.interpret_intentable(metadata=expression.meta, stmt=expression)

        if isinstance(expression, nmx_nodes.Assignment):
            return self.interpret_assignment(assignment=expression)

        elif isinstance(expression, nmx_nodes.SchemedCollection):
            return self.eval_schemed_collection(schemed_collection=expression)

        elif isinstance(expression, nmx_nodes.Collection):
            return self.eval_collection(collection=expression)

        elif isinstance(expression, nmx_nodes.UnaryOperation):
            return self.eval_unary_op(
                operand=expression.operand,
                operation=expression.operation,
                statement=expression,
            )

        elif isinstance(expression, nmx_nodes.BinaryOperation):
            return self.eval_binary_op(
                first=expression.first,
                second=expression.second,
                operation=expression.operation,
                statement=expression,
            )

        elif isinstance(expression, nmx_nodes.SimilarityOperation):
            return self.eval_similarity_op(
                first=expression.first,
                second=expression.second,
                operation=expression.operation,
                qualifier=expression.qualifier,
                statement=expression,
            )

        elif isinstance(expression, nmx_nodes.BuiltinFunction):
            self._push_scope(scope="builtin")

            def __call_builtin(fn):
                value = fn()
                self._emit_call_exit(stmt=expression, trim=True)
                self._pop_scope()
                return value

            if expression.function == BuiltinFunctionEnum.LLM:

                def __llm_call(prompt, *_, **kwargs_):
                    response = Builtin.ask_llm(self.proxies.external, prompt, **kwargs_)
                    self._emit_llm(
                        stmt=expression, prompt=prompt, llm_response=response
                    )
                    return response.text

                function = __llm_call

            elif expression.function == BuiltinFunctionEnum.RETRIEVE:

                def __retrieve(knowledge_base, *_, **kwargs):
                    self._emit_retrieve(
                        stmt=expression, knowledge_base=knowledge_base, **kwargs
                    )
                    return Builtin.retrieve(knowledge_base, **kwargs)

                function = __retrieve

            elif expression.function == BuiltinFunctionEnum.EXPAND:

                def __expand(knowledge_base, *_, **kwargs):
                    self._emit_expand(
                        stmt=expression, knowledge_base=knowledge_base, **kwargs
                    )
                    return Builtin.expand(knowledge_base, **kwargs)

                function = __expand

            elif expression.function == BuiltinFunctionEnum.EXTEND:

                def __extend(knowledge_base, *_, **kwargs):
                    self._emit_extend(
                        stmt=expression, knowledge_base=knowledge_base, **kwargs
                    )
                    return Builtin.extend(knowledge_base, **kwargs)

                function = __extend

            elif expression.function == BuiltinFunctionEnum.GENERALIZE:

                def __generalize(knowledge_base, *_, **kwargs):
                    self._emit_generalize(
                        stmt=expression, knowledge_base=knowledge_base, **kwargs
                    )
                    return Builtin.generalize(knowledge_base, **kwargs)

                function = __generalize
            else:
                function = BUILTIN_FUNCTIONS[expression.function]

            args = [
                self.interpret_expression(expression=arg) for arg in expression.args
            ]

            try:
                _builtin_name = expression.function.name.lower()
                _builtin_prompt = (
                    args[0]
                    if _builtin_name == "llm" and args and isinstance(args[0], str)
                    else ""
                )
                self._emit_line(stmt=expression, trim=True)
                self._emit_call_enter(
                    stmt=expression,
                    trim=True,
                    callable_type="builtin",
                    callable_name=_builtin_name,
                    callable_prompt=_builtin_prompt,
                )

                if len(args) == 0:
                    return __call_builtin(function)

                if len(args) == 1:
                    args = args[0]

                    # TODO: should always to args and kwargs on single struct as input?
                    if (
                        isinstance(args, nmx_runtime.Struct)
                        and args.can_be_seen_as_list()
                    ):
                        args, _ = args.to_args_and_kwargs()
                        # TODO: when calling builtins the number of arguments should match,
                        #  because args may not always be unpacked
                        return __call_builtin(lambda: function(*args))
                    else:
                        return __call_builtin(lambda: function(args))

                return __call_builtin(lambda: function(*args))

            except Exception as e:
                fn_name = expression.function.name
                err_msg = f'"{e}" error in builtin function call "{fn_name}" with arguments "{args}"!'
                raise self._runtime_exception(err_msg, statement=expression)

        elif isinstance(expression, nmx_nodes.MetaExpression):
            # TODO: handle "this" keyword
            label = expression.quals[0]
            if label not in self.metadata:
                err_msg = f'Intentable "{label}" not defined!'
                raise self._runtime_exception(err_msg, statement=expression)

            intentable = self.metadata[label]
            field = ".".join(expression.quals[1:])
            return intentable.get(field, None)

        assert (
            isinstance(expression, (nmx_nodes.Variable, nmx_nodes.SingleValue))
            or isinstance(expression, str)
            or expression is None
        ), (
            f"Expression is not of an unboxable type! Expected either Variable, SingleValue, str or None, got "
            f"{type(expression).__name__} instead."
        )

        return self.unbox_value(expression)

    def eval_loop_block(self, statements: list[nmx_nodes.Statement]):
        continue_flag = False

        for stmt in statements:
            if continue_flag:
                continue

            return_value, return_type = self.interpret_statement(stmt)

            if return_type.is_return():
                return return_value, return_type

            if return_type.is_break():
                return None, return_type

            if return_type.is_continue():
                # set the flag to skip the inner loop's statements
                continue_flag = True
                continue

        return None, ReturnType.NONE

    def eval_schemed_collection(
        self,
        schemed_collection: nmx_nodes.SchemedCollection,
        enclosing_frame: Optional[nmx_runtime.Frame] = None,
    ) -> Struct | None:
        if isinstance(schemed_collection.dataframe, nmx_nodes.Collection):
            frame_name = schemed_collection.value.upper()
            collection = schemed_collection.dataframe
        else:
            assert isinstance(schemed_collection.dataframe, AsFrame)
            frame_name = schemed_collection.dataframe.value.upper()
            collection = schemed_collection.value

        if enclosing_frame is not None:
            if frame_name in enclosing_frame.frames:
                frame: nmx_runtime.Frame = enclosing_frame.frames[frame_name]
            else:
                err_msg = (
                    f'Undefined frame "{frame_name}" in frame "{enclosing_frame.name}"!'
                )
                raise self._runtime_exception(err_msg, statement=schemed_collection)
        else:
            if frame_name not in self.context.frames:
                err_msg = f'Undefined frame "{frame_name}"!'
                raise self._runtime_exception(err_msg, statement=schemed_collection)

            frame: nmx_runtime.Frame = self.context.frames[frame_name]

        struct = self.eval_collection(collection=collection, frame=frame)

        if schemed_collection.apply_type == nmx_nodes.FrameApplyEnum.PRE:
            return frame.apply_prefix(struct)

        return frame.apply_postfix(struct)

    def eval_collection(
        self,
        collection: nmx_nodes.Collection,
        frame: Optional[nmx_runtime.Frame] = None,
    ) -> nmx_runtime.Struct:
        inferred_type = collection.inferred_type
        assert inferred_type == VariableTypeEnum.LIST

        struct = nmx_runtime.Struct()

        if not isinstance(collection.value, list):
            collection.value = [collection.value]

        for value in collection.value:
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, nmx_nodes.SchemedCollection):
                        v = self.eval_schemed_collection(
                            schemed_collection=v, enclosing_frame=frame
                        )
                    elif isinstance(v, nmx_nodes.Collection):
                        v = self.eval_collection(collection=v)
                    else:
                        v = self.interpret_expression(v)

                    struct.set(value=v, key=k)
            else:
                if isinstance(value, nmx_nodes.SchemedCollection):
                    value_ = self.eval_schemed_collection(
                        schemed_collection=value, enclosing_frame=frame
                    )
                elif isinstance(value, nmx_nodes.Collection):
                    value_ = self.eval_collection(collection=value)
                else:
                    value_ = self.interpret_expression(value)

                struct.set(value_)

        return struct

    def eval_similarity_op(
        self,
        first: nmx_nodes.Expression,
        second: nmx_nodes.Expression,
        operation: SimilarityEnum,
        qualifier: nmx_types.SimilarityQualifier | None,
        statement: nmx_nodes.Statement | None = None,
    ):
        a = self.interpret_expression(expression=first)
        b = self.interpret_expression(expression=second)

        def __nmx_similarity_exception() -> nmx_ex.NemantixOperationException:
            error = nmx_ex.NemantixOperationException(
                operand=(a, b),
                operation_name=operation.name,
                statement=statement,
                script=self._get_global_script(),
            )
            self._emit_error(statement, error=error.message)
            return error

        qualifier = qualifier or SimilarityQualifierEnum.CLOSE
        if isinstance(qualifier, tuple):
            qualifier, value = qualifier

            if qualifier == SimilarityQualifierEnum.NUMBER:
                qualifier = Builtin.to_num(value)

        if operation in [SimilarityEnum.SIM, SimilarityEnum.SIM_QUAL]:
            if a is None or b is None:
                raise __nmx_similarity_exception()

            if isinstance(a, (int, float, bool)) or isinstance(b, (int, float, bool)):
                raise __nmx_similarity_exception()

            if (isinstance(a, nmx_runtime.DocRef) and isinstance(b, str)) or (
                isinstance(a, str) and isinstance(b, nmx_runtime.DocRef)
            ):
                if isinstance(a, nmx_runtime.DocRef):
                    doc = a.content
                    query = b
                else:
                    doc = b.content
                    query = a

                similarity = self._compute_similarity(
                    a=doc, b=query, statement=statement
                )

            elif isinstance(a, nmx_runtime.DocRef) and isinstance(
                b, nmx_runtime.DocRef
            ):
                similarity = self._compute_similarity(
                    a=a.content, b=b.content, statement=statement
                )

            elif isinstance(a, nmx_runtime.Struct) and isinstance(b, str):
                return self._filter_by_similarity(
                    struct=a, query=b, qualifier=qualifier
                )

            elif isinstance(a, str) and isinstance(b, nmx_runtime.Struct):
                return self._filter_by_similarity(
                    struct=b, query=a, qualifier=qualifier
                )
            else:
                assert isinstance(a, str) and isinstance(b, str)
                similarity = self._compute_similarity(a, b, statement=statement)

            return self._apply_similarity_qualifier(similarity, qualifier)

        # semantic inclusion
        a = Builtin.to_str(a)
        b = Builtin.to_str(b)

        if operation in [SimilarityEnum.SIM_RIGHT, SimilarityEnum.SIM_QUAL_RIGHT]:
            system_prompt = RIGHT_SEM_INCL_PROMPT
            inclusion_op = f'"{a}" ~> "{b}"'
        else:
            assert operation in [SimilarityEnum.SIM_LEFT, SimilarityEnum.SIM_QUAL_LEFT]
            system_prompt = LEFT_SEM_INCL_PROMPT
            inclusion_op = f'"{a}" <~ "{b}"'

        user_prompt = SEM_INCL_TEMPLATE.format(inclusion_op)
        response = self._semantic_inclusion_with_llm(
            system_prompt, user_prompt, statement=statement
        )
        logger.debug(f"Inclusion score of {inclusion_op} is {response.score}")
        return self._apply_semantic_qualifier(response.score, qualifier)

    def eval_binary_op(
        self,
        first: nmx_nodes.Expression,
        second: nmx_nodes.Expression,
        operation: BinaryOperationEnum,
        statement: nmx_nodes.Statement | None = None,
    ):
        a = self.interpret_expression(expression=first)
        b = self.interpret_expression(expression=second)

        def __nmx_operation_exception() -> nmx_ex.NemantixOperationException:
            error = nmx_ex.NemantixOperationException(
                operand=(a, b),
                operation_name=operation.name,
                statement=statement,
                script=self._get_global_script(),
            )
            self._emit_error(statement, error=error.message)
            return error

        if operation == BinaryOperationEnum.CONCAT:
            if isinstance(a, str) and isinstance(b, str):
                return a + b

            if isinstance(a, nmx_runtime.Struct) and isinstance(b, nmx_runtime.Struct):
                return a.union(b)

            if isinstance(a, nmx_runtime.Struct) or isinstance(b, nmx_runtime.Struct):
                if isinstance(a, nmx_runtime.Struct):
                    return a.append(b)

                return b.append(a)

            raise __nmx_operation_exception()

        if operation == BinaryOperationEnum.FALLBACK:
            if a is not None:
                return a

            return b

        # logical
        if self._is_logical_op(operation):
            if not (isinstance(a, bool) and isinstance(b, bool)):
                raise __nmx_operation_exception()

            if operation == BinaryOperationEnum.LOGICAL_OR:
                return a or b

            if operation == BinaryOperationEnum.LOGICAL_XOR:
                return a ^ b

            if operation == BinaryOperationEnum.LOGICAL_AND:
                return a and b

        # comparisons
        if operation == BinaryOperationEnum.EQ:
            if a is None or b is None:
                return a == b

            if type(a) is type(b):
                return a == b

            raise __nmx_operation_exception()

        if operation == BinaryOperationEnum.NE:
            if a is None or b is None:
                return a != b

            if type(a) is type(b):
                return a != b

            raise __nmx_operation_exception()

        if self._is_comparison_op(operation):
            if type(a) is not type(b):
                raise __nmx_operation_exception()

            if a is None and b is None:
                raise __nmx_operation_exception()

            if operation == BinaryOperationEnum.LT:
                return a < b

            if operation == BinaryOperationEnum.GT:
                return a > b

            if operation == BinaryOperationEnum.LTE:
                return a <= b

            if operation == BinaryOperationEnum.GTE:
                return a >= b

        # arithmetic
        assert self._is_arithmetic_op(operation)

        if isinstance(a, bool) or isinstance(b, bool):
            raise __nmx_operation_exception()

        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise __nmx_operation_exception()

        if operation == BinaryOperationEnum.ADD:
            return a + b

        if operation == BinaryOperationEnum.SUB:
            return a - b

        if operation == BinaryOperationEnum.MUL:
            return a * b

        if operation == BinaryOperationEnum.DIV:
            if b == 0:
                return None

            return a / b

        if operation == BinaryOperationEnum.MOD:
            if b == 0:
                return None

            return a % b

        if operation == BinaryOperationEnum.POW:
            return a**b

        return None

    def eval_unary_op(
        self,
        operand: nmx_nodes.Expression,
        operation: UnaryOperationEnum,
        statement: nmx_nodes.Statement | None = None,
    ):
        value = self.interpret_expression(expression=operand)

        def __nmx_operation_exception() -> nmx_ex.NemantixOperationException:
            error = nmx_ex.NemantixOperationException(
                operand=value, operation_name=operation.name, statement=statement
            )
            self._emit_error(statement, error=error.message)
            return error

        if value is None:
            raise __nmx_operation_exception()

        if operation == UnaryOperationEnum.NEG:
            if isinstance(value, bool):
                raise __nmx_operation_exception()

            if isinstance(value, (int, float)):
                return -value

            raise __nmx_operation_exception()

        if operation == UnaryOperationEnum.POS:
            if isinstance(value, bool):
                raise __nmx_operation_exception()

            if isinstance(value, (int, float)):
                return +value

            raise __nmx_operation_exception()

        assert operation == UnaryOperationEnum.NOT
        if isinstance(value, bool):
            return not value

        raise __nmx_operation_exception()

    def unbox_value(self, value: nmx_nodes.Variable | nmx_nodes.SingleValue | None):
        if value is None:
            return None

        if isinstance(value, nmx_nodes.Variable):
            var = self.context.env.get(value.name)
            value_path = value.path

            if value_path is None:
                value_path = []

            elif isinstance(value_path, str):
                value_path = [value_path]

            if isinstance(var, nmx_runtime.Struct) and len(value_path) > 0:
                struct = var

                # navigate the path
                for name in value_path[:-1]:
                    name = self.unbox_value(name)

                    if name not in struct:
                        var = None
                        break

                    var = struct.get(name)

                    if isinstance(var, nmx_runtime.Struct):
                        struct = var
                        continue

                name = self.unbox_value(value=value_path[-1])
                if len(value_path) == 1:
                    return struct.get(name, None)

                if isinstance(var, nmx_runtime.Struct):
                    return var.get(name, None)

                # R-value match
                return var == name

            elif len(value_path) > 0:
                actual = nmx_runtime.Builtin.type(var)
                err_msg = (
                    f'Cannot read "[{value.name}]" using a path: '
                    f'"[{value.name}]" is {actual}, not a collection.'
                )
                raise self._runtime_exception(err_msg, statement=value)

            return var

        elif isinstance(value, nmx_nodes.SingleValue):
            return self.unbox_token(
                token=value.value, inferred_type=value.inferred_type
            )

        return value.value

    def unbox_token(self, token: str | list, inferred_type: VariableTypeEnum):
        assert not isinstance(token, Token)

        if inferred_type == VariableTypeEnum.NONE:
            return None

        if inferred_type == VariableTypeEnum.INT:
            return int(token)

        if inferred_type == VariableTypeEnum.FLOAT:
            return float(token)

        if inferred_type == VariableTypeEnum.BOOL:
            return bool(token)

        if inferred_type == VariableTypeEnum.FSTRING:
            assert isinstance(token, list)
            values = []

            for value in token:
                if isinstance(value, str):
                    expr = value
                else:
                    expr = self.interpret_expression(expression=value)

                values.append(Builtin.to_str(expr))

            return "".join(values)

        return str(token)

    def pack_return_value(self, value):
        # if primitive; return as it is
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(
            value, (nmx_runtime.Struct, nmx_runtime.DocRef, nmx_runtime.Opaque)
        ):
            # do not box again
            return value

        # for collections wrap each object with an opaque, and create a Struct
        if isinstance(value, (list, tuple, set)):
            struct = nmx_runtime.Struct()

            if isinstance(value, set):
                value = list(value)

            for v in value:
                struct.set(value=self.pack_return_value(v))

            return struct

        elif isinstance(value, dict):
            struct = nmx_runtime.Struct()

            for k, v in value.items():
                struct.set(value=self.pack_return_value(v), key=k)

            return struct

        # value is a python object
        return nmx_runtime.Opaque(obj=value)

    def _semantic_inclusion_with_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        statement: nmx_nodes.Statement | None = None,
    ) -> SimilaritySchema:
        messages = self.llm.messages_from(
            prompts_with_roles=[("system", system_prompt), ("user", user_prompt)]
        )

        result = self.llm.invoke_structured(messages, schema=self.SimilaritySchema)
        self._emit_llm(
            stmt=statement,
            prompt=system_prompt + user_prompt,
            internal=True,
            schema=self.SimilaritySchema,
            llm_response=result,
        )
        response = result.result

        assert isinstance(response, self.SimilaritySchema)
        return response

    def _build_context(self, script: Script, deliberate: Deliberate):
        self._set_global_deliberate(deliberate)
        self._set_global_script(script)

        self._discover_imported_actions(script)
        self._discover_actions(script, deliberate=deliberate)
        self._discover_frames(script)
        self._discover_toolsets_and_imports(script)

    def _discover_frames(self, script: Script):
        for frame in script.frames:
            frame_key = frame.name.upper()

            if frame_key not in self.context.frames:
                frame = self.interpret_frame(frame=frame)
                self.context.frames[frame_key] = frame

    def _discover_toolsets_and_imports(self, script: Script):
        for toolset_decl in script.toolsets_decl:
            if toolset_decl.name not in self.context.toolsets:
                self.interpret_tool_declaration(toolset_decl)
                self.context.toolsets.add(toolset_decl.name)
            else:
                logger.info(f'Toolset "{toolset_decl.name}" already defined.')

        self.interpret_imports(imports=list(script.toolset_imports.values()))

        # importing declared toolset (that are not imported via an import statement)
        for toolset_decl in script.toolsets_decl:
            toolset_name = toolset_decl.name

            if any(toolset_name in tool for tool in self.context.tools.keys()):
                continue

            # create an import statement that imports all @tool-annotated methods
            import_stmt = nmx_nodes.ImportToolsetStatement(
                name=toolset_name, elements=["*"], args=None, alias=None, meta=dict()
            )
            self.interpret_imports(imports=[import_stmt])

    def _get_frame_by_path(
        self, frame_path: str, statement: nmx_nodes.Statement = None
    ) -> nmx_runtime.Frame:
        """Navigates the operational environment to fetch a frame by its fully qualified path."""
        parts = frame_path.upper().split(".")

        # 1. Fetch the root frame from the Interpreter's global frames memory
        root_name = parts[0]
        rt_frame = self.context.frames.get(root_name)

        if not rt_frame:
            err_msg = f"Undefined root frame referenced: {root_name}"
            raise self._runtime_exception(err_msg, statement=statement)

        # 2. Navigate down the subframe path
        for sub_name in parts[1:]:
            rt_frame = rt_frame.frames.get(sub_name)

            if not rt_frame:
                err_msg = (
                    f"Undefined subframe referenced: {sub_name} in path {frame_path}"
                )
                raise self._runtime_exception(err_msg, statement=statement)

        return rt_frame

    def _frame_to_pydantic_schema(
        self,
        frame_path: str,
        resolved_frames: dict | None = None,
        statement: nmx_nodes.Statement | None = None,
    ) -> Type[BaseModel]:
        """
        Dynamically converts a Nemantix Frame (by fully qualified path)
        into a Pydantic BaseModel for structured LLM proxy generation.
        """
        if resolved_frames is None:
            resolved_frames = {}

        frame_path_upper = frame_path.upper()
        if frame_path_upper in resolved_frames:
            return resolved_frames[frame_path_upper]

        # Fetch the runtime frame using the path navigator
        rt_frame = self._get_frame_by_path(frame_path, statement=statement)

        fields = {}

        # Extract slots. In runtime.py, slots are stored in a dict: `self.slots[name.lower()] = dict(types=..., cardinality=...)`
        # We iterate over the items to get both the slot name and its configuration.
        for s_name, slot_info in getattr(rt_frame, "slots", {}).items():
            s_types = slot_info.get("types", [])
            s_card = slot_info.get("cardinality")

            # 1. Map NXS types to Python/Pydantic types
            mapped_types = []
            for t_info in s_types:
                kind = t_info.get("type", t_info.get("name"))
                val = t_info.get("name") if "type" in t_info else None

                if kind == SlotTypesEnum.TEXT:
                    mapped_types.append(str)
                elif kind == SlotTypesEnum.INT:
                    mapped_types.append(int)
                elif kind == SlotTypesEnum.FLOAT:
                    mapped_types.append(float)
                elif kind == SlotTypesEnum.STRUCT:
                    mapped_types.append(dict | list | tuple)
                elif kind == SlotTypesEnum.ENUM:
                    # val is expected to be a list of strings for an ENUM
                    enum_cls = Enum(
                        f"{s_name.capitalize()}Enum", {str(v): str(v) for v in val}
                    )
                    mapped_types.append(enum_cls)
                elif kind == SlotTypesEnum.FRAME:
                    # If a slot is typed as a FRAME, `val` contains the frame name.
                    # It might be fully qualified, or refer to a subframe of the current frame.
                    # If it's a subframe, we construct the fully qualified path.
                    if val.upper() in rt_frame.frames:
                        nested_path = f"{frame_path}.{val}"
                    else:
                        nested_path = val

                    nested_schema = self._frame_to_pydantic_schema(
                        nested_path, resolved_frames, statement=statement
                    )
                    mapped_types.append(nested_schema)

            # Resolve Union types (e.g., TEXT | INT)
            if not mapped_types:
                base_type = Any
            elif len(mapped_types) == 1:
                base_type = mapped_types[0]
            else:
                base_type = Union[tuple(mapped_types)]

            # 2. Apply NXS Cardinality logic
            if not s_card or s_card == "1":
                final_type = base_type
            elif s_card == "0..1":
                final_type = Optional[base_type]
            else:
                final_type = List[base_type]

            # 3. Define Pydantic Field requirements
            is_optional = s_card in ("0..1", "0..*", "*")
            default_val = None if is_optional else ...

            fields[s_name] = (final_type, default_val)

        # Note: pydantic doesn't like dots in model names, so we replace them.
        safe_model_name = frame_path_upper.replace(".", "_")
        pydantic_model = create_model(safe_model_name, **fields)

        resolved_frames[frame_path_upper] = pydantic_model

        return pydantic_model

    def _parse_do_using(self, do: nmx_nodes.DoStatement) -> tuple[list | Any, dict]:
        expression = do.using

        if expression is None:
            return (), {}

        # Strip AST list wrappers
        if isinstance(expression, list):
            if len(expression) == 0:
                return [], {}
    
            expression = expression[0]
        
        if isinstance(expression, nmx_nodes.Assignment):
            return [], {
                expression.var.name: self.interpret_expression(expression.value)
            }

        if isinstance(expression, nmx_nodes.Collection):
            args = []
            kwargs = {}
            keyword_seen = False

            # Safeguard against single-node values
            expr_values = expression.value
            if not isinstance(expr_values, list):
                expr_values = [expr_values]
            
            for stmt in expr_values:
                if isinstance(stmt, nmx_nodes.Assignment):
                    kwargs[stmt.var.name] = self.interpret_expression(stmt.value)
                    keyword_seen = True
                else:
                    if keyword_seen:
                        raise self._runtime_exception(
                            "Positional argument follows nominal argument in 'using'",
                            statement=do,
                        )

                    args.append(self.interpret_expression(expression=stmt))

            return args, kwargs

        args = self.interpret_expression(expression)
        return [args], {}

    def _apply_frame_schema(self, do: nmx_nodes.DoStatement, result, frame_name: str):
        """
        Format the callable result to conform to the named frame schema.

        For a single producing variable the frame is applied directly to the result Struct.
        For multiple producing variables the LLM is asked to map each variable name to a
        frame slot name (no actual values are sent — privacy is preserved), then the
        frame-conforming Struct is built from actual values using that mapping.
        """
        import ast as _ast

        frame_key = frame_name.upper()
        if frame_key not in self.context.frames:
            raise self._runtime_exception(
                f'Undefined frame "{frame_name}" referenced in producing_schema',
                statement=do,
            )

        rt_frame = self.context.frames[frame_key]
        slot_names = list(getattr(rt_frame, "slots", {}).keys())

        # Collect producing variable names from the AST — no values, for privacy
        producing_expr = do.producing
        if isinstance(producing_expr, nmx_nodes.Variable):
            producing_names = [producing_expr.name]
        elif isinstance(producing_expr, nmx_nodes.Collection):
            producing_names = [
                v.name
                for v in producing_expr.value
                if isinstance(v, nmx_nodes.Variable)
            ]
        else:
            return result

        if not producing_names or not slot_names:
            return result

        if len(producing_names) == 1:
            # Single output variable: apply the frame directly without calling the LLM
            packed = self.pack_return_value(result)
            if isinstance(packed, nmx_runtime.Struct):
                return rt_frame.apply_postfix(packed)
            return packed

        # Multiple output variables: ask LLM to map variable names → slot names.
        # Only names are sent, never the actual runtime values.
        prompt = SCHEMA_APPLY_PROMPT.format(
            frame_name=frame_name,
            producing_names=producing_names,
            slot_names=slot_names,
        )

        response = Builtin.ask_llm(self.proxies.external, prompt)
        self._emit_llm(stmt=do, prompt=prompt, llm_response=response, internal=True)

        try:
            name_mapping = _ast.literal_eval(response.text.strip())
            if not isinstance(name_mapping, dict):
                raise ValueError("LLM did not return a dict")
        except Exception:
            # Fallback: positional mapping
            name_mapping = {
                producing_names[i]: slot_names[i]
                for i in range(min(len(producing_names), len(slot_names)))
            }

        # Extract actual values per producing variable from the packed result
        packed = self.pack_return_value(result)
        actual_values = {}
        if isinstance(packed, nmx_runtime.Struct):
            for i, name in enumerate(producing_names):
                actual_values[name] = packed.get(i)
        elif isinstance(result, (list, tuple)):
            for i, name in enumerate(producing_names):
                actual_values[name] = result[i] if i < len(result) else None
        else:
            actual_values[producing_names[0]] = result

        # Build frame-conforming Struct using the LLM-provided mapping
        frame_struct = nmx_runtime.Struct()
        for var_name, slot_name in name_mapping.items():
            if var_name in actual_values and slot_name in slot_names:
                frame_struct.set(value=actual_values[var_name], key=slot_name)

        validated_struct = rt_frame.apply_postfix(frame_struct)

        # Map the validated values back to the producing variable names
        # so interpret_do_statement can correctly unpack them
        final_struct = nmx_runtime.Struct()
        for var_name, slot_name in name_mapping.items():
            final_struct.set(value=validated_struct.get(slot_name), key=var_name)

        return final_struct

    def _unpack_user_inputs(self, expression: nmx_nodes.Expression | None = None):
        assert not isinstance(
            expression, (nmx_nodes.SchemedCollection, nmx_nodes.MetaExpression)
        )
        inputs = self.interpret_expression(expression)

        if inputs is None:
            return []

        if isinstance(inputs, nmx_runtime.Struct):
            args = list(iter(inputs))
            return args

        if not isinstance(inputs, list):
            return [inputs]

        return inputs

    def _discover_imported_actions(self, script: Script):
        required_locations = self.expertise.requires_map.get(script.get_location(), [])

        for location in required_locations:
            required_script = self.expertise.script_by_loc[location]
            self._discover_actions(required_script)

    def _discover_actions(self, script: Script, deliberate: Deliberate | None = None):
        for action in script.actions.values():
            if action.name not in self.context.actions:
                self.context.actions[action.name] = dict(
                    closure=self._action_closure(action),
                    is_global=True,
                    imported_by={},
                    action=action,
                )
            else:
                logger.warning(
                    f'Name "{action.name}" already defined in context.actions!'
                )

        if deliberate is not None:
            for action in deliberate.generated_actions:
                if action.name not in self.context.actions:
                    action_dict = dict(
                        closure=self._action_closure(action),
                        is_global=False,
                        action=action,
                        imported_by={deliberate.name},
                    )

                    self.context.actions[action.name] = action_dict
                    self.context.actions[f"{deliberate.name}.{action.name}"] = (
                        action_dict
                    )
                else:
                    logger.warning(
                        f'Private action "{action.name}" shadowed by global action with same name!'
                    )

    def _set_global_deliberate(self, deliberate: nmx_nodes.Deliberate):
        self.globals["__deliberate"] = deliberate

    def _get_global_deliberate(self) -> nmx_nodes.Deliberate | None:
        return self.globals.get("__deliberate", None)

    def _set_global_script(self, script: Script):
        self.globals["__script"] = script

    def _get_global_script(self) -> Script | None:
        return self.globals.get("__script", None)

    # def _set_block_inputs(
    #     self,
    #     block: nmx_types.PlanOrActionBlock,
    #     provided_args: Any,
    #     callee: nmx_nodes.DoStatement | None = None,
    # ):
    #     pos_args = []
    #     kw_args = {}
    #
    #     if isinstance(block, nmx_nodes.PlanBlock):
    #         deliberate = self._get_global_deliberate()
    #         assert deliberate is not None
    #         block_name = f"plan::{deliberate.name}"
    #     else:
    #         block_name = f'action "{block.name}"'
    #
    #     # Standardize provided arguments into positional (list) and keyword (dict)
    #     if isinstance(provided_args, tuple):
    #         assert len(provided_args) == 2
    #         assert isinstance(provided_args[0], (list, tuple)) and isinstance(
    #             provided_args[1], dict
    #         )
    #         pos_args, kw_args = provided_args
    #
    #     elif isinstance(provided_args, (list, tuple)):
    #         pos_args = provided_args
    #
    #     elif isinstance(provided_args, dict):
    #         kw_args = provided_args
    #
    #     elif provided_args is not None:
    #         pos_args = [provided_args]
    #
    #     provided_kw_keys = set(kw_args.keys())
    #
    #     # Use an independent index tracker for positional arguments
    #     pos_idx = 0
    #
    #     # Map provided arguments to the defined action inputs
    #     for input_arg in block.input:
    #         arg_name = input_arg.name
    #         arg_set = False
    #
    #         if not arg_name:
    #             logger.warning(
    #                 f'Skipping micro-prompt input "{input_arg.prompt.prompt}" '
    #                 f'for block "{block_name}"!'
    #             )
    #             continue
    #
    #         # 1. Consume positional arguments first (Left-to-Right)
    #         if pos_idx < len(pos_args):
    #             arg_val = pos_args[pos_idx]
    #             arg_set = True
    #             pos_idx += 1
    #
    #             # Check for collision: User provided arg positionally AND nominally
    #             if arg_name in kw_args:
    #                 err_msg = (
    #                     f'{block_name} got multiple values for argument "{arg_name}"'
    #                 )
    #                 raise self._runtime_exception(err_msg, statement=callee)
    #
    #         # 2. Consume keyword arguments
    #         elif arg_name in kw_args:
    #             arg_val = kw_args[arg_name]
    #             provided_kw_keys.remove(arg_name)
    #             arg_set = True
    #
    #         # 3. Fallback to default or raise missing error
    #         else:
    #             if input_arg.required:
    #                 err_msg = f'Missing required argument "{arg_name}" for action "{block_name}"!'
    #                 raise self._runtime_exception(err_msg, statement=callee)
    #             else:
    #                 arg_val = self.interpret_expression(input_arg.default)
    #                 arg_set = True
    #
    #         # Bind argument to the operational environment
    #         if arg_set:
    #             self.context.env.set(var_name=arg_name, value=arg_val)
    #
    #     # Validate that NO extra/unknown keyword arguments were provided
    #     if provided_kw_keys:
    #         extra_args = ", ".join(provided_kw_keys)
    #         err_msg = (
    #             f"{block_name} received unexpected keyword arguments: {extra_args}"
    #         )
    #         raise self._runtime_exception(err_msg, statement=callee)
    #
    #     # Validate that NO extra positional arguments were provided
    #     if pos_idx < len(pos_args):
    #         expected_pos_max = len([arg for arg in block.input if arg.name])
    #         err_msg = (
    #             f"{block_name} expects at most {expected_pos_max} positional arguments, "
    #             f"but {len(pos_args)} were provided ({pos_args})."
    #         )
    #         raise self._runtime_exception(err_msg, statement=callee)
    
    def _set_block_inputs(
        self,
        block: nmx_types.PlanOrActionBlock,
        provided_args: Any,
        callee: nmx_nodes.DoStatement | None = None,
    ):
        pos_args = []
        kw_args = {}

        if isinstance(block, nmx_nodes.PlanBlock):
            deliberate = self._get_global_deliberate()
            assert deliberate is not None
            block_name = f"plan::{deliberate.name}"
        else:
            block_name = f'action "{block.name}"'

        # Standardize provided arguments into positional (list) and keyword (dict)
        if isinstance(provided_args, tuple):
            assert len(provided_args) == 2
            assert isinstance(provided_args[0], (list, tuple)) and isinstance(
                provided_args[1], dict
            )
            pos_args, kw_args = provided_args

        elif isinstance(provided_args, (list, tuple)):
            pos_args = provided_args

        elif isinstance(provided_args, dict):
            kw_args = provided_args

        elif provided_args is not None:
            pos_args = [provided_args]

        provided_kw_keys = set(kw_args.keys())

        # Map provided arguments to the defined action inputs
        for i, input_arg in enumerate(block.input):
            arg_name = input_arg.name
            arg_set = False

            if len(arg_name) == 0:
                logger.warning(
                    f'Skipping micro-prompt input "{input_arg.prompt.prompt}" '
                    f'for block "{block_name}"!'
                )
                continue

            # Prioritize keyword arguments
            if arg_name in kw_args:
                arg_val = kw_args[arg_name]
                provided_kw_keys.remove(arg_name)
                arg_set = True

            # Fallback to positional arguments if not provided by keyword
            elif i < len(pos_args):
                arg_val = pos_args[i]
                arg_set = True
            else:
                arg_val = None

            # Validate required arguments
            if not arg_set:
                if input_arg.required:
                    err_msg = f'Missing required argument "{arg_name}" for action "{block_name}"!'
                    raise self._runtime_exception(err_msg, statement=callee)
                else:
                    arg_val = self.interpret_expression(input_arg.default)

            # Bind argument to the operational environment
            self.context.env.set(var_name=arg_name, value=arg_val)

        # Validate that NO extra/unknown keyword arguments were provided
        if provided_kw_keys:
            extra_args = ", ".join(provided_kw_keys)
            err_msg = (
                f"{block_name} received unexpected keyword arguments: {extra_args}"
            )
            raise self._runtime_exception(err_msg, statement=callee)

        # Validate that NO extra positional arguments were provided
        if len(pos_args) > len(block.input):
            err_msg = (
                f"{block_name} expects at most {len(block.input)} positional arguments, "
                f"but {len(pos_args)} were provided ({pos_args})."
            )
            raise self._runtime_exception(err_msg, statement=callee)

    def _action_closure(self, action: nmx_nodes.ActionBlock):
        def wrap(args, callee=None):
            logger.debug(f'calling action "{action.name}"')

            if (callee is not None) and (not isinstance(callee, nmx_nodes.DoStatement)):
                callee = None
                logger.warning(
                    f'Ignoring callee argument as it is not a DoStatement but a "{type(callee)}"!'
                )

            return self.interpret_block(action, args, callee=callee)

        return wrap

    def _event_from_statement(
        self,
        stmt: nmx_nodes.Statement | None,
        event_type: EventType,
        scope=None,
        trim=False,
        **kwargs,
    ):
        scope = scope or self.globals["__scope"]
        deliberate = self._get_global_deliberate()
        assert deliberate is not None
        script = self.expertise.get_script_from_deliberate(
            deliberate_name=deliberate.name
        )

        if stmt is not None:
            file_meta = stmt.meta["file_meta"]
            assert isinstance(file_meta, nmx_nodes.FileMeta)

            start_line, end_line = file_meta.line
            start_column, end_column = file_meta.column

            content = script.read(read_as_lines_list=True)

            if trim:
                payload = "\n".join(
                    line[start_column - 1 : end_column]
                    for line in content[start_line - 1 : end_line]
                )
            else:
                payload = "\n".join(content[start_line - 1 : end_line])
        else:
            start_line, end_line = 0, 0
            payload = "<empty>"

        event = Event(
            type=event_type,
            lines=(start_line, end_line),
            scope="::".join(scope),
            script=script,
            statement=payload.strip(),
            **kwargs,
        )
        return event

    def _emit_event(
        self,
        stmt: nmx_nodes.Statement | None,
        event_type: EventType,
        scope=None,
        **kwargs,
    ):
        event_hub = context.event_hub.get()
        if event_hub is None:
            return

        if not event_hub.has_subscribers(event_type):
            return

        event = self._event_from_statement(stmt, event_type, scope=scope, **kwargs)
        event_hub.emit(event)

    def _emit_line(self, stmt: nmx_nodes.Statement, scope=None, **kwargs):
        self._emit_event(
            stmt,
            event_type=EventType.LINE,
            scope=scope,
            payload=dict(interpreter=self),
            **kwargs,
        )

    def _emit_call_enter(self, stmt: nmx_nodes.Statement, scope=None, **kwargs):
        payload = dict(
            name=kwargs.pop("callable_name"),
            type=kwargs.pop("callable_type"),
            prompt=kwargs.pop("callable_prompt", ""),
        )
        self._emit_event(
            stmt,
            event_type=EventType.CALL_ENTER,
            scope=scope,
            payload=payload,
            **kwargs,
        )

    def _emit_call_exit(self, stmt: nmx_nodes.Statement, scope=None, **kwargs):
        self._emit_event(stmt, event_type=EventType.CALL_EXIT, scope=scope, **kwargs)

    def _emit_error(
        self,
        stmt: nmx_nodes.Statement | None,
        error: Exception | str,
        scope=None,
        **kwargs,
    ):
        if stmt is None:
            logger.warning(f"Skipping error emit due to None statement. Error: {error}")
            return

        self._emit_event(
            stmt,
            event_type=EventType.ERROR,
            scope=scope,
            payload=dict(error=str(error), interpreter=self),
            **kwargs,
        )

    def _emit_llm(
        self,
        stmt: nmx_nodes.Statement | None,
        prompt: str,
        llm_response: LLMResponse | StructuredLLMResponse,
        schema: type[BaseModel] | None = None,
        scope=None,
        internal=False,
        **kwargs,
    ):
        self._emit_event(
            stmt,
            event_type=EventType.LLM,
            scope=scope,
            payload=dict(
                prompt=prompt,
                schema=schema,
                usage=llm_response.usage,
                internal_usage=bool(internal),
                name=llm_response.proxy.get_name(),
            ),
            **kwargs,
        )

    def _emit_retrieve(
        self,
        stmt: nmx_nodes.Statement,
        knowledge_base: NemantixKnowledgeBase,
        scope=None,
        **kwargs,
    ):
        self._emit_event(
            stmt,
            event_type=EventType.RETRIEVE,
            scope=scope,
            payload=dict(knowledge_base=knowledge_base, query=kwargs.pop("query", "")),
            **kwargs,
        )

    def _emit_expand(
        self,
        stmt: nmx_nodes.Statement,
        knowledge_base: NemantixKnowledgeBase,
        scope=None,
        **kwargs,
    ):
        self._emit_event(
            stmt,
            event_type=EventType.EXPAND,
            scope=scope,
            payload=dict(
                knowledge_base=knowledge_base,
                query=f"node_id: {kwargs.pop('node_id', None)}",
            ),
            **kwargs,
        )

    def _emit_extend(
        self,
        stmt: nmx_nodes.Statement,
        knowledge_base: NemantixKnowledgeBase,
        scope=None,
        **kwargs,
    ):
        self._emit_event(
            stmt,
            event_type=EventType.EXTEND,
            scope=scope,
            payload=dict(
                knowledge_base=knowledge_base,
                query=f"node_id: {kwargs.pop('node_id', None)}",
            ),
            **kwargs,
        )

    def _emit_generalize(
        self,
        stmt: nmx_nodes.Statement,
        knowledge_base: NemantixKnowledgeBase,
        scope=None,
        **kwargs,
    ):
        self._emit_event(
            stmt,
            event_type=EventType.GENERALIZE,
            scope=scope,
            payload=dict(
                knowledge_base=knowledge_base,
                query=f"node_id: {kwargs.pop('node_id', None)}",
            ),
            **kwargs,
        )

    def _push_scope(self, scope: str):
        self.globals["__scope"].append(scope)

    def _pop_scope(self):
        if len(self.globals["__scope"]) > 0:
            self.globals["__scope"].pop()

    def _filter_by_similarity(
        self,
        struct: nmx_runtime.Struct,
        query: Any,
        qualifier: SimilarityQualifierEnum | float,
        query_emb: npt.NDArray = None,
    ) -> nmx_runtime.Struct:
        filtered = nmx_runtime.Struct()

        if not isinstance(query_emb, np.ndarray):
            query_emb = self.embedder.embed(Builtin.to_str(query))

        for key, value in struct.items():
            key_ = key if isinstance(key, str) else None
            if isinstance(value, (nmx_runtime.Opaque, nmx_runtime.DocRef)):
                raise NotImplementedError

            if isinstance(value, nmx_runtime.Struct):
                sub_struct = self._filter_by_similarity(
                    struct=value, query=query, qualifier=qualifier, query_emb=query_emb
                )
                if len(sub_struct) > 0:
                    filtered.set(key=key_, value=sub_struct)
            else:
                similarity = self._compute_similarity(
                    a=Builtin.to_str(value), b=None, b_emb=query_emb
                )

                if self._apply_similarity_qualifier(similarity, qualifier):
                    filtered.set(key=key_, value=value)

        return filtered

    def _compute_similarity(
        self, a, b, statement: nmx_nodes.Statement | None = None, **kwargs
    ) -> float:
        if self.embedder is None:
            # use LLM to compute similarity between a and b
            system_prompt = (
                "You are a multilingual language expert, able to understand "
                "whether two words or phrases have a similar semantic meaning. "
                'A similarity expression is denoted as ["a" ~ "b"].'
            )
            user_prompt = (
                f'Task: Evaluate the similarity expression: ["{a}" ~ "{b}"]. '
                f"Return true or false, along with a score in 0-1 range that "
                f"quantifies the degree of similarity (0: means dissimilar or opposite, "
                f"1: means very similar or identical)."
            )

            messages = self.llm.messages_from(
                prompts_with_roles=[("system", system_prompt), ("user", user_prompt)]
            )

            response = self.llm.invoke_structured(
                messages, schema=self.SimilaritySchema
            )
            result = response.result

            self._emit_llm(
                stmt=statement,
                prompt=system_prompt + user_prompt,
                internal=True,
                schema=self.SimilaritySchema,
                llm_response=response,
            )

            logger.debug(f'Similarity score of "{a} ~ {b}" is {result.score}')
            return result.score

        return nmx_runtime.compute_similarity(self.embedder, a=a, b=b, **kwargs)

    def _set_special_variables(self):
        # TODO: add semantics to (all/some) variables?
        self._register_special_var(name="ENV", value=self.external_vars)
        self._register_special_var(
            name="STATE", value=self.agent_state, read_only=False
        )

        # math constants
        # see: https://en.wikipedia.org/wiki/List_of_mathematical_constants
        self._register_special_var(name="PI", value=math.pi)  # Archimede pi
        self._register_special_var(name="E", value=math.e)  # Euler number
        self._register_special_var(name="SQRT_2", value=math.sqrt(2))
        self._register_special_var(name="SQRT_3", value=math.sqrt(3))
        self._register_special_var(
            name="GOLDEN_RATIO", value=(1.0 + math.sqrt(5)) / 2.0
        )
        self._register_special_var(name="LN_2", value=math.log(2))

        # physics constants
        self._register_special_var(name="C", value=299_792_458)  # speed of light in m/s
        self._register_special_var(name="G", value=9.80665)  # Earth's gravity in m/s^2

    def _register_special_var(self, name: str, value, read_only=True):
        self.context.env.set(var_name=name, value=value)

        if read_only:
            self._SPECIAL_VARS.add(name)

    def _wrap_callable_with_try_except(
        self, fn: Callable, statement: nmx_nodes.Statement
    ) -> Callable:
        assert callable(fn)

        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                self._log_exception(e)
                fn_name = (
                    "functools.partial"
                    if type(fn) is functools.partial
                    else fn.__name__
                )

                raise self._runtime_exception(
                    f'Exception in execution of "{fn_name}". Error: {e}',
                    statement=statement,
                )

        return inner

    @staticmethod
    def _apply_similarity_qualifier(
        similarity: float, qualifier: SimilarityQualifierEnum | float
    ) -> bool:
        if qualifier == SimilarityQualifierEnum.FAR:
            return similarity <= 0.4

        if qualifier == SimilarityQualifierEnum.LOOSE:
            return similarity >= 0.6

        if qualifier == SimilarityQualifierEnum.ABOUT:
            return similarity >= 0.75

        if qualifier == SimilarityQualifierEnum.CLOSE:
            return similarity >= 0.85

        if qualifier == SimilarityQualifierEnum.STRICT:
            return similarity >= 0.92

        # NUMBER
        assert isinstance(qualifier, float)
        return similarity >= qualifier

    @staticmethod
    def _apply_semantic_qualifier(
        similarity: float, qualifier: SimilarityQualifierEnum | float
    ) -> bool:
        if qualifier == SimilarityQualifierEnum.FAR:
            return similarity <= 0.45

        if qualifier == SimilarityQualifierEnum.LOOSE:
            return similarity >= 0.7

        if qualifier == SimilarityQualifierEnum.ABOUT:
            return similarity >= 0.82

        if qualifier == SimilarityQualifierEnum.CLOSE:
            return similarity >= 0.9

        if qualifier == SimilarityQualifierEnum.STRICT:
            return similarity >= 0.95

        # NUMBER
        assert isinstance(qualifier, float)
        return similarity >= qualifier

    @staticmethod
    def _is_arithmetic_op(operation: BinaryOperationEnum) -> bool:
        return operation in [
            BinaryOperationEnum.ADD,
            BinaryOperationEnum.SUB,
            BinaryOperationEnum.MUL,
            BinaryOperationEnum.DIV,
            BinaryOperationEnum.MOD,
            BinaryOperationEnum.POW,
        ]

    @staticmethod
    def _is_logical_op(operation: BinaryOperationEnum) -> bool:
        return operation in [
            BinaryOperationEnum.LOGICAL_OR,
            BinaryOperationEnum.LOGICAL_AND,
            BinaryOperationEnum.LOGICAL_XOR,
        ]

    @staticmethod
    def _is_comparison_op(operation: BinaryOperationEnum) -> bool:
        return operation in [
            BinaryOperationEnum.LT,
            BinaryOperationEnum.GT,
            BinaryOperationEnum.LTE,
            BinaryOperationEnum.GTE,
        ]

    def _runtime_exception(
        self, *args, cls=nmx_ex.NemantixRuntimeException, emit=True, **kwargs
    ) -> nmx_ex.NemantixRuntimeException:
        kwargs["script"] = self._get_global_script()

        if emit:
            self._emit_error(stmt=kwargs.get("statement", None), error=args[0])

        return cls(*args, **kwargs)

    @staticmethod
    def _log_exception(exception: Exception):
        logger.error(f"[{exception.__class__.__name__}]: {exception}", exc_info=True)
