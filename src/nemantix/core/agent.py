from __future__ import annotations

import json
import logging
import re
import textwrap
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional, Type

if TYPE_CHECKING:
    from nemantix.knowledge_base.core.nemantix_knowledge_base import KnowledgeBaseConfig

from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from nemantix.common.logger import get_package_logger, update_logger_levels
from nemantix.core import runtime as nmx_runtime
from nemantix.core.exceptions import NemantixException, NemantixRuntimeException
from nemantix.core.executor import Executor
from nemantix.core.expertise import Expertise
from nemantix.core.node import (
    ActionBlock,
    ActionInput,
    ActionOutput,
    Deliberate,
    NodeMeta,
)
from nemantix.hub import Event, EventHub, EventType
from nemantix.llm import AbstractLLMProxy

logger = get_package_logger(__name__)


class Agent:
    def __init__(self, expertise: Expertise, llm_proxy: AbstractLLMProxy | None = None,
                 external_vars: dict[str, Any] | None = None, use_embedder=False,
                 use_knowledge_base=False, build_on_start=True,
                 kb_config: KnowledgeBaseConfig | None = None, log_level=None, **__):
        if log_level is not None:
            logger.info(f"Updating logger level to {logging.getLevelName(log_level)}")
            update_logger_levels(level=log_level)

        if not isinstance(external_vars, dict):
            logger.warning(f'Provided external_vars is not a dict but a "{type(external_vars)}",'
                           f'defaulting to an empty dict.')
            external_vars = dict()

        self.expertise = expertise
        self.expertise.set_external_vars_names([k for k in external_vars.keys()])

        if llm_proxy is None:
            self.llm = self.expertise.coder.llm_proxy
            logger.info("Using Expertise's LLM proxy.")
        else:
            self.llm = llm_proxy

        self.embedder = None
        self.knowledge_base = None
        self.state = AgentState()

        if use_embedder:
            from nemantix.knowledge_base.models.embedding import (
                SentenceTransformerWrapper,
            )
            self.embedder = SentenceTransformerWrapper()

        if use_knowledge_base:
            if kb_config is None:
                raise NemantixException(
                    "The 'kb_config' (KnowledgeBaseConfig) is mandatory when 'use_knowledge_base' is True."
                )

            from nemantix.knowledge_base.core.nemantix_knowledge_base import (
                NemantixKnowledgeBase,
            )

            self.knowledge_base = NemantixKnowledgeBase(config=kb_config)
            self.expertise.set_knowledge_base(knowledge_base=self.knowledge_base)

            if not self.knowledge_base.db.is_service_available():
                logger.error(f"Failed to connect to the database for views: {kb_config.view_ids}")
                raise NemantixException(
                    "The Knowledge Base database service is unavailable. "
                    "Please ensure PostgreSQL is running and accessible."
                )

            logger.info(f"Knowledge Base successfully initialized with scope: {kb_config.view_ids}")

        self.executor = Executor(
            expertise=expertise,
            llm=self.llm,
            embedder=self.embedder,
            knowledge_base=self.knowledge_base,
            external_vars=external_vars,
            agent_state=self.state.get())

        # if there are any nxs, code them
        if build_on_start:
            self.expertise.build()

        if not self.expertise.verify():
            # TODO: better error message?
            raise NemantixRuntimeException("Verification failed for some NXV scripts!")

    def __enter__(self):
        hub = EventHub.get_active_hub(event_type=EventType.MONITOR_START)
        if not hub:
            return

        event = Event(type=EventType.MONITOR_START, lines=(0, 0), scope='agent',
                      script=None, statement='')
        hub.emit(event)

    def __exit__(self, *_):
        hub = EventHub.get_active_hub(event_type=EventType.MONITOR_STOP)
        if not hub:
            return

        event = Event(type=EventType.MONITOR_STOP, lines=(0, 0), scope='agent',
                      script=None, statement='')
        hub.emit(event)

    def run(self, user_request: str, schema: Type[BaseModel] | None = None,
            **kwargs) -> tuple[NemantixException | None, Any]:
        """One-shot agent running mode: it executes the task until completion."""
        logger.info(f"Agent executing request {user_request}")
        exception = None
        outputs = None

        # check for nxs presence
        if not self.expertise.is_fully_coded():
            self.expertise.build()
        else:
            self.expertise.update()

        try:
            outputs = self.executor.execute(user_request=user_request, **kwargs)

            state_struct = self.executor.get_agent_state()
            self.update_state(**state_struct.to_dict())

            if schema is not None:
                outputs = self._format_output(outputs=outputs, schema=schema)

            self.expertise.export()

        except NemantixException as e:
            logger.error(f"[{e.__class__.__name__}]: {e}", exc_info=True)
            exception = e

        return exception, self._unbox_outputs(outputs)

    def update_state(self, **kwargs):
        self.state.update(**kwargs)

    def delete_state(self, *keys):
        self.state.delete(*keys)

    def _format_output(self, outputs: Any, schema: Type[BaseModel]) -> BaseModel:
        """
        Orchestrates the formatting of raw outputs into a Pydantic schema.
        Tries exact programmatic parsing first, then falls back to LLM extraction.
        """
        # METHOD 1: Try the fast, programmatic path
        parsed_result = self._parse_exact(outputs, schema)

        if parsed_result is not None:
            logger.debug("Successfully parsed output programmatically.")
            return parsed_result

        # METHOD 2: Fall back to the LLM path
        logger.info(
            "Exact parsing failed or output is unstructured. Falling back to LLM."
        )
        return self._parse_with_llm(outputs, schema)

    def _unbox_outputs(self, outputs) -> Any:
        if outputs is None:
            return None

        if isinstance(outputs, str):
            return self._unescape_string(outputs)

        if isinstance(outputs, nmx_runtime.Struct):
            if outputs.can_be_seen_as_list():
                values = [self._unbox_outputs(v) for v in outputs.values()]
            else:
                values = {k: self._unbox_outputs(v) for k, v in outputs.items()}

            return values

        elif isinstance(outputs, nmx_runtime.Opaque):
            return outputs.unbox()

        # elif isinstance(outputs, nmx_runtime.DocRef):
        #     return outputs.identifier

        return outputs

    @staticmethod
    def _unescape_string(text):
        # Map valid sequences to their actual control characters
        escapes = {'n': '\n', 't': '\t', 'r': '\r', 'v': '\v', 'f': '\f',
                   'b': '\b', '"': '\"', "'": '\'', '\\': '\\'}

        def replace(match):
            char = match.group(1)
            if char in escapes:
                return escapes[char]

            return char

        return re.sub(r'\\(.)', replace, text)

    @staticmethod
    def _parse_exact(outputs: Any, schema: Type[BaseModel]) -> Optional[BaseModel]:
        """
        Attempts to instantiate the Pydantic model directly from a dict, JSON string,
        or a Nemantix Struct. Returns the validated model or None.
        """

        # 1. Handle Nemantix Struct
        if isinstance(outputs, nmx_runtime.Struct):
            args, kwargs = outputs.to_args_and_kwargs()

            # We try to validate using the named fields (kwargs) inside the Struct
            if kwargs:
                try:
                    return schema.model_validate(kwargs)
                except ValidationError as e:
                    logger.warning(f"Struct to Pydantic validation failed: {e}")
            else:
                logger.info("Struct only contains positional args, leaving to LLM fallback.")

        # 2. Handle Standard Dictionary
        elif isinstance(outputs, dict):
            try:
                return schema.model_validate(outputs)
            except ValidationError as e:
                logger.warning(f"Dict validation failed (missing/wrong fields): {e}")

        # 3. Handle JSON String
        elif isinstance(outputs, str):
            try:
                parsed_dict = json.loads(outputs)
                return schema.model_validate(parsed_dict)
            except (json.JSONDecodeError, ValidationError):
                # Not valid JSON or doesn't match the schema
                pass

        # 4. Handle Primitive Types (int, float, bool, None)
        elif isinstance(outputs, (int, float, bool)) or outputs is None:
            try:
                # This directly validates if your schema is a RootModel
                return schema.model_validate(outputs)
            except ValidationError as e:
                logger.warning(
                    f"Primitive validation failed (schema likely expects a dict, not a scalar): {e}"
                )
                # If it's just a raw primitive but the schema is a complex dictionary,
                # we let it fall through to the LLM to see if the LLM can map it.

        return None

    def _parse_with_llm(self, outputs: Any, schema: Type[BaseModel]) -> BaseModel:
        """
        Uses the LLM proxy to extract data from unstructured text and map it to the schema.
        """
        prompt = (
            "Your task is to extract and format the provided raw data into the exact requested JSON structure.\n"
            "Do not add any conversational text. Only map the raw data to the schema.\n\n"
            f"RAW DATA:\n{outputs}")

        response = self.llm.invoke_structured(prompt=prompt, schema=schema)
        self.executor._emit_llm(usage=response.usage)
        return response.result


# TODO: handle Opaque passing
class ReActAgent(Agent):
    """Agents implementing the Reason-Act-Observe run paradigm"""
    from nemantix.core.tools import Toolset

    class NemantixToolset(Toolset):
        @classmethod
        def register_action(cls, action: ActionBlock, agent: Agent):
            from pathlib import Path

            from nemantix.core.script import Script
            from nemantix.core.source_manager import LocalSourceManager

            script_loc = agent.expertise.action_to_script_loc[action.name]
            script = agent.expertise.script_by_loc[script_loc]

            script_content = script.read(read_as_lines_list=True)
            assert isinstance(script_content, list)
            action_content = agent.expertise.coder._read_node_nxs(script_content, action,
                                                                  read_as_list=True)

            action_body = '\n'.join(action_content[2:-1])
            plan_qualifier = action_content[0]
            micro_prompt = ' '.join(action.prompt.prompt.split())
            deliberate_str = ReActAgent._WRAP_ACTION_TEMPLATE.format(action.name,
                                                                     micro_prompt,
                                                                     plan_qualifier,
                                                                     action_body)
            # create temporary script
            temp_loc = Path(f'./_tmp_{action.name}.nxs')
            temp_script = Script(location=temp_loc, source_manager=LocalSourceManager(),
                                 content=deliberate_str)
            temp_script.parse()

            def call_action(*_, **kwargs):
                deliberate = temp_script.deliberates[action.name]
                action_args = cls.extract_inputs(agent, **kwargs)

                result = agent.executor.interpreter.interpret_coded_request(temp_script,
                                                                            deliberate,
                                                                            action_args)

                return cls._maybe_handle_opaque(agent, result, outputs=action.output)

            inputs = [f'- {arg.name}: required={arg.required}, default={arg.default}'
                      for arg in action.input]
            outputs = [f'- {arg.name} ({arg.prompt.prompt})'
                       for arg in action.output]

            docstring = f"""
            {action.prompt.prompt}
            Inputs:
                {'\n'.join(inputs)}
            Outputs:
                {'\n'.join(outputs)}
            """
            parameters = {arg.name: cls.param_from_input(arg) for arg in action.input}

            cls.REGISTRY[f'{cls.__name__}.{action.name}'] = dict(
                cls=cls,
                cls_name=cls.__name__,
                fn_name=f'call_{action.name}',
                fn=call_action,
                docstring=docstring,
                parameters=parameters)

            return str(temp_loc), temp_script, action.name

        @classmethod
        def register_deliberate(cls, deliberate: Deliberate, agent: 'ReActAgent'):
            def call_deliberate(*_, **kwargs):
                action_args = cls.extract_inputs(agent, **kwargs)
                script = agent.expertise.get_script_from_deliberate(deliberate.name)
                result = agent.executor.interpreter.interpret_coded_request(script, deliberate,
                                                                            user_inputs=action_args)
                return cls._maybe_handle_opaque(agent, result,
                                                outputs=deliberate.get_plan().output)

            plan = deliberate.get_plan()
            inputs = [f'- {arg.name}: required={arg.required}, default={arg.default}'
                      for arg in plan.input]
            outputs = [f'- {arg.name} ({arg.prompt.prompt})'
                       for arg in plan.output]

            node_meta = deliberate.meta['node_meta']
            intent = ''
            if isinstance(node_meta, NodeMeta):
                for ann in node_meta.annotations:
                    if ann.name in ['intent.goal', 'goal']:
                        intent = f'Intent: {ann.value.value}'

            docstring = f"""
            {intent}
            When: {deliberate.when.prompt}
            Inputs:
                {'\n'.join(inputs)}
            Outputs:
                {'\n'.join(outputs)}
            """
            parameters = {arg.name: cls.param_from_input(arg) for arg in plan.input}

            cls.REGISTRY[f'{cls.__name__}.{deliberate.name}'] = dict(
                cls=cls,
                cls_name=cls.__name__,
                fn_name=f'call_{deliberate.name}',
                fn=call_deliberate,
                docstring=docstring.strip(),
                parameters=parameters)

            return deliberate.name
        
        @staticmethod
        def param_from_input(input_arg: ActionInput):
            from collections import namedtuple
            from inspect import Parameter
            from typing import Any

            param = namedtuple('Param', ['annotation', 'default'])
            param.annotation = Any

            if input_arg.required or input_arg.default is None:
                param.default = Parameter.empty
            else:
                param.default = input_arg.default

            return param

        @staticmethod
        def extract_inputs(agent, **kwargs):
            args = []
            for k, v in kwargs.items():
                v_type = 'str'

                # intercept reference to opaque object
                if isinstance(v, str) and v.startswith('__opaque_') and v.endswith('__'):
                    state_struct = agent.state.get()
                    var_name = '_'.join(v.split('_')[3:-2])

                    actual_val = state_struct.get(v)
                    if actual_val is not None:
                        k = var_name
                        v = actual_val  # TODO: passing the opaque is not necessary
                        v_type = "opaque"

                elif isinstance(v, bool):
                    v_type = 'bool'

                elif isinstance(v, int):
                    v_type = 'int'

                elif isinstance(v, float):
                    v_type = 'float'

                elif isinstance(v, list):
                    v_type = 'list'

                elif isinstance(v, dict):
                    v_type = 'dict'

                args.append(dict(name=k, value=v, type=v_type))

            action_args = agent.executor._inputs_from_request(required_inputs=args)
            return action_args

        @staticmethod
        def _maybe_handle_opaque(agent: 'ReActAgent', result: Any, outputs: list[ActionOutput]):
            if isinstance(result, nmx_runtime.Opaque):
                # TODO: generalize
                # get the name of the opaque in out block
                state_key = f'__opaque_{outputs[0].name}__'
                agent.opaque_map[result.identifier] = state_key

                logger.info(f'Setting Opaque object "{result}" in agent state as field "{state_key}"')
                agent.update_state(**{state_key: result})

            return result

    # TODO: include feedback on error?
    class ReActSchema(BaseModel):
        # feedback: str
        thought: str
        tool_name: str
        tool_arguments: dict[str, Any]
        task_completed: bool

    @dataclass
    class RunSchema:
        output: str  # machine-friendly
        answer: str  # human-friendly

    class RunModeEnum(Enum):
        TOOL = auto()
        ACTION = auto()
        DELIBERATE = auto()
        ACTION_TOOL = auto()
        DELIBERATE_TOOL = auto()
        DELIBERATE_ACTION = auto()
        ALL = auto()

    _WRAP_ACTION_TEMPLATE = """
deliberate {} when >> {} <<:
    {}
    plan:
            {}
    __plan
__deliberate
    """
    _WRAP_TOOL_TEMPLATE = """
    """

    def __init__(self, expertise: Expertise, llm_proxy: AbstractLLMProxy | None = None,
                 external_vars: dict | None = None, use_embedder=False,
                 run_mode=RunModeEnum.DELIBERATE, log_level=None,
                 kb_config: KnowledgeBaseConfig | None = None,
                 use_knowledge_base=False, build_on_start=True, **__):
        super().__init__(expertise, llm_proxy, external_vars, use_embedder,
                         use_knowledge_base, build_on_start, kb_config, log_level)

        # bound actions and deliberates as tools for the LLM
        registered_names = []
        should_register_actions = run_mode in [self.RunModeEnum.ACTION, self.RunModeEnum.ALL,
                                               self.RunModeEnum.ACTION_TOOL,
                                               self.RunModeEnum.DELIBERATE_ACTION]
        should_register_deliberates = run_mode in [self.RunModeEnum.ALL, self.RunModeEnum.DELIBERATE,
                                                   self.RunModeEnum.DELIBERATE_ACTION,
                                                   self.RunModeEnum.DELIBERATE_TOOL]
        should_register_tools = run_mode in [self.RunModeEnum.TOOL, self.RunModeEnum.ACTION_TOOL,
                                             self.RunModeEnum.DELIBERATE_TOOL, self.RunModeEnum.ALL]
        update_expertise = []

        for script in self.expertise.script_by_loc.values():
            if should_register_actions:
                for action in script.actions.values():
                    loc_, script_, name_ = self.NemantixToolset.register_action(action, self)
                    registered_names.append(action.name)
                    update_expertise.append((loc_, script_, name_))
            else:
                logger.info(f'Action calling is not supported in script "{script.get_location()}"')

            if should_register_deliberates:
                for deliberate in script.deliberates.values():
                    self.NemantixToolset.register_deliberate(deliberate, self)
                    registered_names.append(deliberate.name)

        if should_register_tools:
            from nemantix.core import Toolset

            for tool_name, tool in Toolset.REGISTRY.items():
                pass

        for loc, script, name in update_expertise:
            self.expertise.script_by_loc[loc] = script
            self.expertise.deliberate_to_script_loc[name] = loc

        self.available_tools = registered_names
        self.history = dict(thoughts=[], tools=[], arguments=[], outputs=[],
                            feedbacks=[], errors=[])
        self.opaque_map = dict()

    # TODO: detect loops
    # TODO: predict a plan, then execute and revise it at each step?
    def run(self, user_request: str, schema: Type[BaseModel] | None = None,
            reformulate_answer=True, human_approval=False, **kwargs) -> RunSchema:
        system_prompt = ('You are an helpful agent assistant. You have access to tools '
                         '(lookup [[tools]]). '
                         'You must assess the goal (lookup [[user_request]]) and '
                         'the current state (lookup [[context]]) forming an '
                         'hypothesis about which tool is required for the next step. '
                         'If the goal is already solved, then predict the task '
                         'as terminated.')
        context = [('system', system_prompt),
                   ('user', user_request)]

        def context_to_str() -> str:
            string = []
            for (role, output) in context:
                string.append(f'role={role}: {output}')

            return '\n'.join(string)

        tools = [(name, self.NemantixToolset.REGISTRY[f'NemantixToolset.{name}'])
                 for name in self.available_tools]
        tools_str = '\n'.join([
            f'tool: {name}; docstring: {tool["docstring"]}; '
            f'arguments: {list(tool["parameters"].keys())} (Provide as JSON native types: int, list, dict, bool, etc.).'
            for (name, tool) in tools
        ])

        # TODO: if you need a tool that is not listed here...
        reason_template = """Given the user request
        [[user_request]]
        "{}"
        and the agent context

        [[context]]
        "{}"
        determine which tool you should call (see [[tools]]), with which arguments 
        (as JSON), and why.

        [[tools]]
        "{}"
        
        MEMORY RULE:
        If a previous tool saved an object to memory and returned a reference ID (e.g., "__opaque_1234__"), 
        and you need to use that object in your next step, pass that exact reference ID string as the 
        argument for the new tool.
        
        NOTE: If the last output answers the [[user_request]] then, no tool call is 
        necessary and the task is solved.
        [[last_output]]
        "{}"
        
        [[errors]]
        "{}"
        If an error occurred during the last tool call, revise it: either select another tool,
        or adjust the input arguments accordingly to avoid the execution error again.
        """

        last_output = ''
        last_error = ''

        # TODO: add internal prompt to initial request?
        # TODO: catch exception and eventually ask user?
        while True:
            prompt = reason_template.format(user_request, context_to_str(),
                                            tools_str, last_output, last_error)

            messages = self.llm.messages_from([('assistant', prompt)])
            result = self.llm.invoke_structured(messages, schema=self.ReActSchema)
            self.executor._emit_llm(usage=result.usage)

            response = result.result
            assert isinstance(response, self.ReActSchema)

            context.append(('assistant', response.thought))
            logger.info(f'though: "{"\n".join(textwrap.wrap(response.thought, width=100))}"')
            self.history['thoughts'].append(response.thought)

            if response.task_completed:
                logger.info('Task complete')

                if reformulate_answer:
                    prompt = (f'Given the response "{last_output}", reformulate it '
                              f'appropriately to answer the user request "{user_request}". '
                              f'Since the task is now complete, do not suggest next steps, and'
                              f'do not add any commentary or opinion."')
                    response = self.llm.invoke(prompt)
                    self.executor._emit_llm(usage=response.usage)

                    return self.RunSchema(last_output, response.text)
                else:
                    return self.RunSchema(last_output, last_output)

            tool_name = response.tool_name
            tool_alias = f'NemantixToolset.{tool_name}'

            if tool_alias in self.NemantixToolset.REGISTRY:
                tool = self.NemantixToolset.get_tool(tool_alias)

                arguments = response.tool_arguments
                logger.info(f'Calling tool "{tool_name}" with arguments: {arguments}')

                try:
                    tool_out = tool(**arguments)
                    last_error = ''

                    last_output = self.convert_to_str(tool_out).replace('\n', ' ')

                    # TODO: add tool args?
                    context.append(('tool', f'{tool_name}: {last_output}'))
                    self.history['arguments'].append(arguments)

                    if human_approval:
                        feedback = self._ask_user(last_output)
                        context.append(('user', feedback))
                        self.history['feedbacks'].append(feedback)

                except Exception as e:
                    # TODO: could also put the error in 'tool' message
                    last_error = f'Error occurred in tool "{tool_name}": "{e}"!'
            else:
                context.append(('tool', f'Tool "{tool_name}" is not a valid toolset name: either '
                                        'make a valid tool call or mark the task as completed if '
                                        'no more tool calls are necessary.'))

            self.history['outputs'].append(last_output)
            self.history['errors'].append(last_error)
            self.history['tools'].append(tool_name)

    # TODO: handle Opaque, etc
    def convert_to_str(self, content) -> str:
        if isinstance(content, nmx_runtime.DocRef):
            return self._doc_to_str(doc=content)

        if isinstance(content, nmx_runtime.Opaque):
            # ref_id = f"__opaque_{content.identifier}__"

            try:
                preview = self._opaque_preview(opaque=content)
            except Exception:
                preview = "<Complex Object>"

            key = self.opaque_map[content.identifier]
            return (f"[Saved to agent internal state 'STATE' as '{key}'. Preview: {preview}\n"
                    f"IMPORTANT: To use this object in your next tool call, you MUST pass the "
                    f"EXACT string '{key}' as the input argument.]")

        if isinstance(content, nmx_runtime.Struct):
            args, kwargs = content.to_args_and_kwargs()
            string = []

            for i, arg in enumerate(args):
                if isinstance(arg, nmx_runtime.DocRef):
                    arg = self._doc_to_str(doc=arg)

                elif isinstance(arg, (nmx_runtime.Struct, nmx_runtime.Opaque)):
                    arg = self.convert_to_str(content=arg)

                string.append(f'{i}: {arg}')

            for k, v in kwargs.items():
                if isinstance(v, nmx_runtime.DocRef):
                    v = self._doc_to_str(doc=v)

                elif isinstance(v, (nmx_runtime.Struct, nmx_runtime.Opaque)):
                    v = self.convert_to_str(content=v)

                string.append(f'{k}: {v}')

            return f"{{{', '.join(string)}}}"

        return str(content)

    @staticmethod
    def _opaque_preview(opaque: nmx_runtime.Opaque):
        """Helper to generate a lightweight text summary of complex objects."""
        obj = opaque.unbox()
        type_name = type(obj).__name__

        try:
            import pandas as pd

            if isinstance(obj, pd.DataFrame):
                return f"DataFrame(shape={obj.shape}, columns={list(obj.columns)})"
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return f"np.ndarray({str(obj)})"
        except ImportError:
            pass

        # Fallback for generic objects
        return f"<{type_name} object>"
    
    @staticmethod
    def _ask_user(output) -> str:
        lines = textwrap.wrap(output, width=100)
        user_prompt = (f'[Last step output]\n{"\n".join(lines)}\n\n[Feedback]'
                       f'\nWrite user feedback: ')
        feedback = input(user_prompt)
        return feedback

    @staticmethod
    def _doc_to_str(doc: nmx_runtime.DocRef) -> str:
        return (f'{{"node_id": "{doc.node_id}", "content": "{doc.content}",'
                f'"breadcrumbs": "{doc.breadcrumbs}"}}')


class AgentState:
    def __init__(self, **kwargs):
        self._state = nmx_runtime.Struct()
        self.update(**kwargs)

    def get(self) -> nmx_runtime.Struct:
        return self._state

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self._state.set(key=k, value=self._box_value(v))

    def delete(self, *keys):
        for k in keys:
            self._state.pop(k, None)

    def _box_value(self, value):
        if value is None:
            return None

        if isinstance(value, (bool, str, int, float, nmx_runtime.DocRef,
                              nmx_runtime.Struct, nmx_runtime.Opaque)):
            return value

        if isinstance(value, (list, tuple, set)):
            struct = nmx_runtime.Struct()

            for v in value:
                v = self._box_value(v)
                struct.set(value=v, key=None)

            return struct

        if isinstance(value, dict):
            struct = nmx_runtime.Struct()

            for k, v in value.items():
                v = self._box_value(v)
                struct.set(value=v, key=k)

            return struct

        return nmx_runtime.Opaque(obj=value)

    def __repr__(self):
        size = len("Struct(")
        return f"AgentState({str(self._state)[size:-1]})"
