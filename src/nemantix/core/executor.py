import json
import textwrap
from pathlib import Path
from typing import Any, Iterable, Optional

from lark import LarkError
from pydantic import BaseModel

from nemantix.common import context
from nemantix.common.logger import get_package_logger
from nemantix.core import node as nmx_nodes
from nemantix.core.coder import CodeOperationEnum
from nemantix.core.exceptions import (
    NemantixException,
    NemantixParserException,
    NemantixRuntimeException,
)
from nemantix.core.expertise import Expertise, FallbackEnum
from nemantix.core.interpreter import Interpreter
from nemantix.core.node import FileMeta, PlanQualifierEnum
from nemantix.core.parser import AstTransformer, _get_fstring_parser
from nemantix.core.prompt import (
    DELIBERATE_SELECTION_PROMPT,
    FALLBACK_IDENTITY_PROMPT,
    REQUEST_PARSING_PROMPT,
)
from nemantix.core.runtime import ExternalVariables, Struct
from nemantix.core.script import Script
from nemantix.core.source_manager import LocalSourceManager
from nemantix.hub.events import Event, EventType
from nemantix.llm import (
    AbstractLLMProxy,
    LLMProxyConfig,
    LLMResponse,
    StructuredLLMResponse,
)

logger = get_package_logger(__name__)


class Executor:
    """Code execution according to user request"""

    _CODED_TEMPLATE = """
        deliberate x when >> x <<:
            plan:
                in:
                    x (optional)
                __
                out:
                __
                body:
                    {}
                __
            __
        __
    """
    _NO_DELIBERATE = "<NONE>"
    _TEMP_PATH = Path("_temp.nxs")

    class SelectionSchema(BaseModel):
        name: str
        motivation: Optional[str]

    class IdentitySchema(BaseModel):
        name: str
        when: str
        guidelines: str

    class PhaseEvent:
        def __init__(
            self, executor: "Executor", phase: str, deliberate: nmx_nodes.Deliberate
        ):
            self.executor = executor
            self.phase = phase
            self.deliberate = deliberate

        def __enter__(self):
            self.executor._emit_executor_event(
                EventType.EXECUTOR_PHASE_START,
                phase=self.phase,
                deliberate=self.deliberate.name,
            )

        def __exit__(self, *args):
            self.executor._emit_executor_event(
                EventType.EXECUTOR_PHASE_END,
                phase=self.phase,
                deliberate=self.deliberate.name,
            )

    # TODO: deprecate "llm" argument
    def __init__(
        self,
        expertise: Expertise,
        proxy_config: LLMProxyConfig,
        llm: AbstractLLMProxy | None = None,
        embedder: Optional[Any] = None,
        external_vars: Optional[dict] = None,
        knowledge_base: Optional[Any] = None,
        agent_state: Struct | None = None,
    ):
        assert isinstance(llm, AbstractLLMProxy)

        if embedder is not None:
            from nemantix.knowledge_base.models.embedding import TextEmbedding

            assert isinstance(embedder, TextEmbedding)

        if knowledge_base is not None:
            from nemantix.knowledge_base import NemantixKnowledgeBase

            assert isinstance(knowledge_base, NemantixKnowledgeBase)

        if llm is not None:
            logger.warning(
                'Argument "llm" will be deprecated in favor of "proxy_config"!'
            )

        if external_vars is None:
            external_vars = {}
        elif not isinstance(external_vars, dict):
            logger.warning(
                f'Provided "external_vars" are ignored because not a dict but a "{type(external_vars)}"!'
            )
            external_vars = {}

        assert isinstance(external_vars, dict)
        self.expertise = expertise
        self._volatile_state: tuple | None = None

        self.proxies = proxy_config
        self.llm = llm or self.proxies.internal
        self.embedder = embedder
        self.knowledge_base = knowledge_base
        self.external_vars = ExternalVariables(**external_vars)

        self.interpreter = Interpreter(
            expertise=self.expertise,
            embedder=self.embedder,
            llm=self.llm,
            knowledge_base=self.knowledge_base,
            agent_state=agent_state,
            external_variables=self.external_vars,
            proxy_config=self.proxies,
        )

    def execute(self, user_request: str, **__):
        self._emit_request(user_request)

        # build nxc if is not already
        if not self.expertise.is_fully_coded():
            self.expertise.build()
        else:
            self.expertise.update()

        # check if the request is coded
        uncoded = False
        inputs = None

        # Phase 1: parse the request (coded or uncoded via LLM)
        self._emit_executor_event(EventType.EXECUTOR_PHASE_START, phase="parse_request")
        try:
            # check if coded
            script, deliberate, inputs = self.parse_coded_request(user_request)

        except (LarkError, SyntaxError, NemantixParserException):
            # otherwise uncoded
            logger.info("Request is uncoded, processing with LLM...")

            deliberate = self.parse_uncoded_request(user_request)
            script = self.expertise.get_script_from_deliberate(
                deliberate_name=deliberate.name
            )
            uncoded = True

        self._emit_executor_event(
            EventType.EXECUTOR_PHASE_END,
            phase="parse_request",
            deliberate=deliberate.name,
            uncoded=uncoded,
        )

        # in every case, if the plan is not frozen, it must be completed
        plan = deliberate.get_plan()
        if plan is not None:
            qualifier = plan.qualifier[1] if plan.qualifier is not None else None
        else:
            qualifier = (
                deliberate.qualifier[1] if deliberate.qualifier is not None else None
            )

        if not plan or qualifier != PlanQualifierEnum.FROZEN:
            logger.info(f"Coding not frozen plan of deliberate {deliberate.name}")

            # Phase 2: code the deliberate
            with self.PhaseEvent(self, phase="code_deliberate", deliberate=deliberate):
                new_content = self.expertise.coder.code_deliberate(
                    coding_level=CodeOperationEnum.COMPLETE,
                    deliberate_name=deliberate.name,
                    script=script,
                    required_scripts=self.expertise.get_required_scripts(script),
                    user_request=user_request,
                )
                # update script in expertise (with coded deliberate)
                script = self.expertise.update_script_content(
                    script.get_location(), deliberate, new_content
                )

        # check for not frozen actions that are required by this deliberate, and update all affected scripts
        modified = self.uncoded_actions_runtime_coding(
            deliberate=deliberate, script=script, request=user_request
        )

        if modified:
            output_dir = script.source_manager.get_default_export_location()
            new_filename = (
                script.source_manager.get_file_name(script.get_location())
                + "_runtime.nxc"
            )
            export_path = script.source_manager.join(output_dir, new_filename)
            script.write(script.content, source_manager=None, location=export_path)

        try:
            if uncoded:
                return self.execute_uncoded_request(user_request, deliberate, script)

            return self.interpreter.interpret_coded_request(
                script, request_deliberate=deliberate, user_inputs=inputs
            )
        finally:
            self._cleanup_volatile_deliberate()

    def execute_uncoded_request(
        self, user_request: str, deliberate: nmx_nodes.Deliberate, script: Script
    ):
        with self.PhaseEvent(self, phase="parse_inputs", deliberate=deliberate):
            user_inputs = self._parse_user_inputs(
                request=user_request, script=script, deliberate=deliberate
            )

        return self.interpreter.interpret(
            deliberate=deliberate, user_inputs=user_inputs
        )

    def get_agent_state(self) -> Struct:
        return self.interpreter.agent_state

    def uncoded_actions_runtime_coding(
        self, deliberate: nmx_nodes.Deliberate, script: Script, request: str
    ):
        """Code all not completed actions used by the deliberate."""
        visible_actions = self.expertise.get_visible_actions_names(script)
        modified = False

        def visit_stmt(stmt):
            nonlocal modified

            if isinstance(stmt, nmx_nodes.DoStatement):
                callable_type = stmt.callable_type

                if callable_type == nmx_nodes.CallableTypeEnum.ACTION or (
                    callable_type is None and stmt.name in visible_actions
                ):
                    logger.debug(str(stmt))
                    action_name = stmt.name

                    if stmt.name in self.expertise.action_to_script_loc:
                        # action is global
                        action_loc = self.expertise.action_to_script_loc[stmt.name]
                        action_script = self.expertise.script_by_loc[action_loc]
                        action = action_script.actions[action_name]

                    elif (
                        f"{deliberate.name}.{stmt.name}"
                        in self.expertise.private_action_to_script_loc
                    ):
                        # action is deliberate-private
                        action_name = f"{deliberate.name}.{stmt.name}"
                        action_loc = self.expertise.private_action_to_script_loc[
                            action_name
                        ]
                        action_script = self.expertise.script_by_loc[action_loc]
                        action = action_script.private_actions[action_name]
                    else:
                        raise NemantixRuntimeException(
                            f'Cannot find action "{stmt.name}" in either global or '
                            f'private actions of deliberate "{deliberate.name}"!'
                        )

                    action_qualifier = action.get_qualifier()

                    if (
                        action_qualifier is None
                        or action_qualifier[1] != PlanQualifierEnum.FROZEN
                    ):
                        logger.info(
                            f"Found a call to a non-frozen action '{stmt.name}': "
                            f"runtime coding using request..."
                        )

                        required_scripts = self.expertise.get_required_scripts(
                            action_script
                        )

                        assert isinstance(action_name, str)
                        res = self.expertise.coder.code_action(
                            coding_level=CodeOperationEnum.COMPLETE,
                            action_name=action_name,
                            script=action_script,
                            required_scripts=required_scripts,
                            user_request=request,
                        )

                        self.expertise.update_script_content(action_loc, action, res)
                        modified = True

            elif isinstance(stmt, nmx_nodes.BlockStatement):
                for child in stmt.children:
                    visit_stmt(child)

        for statement in deliberate.get_plan().children or []:
            visit_stmt(statement)

        return modified

    # TODO: fallback deliberate should only occur if enabled and if the request
    #  is outside the provided deliberates but still within the same domain!
    def parse_uncoded_request(self, request: str) -> nmx_nodes.Deliberate:
        selection_response = self._select_deliberate_name(request)
        deliberate_name = selection_response.name
        motivation = selection_response.motivation or "None"

        if deliberate_name.upper() == self._NO_DELIBERATE:
            if self.expertise.allow_fallback_deliberate == FallbackEnum.NONE:
                raise NemantixRuntimeException(
                    f'Request "{request}" cannot be answered by provided deliberates!\n'
                    f'Motivation: "{motivation}".'
                )

            # no existing fallback name — fall back to any registered fallback
            fallback_name = next(iter(self.expertise.fallback_names), None)
            if fallback_name is None:
                raise NemantixRuntimeException(
                    f'Request "{request}" cannot be answered and no fallback '
                    f'deliberate is available. Motivation: "{motivation}".'
                )
            if self.expertise.allow_fallback_deliberate == FallbackEnum.PERSISTENT:
                deliberate_name = self._promote_fallback(request, fallback_name)
            elif self.expertise.allow_fallback_deliberate == FallbackEnum.VOLATILE:
                deliberate_name = self._promote_fallback(
                    request, fallback_name, volatile=True
                )

        elif self.expertise.is_fallback(deliberate_name):
            if self.expertise.allow_fallback_deliberate == FallbackEnum.PERSISTENT:
                deliberate_name = self._promote_fallback(request, deliberate_name)
            elif self.expertise.allow_fallback_deliberate == FallbackEnum.VOLATILE:
                deliberate_name = self._promote_fallback(
                    request, deliberate_name, volatile=True
                )

        script = self.expertise.get_script_from_deliberate(deliberate_name)
        return script.deliberates[deliberate_name]

    def parse_coded_request(
        self, request: str
    ) -> tuple[Script, nmx_nodes.Deliberate, nmx_nodes.Expression | None]:
        # temp script
        coded_request = self._CODED_TEMPLATE.format(request)
        logger.debug(coded_request)

        temp_script = Script(
            location=self._TEMP_PATH,
            source_manager=LocalSourceManager(),
            content=coded_request,
        )
        temp_script.parse()
        logger.debug(
            f"Deliberates: {str([v.name for v in temp_script.deliberates.values()])},"
            f"\nContent: {temp_script.content}"
        )

        temp_deliberate = list(temp_script.deliberates.values())[0]
        logger.debug(temp_deliberate)

        statements = temp_deliberate.get_plan().children or []
        if len(statements) > 1:
            raise NemantixParserException(
                f"Only one statement allowed in coded request: found {len(statements)}!"
            )
        if not isinstance(statements[0], nmx_nodes.DoStatement):
            raise NemantixParserException("Only do statement allowed in coded request!")

        logger.info("Parsed coded request.")

        do_stmt = statements[0]
        assert isinstance(do_stmt, nmx_nodes.DoStatement)

        # lookup the deliberate (or script) that contains the requested action
        node_name = do_stmt.name

        if node_name not in self.expertise.deliberate_to_script_loc:
            raise NemantixRuntimeException(
                f'Coded request: deliberate "{node_name}" does not exist!'
            )

        loc = self.expertise.deliberate_to_script_loc[node_name]
        script = self.expertise.script_by_loc[loc]
        deliberate = script.deliberates[node_name]

        return script, deliberate, do_stmt.using

    def _parse_user_inputs(
        self,
        request: str,
        deliberate: nmx_nodes.Deliberate,
        script: Script,
        max_attempts=6,
    ) -> nmx_nodes.Collection | None:
        correction = ""
        file_meta = deliberate.meta.get("file_meta")
        assert isinstance(file_meta, FileMeta)

        action_info = script.delib_semantics_map[deliberate.name].to_dict()
        action_text = str(
            {"inputs": action_info["ins"], "outputs": action_info["outs"]}
        )

        for i in range(max_attempts):
            prompt = REQUEST_PARSING_PROMPT.format(request, action_text, correction)
            logger.debug(f'Request parsing prompt: "{prompt}"')
            response = self.llm.invoke(prompt)
            self._emit_llm(llm_response=response)
            answer = response.text

            try:
                inputs = json.loads(answer)
                logger.debug(f'Extracted inputs from request: "{inputs}".')

                if isinstance(inputs, dict):
                    for possible_key in ["inputs", "parameters", "data", "args"]:
                        if possible_key in inputs and isinstance(
                            inputs[possible_key], (list, tuple)
                        ):
                            inputs = inputs[possible_key]
                            break

                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]

                for input_spec in inputs:
                    missing_fields = []

                    if "name" not in input_spec:
                        missing_fields.append("name")

                    if "type" not in input_spec:
                        missing_fields.append("type")

                    if "value" not in input_spec:
                        missing_fields.append("value")

                    if len(missing_fields) > 0:
                        raise NemantixRuntimeException(
                            f'Missing fields: "{missing_fields}"'
                        )

                return self._inputs_from_request(required_inputs=inputs)

            except (
                json.JSONDecodeError,
                RuntimeError,
                LarkError,
                NemantixException,
            ) as e:
                logger.warning(f"[{e.__class__.__name__}]: {e}")
                correction = (
                    f'Your response was faulty. Adjust to solve the error "{e}". '
                    f"You have {max_attempts - i - 1} attempts left."
                )

        logger.error(f'Cannot format user request "{request}" as JSON!', exc_info=True)
        return None

    @classmethod
    def _inputs_from_request(
        cls, required_inputs: list[dict]
    ) -> nmx_nodes.Collection | None:
        inputs = []
        parser = _get_fstring_parser()
        transformer = AstTransformer()
        meta = None

        for input_spec in required_inputs:
            name = input_spec["name"].strip()

            if not name.isidentifier():
                raise NemantixRuntimeException(
                    f'"{name}" is not a valid variable name!'
                )

            raw_kind = input_spec.get("type")
            value = input_spec.get("value", None)

            # 1. Safely handle missing/None type by inferring from value
            if raw_kind is None:
                if value is None:
                    kind = "none"
                elif isinstance(
                    value, bool
                ):  # Must check before int (bool is a subclass of int)
                    kind = "bool"
                elif isinstance(value, int):
                    kind = "int"
                elif isinstance(value, float):
                    kind = "float"
                elif isinstance(value, str):
                    kind = "str"
                elif isinstance(value, list):
                    kind = "list"
                elif isinstance(value, dict):
                    kind = "dict"
                else:
                    kind = "none"
            else:
                # 2. Safely cast to string in case the LLM returned an unexpected type object
                kind = str(raw_kind).strip().lower()

            if kind in ["str", "string"]:
                value = str(value).replace('"', '\\"')
                stmt = f'[[{name}] = "{value}"]'

            elif kind in [
                "int",
                "integer",
                "float",
                "double",
                "numeric",
                "number",
                "num",
            ]:
                stmt = f"[[{name}] = {value}]"

            elif kind in ["bool", "boolean"]:
                if value:
                    stmt = f"[[{name}] = true]"
                else:
                    stmt = f"[[{name}] = false]"

            elif kind in ["none", "nan", "null", "nil"] or value is None:
                stmt = f"[[{name}] = none]"

            elif kind in ["list", "array", "sequence"]:
                assert isinstance(value, Iterable)
                buffer = []

                for v in value:
                    # Format elements properly based on their Python type
                    if isinstance(v, str):
                        buffer.append(f'"{v}"')
                    elif isinstance(v, bool):
                        buffer.append("true" if v else "false")
                    elif v is None:
                        buffer.append("none")
                    elif isinstance(v, dict):
                        struct_str = cls._parse_dict_to_struct(v)
                        buffer.append(struct_str)
                    else:
                        buffer.append(str(v))  # Handles int and float

                if len(buffer) > 0:
                    # NXS uses parentheses () for list literals
                    if len(buffer) == 1:
                        buffer.append("")
                    stmt = f"[[{name}] = ({', '.join(buffer)})]"
                else:
                    stmt = f"[[{name}] = ()]"

            elif kind == "opaque":
                stmt = f"[[{name}] = [STATE:__opaque_{value}__]]"
            else:
                assert kind in ["dict", "dictionary"]
                assert isinstance(value, dict)
                struct_str = cls._parse_dict_to_struct(value)
                stmt = f"[[{name}] = {struct_str}]"

            logger.debug(f'parsing: "{stmt}"')

            tree = parser.parse(stmt)
            tree = transformer.transform(tree)

            assert len(tree.children) == 1
            inputs.append(tree.children[0])
            meta = tree.children[0].meta

        if len(inputs) == 0:
            return None

        collection = nmx_nodes.Collection(
            value=inputs, meta=meta, inferred_type=nmx_nodes.VariableTypeEnum.LIST
        )
        return collection

    def _promote_fallback(
        self, request: str, fallback_name: str, volatile: bool = False
    ) -> str:
        """Turn a fallback deliberate into a concrete, reusable deliberate.

        1. Ask the LLM for a new identity (name / when / guidelines) generalizing
           the request.
        2. Swap the fallback block in-script with a new-deliberate skeleton carrying
           that identity.
        3. Code the plan of the new deliberate from the user request using the
           existing `Coder.code_deliberate(COMPLETE, ...)` path.
        4. Code any non-frozen actions referenced by the new plan.
        5. Append a fresh fallback deliberate to the same script so it always has
           one uncoded fallback available.
        Returns the name of the newly promoted deliberate.
        """
        script_loc = self.expertise.deliberate_to_script_loc[fallback_name]
        script = self.expertise.script_by_loc[script_loc]
        fallback_deliberate = script.deliberates[fallback_name]

        if volatile:
            original_content = (
                script.content
                if isinstance(script.content, str)
                else "\n".join(script.content)
            )

        # 1. generate identity
        identity = self._generate_deliberate_identity(request)
        new_name = self._resolve_unique_deliberate_name(identity.name, fallback_name)
        logger.info(
            f'Promoting fallback "{fallback_name}" on script "{script_loc}" '
            f'into new deliberate "{new_name}".'
        )

        # 2. replace the fallback block with the new-deliberate skeleton
        new_block = self.expertise.build_promoted_deliberate_block(
            name=new_name, when=identity.when, guidelines=identity.guidelines
        )

        script = self.expertise.update_script_content(
            script_loc, original_node=fallback_deliberate, new_content=new_block
        )

        # fallback name is gone from the script; drop it from tracking
        self.expertise.discard_fallback_name(fallback_name)

        # 3. code the plan of the new deliberate using the user request
        required_scripts = self.expertise.get_required_scripts(script)

        coded_content = self.expertise.coder.code_deliberate(
            coding_level=CodeOperationEnum.COMPLETE,
            deliberate_name=new_name,
            script=script,
            required_scripts=required_scripts,
            user_request=request,
        )
        coded_content = textwrap.dedent(coded_content)

        # TODO: fix indentation
        script = self.expertise.update_script_content(
            script_loc,
            original_node=script.deliberates[new_name],
            new_content=coded_content,
        )

        new_deliberate = script.deliberates[new_name]

        # 4. code any non-frozen actions referenced by the coded plan
        self.uncoded_actions_runtime_coding(
            deliberate=new_deliberate, script=script, request=request
        )

        if not volatile:
            # update coded scripts
            self.expertise.export()
            # 5. append a fresh fallback for future requests
            self.expertise.append_fallback_deliberate(script_loc)
        else:
            self._volatile_state = (script_loc, original_content, fallback_name)

        return new_name

    def _cleanup_volatile_deliberate(self) -> None:
        if self._volatile_state is None:
            return
        script_loc, original_content, fallback_name = self._volatile_state
        self._volatile_state = None

        script = self.expertise.script_by_loc[script_loc]
        script.content = original_content
        script.parse()
        self.expertise._refresh_script_maps(script_loc)
        self.expertise.fallback_names.add(fallback_name)
        self.expertise.fallback_name_by_script_loc[script_loc] = fallback_name

    def _generate_deliberate_identity(self, request: str) -> IdentitySchema:
        existing_names = sorted(
            set(self.expertise.deliberate_to_script_loc.keys())
            | self.expertise.fallback_names
        )
        prompt = FALLBACK_IDENTITY_PROMPT.format(
            request=request,
            existing_names=", ".join(existing_names) if existing_names else "(none)",
        )
        logger.debug(f'Fallback identity prompt:\n"{prompt}"')

        response = self.llm.invoke_structured(prompt, schema=self.IdentitySchema)
        self._emit_llm(llm_response=response)
        identity = response.result

        # minimal sanitization: ensure name is a valid identifier
        name = (identity.name or "").strip()
        if not name.isidentifier():
            name = "".join(c for c in name if c.isalnum() or c == "_")
            if not name or not name[0].isalpha():
                name = "PromotedDeliberate"
            identity.name = name

        assert isinstance(identity, self.IdentitySchema)
        return identity

    def _resolve_unique_deliberate_name(self, proposed: str, fallback_name: str) -> str:
        """Ensure the new deliberate name does not collide with existing ones.

        The currently-being-promoted fallback name is excluded from the collision
        check since it is about to be removed.
        """
        existing = (
            set(self.expertise.deliberate_to_script_loc.keys())
            | self.expertise.fallback_names
        )
        existing.discard(fallback_name)
        if proposed not in existing:
            return proposed
        suffix = 2
        while f"{proposed}_{suffix}" in existing:
            suffix += 1
        return f"{proposed}_{suffix}"

    def _select_deliberate_name(
        self, request: str, prompt_correction=""
    ) -> SelectionSchema:
        """Uses a LLM to predict the most suitable deliberate given the request"""
        if self.llm is None:
            raise NemantixRuntimeException("LLM is not configured")

        deliberate_sem_map = self.expertise.get_all_deliberates_semantics()
        prompt = DELIBERATE_SELECTION_PROMPT.format(
            deliberate_sem_map, request, prompt_correction
        )
        logger.debug(f'Prompt used for deliberate selection:\n"{prompt}"')

        response = self.llm.invoke_structured(prompt, schema=self.SelectionSchema)
        self._emit_llm(llm_response=response)
        return response.result

    @staticmethod
    def _emit_executor_event(event_type: EventType, phase: str, **extra_payload):
        hub = context.event_hub.get()
        if hub is None or not hub.has_subscribers(event_type):
            return
        event = Event(
            type=event_type,
            lines=(0, 0),
            scope="executor",
            script=None,
            statement="",
            payload=dict(phase=phase, **extra_payload),
        )
        hub.emit(event)

    @staticmethod
    def _emit_llm(llm_response: LLMResponse | StructuredLLMResponse) -> None:
        hub = context.event_hub.get()
        if hub is None or not hub.has_subscribers(EventType.LLM):
            return
        event = Event(
            type=EventType.LLM,
            lines=(0, 0),
            scope="executor",
            script=None,
            statement="",
            payload=dict(
                usage=llm_response.usage,
                name=llm_response.proxy.get_name(),
                internal_usage=True,
            ),
        )
        hub.emit(event)

    @staticmethod
    def _emit_request(user_request: str):
        hub = context.event_hub.get()
        if hub is None or not hub.has_subscribers(EventType.USER_REQUEST):
            return

        event = Event(
            type=EventType.USER_REQUEST,
            lines=(0, 0),
            scope="executor",
            script=None,
            statement="",
            payload=dict(request=user_request),
        )
        hub.emit(event)

    @staticmethod
    def _parse_value(v_):
        if isinstance(v_, str):
            v_ = f'"{v_.replace('"', '\\"')}"'
        elif v_ is None or isinstance(v_, bool):
            v_ = str(v_).lower()
        else:
            v_ = str(v_)

        return v_

    @classmethod
    def _parse_dict_to_struct(cls, dict_to_parse: dict):
        buffer = []
        if "__opaque__" in dict_to_parse:
            opaque = f"[STATE:{dict_to_parse['__opaque__']}]"
            return opaque
        for k_, v_ in dict_to_parse.items():
            v_ = cls._parse_value(v_)
            try:
                k_ = int(k_)
            except ValueError:
                pass
            if isinstance(k_, int):
                buffer.append(v_)
            else:
                assert isinstance(k_, str)
                buffer.append(f"{k_}: {v_}")
        struct_str = f"({', '.join(buffer)})"
        return struct_str
