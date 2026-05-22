import json
import textwrap
import traceback
from enum import Enum

from lark import LarkError

from nemantix.common import context
from nemantix.common.logger import get_package_logger
from nemantix.core.exceptions import (
    NemantixException,
    NemantixParserException,
    NemantixRuntimeException,
)
from nemantix.core.node import (
    ActionBlock,
    BlockStatement,
    Deliberate,
    DoStatement,
    FileMeta,
    Frame,
    ImportToolsetStatement,
    MicroPrompt,
    Statement,
)
from nemantix.core.parser import _get_frame_parser
from nemantix.core.prompt import (
    CODING_ADDITIONAL_INFO,
    CODING_DELIBERATE_ADDITIONAL_INFO,
    CODING_SYSTEM_PROMPT,
    COMPILATION_ACTION,
    COMPILATION_DELIBERATE,
    COMPILATION_DELIBERATE_BREAKDOWN,
    COMPLETE_ACTION_RULES,
    COMPLETE_DELIBERATE_BREAKDOWN_RULES,
    COMPLETE_DELIBERATE_RULES,
    DRAFT_ACTION_RULES,
    DRAFT_DELIBERATE_BREAKDOWN_RULES,
    DRAFT_DELIBERATE_RULES,
    EVALUATE_ACTION_RULES,
    EVALUATE_DELIBERATE_BREAKDOWN_RULES,
    EVALUATE_DELIBERATE_RULES,
    FIX_GENERATION,
    GEN_FRAME_PROMPT,
    GEN_TOOLSET_PROMPT,
    USER_REQUEST,
)
from nemantix.core.runtime import get_globals
from nemantix.core.script import Script
from nemantix.core.tools import Toolset
from nemantix.hub.events import Event, EventType
from nemantix.llm import AbstractLLMProxy
from nemantix.llm.abstract_proxy import LLMUsage

logger = get_package_logger(__name__)


class CodingModeEnum(Enum):
    DELIBERATE_LEVEL = 1
    ACTION_LEVEL = 0


class CodeOperationEnum(Enum):
    SKIP = 0
    DRAFT = 1
    COMPLETE = 2
    EVALUATE = 3


qualifier_coding_map = {
    # first none
    "none->undefined": CodeOperationEnum.SKIP,
    "none->drafted": CodeOperationEnum.DRAFT,
    "none->frozen": CodeOperationEnum.COMPLETE,
    # second none
    "undefined->none": CodeOperationEnum.EVALUATE,
    "drafted->none": CodeOperationEnum.EVALUATE,
    "frozen->none": CodeOperationEnum.SKIP,
    # all
    "undefined->drafted": CodeOperationEnum.DRAFT,
    "undefined->frozen": CodeOperationEnum.COMPLETE,
    "drafted->frozen": CodeOperationEnum.COMPLETE,
    # same
    "none->none": CodeOperationEnum.EVALUATE,
    "undefined->undefined": CodeOperationEnum.SKIP,
    "drafted->drafted": CodeOperationEnum.EVALUATE,
    "frozen->frozen": CodeOperationEnum.SKIP,
}


class Coder:
    def __init__(self, llm_proxy: AbstractLLMProxy):
        self.llm_proxy = llm_proxy
        self.action_semantics_map: dict[str, dict[str, str]] = {}  # dict[deliberate_name, dict[action_name, semantics]]
        self.runtime_globals = get_globals()
        self.external_vars_names = None
        self.knowledge_base = None
        self.enable_fixer = False

    def coding(self, script: Script, required_scripts: list[Script], external_vars_names: list[str] = None):
        """
        Performs the coding of the parsed NXS content into NXC output
        Args:
            :param script: Script to be coded
            :param required_scripts: Scripts required by the script to be coded
            :param external_vars_names: names of the Agent's external variables
        """
        self.external_vars_names = external_vars_names if external_vars_names else self.external_vars_names
        # code nxs-declared toolsets
        if len(script.toolsets_decl) > 0:
            toolset_coding_result = self.code_script_toolsets(script)
            script.update(content=toolset_coding_result, enable_fixer=self.enable_fixer)

        # code actions
        action_coding_result = self.code_script_actions(script, required_scripts)
        script.update(content=action_coding_result, enable_fixer=self.enable_fixer)

        # add deliberate
        deliberate_coding_result = self.code_script_deliberates(script, required_scripts)
        script.update(content=deliberate_coding_result, enable_fixer=self.enable_fixer)

        # add frames
        coding_result = self.code_script_frames(script)

        return coding_result

    def code_script_actions(self, script: Script, required_scripts: list[Script]):
        """Code all actions in a Script"""
        actions = script.actions.values()

        non_deliberate_list = script.requires + list(
            script.toolset_imports.values()) + script.toolsets_decl + script.frames
        deliberate_list = list(script.deliberates.values())

        # Deliberates must be rewritten exactly as they are, just like the non_deliberates
        non_action_list = non_deliberate_list + deliberate_list
        content_list = script.read_as_list()

        ## first, copy all non-deliberate to nxc ##
        new_content = ""
        for n in non_action_list:
            nxs_content = self._read_node_nxs(script_content_list=content_list, node=n, read_as_list=False)
            new_content += nxs_content + "\n"

        # 1. Extract the actions from the required scripts and determine their semantics
        # 2. For each action, create a prompt that includes actions, tools, and frames (no deliberations)
        for action in actions:
            logger.info(f"Processing action {action.name}")
            qual = action.qualifier
            qual_str = "no"

            # qualified plan
            if qual is not None:
                qual_str = f"{qual[0].value}->{qual[1].value}"
                coding_type = qualifier_coding_map[qual_str]
            else:
                # like '_ -> _ = evaluate'
                coding_type = CodeOperationEnum.EVALUATE

            # coding procedure
            if coding_type == CodeOperationEnum.SKIP:
                nxs_content = self._read_node_nxs(script_content_list=content_list, node=action, read_as_list=False)
                new_content += nxs_content + "\n"
                logger.info(f"Skipping action '{action.name}' with {qual_str} completion")
                continue
            else:  # draft or frozen or evaluate
                logger.info(f"Coding action '{action.name}' with {qual_str} completion")
                assert action.name is not None
                res = self.code_action(coding_type, action.name, script, required_scripts)
                new_content += res + "\n"

            new_content += "\n"

        return new_content

    def code_script_deliberates(self, script: Script, required_scripts: list[Script]):
        """Code all actions in a Script"""
        deliberates = script.deliberates.values()
        non_deliberate_list = (script.requires + list(script.toolset_imports.values()) +
                               script.toolsets_decl + script.frames + list(script.actions.values()))
        content_list = script.read_as_list()

        ## first, copy all non-deliberate to nxc ##
        new_content = ""
        for n in non_deliberate_list:
            nxs_content = self._read_node_nxs(script_content_list=content_list, node=n, read_as_list=False)
            new_content += nxs_content + "\n"

        # 1. Extract the actions from the required scripts and determine their semantics
        # 2. For each action, create a prompt that includes actions, tools, and frames (no deliberations)
        for deliberate in deliberates:
            qual = deliberate.qualifier  # same as plan qualifier
            qual_str = "no"

            # qualified plan
            if qual is not None:
                qual_str = f"{qual[0].value}->{qual[1].value}"
                coding_type = qualifier_coding_map[qual_str]
            else:
                # like '_ -> _ = evaluate'
                coding_type = CodeOperationEnum.EVALUATE

            # coding procedure
            if coding_type == CodeOperationEnum.SKIP:
                nxs_content = self._read_node_nxs(script_content_list=content_list, node=deliberate, read_as_list=False)
                new_content += nxs_content + "\n"
                logger.info(f"Skipping deliberate '{deliberate.name}' with {qual_str} completion")
                continue
            else:  # draft or frozen
                logger.info(f"Coding deliberate '{deliberate.name}' with {qual_str} completion")
                res = self.code_deliberate(coding_type, deliberate.name, script, required_scripts)
                new_content += res + "\n"

            new_content += "\n"

        return new_content

    def code_script_frames(self, script: Script, max_retries=6):
        """Code all frames in a Script"""
        defined_frames = {frame.name: frame for frame in script.frames}
        frame_usages = self._extract_do_schema(script)
        coded_frames = []
        frame_parser = _get_frame_parser()
        content_list = script.read_as_list()

        for frame_name, usages in frame_usages.items():
            prev_frame = None

            if self._is_partial_frame(frame=defined_frames.get(frame_name, None)):
                prev_frame = self._read_node_nxs(script_content_list=content_list,
                                                 node=defined_frames[frame_name],
                                                 read_as_list=False)
                assert isinstance(prev_frame, str)

            elif frame_name in defined_frames:
                # skip as already coded or defined
                coded_frames.append(defined_frames[frame_name])
                continue

            # code frame
            prev_error = None
            frame = None
            attempt = 0

            self._emit_coding_start(script, scope=frame_name, kind='frame')
            for attempt in range(max_retries):
                frame = self.generate_frame(frame_name, usages, previous_frame=prev_frame,
                                            previous_error=prev_error)
                try:
                    frame_parser.parse(frame)
                    break
                except SyntaxError as e:
                    prev_error = str(e)
                    prev_frame = frame
                    logger.warning(f'Error during frame coding: {e}')

            if frame is None:
                self._emit_coding_error(error=f'Cannot code frame "{frame_name}"',
                                        code=frame or '', lines=(0, 0),
                                        scope='frame')
                raise NemantixRuntimeException(f'Cannot code frame "{frame_name}"')

            self._emit_coding_end(script, scope=frame_name, kind='frame', attempts=attempt + 1)
            coded_frames.append(frame)

        # copy all the nodes in the right order
        node_list = (script.requires + list(script.toolset_imports.values()) +
                     script.toolsets_decl + coded_frames + list(script.actions.values()) +
                     list(script.deliberates.values()))

        new_content = ""
        for n in node_list:
            if isinstance(n, str):
                nxs_content = n
            else:
                nxs_content = self._read_node_nxs(script_content_list=content_list, node=n,
                                                  read_as_list=False)
            new_content += nxs_content + "\n\n"

        return new_content

    def code_script_toolsets(self, script: Script, max_retries: int = 6):
        """
        :param script:
        :param max_retries:
        """
        content_list = script.read_as_list()
        implemented_nxs = []

        for declaration in script.toolsets_decl:
            generated_code = None
            locals_var = self.runtime_globals
            last_error = ""
            code = None

            prompt = declaration.prompt.prompt
            toolset_name = declaration.name
            description = declaration.prompt.prompt
            nxs_content = self._read_node_nxs(script_content_list=content_list, node=declaration, read_as_list=False)

            # 1. Flatten the loops and filter early to reduce indentation
            matching_imports = (imp for name, imp in script.toolset_imports.items()
                                if (len(name.split(":")) == 1 and name == toolset_name)
                                or (len(name.split(":")) > 1 and name.split(":")[0] == toolset_name))

            imports_str, toolset_alias, tools_name = self._extract_import_str(list(matching_imports),
                                                                              script.read_as_list())
            toolset_alias.add(toolset_name)
            do_str = self._extract_do_str(toolset_alias, script)
            should_generate = False

            for attempt in range(max_retries):
                index = prompt.find("class ")

                if index == -1 or attempt > 0:
                    if not should_generate:
                        self._emit_coding_start(script, scope=declaration.name, kind='toolset')
                        should_generate = True

                    if attempt > 0:
                        logger.info(
                            f"Attempt {attempt + 1}: Execution failed previously. Regenerating code for {declaration.name}...")
                    else:
                        logger.info(f"No class found for {declaration.name}. Generating dynamically...")

                    if index != -1:
                        code = prompt[index:]
                        description = prompt[:index]

                    generated_code = self.generate_tool(
                        do_str=do_str,
                        toolset_name=toolset_name,
                        imports_str=imports_str,
                        description=description,
                        previous_error=last_error,
                        previous_code=code)

                    logger.debug(f"ATTEMPT: {attempt}\n->GENERATED CODE: {generated_code}")
                    prompt = generated_code
                    index = prompt.find("class ")

                    if index == -1:
                        if attempt == max_retries - 1:
                            raise NemantixRuntimeException(
                                f"Malformed tool declaration (generation failed completely): {declaration.name}",
                                statement=declaration, script=script)

                        continue

                # lookup for imports before class definition
                import_index = prompt.find("import ")
                from_index = prompt.find("from ")

                if import_index > -1:
                    index = min(index, import_index)

                if from_index > -1:
                    index = min(index, from_index)

                code = prompt[index:]
                logger.debug(f'Exec on "\n{code}\n"')

                try:
                    exec(code, self.runtime_globals, locals_var)
                    toolset_info = locals_var.get(declaration.name)

                    for tool_name in tools_name:
                        if toolset_info is None:
                            raise NemantixRuntimeException(
                                f"{tool_name} not implemented in {declaration.name}")

                        try:
                            tool_fn = toolset_info.get_tool(f"{declaration.name}.{tool_name}")
                            logger.debug(f"{tool_fn} implemented correctly")

                        except Exception:
                            raise NemantixRuntimeException(f"{tool_name} not implemented in {declaration.name}")

                    if not issubclass(locals_var.get(declaration.name), Toolset):
                        logger.warning("Toolset generated dynamically is not a subclass of Toolset")

                        raise NemantixRuntimeException(
                            f"{declaration.name} is not a subclass of Toolset")

                    elif generated_code is not None:
                        indented_code = textwrap.indent(generated_code, "    ")
                        indented_code = textwrap.dedent(indented_code)

                        str_toolset = (
                            f"toolset {toolset_name}:\n"
                            f"  >>>\n"
                            f"{indented_code}\n"
                            f"  <<<\n"
                            f"__toolset")

                        nxs_content = str_toolset
                        implemented_nxs.append(nxs_content)
                        self._emit_coding_end(script, scope=declaration.name,
                                              attempts=attempt + 1, kind='toolset')
                        break
                    else:
                        implemented_nxs.append(nxs_content)
                        break

                except Exception as e:
                    logger.warning(f"Execution failed on attempt {attempt + 1} for {declaration.name}: {e}")
                    last_error += f"\n{traceback.format_exc()}"

                    if attempt == max_retries - 1:
                        self._emit_coding_error(error='Invalid toolset code', scope='toolset',
                                                code=generated_code or '', lines=(0, 0))
                        raise NemantixRuntimeException(
                            f"Invalid toolset code! Failed to execute after {max_retries} attempts.",
                            statement=declaration, script=script)

        logger.debug(f"Old declarations\n{[decl.name for decl in script.toolsets_decl]}")
        for i, impl in enumerate(implemented_nxs):
            logger.debug(f"IMPLEMENTED\n{implemented_nxs[i]}")

            decl = script.toolsets_decl[i]
            file_meta: FileMeta = decl.meta["file_meta"]
            start_line, end_line = file_meta.line[0] - 1, file_meta.line[1] - 1

            replaced = self.replace_nxs_code_block(content_list, start_line, end_line, impl, indent=False)
            script.content = replaced
            logger.debug(f"REPLACED\n{replaced}")
            script.parse()

        return script.content

    def code_action(self, coding_level: CodeOperationEnum, action_name: str, script: Script,
                    required_scripts: list[Script], user_request=None):
        self._emit_coding_start(script, scope=action_name, kind='action')

        if action_name in script.actions:
            action = script.actions[action_name]
        else:
            action = script.private_actions[action_name]

        script_content_list = script.read_as_list()
        messages = self._build_action_coding_prompt(script_content_list, action, coding_level,
                                                    user_request, script, required_scripts)

        response = self.llm_proxy.invoke_grammar_based(messages)
        self._emit_llm(scope=action_name, usage=response.usage)
        resp = response.text

        file_meta = action.meta["file_meta"]
        assert isinstance(file_meta, FileMeta)
        end_line = file_meta.line[1] - 1

        orig_code = self._read_node_nxs(script_content_list=script_content_list, node=action, read_as_list=True)
        assert isinstance(orig_code, list)

        relative_start = 0
        relative_end = end_line
        res, attempts = self._check_and_fix_generated_code(messages, resp, relative_start, relative_end, orig_code,
                                                           scope=action_name)

        # TODO: should handle missing or removed @completion qualifier?

        self._emit_coding_end(script, scope=action_name, attempts=attempts, kind='action', request=user_request)
        return res

    def code_deliberate(self, coding_level: CodeOperationEnum, deliberate_name: str, script: Script,
                        required_scripts: list[Script], user_request=None):
        self._emit_coding_start(script, scope=deliberate_name, kind='deliberate')
        deliberate = script.deliberates[deliberate_name]
        qual = deliberate.qualifier

        script_content_list = script.read_as_list()
        messages = self._build_deliberate_coding_prompt(script_content_list, deliberate, coding_level, user_request,
                                                        script, required_scripts)

        logger.debug(f"Coding deliberate with message:\n{messages}")

        response = self.llm_proxy.invoke_grammar_based(messages)
        self._emit_llm(scope=deliberate_name, usage=response.usage)
        resp = response.text

        deliberate_file_meta = deliberate.meta["file_meta"]
        assert isinstance(deliberate_file_meta, FileMeta)
        deliberate_start_line = deliberate_file_meta.line[0] - 1

        deliberate_original_code = self._read_node_nxs(script_content_list=script_content_list,
                                                       node=deliberate, read_as_list=True)
        if deliberate.get_plan() is not None:
            plan = deliberate.get_plan()
            plan_meta = plan.meta["file_meta"]

            assert isinstance(plan_meta, FileMeta)
            plan_start_line, plan_end_line = plan_meta.line[0] - 1, plan_meta.line[1] - 1
        else:
            plan_meta = deliberate.meta["file_meta"]
            assert isinstance(plan_meta, FileMeta)
            plan_start_line, plan_end_line = plan_meta.line[1] - 1, plan_meta.line[1] - 2

        relative_start = plan_start_line - deliberate_start_line
        relative_end = relative_start + plan_end_line - plan_start_line

        while deliberate_original_code[relative_start].strip().startswith("@"):
            relative_start += 1  # start replacing from 'plan' not from annotations

        assert isinstance(deliberate_original_code, list)
        res, attempts = self._check_and_fix_generated_code(messages, resp, relative_start, relative_end,
                                                           deliberate_original_code, scope=deliberate.name)

        # Check for qualifier consistency
        temp_scr = Script("_temp.nxs", None, content=res)
        temp_scr.parse(enable_fixer=self.enable_fixer)

        # copy old qualifier if the coding removed it
        coded_qual = [v for v in temp_scr.deliberates.values()][0].qualifier
        if coded_qual is None and qual is not None:
            res = "@completion: " + f'{qual[0].value}->{qual[1].value}' + "\n" + res

        # add none->none qualifier if there was no @completion
        elif coded_qual is None and qual is None:
            res = "@completion: _->_ \n" + res

        self._emit_coding_end(script, scope=deliberate_name, attempts=attempts, kind='deliberate', request=user_request)
        return res

    def generate_frame(self, frame_name: str, usages: list[str], previous_frame: str = None,
                       previous_error: str = None) -> str:
        user_content = (
            f"Name: {frame_name}\n"
            f"Usage Statements:\n{'\n'.join(usages)}\n")

        if previous_frame:
            user_content += (
                f"\n--- PREVIOUS FRAME ---\n{previous_frame}\n---------------------\n")

        if previous_error:
            user_content += (
                f"\nATTENTION: The previously generated Frame failed during parsing with "
                f"the following error:\n{previous_error}\n"
                f"Analyze this error in the context of the previous frame. "
                f"Ensure you fix the specific line "
                f"or logic causing the issue in this new attempt.")

        # Create the message array for invoke method
        messages = [
            dict(role="system", content=GEN_FRAME_PROMPT),
            dict(role="user", content=user_content)]

        response = self.llm_proxy.invoke(prompt=messages)
        self._emit_llm(scope=frame_name, usage=response.usage)
        generated_frame = response.text

        logger.debug(f"LLM Generated Frame:\n{generated_frame}")
        return generated_frame

    def generate_tool(self, toolset_name, imports_str, do_str, description,
                      previous_error: str | None = None, previous_code: str | None = None) -> str:
        self._emit_coding_start(script=None, scope=toolset_name, kind='tool')

        user_content = (
            f"Name: {toolset_name}\n"
            f"Import Statement: {imports_str}\n"
            f"Usage Statements:\n{do_str}\n"
            f"Description: {description}\n")

        if previous_code:
            user_content += (
                f"\n--- PREVIOUS CODE ---\n{previous_code}\n---------------------\n")

        if previous_error:
            user_content += (
                f"\nATTENTION: The previously generated code failed during execution with the following error:\n"
                f"{previous_error}\n"
                f"Analyze this error in the context of the previous code. Ensure you fix the specific line or logic causing the issue in this new attempt."
            )

        # Create the message array for invoke method
        messages = [
            dict(role="system", content=GEN_TOOLSET_PROMPT),
            dict(role="user", content=user_content),
        ]

        response = self.llm_proxy.invoke(prompt=messages)
        self._emit_llm(scope=toolset_name, usage=response.usage)
        generated_code = response.text

        # Defensive cleanup: LLMs sometimes still output Markdown blocks despite instructions
        if generated_code.strip().startswith("```"):
            lines = generated_code.strip().split("\n")
            # Remove the first line (e.g., ```python) and the last line (```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            generated_code = "\n".join(lines).strip()

        logger.debug(f"LLM Generated Toolset Code:\n{generated_code}")
        self._emit_coding_end(script=None, scope=toolset_name, kind='tool', attempts=1)
        return generated_code

    @classmethod
    def _is_partial_frame(cls, frame: Frame | None) -> bool:
        if frame is None:
            return False

        for slot in frame.children:
            if isinstance(slot, MicroPrompt):
                return True

            if isinstance(slot, Frame):
                if cls._is_partial_frame(frame=slot):
                    return True

        return False

    def _extract_do_str(self, toolsets_alias, script: Script):
        extracted_contents = []
        seen_contents = set()
        script_content_list = script.read_as_list()

        actions = script.actions
        deliberates = script.deliberates

        do_stmts = []

        def _collect_do_statements(nodes, collected):
            if not nodes:
                return

            for node_ in nodes:
                if not node_:
                    continue
                if isinstance(node_, DoStatement):
                    collected.append(node_)

                # Se è un BlockNode, scendi nei figli
                if isinstance(node_, BlockStatement):
                    _collect_do_statements(getattr(node_, "children", []), collected)

        # 1. Action Children
        for _, action_node in actions.items():
            if not action_node:
                continue
            _collect_do_statements(getattr(action_node, "children", []), do_stmts)

        # 2. Deliberate children
        #    - Plan content
        #    - eventual generated_actions content
        for deliberate in deliberates.values():
            if not deliberate:
                continue
            plan = deliberate.get_plan()
            if plan:
                _collect_do_statements(getattr(plan, "children", []), do_stmts)
            generated_actions = getattr(deliberate, "generated_actions", None)
            if generated_actions:
                _collect_do_statements(getattr(generated_actions, "children", []), do_stmts)

        # 3. Extract snippets
        for node in do_stmts:
            toolset_name = node.name.split(".")[0] if node.name else ""

            if toolset_name not in toolsets_alias:
                continue

            snippet = self._read_node_nxs(script_content_list, node, read_as_list=False)

            if snippet not in seen_contents:
                extracted_contents.append(snippet)

                assert isinstance(snippet, str)
                seen_contents.add(snippet)

        return "\n".join(extracted_contents)

    def _extract_import_str(self, import_statements: list[ImportToolsetStatement], script_content_list: list[str]):
        extracted_imports = []
        tools_alias = set()
        tools_name = set()

        for imp in import_statements:
            # Update tracking sets
            if imp.alias:
                tools_alias.add(imp.alias)
            if imp.elements:
                tools_name.update(imp.elements)

            # Read nxs
            snippet = self._read_node_nxs(script_content_list, imp, read_as_list=False)
            if snippet:
                assert isinstance(snippet, str)
                extracted_imports.append(snippet.strip())

        return "\n".join(extracted_imports), tools_alias, tools_name

    def _extract_do_schema(self, script: Script) -> dict[str, list[str]]:
        seen_contents = set()
        script_content_list = script.read_as_list()
        do_stmts = []
        frames_with_usages = dict()

        def _collect_do_statements(nodes, collected):
            if not nodes:
                return

            for node_ in nodes:
                if not node_:
                    continue

                if isinstance(node_, DoStatement):
                    if node_.producing_schema is not None:
                        collected.append(node_)

                # Se è un BlockNode, scendi nei figli
                if isinstance(node_, BlockStatement):
                    _collect_do_statements(node_.children or [], collected)

        # actions
        for action_node in script.actions.values():
            if not action_node:
                continue

            _collect_do_statements(getattr(action_node, "children", []), do_stmts)

        # deliberates
        for deliberate in script.deliberates.values():
            if not deliberate:
                continue

            plan = deliberate.get_plan()
            if plan:
                _collect_do_statements(getattr(plan, "children", []), do_stmts)

            generated_actions = getattr(deliberate, "generated_actions", None)

            if generated_actions:
                _collect_do_statements(getattr(generated_actions, "children", []), do_stmts)

        # Extract snippets
        for node in do_stmts:
            frame_name = str(node.producing_schema)
            frames_with_usages.setdefault(frame_name, [])

            snippet = self._read_node_nxs(script_content_list, node, read_as_list=False)
            assert isinstance(snippet, str)

            if snippet not in seen_contents:
                frames_with_usages[frame_name].append(snippet.strip())

                assert isinstance(snippet, str)
                seen_contents.add(snippet)

        return frames_with_usages

    @staticmethod
    def _read_node_nxs(script_content_list: list[str], node: Statement, read_as_list=False):
        file_meta = node.meta['file_meta']
        assert isinstance(file_meta, FileMeta)

        node_start_line, node_end_line = file_meta.line[0] - 1, file_meta.line[1]
        orig_code = script_content_list[node_start_line:node_end_line]

        if not read_as_list:
            orig_code = "\n".join(orig_code)

        return orig_code

    def _build_action_coding_prompt(self, script_content: list[str], action: ActionBlock,
                                    coding_level: CodeOperationEnum,
                                    user_request: str, script: Script, required_scripts: list[Script]):
        system_prompt = CODING_SYSTEM_PROMPT

        rules = ""
        # rules on the basis of expected coding output
        if coding_level == CodeOperationEnum.DRAFT:
            rules = DRAFT_ACTION_RULES
        elif coding_level == CodeOperationEnum.COMPLETE:
            rules = COMPLETE_ACTION_RULES
        elif coding_level == CodeOperationEnum.EVALUATE:
            rules = EVALUATE_ACTION_RULES

        # add user request if present (runtime)
        user_req_prompt = "" if user_request is None else USER_REQUEST.format(user_request=user_request)

        orig_code = self._read_node_nxs(script_content_list=script_content, node=action, read_as_list=True)

        # build prompt
        task = COMPILATION_ACTION.format(rules=rules, action_nxs="\n".join(orig_code))

        ## Add imported tools and actions ##
        # Tools
        imported_tools = list(script.toolset_imports.values())
        toolset_info_map = self._extract_toolset_docs_map(imported_tools)
        available_tools = json.dumps(toolset_info_map, indent=2, ensure_ascii=False)

        # Actions
        action_semantics_map = self._extract_actions_semantics(script, required_scripts)
        available_actions = json.dumps(action_semantics_map, indent=2, ensure_ascii=False)

        # Frames
        frames_nxs = ""
        content = script.read_as_list()

        for frame in script.frames:
            frames_nxs += self._read_node_nxs(script_content_list=content, node=frame, read_as_list=False)
            frames_nxs += "\n"

        ## Aggregate
        task += CODING_ADDITIONAL_INFO.format(tools=str(available_tools), actions=str(available_actions),
                                              frames=frames_nxs, ENV_vars=self.external_vars_names)
        task += user_req_prompt
        messages = [{"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": task}]}]

        return messages

    def _build_deliberate_coding_prompt(self, script_content: list[str], deliberate: Deliberate,
                                        coding_level: CodeOperationEnum,
                                        user_request: str, script: Script, required_scripts: list[Script]):
        system_prompt = CODING_SYSTEM_PROMPT

        try:
            breakdown = deliberate.get_annotation_value("breakdown")
            breakdown = breakdown.value
        except NemantixException:
            breakdown = None

        logger.info("Breakdown: " + str(breakdown))

        # rules on the basis of expected coding output
        rules = ""
        # rules on the basis of expected coding output
        if coding_level == CodeOperationEnum.DRAFT:
            rules = DRAFT_DELIBERATE_RULES if not breakdown else DRAFT_DELIBERATE_BREAKDOWN_RULES
        elif coding_level == CodeOperationEnum.COMPLETE:
            rules = COMPLETE_DELIBERATE_RULES if not breakdown else COMPLETE_DELIBERATE_BREAKDOWN_RULES
        elif coding_level == CodeOperationEnum.EVALUATE:
            rules = EVALUATE_DELIBERATE_RULES if not breakdown else EVALUATE_DELIBERATE_BREAKDOWN_RULES

        # add user request if present (runtime)
        user_req_prompt = "" if user_request is None else USER_REQUEST.format(user_request=user_request)

        orig_code = self._read_node_nxs(script_content_list=script_content, node=deliberate, read_as_list=True)

        # build prompt
        task = COMPILATION_DELIBERATE.format(rules=rules, deliberate_nxs="\n".join(orig_code)) if not breakdown \
            else COMPILATION_DELIBERATE_BREAKDOWN.format(rules=rules, deliberate_nxs="\n".join(orig_code))

        ## Add imported tools and actions ##
        # Tools
        imported_tools = list(script.toolset_imports.values())
        toolset_info_map = self._extract_toolset_docs_map(imported_tools)
        available_tools = json.dumps(toolset_info_map, indent=2, ensure_ascii=False)
        # Actions
        action_semantics_map = self._extract_actions_semantics(script, required_scripts)
        available_actions = json.dumps(action_semantics_map, indent=2, ensure_ascii=False)

        # Frames
        frames_nxs = ""
        content = script.read_as_list()

        for frame in script.frames:
            frames_nxs += self._read_node_nxs(script_content_list=content, node=frame, read_as_list=False)
            frames_nxs += "\n"

        # deliberates
        available_deliberates_map = self.get_deliberate_semantics(deliberate)
        available_deliberates = json.dumps(available_deliberates_map, indent=2, ensure_ascii=False)

        ## Aggregate
        task += CODING_DELIBERATE_ADDITIONAL_INFO.format(tools=str(available_tools), actions=str(available_actions),
                                                         frames=frames_nxs, deliberates=available_deliberates,
                                                         knowledge_base=self.query_knowledge_base(deliberate),
                                                         ENV_vars=self.external_vars_names)
        task += user_req_prompt
        messages = [{"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": task}]}]

        return messages

    def query_knowledge_base(self, deliberate: Deliberate) -> str:
        from nemantix.core.runtime import Builtin
        from nemantix.knowledge_base.core.nemantix_knowledge_base import (
            NemantixKnowledgeBase,
        )

        if not isinstance(self.knowledge_base, NemantixKnowledgeBase):
            return ''

        guidelines = deliberate.guidelines.prompt
        kb_prompt = ('You have access to a knowledge base. You must determine a suitable query'
                     ' to retrieve chunks that should help implement the following guidelines:\n'
                     f'[guidelines]\n{guidelines}\nOnly output the query, without any commentary,'
                     f'next steps, opinions, suggestions, etc.')

        response = self.llm_proxy.invoke(kb_prompt)
        self._emit_llm(scope='coder-knowledge_base', usage=response.usage)
        query = response.text

        chunks = Builtin.retrieve(self.knowledge_base, query)
        if len(chunks) == 0:
            return ''

        content = '\n'.join([chunk.content for chunk in chunks])
        prompt_ = (f'Given the query "{query}" and the retrieved content "{content}",'
                   'reformulate it such that to answer the query. if the content is not relevant'
                   ' output an empty string. Do not add any commentary, suggestions, next steps,'
                   'opinions, etc.')

        response = self.llm_proxy.invoke(prompt_)
        self._emit_llm(scope='coder-knowledge_base', usage=response.usage)
        return response.text

    @staticmethod
    def get_deliberate_semantics(deliberate: Deliberate):
        semantics = {"deliberate_name": deliberate.name, "when": deliberate.when.prompt,
                     "guidelines": deliberate.guidelines.prompt if deliberate.guidelines else None}

        return semantics

    @staticmethod
    def replace_nxs_code_block(code: list[str], block_start_line: int, block_end_line: int, new_code_block: str,
                               indent=True) -> str:
        if indent:
            indented = textwrap.indent(new_code_block, " " * 4)
        else:
            indented = new_code_block
        new_code = ("\n".join(code[0:block_start_line]) + "\n" +
                    indented + "\n" + "\n".join(code[block_end_line + 1:len(code) + 1]))

        return new_code

    def _check_and_fix_generated_code(self, messages, result, block_start_line: int, block_end_line: int,
                                      original_code: list[str], retry_count=6, scope: str | None = None) -> tuple:
        for attempt in range(retry_count):
            messages.append({"role": "assistant", "content": [{"type": "output_text", "text": result}]})
            try:
                replaced = self.replace_nxs_code_block(original_code, block_start_line, block_end_line, result,
                                                       indent=True if block_start_line != 0 else False)
                logger.debug("---- Replaced: ----\n" + str(replaced) + "\n------------------\n")

                temp_scr = Script("_temp.nxs", None, content=replaced)
                temp_scr.parse()
                self._check_deliberate_plan_existence(temp_scr)

            except (LarkError, NemantixException, SyntaxError) as e:
                exc_str = f"{type(e).__name__}: {e}"
                logger.warning(f"Checking produced wrong syntax: {exc_str}")

                # create prompt for fixing
                fix_prompt = FIX_GENERATION + exc_str
                messages.append({"role": "user", "content": [{"type": "input_text", "text": fix_prompt}]})
                retry_response = self.llm_proxy.invoke_grammar_based(messages)
                self._emit_llm(scope=scope or 'coder', usage=retry_response.usage)
                result = retry_response.text
                logger.debug("---- Generated: ----\n" + str(result) + "\n------------------\n")
            else:
                logger.info("Code ok.")
                return replaced, attempt + 1

        self._emit_coding_error(error='Could not generate parsable code', code=original_code,
                                scope=scope, lines=(block_start_line, block_end_line))
        raise NemantixException(f"Could not generate parsable code after {retry_count} attempts.")

    @staticmethod
    def _check_deliberate_plan_existence(temp_scr: Script):
        for deliberate in temp_scr.deliberates.values():
            if not deliberate.get_plan():
                raise NemantixParserException(f"Deliberate '{deliberate.name}' does not have a plan block."
                                              "All deliberates must have a plan block.")

    @staticmethod
    def _extract_toolset_docs_map(import_stmt_list: list[ImportToolsetStatement]):
        toolset_map = {}
        toolset_classes = Toolset.get_registered_classes()

        if len(toolset_classes) == 0:
            return toolset_map

        for import_stmt in import_stmt_list:
            toolset = import_stmt.name
            tools = import_stmt.elements

            # get class to use class methods to extract docstring
            try:
                cls: Toolset = [cls for cls in toolset_classes if cls.__name__ == toolset][0]
            except IndexError:
                raise NemantixException(f"Trying to import a non-available toolset '{toolset}' in nxs. Please, "
                                        f"provide the Toolset class to the Logic.")
            tools_info = cls.get_tool_descriptions()

            # filter only imported tools from toolset (*=import all)
            if (isinstance(tools, str) and tools != "*") or isinstance(tools, list):
                name = import_stmt.get_aliased_name()
                toolset_map[name] = {k: v for k, v in tools_info.items() if k in tools}
                # no match found
                if len(toolset_map[name].keys()) == 0:
                    raise NemantixException(f"No tool named {tools} in toolset {toolset}")
            else:
                name = import_stmt.get_aliased_name()
                toolset_map[name] = tools_info

        return toolset_map

    @staticmethod
    def _extract_actions_semantics(script: Script, required_scripts: list[Script]):
        action_semantics_map = script.action_semantics_map

        # build map of all deliberates # TODO handle same name deliberates (?)
        for req_scr in required_scripts:
            action_semantics_map.update(req_scr.action_semantics_map)

        action_semantics_map = {k: v.to_dict() for k, v in action_semantics_map.items()}

        return action_semantics_map

    @staticmethod
    def _emit_coding_start(script: Script | None, scope: str, kind: str):
        event_hub = context.event_hub.get()
        if (event_hub is None or
                not event_hub.has_subscribers(event_type=EventType.CODING_START)):
            return

        event = Event(type=EventType.CODING_START, lines=(0, 0), scope=str(scope),
                      script=script, statement='', payload=dict(type=str(kind)))
        event_hub.emit(event)

    def _emit_coding_end(self, script: Script | None, scope: str, kind: str, attempts=1,
                         request: str = None):
        event_hub = context.event_hub.get()
        if (event_hub is None or
                not event_hub.has_subscribers(event_type=EventType.CODING_END)):
            return
        model = self.llm_proxy.model_name if hasattr(self.llm_proxy,
                                                     "model_name") else self.llm_proxy.__class__.__name__
        event = Event(type=EventType.CODING_END, lines=(0, 0), scope=str(scope),
                      script=script, statement='',
                      payload=dict(type=str(kind), attempts=int(attempts), request=request, model=model))
        event_hub.emit(event)

    @staticmethod
    def _emit_coding_error(error: str, code: str | list[str], lines: tuple[int, int],
                           scope: str | None = None) -> None:
        hub = context.event_hub.get()
        if hub is None or not hub.has_subscribers(EventType.CODING_ERROR):
            return

        if isinstance(code, list):
            code = '\n'.join(code)

        event = Event(type=EventType.CODING_ERROR, lines=lines,
                      scope=scope or 'coder', script=None, statement='',
                      payload=dict(error=error, code=code))
        hub.emit(event)

    def _emit_llm(self, scope: str, usage: LLMUsage) -> None:
        event_hub = context.event_hub.get()
        if (event_hub is None or
                not event_hub.has_subscribers(event_type=EventType.LLM)):
            return

        event = Event(type=EventType.LLM, lines=(0, 0), scope=str(scope),
                      script=None, statement='',
                      payload=dict(usage=usage, name=self.llm_proxy.get_name(),
                                   internal_usage=True))
        event_hub.emit(event)
