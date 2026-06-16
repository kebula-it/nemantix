from enum import Enum
from pathlib import Path
from typing import Any

from nemantix.common.logger import get_package_logger
from nemantix.core.custom_types import PathLike
from nemantix.core.exceptions import NemantixException
from nemantix.core.node import (
    ActionBlock,
    BlockStatement,
    Deliberate,
    DoStatement,
    Frame,
    ImportToolsetStatement,
    PythonToolDeclaration,
    Require,
    SingleValue,
)
from nemantix.core.parser import ParserLark
from nemantix.core.source_manager import SourceManager

logger = get_package_logger(__name__)


class ScriptTypeEnum(Enum):
    NXS = "nxs"
    NXC = "nxc"
    NXV = "nxv"


extension_map = {
    "nxs": ScriptTypeEnum.NXS,
    "nxc": ScriptTypeEnum.NXC,
    "nxv": ScriptTypeEnum.NXV,
}


def _nxc_deliberate_completeness_check(deliberates: list[Deliberate], location: PathLike):
    for deliberate in deliberates:
        qual = deliberate.qualifier if deliberate.qualifier else None
        if qual and qual[1].value == "frozen" and not deliberate.get_plan():
            raise NemantixException(
                f"The deliberate '{deliberate.name}' does not have a plan and has a '{qual[1].value}' completion level."
                f"\nThe script '{location}' shouldn't be an nxc.")


class Script:
    class SemanticInfo:
        def __init__(self, semantics, ins, outs, summary=None, body=None):
            self.semantics = semantics
            self.ins = ins
            self.outs = outs
            self.body = body
            self.summary = summary

        def to_dict(self):
            d = {"semantics": self.semantics,
                "ins": [str(i) for i in self.ins],
                "outs": [str(o) for o in self.outs],
                }
            if self.summary is not None:
                d["summary"] = self.summary
            if self.body is not None:
                d["body"] = str(self.body)
            return d

    def __init__(self, location: PathLike, source_manager: SourceManager, content: str = None):
        self._location = Path(location) if not isinstance(location, Path) else location
        self.source_manager = source_manager

        if self.source_manager:
            ext = self.source_manager.get_file_extension(self._location).lower()
        else:
            ext = str(self._location).split(".")[-1].lower()
        if ext not in extension_map:
            raise NemantixException(f"Script {location} must have .nxs/.nxc/.nxv extension.")
        self.type = extension_map[ext]

        self.content = content

        self.parser = ParserLark()
        self.deliberates: dict[Any, Deliberate] = {}  # {name:Deliberate}
        self.actions: dict[Any, ActionBlock] = {}  # {name:ActionBlock}
        self.private_actions: dict[Any, ActionBlock] = {}  # {name:ActionBlock}
        self.requires = []
        self.frames = []
        self.toolsets_decl = []
        self.toolset_imports = {}  # {name:ImportStatement}
        self.delib_semantics_map = {}
        self.action_semantics_map = {}

        self._nodes = None

        if self.type == ScriptTypeEnum.NXC:
            _nxc_deliberate_completeness_check(list(self.deliberates.values()), self._location)
        # self.parse()

    def read_as_list(self, update=False) -> list[str]:
        content_list = self.read(update=bool(update), read_as_lines_list=True)
        assert isinstance(content_list, list)
        return content_list

    def read(self, update=False, read_as_lines_list=True) -> str | list[str]:
        if not self.content or update:
            self.content = self.source_manager.read(self._location, read_as_lines_list)

        if read_as_lines_list:
            if isinstance(self.content, str):
                return self.content.split("\n")
        else:
            if isinstance(self.content, list):
                return "\n".join(self.content)

        return self.content

    def write(self, content: str = None, source_manager: SourceManager = None, location: PathLike = None):
        source_manager = source_manager or self.source_manager
        over_write = False

        location = location or self._location
        if str(location) == str(self._location):
            over_write = True

        if not self.content or self.content == "":
            if not content:
                raise NemantixException("No content provided to write.")
            else:
                new_content = content
        else:
            new_content = self.content

        assert source_manager is not None
        source_manager.write(location, new_content, mode="w")

        if over_write:
            self.content = new_content
            self.parse()

    def parse(self, verbose=False, enable_fixer=False):
        # re-init
        self.toolset_imports = {}  # {name:ImportStatement}
        self.deliberates = {}  # {name:Deliberate}
        self.actions = {}  # {name:ActionBlock}
        self.private_actions = {}  # {name:ActionBlock}
        self.requires = []
        self.frames = []
        self.toolsets_decl = []
        self.delib_semantics_map = {}
        self.action_semantics_map = {}
        self._nodes = None

        if not self.content:
            self.read(read_as_lines_list=False)

        self._nodes = self.parser.parse(self.content or '', self._location,
                                        verbose=bool(verbose),
                                        enable_fixer=bool(enable_fixer))
        # build structures
        for n in self._nodes:
            if isinstance(n, Deliberate):
                self.deliberates[n.name] = n

                if n.name in self.delib_semantics_map:
                    raise NemantixException(f"Cannot instantiate two deliberates with the same name ({n.name})")

                semantics = n.guidelines.prompt if n.guidelines is not None else None
                plan = n.get_plan()

                if plan:
                    ins = plan.input
                    outs = plan.output
                else:
                    ins = None
                    outs = None

                summary = None
                try:
                    summary = n.get_annotation_value("intent.summary")
                except NemantixException:
                    pass
                summary = summary.value if isinstance(summary, SingleValue) else summary
                self.delib_semantics_map[n.name] = self.SemanticInfo(semantics, ins, outs, summary=summary)

                # add deliberate private actions with deliberate.name prefix
                for private_action in n.generated_actions:
                    key = f'{n.name}.{private_action.name}'

                    if key in self.private_actions:
                        raise NemantixException(f'Action "{private_action.name}" already defined in'
                                                f'deliberate "{n.name}"!')
                    self.private_actions[key] = private_action

            elif isinstance(n, ActionBlock):
                self.actions[n.name] = n

                if n.name in self.action_semantics_map:
                    raise NemantixException(f"Cannot instantiate two actions with the same name ({n.name})")

                semantics = n.prompt.prompt
                ins = n.input
                outs = n.output
                summary = None
                try:
                    summary = n.get_annotation_value("intent.summary")
                except NemantixException:
                    pass
                summary = summary.value if isinstance(summary, SingleValue) else summary
                self.action_semantics_map[n.name] = self.SemanticInfo(semantics, ins, outs,
                                                                      summary=summary)
            elif isinstance(n, Require):
                self.requires.append(n)

            elif isinstance(n, PythonToolDeclaration):
                self.toolsets_decl.append(n)

            elif isinstance(n, Frame):
                self.frames.append(n)

            elif isinstance(n, ImportToolsetStatement):
                name = n.get_aliased_name()
                self.toolset_imports[name] = n

    def update(self, content: str | None = None, enable_fixer=False):
        """Updates the script content:
            - if `enable_fixer=True`: the FixerTransformer is applied on the AST"""
        if content is None:
            content = self.read(read_as_lines_list=False)

        self.content = content
        self.parse(enable_fixer=enable_fixer)

        if not enable_fixer:
            return
        else:
            logger.info(f'[EXPERIMENTAL] Applying FixerTransformer on '
                        f'Script "{self.get_location()}"')

        content_lines = self.read(read_as_lines_list=True)

        # Collect all DoStatements recursively
        do_stmts = []

        def _collect(nodes):
            if not nodes:
                return

            for n in nodes:
                if isinstance(n, DoStatement):
                    do_stmts.append(n)

                if isinstance(n, BlockStatement) and hasattr(n, "children"):
                    _collect(n.children)

                # Also check generated_actions inside Deliberate
                if isinstance(n, Deliberate) and getattr(n, "generated_actions", None):
                    _collect(n.generated_actions)

        # Dig into the top-level script objects
        _collect(self.deliberates.values())
        _collect(self.actions.values())

        # Sort DoStatements from bottom to top (reverse order by line number)
        # This guarantees that string replacements don't shift the line indices of earlier nodes!
        do_stmts.sort(key=lambda n: n.meta["file_meta"].line[0], reverse=True)

        for do_node in do_stmts:
            file_meta = do_node.meta["file_meta"]

            # FileMeta is 1-indexed; convert to 0-indexed for Python arrays
            start_line = file_meta.line[0] - 1
            end_line = file_meta.line[1] - 1
            is_multiline = file_meta.line[1] - file_meta.line[0] > 0

            if is_multiline:
                end_line += 1  # to account for final "__do"

            # Extract the old code to see if the Fixer actually changed anything
            old_code = "\n".join(content_lines[start_line:end_line + 1])
            new_code_stripped = do_node.to_nxs()

            if old_code.strip() != new_code_stripped.strip():
                logger.info(f"Fixer applying change: '{old_code.strip()}' -> '{new_code_stripped}'")

                # Preserve the original indentation
                indent_spaces = len(content_lines[start_line]) - len(content_lines[start_line].lstrip())
                indented_new_code = (" " * indent_spaces) + new_code_stripped

                # Splice the new code into the content lines list
                content_lines = (content_lines[:start_line] + [indented_new_code] +
                                 content_lines[end_line + 1:])

        # Save and perform a final clean parse
        self.content = "\n".join(content_lines)
        self.parse(enable_fixer=False)

        if self.type == ScriptTypeEnum.NXC:
            _nxc_deliberate_completeness_check(list(self.deliberates.values()), self._location)

    def get_location(self) -> str:
        return self.source_manager.location_to_str(self._location)

    def get_location_with_extension(self, ext: str | ScriptTypeEnum):
        """Returns a new location with changed file extension"""
        if isinstance(ext, str):
            ext_ = ext.replace('.', '').lower()
            ext_enum = extension_map.get(ext_, None)

            if ext_enum is None:
                raise NemantixException(f"Invalid script extension '{ext}'")
            else:
                ext = ext_enum.value
        else:
            assert (isinstance(ext, ScriptTypeEnum))
            ext = ext.value

        return self.source_manager.change_file_extension(self._location, ext=f'.{ext}')

    def print_ast(self):
        logger.debug(f"--> AST of Script {str(self._location)} <--")
        self.parser.print_ast(self._nodes)

    def get_ast(self, update: bool = False):
        if not self._nodes or update:
            self.parse(enable_fixer=False)
        return self._nodes

    def __str__(self):
        return f"Script of type {self.type.value} at location:{self.get_location()}"
