from collections import defaultdict, deque
from enum import Enum

from nemantix.common import context
from nemantix.common.logger import get_package_logger
from nemantix.core.coder import Coder, Judge
from nemantix.core.custom_types import PathLike
from nemantix.core.exceptions import NemantixException
from nemantix.core.node import ActionBlock, BlockStatement, FileMeta
from nemantix.core.script import Script, ScriptTypeEnum
from nemantix.core.source_manager import (
    LocalSourceManager,
    MultiSourceResolver,
    SourceManager,
)
from nemantix.core.tools import Toolset
from nemantix.hub import Event, EventHub, EventType
from nemantix.hub.event_hub import Observable
from nemantix.llm import AbstractLLMProxy, Credentials, LLMProxyFactory
from nemantix.security.verifier import BaseVerifier

logger = get_package_logger(__name__)


def _topo_order(
    imports_map: dict[str, list[str]], only_internal: bool = True
) -> list[str]:
    nodes = set(imports_map.keys())
    if not only_internal:
        # include not present as keys
        for deps in imports_map.values():
            nodes.update(deps)

    indegree = {n: 0 for n in nodes}
    # reverse graph: for each dep -> who depends on from dep
    dependents_of = defaultdict(set)

    for f, deps in imports_map.items():
        # dedup to avoid double count if list has duplicates
        for d in set(deps):
            if only_internal and d not in nodes:
                continue
            dependents_of[d].add(f)
            indegree[f] += 1

    # start from file with 0 dependencies
    q = deque(sorted([n for n in nodes if indegree[n] == 0]))
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for m in sorted(dependents_of[n]):  # sorted
            indegree[m] -= 1
            if indegree[m] == 0:
                q.append(m)

    # if not all are taken, there's a cycle (A imports B an B imports A, ecc.)
    if len(order) != len(nodes):
        remaining = sorted([n for n in nodes if indegree[n] > 0])
        raise ValueError(f"Dependencies cycle detected among: {remaining}")

    return order


class FallbackEnum(Enum):
    NONE = 0
    VOLATILE = 1
    PERSISTENT = 2


class JsonParsingMode(Enum):
    """How a JSON string operand is parsed during frame application.

    - STRICT: json.loads only; invalid JSON raises a clear runtime error.
    - LENIENT: on a decode error, an LLM is asked to repair the JSON, then reparse.
    """

    STRICT = "strict"
    LENIENT = "lenient"


class Expertise:
    """Orchestrates nxs/nxc/nxs Script"""

    FALLBACK_DELIBERATE_PREFIX = "__Fallback__"

    # Template for the fallback deliberate injected into each user script.
    # `@completion: undefined->undefined` keeps the Coder from coding it in `build()`
    # (SKIP in qualifier_coding_map), while still leaving it selectable by the LLM
    # at runtime. The body is intentionally empty — the fallback is never executed;
    # it is promoted into a real deliberate before interpretation.
    FALLBACK_DELIBERATE_TEMPLATE = (
        "\n"
        "@completion: undefined->undefined\n"
        "deliberate {name} when >>> Select this deliberate only as a last-resort"
        " fallback: it signals that no other deliberate in this script matches"
        " the user request and that a new deliberate must be synthesised from"
        " the request itself. <<<:\n"
        "    mandate:\n"
        "        >>> Serve as a placeholder that will be promoted into a new"
        " deliberate tailored to the current user request. Do not execute this"
        " block directly; it must be replaced by a concrete deliberate before"
        " interpretation. <<<\n"
        "    __mandate\n"
        "\n"
        "    plan:\n"
        "        body:\n"
        "        __body\n"
        "    __plan\n"
        "__deliberate\n"
    )

    # Template for a freshly promoted deliberate. Its plan body is empty and
    # will be filled by `Coder.code_deliberate(COMPLETE, ...)`. The `_->frozen`
    # qualifier tells the Executor's runtime loop that, once this coding pass
    # finishes, the plan must not be re-coded on subsequent invocations.
    PROMOTED_DELIBERATE_TEMPLATE = (
        "\n"
        "@completion: _->frozen\n"
        "deliberate {name} when >>> {when} <<<:\n"
        "    mandate:\n"
        "        >>> {mandate} <<<\n"
        "    __mandate\n"
        "\n"
        "    plan:\n"
        "        body:\n"
        "        __body\n"
        "    __plan\n"
        "__deliberate\n"
    )

    def __init__(
        self,
        script_list: list[Script],
        coder: Coder,
        verifier: BaseVerifier,
        observers: list | None = None,
        export_location: PathLike = None,
        export=True,
        allow_fallback_deliberate: FallbackEnum = FallbackEnum.NONE,
        experimental_enhance_coding=False,
        experimental_enable_fixer=False,
        experimental_include_action_body_in_semantics=False,
        json_parsing: JsonParsingMode | str = JsonParsingMode.STRICT,
        search_environments: list[tuple[PathLike, SourceManager]] | None = None,
    ):
        assert isinstance(verifier, BaseVerifier)

        self.json_parsing = JsonParsingMode(json_parsing)

        if search_environments is None:
            search_environments = [(".", LocalSourceManager())]

        self.resolver = MultiSourceResolver(search_environments)

        self.toolset_classes = Toolset.get_registered_classes()
        self.script_by_loc: dict[str, Script] = {}
        self.requires_map: dict[str, list[str]] = {}

        self.coder = coder
        self.coder.enable_fixer = bool(experimental_enable_fixer)
        self.coder.include_action_body_in_semantics = bool(
            experimental_include_action_body_in_semantics
        )

        self.verifier = verifier
        self.export_location = export_location
        self.external_vars_names = None
        self.do_export = export

        self.enhance_coding = bool(experimental_enhance_coding)
        self.knowledge_base = None

        if observers:
            event_hub = context.event_hub.get()
            if event_hub is None:
                logger.info("Setting event-hub context instance.")
                context.event_hub.set(EventHub())
                event_hub = context.event_hub.get()

            assert event_hub is not None
            for observer in observers:
                assert isinstance(observer, Observable)
                observer.subscribe(event_hub)

        # Tracking for per-script fallback deliberates.
        self._fallback_counter: int = 0
        self.fallback_names: set[str] = set()
        self.fallback_name_by_script_loc: dict[str, str] = {}
        self.allow_fallback_deliberate = allow_fallback_deliberate

        # read nx_ and create map {location:script} and 'requires' map:
        for script in script_list:
            if self.allow_fallback_deliberate != FallbackEnum.NONE:
                # inject per-script fallback deliberate into script content
                self._inject_fallback_into_script(script)

            script.parse()

            self.script_by_loc[script.get_location()] = script

        # build requires map
        for loc, script in self.script_by_loc.items():
            resolved_requires = []

            for require_node in script.requires:
                resolved_path = self.resolver.resolve(require_node.file_path)
                resolved_requires.append(resolved_path)

            self.requires_map[loc] = resolved_requires

        self.source_ordered_list = []  # coding order basing on imports
        self.deliberate_to_script_loc = {}  # delib_name:script_loc
        self.action_to_script_loc = {}  # action_name:script_loc
        self.private_action_to_script_loc = {}  # action_name:script_loc

    def set_external_vars_names(self, external_vars_names: list[str]) -> None:
        if isinstance(external_vars_names, dict):
            self.external_vars_names = [k for k in external_vars_names.keys()]
        elif isinstance(external_vars_names, list) or external_vars_names is None:
            self.external_vars_names = external_vars_names
        else:
            raise NemantixException(
                "external_vars_names must be the list of names of the variables"
            )

    def set_knowledge_base(self, knowledge_base):
        from nemantix.knowledge_base.core.nemantix_knowledge_base import (
            NemantixKnowledgeBase,
        )

        if self.enhance_coding and isinstance(knowledge_base, NemantixKnowledgeBase):
            logger.info(
                "[EXPERIMENTAL] Using the Knowledge-base to enhance the coding."
            )
            self.coder.knowledge_base = knowledge_base

    def build(self):
        # get coding order basing on imports
        self.source_ordered_list = _topo_order(self.requires_map)
        event_hub = context.event_hub.get()

        for source_loc in self.source_ordered_list:
            # For each source, I check what type it is:
            # - If it's nxs, then I generate the code by building the tool and action context, and then I can simply
            #   replace the content of the current script while leaving all includes and lists unchanged,
            #   and finally perform an export
            # - If it's nxc, then I skip it
            # - If it's nxv, do I need to verify it?
            script = self.script_by_loc[source_loc]
            coded = False

            if script.type == ScriptTypeEnum.NXC:
                pass  # no need to complete code in this point
            elif script.type == ScriptTypeEnum.NXS:
                # NOTE: avoid coding of fallback deliberate
                required_scripts = self.get_required_scripts(script)

                nxc_content = self.coder.coding(
                    script=script,
                    required_scripts=required_scripts,
                    external_vars_names=self.external_vars_names,
                )

                # partial replacement (only content)
                script.content = nxc_content
                script.parse()  # update AST etc
                script.type = ScriptTypeEnum.NXC  # force type
                self.script_by_loc[source_loc] = script
                coded = True
            elif script.type == ScriptTypeEnum.NXV:
                pass  # no need to complete code in this point

            # emit expertise build for all type of script (type sent in scope, also)
            if event_hub is not None:
                name = (
                    self.coder.llm_proxy.model_name
                    if hasattr(self.coder.llm_proxy, "model_name")
                    else self.coder.llm_proxy.__class__.__name__
                )
                event = Event(
                    type=EventType.EXPERTISE_BUILD,
                    lines=(0, 0),
                    scope=str(script.type.value),
                    script=script,
                    statement="",
                    payload={"model": name, "coded": coded},
                )
                event_hub.emit(event=event)

            self.update(source_ordered_list=self.source_ordered_list)

        self.export()

    def update(self, source_ordered_list: list[str] | None = None):
        """Discovers actions and deliberates for each script"""
        if source_ordered_list is None:
            self.source_ordered_list = _topo_order(self.requires_map)
        else:
            self.source_ordered_list = source_ordered_list

        for source_loc in self.source_ordered_list:
            for deliberate in self.script_by_loc[source_loc].deliberates.values():
                self.deliberate_to_script_loc[deliberate.name] = source_loc

            # map action name to source location (for execution)
            for action in self.script_by_loc[source_loc].actions.values():
                self.action_to_script_loc[action.name] = source_loc

            for key in self.script_by_loc[source_loc].private_actions.keys():
                self.private_action_to_script_loc[key] = source_loc

    def export(self):
        if self.do_export:
            for script in self.script_by_loc.values():
                # get default export location or set to chosen export location
                output_dir = (
                    self.export_location
                    if self.export_location
                    else script.source_manager.get_default_export_location()
                )

                new_filename = (
                    script.source_manager.get_file_name(script.get_location()) + ".nxc"
                )
                export_path = script.source_manager.join(output_dir, new_filename)

                # do not export fallback deliberates
                if script.get_location() in self.fallback_name_by_script_loc:
                    fallback_name = self.fallback_name_by_script_loc[
                        script.get_location()
                    ]
                    fallback_deliberate = script.deliberates[fallback_name]

                    file_meta = fallback_deliberate.meta["file_meta"]
                    assert isinstance(file_meta, FileMeta)
                    start_line, end_line = file_meta.line

                    original_content = script.content
                    content = script.read_as_list()

                    for _ in range(end_line - start_line + 1):
                        content.pop(start_line - 1)

                    content = "\n".join(content)
                    script.content = content
                    script.write(content, source_manager=None, location=export_path)

                    # restore original content
                    script.content = original_content
                    script.parse()
                else:
                    script.write(
                        script.content, source_manager=None, location=export_path
                    )
        else:
            logger.info("Export deactivated.")

    def verify(self) -> bool:
        for script in self.script_by_loc.values():
            if script.type != ScriptTypeEnum.NXV:
                continue

            if not self.verifier.verify(script):
                return False

        return True

    def get_visible_actions_names(self, script) -> list[str]:
        actions = [a.name for a in script.actions.values()]

        for required_script in self.get_required_scripts(script):
            actions.extend([a.name for a in required_script.actions.values()])

        return actions

    def is_fully_coded(self):
        for script in self.script_by_loc.values():
            if script.type == ScriptTypeEnum.NXS:
                return False

        return True

    def get_all_deliberates_semantics(self) -> list:
        deliberate_semantics = []

        for script in self.script_by_loc.values():
            for deliberate in script.deliberates.values():
                deliberate_semantics.append(
                    self.coder.get_deliberate_semantics(deliberate)
                )

        return deliberate_semantics

    def get_script_from_deliberate(self, deliberate_name: str) -> Script:
        location = self.deliberate_to_script_loc.get(deliberate_name, None)
        if location is None:
            raise NemantixException(
                f'Cannot find script location for deliberate "{deliberate_name}"!'
            )

        assert isinstance(location, str)
        script = self.script_by_loc.get(location, None)
        if script is None:
            raise NemantixException(f'No script at requested location "{location}"!"')

        return script

    def update_script_content(
        self, script_location: str, original_node: BlockStatement, new_content: str
    ):
        orig_code = self.script_by_loc[script_location].content

        if isinstance(orig_code, str):
            orig_code = orig_code.split("\n")

        assert isinstance(orig_code, list)
        file_meta = original_node.meta["file_meta"]
        assert isinstance(file_meta, FileMeta)
        start_line, end_line = file_meta.line[0] - 1, file_meta.line[1] - 1

        repl_content = self.coder.replace_nxs_code_block(
            orig_code,
            start_line,
            end_line,
            new_content,
            indent=False if start_line != 0 else False,
        )
        # rebuild script content
        self.script_by_loc[script_location].content = repl_content
        self.script_by_loc[script_location].parse()

        # refresh name → location maps (also purges stale entries)
        self._refresh_script_maps(script_location)

        # map private action name to source location (for execution)
        for action in self.script_by_loc[script_location].private_actions.values():
            self.private_action_to_script_loc[action.name] = script_location

        name = original_node.name if hasattr(original_node, "name") else ""
        node_type = "action" if isinstance(original_node, ActionBlock) else "deliberate"
        model = (
            self.coder.llm_proxy.model_name
            if hasattr(self.coder.llm_proxy, "model_name")
            else self.coder.llm_proxy.__class__.__name__
        )

        if (event_hub := context.event_hub.get()) is not None:
            event = Event(
                type=EventType.SCRIPT_UPDATE,
                lines=(0, 0),
                scope=name,
                script=self.script_by_loc[script_location],
                statement="",
                payload={"model": model, "type": node_type},
            )
            event_hub.emit(event)

        return self.script_by_loc[script_location]

    def get_required_scripts(self, script: Script) -> list[Script]:
        requirements = self.requires_map.get(script.get_location(), [])
        try:
            return [self.script_by_loc[loc] for loc in requirements]
        except KeyError as e:
            raise NemantixException(
                f"Could not find required script: {str(e)}.\n Script required by '{script.get_location()}'."
                f"\nPlease check the path in the require or check you have passed the script to the Expertise."
            )

    @classmethod
    def from_local_scripts(
        cls,
        paths: list[PathLike] | list[Script],
        verifier: BaseVerifier,
        llm: AbstractLLMProxy | None = None,
        export_location: PathLike = None,
        export=True,
        create_summary=False,
        search_paths: list[PathLike] | None = None,
        **kwargs,
    ) -> "Expertise":
        """Instantiates an Expertise assuming local source files or Script list."""
        if not all(isinstance(p, Script) for p in paths):
            scripts = [
                Script(location=path, source_manager=LocalSourceManager())
                for path in paths
            ]
        else:
            scripts = paths

        if search_paths is None:
            search_paths = [
                ".",
                export_location or LocalSourceManager().get_default_export_location(),
            ]

        observers = kwargs.pop("observers", None)
        enable_fixer = kwargs.pop("experimental_enable_fixer", False)
        allow_fallback = kwargs.pop("allow_fallback_deliberate", FallbackEnum.NONE)
        enhance_coding = kwargs.pop("experimental_enhance_coding", False) or kwargs.pop(
            "enhance_coding", False
        )
        include_body_semantics = kwargs.pop(
            "experimental_include_action_body_in_semantics", False
        )
        experimental_llm_judge = kwargs.pop("experimental_llm_judge", False)
        json_parsing = kwargs.pop("json_parsing", JsonParsingMode.STRICT)

        logger.debug(f"Allow fallback value = {allow_fallback} ")

        if llm is None:
            logger.info(
                f"Instantiating a default LLM proxy for the Coder: "
                f"vendor {kwargs.get('vendor', 'openai')}; "
                f"model {kwargs.get('model', 'gpt-5-mini')}."
            )
            llm = cls.get_default_llm(**kwargs)

        assert isinstance(llm, AbstractLLMProxy)
        judge = Judge.LLM if experimental_llm_judge else None
        coder = Coder(llm_proxy=llm, create_summary=create_summary, judge=judge)

        return Expertise(
            script_list=scripts,
            coder=coder,
            verifier=verifier,
            observers=observers,
            export_location=export_location,
            export=export,
            allow_fallback_deliberate=allow_fallback,
            experimental_enhance_coding=enhance_coding,
            experimental_enable_fixer=enable_fixer,
            experimental_include_action_body_in_semantics=include_body_semantics,
            json_parsing=json_parsing,
            search_environments=[(path, LocalSourceManager()) for path in search_paths],
        )

    @staticmethod
    def get_default_llm(
        vendor="openai",
        model="gpt-5-mini",
        temperature=1,
        **kwargs,
    ) -> AbstractLLMProxy:
        """Returns a default LLM proxy."""
        cred_manager = Credentials()
        AbstractLLMProxy.set_credentials_manager(cred_manager)

        llm = LLMProxyFactory.create_llm_proxy(
            vendor, model_name=model, temperature=float(temperature), **kwargs
        )
        return llm

    def _make_fallback_name(self) -> str:
        """Return a deliberate name that is globally unique across all scripts.

        Uses a monotonic counter so newly appended fallbacks never collide with
        previously promoted names or with fallbacks on other scripts.
        """
        while True:
            self._fallback_counter += 1
            name = f"{self.FALLBACK_DELIBERATE_PREFIX}{self._fallback_counter}"

            if name in self.fallback_names:
                continue

            if name in getattr(self, "deliberate_to_script_loc", {}):
                continue

            return name

    def _inject_fallback_into_script(self, script: Script) -> str:
        """Append a fallback deliberate block to the given script's content.

        Loads `script.content` from disk if not already loaded, appends the
        fallback template with a fresh unique name, and registers the name in
        `fallback_names` / `fallback_name_by_script_loc`. Returns the fallback name.
        """
        if not script.content:
            script.read(read_as_lines_list=False)

        content = script.content
        if isinstance(content, list):
            content = "\n".join(content)
        else:
            assert isinstance(content, str)

        fallback_name = self._make_fallback_name()
        block = self.FALLBACK_DELIBERATE_TEMPLATE.format(name=fallback_name)
        script.content = content.rstrip("\n") + "\n" + block

        loc = script.get_location()
        self.fallback_names.add(fallback_name)
        self.fallback_name_by_script_loc[loc] = fallback_name
        return fallback_name

    def is_fallback(self, deliberate_name: str) -> bool:
        return deliberate_name in self.fallback_names

    def get_fallback_name(self, script_loc: str) -> str | None:
        return self.fallback_name_by_script_loc.get(script_loc)

    def build_promoted_deliberate_block(
        self, name: str, when: str, mandate: str
    ) -> str:
        """Build the NXS text for a newly promoted deliberate (empty plan)."""
        safe_when = (when or "").replace("<<<", "<<").replace(">>>", ">>").strip()
        safe_mandate = (mandate or "").replace("<<<", "<<").replace(">>>", ">>").strip()

        return self.PROMOTED_DELIBERATE_TEMPLATE.format(
            name=name, when=safe_when, mandate=safe_mandate
        )

    def append_fallback_deliberate(self, script_loc: str) -> str:
        """Append a fresh fallback deliberate to the given script and refresh maps.

        Intended to be called AFTER a previous fallback on this script was promoted
        to a concrete deliberate, so that the script always carries exactly one
        uncoded fallback available for future requests.
        """
        script = self.script_by_loc[script_loc]
        fallback_name = self._make_fallback_name()
        block = self.FALLBACK_DELIBERATE_TEMPLATE.format(name=fallback_name)

        content = script.content
        if isinstance(content, list):
            content = "\n".join(content)
        else:
            assert isinstance(content, str)

        script.content = content.rstrip("\n") + "\n" + block

        script.parse()
        self._refresh_script_maps(script_loc)

        self.fallback_names.add(fallback_name)
        self.fallback_name_by_script_loc[script_loc] = fallback_name
        return fallback_name

    def discard_fallback_name(self, fallback_name: str) -> None:
        """Remove a fallback name from tracking maps (after it was promoted)."""
        self.fallback_names.discard(fallback_name)
        for loc, name in list(self.fallback_name_by_script_loc.items()):
            if name == fallback_name:
                del self.fallback_name_by_script_loc[loc]

        self.deliberate_to_script_loc.pop(fallback_name, None)

    def _refresh_script_maps(self, script_loc: str) -> None:
        """Rebuild the name → location maps for a script that was just reparsed."""
        script = self.script_by_loc[script_loc]
        # purge stale entries that used to belong to this script
        for name, loc in list(self.deliberate_to_script_loc.items()):
            if loc == script_loc and name not in script.deliberates:
                del self.deliberate_to_script_loc[name]
        for name, loc in list(self.action_to_script_loc.items()):
            if loc == script_loc and name not in script.actions:
                del self.action_to_script_loc[name]
        for name, loc in list(self.private_action_to_script_loc.items()):
            if loc == script_loc and name not in script.private_actions:
                del self.private_action_to_script_loc[name]
        for deliberate in script.deliberates.values():
            self.deliberate_to_script_loc[deliberate.name] = script_loc
        for action in script.actions.values():
            self.action_to_script_loc[action.name] = script_loc
        for key in script.private_actions.keys():
            self.private_action_to_script_loc[key] = script_loc
