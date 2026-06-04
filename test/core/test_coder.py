from __future__ import annotations

from dataclasses import dataclass

import pytest
from lark import LarkError

from nemantix.core import coder as coder_module
from nemantix.core.coder import CodeOperationEnum, Coder, qualifier_coding_map
from nemantix.core.exceptions import NemantixException
from nemantix.core.node import (
    ActionBlock,
    BlockStatement,
    CallableTypeEnum,
    Deliberate,
    DoStatement,
    FileMeta,
    Frame,
    MicroPrompt,
    PlanQualifierEnum,
)
from nemantix.core.script import Script
from nemantix.core.tools import Toolset
from nemantix.hub.event_hub import EventHub
from nemantix.hub.events import Event, EventType
from nemantix.llm.abstract_proxy import LLMResponse, LLMUsage

nemantix = pytest.importorskip("nemantix")


class MyToolset(Toolset):
    @classmethod
    def get_tool_descriptions(cls):
        return {
            "fake_tool": "prints ciao",
            "fake_tool2": "prints arrivederci",
        }


@dataclass
class ImportToolsetStmt:
    name: str
    elements: object
    alias: str | None = None

    def get_aliased_name(self):
        return f"{self.name}:{self.alias}" if self.alias else f"{self.name}"



@dataclass
class ActionInfo:
    semantics: str
    ins: list[str]
    outs: list[str]

    def to_dict(self):
        return {
            "semantics": self.semantics,
            "ins": [str(i) for i in self.ins],
            "outs": [str(o) for o in self.outs],
        }


class FakeLLMProxy:
    def __init__(self, responses: list[str], raw_responses: list[str] | None = None,
                 usage: LLMUsage | None = None):
        self._responses = list(responses)
        self._raw_responses = list(raw_responses or [])
        self.calls: list[list[dict]] = []
        self.raw_calls: list[object] = []
        self._usage = usage or LLMUsage(input_tokens=0, output_tokens=0)

    def get_name(self) -> str:
        return 'Fake-LLM'

    def invoke_grammar_based(self, messages):
        self.calls.append(messages)
        if not self._responses:
            raise RuntimeError("No more fake responses configured")
        return LLMResponse(text=self._responses.pop(0), tool_calls=[], usage=self._usage)

    def invoke(self, prompt):
        self.raw_calls.append(prompt)
        if not self._raw_responses:
            raise RuntimeError("No more fake raw responses configured")
        return LLMResponse(text=self._raw_responses.pop(0), tool_calls=[], usage=self._usage)


@dataclass
class DummyFileMeta(FileMeta):
    line: tuple[int, int]


class DummyStatement:
    def __init__(self, line_start: int, line_end: int):
        self.meta = {"file_meta": DummyFileMeta((line_start, line_end))}


class DummyAction(DummyStatement):
    def __init__(self, name: str, line_start: int, line_end: int, qualifier=None, children=None):
        super().__init__(line_start, line_end)
        self.name = name
        self.qualifier = qualifier
        self.children = children or []


class DummyWhen:
    def __init__(self, prompt: str):
        self.prompt = prompt


class DummyGuidelines:
    def __init__(self, prompt: str):
        self.prompt = prompt


class DummyDeliberate(DummyStatement):
    def __init__(self, name: str, line_start: int, line_end: int, qualifier=None):
        super().__init__(line_start, line_end)
        self.name = name
        self.qualifier = qualifier
        self.when = DummyWhen("when text")
        self.guidelines = DummyGuidelines("guidelines text")
        self.generated_actions = None
        self._annotations = {}

    def get_plan(self):
        return None

    def get_annotation_value(self, name):
        return self._annotations.get(name)


def make_do_statement(name: str, line_start: int, line_end: int):
    node = DoStatement.__new__(DoStatement)
    node.name = name
    node.meta = {"file_meta": FileMeta((line_start, line_end), (line_start+1, line_end+1)), "node_meta":None}
    return node


def make_block_statement(children):
    node = BlockStatement.__new__(BlockStatement)
    node.children = children
    return node


# ============================================================
# Tests
# ============================================================


def test_qualifier_coding_map_matches_new_coder_rules():
    assert qualifier_coding_map["none->undefined"] is CodeOperationEnum.SKIP
    assert qualifier_coding_map["none->drafted"] is CodeOperationEnum.DRAFT
    assert qualifier_coding_map["none->frozen"] is CodeOperationEnum.COMPLETE
    assert qualifier_coding_map["drafted->none"] is CodeOperationEnum.EVALUATE
    assert qualifier_coding_map["frozen->frozen"] is CodeOperationEnum.SKIP


def test_extract_toolset_docs_map_import_all(monkeypatch):
    monkeypatch.setattr(Toolset, "get_registered_classes", classmethod(lambda cls: [MyToolset]))

    stmt = ImportToolsetStmt(name="MyToolset", elements="*")
    out = Coder._extract_toolset_docs_map([stmt])

    assert out == {
        "MyToolset": {
            "fake_tool": "prints ciao",
            "fake_tool2": "prints arrivederci",
        }
    }


def test_extract_toolset_docs_map_import_subset_list(monkeypatch):
    monkeypatch.setattr(Toolset, "get_registered_classes", classmethod(lambda cls: [MyToolset]))

    stmt = ImportToolsetStmt(name="MyToolset", elements=["fake_tool2"])
    out = Coder._extract_toolset_docs_map([stmt])

    assert out == {"MyToolset": {"fake_tool2": "prints arrivederci"}}


def test_extract_toolset_docs_map_raises_if_toolset_not_available(monkeypatch):
    monkeypatch.setattr(Toolset, "get_registered_classes", classmethod(lambda cls: [MyToolset]))

    stmt = ImportToolsetStmt(name="MissingToolset", elements="*")
    with pytest.raises(NemantixException, match=r"non-available toolset 'MissingToolset'"):
        Coder._extract_toolset_docs_map([stmt])


def test_extract_actions_semantics_merges_required_scripts():
    coder = Coder(llm_proxy=FakeLLMProxy(responses=["unused"]))

    script = type("S", (), {})()
    script.action_semantics_map = {
        "A1": ActionInfo("sem1", [], []),
        "A2": ActionInfo("sem2", ["i"], ["o"]),
    }

    required_script = type("RS", (), {})()
    required_script.action_semantics_map = {
        "B1": ActionInfo("bsem", [], []),
    }

    out = coder._extract_actions_semantics(script=script, required_scripts=[required_script])

    assert out == {
        "A1": {"semantics": "sem1", "ins": [], "outs": []},
        "A2": {"semantics": "sem2", "ins": ["i"], "outs": ["o"]},
        "B1": {"semantics": "bsem", "ins": [], "outs": []},
    }


def test_extract_do_str_collects_nested_matching_toolset_calls():
    coder = Coder(llm_proxy=FakeLLMProxy(responses=["unused"]))
    script = type("S", (), {})()
    script.content = "do myts.first\ndo otherts.skip\ndo myts.first\ndo myts.second"
    script.read = lambda **__: script.content.split('\n')
    script.read_as_list = lambda **__: script.content.split('\n')
    script.actions = {
        "A": DummyAction(
            "A",
            1,
            1,
            children=[
                make_do_statement("myts.first", 1, 1),
                make_block_statement(children=[make_do_statement("otherts.skip", 2, 2)]),
            ],
        )
    }
    deliberate = DummyDeliberate("D", 1, 1)
    deliberate.get_plan = lambda: make_block_statement(
        children=[
            make_do_statement("myts.first", 3, 3),
            make_do_statement("myts.second", 4, 4),
        ]
    )
    deliberate.generated_actions = None
    script.deliberates = {"D": deliberate}

    out = coder._extract_do_str({"myts"}, script)

    assert out == "do myts.first\ndo myts.second"


def test_replace_nxs_code_block_respects_indent_flag():
    code = ["a", "b", "c"]

    out_no_indent = Coder.replace_nxs_code_block(code, 1, 1, "X\nY", indent=False)
    out_indent = Coder.replace_nxs_code_block(code, 1, 1, "X\nY", indent=True)

    assert out_no_indent == "a\nX\nY\nc"
    assert out_indent == "a\n    X\n    Y\nc"


def test_check_and_fix_generated_code_success_first_try(monkeypatch):
    llm = FakeLLMProxy(responses=["unused"])
    coder = Coder(llm_proxy=llm)

    monkeypatch.setattr(Script, "parse", lambda self: None)

    original = ["A", "B", "C", "D"]
    out, _ = coder._check_and_fix_generated_code(messages=[], result="X\nY", block_start_line=1, block_end_line=2,
                                              original_code=original)

    assert out == "A\n    X\n    Y\nD"
    assert llm.calls == []


def test_check_and_fix_generated_code_retries_then_succeeds(monkeypatch):
    llm = FakeLLMProxy(responses=["fixed"])
    coder = Coder(llm_proxy=llm)

    calls = {"n": 0}

    def parse_side_effect(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LarkError("bad syntax")
        return None

    monkeypatch.setattr(Script, "parse", parse_side_effect)

    out, _ = coder._check_and_fix_generated_code(messages=[], result="bad", block_start_line=0, block_end_line=0,
                                              original_code=["ORIG"])

    assert "fixed" in out
    assert len(llm.calls) == 1


def test_check_and_fix_generated_code_raises_after_max_attempts(monkeypatch):
    llm = FakeLLMProxy(responses=["still bad"] * 6)
    coder = Coder(llm_proxy=llm)

    monkeypatch.setattr(Script, "parse", lambda self: (_ for _ in ()).throw(LarkError("always bad")))

    with pytest.raises(NemantixException, match=r"Could not generate parsable code after 6 attempts"):
        coder._check_and_fix_generated_code(messages=[], result="bad", block_start_line=0, block_end_line=0,
                                            original_code=["ORIG"])

    assert len(llm.calls) == 6


def test_code_script_deliberates_uses_deliberate_qualifier_and_copies_non_deliberates(monkeypatch):
    llm = FakeLLMProxy(responses=["unused"])
    coder = Coder(llm_proxy=llm)

    called = []

    def fake_code_deliberate(coding_type, deliberate_name, script, required_scripts):
        called.append((coding_type, deliberate_name))
        return f"CODED::{deliberate_name}"

    monkeypatch.setattr(coder, "code_deliberate", fake_code_deliberate)

    content_lines = [
        "REQ_LINE",
        "FRAME_LINE",
        "ACTION_LINE",
        "DELIB_SKIP",
        "DELIB_CODE",
    ]

    req = DummyStatement(1, 1)
    frame = DummyStatement(2, 2)
    action = DummyAction("A", 3, 3)
    d_skip = DummyDeliberate("D_SKIP", 4, 4, qualifier=(PlanQualifierEnum.FROZEN, PlanQualifierEnum.FROZEN))
    d_code = DummyDeliberate("D_CODE", 5, 5, qualifier=None)

    script = type("ScriptObj", (), {})()
    script.content = "\n".join(content_lines)
    script.requires = [req]
    script.frames = [frame]
    script.toolsets_decl = []
    script.toolset_imports = {}
    script.actions = {"A": action}
    script.deliberates = {"D_SKIP": d_skip, "D_CODE": d_code}
    script.read_as_list = lambda *args: content_lines

    out = coder.code_script_deliberates(script=script, required_scripts=[])

    assert "REQ_LINE\n" in out
    assert "FRAME_LINE\n" in out
    assert "ACTION_LINE\n" in out
    assert "DELIB_SKIP\n" in out
    assert "CODED::D_CODE\n" in out
    assert called == [(CodeOperationEnum.EVALUATE, "D_CODE")]


def test_generate_tool_removes_markdown_fences():
    llm = FakeLLMProxy(responses=[], raw_responses=["```python\nclass X:\n    pass\n```"])
    coder = Coder(llm_proxy=llm)

    out = coder.generate_tool(
        toolset_name="MyToolset",
        imports_str="import something",
        do_str="do myts.tool",
        description="desc",
    )

    assert out == "class X:\n    pass"


# =============================================================================
# LLM usage events emitted by the coder
# =============================================================================

def _capture_events(hub: EventHub, event_type: EventType) -> list[Event]:
    captured = []
    hub.subscribe(event_type, captured.append)
    return captured


def test_generate_tool_emits_llm_event_with_usage(isolated_event_hub):
    usage = LLMUsage(input_tokens=50, output_tokens=20)
    llm = FakeLLMProxy(responses=[], raw_responses=["class X: pass"], usage=usage)

    coder = Coder(llm_proxy=llm)
    hub = isolated_event_hub

    llm_events = _capture_events(hub, EventType.LLM)
    coding_start_events = _capture_events(hub, EventType.CODING_START)
    coding_end_events = _capture_events(hub, EventType.CODING_END)

    coder.generate_tool(toolset_name="T", imports_str="", do_str="", description="")

    assert len(coding_start_events) == 1
    assert len(coding_end_events) == 1
    assert len(llm_events) == 1
    assert llm_events[0].payload["usage"].input_tokens == 50
    assert llm_events[0].payload["usage"].output_tokens == 20
    # CODING_END no longer carries usage — the Profiler attributes it via the LLM event
    assert "usage" not in coding_end_events[0].payload


def test_check_and_fix_retries_emit_one_llm_event_per_retry(monkeypatch, isolated_event_hub):
    usage = LLMUsage(input_tokens=30, output_tokens=10)
    llm = FakeLLMProxy(responses=["fixed"] * 2, usage=usage)
    coder = Coder(llm_proxy=llm)
    llm_events = _capture_events(isolated_event_hub, EventType.LLM)

    calls = {"n": 0}

    def parse_side_effect(_):
        calls["n"] += 1
        if calls["n"] == 1:
            from lark import LarkError
            raise LarkError("bad syntax")

    monkeypatch.setattr(Script, "parse", parse_side_effect)

    coder._check_and_fix_generated_code(messages=[], result="bad", block_start_line=0,
                                        block_end_line=0, original_code=["ORIG"],
                                        scope="test")

    # One retry LLM call → one LLM event with the retry's usage
    assert len(llm_events) == 1
    assert llm_events[0].payload["usage"].input_tokens == 30
    assert llm_events[0].payload["usage"].output_tokens == 10


# =============================================================================
# code_do_as_frames Tests
# =============================================================================


def test_code_do_as_frames_invalid_node():
    """Test that passing a node that is neither ActionBlock nor Deliberate returns None, None."""
    coder = Coder(llm_proxy=FakeLLMProxy(responses=[], raw_responses=[]))
    script = type("ScriptObj", (), {})()

    res_code, res_frames = coder.code_do_as_frames(script, DummyStatement(1, 2), [])

    assert res_code is None
    assert res_frames is None


def test_code_do_as_frames_no_generative_schemas():
    """Test that passing an ActionBlock with no generative DoStatements returns None, None."""
    coder = Coder(llm_proxy=FakeLLMProxy(responses=[], raw_responses=[]))
    script = type("ScriptObj", (), {})()
    script.read_as_list = lambda: []

    normal_do = make_do_statement("my_tool", 2, 2)
    normal_do.producing_schema = "ALREADY_EXISTING_FRAME"  # Not a MicroPrompt

    action = ActionBlock.__new__(ActionBlock)
    action.name = "A"
    action.meta = {"file_meta": FileMeta((1, 3), (0, 0))}
    action.children = [normal_do]

    res_code, res_frames = coder.code_do_as_frames(script, action, [])

    assert res_code is None
    assert res_frames is None


def test_code_do_as_frames_success_action_block(monkeypatch):
    """Test successful generation and replacement of a generative schema within an ActionBlock."""
    llm = FakeLLMProxy(
        responses=[],
        raw_responses=["frame NEW_FRAME:\n  slot example of type TEXT\n__frame"],
    )
    coder = Coder(llm_proxy=llm)

    monkeypatch.setattr(coder, "_extract_actions_semantics", lambda *args: {})
    monkeypatch.setattr(coder, "_extract_toolset_docs_map", lambda *args: {})

    class DummyParser:
        def parse(self, text):
            pass

    monkeypatch.setattr(coder_module, "_get_frame_parser", lambda: DummyParser())

    def mock_script_parse(self, **kwargs):
        self.frames = [Frame("NEW_FRAME", meta={"file_meta": FileMeta((1, 1), (1, 1))})]
        self.actions = {}
        self.deliberates = {}

    monkeypatch.setattr(Script, "parse", mock_script_parse)

    script = type("ScriptObj", (), {})()
    script.content = "action A:\n  do my_tool as >> make a frame <<\n__action"
    script.read_as_list = lambda: script.content.split("\n")
    script.toolset_imports = {}

    do_node = make_do_statement("my_tool", 2, 2)
    do_node.callable_type = CallableTypeEnum.TOOL
    do_node.producing_schema = MicroPrompt(
        "make a frame", meta={"file_meta": FileMeta((2, 2), (2, 2))}
    )
    do_node.to_nxs = lambda **kwargs: "do my_tool as {NEW_FRAME}"

    action = ActionBlock.__new__(ActionBlock)
    action.name = "A"
    action.meta = {"file_meta": FileMeta((1, 3), (0, 0))}
    action.children = [do_node]

    updated_code, coded_frames = coder.code_do_as_frames(script, action, [])

    assert len(llm.raw_calls) == 1
    assert "frame NEW_FRAME" in coded_frames
    assert "do my_tool as {NEW_FRAME}" in updated_code


def test_code_do_as_frames_success_deliberate(monkeypatch):
    """Test successful generative schema replacement within a Deliberate's plan."""
    llm = FakeLLMProxy(
        responses=[], raw_responses=["frame D_FRAME:\n  slot test\n__frame"]
    )
    coder = Coder(llm_proxy=llm)

    monkeypatch.setattr(coder, "_extract_actions_semantics", lambda *args: {})
    monkeypatch.setattr(coder, "_extract_toolset_docs_map", lambda *args: {})

    class DummyParser:
        def parse(self, text):
            pass

    monkeypatch.setattr(coder_module, "_get_frame_parser", lambda: DummyParser())

    def mock_script_parse(self, **kwargs):
        self.frames = [Frame("D_FRAME", meta={"file_meta": FileMeta((1, 1), (1, 1))})]
        self.actions = {}
        self.deliberates = {}

    monkeypatch.setattr(Script, "parse", mock_script_parse)

    script = type("ScriptObj", (), {})()
    script.content = "deliberate D:\n  plan:\n    do my_action as >> gen frame <<\n  __plan\n__deliberate"
    script.read_as_list = lambda: script.content.split("\n")
    script.toolset_imports = {}

    do_node = make_do_statement("my_action", 3, 3)
    do_node.callable_type = CallableTypeEnum.ACTION
    do_node.producing_schema = MicroPrompt(
        "gen frame", meta={"file_meta": FileMeta((3, 3), (3, 3))}
    )
    do_node.to_nxs = lambda **kwargs: "do my_action as {D_FRAME}"

    deliberate = Deliberate.__new__(Deliberate)
    deliberate.name = "D"
    deliberate.meta = {"file_meta": FileMeta((1, 5), (0, 0))}
    deliberate.generated_actions = []

    plan = BlockStatement.__new__(BlockStatement)
    plan.children = [do_node]
    deliberate.get_plan = lambda: plan

    updated_code, coded_frames = coder.code_do_as_frames(script, deliberate, [])

    assert len(llm.raw_calls) == 1
    assert "frame D_FRAME" in coded_frames
    assert "do my_action as {D_FRAME}" in updated_code


def test_code_do_as_frames_retries_on_parser_error(monkeypatch):
    """Test that the function retries generating the frame if a parsing error occurs."""
    llm = FakeLLMProxy(
        responses=[], raw_responses=["invalid", "frame FIXED_FRAME:\n__frame"]
    )
    coder = Coder(llm_proxy=llm)

    monkeypatch.setattr(coder, "_extract_actions_semantics", lambda *args: {})
    monkeypatch.setattr(coder, "_extract_toolset_docs_map", lambda *args: {})

    class DummyParser:
        def parse(self, text):
            if "invalid" in text:
                raise SyntaxError("bad syntax")

    monkeypatch.setattr(coder_module, "_get_frame_parser", lambda: DummyParser())

    def mock_script_parse(self, **kwargs):
        if "invalid" in self.content:
            raise SyntaxError("bad syntax")
        self.frames = [
            Frame("FIXED_FRAME", meta={"file_meta": FileMeta((1, 1), (1, 1))})
        ]
        self.actions = {}
        self.deliberates = {}

    monkeypatch.setattr(Script, "parse", mock_script_parse)

    script = type("ScriptObj", (), {})()
    script.content = "action A:\n  do my_tool as >> gen <<\n__action"
    script.read_as_list = lambda: script.content.split("\n")
    script.toolset_imports = {}

    do_node = make_do_statement("my_tool", 2, 2)
    do_node.producing_schema = MicroPrompt(
        "gen", meta={"file_meta": FileMeta((2, 2), (2, 2))}
    )
    do_node.callable_type = None
    do_node.to_nxs = lambda **kwargs: "do my_tool as {FIXED_FRAME}"

    action = ActionBlock.__new__(ActionBlock)
    action.name = "A"
    action.meta = {"file_meta": FileMeta((1, 3), (0, 0))}
    action.children = [do_node]

    updated_code, coded_frames = coder.code_do_as_frames(script, action, [])

    assert len(llm.raw_calls) == 2
    assert "frame FIXED_FRAME" in coded_frames
    assert "do my_tool as {FIXED_FRAME}" in updated_code


def test_code_do_as_frames_raises_exception_max_retries(monkeypatch):
    """Test that an exception is raised when max retries are exceeded due to continuous parser errors."""
    llm = FakeLLMProxy(responses=[], raw_responses=["invalid"] * 6)
    coder = Coder(llm_proxy=llm)

    monkeypatch.setattr(coder, "_extract_actions_semantics", lambda *args: {})
    monkeypatch.setattr(coder, "_extract_toolset_docs_map", lambda *args: {})

    class DummyParser:
        def parse(self, text):
            raise SyntaxError("bad syntax")

    monkeypatch.setattr(coder_module, "_get_frame_parser", lambda: DummyParser())

    script = type("ScriptObj", (), {})()
    script.content = "action A:\n  do my_tool as >> gen <<\n__action"
    script.read_as_list = lambda: script.content.split("\n")
    script.toolset_imports = {}

    do_node = make_do_statement("my_tool", 2, 2)
    do_node.callable_type = None
    do_node.producing_schema = MicroPrompt(
        "gen", meta={"file_meta": FileMeta((2, 2), (2, 2))}
    )

    action = ActionBlock.__new__(ActionBlock)
    action.name = "A"
    action.meta = {"file_meta": FileMeta((1, 3), (0, 0))}
    action.children = [do_node]

    # Coder falls through to Script.parse() which throws a SyntaxError on "invalid"
    with pytest.raises(SyntaxError):
        coder.code_do_as_frames(script, action, [])

    assert len(llm.raw_calls) == 6
