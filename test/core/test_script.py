# test_script.py

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemantix.core.exceptions import NemantixException
from nemantix.core.script import Script, ScriptTypeEnum, extension_map

# ==========================================
# Dummy AST node types (to satisfy isinstance)
# ==========================================


class DummyRequire:
    pass


class DummyPythonToolDeclaration:
    pass


class DummyFrame:
    pass


class DummyImportToolsetStatement:
    def __init__(self, alias: str):
        self._alias = alias

    def get_aliased_name(self):
        return self._alias


class DummyPrompt:
    def __init__(self, prompt: str):
        self.prompt = prompt


class DummyGuidelines:
    def __init__(self, prompt: str):
        self.prompt = prompt


class DummyAction:
    def __init__(self, name: str, semantics: str, ins=None, outs=None):
        self.name = name
        self.prompt = DummyPrompt(semantics)
        self.input = ins or []
        self.output = outs or []
        self.children = []

    @staticmethod
    def get_annotation_value(*_: str):
        return None


class DummyPlan:
    def __init__(self, children, ins=None, outs=None):
        self.children = children
        self.input = ins or []
        self.output = outs or []


class DummyDeliberate:
    def __init__(self, name: str, actions: list[DummyAction], guidelines=None):
        self.name = name
        self._plan = DummyPlan(
            actions,
            ins=actions[0].input if actions else [],
            outs=actions[0].output if actions else [],
        )
        self.guidelines = guidelines
        self.generated_actions = {}

    def get_plan(self):
        return self._plan

    @staticmethod
    def get_annotation_value(*_: str):
        return None


# ==========================================
# Helpers
# ==========================================


@pytest.fixture
def mock_source_manager():
    m = MagicMock()
    # get_file_extension returns the bare extension (no dot), matching extension_map keys
    m.get_file_extension.side_effect = lambda p: str(p).rsplit(".", 1)[-1]
    m.location_to_str.side_effect = lambda p: str(p.as_posix())
    return m


# ==========================================
# Tests for extension/type mapping
# ==========================================


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("x.nxs", ScriptTypeEnum.NXS),
        ("x.nxc", ScriptTypeEnum.NXC),
        ("x.nxv", ScriptTypeEnum.NXV),
    ],
)
def test_script_init_valid_extensions_sets_type(
    filename, expected, mock_source_manager
):
    scr = Script(location=filename, source_manager=mock_source_manager, content="dummy")
    assert scr.type == expected


def test_script_init_valid_extensions_without_source_manager():
    scr = Script(location="x.nxs", source_manager=None, content="dummy")
    assert scr.type == ScriptTypeEnum.NXS


def test_script_init_invalid_extension_raises(mock_source_manager):
    with pytest.raises(NemantixException, match=r"must have \.nxs/\.nxc/\.nxv"):
        Script(location="x.txt", source_manager=mock_source_manager, content="dummy")


def test_extension_map_contains_expected_keys():
    assert extension_map["nxs"] == ScriptTypeEnum.NXS
    assert extension_map["nxc"] == ScriptTypeEnum.NXC
    assert extension_map["nxv"] == ScriptTypeEnum.NXV


# ==========================================
# Tests for Script.read
# ==========================================


def test_read_reads_from_source_manager_when_content_missing(mock_source_manager):
    mock_source_manager.read.return_value = ["a\n", "b\n"]
    scr = Script(location="x.nxs", source_manager=mock_source_manager, content=None)

    out = scr.read(update=False, read_as_lines_list=True)

    assert out == ["a\n", "b\n"]
    assert scr.content == ["a\n", "b\n"]
    mock_source_manager.read.assert_called_once_with(Path("x.nxs"), True)


def test_read_does_not_reread_if_content_present(mock_source_manager):
    scr = Script(
        location="x.nxs", source_manager=mock_source_manager, content="already"
    )
    out = scr.read(update=False, read_as_lines_list=False)

    assert out == "already"
    mock_source_manager.read.assert_not_called()


def test_read_update_true_forces_reread(mock_source_manager):
    scr = Script(location="x.nxs", source_manager=mock_source_manager, content="old")
    mock_source_manager.read.return_value = "new"

    out = scr.read(update=True, read_as_lines_list=False)

    assert out == "new"
    assert scr.content == "new"
    mock_source_manager.read.assert_called_once_with(Path("x.nxs"), False)


def test_read_converts_string_to_lines_when_requested(mock_source_manager):
    scr = Script(location="x.nxs", source_manager=mock_source_manager, content="a\nb")

    out = scr.read(update=False, read_as_lines_list=True)

    assert out == ["a", "b"]


def test_read_converts_lines_to_string_when_requested(mock_source_manager):
    scr = Script(
        location="x.nxs", source_manager=mock_source_manager, content=["a", "b"]
    )

    out = scr.read(update=False, read_as_lines_list=False)

    assert out == "a\nb"


# ==========================================
# Tests for Script.write
# ==========================================


def test_write_raises_if_no_existing_content_and_no_new_content(mock_source_manager):
    scr = Script(location="x.nxs", source_manager=mock_source_manager, content=None)

    with pytest.raises(NemantixException, match="No content provided to write"):
        scr.write(content=None)


def test_write_uses_existing_self_content_when_present(mock_source_manager):
    scr = Script(
        location="x.nxs", source_manager=mock_source_manager, content="SELF_CONTENT"
    )

    scr.write(content="IGNORED", location="other.nxs")

    mock_source_manager.write.assert_called_once_with(
        "other.nxs", "SELF_CONTENT", mode="w"
    )
    assert scr.content == "SELF_CONTENT"


def test_write_uses_new_content_when_self_content_missing(mock_source_manager):
    scr = Script(location="x.nxs", source_manager=mock_source_manager, content=None)

    scr.write(content="NEW_CONTENT", location="other.nxs")

    mock_source_manager.write.assert_called_once_with(
        "other.nxs", "NEW_CONTENT", mode="w"
    )
    assert scr.content is None


def test_write_overwrite_same_location_updates_content_and_triggers_parse(
    mock_source_manager,
):
    scr = Script(location="x.nxs", source_manager=mock_source_manager, content="ABC")
    scr.parse = MagicMock()

    scr.write(location="x.nxs")

    mock_source_manager.write.assert_called_once_with("x.nxs", "ABC", mode="w")
    scr.parse.assert_called_once()
    assert scr.content == "ABC"


def test_write_different_location_does_not_trigger_parse(mock_source_manager):
    scr = Script(location="x.nxs", source_manager=mock_source_manager, content="ABC")
    scr.parse = MagicMock()

    scr.write(location="y.nxs")

    mock_source_manager.write.assert_called_once_with("y.nxs", "ABC", mode="w")
    scr.parse.assert_not_called()
    assert scr.content == "ABC"


# ==========================================
# Tests for Script.parse
# ==========================================


def test_parse_reads_content_if_missing_then_calls_parser(
    mock_source_manager, monkeypatch
):
    import nemantix.core.script as script_module

    monkeypatch.setattr(script_module, "Deliberate", DummyDeliberate)
    monkeypatch.setattr(script_module, "ActionBlock", DummyAction)
    monkeypatch.setattr(script_module, "Require", DummyRequire)
    monkeypatch.setattr(
        script_module, "PythonToolDeclaration", DummyPythonToolDeclaration
    )
    monkeypatch.setattr(script_module, "Frame", DummyFrame)
    monkeypatch.setattr(
        script_module, "ImportToolsetStatement", DummyImportToolsetStatement
    )

    mock_source_manager.read.return_value = "CONTENT"

    scr = Script(location="x.nxs", source_manager=mock_source_manager, content=None)
    scr.parser.parse = MagicMock(return_value=[])

    scr.parse()

    mock_source_manager.read.assert_called_once_with(Path("x.nxs"), False)
    scr.parser.parse.assert_called_once_with(
        "CONTENT", Path("x.nxs"), verbose=False, enable_fixer=False
    )


def test_parse_builds_structures_and_semantics_map(mock_source_manager, monkeypatch):
    import nemantix.core.script as script_module

    monkeypatch.setattr(script_module, "Deliberate", DummyDeliberate)
    monkeypatch.setattr(script_module, "ActionBlock", DummyAction)
    monkeypatch.setattr(script_module, "Require", DummyRequire)
    monkeypatch.setattr(
        script_module, "PythonToolDeclaration", DummyPythonToolDeclaration
    )
    monkeypatch.setattr(script_module, "Frame", DummyFrame)
    monkeypatch.setattr(
        script_module, "ImportToolsetStatement", DummyImportToolsetStatement
    )

    actions = [
        DummyAction(name="A1", semantics="do X", ins=["i1"], outs=["o1"]),
        DummyAction(name="A2", semantics="do Y", ins=["i2"], outs=["o2"]),
    ]

    delib = DummyDeliberate(
        name="D1", actions=actions, guidelines=DummyGuidelines("follow D1")
    )
    req = DummyRequire()
    tool = DummyPythonToolDeclaration()
    frame = DummyFrame()
    toolset_import = DummyImportToolsetStatement("tool_alias")

    scr = Script(location="x.nxs", source_manager=mock_source_manager, content="RAW")
    scr.parser.parse = MagicMock(
        return_value=[*actions, delib, req, tool, frame, toolset_import]
    )

    scr.parse()

    assert "D1" in scr.deliberates
    assert scr.deliberates["D1"] is delib

    assert "A1" in scr.actions
    assert scr.actions["A1"] is actions[0]
    assert "A2" in scr.actions
    assert scr.actions["A2"] is actions[1]

    assert scr.requires == [req]
    assert scr.toolsets_decl == [tool]
    assert scr.frames == [frame]
    assert scr.toolset_imports["tool_alias"] is toolset_import

    assert "D1" in scr.delib_semantics_map
    d_info = scr.delib_semantics_map["D1"]
    assert d_info.semantics == "follow D1"
    assert d_info.ins == ["i1"]
    assert d_info.outs == ["o1"]
    assert d_info.to_dict() == {"semantics": "follow D1", "ins": ["i1"], "outs": ["o1"]}

    assert "A1" in scr.action_semantics_map
    info = scr.action_semantics_map["A1"]
    assert info.semantics == "do X"
    assert info.ins == ["i1"]
    assert info.outs == ["o1"]
    assert info.to_dict() == {"semantics": "do X", "ins": ["i1"], "outs": ["o1"]}


def test_parse_deliberate_without_guidelines_sets_none_semantics(
    mock_source_manager, monkeypatch
):
    import nemantix.core.script as script_module

    monkeypatch.setattr(script_module, "Deliberate", DummyDeliberate)
    monkeypatch.setattr(script_module, "ActionBlock", DummyAction)
    monkeypatch.setattr(script_module, "Require", DummyRequire)
    monkeypatch.setattr(
        script_module, "PythonToolDeclaration", DummyPythonToolDeclaration
    )
    monkeypatch.setattr(script_module, "Frame", DummyFrame)
    monkeypatch.setattr(
        script_module, "ImportToolsetStatement", DummyImportToolsetStatement
    )

    action = DummyAction(name="A1", semantics="do X", ins=["i1"], outs=["o1"])
    delib = DummyDeliberate(name="D1", actions=[action], guidelines=None)

    scr = Script(location="x.nxs", source_manager=mock_source_manager, content="RAW")
    scr.parser.parse = MagicMock(return_value=[action, delib])

    scr.parse()

    info = scr.delib_semantics_map["D1"]
    assert info.semantics is None
    assert info.ins == ["i1"]
    assert info.outs == ["o1"]


def test_parse_duplicate_deliberate_name_raises(mock_source_manager, monkeypatch):
    import nemantix.core.script as script_module

    monkeypatch.setattr(script_module, "Deliberate", DummyDeliberate)
    monkeypatch.setattr(script_module, "ActionBlock", DummyAction)
    monkeypatch.setattr(script_module, "Require", DummyRequire)
    monkeypatch.setattr(
        script_module, "PythonToolDeclaration", DummyPythonToolDeclaration
    )
    monkeypatch.setattr(script_module, "Frame", DummyFrame)
    monkeypatch.setattr(
        script_module, "ImportToolsetStatement", DummyImportToolsetStatement
    )

    d1 = DummyDeliberate(name="DUP", actions=[DummyAction("A", "s")])
    d2 = DummyDeliberate(name="DUP", actions=[DummyAction("B", "t")])

    scr = Script(location="x.nxs", source_manager=mock_source_manager, content="RAW")
    scr.parser.parse = MagicMock(return_value=[d1, d2])

    with pytest.raises(NemantixException, match=r"same name \(DUP\)"):
        scr.parse()


def test_parse_duplicate_action_name_raises(mock_source_manager, monkeypatch):
    import nemantix.core.script as script_module

    monkeypatch.setattr(script_module, "Deliberate", DummyDeliberate)
    monkeypatch.setattr(script_module, "ActionBlock", DummyAction)
    monkeypatch.setattr(script_module, "Require", DummyRequire)
    monkeypatch.setattr(
        script_module, "PythonToolDeclaration", DummyPythonToolDeclaration
    )
    monkeypatch.setattr(script_module, "Frame", DummyFrame)
    monkeypatch.setattr(
        script_module, "ImportToolsetStatement", DummyImportToolsetStatement
    )

    actions = [
        DummyAction(name="A", semantics="s1"),
        DummyAction(name="A", semantics="s2"),
    ]

    scr = Script(location="x.nxs", source_manager=mock_source_manager, content="RAW")
    scr.parser.parse = MagicMock(return_value=actions)

    with pytest.raises(NemantixException, match=r"same name \(A\)"):
        scr.parse()


def test_get_location_returns_string_path(mock_source_manager):
    scr = Script(
        location=Path("x.nxs"), source_manager=mock_source_manager, content="RAW"
    )
    assert scr.get_location() == "x.nxs"
