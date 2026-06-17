from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemantix.core.script import ScriptTypeEnum
from nemantix.core.expertise import _topo_order, Expertise
from nemantix.core.source_manager import LocalSourceManager
from nemantix.security.verifier import DebugVerifier
from nemantix.core.exceptions import NemantixException


# ==========================================
# Tests for _topo_order
# ==========================================


def test_topo_order_simple_chain_internal_only():
    """
    a depends on b and c; b depends on c; c has no deps => order: c, b, a
    """
    imports_map = {
        "a": ["b", "c"],
        "b": ["c"],
        "c": [],
    }

    order = _topo_order(imports_map, only_internal=True)
    assert order == ["c", "b", "a"]


def test_topo_order_dedups_dependencies():
    """
    Duplicated deps should not increase indegree twice.
    """
    imports_map = {"a": ["b", "b"], "b": []}
    order = _topo_order(imports_map, only_internal=True)
    assert order == ["b", "a"]


def test_topo_order_ignores_external_deps_when_only_internal_true():
    """
    If only_internal=True, deps not present as keys are ignored.
    """
    imports_map = {"a": ["ext_lib"]}
    order = _topo_order(imports_map, only_internal=True)
    assert order == ["a"]


def test_topo_order_includes_external_nodes_when_only_internal_false():
    """
    If only_internal=False, deps not present as keys are included as nodes.
    """
    imports_map = {"a": ["ext_lib"]}
    order = _topo_order(imports_map, only_internal=False)
    assert order == ["ext_lib", "a"]


def test_topo_order_cycle_detection_raises():
    imports_map = {"a": ["b"], "b": ["a"]}
    with pytest.raises(ValueError, match=r"Dependencies cycle detected among:"):
        _topo_order(imports_map, only_internal=True)


# ==========================================
# Tests for Expertise
# ==========================================


@dataclass
class DummyRequire:
    file_path: str


def _named_obj(name: str):
    obj = MagicMock()
    obj.name = name
    return obj


def _mk_script(
    location: str,
    stype: ScriptTypeEnum,
    requires: list[str] | None = None,
    deliberates=None,
    actions=None,
):
    """
    Creates a lightweight "Script-like" object (MagicMock) with the attributes
    Expertise expects.
    """
    requires = requires or []
    deliberates = deliberates or {}
    actions = actions or {}

    s = MagicMock()
    s.location = location
    s.type = stype
    s.content = f"CONTENT::{location}"
    s.requires = [DummyRequire(r) for r in requires]
    s.deliberates = deliberates
    s.actions = actions
    s.parse = MagicMock()
    s.write = MagicMock()
    s.source_manager = LocalSourceManager()
    s.get_location = lambda: s.location
    return s


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_expertise_init_parses_scripts_and_builds_maps(mock_get_tools):
    s1 = _mk_script(
        "a.nxs",
        ScriptTypeEnum.NXS,
        requires=["b.nxs"],
        deliberates={"DA": _named_obj("DA")},
        actions={"ACT_A": _named_obj("ACT_A")},
    )
    s2 = _mk_script(
        "b.nxs",
        ScriptTypeEnum.NXS,
        requires=[],
        deliberates={"DB": _named_obj("DB")},
        actions={"ACT_B": _named_obj("ACT_B")},
    )
    coder = MagicMock()

    exp = Expertise([s1, s2], coder, verifier=DebugVerifier())

    s1.parse.assert_called_once()
    s2.parse.assert_called_once()

    assert exp.script_by_loc["a.nxs"] is s1
    assert exp.script_by_loc["b.nxs"] is s2
    assert exp.requires_map == {"a.nxs": ["b.nxs"], "b.nxs": []}
    mock_get_tools.assert_called_once()


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_expertise_build_codes_nxs_to_nxc_updates_maps_and_exports(
    mock_get_tools, tmp_path, monkeypatch
):
    """
    - Ensures topo ordering is respected
    - NXS scripts are sent to coder.coding and then converted to NXC
    - deliberate_to_script_loc / action_to_script_loc get filled
    - export writes to ./coding_output/<name>.nxc
    """
    monkeypatch.chdir(tmp_path)

    b = _mk_script(
        "b.nxs",
        ScriptTypeEnum.NXS,
        requires=[],
        deliberates={"DelibB": _named_obj("DelibB")},
        actions={"ActionB": _named_obj("ActionB")},
    )
    a = _mk_script(
        "a.nxs",
        ScriptTypeEnum.NXS,
        requires=["b.nxs"],
        deliberates={"DelibA": _named_obj("DelibA")},
        actions={"ActionA": _named_obj("ActionA")},
    )

    coder = MagicMock()
    call_log = []

    def coding_side_effect(*, script, required_scripts, external_vars_names):
        call_log.append(
            (
                script.location,
                [rs.location for rs in required_scripts],
                external_vars_names,
            )
        )
        return f"CODED::{script.location}"

    coder.coding.side_effect = coding_side_effect

    exp = Expertise([a, b], coder, verifier=DebugVerifier())
    exp.set_external_vars_names(["tenant_id", "user_id"])
    exp.build()

    assert call_log == [
        ("b.nxs", [], ["tenant_id", "user_id"]),
        ("a.nxs", ["b.nxs"], ["tenant_id", "user_id"]),
    ]

    assert exp.script_by_loc["a.nxs"].type == ScriptTypeEnum.NXC
    assert exp.script_by_loc["b.nxs"].type == ScriptTypeEnum.NXC
    assert exp.script_by_loc["a.nxs"].content == "CODED::a.nxs"
    assert exp.script_by_loc["b.nxs"].content == "CODED::b.nxs"

    assert a.parse.call_count >= 2
    assert b.parse.call_count >= 2

    assert exp.deliberate_to_script_loc["DelibA"] == "a.nxs"
    assert exp.deliberate_to_script_loc["DelibB"] == "b.nxs"
    assert exp.action_to_script_loc["ActionA"] == "a.nxs"
    assert exp.action_to_script_loc["ActionB"] == "b.nxs"

    expected_a_loc = Path("coding_output/a.nxc")
    expected_b_loc = Path("coding_output/b.nxc")

    a.write.assert_called_with(a.content, source_manager=None, location=expected_a_loc)
    b.write.assert_called_with(b.content, source_manager=None, location=expected_b_loc)


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_expertise_build_skips_nxc_scripts_and_export_flag_in_build_does_not_override_constructor(
    mock_get_tools, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    nxc = _mk_script(
        "already.nxc",
        ScriptTypeEnum.NXC,
        requires=[],
        deliberates={"D": _named_obj("D")},
        actions={"A": _named_obj("A")},
    )
    coder = MagicMock()

    exp = Expertise([nxc], coder, verifier=DebugVerifier(), export=False)
    exp.build()

    coder.coding.assert_not_called()
    assert exp.deliberate_to_script_loc["D"] == "already.nxc"
    assert exp.action_to_script_loc["A"] == "already.nxc"
    nxc.write.assert_not_called()


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_get_all_deliberates_semantics_collects_from_all_scripts(mock_get_tools):
    d1, d2, d3 = MagicMock(), MagicMock(), MagicMock()
    d1.name, d2.name, d3.name = "D1", "D2", "D3"

    s1 = _mk_script("a.nxs", ScriptTypeEnum.NXS, deliberates={"D1": d1, "D2": d2})
    s2 = _mk_script("b.nxs", ScriptTypeEnum.NXS, deliberates={"D3": d3})

    coder = MagicMock()
    coder.get_deliberate_semantics.side_effect = lambda d: f"SEM::{id(d)}"

    exp = Expertise([s1, s2], coder, verifier=DebugVerifier())
    sem = exp.get_all_deliberates_semantics()

    assert len(sem) == 3
    assert coder.get_deliberate_semantics.call_count == 3


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_set_external_vars_names_accepts_dict_list_and_none(mock_get_tools):
    exp = Expertise([], MagicMock(), verifier=DebugVerifier())

    exp.set_external_vars_names({"foo": 1, "bar": 2})
    assert exp.external_vars_names == ["foo", "bar"]

    exp.set_external_vars_names(["x", "y"])
    assert exp.external_vars_names == ["x", "y"]

    exp.set_external_vars_names(None)
    assert exp.external_vars_names is None


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_set_external_vars_names_rejects_invalid_types(mock_get_tools):
    exp = Expertise([], MagicMock(), verifier=DebugVerifier())

    with pytest.raises(NemantixException, match="external_vars_names"):
        exp.set_external_vars_names("not-a-list")


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_get_visible_actions_names_includes_required_scripts_actions(mock_get_tools):
    dep = _mk_script(
        "dep.nxc",
        ScriptTypeEnum.NXC,
        actions={"dep_action": _named_obj("dep_action")},
    )
    root = _mk_script(
        "root.nxc",
        ScriptTypeEnum.NXC,
        requires=["dep.nxc"],
        actions={"root_action": _named_obj("root_action")},
    )

    exp = Expertise([root, dep], MagicMock(), verifier=DebugVerifier())

    assert exp.get_visible_actions_names(root) == ["root_action", "dep_action"]


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_verify_only_checks_nxv_scripts(mock_get_tools):
    nxv = _mk_script("check.nxv", ScriptTypeEnum.NXV)
    nxs = _mk_script("skip.nxs", ScriptTypeEnum.NXS)

    exp = Expertise([nxv, nxs], MagicMock(), verifier=DebugVerifier())

    assert exp.verify() is True


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_get_script_from_deliberate_returns_script_and_raises_for_missing(
    mock_get_tools,
):
    script = _mk_script(
        "a.nxs", ScriptTypeEnum.NXS, deliberates={"DelibA": _named_obj("DelibA")}
    )
    exp = Expertise([script], MagicMock(), verifier=DebugVerifier())
    exp.deliberate_to_script_loc["DelibA"] = "a.nxs"

    assert exp.get_script_from_deliberate("DelibA") is script

    with pytest.raises(NemantixException, match="Cannot find script location"):
        exp.get_script_from_deliberate("Missing")


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_is_fully_coded_false_if_any_nxs_present(mock_get_tools):
    s1 = _mk_script("a.nxs", ScriptTypeEnum.NXS)
    coder = MagicMock()
    exp = Expertise([s1], coder, verifier=DebugVerifier())

    assert exp.is_fully_coded() is False


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_is_fully_coded_true_if_no_nxs_present(mock_get_tools):
    s1 = _mk_script("a.nxc", ScriptTypeEnum.NXC)
    s2 = _mk_script("b.nxv", ScriptTypeEnum.NXV)
    coder = MagicMock()
    exp = Expertise([s1, s2], coder, verifier=DebugVerifier())

    assert exp.is_fully_coded() is True
