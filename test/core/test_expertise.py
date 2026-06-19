from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemantix.core.exceptions import NemantixException
from nemantix.core.expertise import Expertise, _topo_order
from nemantix.core.script import ScriptTypeEnum
from nemantix.core.source_manager import LocalSourceManager
from nemantix.security.verifier import DebugVerifier

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

    # Initialize the source manager
    s.source_manager = LocalSourceManager()
    s.get_location = lambda: s.source_manager.location_to_str(s.location)

    return s


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_expertise_init_parses_scripts_and_builds_maps(
    mock_get_tools, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    Path("a.nxs").touch()
    Path("b.nxs").touch()

    # Calculate the expected absolute posix strings
    abs_a = Path("a.nxs").resolve().as_posix()
    abs_b = Path("b.nxs").resolve().as_posix()

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

    # Assert against the absolute paths
    assert exp.script_by_loc[abs_a] is s1
    assert exp.script_by_loc[abs_b] is s2
    assert exp.requires_map == {abs_a: [abs_b], abs_b: []}
    mock_get_tools.assert_called_once()


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_expertise_build_codes_nxs_to_nxc_updates_maps_and_exports(
    mock_get_tools, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    Path("a.nxs").touch()
    Path("b.nxs").touch()

    abs_a = Path("a.nxs").resolve().as_posix()
    abs_b = Path("b.nxs").resolve().as_posix()

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

    # Ensure the maps contain absolute paths
    assert exp.script_by_loc[abs_a].type == ScriptTypeEnum.NXC
    assert exp.script_by_loc[abs_b].type == ScriptTypeEnum.NXC
    assert exp.deliberate_to_script_loc["DelibA"] == abs_a
    assert exp.deliberate_to_script_loc["DelibB"] == abs_b
    assert exp.action_to_script_loc["ActionA"] == abs_a
    assert exp.action_to_script_loc["ActionB"] == abs_b


@patch("nemantix.core.expertise.Toolset.get_registered_classes", return_value=[])
def test_expertise_build_skips_nxc_scripts_and_export_flag_in_build_does_not_override_constructor(
    mock_get_tools, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    # It is good practice to touch the file so the resolver sees it, just in case
    Path("already.nxc").touch()

    # Calculate the expected absolute path dynamically
    abs_nxc = Path("already.nxc").resolve().as_posix()

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

    # Assert against the absolute path variable instead of the hardcoded relative string
    assert exp.deliberate_to_script_loc["D"] == abs_nxc
    assert exp.action_to_script_loc["A"] == abs_nxc


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
def test_get_visible_actions_names_includes_required_scripts_actions(
    mock_get_tools, tmp_path, monkeypatch
):
    # Setup temporary files
    monkeypatch.chdir(tmp_path)
    Path("root.nxc").touch()
    Path("dep.nxc").touch()

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
    mock_get_tools, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    Path("a.nxs").touch()
    abs_a = Path("a.nxs").resolve().as_posix()

    script = _mk_script(
        "a.nxs", ScriptTypeEnum.NXS, deliberates={"DelibA": _named_obj("DelibA")}
    )
    exp = Expertise([script], MagicMock(), verifier=DebugVerifier())

    # We must explicitly set this in the test because _mk_script bypasses self.update()
    exp.deliberate_to_script_loc["DelibA"] = abs_a

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
