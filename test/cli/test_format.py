"""Tests for `nemantix format` CLI subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from nemantix.cli.format import handle_format, register


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CLEAN_NXS = "action foo >> foo <<:\n  body:\n    >> x <<\n  __body\n__action"
_DIRTY_NXS = (
    "action foo >> foo <<:\n"
    "    body:\n"  # 4-space indent → NXF001
    "        >> x <<\n"
    "    __body\n"
    "__action"
)


def _make_subparsers() -> argparse._SubParsersAction:  # type: ignore[type-arg]
    p = argparse.ArgumentParser()
    return p.add_subparsers()


def _args(**kwargs) -> argparse.Namespace:
    defaults = {"check": False, "permissive": False, "files": []}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


def test_register_adds_format_subparser():
    subs = _make_subparsers()
    register(subs)
    p = argparse.ArgumentParser()
    subs2 = p.add_subparsers(dest="sub")
    register(subs2)
    ns = p.parse_args(["format", "--check", "a.nxs"])
    assert ns.sub == "format"
    assert ns.check is True
    assert ns.files == ["a.nxs"]


def test_register_returns_parser():
    subs = _make_subparsers()
    result = register(subs)
    assert isinstance(result, argparse.ArgumentParser)


# ---------------------------------------------------------------------------
# --check mode
# ---------------------------------------------------------------------------


def test_check_clean_file_exits_zero(tmp_path: Path):
    f = tmp_path / "clean.nxs"
    f.write_text(_CLEAN_NXS)
    args = _args(check=True, files=[str(f)])
    assert handle_format(args) == 0


def test_check_dirty_file_exits_one(tmp_path: Path):
    f = tmp_path / "dirty.nxs"
    f.write_text(_DIRTY_NXS)
    args = _args(check=True, files=[str(f)])
    assert handle_format(args) == 1


def test_check_prints_violations(tmp_path: Path, capsys: pytest.CaptureFixture):
    f = tmp_path / "dirty.nxs"
    f.write_text(_DIRTY_NXS)
    args = _args(check=True, files=[str(f)])
    handle_format(args)
    out = capsys.readouterr().out
    assert "NXF001" in out


def test_check_does_not_modify_file(tmp_path: Path):
    f = tmp_path / "dirty.nxs"
    f.write_text(_DIRTY_NXS)
    args = _args(check=True, files=[str(f)])
    handle_format(args)
    assert f.read_text() == _DIRTY_NXS


def test_check_exits_zero_when_all_clean(tmp_path: Path):
    files = []
    for i in range(3):
        f = tmp_path / f"clean{i}.nxs"
        f.write_text(_CLEAN_NXS)
        files.append(str(f))
    args = _args(check=True, files=files)
    assert handle_format(args) == 0


def test_check_exits_one_if_any_dirty(tmp_path: Path):
    clean = tmp_path / "clean.nxs"
    clean.write_text(_CLEAN_NXS)
    dirty = tmp_path / "dirty.nxs"
    dirty.write_text(_DIRTY_NXS)
    args = _args(check=True, files=[str(clean), str(dirty)])
    assert handle_format(args) == 1


# ---------------------------------------------------------------------------
# format in-place mode
# ---------------------------------------------------------------------------


def test_format_inplace_rewrites_dirty_file(tmp_path: Path):
    f = tmp_path / "dirty.nxs"
    f.write_text(_DIRTY_NXS)
    args = _args(check=False, files=[str(f)])
    rc = handle_format(args)
    assert rc == 0
    assert f.read_text() != _DIRTY_NXS
    assert "  body:" in f.read_text()


def test_format_inplace_clean_file_unchanged(tmp_path: Path):
    f = tmp_path / "clean.nxs"
    f.write_text(_CLEAN_NXS)
    args = _args(check=False, files=[str(f)])
    handle_format(args)
    assert f.read_text() == _CLEAN_NXS


# ---------------------------------------------------------------------------
# .nxv files — always check-only
# ---------------------------------------------------------------------------


def test_nxv_file_never_modified(tmp_path: Path):
    f = tmp_path / "signed.nxv"
    f.write_text(_DIRTY_NXS)
    args = _args(check=False, files=[str(f)])
    handle_format(args)
    assert f.read_text() == _DIRTY_NXS


def test_nxv_file_check_mode_reports_violations(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    f = tmp_path / "signed.nxv"
    f.write_text(_DIRTY_NXS)
    args = _args(check=False, files=[str(f)])
    rc = handle_format(args)
    out = capsys.readouterr().out
    assert "NXF001" in out
    assert rc == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_file_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture):
    args = _args(check=True, files=[str(tmp_path / "nonexistent.nxs")])
    rc = handle_format(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert (
        "not found" in err.lower()
        or "no such" in err.lower()
        or "nonexistent" in err.lower()
    )


def test_permissive_flag_preserves_bare_closers(tmp_path: Path):
    src = "action foo >> foo <<:\n  body:\n    >> x <<\n  __\n__"
    f = tmp_path / "bare.nxs"
    f.write_text(src)
    args = _args(check=False, permissive=True, files=[str(f)])
    handle_format(args)
    content = f.read_text()
    lines = content.splitlines()
    assert lines[-2] == "  __"
    assert lines[-1] == "__"


def test_syntax_error_prints_message_and_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    f = tmp_path / "broken.nxs"
    f.write_text(
        "action foo >> foo <<:\n  body:\n    >> x <<\n  __body\n# missing closer"
    )
    args = _args(check=True, files=[str(f)])
    rc = handle_format(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "broken.nxs" in err


def test_syntax_error_continues_to_next_file(tmp_path: Path):
    broken = tmp_path / "broken.nxs"
    broken.write_text(
        "action foo >> foo <<:\n  body:\n    >> x <<\n  __body\n# missing closer"
    )
    clean = tmp_path / "clean.nxs"
    clean.write_text(_CLEAN_NXS)
    args = _args(check=True, files=[str(broken), str(clean)])
    rc = handle_format(args)
    assert rc == 1  # broken causes rc=1
    assert clean.read_text() == _CLEAN_NXS  # clean file was still checked, not aborted


def test_multiple_files_reports_violations_from_each(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    f1 = tmp_path / "a.nxs"
    f2 = tmp_path / "b.nxs"
    f1.write_text(_DIRTY_NXS)
    f2.write_text(_DIRTY_NXS)
    args = _args(check=True, files=[str(f1), str(f2)])
    handle_format(args)
    out = capsys.readouterr().out
    assert str(f1) in out
    assert str(f2) in out
