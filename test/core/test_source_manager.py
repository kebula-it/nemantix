from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from nemantix.core.exceptions import NemantixException
from nemantix.core.source_manager import LocalSourceManager, MultiSourceResolver

# ==========================================
# Tests for LocalSourceManager __init__
# ==========================================


def test_local_source_manager_init_with_default_export_path_none():
    mgr = LocalSourceManager(max_file_cache=3)

    assert mgr.max_file_cache == 3
    assert mgr.default_export_location == Path("./coding_output")


def test_local_source_manager_init_with_string_default_export_path():
    mgr = LocalSourceManager(
        max_file_cache=3,
        default_export_path="./runs/exp",
    )

    assert mgr.max_file_cache == 3
    assert mgr.default_export_location == Path("./runs/exp")


def test_local_source_manager_init_with_path_default_export_path():
    default_path = Path("./runs/exp")
    mgr = LocalSourceManager(
        max_file_cache=3,
        default_export_path=default_path,
    )

    assert mgr.max_file_cache == 3
    assert mgr.default_export_location == default_path


def test_empty_file_cache_resets_internal_cache():
    mgr = LocalSourceManager(max_file_cache=2)
    mgr._open_files_path = [Path("a.txt")]
    mgr._open_files_content = [["x\n"]]

    mgr.empty_file_cache()

    assert mgr._open_files_path == []
    assert mgr._open_files_content == []


# ==========================================
# Tests for LocalSourceManager.read
# ==========================================


def test_read_returns_lines_list_by_default(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("a\nb\n", encoding="utf-8")

    mgr = LocalSourceManager(max_file_cache=5)
    out = mgr.read(f)

    assert out == ["a\n", "b\n"]
    assert isinstance(out, list)


def test_read_returns_string_when_read_as_lines_list_false(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("a\nb\n", encoding="utf-8")

    mgr = LocalSourceManager(max_file_cache=5)
    out = mgr.read(str(f), read_as_lines_list=False)

    assert out == "a\nb\n"
    assert isinstance(out, str)


def test_read_raises_if_path_does_not_exist(tmp_path: Path):
    mgr = LocalSourceManager()
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(NemantixException, match="does not exist"):
        mgr.read(missing_file)


def test_read_uses_cache_and_does_not_reopen_file(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("hello\n", encoding="utf-8")

    mgr = LocalSourceManager(max_file_cache=5)

    original_open = Path.open

    with patch(
        "pathlib.Path.open",
        autospec=True,
        side_effect=lambda self, *a, **kw: original_open(self, *a, **kw),
    ) as mocked_open:
        first = mgr.read(f)
        second = mgr.read(f)

    assert first == ["hello\n"]
    assert second == ["hello\n"]
    assert mocked_open.call_count == 1


def test_read_cache_eviction_fifo_when_exceeding_max_file_cache(tmp_path: Path):
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f3 = tmp_path / "f3.txt"

    f1.write_text("1\n", encoding="utf-8")
    f2.write_text("2\n", encoding="utf-8")
    f3.write_text("3\n", encoding="utf-8")

    mgr = LocalSourceManager(max_file_cache=2)

    original_open = Path.open

    with patch(
        "pathlib.Path.open",
        autospec=True,
        side_effect=lambda self, *a, **kw: original_open(self, *a, **kw),
    ) as mocked_open:
        mgr.read(f1)
        mgr.read(f2)
        mgr.read(f3)
        mgr.read(f1)

    assert mocked_open.call_count == 4
    assert mgr._open_files_path == [f3, f1]


def test_read_returns_cached_content_even_if_file_changes_on_disk(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("old\n", encoding="utf-8")

    mgr = LocalSourceManager(max_file_cache=5)

    first = mgr.read(f)

    f.write_text("new\n", encoding="utf-8")

    second = mgr.read(f)

    assert first == ["old\n"]
    assert second == ["old\n"]


# ==========================================
# Tests for LocalSourceManager.write
# ==========================================


def test_write_creates_parent_directories_and_writes_content(tmp_path: Path):
    mgr = LocalSourceManager()

    out_path = tmp_path / "a" / "b" / "file.txt"
    mgr.write(out_path, "hello", mode="w")

    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == "hello"


def test_write_accepts_string_path(tmp_path: Path):
    mgr = LocalSourceManager()

    out_path = tmp_path / "file.txt"
    mgr.write(str(out_path), "hello", mode="w")

    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == "hello"


def test_write_with_list_joins_with_newlines(tmp_path: Path):
    mgr = LocalSourceManager()

    out_path = tmp_path / "file.txt"
    mgr.write(out_path, ["a", "b", "c"], mode="w")

    assert out_path.read_text(encoding="utf-8") == "a\nb\nc"


def test_write_mode_a_creates_file_if_missing(tmp_path: Path):
    mgr = LocalSourceManager()

    out_path = tmp_path / "missing.txt"

    assert not out_path.exists()

    mgr.write(out_path, "first", mode="a")

    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == "first"


def test_write_mode_a_appends_if_file_exists(tmp_path: Path):
    mgr = LocalSourceManager()

    out_path = tmp_path / "file.txt"
    out_path.write_text("base-", encoding="utf-8")

    mgr.write(out_path, "append", mode="a")

    assert out_path.read_text(encoding="utf-8") == "base-append"


def test_write_mode_w_overwrites(tmp_path: Path):
    mgr = LocalSourceManager()

    out_path = tmp_path / "file.txt"
    out_path.write_text("old", encoding="utf-8")

    mgr.write(out_path, "new", mode="w")

    assert out_path.read_text(encoding="utf-8") == "new"


def test_write_invalidates_cached_file(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("old\n", encoding="utf-8")

    mgr = LocalSourceManager(max_file_cache=5)

    assert mgr.read(f) == ["old\n"]
    assert f in mgr._open_files_path

    mgr.write(f, "new\n", mode="w")

    assert f not in mgr._open_files_path
    assert mgr.read(f) == ["new\n"]


# ==========================================
# Tests for path helpers
# ==========================================


def test_join_with_path_and_string():
    mgr = LocalSourceManager()

    result = mgr.join(Path("./runs"), "exp")

    assert result == Path("./runs") / "exp"


def test_get_file_extension_from_file_path(tmp_path: Path):
    mgr = LocalSourceManager()

    f = tmp_path / "script.nxs"
    f.write_text("content", encoding="utf-8")

    assert mgr.get_file_extension(f) == "nxs"


def test_get_file_extension_from_string_path(tmp_path: Path):
    mgr = LocalSourceManager()

    f = tmp_path / "script.nxc"
    f.write_text("content", encoding="utf-8")

    assert mgr.get_file_extension(str(f)) == "nxc"


def test_get_file_extension_raises_for_directory(tmp_path: Path):
    mgr = LocalSourceManager()

    with pytest.raises(NemantixException, match="is a directory"):
        mgr.get_file_extension(tmp_path)


def test_get_file_name_returns_stem(tmp_path: Path):
    mgr = LocalSourceManager()

    f = tmp_path / "compile_deliberate.nxs"
    f.write_text("content", encoding="utf-8")

    assert mgr.get_file_name(f) == "compile_deliberate"


def test_get_file_name_raises_for_directory(tmp_path: Path):
    mgr = LocalSourceManager()

    with pytest.raises(NemantixException, match="is a directory"):
        mgr.get_file_name(tmp_path)


def test_get_file_name_with_extension(tmp_path: Path):
    mgr = LocalSourceManager()

    f = tmp_path / "compile_deliberate.nxs"
    f.write_text("content", encoding="utf-8")

    assert mgr.get_file_name_with_extension(f) == "compile_deliberate.nxs"


def test_get_file_name_with_extension_raises_for_directory(tmp_path: Path):
    mgr = LocalSourceManager()

    with pytest.raises(NemantixException, match="is a directory"):
        mgr.get_file_name_with_extension(tmp_path)


def test_get_default_export_location():
    mgr = LocalSourceManager(default_export_path="./runs/exp")

    assert mgr.get_default_export_location() == Path("./runs/exp")


def test_get_default_export_path_alias():
    mgr = LocalSourceManager(default_export_path="./runs/exp")

    assert mgr.get_default_export_path() == Path("./runs/exp")


def test_exists_returns_true_for_existing_file(tmp_path: Path):
    mgr = LocalSourceManager()

    f = tmp_path / "file.txt"
    f.write_text("content", encoding="utf-8")

    assert mgr.exists(f) is True


def test_exists_returns_true_for_existing_directory(tmp_path: Path):
    mgr = LocalSourceManager()

    assert mgr.exists(tmp_path) is True


def test_exists_returns_false_for_missing_path(tmp_path: Path):
    mgr = LocalSourceManager()

    assert mgr.exists(tmp_path / "missing") is False


def test_get_files_in_location_returns_only_files(tmp_path: Path):
    mgr = LocalSourceManager()

    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    d1 = tmp_path / "folder"

    f1.write_text("a", encoding="utf-8")
    f2.write_text("b", encoding="utf-8")
    d1.mkdir()

    result = mgr.get_files_in_location(tmp_path)

    assert sorted(result, key=lambda p: p.name) == [f1, f2]


def test_create_location_creates_directory(tmp_path: Path):
    mgr = LocalSourceManager()

    location = tmp_path / "a" / "b"

    result = mgr.create_location(location)

    assert location.is_dir()
    assert result is not None


def test_append_to_folder_name_appends_postfix_to_existing_directory(tmp_path: Path):
    mgr = LocalSourceManager()

    folder = tmp_path / "folder"
    folder.mkdir()

    result = mgr.append_to_folder_name(folder, "_1")

    assert result == Path(str(folder) + "_1")


def test_append_to_folder_name_raises_if_location_is_not_directory(tmp_path: Path):
    mgr = LocalSourceManager()

    f = tmp_path / "file.txt"
    f.write_text("content", encoding="utf-8")

    with pytest.raises(NemantixException, match="is not a directory"):
        mgr.append_to_folder_name(f, "_1")


def test_is_dir_returns_true_for_directory(tmp_path: Path):
    mgr = LocalSourceManager()

    assert mgr.is_dir(tmp_path) is True


def test_is_dir_returns_false_for_file(tmp_path: Path):
    mgr = LocalSourceManager()

    f = tmp_path / "file.txt"
    f.write_text("content", encoding="utf-8")

    assert mgr.is_dir(f) is False


def test_location_to_str_returns_posix_string():
    mgr = LocalSourceManager()

    path = Path("runs") / "exp" / "file.txt"
    result = mgr.location_to_str(path)

    expected = path.resolve().as_posix()
    assert result == expected


def test_change_file_extension_with_dot(tmp_path: Path):
    mgr = LocalSourceManager()

    f = tmp_path / "file.nxs"

    result = mgr.change_file_extension(f, ".nxc")

    assert result == tmp_path / "file.nxc"


def test_change_file_extension_without_dot(tmp_path: Path):
    mgr = LocalSourceManager()

    f = tmp_path / "file.nxs"

    result = mgr.change_file_extension(f, "nxc")

    assert result == tmp_path / "file.nxc"


# ==========================================
# Tests for MultiSourceResolver
# ==========================================


def test_multi_source_resolver_finds_script_in_first_environment(tmp_path: Path):
    env1 = tmp_path / "env1"
    env1.mkdir()

    target_script = env1 / "script.nxs"
    target_script.touch()

    mgr = LocalSourceManager()
    resolver = MultiSourceResolver([(env1, mgr)])

    result = resolver.resolve("script.nxs")

    # It should resolve using the source manager's stringification method
    assert result == mgr.location_to_str(target_script)


def test_multi_source_resolver_falls_back_to_subsequent_environments(tmp_path: Path):
    env1 = tmp_path / "env1"
    env2 = tmp_path / "env2"
    env1.mkdir()
    env2.mkdir()

    # Put the script only in the second environment
    target_script = env2 / "fallback.nxc"
    target_script.touch()

    mgr = LocalSourceManager()
    resolver = MultiSourceResolver([(env1, mgr), (env2, mgr)])

    result = resolver.resolve("fallback.nxc")

    # It should skip env1 and find it in env2
    assert result == mgr.location_to_str(target_script)


def test_multi_source_resolver_raises_exception_if_not_found(tmp_path: Path):
    env1 = tmp_path / "env1"
    env1.mkdir()

    mgr = LocalSourceManager()
    resolver = MultiSourceResolver([(env1, mgr)])

    with pytest.raises(
        NemantixException, match="Required script 'missing.nxs' not found"
    ):
        resolver.resolve("missing.nxs")


# ==========================================
# Tests for path helpers (location_to_str normalization)
# ==========================================


def test_location_to_str_returns_resolved_posix_string(tmp_path: Path):
    mgr = LocalSourceManager()
    f = tmp_path / "runs" / "exp" / "file.txt"

    expected = f.resolve().as_posix()
    result = mgr.location_to_str(f)

    assert result == expected


def test_location_to_str_resolves_parent_directory_traversal(tmp_path: Path):
    mgr = LocalSourceManager()

    # Create a base structure: /tmp/.../base/folder
    base = tmp_path / "base"
    folder = base / "folder"

    # Construct a path that uses '../' to step back out of 'folder'
    traversal_path = folder / ".." / "script.nxs"

    # The expected resolution should just be /tmp/.../base/script.nxs
    expected = (base / "script.nxs").resolve().as_posix()
    result = mgr.location_to_str(traversal_path)

    assert result == expected


def test_location_to_str_resolves_current_directory_traversal(tmp_path: Path):
    mgr = LocalSourceManager()

    base = tmp_path / "base"

    # Construct a path that uses './'
    current_path = base / "." / "script.nxs"

    # The expected resolution should just ignore the './'
    expected = (base / "script.nxs").resolve().as_posix()
    result = mgr.location_to_str(current_path)

    assert result == expected


def test_location_to_str_consistency_across_equivalent_paths(
    tmp_path: Path, monkeypatch
):
    mgr = LocalSourceManager()

    # Move working directory to tmp_path so relative string paths work reliably
    monkeypatch.chdir(tmp_path)
    Path("subfolder").mkdir()

    # Three completely different ways to express the exact same file location
    path1 = Path("subfolder/script.nxs")
    path2 = Path("./subfolder/../subfolder/script.nxs")
    path3 = (tmp_path / "subfolder" / "script.nxs").resolve()

    str1 = mgr.location_to_str(path1)
    str2 = mgr.location_to_str(path2)
    str3 = mgr.location_to_str(path3)

    # If the normalization works, all three must produce the exact same dictionary key string
    assert str1 == str2 == str3
