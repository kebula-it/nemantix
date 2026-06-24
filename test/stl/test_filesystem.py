import os
import shutil
from pathlib import Path

import pytest

from nemantix.core import Toolset
from nemantix.stl.local_filesystem.base import LocalFileSystemToolset

# Define a real directory relative to this test file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "test_sandbox")


@pytest.fixture(scope="session", autouse=True)
def setup_real_dir():
    """Ensures the real root sandbox directory exists before any tests run."""
    os.makedirs(TEST_DIR, exist_ok=True)
    yield
    shutil.rmtree(TEST_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_sandbox():
    """Wipes the real directory clean before EVERY test to ensure isolation."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)
    yield


class TestFileSystemToolkit:
    # --- Initialization & Security ---

    def test_init_invalid_dir(self):
        """Test that initializing with a non-existent directory raises an error."""
        with pytest.raises(FileNotFoundError):
            LocalFileSystemToolset(root_dir="/path/that/does/not/exist")

    def test_sandbox_enforcement(self):
        """Test that accessing files outside the root is blocked."""
        ts = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.read_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )

        # Attempt to access a file outside the sandbox using '..'
        result = ts("../some_sensitive_file")
        assert "Access denied" in result
        assert "outside the sandbox" in result

    # --- Write & Read Operations ---

    def test_write_and_read_file(self):
        """Test writing a file and reading it back."""
        filename = "test_doc.txt"
        content = "Hello, World!"

        ts_write = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.write_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )
        ts_read = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.read_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )

        # 1. Write the file
        write_result = ts_write(filename, content)
        assert "Successfully wrote" in write_result

        # Verify physical file existence using pathlib
        assert (Path(TEST_DIR) / filename).exists()
        assert (Path(TEST_DIR) / filename).read_text(encoding="utf-8") == content

        # 2. Read the file using the tool
        read_result = ts_read(filename)
        assert read_result == content

    def test_write_creates_subdirectories(self):
        """Test that write_file automatically creates missing parent directories."""
        filepath = "deeply/nested/folder/note.txt"
        ts_write = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.write_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )

        ts_write(filepath, "content")

        assert (Path(TEST_DIR) / "deeply/nested/folder/note.txt").exists()

    def test_read_non_existent_file(self):
        """Test reading a file that does not exist."""
        ts_read = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.read_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )

        result = ts_read("ghost.txt")
        assert "Error: File 'ghost.txt' not found" in result

    # --- List & Info Operations ---

    def test_list_files(self):
        """Test listing files in a directory."""
        ts_write = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.write_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )
        ts_create_dir = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.create_directory",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )
        ts_list = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.list_files",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )

        ts_write("a.txt", "A")
        ts_create_dir("subdir")

        result = ts_list(".")
        assert "a.txt" in result
        assert "subdir" in result
        assert "[DIR]" in result

    def test_get_file_info(self):
        """Test retrieving file metadata."""
        content = "12345"  # 5 bytes
        ts_write = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.write_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )
        ts_info = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.get_file_info",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )

        ts_write("data.bin", content)

        result = ts_info("data.bin")
        assert "Size: 5 bytes" in result

    # --- Move, Replace, Delete ---

    def test_move_file(self):
        """Test moving/renaming a file."""
        ts_write = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.write_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )
        ts_move = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.move_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )

        ts_write("old.txt", "data")

        result = ts_move("old.txt", "new.txt")
        assert "Successfully moved" in result

        assert not (Path(TEST_DIR) / "old.txt").exists()
        assert (Path(TEST_DIR) / "new.txt").exists()

    def test_replace_file(self):
        """Test atomic replacement of a file."""
        ts_write = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.write_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )
        ts_replace = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.replace_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )

        ts_write("target.json", "Old Config")
        ts_write("new_config.tmp", "New Config")

        result = ts_replace("new_config.tmp", "target.json")
        assert "Successfully replaced" in result

        # Verify target has new content and temp file is gone
        assert (Path(TEST_DIR) / "target.json").read_text(
            encoding="utf-8"
        ) == "New Config"
        assert not (Path(TEST_DIR) / "new_config.tmp").exists()

    def test_delete_file(self):
        """Test deleting a specific file."""
        ts_write = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.write_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )
        ts_delete = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.delete_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )

        ts_write("trash.txt", "junk")

        result = ts_delete("trash.txt")
        assert "Successfully deleted" in result
        assert not (Path(TEST_DIR) / "trash.txt").exists()

    def test_delete_directory_recursive(self):
        """Test recursive deletion of a directory."""
        ts_write = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.write_file",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )
        ts_delete_dir = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.delete_directory",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )

        ts_write("folder/subfile.txt", "content")

        result = ts_delete_dir("folder")
        assert "Successfully deleted" in result
        assert not (Path(TEST_DIR) / "folder").exists()

    def test_prevent_root_deletion(self):
        """Test that the toolkit prevents deleting the root sandbox itself."""
        ts_delete_dir = Toolset.get_tool(
            tool_name="LocalFileSystemToolset.delete_directory",
            instance_alias="LocalFileSystemToolset",
            instance_args=(TEST_DIR,),
        )

        # "." resolves to the root directory
        result = ts_delete_dir(".")
        assert "Error: Cannot delete the root sandbox directory" in result
