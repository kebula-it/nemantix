import os
import shutil
from pathlib import Path
from nemantix.core import tool, Toolset


class LocalFileSystemToolset(Toolset):
    """
    A Toolset for safe file system operations within a sandboxed directory.
    Enforces that all operations occur strictly within 'root_dir'.
    """

    def __init__(self, root_dir: str):
        """
        Initialize the toolset with a sandbox root directory.

        Args:
            root_dir (str): The absolute path to the directory where operations are allowed.
        """
        super().__init__()
        self.root_dir = Path(root_dir).resolve()

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist.")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Root path '{root_dir}' is not a directory.")

    def _get_safe_path(self, relative_path: str) -> Path:
        """
        Internal helper to resolve a path and ensure it stays inside root_dir.
        """
        # Resolve path against root
        target_path = (self.root_dir / relative_path).resolve()

        # Check if the resulting path is still inside root_dir
        if not str(target_path).startswith(str(self.root_dir)):
            raise PermissionError(
                f"Access denied: '{relative_path}' is outside the sandbox."
            )

        return target_path

    # --- Read / List Operations ---

    @tool
    def list_files(self, directory: str = ".") -> str:
        """
        List files and subdirectories in a given directory (relative to root).

        Args:
            directory (str, optional): The directory to list. Defaults to "." (root).

        Returns:
            str: A formatted list of files and directories, or an error message.

        Example call:
            list_files(
                directory="documents/reports"
            )
        """
        try:
            target_path = self._get_safe_path(directory)

            if not target_path.exists():
                return f"Error: Directory '{directory}' does not exist."
            if not target_path.is_dir():
                return f"Error: '{directory}' is a file, not a directory."

            items = os.listdir(target_path)
            if not items:
                return f"Directory '{directory}' is empty."

            output = [f"Contents of '{directory}':"]
            for item in items:
                item_path = target_path / item
                prefix = "[DIR] " if item_path.is_dir() else "[FILE]"
                output.append(f"{prefix} {item}")
            return "\n".join(output)

        except Exception as e:
            return f"Error listing files: {str(e)}"

    @tool
    def read_file(self, file_path: str) -> str:
        """
        Read the contents of a text file (UTF-8).

        Args:
            file_path (str): The path to the file to read.

        Returns:
            str: The content of the file, or an error message.

        Example call:
            read_file(
                file_path="config/settings.json"
            )
        """
        try:
            target_path = self._get_safe_path(file_path)
            if not target_path.exists() or not target_path.is_file():
                return f"Error: File '{file_path}' not found or is a directory."

            with open(target_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @tool
    def get_file_info(self, file_path: str) -> str:
        """
        Get metadata about a file (size, modification time).

        Args:
            file_path (str): The path to the file.

        Returns:
            str: Metadata string including file size.

        Example call:
            get_file_info(
                file_path="logs/error.log"
            )
        """
        try:
            target_path = self._get_safe_path(file_path)
            if not target_path.exists():
                return f"Error: Path '{file_path}' does not exist."
            stat = target_path.stat()
            return f"File: {file_path}\nSize: {stat.st_size} bytes"
        except Exception as e:
            return f"Error getting info: {str(e)}"

    # --- Write / Create Operations ---

    @tool
    def write_file(self, file_path: str, content: str) -> str:
        """
        Write content to a file. OVERWRITES existing files.
        Automatically creates missing parent directories.

        Args:
            file_path (str): The path where the file should be written.
            content (str): The text content to write.

        Returns:
            str: Success message.

        Example call:
            write_file(
                file_path="notes/todo.txt",
                content="1. Buy milk\n2. Walk dog"
            )
        """
        try:
            target_path = self._get_safe_path(file_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to '{file_path}'."
        except Exception as e:
            return f"Error writing file: {str(e)}"

    @tool
    def create_directory(self, directory_path: str) -> str:
        """
        Create a new directory.
        Creates intermediate parent directories if they don't exist.

        Args:
            directory_path (str): The path of the directory to create.

        Returns:
            str: Success message.

        Example call:
            create_directory(
                directory_path="projects/python/src"
            )
        """
        try:
            target_path = self._get_safe_path(directory_path)
            target_path.mkdir(parents=True, exist_ok=True)
            return f"Successfully created directory '{directory_path}'."
        except Exception as e:
            return f"Error creating directory: {str(e)}"

    # --- Move / Replace / Delete Operations ---

    @tool
    def move_file(self, src_path: str, dst_path: str) -> str:
        """
        Move or rename a file or directory.
        Fails if the destination already exists.

        Args:
            src_path (str): The current path of the file/directory.
            dst_path (str): The new path or name.

        Returns:
            str: Success or error message.

        Example call:
            move_file(
                src_path="temp_data.txt",
                dst_path="archive/data_2023.txt"
            )
        """
        try:
            source = self._get_safe_path(src_path)
            destination = self._get_safe_path(dst_path)

            if not source.exists():
                return f"Error: Source '{src_path}' does not exist."
            if destination.exists():
                return f"Error: Destination '{dst_path}' already exists."

            # Ensure parent of destination exists
            destination.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(source), str(destination))
            return f"Successfully moved '{src_path}' to '{dst_path}'."
        except Exception as e:
            return f"Error moving file: {str(e)}"

    @tool
    def replace_file(self, src_path: str, dst_path: str) -> str:
        """
        Atomically replace the destination file with the source file.
        Useful for safely updating a file.

        Args:
            src_path (str): The path to the new file version.
            dst_path (str): The path to the file being replaced.

        Returns:
            str: Success or error message.

        Example call:
            replace_file(
                src_path="config.tmp",
                dst_path="config.json"
            )
        """
        try:
            source = self._get_safe_path(src_path)
            destination = self._get_safe_path(dst_path)

            if not source.exists():
                return f"Error: Source '{src_path}' does not exist."

            # Ensure parent of destination exists
            destination.parent.mkdir(parents=True, exist_ok=True)

            os.replace(source, destination)
            return f"Successfully replaced '{dst_path}' with '{src_path}'."
        except Exception as e:
            return f"Error replacing file: {str(e)}"

    @tool
    def delete_file(self, file_path: str) -> str:
        """
        Permanently delete a file.

        Args:
            file_path (str): The path to the file to delete.

        Returns:
            str: Success or error message.

        Example call:
            delete_file(
                file_path="cache/temp.log"
            )
        """
        try:
            target_path = self._get_safe_path(file_path)

            if not target_path.exists():
                return f"Error: File '{file_path}' does not exist."
            if not target_path.is_file():
                return f"Error: '{file_path}' is a directory. Use delete_directory."

            os.remove(target_path)
            return f"Successfully deleted file '{file_path}'."
        except Exception as e:
            return f"Error deleting file: {str(e)}"

    @tool
    def delete_directory(self, directory_path: str) -> str:
        """
        Recursively delete a directory and all its contents.

        Args:
            directory_path (str): The path to the directory to remove.

        Returns:
            str: Success or error message.

        Example call:
            delete_directory(
                directory_path="temp_build_files"
            )
        """
        try:
            target_path = self._get_safe_path(directory_path)

            # Extra safety: Prevent deleting the root sandbox itself
            if target_path == self.root_dir:
                return "Error: Cannot delete the root sandbox directory."

            if not target_path.exists():
                return f"Error: Directory '{directory_path}' does not exist."
            if not target_path.is_dir():
                return f"Error: '{directory_path}' is a file. Use delete_file."

            shutil.rmtree(target_path)
            return f"Successfully deleted directory '{directory_path}'."
        except Exception as e:
            return f"Error deleting directory: {str(e)}"
