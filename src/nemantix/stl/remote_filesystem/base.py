import ftplib
import io
import os
import socket
import traceback
import paramiko

from nemantix.core import tool, Toolset
from nemantix.common.logger import get_package_logger

logger = get_package_logger(__name__)


class RemoteFileSystemToolset(Toolset):
    """
    A unified toolset for interacting with remote file servers via FTP, FTPS, or SFTP.
    Supports file transfer, internal moves, and directory management with persistent connections.
    """

    def __init__(
        self,
        host: str,
        user: str,
        password: str = "",
        port: int | None = None,
        protocol: str = "ftp",
        timeout: int = 30,
    ):
        super().__init__()
        self.host = host
        self.user = user
        self.password = password
        self.protocol = protocol.lower()
        self.timeout = timeout

        # Set default ports if not provided
        if port is None:
            self.port = 22 if self.protocol == "sftp" else 21
        else:
            self.port = port

        # Internal state for persistence
        self._ftp_client = None  # For FTP and FTPS
        self._ssh_client = None  # For SFTP
        self._sftp_client = None  # For SFTP

    def _get_ftp_connection(self):
        """Internal helper to manage FTP/FTPS connections."""
        if self._ftp_client:
            try:
                self._ftp_client.voidcmd("NOOP")
                return self._ftp_client
            except (OSError, ftplib.Error, socket.error):
                try:
                    self._ftp_client.close()
                except Exception:
                    pass
                self._ftp_client = None

        logger.debug(
            f"Connecting {self.protocol.upper()} to {self.host}:{self.port}..."
        )

        if self.protocol == "ftps":
            ftp = ftplib.FTP_TLS(timeout=self.timeout)
        else:
            ftp = ftplib.FTP(timeout=self.timeout)

        ftp.connect(self.host, self.port)
        ftp.login(self.user, self.password)

        if self.protocol == "ftps":
            ftp.prot_p()

        self._ftp_client = ftp
        return ftp

    def _get_sftp_connection(self):
        """Internal helper to manage SFTP (SSH) connections."""
        if (
            self._ssh_client
            and self._ssh_client.get_transport()
            and self._ssh_client.get_transport().is_active()
        ):
            return self._sftp_client

        logger.debug(f"Connecting SFTP to {self.host}:{self.port}...")

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh.connect(
            hostname=self.host,
            port=self.port,
            username=self.user,
            password=self.password,
            timeout=self.timeout,
        )

        self._ssh_client = ssh
        self._sftp_client = ssh.open_sftp()
        return self._sftp_client

    def _reset_connection(self):
        """Force close connections so they are recreated on next retry."""
        try:
            if self._ftp_client:
                self._ftp_client.quit()
        except Exception:
            pass
        try:
            if self._sftp_client:
                self._sftp_client.close()
            if self._ssh_client:
                self._ssh_client.close()
        except Exception:
            pass
        self._ftp_client = None
        self._sftp_client = None
        self._ssh_client = None

    # --- Core File Operations ---

    @tool
    def list_files(self, directory: str = ".") -> str:
        """
        List files and directories in the specified path on the remote server.

        Args:
            directory (str): The directory path to list. Defaults to current directory "."

        Returns:
            str: A newline-separated string of file names.
        """
        try:
            if self.protocol == "sftp":
                sftp = self._get_sftp_connection()
                if directory == ".":
                    directory = sftp.normalize(".")
                files = sftp.listdir(directory)
                return (
                    "\n".join(files) if files else f"Directory '{directory}' is empty."
                )
            else:
                ftp = self._get_ftp_connection()
                ftp.cwd(directory)
                lines = []
                ftp.retrlines("LIST", lines.append)
                return (
                    "\n".join(lines) if lines else f"Directory '{directory}' is empty."
                )
        except Exception as e:
            self._reset_connection()
            return f"Error listing files in '{directory}': {str(e)}"

    @tool
    def read_file(self, file_path: str) -> str:
        """
        Download and read the content of a text file from the remote server directly into memory.

        Args:
            file_path (str): The full path to the file on the server.

        Returns:
            str: The content of the file decoded as UTF-8.
        """
        try:
            bio = io.BytesIO()
            if self.protocol == "sftp":
                sftp = self._get_sftp_connection()
                with sftp.open(file_path, "r") as remote_file:
                    return remote_file.read().decode("utf-8")
            else:
                ftp = self._get_ftp_connection()
                ftp.retrbinary(f"RETR {file_path}", bio.write)
                return bio.getvalue().decode("utf-8")
        except Exception as e:
            self._reset_connection()
            return f"Error reading file '{file_path}': {str(e)}"

    @tool
    def write_file(self, file_path: str, content: str) -> str:
        """
        Upload text content directly to a file on the remote server. Overwrites if exists.

        Args:
            file_path (str): The destination path on the server including the filename.
            content (str): The text content to write.

        Returns:
            str: Confirmation message.
        """
        try:
            if self.protocol == "sftp":
                sftp = self._get_sftp_connection()
                with sftp.open(file_path, "w") as remote_file:
                    remote_file.write(content)
                return f"Successfully wrote {len(content)} chars to '{file_path}'."
            else:
                ftp = self._get_ftp_connection()
                bio = io.BytesIO(content.encode("utf-8"))
                ftp.storbinary(f"STOR {file_path}", bio)
                return f"Successfully wrote {len(content)} chars to '{file_path}'."
        except Exception as e:
            self._reset_connection()
            return f"Error writing to '{file_path}': {str(e)}"

    # --- Transfer Operations (Local <-> Remote) ---

    @tool
    def upload_file(self, local_path: str, remote_path: str) -> str:
        """
        Upload a file from the local file system to the remote server.

        Args:
            local_path (str): The path to the file on the local machine.
            remote_path (str): The destination path on the remote server.

        Returns:
            str: Confirmation message.

        Example call:
            upload_file(
                local_path="./data/report.csv",
                remote_path="/var/www/uploads/report.csv"
            )
        """
        if not os.path.exists(local_path):
            return f"Error: Local file '{local_path}' does not exist."

        try:
            if self.protocol == "sftp":
                sftp = self._get_sftp_connection()
                sftp.put(local_path, remote_path)
            else:
                ftp = self._get_ftp_connection()
                with open(local_path, "rb") as f:
                    ftp.storbinary(f"STOR {remote_path}", f)
            return f"Successfully uploaded '{local_path}' to '{remote_path}'."
        except Exception as e:
            self._reset_connection()
            return f"Error uploading file: {str(e)}"

    @tool
    def download_file(self, remote_path: str, local_path: str) -> str:
        """
        Download a file from the remote server to the local file system.

        Args:
            remote_path (str): The path to the file on the remote server.
            local_path (str): The destination path on the local machine.

        Returns:
            str: Confirmation message.

        Example call:
            download_file(
                remote_path="/var/www/logs/error.log",
                local_path="./logs/server_error.log"
            )
        """
        try:
            if self.protocol == "sftp":
                sftp = self._get_sftp_connection()
                sftp.get(remote_path, local_path)
            else:
                ftp = self._get_ftp_connection()
                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {remote_path}", f.write)
            return f"Successfully downloaded '{remote_path}' to '{local_path}'."
        except Exception as e:
            self._reset_connection()
            return f"Error downloading file: {str(e)}"

    # --- File Management (Move, Delete, Directories) ---

    @tool
    def move_file(self, source_path: str, destination_path: str) -> str:
        """
        Move or rename a file/directory internally on the remote server.

        Args:
            source_path (str): The current path of the file/directory.
            destination_path (str): The new path or name.

        Returns:
            str: Confirmation message.

        Example call:
            move_file(
                source_path="/uploads/temp.txt",
                destination_path="/processed/final.txt"
            )
        """
        try:
            if self.protocol == "sftp":
                sftp = self._get_sftp_connection()
                sftp.rename(source_path, destination_path)
            else:
                ftp = self._get_ftp_connection()
                ftp.rename(source_path, destination_path)
            return f"Successfully moved '{source_path}' to '{destination_path}'."
        except Exception as e:
            self._reset_connection()
            return f"Error moving file: {str(e)}"

    @tool
    def create_directory(self, directory_path: str) -> str:
        """
        Create a new directory on the remote server.

        Args:
            directory_path (str): The path of the new directory to create.

        Returns:
            str: Confirmation message.

        Example call:
            create_directory(
                directory_path="/uploads/2023"
            )
        """
        try:
            if self.protocol == "sftp":
                sftp = self._get_sftp_connection()
                sftp.mkdir(directory_path)
            else:
                ftp = self._get_ftp_connection()
                ftp.mkd(directory_path)
            return f"Successfully created directory '{directory_path}'."
        except Exception as e:
            self._reset_connection()
            return f"Error creating directory: {str(e)}"

    @tool
    def delete_directory(self, directory_path: str) -> str:
        """
        Remove a directory from the remote server.
        Note: The directory usually must be empty.

        Args:
            directory_path (str): The path of the directory to remove.

        Returns:
            str: Confirmation message.

        Example call:
            delete_directory(
                directory_path="/uploads/temp"
            )
        """
        try:
            if self.protocol == "sftp":
                sftp = self._get_sftp_connection()
                sftp.rmdir(directory_path)
            else:
                ftp = self._get_ftp_connection()
                ftp.rmd(directory_path)
            return f"Successfully deleted directory '{directory_path}'."
        except Exception as e:
            self._reset_connection()
            return f"Error deleting directory: {str(e)} {traceback.format_exc()}"

    @tool
    def delete_file(self, file_path: str) -> str:
        """
        Delete a specific file from the remote server.

        Args:
            file_path (str): The full path of the file to delete.

        Returns:
            str: Confirmation message.

        Example call:
            delete_file(
                file_path="/temp/cache_dump.tmp"
            )
        """
        try:
            if self.protocol == "sftp":
                sftp = self._get_sftp_connection()
                sftp.remove(file_path)
            else:
                ftp = self._get_ftp_connection()
                ftp.delete(file_path)
            return f"Successfully deleted '{file_path}'."
        except Exception as e:
            self._reset_connection()
            return f"Error deleting file: {str(e)}"
