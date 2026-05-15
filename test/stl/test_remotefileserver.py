import os
import json
import threading
import pytest

# Ephemeral FTP server imports
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

from nemantix.core import Toolset
from nemantix.stl.remote_filesystem.base import RemoteFileSystemToolset

# ==========================================
# 1. SFTP CONFIGURATION & FIXTURES (LIVE)
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "sftp_config.json")
REMOTE_SANDBOX = "/tmp/pytest_sftp_sandbox"


@pytest.fixture(scope="session")
def real_credentials():
    """Loads real SFTP credentials from a JSON file. Skips tests if missing."""
    if not os.path.exists(CONFIG_PATH):
        pytest.skip(f"Config file {CONFIG_PATH} missing. Skipping live SFTP tests.")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = ["host", "user", "password", "protocol"]
    if not all(k in config for k in required_keys):
        pytest.skip("JSON is missing required credential fields. Skipping tests.")

    return config


@pytest.fixture(autouse=True)
def setup_and_teardown_sandbox(real_credentials):
    """
    Creates the sandbox directory on the remote server before each test,
    and attempts to clean it up afterward so we don't leave junk on the server.
    """
    args = (
        real_credentials["host"],
        real_credentials["user"],
        real_credentials["password"],
        22,
        real_credentials["protocol"],
    )

    # Initialize directly for setup/teardown maintenance so we don't pollute the Toolset cache
    ts = RemoteFileSystemToolset(*args)

    # Setup
    ts.create_directory(REMOTE_SANDBOX)

    yield  # Test runs here

    # Teardown
    try:
        ts.list_files(REMOTE_SANDBOX)
        ts.delete_file(f"{REMOTE_SANDBOX}/test_write.txt")
        ts.delete_file(f"{REMOTE_SANDBOX}/uploaded.txt")
        ts.delete_directory(f"{REMOTE_SANDBOX}/subfolder")
        ts.delete_directory(REMOTE_SANDBOX)
    except Exception as e:
        print(f"Cleanup warning: {e}")


# ==========================================
# 2. FTP CONFIGURATION & FIXTURES (EPHEMERAL)
# ==========================================


@pytest.fixture(scope="session")
def ftp_server(tmp_path_factory):
    """
    Spins up a local, ephemeral FTP server in a background thread.
    The server lives for the duration of the test session and is torn down afterward.
    """
    ftp_root = tmp_path_factory.mktemp("ftp_root")

    authorizer = DummyAuthorizer()
    authorizer.add_user("testuser", "testpass", str(ftp_root), perm="elradfmwMT")

    handler = FTPHandler
    handler.authorizer = authorizer

    # Bind to localhost on port 0 to allow the OS to pick an open random port
    server = FTPServer(("127.0.0.1", 0), handler)

    # Use .address instead of .server_address for pyftpdlib
    host, port = server.address

    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    yield {
        "host": host,
        "port": port,
        "user": "testuser",
        "password": "testpass",
        "protocol": "ftp",
        "root_dir": str(ftp_root),
    }

    server.close_all()


@pytest.fixture
def ftp_args(ftp_server):
    """Returns the initialization tuple for the RemoteFileSystemToolset."""
    return (
        ftp_server["host"],
        ftp_server["user"],
        ftp_server["password"],
        ftp_server["port"],
        ftp_server["protocol"],
    )


# ==========================================
# 3. LOCAL FTP TESTS
# ==========================================


class TestEphemeralFTP:
    def test_write_and_read_ftp(self, ftp_args):
        """Test writing a file to the ephemeral server and reading it back."""
        # Use an alias to ensure we don't mix cache instances with the SFTP tests
        ts_write = Toolset.get_tool(
            "RemoteFileSystemToolset.write_file",
            instance_alias="ftp_env",
            instance_args=ftp_args,
        )
        ts_read = Toolset.get_tool(
            "RemoteFileSystemToolset.read_file",
            instance_alias="ftp_env",
            instance_args=ftp_args,
        )

        remote_file = "test_write.txt"
        content = "Hello from Ephemeral FTP!"

        write_result = ts_write(file_path=remote_file, content=content)
        assert "Successfully wrote" in write_result

        read_result = ts_read(file_path=remote_file)
        assert read_result == content

    def test_list_files_ftp(self, ftp_args):
        """Test listing files in the remote directory."""
        ts_write = Toolset.get_tool(
            "RemoteFileSystemToolset.write_file",
            instance_alias="ftp_env",
            instance_args=ftp_args,
        )
        ts_list = Toolset.get_tool(
            "RemoteFileSystemToolset.list_files",
            instance_alias="ftp_env",
            instance_args=ftp_args,
        )

        remote_file = "list_test.txt"
        ts_write(file_path=remote_file, content="Data")

        result = ts_list(directory=".")
        assert remote_file in result

    def test_directory_operations(self, ftp_args):
        """Test creating and deleting remote directories."""
        ts_create_dir = Toolset.get_tool(
            "RemoteFileSystemToolset.create_directory",
            instance_alias="ftp_env",
            instance_args=ftp_args,
        )
        ts_list = Toolset.get_tool(
            "RemoteFileSystemToolset.list_files",
            instance_alias="ftp_env",
            instance_args=ftp_args,
        )
        ts_delete_dir = Toolset.get_tool(
            "RemoteFileSystemToolset.delete_directory",
            instance_alias="ftp_env",
            instance_args=ftp_args,
        )

        remote_folder = "new_subfolder"

        create_result = ts_create_dir(directory_path=remote_folder)
        assert "Successfully created" in create_result

        list_result = ts_list(directory=".")
        assert remote_folder in list_result

        delete_result = ts_delete_dir(directory_path=remote_folder)
        assert "Successfully deleted" in delete_result


# ==========================================
# 4. LIVE SFTP TESTS
# ==========================================


@pytest.mark.external
class TestLiveSFTP:
    def test_write_and_read_sftp(self, real_credentials):
        """Test physically writing a file to the server and reading it back."""
        args = (
            real_credentials["host"],
            real_credentials["user"],
            real_credentials["password"],
            22,
            real_credentials["protocol"],
        )

        # Added instance_alias="sftp_env" to keep SFTP instance separate from FTP instance
        ts_write = Toolset.get_tool(
            "RemoteFileSystemToolset.write_file",
            instance_alias="sftp_env",
            instance_args=args,
        )
        ts_read = Toolset.get_tool(
            "RemoteFileSystemToolset.read_file",
            instance_alias="sftp_env",
            instance_args=args,
        )

        remote_file = f"{REMOTE_SANDBOX}/test_write.txt"
        content = "Hello from Pytest over real SFTP!"

        write_result = ts_write(file_path=remote_file, content=content)
        assert "Successfully wrote" in write_result

        read_result = ts_read(file_path=remote_file)
        assert read_result == content

    def test_list_files_sftp(self, real_credentials):
        """Test listing files in the remote directory."""
        args = (
            real_credentials["host"],
            real_credentials["user"],
            real_credentials["password"],
            22,
            real_credentials["protocol"],
        )

        ts_write = Toolset.get_tool(
            "RemoteFileSystemToolset.write_file",
            instance_alias="sftp_env",
            instance_args=args,
        )
        ts_list = Toolset.get_tool(
            "RemoteFileSystemToolset.list_files",
            instance_alias="sftp_env",
            instance_args=args,
        )

        remote_file = f"{REMOTE_SANDBOX}/test_write.txt"
        ts_write(file_path=remote_file, content="Data")

        result = ts_list(directory=REMOTE_SANDBOX)
        assert "test_write.txt" in result

    def test_upload_file_sftp(self, real_credentials, tmp_path):
        """Test uploading a local file from your machine to the remote server."""
        args = (
            real_credentials["host"],
            real_credentials["user"],
            real_credentials["password"],
            22,
            real_credentials["protocol"],
        )

        ts_upload = Toolset.get_tool(
            "RemoteFileSystemToolset.upload_file",
            instance_alias="sftp_env",
            instance_args=args,
        )
        ts_read = Toolset.get_tool(
            "RemoteFileSystemToolset.read_file",
            instance_alias="sftp_env",
            instance_args=args,
        )

        local_file = tmp_path / "local_test.txt"
        local_content = "This is a local file uploaded via SFTP."
        local_file.write_text(local_content)

        remote_file = f"{REMOTE_SANDBOX}/uploaded.txt"

        upload_result = ts_upload(local_path=str(local_file), remote_path=remote_file)
        assert "Successfully uploaded" in upload_result

        read_result = ts_read(file_path=remote_file)
        assert read_result == local_content

    def test_directory_operations_sftp(self, real_credentials):
        """Test creating and deleting remote directories."""
        args = (
            real_credentials["host"],
            real_credentials["user"],
            real_credentials["password"],
            22,
            real_credentials["protocol"],
        )

        ts_create_dir = Toolset.get_tool(
            "RemoteFileSystemToolset.create_directory",
            instance_alias="sftp_env",
            instance_args=args,
        )
        ts_delete_dir = Toolset.get_tool(
            "RemoteFileSystemToolset.delete_directory",
            instance_alias="sftp_env",
            instance_args=args,
        )

        remote_folder = f"{REMOTE_SANDBOX}/subfolder"

        create_result = ts_create_dir(directory_path=remote_folder)
        assert "Successfully created" in create_result

        delete_result = ts_delete_dir(directory_path=remote_folder)
        assert "Successfully deleted" in delete_result
