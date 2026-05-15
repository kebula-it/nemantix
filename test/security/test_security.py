import pytest
from pathlib import Path

from nemantix.security.ecdsa import generate_keys
from nemantix.security.signer import Signer
from nemantix.security.verifier import Verifier, DebugVerifier
from nemantix.core.script import Script
from nemantix.core.source_manager import LocalSourceManager


# --- Fixtures ---


@pytest.fixture
def key_dir(tmp_path: Path) -> Path:
    """Fixture to generate keys in a temporary directory and return the path."""
    generate_keys(tmp_path)
    return tmp_path


@pytest.fixture
def private_key_path(key_dir: Path) -> Path:
    return key_dir / "nmx_ecdsa_private.pem"


@pytest.fixture
def public_key_path(key_dir: Path) -> Path:
    return key_dir / "nmx_ecdsa_public.pem"


@pytest.fixture
def dummy_file(tmp_path: Path) -> Path:
    """Fixture to create a dummy file for signing/verifying."""
    file_path = tmp_path / "test_data.nxc"
    file_path.write_text(
        """
    deliberate dummy when >> condition <<:
    plan:
        body:
        __
    __plan
    __deliberate
    """
    )
    return file_path


@pytest.fixture
def dummy_script(dummy_file: Path) -> Script:
    """Fixture to create a dummy file for signing/verifying."""
    script = Script(location=dummy_file, source_manager=LocalSourceManager())
    return script


# --- Tests ---


def test_generate_keys(tmp_path: Path):
    """Test that keys are generated successfully."""
    generate_keys(tmp_path)

    private_key = tmp_path / "nmx_ecdsa_private.pem"
    pub_key = tmp_path / "nmx_ecdsa_public.pem"

    assert private_key.exists() and private_key.is_file()
    assert pub_key.exists() and pub_key.is_file()
    assert private_key.stat().st_size > 0
    assert pub_key.stat().st_size > 0


def test_signer_initialization_fails_with_invalid_path():
    """Test that Signer asserts if the private key path doesn't exist."""
    with pytest.raises(AssertionError):
        Signer("non_existent_key.pem")


def test_signer_creates_signature(private_key_path: Path, dummy_script: Script):
    """Test that the signer generates a .sig file alongside the original file."""
    signer = Signer(private_key_path)
    sig_script = signer.sign(dummy_script)
    sig_path = Path(sig_script.get_location())

    assert sig_path.exists()
    assert sig_path.name == Path(dummy_script.get_location()).with_suffix(".nxv").name
    assert sig_path.stat().st_size > 0


def test_verify_valid_signature(
    private_key_path: Path, public_key_path: Path, dummy_script: Script
):
    """Test full cycle: sign a file and verify it successfully."""
    signer = Signer(private_key_path)
    signed_script = signer.sign(dummy_script)

    verifier = Verifier(public_key_path)
    is_valid = verifier.verify(signed_script)

    assert is_valid is True


def test_verify_invalid_signature_data_tampered(
    private_key_path: Path,
    public_key_path: Path,
    dummy_script: Script,
    dummy_file: Path,
):
    """Test that verification fails if the data is modified after signing."""
    signer = Signer(private_key_path)
    signer.sign(dummy_script)

    # Tamper with the original file
    with open(dummy_file, "a") as f:
        f.write("tampered data")

    script = Script(location=dummy_file, source_manager=LocalSourceManager())

    verifier = Verifier(public_key_path)
    is_valid = verifier.verify(script)

    assert is_valid is False


def test_verify_missing_signature(public_key_path: Path, dummy_script: Script):
    """Test that verification handles FileNotFoundError when the .sig file is missing."""
    verifier = Verifier(public_key_path)
    is_valid = verifier.verify(dummy_script)

    assert is_valid is False


def test_debug_verifier(dummy_script: Script):
    """Test that DebugVerifier always returns True, even without a signature."""
    debug_verifier = DebugVerifier()

    assert debug_verifier.verify(dummy_script) is True

    script = Script(
        location="some_random_nonexistent_path.nxs", source_manager=LocalSourceManager()
    )
    assert debug_verifier.verify(script) is True
