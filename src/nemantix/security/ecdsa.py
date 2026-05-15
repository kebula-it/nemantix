from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from nemantix.common.logger import get_package_logger
from nemantix.core.custom_types import PathLike


logger = get_package_logger(__name__)


def generate_keys(base_path: PathLike):
    """Elliptic curve key pair generation"""
    private_key = ec.generate_private_key(ec.SECP256R1())
    base_path = Path(base_path)
    assert base_path.is_dir()

    with open(base_path / "nmx_ecdsa_private.pem", "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    public_key = private_key.public_key()
    with open(base_path / "nmx_ecdsa_public.pem", "wb") as f:
        f.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

    logger.info("ECDSA Keys generated successfully.")
