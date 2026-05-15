from abc import ABC, abstractmethod
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

from nemantix.core.custom_types import PathLike
from nemantix.core.script import Script, ScriptTypeEnum
from nemantix.common.logger import get_package_logger
from nemantix.security.signer import SIGNATURE_HEADER

logger = get_package_logger(__name__)


class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, script: Script) -> bool:
        pass


class DebugVerifier(BaseVerifier):
    """Dummy verifier for development purposes"""

    def verify(self, script: Script) -> bool:
        return True


class Verifier(BaseVerifier):
    def __init__(self, public_key_path: PathLike):
        super().__init__()

        self.public_key_path = Path(public_key_path)
        assert self.public_key_path.exists() and self.public_key_path.is_file()

        with open(self.public_key_path, "rb") as key_file:
            self.public_key = serialization.load_pem_public_key(key_file.read())

    def verify(self, script: Script) -> bool:
        """Verifies the authenticity of the given script"""
        location = script.get_location()

        if script.type != ScriptTypeEnum.NXV:
            # TODO: should raise error or return false if not NXV?
            logger.warning(f'"{location}" does not have an .nxv extension.')

        try:
            lines = script.read(read_as_lines_list=True)

            # Check for the signature header
            content_lines = []
            signature = None

            for line in lines:
                if line.startswith(SIGNATURE_HEADER):
                    if signature is None:
                        signature = line.strip().split(SIGNATURE_HEADER)[1].strip()
                else:
                    content_lines.append(line)

            if signature is None:
                logger.error(f'"{location}" is missing the NXV signature header.')
                raise InvalidSignature("missing NXV signature header!")

            data = "\n".join(lines[1:])
            data = bytearray(data, "utf-8")

            try:
                # Convert the hex string back into raw binary bytes
                signature = bytes.fromhex(signature)
            except ValueError:
                logger.error(
                    f'"{location}" has an invalid hexadecimal signature format.'
                )
                return False

            self.public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))

            logger.info(f'"{location}" verified successfully.')
            return True

        except (FileNotFoundError, InvalidSignature) as e:
            logger.error(f'Verification failed for "{location}": {e}!', exc_info=True)
            return False
