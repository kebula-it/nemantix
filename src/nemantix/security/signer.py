from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

from nemantix.core.custom_types import PathLike
from nemantix.core.script import Script
from nemantix.common.logger import get_package_logger

logger = get_package_logger(__name__)

SIGNATURE_HEADER = "# NXV-SIGN: "


class Signer:
    def __init__(self, private_key_path: PathLike):
        self.private_key_path = Path(private_key_path)
        assert self.private_key_path.exists() and self.private_key_path.is_file()

        with open(self.private_key_path, "rb") as key_file:
            self.private_key = serialization.load_pem_private_key(
                key_file.read(), password=None
            )

    def sign(self, script: Script) -> Script:
        """Signs the given script, and returns the signature path"""
        assert isinstance(script, Script)

        data = script.read(read_as_lines_list=False)
        data = str(data).replace("\n\r", "\n")

        # skip old signature, if any
        content_lines = []

        for line in data.split("\n"):
            if not line.startswith(SIGNATURE_HEADER):
                content_lines.append(line.replace("\n", ""))

        data = "\n".join(content_lines)
        data = bytearray(data, "utf-8")

        signature = self.private_key.sign(data, ec.ECDSA(hashes.SHA256()))
        signature = str(signature.hex().encode("utf-8"))[2:-1]

        content_lines = [f"{SIGNATURE_HEADER} {signature}"] + content_lines
        nxv_content = "\n".join(content_lines)

        output_path = script.get_location_with_extension(ext="nxv")
        nxv_script = Script(location=output_path, source_manager=script.source_manager)
        nxv_script.write(content=nxv_content, location=output_path)

        logger.info(
            f"Signed {nxv_script.get_location()}. Signature size: {len(signature)} bytes."
        )
        return nxv_script
