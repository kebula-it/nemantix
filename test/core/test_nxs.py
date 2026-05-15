from pathlib import Path
import pytest
from lark import Lark, exceptions

# --- Paths resolved from this file ---
HERE = Path(__file__).parent
GRAMMAR_FILE = HERE / "../../src/nemantix/core/nxs_v2_grammar.lark"
EXAMPLES_DIR = HERE / "test_scripts"
EXAMPLE_FILES = sorted(map(str, EXAMPLES_DIR.glob("*.nxs")))  # list[str]


@pytest.fixture(scope="session")
def nxs_parser():
    try:
        grammar = GRAMMAR_FILE.read_text(encoding="utf-8")
        return Lark(grammar, start="start")
    except FileNotFoundError:
        pytest.fail(f"Grammar file not found: '{GRAMMAR_FILE}'", pytrace=False)


@pytest.mark.parametrize("filepath", EXAMPLE_FILES)
def test_parse_example_file(nxs_parser, filepath):
    try:
        script_content = Path(filepath).read_text(encoding="utf-8")
        if not script_content.strip():
            print(f"Note: {filepath} is empty, which is valid.")
            return
        nxs_parser.parse(script_content)
    except FileNotFoundError:
        pytest.fail(f"Test file disappeared: {filepath}", pytrace=False)
    except exceptions.LarkError as e:
        pytest.fail(
            f"Failed to parse '{filepath}'.\n--- Lark Error ---\n{e}\n------------------",
            pytrace=False,
        )
