import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemantix.core.executor import Executor
from nemantix.core.node import FileMeta, Deliberate
from nemantix.llm import AbstractLLMProxy
from nemantix.llm.abstract_proxy import LLMResponse, LLMUsage
from nemantix.security.verifier import BaseVerifier
from nemantix.core.source_manager import LocalSourceManager
from nemantix.core.script import Script
from nemantix.core.coder import Coder
from nemantix.core.expertise import Expertise
from nemantix.security.verifier import DebugVerifier

HERE = Path(__file__).parent


@pytest.fixture(scope='module')
def example_nxs(project_root: Path):
    return HERE / 'test_scripts' / 'deliberate_selection.nxs'


@pytest.fixture(scope='module')
def example_nxc(example_nxs):
    file_path = example_nxs.with_suffix('.nxc')

    if not file_path.exists():
        data = example_nxs.read_text(encoding='utf-8')
        file_path.write_text(data, encoding='utf-8')

    return file_path


@pytest.fixture(scope='module')
def mock_llm():
    """
    Creates a mock LLM that behaves predictably for tests.
    """
    llm = MagicMock(spec=AbstractLLMProxy)
    llm.llm_proxy = "fake"
    # Configure the mock to return a known deliberate name found in the NXS example
    llm.invoke.return_value = LLMResponse(text='SummarizeSupportTicket', tool_calls=[], usage=LLMUsage(input_tokens=0, output_tokens=0))
    return llm


@pytest.fixture(scope='module')
def mock_verifier():
    """
    Creates a mock Verifier that behaves predictably for tests.
    """
    verifier = MagicMock(spec=BaseVerifier)
    verifier.verify.return_value = True
    return verifier


@pytest.fixture(scope='module')
def shared_executor(example_nxc, mock_verifier, mock_llm):
    """
    Initializes the Executor once per module with a mock LLM,
    mocked SourceManager, and mocked Coder.
    """
    script = Script(location=str(example_nxc), source_manager=LocalSourceManager())
    script.parse()

    # 3. Mock the Coder
    mock_coder = MagicMock(spec=Coder)
    mock_coder.code_deliberate.return_value = "mocked nxc content"
    mock_coder.llm_proxy = "mocked llm"

    # 4. Instantiate Expertise
    expertise = Expertise(script_list=[script], coder=mock_coder, verifier=DebugVerifier())
    expertise.build()

    return Executor(expertise=expertise, llm=mock_llm)


@pytest.fixture(scope='function')
def executor(shared_executor):
    """
    Yields the shared executor instance for tests to use.
    Crucially, it handles cleanup by restoring the original interpreter
    after a test runs.
    """
    # Save the real interpreter state before the test runs
    real_interpreter = shared_executor.interpreter

    yield shared_executor

    # Restore the real interpreter after the test finishes
    shared_executor.interpreter = real_interpreter


def test_parse_user_inputs_retry_logic(executor, mock_llm):
    """
    Verifies that _parse_user_inputs retries when the LLM returns invalid JSON,
    and succeeds if a subsequent attempt provides valid JSON.
    """
    # Reset the mock's call history from previous tests
    mock_llm.invoke.reset_mock()

    # Setup mock LLM to fail first, then succeed
    invalid_json = "This is not JSON at all"
    valid_json = json.dumps([{"name": "retry_target", "type": "int", "value": 99}])

    # side_effect allows us to return different values on consecutive calls
    mock_llm.invoke.side_effect = [
        LLMResponse(text=invalid_json, tool_calls=[], usage=LLMUsage(input_tokens=0, output_tokens=0)),
        LLMResponse(text=valid_json, tool_calls=[], usage=LLMUsage(input_tokens=0, output_tokens=0)),
    ]

    # Create a mock action block
    mock_deliberate = MagicMock(spec=Deliberate)
    mock_deliberate.name = "SummarizeSupportTicket"  # Set the name attribute for the action
    mock_file_meta = MagicMock(spec=FileMeta)
    mock_file_meta.line = (0, 1)
    mock_deliberate.meta = dict(file_meta=mock_file_meta)

    # Get a real Script instance
    script = executor.expertise.get_script_from_deliberate('SummarizeSupportTicket')

    # Execute (should internally fail once, catch the error, retry, and succeed)
    result = executor._parse_user_inputs(request="Try again", deliberate=mock_deliberate,
                                         script=script)

    # Assert that the result is not None (indicating successful parsing)
    assert result is not None
    assert len(result.value) == 1  # Assuming that one input was parsed successfully

    # Ensure that the LLM's 'invoke' method was called twice (once for failure, once for success)
    assert mock_llm.invoke.call_count == 2