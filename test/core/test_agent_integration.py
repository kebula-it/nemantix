import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from nemantix.core.agent import Agent
from nemantix.core.coder import Coder
from nemantix.core.expertise import Expertise
from nemantix.core.script import Script
from nemantix.core.source_manager import LocalSourceManager
from nemantix.llm import AbstractLLMProxy
from nemantix.llm.abstract_proxy import LLMResponse, LLMUsage, StructuredLLMResponse
from nemantix.security.verifier import DebugVerifier

HERE = Path(__file__).parent


@pytest.fixture
def nxc_file():
    """Points to the existing deliberate_selection.nxc provided in the test_scripts folder."""
    file_path = HERE / "test_scripts" / "deliberate_selection.nxc"

    if not file_path.exists():
        pytest.fail(f"Could not find {file_path}. Ensure it is placed correctly.")

    return file_path


@pytest.fixture
def mock_credentials(tmp_path):
    """Creates a temporary dummy credentials file for Agent initialization."""
    cred_file = tmp_path / "credentials.json"
    cred_file.write_text('{"api_key": "fake"}')
    return cred_file


def mock_llm_proxy():
    """
    Mocks the LLM proxy for tests.
    Emulates the LLM response without the need of an API KEY.
    """
    mock_llm = MagicMock(spec=AbstractLLMProxy)

    def _resp(text):
        return LLMResponse(
            text=text,
            tool_calls=[],
            usage=LLMUsage(input_tokens=0, output_tokens=0),
            proxy=mock_llm,
        )

    def side_effect(prompt, **__):
        # 1. Deliberate Selection Phase
        if "find the name of the deliberate statement" in prompt:
            return _resp("GenerateTicket")

        # 2. JSON Input Extraction Phase
        if "extract the (possible) inputs" in prompt:
            return _resp(
                json.dumps(
                    [
                        {"name": "error_code", "type": "str", "value": "500"},
                        {
                            "name": "description",
                            "type": "str",
                            "value": "Internal Server Error",
                        },
                    ]
                )
            )

        return _resp("")

    class _SelectionSchema(BaseModel):
        name: str
        motivation: str

    def invoke_structured_side_effect(prompt, schema=None, **__):
        return StructuredLLMResponse(
            result=_SelectionSchema(
                name="GenerateTicket", motivation="matches error ticket creation"
            ),
            usage=LLMUsage(input_tokens=0, output_tokens=0),
            proxy=mock_llm,
        )

    mock_llm.invoke.side_effect = side_effect
    mock_llm.invoke_structured.side_effect = invoke_structured_side_effect
    return mock_llm


def test_agent_run_nlp_request(
    nxc_file, mock_credentials, dummy_llm_proxy_config_class
):
    """
    Tests the agent using a natural language request.
    Relies on the LLM mock to route to GenerateTicket and extract inputs.
    """
    # Create a real Script instance and parse the NXC file
    script = Script(location=str(nxc_file), source_manager=LocalSourceManager())
    script.parse()  # Parsing the NXC file

    # Create a mocked LLM proxy
    llm = mock_llm_proxy()

    # Create the Expertise instance
    coder = Coder(llm)
    expertise = Expertise(script_list=[script], coder=coder, verifier=DebugVerifier())
    expertise.build()

    # Instantiate the Agent with real Expertise and mocked LLM proxy
    agent = Agent(
        expertise=expertise,  # Use the real Expertise
        llm_proxy=None,
        proxy_config=dummy_llm_proxy_config_class(llm),
        external_vars={},
        use_embedder=False,
        use_knowledge_base=False,
        build_on_start=True,
    )

    # Natural Language Request
    request = "We have a 500 Internal Server Error"
    exception, outputs = agent.run(user_request=request)

    # Check if there was no exception
    assert exception is None
    assert outputs is not None

    # GenerateTicket returns a list of 3 variables: [[ticket_id], [submission_status], [submission_payload]]
    assert outputs[0] == "TCK-500"
    assert outputs[1] == "pending"

    # Verify the nested dictionary structures compiled from the NXC file
    payload = outputs[2]
    assert payload["endpoint"] is None
    assert payload["method"] == "POST"


def test_agent_run_coded_request(nxc_file, mock_credentials):
    """
    Tests the agent using an explicit coded 'do' statement (NXS syntax).
    This proves that the system bypasses the LLM completely when given executable code.
    """
    # Create a real Script instance and parse the NXC file
    script = Script(location=str(nxc_file), source_manager=LocalSourceManager())
    script.parse()  # Parsing the NXC file

    # Create a mocked LLM proxy
    llm = mock_llm_proxy()

    # Create the Expertise instance
    coder = Coder(llm)
    expertise = Expertise(script_list=[script], coder=coder, verifier=DebugVerifier())
    expertise.build()

    # Instantiate the Agent with real Expertise and mocked LLM proxy
    agent = Agent(
        expertise=expertise,  # Use the real Expertise
        llm_proxy=llm,
        external_vars={},
        use_embedder=False,
        use_knowledge_base=False,
        build_on_start=True,
    )

    # Coded Request: Explicitly call the deliberate with defined parameters
    request = (
        'do GenerateTicket using [[error_code]="404", [description]="Not Found"] '
        "producing [[a], [b], [c]]"
    )
    exception, outputs = agent.run(user_request=request)

    # Check if there was no exception
    assert exception is None
    assert outputs is not None

    # Verify the output was processed exactly as requested
    assert outputs[0] == "TCK-404"
    assert outputs[1] == "pending"
    assert outputs[2]["method"] == "POST"
