import collections
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from nemantix.core.exceptions import NemantixRuntimeException
from nemantix.core.runtime import DocRef, Struct
from nemantix.experimental.stateful_agent import StatefulAgent


class DummySchema(BaseModel):
    key: str


@pytest.fixture
def mock_expertise():
    """Provides a dummy Expertise mock for Agent initialization."""
    return MagicMock()


@pytest.fixture
def stateful_agent(mock_expertise):
    """Fixture to provide a StatefulAgent instance without building on start."""
    with patch("nemantix.core.agent.Executor"):
        return StatefulAgent(
            expertise=mock_expertise,
            max_history_turns=2,
            strings_max_size=50,
            build_on_start=False,
        )


def test_initialization(mock_expertise):
    """Tests if deque and limits are initialized properly[cite: 6]."""
    agent = StatefulAgent(
        expertise=mock_expertise,
        max_history_turns=16,
        strings_max_size=1000,
        build_on_start=False,
    )
    assert agent.max_history_turns == 16
    assert agent.max_strings_size == 1000
    assert isinstance(agent.chat_history, collections.deque)
    assert agent.chat_history.maxlen == 16


def test_augmented_request_empty_history(stateful_agent):
    """Test that the first request is passed exactly as-is without a transcript[cite: 6]."""
    req = "Run analysis"
    assert stateful_agent._build_augmented_request(req) == req


@patch("nemantix.core.agent.Agent.run")
def test_chat_history_is_appended_on_success(mock_super_run, stateful_agent):
    """Verifies that a successful run is logged in the history[cite: 6]."""
    mock_super_run.return_value = (None, "Analysis complete")

    exception, output = stateful_agent.run(user_request="Run analysis")

    assert exception is None
    assert output == "Analysis complete"
    assert len(stateful_agent.chat_history) == 1

    logged_turn = stateful_agent.chat_history[0]
    assert logged_turn["user"] == "Run analysis"
    assert logged_turn["agent"] == "Analysis complete"


def test_augmented_request_with_history(stateful_agent):
    """Verifies the prompt generation when history is present[cite: 6]."""
    stateful_agent.chat_history.append({"user": "First request", "agent": "Output A"})

    augmented = stateful_agent._build_augmented_request("Second request")

    assert "[Previous Transcript]" in augmented
    assert "User: First request" in augmented
    assert "Agent: Output A" in augmented
    assert "\n[Current Request]" in augmented
    assert "User: Second request" in augmented


@patch("nemantix.core.agent.Agent.run")
def test_rolling_window_eviction(mock_super_run, stateful_agent):
    """Tests if the oldest transcript is removed when max_history_turns is exceeded[cite: 6]."""
    assert stateful_agent.chat_history.maxlen == 2

    # Run 3 times, exceeding maxlen of 2
    mock_super_run.return_value = (None, "Out 1")
    stateful_agent.run("Req 1")

    mock_super_run.return_value = (None, "Out 2")
    stateful_agent.run("Req 2")

    mock_super_run.return_value = (None, "Out 3")
    stateful_agent.run("Req 3")

    assert len(stateful_agent.chat_history) == 2
    assert stateful_agent.chat_history[0]["user"] == "Req 2"
    assert stateful_agent.chat_history[1]["user"] == "Req 3"


@patch("nemantix.core.agent.Agent.run")
def test_error_handling_in_history(mock_super_run, stateful_agent):
    """Verifies that if the Executor raises a NemantixException, the Agent logs the error text[cite: 6]."""
    error_instance = NemantixRuntimeException("Database timeout")
    mock_super_run.return_value = (error_instance, None)

    exception, output = stateful_agent.run(user_request="Fetch records")

    assert exception is error_instance
    assert output is None

    logged_turn = stateful_agent.chat_history[0]
    assert "Execution failed with error: Database timeout" in logged_turn["agent"]


def test_clear_history(stateful_agent):
    """Ensures manual clearing of the rolling window works[cite: 6]."""
    stateful_agent.chat_history.append({"user": "Q", "agent": "A"})
    assert len(stateful_agent.chat_history) == 1

    stateful_agent.clear_history()
    assert len(stateful_agent.chat_history) == 0


def test_format_output_truncation(stateful_agent):
    """Ensures massively long string outputs are truncated to max_strings_size[cite: 6]."""
    # strings_max_size is set to 50 in the fixture
    massive_output = "A" * 100
    formatted = stateful_agent._format_output_for_history(massive_output)

    assert formatted.startswith("A" * 50)
    assert formatted.endswith("... [Truncated for length]")
    assert len(formatted) == 50 + len("... [Truncated for length]")


def test_format_output_none(stateful_agent):
    """Ensures None outputs are safely handled[cite: 6]."""
    formatted = stateful_agent._format_output_for_history(None)
    assert formatted == "<Task completed silently>"


def test_format_output_basemodel(stateful_agent):
    """Ensures Pydantic models are serialized to JSON[cite: 6]."""
    model = DummySchema(key="value")
    formatted = stateful_agent._format_output_for_history(model)
    assert formatted == '{"key":"value"}'


def test_format_output_dict(stateful_agent):
    """Ensures standard dictionaries are cast to string[cite: 6]."""
    data = {"hello": "world"}
    formatted = stateful_agent._format_output_for_history(data)
    assert formatted == "{'hello': 'world'}"


def test_format_output_docref(stateful_agent):
    """Ensures DocRef formatting works natively through ReActAgent._doc_to_str."""

    class MockDocRef(DocRef):
        def __init__(self):
            super().__init__(
                node_id="12345",
                score=0.0,
                breadcrumbs="root > folder > file",
                content="Summary of document",
            )

    mock_doc = MockDocRef()
    expected_str = '{"node_id": "12345", "content": "Summary of document", "breadcrumbs": "root > folder > file"}'

    formatted = stateful_agent._format_output_for_history(mock_doc)

    assert formatted == expected_str


def test_struct_to_str_recursive(stateful_agent):
    """Ensures Struct inputs unpack into string representations recursively."""
    # Main Struct
    mock_struct = MagicMock()
    mock_struct.__class__ = Struct

    # Inner Struct (simulating recursion)
    inner_struct = MagicMock()
    inner_struct.__class__ = Struct
    inner_struct.to_args_and_kwargs.return_value = (["nested_arg"], {})

    # A DocRef instance with attributes for the real _doc_to_str method
    mock_doc = MagicMock()
    mock_doc.__class__ = DocRef
    mock_doc.node_id = "doc_1"
    mock_doc.content = "content_1"
    mock_doc.breadcrumbs = "bread_1"
    expected_doc_str = (
        '{"node_id": "doc_1", "content": "content_1", "breadcrumbs": "bread_1"}'
    )

    # Set up the outer struct to return these mixed types
    mock_struct.to_args_and_kwargs.return_value = (
        ["basic_arg", inner_struct],
        {"doc_key": mock_doc, "str_key": "val"},
    )

    formatted = stateful_agent._struct_to_str(mock_struct)

    assert "0: basic_arg" in formatted
    assert "1: {0: nested_arg}" in formatted
    assert f"doc_key: {expected_doc_str}" in formatted
    assert "str_key: val" in formatted
