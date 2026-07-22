import json
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel, RootModel

from nemantix.core.agent import Agent
from nemantix.core.exceptions import NemantixRuntimeException
from nemantix.core.runtime import Struct
from nemantix.llm import AbstractLLMProxy


# ==========================================
# Tests for Agent
# ==========================================
class DummySchema(BaseModel):
    name: str
    status: str


@pytest.fixture
def mock_nxs_file(tmp_path):
    """Creates a temporary NXS script."""
    nxs_file = tmp_path / "test.nxs"
    nxs_file.write_text(
        "deliberate TestMain when >> test <<: plan: action TestMain: body: return 1 __body __action __plan __deliberate"
    )
    return nxs_file


@pytest.fixture
def mock_agent(mock_nxs_file):
    """Fixture to provide an Agent instance with mocked dependencies."""
    # Mocking Executor and LLM
    with patch("nemantix.core.agent.Executor") as MockExecutor:
        # Pre-configure the mock instance so it returns a mock output
        mock_executor_instance = MockExecutor.return_value
        mock_executor_instance.execute.return_value = {"status": "success"}

        mock_llm = MagicMock(spec=AbstractLLMProxy)

        # Initialize the Agent with the mock dependencies
        agent = Agent(
            expertise=MagicMock(),  # Mock Expertise (you can further mock the needed methods)
            llm_proxy=mock_llm,
            external_vars={},
            use_embedder=False,
            use_knowledge_base=False,
            build_on_start=True,
        )

        return agent


@patch("nemantix.core.agent.Executor")
def test_agent_run_success(mock_executor_class, mock_nxs_file):
    """
    Verifies the one-shot running mode where the Agent executes the task until completion
    without errors and updates its state.
    """
    mock_executor_instance = MagicMock()
    # Mock execution output
    mock_executor_instance.execute.return_value = {"status": "success"}

    # Mock the returned state struct
    mock_state_struct = MagicMock(spec=Struct)
    mock_state_struct.to_dict.return_value = {"var1": "value1"}
    mock_executor_instance.get_agent_state.return_value = mock_state_struct

    mock_executor_class.return_value = mock_executor_instance

    agent = Agent(expertise=MagicMock(), llm_proxy=MagicMock())

    # Execute
    exception, outputs = agent.run(user_request="Do a test run")

    # Assert
    assert exception is None
    assert outputs == {"status": "success"}

    mock_executor_instance.execute.assert_called_once_with(user_request="Do a test run")

    # Verify state was updated
    assert agent.state.get().get("var1") == "value1"


@patch("nemantix.core.agent.Executor")
def test_agent_run_nemantix_exception(mock_executor_class, mock_nxs_file):
    """
    Verifies that if the Executor raises a NemantixException, the Agent catches it
    and returns it as the first element of the tuple.
    """
    mock_executor_instance = MagicMock()
    # Force the executor to raise an exception
    error_instance = NemantixRuntimeException("Test error")
    mock_executor_instance.execute.side_effect = error_instance
    mock_executor_class.return_value = mock_executor_instance

    agent = Agent(expertise=MagicMock(), llm_proxy=MagicMock())

    # Execute
    exception, outputs = agent.run(user_request="Do a failing task")

    # Assert
    assert exception is error_instance
    assert outputs is None


# Test Knowledge Base configuration mismatches


@patch("nemantix.core.agent.Executor")
def test_agent_warns_when_kb_config_provided_without_use_knowledge_base(
    mock_executor_class,
):
    """A kb_config passed alongside use_knowledge_base=False is silently ignored;
    this should be flagged with a warning so the misconfiguration isn't missed."""
    with patch("nemantix.core.agent.logger.warning") as mock_warning:
        agent = Agent(
            expertise=MagicMock(),
            llm_proxy=MagicMock(),
            use_knowledge_base=False,
            kb_config=MagicMock(),
        )

    assert agent.knowledge_base is None
    assert any(
        "kb_config" in call.args[0] and "use_knowledge_base" in call.args[0]
        for call in mock_warning.call_args_list
    )


@patch("nemantix.core.agent.Executor")
def test_agent_no_kb_warning_when_config_and_flag_are_consistent(mock_executor_class):
    """No kb-related warning should fire when kb_config and use_knowledge_base agree
    (both left at their defaults)."""
    with patch("nemantix.core.agent.logger.warning") as mock_warning:
        Agent(
            expertise=MagicMock(),
            llm_proxy=MagicMock(),
            use_knowledge_base=False,
            kb_config=None,
        )

    assert not any(
        "kb_config" in call.args[0] and "use_knowledge_base" in call.args[0]
        for call in mock_warning.call_args_list
    )


# Test Output Formatting
class TestOutputFormat:
    def test_run_with_dict(self, mock_agent):
        """Test that a valid dictionary bypasses the LLM and parses instantly."""
        # Arrange
        mock_agent.executor.execute.return_value = {
            "name": "TestTask",
            "status": "done",
        }

        # Act
        exception, result = mock_agent.run(user_request="Do task", schema=DummySchema)

        # Assert
        assert exception is None
        assert isinstance(result, DummySchema)
        assert result.name == "TestTask"
        assert result.status == "done"

    def test_run_with_json_string(self, mock_agent):
        """Test that a valid JSON string bypasses the LLM and parses instantly."""
        # Arrange
        raw_json = json.dumps({"name": "StringTask", "status": "pending"})
        mock_agent.executor.execute.return_value = raw_json

        # Act
        exception, result = mock_agent.run(user_request="Do task", schema=DummySchema)

        # Assert
        assert exception is None
        assert isinstance(result, DummySchema)
        assert result.name == "StringTask"

    def test_run_with_nemantix_struct(self, mock_agent):
        """Test that a Struct with valid kwargs bypasses the LLM."""
        # Arrange
        struct_output = Struct(name="StructTask", status="active")
        mock_agent.executor.execute.return_value = struct_output

        # Act
        exception, result = mock_agent.run(
            user_request="Do struct task", schema=DummySchema
        )

        # Assert
        assert exception is None
        assert isinstance(result, DummySchema)
        assert result.name == "StructTask"

    def test_run_with_primitive_int(self, mock_agent):
        """Test that returning a primitive integer works with RootModel."""
        # Arrange
        mock_agent.executor.execute.return_value = 42

        # Act
        exception, result = mock_agent.run(
            user_request="Count items", schema=RootModel[int]
        )

        # Assert
        assert exception is None
        assert isinstance(result, RootModel)
        assert result.root == 42

    def test_run_primitive_bool(self, mock_agent):
        """Test that returning a primitive boolean works with RootModel."""
        # Arrange
        mock_agent.executor.execute.return_value = True

        # Act
        exception, result = mock_agent.run(
            user_request="Check flag", schema=RootModel[bool]
        )

        # Assert
        assert exception is None
        assert result.root is True

    @patch.object(Agent, "_parse_with_llm")
    def test_run_fallback_invalid_dict(self, mock_parse_with_llm, mock_agent):
        """Test that a dictionary missing required fields triggers the LLM fallback."""
        # Arrange
        mock_agent.executor.execute.return_value = {"name": "BrokenTask"}

        expected_model = DummySchema(
            name="BrokenTask", status="unknown_inferred_by_llm"
        )
        mock_parse_with_llm.return_value = expected_model

        # Act
        exception, result = mock_agent.run(
            user_request="Parse broken dict", schema=DummySchema
        )

        # Assert
        assert exception is None
        # Ensure it fell back to the LLM because of the ValidationError
        mock_parse_with_llm.assert_called_once()
        assert result.name == "BrokenTask"
