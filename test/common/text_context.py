from nemantix.common.context import event_hub


def test_event_hub_context_var():
    """Tests the default state and context isolation of the event_hub ContextVar."""
    # Default value should be None
    assert event_hub.get() is None

    # Set a mock value and keep the reset token
    token = event_hub.set("mock_hub_instance")
    assert event_hub.get() == "mock_hub_instance"

    # Reset back to the original state
    event_hub.reset(token)
    assert event_hub.get() is None
