from nemantix.common import context
from nemantix.hub import EventType, emit_json_parse


def test_emit_json_parse_noop_without_subscriber(isolated_event_hub):
    """With no JSON_PARSE subscriber, emit_json_parse records nothing."""
    emit_json_parse(True, "frame_apply", name="gpt-4")
    assert isolated_event_hub._events_by_type[EventType.JSON_PARSE] == []


def test_emit_json_parse_emits_with_subscriber(isolated_event_hub):
    """With a subscriber, a JSON_PARSE Event is emitted with the expected payload."""
    captured = []
    isolated_event_hub.subscribe(EventType.JSON_PARSE, captured.append)

    emit_json_parse(
        False, "structured_output", error="boom", name="gpt-4", mode="strict"
    )

    assert len(captured) == 1
    event = captured[0]
    assert event.type is EventType.JSON_PARSE
    assert event.payload == dict(
        success=False,
        source="structured_output",
        error="boom",
        name="gpt-4",
        mode="strict",
    )


def test_emit_json_parse_noop_without_hub():
    """No active hub in context -> emit_json_parse is a safe no-op (does not raise)."""
    token = context.event_hub.set(None)
    try:
        emit_json_parse(True, "frame_apply")
    finally:
        context.event_hub.reset(token)
