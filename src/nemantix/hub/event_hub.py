from abc import ABC, abstractmethod
from typing import Callable

from nemantix.common import context
from nemantix.hub.events import Event, EventType


class EventHub:
    def __init__(self):
        self._events = []
        self._events_by_type = {event_type: [] for event_type in EventType}

        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = {
            event_type: [] for event_type in EventType
        }

    @staticmethod
    def get_active_hub(event_type: EventType) -> "EventHub | None":
        """Returns an EventHub instance (of current context) if it exists and if it subscribes to
        the provided event type.
        """
        hub = context.event_hub.get()
        if hub is not None and hub.has_subscribers(event_type):
            return hub

        return None

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """Registers a callback to be triggered on a specific event type."""
        self._subscribers[event_type].append(callback)

    def has_subscribers(self, event_type: EventType) -> bool:
        return len(self._subscribers[event_type]) > 0

    def emit(self, event: Event):
        """Broadcasts the event to all registered callbacks for that event type."""
        self._events.append(event)
        self._events_by_type[event.type].append(event)

        for callback in self._subscribers[event.type]:
            callback(event)


def emit_json_parse(
    success: bool,
    source: str,
    *,
    error: str | None = None,
    scope: str = "",
    script=None,
    lines: tuple[int, int] = (0, 0),
    statement: str = "",
    **extra,
) -> None:
    """Emit a JSON_PARSE event from any layer.

    Layer-agnostic emitter: no-ops when there is no active hub or no subscriber
    for JSON_PARSE. `source` labels the parse site (e.g. "frame_apply",
    "request_inputs", "structured_output"); `extra` may carry
    `name` (the responsible LLM), `mode`, `repaired`, etc.
    """
    hub = context.event_hub.get()
    if hub is None or not hub.has_subscribers(EventType.JSON_PARSE):
        return

    hub.emit(
        Event(
            type=EventType.JSON_PARSE,
            lines=lines,
            scope=scope,
            script=script,
            statement=statement,
            payload=dict(success=success, source=source, error=error, **extra),
        )
    )


class Observable(ABC):
    @abstractmethod
    def subscribe(self, event_hub: EventHub):
        pass

    @staticmethod
    def get_script_location(event: Event) -> str | None:
        if event.script is not None:
            return event.script.get_location()

        return None
