from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable

from nemantix.common import context
from nemantix.hub.events import BaseEventType, Event, EventType


class EventHub:
    def __init__(self):
        self._events = []
        self._events_by_type: dict[BaseEventType, list[Event]] = defaultdict(list)
        self._subscribers: dict[
            BaseEventType, list[Callable[[Event], None]]
        ] = defaultdict(list)

    @staticmethod
    def get_active_hub(event_type: BaseEventType) -> "EventHub | None":
        """Returns an EventHub instance (of current context) if it exists and if it subscribes to
        the provided event type.
        """
        hub = context.event_hub.get()
        if hub is not None and hub.has_subscribers(event_type):
            return hub

        return None

    def subscribe(self, event_type: BaseEventType, callback: Callable[[Event], None]):
        """Registers a callback to be triggered on a specific event type."""
        self._subscribers[event_type].append(callback)

    def has_subscribers(self, event_type: BaseEventType) -> bool:
        return len(self._subscribers[event_type]) > 0

    def emit(self, event: Event):
        """Broadcasts the event to all registered callbacks for that event type."""
        self._events.append(event)
        self._events_by_type[event.type].append(event)

        for callback in self._subscribers[event.type]:
            callback(event)


class Observable(ABC):
    @abstractmethod
    def subscribe(self, event_hub: EventHub):
        pass

    @staticmethod
    def get_script_location(event: Event) -> str | None:
        if event.script is not None:
            return event.script.get_location()

        return None
