from contextvars import ContextVar

event_hub: ContextVar = ContextVar("event_hub", default=None)
security_context: ContextVar = ContextVar("security_context", default=None)
