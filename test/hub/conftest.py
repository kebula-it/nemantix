from pathlib import Path
from nemantix.common import context
from nemantix.hub import EventHub
import pytest


@pytest.fixture(scope="session")
def project_root():
    """Returns the root of the project based on the location of this file."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated_event_hub():
    """
    Automatically creates a fresh EventHub and binds it to the ContextVar
    for every single test. Cleans up afterward.
    """
    # 1. Create a fresh hub
    hub = EventHub()

    # 2. Bind it to the context for this specific test
    token = context.event_hub.set(hub)

    # 3. Yield the hub. Tests can optionally request this fixture by name
    # if they need to manually attach tools or inspect the hub.
    yield hub

    # 4. Teardown: Reset the context after the test finishes
    context.event_hub.reset(token)
