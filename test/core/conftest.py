import os
from pathlib import Path

import pytest

from nemantix.common import context
from nemantix.hub import EventHub
from nemantix.llm import LLMProxyConfig


class DummyProxyConfig(LLMProxyConfig):
    def __init__(self, dummy_llm, *_, **__):
        self.llm = dummy_llm

    def __getattr__(self, name):
        return self.llm


@pytest.fixture(scope="session")
def project_root():
    """Returns the root of the project based on the location of this file."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def grammar_path(project_root):
    """
    Locates the nxs_v2_grammar.lark file.
    Assumes standard directory structure: nemantix/core/nxs_v2_grammar.lark
    """
    # Adjust "src" depending on if your project uses a src layout or flat layout
    candidate = project_root / "src" / "nemantix" / "core" / "nxs_v2_grammar.lark"

    # Fallback for flat layout if needed
    if not candidate.exists():
        candidate = project_root / "core" / "nxs_v2_grammar.lark"

    if candidate.exists():
        return str(candidate)

    # Allow env override
    envp = os.environ.get("NXS_GRAMMAR")
    if envp and Path(envp).exists():
        return envp

    pytest.fail("nxs_v2_grammar.lark not found. Set NXS_GRAMMAR env var.")
    return None


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


@pytest.fixture(scope="session")
def dummy_llm_proxy_config_class():
    """Returns the DummyProxyConfig class so it can be instantiated in tests."""
    return DummyProxyConfig
