from nemantix.llm.credentials import Credentials


def test_get_api_key_success(monkeypatch):
    """Test retrieving keys strictly from environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("GOOGLE_API_KEY", "gk-env")

    c = Credentials()
    assert c.get_api_key("openai_api_key") == "sk-env"
    assert c.get_api_key("google_api_key") == "gk-env"


def test_env_cache(monkeypatch):
    """Test that the credentials manager caches the value after the first retrieval."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    c = Credentials()

    assert c.get_api_key("openai_api_key") == "sk-env"

    # Modify env to ensure it reads from the internal cache, not re-reading os.getenv
    monkeypatch.setenv("OPENAI_API_KEY", "sk-changed")
    assert c.get_api_key("openai_api_key") == "sk-env"


def test_missing_returns_none(monkeypatch):
    """Test that a missing key safely returns None without raising."""
    # Ensure the environment variable is explicitly removed for the test
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)

    c = Credentials()
    val = c.get_api_key("no_such_key")
    assert val is None
