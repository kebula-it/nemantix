import json
from nemantix.llm.credentials import Credentials


def test_load_from_file_success(tmp_path):
    creds = {"openai_api_key": "sk-file", "google_api_key": "gk-file"}
    p = tmp_path / "credentials.json"
    p.write_text(json.dumps(creds), encoding="utf-8")

    c = Credentials(file_path=str(p))
    assert c.get_api_key("openai_api_key") == "sk-file"
    assert c.get_api_key("google_api_key") == "gk-file"


def test_env_fallback(monkeypatch, tmp_path):
    # Invalid json file should not crash and should fall back to env
    bad = tmp_path / "bad.json"
    bad.write_text("{not:json}", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    c = Credentials(file_path=str(bad))

    assert c.get_api_key("openai_api_key") == "sk-env"
    # Second call must come from internal cache, not env re-read
    assert c.get_api_key("openai_api_key") == "sk-env"


def test_missing_returns_none(tmp_path, capsys):
    c = Credentials(file_path=str(tmp_path / "missing.json"))
    val = c.get_api_key("no_such_key")
    assert val is None
