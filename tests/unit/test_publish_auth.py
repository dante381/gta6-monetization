"""auth.get_credentials with fake token files (never the real token or client secret)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from clipper import paths
from clipper.publish import auth

FAKE_TOKEN = {"token": "fake-access", "refresh_token": "fake-refresh", "client_id": "fake.apps",
              "client_secret": "fake-secret", "token_uri": "https://oauth2.googleapis.com/token",
              "scopes": auth.SCOPES}


@pytest.fixture
def fake_paths(tmp_path, monkeypatch):
    tok = tmp_path / "secrets" / "token.json"
    sec = tmp_path / "client_secret.json"
    monkeypatch.setattr(paths, "TOKEN", tok)
    monkeypatch.setattr(paths, "CLIENT_SECRET", sec)
    return tok, sec


def _write_token(path, expired: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = dict(FAKE_TOKEN)
    exp = datetime.now(timezone.utc).replace(tzinfo=None) + (timedelta(hours=-1) if expired else timedelta(hours=1))
    d["expiry"] = exp.strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(d), encoding="utf-8")


def test_scope_is_upload_only():
    assert auth.SCOPES == ["https://www.googleapis.com/auth/youtube.upload"]


def test_missing_token_raises(fake_paths):
    with pytest.raises(auth.AuthRequired) as ei:
        auth.get_credentials()
    msg = str(ei.value)
    assert "python -m clipper auth" in msg and "Publish app" in msg


def test_valid_token_loaded(fake_paths):
    tok, _ = fake_paths
    _write_token(tok, expired=False)
    creds = auth.get_credentials()
    assert creds.valid and creds.refresh_token == "fake-refresh"


def test_refresh_saves_atomically(fake_paths, monkeypatch):
    from google.oauth2.credentials import Credentials
    tok, _ = fake_paths
    _write_token(tok, expired=True)

    def fake_refresh(self, request):
        self.token = "new-access"
        self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)
    monkeypatch.setattr(Credentials, "refresh", fake_refresh)
    creds = auth.get_credentials()
    assert creds.token == "new-access"
    assert json.loads(tok.read_text(encoding="utf-8"))["token"] == "new-access"
    assert not tok.with_name("token.json.tmp").exists()


def test_refresh_error_raises_auth_required(fake_paths, monkeypatch):
    from google.auth.exceptions import RefreshError
    from google.oauth2.credentials import Credentials
    tok, _ = fake_paths
    _write_token(tok, expired=True)

    def bad_refresh(self, request):
        raise RefreshError("invalid_grant: Token has been expired or revoked.")
    monkeypatch.setattr(Credentials, "refresh", bad_refresh)
    with pytest.raises(auth.AuthRequired):
        auth.get_credentials()


def test_interactive_flow_saves_token(fake_paths, monkeypatch):
    tok, sec = fake_paths
    sec.write_text("{}", encoding="utf-8")
    seen = {}

    class FakeCreds:
        def to_json(self):
            return json.dumps({"token": "t"})

    class FakeFlow:
        def run_local_server(self, port, open_browser):
            seen.update(port=port, open_browser=open_browser)
            return FakeCreds()

    from google_auth_oauthlib import flow as flow_mod

    def from_file(path, scopes):
        seen.update(path=path, scopes=scopes)
        return FakeFlow()
    monkeypatch.setattr(flow_mod.InstalledAppFlow, "from_client_secrets_file", staticmethod(from_file))
    auth.get_credentials(interactive=True)
    assert seen == {"path": str(sec), "scopes": auth.SCOPES, "port": 0, "open_browser": True}
    assert json.loads(tok.read_text(encoding="utf-8")) == {"token": "t"}


def test_main_without_client_secret(fake_paths, capsys):
    assert auth.main([]) == 1
    out = capsys.readouterr().out
    assert "Advanced" in out and "unsafe" in out
