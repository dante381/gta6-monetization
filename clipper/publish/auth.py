"""YouTube OAuth: load/refresh the saved token, one-time browser sign-in, API client.

The token lives at paths.TOKEN (secrets/token.json) and the OAuth client at
paths.CLIENT_SECRET (client_secret.json in the project folder). Neither is ever logged.

Run once:  python -m clipper auth
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .. import paths

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

log = logging.getLogger("clipper.publish.auth")

FIX_STEPS = (
    "To fix: (1) in Google Cloud Console > Google Auth Platform > Audience, click 'Publish app' "
    "(in Testing mode the sign-in expires after 7 days), then (2) run "
    "`python -m clipper auth` once on the laptop and sign in."
)


class AuthRequired(RuntimeError):
    """No usable YouTube token: a human has to sign in again."""


def _token_path() -> Path:
    return Path(paths.TOKEN)


def _client_secret_path() -> Path:
    return Path(paths.CLIENT_SECRET)


def save_token(creds, path: Path | None = None) -> None:
    """Write the token atomically (temp file in the same folder, then replace)."""
    path = Path(path or _token_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _interactive_flow():
    from google_auth_oauthlib.flow import InstalledAppFlow

    secret = _client_secret_path()
    if not secret.exists():
        raise AuthRequired(
            f"OAuth client file not found at {secret}. In Google Cloud Console create a Desktop "
            "OAuth client, download it and save it there as client_secret.json.")
    flow = InstalledAppFlow.from_client_secrets_file(str(secret), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)
    save_token(creds)
    return creds


def get_credentials(interactive: bool = False):
    """Return valid google.oauth2 Credentials for the youtube.upload scope.

    Loads paths.TOKEN and refreshes it when expired (saving it back). Raises AuthRequired when
    the token is missing, unreadable, revoked or expired for good (RefreshError, e.g. the 7-day
    expiry in Testing mode). `interactive=True` runs the browser sign-in instead.
    Network errors during a refresh are NOT AuthRequired; they propagate so the caller retries.
    """
    if interactive:
        return _interactive_flow()

    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token = _token_path()
    if not token.exists():
        raise AuthRequired(f"YouTube sign-in needed: no saved token at {token}. {FIX_STEPS}")
    try:
        creds = Credentials.from_authorized_user_file(str(token), SCOPES)
    except (ValueError, KeyError, OSError) as e:
        raise AuthRequired(f"YouTube sign-in needed: the saved token can't be read "
                           f"({type(e).__name__}). {FIX_STEPS}") from e

    if creds.valid:
        return creds
    if not creds.refresh_token:
        raise AuthRequired(f"YouTube sign-in needed: the saved token has no refresh token. {FIX_STEPS}")
    try:
        creds.refresh(Request())
    except RefreshError as e:
        raise AuthRequired(f"YouTube sign-in expired or was revoked ({e.__class__.__name__}). "
                           f"{FIX_STEPS}") from e
    save_token(creds)
    return creds


def build_youtube(creds):
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


INSTRUCTIONS = """\
YouTube sign-in for GTA 6 Stream Auto Clipper
=============================================
A browser window opens. Sign in with the Google account that owns your YouTube channel.

Because the app isn't verified by Google, you'll see "Google hasn't verified this app":
  1. Click "Advanced".
  2. Click "Go to <app name> (unsafe)". It's your own app, so this is expected.
  3. Allow "Manage your YouTube videos" (upload only), then close the tab when it says done.

The token is saved only on this laptop, in secrets/token.json.
Tip: publish the app first (Google Auth Platform > Audience > Publish app); in Testing mode
the sign-in stops working after 7 days.
"""


def _clear_auth_hold() -> None:
    """After a good sign-in, lift a publish hold that was set only because auth failed."""
    try:
        from ..state import StateDB
        db = StateDB()
        try:
            if db.publish_hold() and db.kv_get("hold.publish_reason") == "auth":
                db.set_publish_hold(False, "")
                print("Publishing hold (auth) cleared.")
        finally:
            db.close()
    except Exception as e:  # state is optional here
        log.warning("could not clear the auth publish hold: %s", e)


def main(argv: list[str] | None = None) -> int:
    print(INSTRUCTIONS)
    if not _client_secret_path().exists():
        print(f"ERROR: {_client_secret_path()} not found. Create a Desktop OAuth client in Google "
              "Cloud Console and save its JSON there as client_secret.json.")
        return 1
    try:
        get_credentials(interactive=True)
    except AuthRequired as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: sign-in failed: {type(e).__name__}: {e}")
        return 1
    print(f"Signed in. Token saved to {_token_path()}.")
    _clear_auth_hold()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
