"""YouTube live stream discovery via yt-dlp.

- live_status(url, cfg)   -> {"status", "video_id", "title"}
- resolve_hls(url, cfg)   -> [video_url, audio_url] or [muxed_url]
- probe(url, cfg)         -> both in ONE yt-dlp call (the recorder uses this on reconnect)
- wait_until_live(url, cfg, hb) polls every cfg.stream.live_poll_s

The yt-dlp subprocess runner is injectable (`runner=`) so tests never touch the network. A runner
is `runner(cmd: list[str], *, timeout: float, on_tick: callable|None) -> (returncode, stdout, stderr)`.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any, Callable, Protocol

from .. import paths
from ..env import subprocess_env

log = logging.getLogger(__name__)

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# Prefer HLS (m3u8): ffmpeg records it directly with -c copy. DASH-only streams (some big/DVR
# broadcasts) can't be recorded that way, so they are only a last-resort fallback and resolve_hls
# turns them into a clear StreamError.
DEFAULT_FORMAT = ("bv*[height<=720][protocol^=m3u8]+ba[protocol^=m3u8]/b[height<=720][protocol^=m3u8]"
                  "/bv*[height<=720]+ba/b[height<=720]")
STATUSES = ("is_live", "is_upcoming", "was_live", "not_live", "post_live", "error")
INFO_TEMPLATE = "%(.{id,title,live_status,is_live,protocol})j"


class StreamError(RuntimeError):
    """yt-dlp could not resolve playable URLs."""


class Runner(Protocol):
    def __call__(self, cmd: list[str], *, timeout: float,
                 on_tick: Callable[[], None] | None) -> tuple[int, str, str]: ...


def _get(cfg: Any, section: str, key: str, default: Any) -> Any:
    try:
        v = cfg[section].get(key, default)
    except (KeyError, AttributeError, TypeError):
        return default
    return default if v in (None, "") else v


def subprocess_runner(cmd: list[str], *, timeout: float,
                      on_tick: Callable[[], None] | None = None) -> tuple[int, str, str]:
    """Default runner: real subprocess, polled every <=5 s so the caller's heartbeat keeps ticking."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                            env=subprocess_env(), creationflags=NO_WINDOW)
    deadline = time.monotonic() + timeout
    while True:
        try:
            out, err = proc.communicate(timeout=5.0)
            break
        except subprocess.TimeoutExpired:
            if on_tick:
                on_tick()
            if time.monotonic() > deadline:
                from ..ffmpeg import kill_tree
                kill_tree(proc)
                try:
                    out, err = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    out, err = b"", b""
                return -9, out.decode("utf-8", "replace"), f"yt-dlp timed out after {timeout:.0f}s\n" + \
                    err.decode("utf-8", "replace")
    return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


DEFAULT_RUNNER: Runner = subprocess_runner


def ytdlp_base() -> list[str]:
    return [str(paths.VENV / "Scripts" / "yt-dlp.exe"), "--cache-dir", str(paths.CACHE / "yt-dlp"),
            "--no-warnings"]


def _classify_error(stderr: str) -> str:
    s = stderr.lower()
    if "not currently live" in s or "is not live" in s:
        return "not_live"
    if "will begin" in s or "premieres in" in s or "upcoming" in s:
        return "is_upcoming"
    if "has ended" in s or "was live" in s or "recording is not available" in s:
        return "was_live"
    return "error"


def _status_of(info: dict) -> str:
    st = info.get("live_status")
    if st in STATUSES:
        return st
    if info.get("is_live") is True:
        return "is_live"
    return "not_live"


def probe(url: str, cfg: Any, *, want_urls: bool = True, runner: Runner | None = None,
          on_tick: Callable[[], None] | None = None) -> dict:
    """One yt-dlp call. Returns {"status", "video_id", "title", "urls", "error"}.

    Never raises: failures give status "error" (or not_live/is_upcoming parsed from stderr).
    """
    runner = runner or DEFAULT_RUNNER
    fmt = _get(cfg, "stream", "format", DEFAULT_FORMAT)
    timeout = float(_get(cfg, "stream", "ytdlp_timeout_s", 180))
    cmd = [*ytdlp_base(), "--ignore-no-formats-error", "--no-playlist", "-f", fmt, "--print", INFO_TEMPLATE]
    if want_urls:
        cmd += ["--print", "urls"]
    cmd.append(url)
    res = {"status": "error", "video_id": None, "title": None, "urls": [], "error": ""}
    try:
        rc, out, err = runner(cmd, timeout=timeout, on_tick=on_tick)
    except Exception as e:  # runner itself failed (exe missing, ...)
        res["error"] = f"{type(e).__name__}: {e}"
        return res
    info: dict | None = None
    urls: list[str] = []
    for line in (out or "").splitlines():
        line = line.strip()
        if not line or line == "NA":
            continue
        if info is None and line.startswith("{"):
            try:
                info = json.loads(line)
                continue
            except json.JSONDecodeError:
                pass
        if info is not None:
            urls.append(line)
    if info is None:
        res["status"] = _classify_error(err or "")
        res["error"] = (err or "").strip()[-1000:] or f"yt-dlp exit {rc}, no info"
        return res
    res.update(status=_status_of(info), video_id=info.get("id"), title=info.get("title"), urls=urls,
               protocol=info.get("protocol"))
    if rc != 0:
        res["error"] = (err or "").strip()[-1000:]
    return res


def live_status(url: str, cfg: Any, *, runner: Runner | None = None,
                on_tick: Callable[[], None] | None = None) -> dict:
    """{"status": is_live|is_upcoming|was_live|not_live|post_live|error, "video_id", "title"}."""
    r = probe(url, cfg, want_urls=False, runner=runner, on_tick=on_tick)
    return {"status": r["status"], "video_id": r["video_id"], "title": r["title"]}


def resolve_hls(url: str, cfg: Any, *, runner: Runner | None = None,
                on_tick: Callable[[], None] | None = None) -> list[str]:
    """1 (muxed) or 2 (video, audio) playable URLs. Raises StreamError when nothing is playable."""
    r = probe(url, cfg, want_urls=True, runner=runner, on_tick=on_tick)
    if not r["urls"]:
        raise StreamError(f"no playable URLs (status={r['status']}): {r['error'][-300:]}")
    proto = str(r.get("protocol") or "").lower()
    if "dash" in proto:
        raise StreamError(f"this stream is only offered as DASH (protocol {proto}); ffmpeg cannot record it "
                          f"with -c copy. HLS (m3u8) is required, which is what a normal PS5 live stream gives.")
    return r["urls"][:2]


def sleep_ticking(seconds: float, hb: Any = None, sleep: Callable[[float], None] = time.sleep) -> None:
    """Sleep in <=5 s steps, ticking the heartbeat between steps."""
    left = max(0.0, float(seconds))
    while left > 0:
        if hb is not None:
            hb.tick()
        step = min(5.0, left)
        sleep(step)
        left -= step
    if hb is not None:
        hb.tick()


def wait_until_live(url: str, cfg: Any, hb: Any = None, *, runner: Runner | None = None,
                    sleep: Callable[[float], None] = time.sleep, max_checks: int | None = None) -> dict:
    """Block until `url` is live; returns the live_status dict. Polls every cfg.stream.live_poll_s."""
    poll = float(_get(cfg, "stream", "live_poll_s", 60))
    on_tick = hb.tick if hb is not None else None
    last = None
    n = 0
    while True:
        st = live_status(url, cfg, runner=runner, on_tick=on_tick)
        if st["status"] == "is_live" and st["video_id"]:
            log.info("live: %s (%s)", st["video_id"], st["title"])
            return st
        if st["status"] != last:
            log.info("waiting for live (status=%s) %s", st["status"], url)
            last = st["status"]
        n += 1
        if max_checks is not None and n >= max_checks:
            return st
        sleep_ticking(poll, hb, sleep)
