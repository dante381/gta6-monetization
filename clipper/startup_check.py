"""Pre-stream checks: tools, GPU encode/decode (no CPU path), Claude login, yt-dlp, disk, secrets, music.

`python -m clipper check` prints a table; exit code 1 if a blocking check fails.
GPU failures wait and retry (the spec: "the job waits and retries; it never switches to the CPU").
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass

from . import config as config_mod
from . import ffmpeg, paths
from .env import subprocess_env

log = logging.getLogger("check")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    blocking: bool = True


def _run(cmd, timeout=120) -> subprocess.CompletedProcess:
    return subprocess.run(list(map(str, cmd)), capture_output=True, text=True, env=subprocess_env(),
                          creationflags=NO_WINDOW, timeout=timeout, encoding="utf-8", errors="replace")


def check_tools() -> list[Check]:
    out = []
    for name, p in [("ffmpeg", paths.FFMPEG_DIR / "ffmpeg.exe"), ("ffprobe", paths.FFMPEG_DIR / "ffprobe.exe"),
                    ("deno", paths.DENO_DIR / "deno.exe"), ("yt-dlp", paths.VENV / "Scripts" / "yt-dlp.exe")]:
        out.append(Check(f"tool:{name}", p.exists(), str(p)))
    return out


def check_ffmpeg_features() -> Check:
    try:
        enc = _run([ffmpeg.FFMPEG, "-hide_banner", "-encoders"]).stdout
        dec = _run([ffmpeg.FFMPEG, "-hide_banner", "-decoders"]).stdout
        flt = _run([ffmpeg.FFMPEG, "-hide_banner", "-filters"]).stdout
    except Exception as e:  # noqa: BLE001
        return Check("ffmpeg:features", False, str(e))
    need = {"h264_nvenc": enc, "h264_cuvid": dec, "scale_cuda": flt, "overlay_cuda": flt, "hwupload_cuda": flt}
    missing = [k for k, v in need.items() if k not in v]
    return Check("ffmpeg:features", not missing, "missing: " + ", ".join(missing) if missing else "nvenc/cuvid/cuda filters ok")


def check_gpu(max_wait_s: float = 0.0) -> list[Check]:
    """NVENC encode + NVDEC decode test. Waits/retries up to max_wait_s on GPU errors."""
    paths.TMP.mkdir(parents=True, exist_ok=True)
    test = paths.TMP / "gpu_check.mp4"
    res = []
    try:
        ffmpeg.run_gpu(["-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=2", "-c:v", "h264_nvenc",
                        "-preset", "p5", str(test)], max_wait_s=max_wait_s, retry_wait_s=15, timeout=60)
        res.append(Check("gpu:nvenc", True, "h264_nvenc encode ok"))
    except ffmpeg.FfmpegError as e:
        res.append(Check("gpu:nvenc", False, str(e)[:300]))
        return res
    try:
        ffmpeg.run_gpu([*ffmpeg.GPU_DECODE, "-i", str(test), "-vf", "scale_cuda=320:180", "-f", "null", "-"],
                       max_wait_s=max_wait_s, retry_wait_s=15, timeout=60)
        res.append(Check("gpu:nvdec", True, "cuda decode + scale_cuda ok"))
    except ffmpeg.FfmpegError as e:
        res.append(Check("gpu:nvdec", False, str(e)[:300]))
    test.unlink(missing_ok=True)
    try:
        smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total",
                    "--format=csv,noheader"], timeout=20).stdout.strip()
        res.append(Check("gpu:info", True, smi, blocking=False))
    except Exception as e:  # noqa: BLE001
        res.append(Check("gpu:info", False, str(e), blocking=False))
    return res


def check_claude() -> Check:
    exe = shutil.which("claude")
    if not exe:
        return Check("claude:login", False, "claude CLI not on PATH")
    try:
        r = _run(["claude", "auth", "status"], timeout=60)
        info = json.loads(r.stdout or "{}")
    except Exception as e:  # noqa: BLE001
        return Check("claude:login", False, f"auth status failed: {e}")
    ok = bool(info.get("loggedIn"))
    detail = f"logged in ({info.get('authMethod')}, plan {info.get('planTier') or info.get('subscriptionType')})" \
        if ok else "not logged in - run `claude` once and /login"
    exp = info.get("loginExpiresAt")
    if exp:
        detail += f", login expires {exp[:10]}"
    return Check("claude:login", ok, detail)


def update_ytdlp() -> Check:
    try:
        r = _run([paths.VENV_PY, "-m", "pip", "install", "-U", "--disable-pip-version-check", "-q",
                  "yt-dlp[default]"], timeout=300)
        v = _run([paths.VENV / "Scripts" / "yt-dlp.exe", "--version"], timeout=60).stdout.strip()
        return Check("yt-dlp:update", r.returncode == 0, f"version {v}" + ("" if r.returncode == 0 else
                                                                          f" (update failed: {r.stderr[-200:]})"),
                     blocking=False)
    except Exception as e:  # noqa: BLE001
        return Check("yt-dlp:update", False, str(e), blocking=False)


def check_disk(min_free_gb: float = 20.0) -> Check:
    free = shutil.disk_usage(paths.ROOT).free / 2**30
    return Check("disk:free", free >= min_free_gb, f"{free:.0f} GB free on project drive")


def check_publishing(cfg) -> list[Check]:
    test_mode = bool(cfg.locked.test_mode)
    out = [Check("mode", True, "TEST MODE: no uploads" if test_mode else "LIVE MODE: uploads enabled", blocking=False)]
    out.append(Check("youtube:client_secret", paths.CLIENT_SECRET.exists(),
                     "client_secret.json present" if paths.CLIENT_SECRET.exists() else
                     "client_secret.json missing in project root", blocking=not test_mode))
    out.append(Check("youtube:token", paths.TOKEN.exists(),
                     "token present" if paths.TOKEN.exists() else "no token - run `python -m clipper auth`",
                     blocking=not test_mode))
    if config_mod.paused(cfg):
        out.append(Check("kill_switch", False, "PAUSE file present: uploads stopped", blocking=False))
    return out


def check_assets() -> list[Check]:
    music = [p for p in paths.MUSIC.glob("*") if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg", ".flac")]
    huds = [p for p in paths.HUD_REFS.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    return [Check("music", bool(music), f"{len(music)} royalty-free tracks" if music else
                  "music/ is empty: radio sections will be muted", blocking=False),
            Check("hud_refs", True, f"{len(huds)} HUD reference frames", blocking=False)]


def run_all(cfg=None, gpu_wait_s: float = 0.0, update: bool = True) -> list[Check]:
    cfg = cfg or config_mod.load_config()
    checks: list[Check] = []
    checks += check_tools()
    checks.append(check_ffmpeg_features())
    checks += check_gpu(gpu_wait_s)
    checks.append(check_claude())
    if update:
        checks.append(update_ytdlp())
    checks.append(check_disk())
    checks += check_publishing(cfg)
    checks += check_assets()
    return checks


def print_checks(checks: list[Check]) -> bool:
    ok = True
    for c in checks:
        mark = "OK  " if c.ok else ("FAIL" if c.blocking else "WARN")
        if not c.ok and c.blocking:
            ok = False
        print(f"[{mark}] {c.name:24s} {c.detail}")
    print("\nAll blocking checks passed." if ok else "\nBlocking checks FAILED.")
    return ok


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="clipper check")
    ap.add_argument("--no-update", action="store_true", help="skip yt-dlp update")
    ap.add_argument("--gpu-wait", type=float, default=0.0, help="seconds to keep retrying GPU checks")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    ok = print_checks(run_all(gpu_wait_s=a.gpu_wait, update=not a.no_update))
    print(f"({time.monotonic() - t0:.0f}s)")
    return 0 if ok else 1
