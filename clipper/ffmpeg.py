"""ffmpeg / ffprobe runner.

GPU ONLY: video decode/scale/overlay/encode use CUDA/NVDEC/NVENC. If the GPU path fails with a
GPU-type error, GpuUnavailable is raised; `run_gpu` waits and retries. Nothing here ever swaps
in a CPU codec.
"""
from __future__ import annotations

import collections
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from . import paths
from .env import subprocess_env

log = logging.getLogger(__name__)

FFMPEG = str(paths.FFMPEG_DIR / "ffmpeg.exe")
FFPROBE = str(paths.FFMPEG_DIR / "ffprobe.exe")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Decode on the GPU, keep frames in GPU memory.
GPU_DECODE = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
# Decode on the GPU, frames delivered to system memory (for small analysis frames only).
GPU_DECODE_TO_RAM = ["-hwaccel", "cuda"]

GPU_ERROR_PATTERNS = (
    "no nvenc capable devices", "openencodesessionex failed", "cannot load nvcuda", "cannot load nvencodeapi",
    "cuda_error", "cuinit", "cuctxcreate", "failed setup for format cuda", "hwaccel initialisation returned error",
    "device creation failed", "no device available for decoder", "nvenc", "cuvid", "nvdec",
    "out of memory", "driver does not support the required nvenc api version",
)


class FfmpegError(RuntimeError):
    def __init__(self, msg: str, returncode: int | None = None, stderr: str = ""):
        super().__init__(msg)
        self.returncode = returncode
        self.stderr = stderr


class GpuUnavailable(FfmpegError):
    """GPU encoder/decoder not usable right now. Caller waits and retries; never falls back to CPU."""


class FfmpegTimeout(FfmpegError):
    pass


@dataclass
class FfResult:
    returncode: int
    stdout: bytes | None
    stderr: str


def ffmpeg_cmd(args: Sequence[str], loglevel: str = "error") -> list[str]:
    return [FFMPEG, "-hide_banner", "-nostdin", "-y", "-loglevel", loglevel, *map(str, args)]


def is_gpu_error(stderr: str) -> bool:
    s = stderr.lower()
    return any(p in s for p in GPU_ERROR_PATTERNS)


def popen(cmd: Sequence[str], **kw) -> subprocess.Popen:
    kw.setdefault("env", subprocess_env())
    kw.setdefault("creationflags", NO_WINDOW)
    return subprocess.Popen(list(map(str, cmd)), **kw)


def _drain(stream, sink, _keep_all: bool) -> None:
    """Read a pipe to EOF into `sink` (a list, or a bounded deque for stderr tails)."""
    try:
        for chunk in iter(lambda: stream.read(65536), b""):
            sink.append(chunk)
    finally:
        stream.close()


def run(args: Sequence[str], *, timeout: float | None = None, on_tick: Callable[[], None] | None = None,
        tick_s: float = 5.0, capture_stdout: bool = False, check: bool = True, loglevel: str = "error",
        raw_cmd: bool = False, stdin_data: bytes | None = None) -> FfResult:
    """Run ffmpeg with `args` (without the ffmpeg prefix unless raw_cmd=True).

    Polls every tick_s seconds and calls on_tick (heartbeat) while waiting.
    Raises GpuUnavailable / FfmpegError on failure when check=True, FfmpegTimeout on timeout.
    """
    cmd = list(map(str, args)) if raw_cmd else ffmpeg_cmd(args, loglevel)
    proc = popen(cmd, stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
                 stderr=subprocess.PIPE, stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL)
    err_tail: collections.deque[bytes] = collections.deque(maxlen=64)
    out_buf: list[bytes] = []
    threads = [threading.Thread(target=_drain, args=(proc.stderr, err_tail, False), daemon=True)]
    if capture_stdout:
        threads.append(threading.Thread(target=_drain, args=(proc.stdout, out_buf, True), daemon=True))
    for t in threads:
        t.start()
    if stdin_data is not None:
        try:
            proc.stdin.write(stdin_data)
        finally:
            proc.stdin.close()
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            proc.wait(timeout=tick_s)
            break
        except subprocess.TimeoutExpired:
            if on_tick:
                on_tick()
            if deadline is not None and time.monotonic() > deadline:
                kill_tree(proc)
                raise FfmpegTimeout(f"ffmpeg timed out after {timeout}s: {' '.join(cmd[:12])}")
    for t in threads:
        t.join(timeout=5)
    stderr = b"".join(err_tail).decode("utf-8", "replace")[-6000:]
    res = FfResult(proc.returncode, b"".join(out_buf) if capture_stdout else None, stderr)
    if check and proc.returncode != 0:
        msg = f"ffmpeg exited {proc.returncode}: {stderr.strip()[-800:]}"
        if is_gpu_error(stderr):
            raise GpuUnavailable(msg, proc.returncode, stderr)
        raise FfmpegError(msg, proc.returncode, stderr)
    return res


def run_gpu(args: Sequence[str], *, retry_wait_s: float = 30.0, max_wait_s: float | None = None,
            on_tick: Callable[[], None] | None = None, **kw) -> FfResult:
    """run() that waits and retries while the GPU is unavailable. Never switches to CPU."""
    started = time.monotonic()
    attempt = 0
    while True:
        try:
            return run(args, on_tick=on_tick, **kw)
        except GpuUnavailable as e:
            attempt += 1
            if max_wait_s is not None and time.monotonic() - started > max_wait_s:
                raise
            log.warning("GPU unavailable (attempt %d), waiting %.0fs: %s", attempt, retry_wait_s, str(e)[:300])
            end = time.monotonic() + retry_wait_s
            while time.monotonic() < end:
                time.sleep(min(5.0, max(0.0, end - time.monotonic())))
                if on_tick:
                    on_tick()


def grab_frames(src: str | Path, times: Sequence[float], out_dir: str | Path, width: int = 1280,
                prefix: str = "still", on_tick: Callable[[], None] | None = None) -> list[Path]:
    """Save one JPEG per time (seconds into `src`) using GPU decode + GPU scaling.

    Used for judge frames. Only the finished small frame is downloaded to RAM.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs: list[Path] = []
    for i, t in enumerate(times):
        p = out_dir / f"{prefix}_{i:02d}_{t:.1f}.jpg"
        run_gpu([*GPU_DECODE, "-ss", f"{max(0.0, t):.3f}", "-i", str(src), "-frames:v", "1",
                 "-vf", f"scale_cuda={width}:-2:format=yuv420p,hwdownload,format=yuv420p",
                 "-q:v", "3", str(p)], on_tick=on_tick, timeout=120)
        outs.append(p)
    return outs


def probe(path: str | Path) -> dict:
    cmd = [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    out = subprocess.run(cmd, capture_output=True, env=subprocess_env(), creationflags=NO_WINDOW, timeout=120)
    if out.returncode != 0:
        raise FfmpegError(f"ffprobe failed: {out.stderr.decode('utf-8', 'replace')[-500:]}", out.returncode)
    return json.loads(out.stdout.decode("utf-8", "replace") or "{}")


def duration(path: str | Path) -> float:
    info = probe(path)
    d = info.get("format", {}).get("duration")
    if d is None:
        for s in info.get("streams", []):
            if s.get("duration"):
                return float(s["duration"])
        return 0.0
    return float(d)


def kill_tree(proc: subprocess.Popen | int) -> None:
    import psutil
    pid = proc if isinstance(proc, int) else proc.pid
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for child in parent.children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.kill()
    except psutil.NoSuchProcess:
        pass
