"""Clip production end to end on the GPU (synthetic 300 s chunk: radio 100-160 s, cutscene 170-230 s)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from clipper import ffmpeg, paths
from clipper.env import subprocess_env
from clipper.models import Candidate, Chunk, ClipMeta
from clipper.production import produce, produce_clip

pytestmark = pytest.mark.gpu

CHORD = (440.0, 554.37, 659.25)       # synthetic radio
CUTSCENE_HZ = 220.0                   # synthetic cutscene score
TONES = {"a_tone.wav": 1000.0, "b_tone.m4a": 300.0}
CPU_VIDEO_ENCODERS = ("libx264", "libx265", "libopenh264", "libvpx", "libaom", "libsvtav1", "mpeg4",
                      "h264_mf", "h264_qsv", "h264_amf", "libxvid")


# ------------------------------------------------------------------ fixtures / helpers
@pytest.fixture(scope="module")
def music_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("music")
    for name, hz in TONES.items():
        dur = 8 if name.endswith(".wav") else 90      # one short (loops), one long (offset seek)
        ffmpeg.run(["-f", "lavfi", "-i", f"sine=frequency={hz}:duration={dur}:sample_rate=48000",
                    "-ac", "2", str(d / name)])
    return d


@pytest.fixture
def with_music(monkeypatch, music_dir):
    monkeypatch.setattr(paths, "MUSIC", music_dir)
    return music_dir


@pytest.fixture
def no_music(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "MUSIC", tmp_path / "empty_music")
    return tmp_path / "empty_music"


@pytest.fixture
def gcfg(cfg):
    cfg.production.gpu_max_wait_s = 0          # a test must fail fast, not wait for the GPU forever
    return cfg


@pytest.fixture
def chunk(synthetic_chunk):
    path, _truth = synthetic_chunk
    return Chunk(0, str(path), start_s=0.0, duration_s=300.0, stream_id="test-stream")


@pytest.fixture(scope="module")
def split_chunks(synthetic_chunk, tmp_path_factory):
    """The synthetic chunk split at 150 s into two TS files (stream copy), as the recorder would."""
    src, _ = synthetic_chunk
    d = tmp_path_factory.mktemp("split")
    a, b = d / "chunk_00000.ts", d / "chunk_00001.ts"
    ffmpeg.run(["-i", str(src), "-t", "150", "-map", "0", "-c", "copy", "-f", "mpegts", str(a)])
    ffmpeg.run(["-ss", "150", "-i", str(src), "-map", "0", "-c", "copy", "-f", "mpegts", str(b)])
    da = ffmpeg.duration(a)
    return [Chunk(0, str(a), 0.0, da, stream_id="test-stream"),
            Chunk(1, str(b), da, ffmpeg.duration(b), stream_id="test-stream")]


@pytest.fixture
def commands(monkeypatch):
    """Record every ffmpeg arg list the production code runs."""
    seen: list[list[str]] = []
    real = ffmpeg.run_gpu

    def spy(args, **kw):
        seen.append([str(a) for a in args])
        return real(args, **kw)
    monkeypatch.setattr(ffmpeg, "run_gpu", spy)
    return seen


def cand(start, end, *, cid="test-stream-0-1", route="publish", hook="5 stars in 30 seconds",
         radio=(), track="gameplay", scene_tag=None) -> Candidate:
    return Candidate(id=cid, chunk_idx=0, start_s=start, peak_s=(start + end) / 2, end_s=end, track=track,
                     final_score=80.0, signal_score=70.0, route=route, radio_ranges=list(radio),
                     scene_tag=scene_tag,
                     meta=ClipMeta(title="GTA 6 test", description="d", hashtags=["#Shorts"], hook=hook))


def vprobe(path) -> dict:
    info = ffmpeg.probe(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    return {"w": v["width"], "h": v["height"], "vcodec": v["codec_name"], "acodec": a["codec_name"],
            "sr": int(a["sample_rate"]), "ch": a["channels"], "dur": float(info["format"]["duration"]),
            "sar": v.get("sample_aspect_ratio", "1:1")}


def audio(path, sr=8000) -> np.ndarray:
    """Decode the clip's audio to mono float32 (test-side analysis)."""
    out = subprocess.run([ffmpeg.FFMPEG, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True, env=subprocess_env())
    return np.frombuffer(out.stdout, dtype=np.float32)


def band(x: np.ndarray, sr: int, t0: float, t1: float, hz: float) -> float:
    seg = x[int(t0 * sr):int(t1 * sr)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) / len(seg)
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    i = int(np.argmin(np.abs(f - hz)))
    return float(spec[max(0, i - 3):i + 4].max())


def rms(x: np.ndarray, sr: int, t0: float, t1: float) -> float:
    seg = x[int(t0 * sr):int(t1 * sr)]
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))


def assert_short(path: Path, dur: float):
    p = vprobe(path)
    assert (p["w"], p["h"], p["vcodec"]) == (1080, 1920, "h264"), p
    assert (p["acodec"], p["sr"], p["ch"]) == ("aac", 48000, 2), p
    assert abs(p["dur"] - dur) < 0.5, p
    assert p["sar"] in ("1:1", "0:1"), p
    return p


# ------------------------------------------------------------------ tests
def test_single_chunk_clip(gcfg, chunk, with_music, commands, isolated_data):
    ticks = []
    c = cand(20, 32)
    sc = produce_clip(c, [chunk], gcfg, on_tick=lambda: ticks.append(1))
    out = paths.DATA.ready / f"{c.id}.mp4"
    assert Path(sc.clip_path) == out and out.exists()
    assert_short(out, 12)
    side = json.loads((paths.DATA.ready / f"{c.id}.json").read_text(encoding="utf-8"))
    assert side["clip_id"] == c.id and side["route"] == "publish" and side["hook"] == c.meta.hook
    assert side["candidate"]["id"] == c.id and side["created_at"] > 0 and side["music_track"] is None
    assert abs(side["duration_s"] - 12) < 0.5
    assert ticks, "on_tick never called"
    assert not list(paths.DATA.backup.iterdir())
    assert not list(paths.DATA.work.iterdir()), "work files left behind"
    ev = [json.loads(l) for l in (paths.DATA.logs / "clips.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["kind"] == "clip_produced" and e["clip_id"] == c.id for e in ev)
    cmd = " ".join(commands[0])
    assert "overlay_cuda=0:220[bgh]" in cmd and "-loop" not in commands[0]   # hook overlaid, single frame
    # frame check artefact for a human/agent look (.tmp/production_frame_check_00_6.0.jpg)
    ffmpeg.grab_frames(out, [6.0], paths.TMP, width=1080, prefix="production_frame_check")


def test_cross_chunk_clip_concat(gcfg, split_chunks, with_music, commands):
    a, b = split_chunks
    c = cand(140, 170, cid="test-stream-0-cross", hook="")
    sc = produce_clip(c, [b, a], gcfg)
    assert_short(Path(sc.clip_path), 30)
    cmd = commands[0]
    assert cmd.count("concat") == 2 and cmd.count("-ss") == 2      # both video inputs, same source/seek
    i = cmd.index("-ss")
    assert abs(float(cmd[i + 1]) - 140.0) < 0.01
    # seek accuracy across the join: radio chord until stream 160 s (clip 20 s), cutscene after 170 s
    x = audio(sc.clip_path)
    assert band(x, 8000, 2, 18, 440) > 10 * band(x, 8000, 22, 29, 440)


def test_empty_hook_backup_route(gcfg, chunk, no_music, commands):
    c = cand(30, 40, cid="test-stream-0-nohook", route="backup", hook="  ")
    sc = produce_clip(c, [chunk], gcfg)
    assert Path(sc.clip_path).parent == paths.DATA.backup
    assert (paths.DATA.backup / f"{c.id}.json").exists()
    assert not list(paths.DATA.ready.iterdir())
    assert_short(Path(sc.clip_path), 10)
    g = commands[0][commands[0].index("-filter_complex") + 1]
    assert "hwupload_cuda" not in g and "[bg][fg]overlay_cuda" in g
    assert not any(a.endswith(".png") for a in commands[0])


def test_radio_replaced_by_music(gcfg, chunk, with_music, commands):
    c = cand(95, 125, cid="test-stream-0-radio", radio=[(100, 160)])
    sc = produce_clip(c, [chunk], gcfg)
    assert_short(Path(sc.clip_path), 30)
    assert sc.music_track in TONES
    tone = TONES[sc.music_track]
    g = commands[0][commands[0].index("-filter_complex") + 1]
    assert "amix" in g and "eval=frame" in g
    x = audio(sc.clip_path)
    # inside the radio range (clip 5..30 s): music dominates, the chord is gone
    music_e = band(x, 8000, 7, 29, tone)
    for hz in CHORD:
        assert band(x, 8000, 7, 29, hz) < 0.05 * music_e, (hz, music_e)
    # before the fade (clip 0..3.5 s): no music yet
    assert band(x, 8000, 0, 3.5, tone) < 0.05 * music_e


def test_radio_muted_without_music(gcfg, chunk, no_music):
    c = cand(95, 125, cid="test-stream-0-mute", radio=[(100, 160)])
    sc = produce_clip(c, [chunk], gcfg)
    assert sc.music_track is None
    x = audio(sc.clip_path)
    assert rms(x, 8000, 7, 29) < 1e-3                       # muted
    assert rms(x, 8000, 0, 3.5) > 1e-3                      # game audio (noise floor) kept before it


def test_ducked_cutscene(gcfg, chunk, with_music):
    normal = produce_clip(cand(178, 198, cid="test-stream-0-cs", track="cutscene", scene_tag="heist_intro"),
                          [chunk], gcfg)
    assert normal.music_track is None
    gcfg.production.duck_scene_types = ["heist_intro"]
    ducked = produce_clip(cand(178, 198, cid="test-stream-0-csduck", track="cutscene",
                               scene_tag="Heist_Intro"), [chunk], gcfg)
    assert ducked.music_track in TONES
    xn, xd = audio(normal.clip_path), audio(ducked.clip_path)
    ratio = band(xd, 8000, 3, 17, CUTSCENE_HZ) / band(xn, 8000, 3, 17, CUTSCENE_HZ)
    assert 0.15 < ratio < 0.4, ratio                        # game audio ducked to ~duck_volume (0.25)
    tone = TONES[ducked.music_track]
    assert band(xd, 8000, 3, 17, tone) > 5 * band(xn, 8000, 3, 17, tone)   # music bed under it


def test_atomic_output_on_failure(gcfg, chunk, with_music, monkeypatch):
    real = ffmpeg.run_gpu

    def half_then_fail(args, **kw):
        Path(args[-1]).write_bytes(b"\x00" * 1000)          # partial file, then the "crash"
        raise ffmpeg.FfmpegError("simulated crash mid-render")
    monkeypatch.setattr(ffmpeg, "run_gpu", half_then_fail)
    c = cand(20, 30, cid="test-stream-0-fail", radio=[(22, 25)])
    with pytest.raises(ffmpeg.FfmpegError):
        produce_clip(c, [chunk], gcfg)
    assert not list(paths.DATA.ready.iterdir()) and not list(paths.DATA.backup.iterdir())
    assert not list(paths.DATA.work.iterdir())

    # a render that comes out wrong (validation fails) also leaves nothing behind
    monkeypatch.setattr(ffmpeg, "run_gpu", real)
    monkeypatch.setattr(produce, "_probe_video", lambda p: {"width": 1080, "height": 1080, "coded_width": 1080,
                                                            "codec": "h264", "duration": 10.0})
    with pytest.raises(produce.ProductionError):
        produce_clip(c, [chunk], gcfg)
    assert not list(paths.DATA.ready.iterdir()) and not list(paths.DATA.work.iterdir())


def test_width_fix_remux(gcfg, chunk, no_music, commands):
    gcfg.production.crop_right_px = 0                       # no bsf crop -> coded 1088 wide -> remux path
    sc = produce_clip(cand(50, 58, cid="test-stream-0-remux"), [chunk], gcfg)
    assert_short(Path(sc.clip_path), 8)
    if len(commands) == 2:                                  # only if this build really outputs 1088
        assert "copy" in commands[1] and any("crop_right=8" in a for a in commands[1])


def test_never_cpu_encoder(gcfg, chunk, with_music, commands):
    gcfg.production.duck_scene_types = ["x"]
    produce_clip(cand(100, 110, cid="test-stream-0-gpu", radio=[(104, 108)], track="cutscene", scene_tag="x"),
                 [chunk], gcfg)
    assert commands
    for cmd in commands:
        low = [a.lower() for a in cmd]
        for enc in CPU_VIDEO_ENCODERS:
            assert enc not in low, (enc, cmd)
        for i, a in enumerate(low):
            if a in ("-c:v", "-vcodec", "-codec:v"):
                assert low[i + 1] in ("h264_nvenc", "copy") or low[i + 1].endswith("_cuvid"), cmd
            if a == "-c":
                assert low[i + 1] == "copy"
        joined = " ".join(low)
        assert "hwdownload" not in joined and " -vf " not in f" {joined} "
        if "-filter_complex" in low:
            assert "h264_nvenc" in low and low.count("-hwaccel") == 2
