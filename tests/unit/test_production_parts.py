"""Clip production: pure parts (no GPU): hook PNG, audio graph, cut planning, geometry, music pick."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from clipper.models import Chunk
from clipper.production import audio_mix, cut, hooktext, layout, music


# ------------------------------------------------------------------ hook text
def test_hook_png_rendered_with_alpha(cfg, tmp_path):
    p = hooktext.render_hook("5 stars in 30 seconds", cfg, tmp_path / "hook.png")
    assert p is not None and p.exists()
    im = Image.open(p)
    assert im.mode == "RGBA" and im.width == 1080 and im.height % 2 == 0
    alpha = np.asarray(im)[:, :, 3]
    assert alpha.max() == 255 and alpha.min() == 0           # text drawn, background transparent
    assert im.getchannel("A").getbbox() is not None
    rgb = np.asarray(im)[:, :, :3][alpha == 255]
    assert (rgb.min(axis=1) > 240).any(), "white fill expected"
    assert (rgb.max(axis=1) < 15).any(), "black stroke expected"


def test_hook_long_text_wraps_and_shrinks(cfg, tmp_path):
    text = "He drove a stolen fire truck through the entire police blockade and somehow made it"
    lines, font, stroke = hooktext.layout_hook(hooktext.clean_text(text, 0), cfg)
    assert 1 <= len(lines) <= 2
    assert font.size <= cfg.production.hook_font_size
    p = hooktext.render_hook(text, cfg, tmp_path / "long.png")
    im = Image.open(p)
    bbox = im.getbbox()
    margin = cfg.production.get("hook_margin_px", 60)
    assert bbox[0] >= margin - 8 and bbox[2] <= 1080 - margin + 8   # within the side margins
    assert im.height < 656 - cfg.production.hook_top_px                 # fits above the fg


def test_hook_empty_or_emoji_only_skips(cfg, tmp_path):
    assert hooktext.render_hook("", cfg, tmp_path / "e.png") is None
    assert hooktext.render_hook("   \U0001F525\U0001F525 ", cfg, tmp_path / "e2.png") is None
    assert hooktext.clean_text("WASTED \U0001F480 again", 40) == "WASTED again"


def test_hook_font_fallback(cfg, tmp_path):
    cfg.production.font_path = "C:/nope/missing.ttf"
    assert hooktext.find_font(cfg.production.font_path).suffix.lower() == ".ttf"
    assert hooktext.render_hook("still works", cfg, tmp_path / "f.png") is not None


# ------------------------------------------------------------------ audio graph
def test_clip_ranges_relative_and_clamped():
    r = audio_mix.clip_ranges([(100, 160), (90, 102), (300, 310)], start_s=95, duration=40)
    assert r == [(0.0, 40.0)]
    assert audio_mix.clip_ranges([(10, 20)], start_s=30, duration=10) == []
    assert audio_mix.clip_ranges([(32, 35), (36, 38)], 30, 10) == [(2.0, 5.0), (6.0, 8.0)]


def _eval_weight(expr: str, t: float) -> float:
    py = expr.replace("clip(", "_clip(")
    return eval(py, {"_clip": lambda x, a, b: min(max(x, a), b), "t": t})


def test_radio_weight_expression_mutes_inside_range():
    e = audio_mix.radio_weight_expr([(5.0, 20.0)], 1.0)
    assert _eval_weight(e, 5.0) == 1 and _eval_weight(e, 12) == 1 and _eval_weight(e, 20.0) == 1
    assert _eval_weight(e, 3.9) == 0 and _eval_weight(e, 21.1) == 0
    assert 0 < _eval_weight(e, 4.5) < 1
    e0 = audio_mix.radio_weight_expr([(0.0, 3.0)], 1.0)
    assert _eval_weight(e0, 0.0) == 1                     # range at the clip start: muted from t=0


def test_build_audio_variants():
    plain = audio_mix.build_audio("1:a", "3:a", [], 30, duck=False, fade_s=1, music_volume=0.8, duck_volume=0.25)
    assert not plain.music_used and "volume" not in plain.filter and "[3:a]" not in plain.filter
    radio = audio_mix.build_audio("1:a", "3:a", [(5, 20)], 30, duck=False, fade_s=1, music_volume=0.8,
                                  duck_volume=0.25)
    assert radio.music_used and "amix" in radio.filter and "eval=frame" in radio.filter
    assert "volume='1-clip(" in radio.filter
    muted = audio_mix.build_audio("1:a", None, [(5, 20)], 30, duck=False, fade_s=1, music_volume=0.8,
                                  duck_volume=0.25)
    assert not muted.music_used and "amix" not in muted.filter and "volume='1-clip(" in muted.filter
    duck = audio_mix.build_audio("1:a", "3:a", [], 30, duck=True, fade_s=1, music_volume=0.8, duck_volume=0.25)
    assert duck.ducked and "volume='0.25*(1)'" in duck.filter and "afade=t=in" in duck.filter
    for ap in (plain, radio, muted, duck):
        assert ap.filter.endswith("[aout]")
        assert "channel_layouts=stereo" in ap.filter and "aresample=48000" in ap.filter


# ------------------------------------------------------------------ cut planning
def test_plan_cut_single_and_concat(tmp_path):
    a = Chunk(3, str(tmp_path / "a.ts"), start_s=600.0, duration_s=300.0)
    b = Chunk(4, str(tmp_path / "b.ts"), start_s=900.0, duration_s=300.0)
    one = cut.plan_cut(650, 680, [a], tmp_path)
    assert one.input_args == ["-i", a.path] and one.seek_s == 50 and one.duration_s == 30
    assert one.concat_list is None
    one2 = cut.plan_cut(650, 680, [b, a], tmp_path)          # extra non-overlapping chunk ignored
    assert one2.input_args == ["-i", a.path]
    two = cut.plan_cut(890, 920, [b, a], tmp_path)
    assert two.input_args[:4] == ["-f", "concat", "-safe", "0"] and two.seek_s == 290
    txt = two.concat_list.read_text(encoding="utf-8").splitlines()
    assert txt[0] == "ffconcat version 1.0" and txt[1].endswith("a.ts'") and txt[2].endswith("b.ts'")
    with pytest.raises(ValueError):
        cut.plan_cut(10, 10, [a], tmp_path)


# ------------------------------------------------------------------ geometry / graph
def test_geometry_720p():
    g = layout.geometry(layout.SourceInfo(1280, 720, "h264", True))
    assert (g.crop_l, g.crop_r, g.crop_w) == (437, 438, 405)
    assert (g.fg_w, g.fg_h, g.fg_y) == (1080, 608, 656)
    g2 = layout.geometry(layout.SourceInfo(1920, 1080, "h264", True), blur=[27, 48])
    assert g2.crop_w == 608 and (g2.blur_w, g2.blur_h) == (28, 48) and g2.fg_h == 608


def test_video_filter_matches_verified_pattern():
    g = layout.geometry(layout.SourceInfo(1280, 720, "h264", True))
    f = layout.video_filter(g, 2, 220)
    assert f == ("[2:v]format=yuva420p,hwupload_cuda[hk];"
                 "[0:v]scale_cuda=54:96:format=yuv420p,scale_cuda=1080:1920:interp_algo=bicubic,setsar=1[bg];"
                 "[bg][hk]overlay_cuda=0:220[bgh];"
                 "[1:v]scale_cuda=1080:608:format=yuv420p[fg];"
                 "[bgh][fg]overlay_cuda=0:656[vout]")
    nohook = layout.video_filter(g, None, 220)
    assert "hk" not in nohook and "[bg][fg]overlay_cuda=0:656[vout]" in nohook
    ins = layout.video_inputs(["-i", "x.ts"], 12.5, 30, "h264", g)
    assert ins.count("-ss") == 2 and ins.count("x.ts") == 2 and "h264_cuvid" in ins
    assert ins[ins.index("-crop") + 1] == "0x0x437x438"


# ------------------------------------------------------------------ music pick
def test_music_pick_deterministic(tmp_path, monkeypatch):
    from clipper import paths
    monkeypatch.setattr(paths, "MUSIC", tmp_path / "music")
    assert music.pick_track("x", 30) is None                  # missing folder -> None
    (tmp_path / "music").mkdir()
    for n in ("a.mp3", "b.wav", "c.m4a", "notes.txt"):
        (tmp_path / "music" / n).write_bytes(b"")
    names = [p.name for p in music.list_tracks()]
    assert names == ["a.mp3", "b.wav", "c.m4a"]
    monkeypatch.setattr(music.ffmpeg, "duration", lambda p: 200.0)
    p1, p2 = music.pick_track("cand-1", 30), music.pick_track("cand-1", 30)
    assert p1.path == p2.path and p1.offset_s == p2.offset_s and 0 <= p1.offset_s <= 170
    picks = {music.pick_track(f"cand-{i}", 30).path.name for i in range(30)}
    assert len(picks) > 1
    monkeypatch.setattr(music.ffmpeg, "duration", lambda p: 10.0)
    short = music.pick_track("cand-1", 30)
    assert short.loop and short.offset_s == 0
