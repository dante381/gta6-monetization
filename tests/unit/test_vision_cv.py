"""Vision unit tests without GPU: CV measures, grids, prompt, Claude answer parsing, merge."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from clipper import paths
from clipper.analysis import grid_ai, vision
from clipper.claude_cli import ClaudeError
from clipper.models import VisualFrame


# --------------------------------------------------------------------------- letterbox
def _frame(h=180, w=320, mid=120, band=0, band_frac=0.12):
    g = np.full((h, w), mid, dtype=np.uint8)
    b = int(h * band_frac)
    g[:b] = band
    g[h - b:] = band
    return g


def test_letterbox_bars_detected(cfg):
    assert vision.letterbox_flag(_frame(), cfg)
    assert vision.letterbox_flag(_frame(h=36, w=64), cfg)


def test_letterbox_not_on_black_or_plain_frames(cfg):
    assert not vision.letterbox_flag(np.zeros((180, 320), np.uint8), cfg)           # fully black
    assert not vision.letterbox_flag(np.full((180, 320), 120, np.uint8), cfg)       # no bars
    assert not vision.letterbox_flag(_frame(band=60), cfg)                          # grey bands
    g = _frame()
    g[:18, ::10] = 255                                                               # lights in the "bar"
    assert not vision.letterbox_flag(g, cfg)


# --------------------------------------------------------------------------- motion
def _series(cfg, frames, fps=2.0):
    acc = vision._MotionAccumulator(cfg, (64, 36))
    for f in frames:
        acc.add(f)
    return acc.series(fps)


def test_motion_static_flash_scene(cfg):
    rng = np.random.default_rng(0)
    still = [np.full((36, 64), 0, np.uint8)] * 10
    noisy = [rng.integers(0, 255, (36, 64), dtype=np.uint8) for _ in range(10)]
    m = _series(cfg, still + noisy)
    assert m.static[1:8].all() and not m.static[12:].any()
    assert m.motion[:9].max() < 0.01 and m.motion[11:].min() > 0.5
    assert len(m.t) == 20 and m.t[1] == 0.5


def test_motion_flash_and_cut(cfg):
    dark = np.full((36, 64), 40, np.uint8)
    bright = np.full((36, 64), 220, np.uint8)
    m = _series(cfg, [dark, dark, bright, bright])
    assert m.flash[2] == 1.0 and m.flash[3] == 0.0
    assert m.scene_change[2] > 0.9


def test_no_flash_out_of_black(cfg):
    m = _series(cfg, [np.zeros((36, 64), np.uint8), np.full((36, 64), 200, np.uint8)])
    assert m.flash[1] == 0.0


def test_shake_from_global_shift(cfg):
    rng = np.random.default_rng(1)
    base = (rng.random((60, 100)) * 255).astype(np.uint8)
    base = np.asarray(Image.fromarray(base).resize((100, 60), Image.BILINEAR))
    a, b = base[10:46, 10:74], base[12:48, 13:77]     # shifted by (3, 2) px
    m = _series(cfg, [a, a, b])
    assert m.shake[1] == 0.0
    assert m.shake[2] > 0.5


# --------------------------------------------------------------------------- grids
def _fake_frames(tmp_path, n, interval=3.0):
    fp = {}
    for i in range(n):
        p = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (320, 180), (20 + i * 5, 90, 160)).save(p)
        fp[i * interval] = str(p)
    return fp


def test_build_grids_layout_and_labels(cfg, tmp_path):
    fp = _fake_frames(tmp_path, 45)
    grids = vision.build_grids(fp, cfg, tmp_path / "g")
    assert [len(ts) for _, ts in grids] == [20, 20, 5]
    assert grids[1][1][0] == 60.0
    lab = vision.DEFAULTS["vision_label_px"]
    im = np.asarray(Image.open(grids[0][0]).convert("RGB"))
    assert im.shape == (5 * (180 + lab), 4 * 320, 3)
    # every label strip has yellow text pixels
    for n in range(20):
        x, y = (n % 4) * 320, (n // 4) * (180 + lab)
        strip = im[y:y + lab, x:x + 320].astype(int)
        yellow = (strip[..., 0] > 180) & (strip[..., 1] > 180) & (strip[..., 2] < 100)
        assert yellow.sum() > 20, f"tile {n} has no label"


# --------------------------------------------------------------------------- prompt
def test_prompt_has_timestamps_paths_and_vocab(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "HUD_REFS", tmp_path / "no_refs")
    grids = vision.build_grids(_fake_frames(tmp_path, 25), cfg, tmp_path / "g")
    prompt, dirs = grid_ai.build_prompt(grids, cfg)
    ts_line = next(l for l in prompt.splitlines() if l.startswith("TIMESTAMPS:"))
    assert ts_line == "TIMESTAMPS: " + ",".join(str(i * 3) for i in range(25))
    for g, _ in grids:
        assert str(g) in prompt
    for word in ("wanted_up", "wasted", "busted", "mission_passed", "loading", "scene_tag", "title_text"):
        assert word in prompt
    assert "{{" not in prompt
    assert dirs == [(tmp_path / "g").resolve()]


def test_prompt_lists_hud_refs(cfg, tmp_path, monkeypatch):
    refs = tmp_path / "hud"
    refs.mkdir()
    for i in range(8):
        Image.new("RGB", (8, 8)).save(refs / f"wasted_{i}.png")
    (refs / "notes.txt").write_text("x")
    monkeypatch.setattr(paths, "HUD_REFS", refs)
    grids = vision.build_grids(_fake_frames(tmp_path, 3), cfg, tmp_path / "g")
    prompt, dirs = grid_ai.build_prompt(grids, cfg)
    n = cfg.analysis.hud_ref_max_images
    assert sum(1 for i in range(8) if f"wasted_{i}.png" in prompt) == n
    assert "notes.txt" not in prompt and "reference examples of HUD events".lower() in prompt.lower()
    assert refs.resolve() in dirs
    prompt2, dirs2 = grid_ai.build_prompt(grids, cfg, hud_refs=False)
    assert "wasted_0.png" not in prompt2 and refs.resolve() not in dirs2


# --------------------------------------------------------------------------- parsing
def test_parse_frames_snaps_and_cleans():
    data = {"frames": [
        {"t": 0, "kind": "gameplay", "hud_events": ["wanted_up", "bogus", "wanted_up"], "wanted_stars": 9,
         "title_text": "  ", "action": 14, "scene_tag": "  City  Street "},
        {"t": 3.4, "kind": "weird", "hud_events": "wasted", "wanted_stars": 1, "title_text": "MISSION ONE",
         "action": "5", "scene_tag": None},
        {"t": 3.0, "kind": "menu", "hud_events": [], "wanted_stars": 0, "title_text": None, "action": 0,
         "scene_tag": "dup"},
        {"t": 99, "kind": "menu"}, "junk",
    ]}
    fr = grid_ai.parse_frames(data, [0, 3, 6], 3.0)
    assert [f.t for f in fr] == [0, 3]
    a, b = fr
    assert a.hud_events == ["wanted_up"] and a.wanted_stars == 5 and a.action == 10
    assert a.title_text is None and a.scene_tag == "city street" and a.from_ai
    assert b.kind == "other" and b.hud_events == ["wasted"] and b.title_text == "MISSION ONE" and b.action == 5


@pytest.mark.parametrize("bad", [None, [], {"x": 1}, {"frames": "no"}, {"frames": [{"t": 500}]}])
def test_parse_frames_rejects_garbage(bad):
    with pytest.raises(ClaudeError):
        grid_ai.parse_frames(bad, [0, 3], 3.0)


def test_label_grids_partial_answer_and_call_args(cfg, tmp_path, monkeypatch):
    seen = {}

    def fake_call(prompt, **kw):
        seen.update(kw, prompt=prompt)
        return {"frames": [{"t": 3, "kind": "cutscene", "hud_events": [], "wanted_stars": 0,
                            "title_text": "ACT I", "action": 2, "scene_tag": "rooftop"}]}

    monkeypatch.setattr(grid_ai.claude_cli, "call", fake_call)
    grids = vision.build_grids(_fake_frames(tmp_path, 4), cfg, tmp_path / "g")
    fr = grid_ai.label_grids(grids, cfg)
    assert [f.t for f in fr] == [3.0]
    assert seen["schema"] == "grid" and seen["effort"] == cfg.claude.effort_grid == "low"
    assert seen["model"] == cfg.claude.model
    assert seen["timeout_s"] == cfg.claude.timeout_s


def test_label_grids_missing_binary_is_unavailable(cfg, tmp_path, monkeypatch):
    from clipper.claude_cli import ClaudeUnavailable

    def boom(*a, **k):
        raise FileNotFoundError("claude")

    monkeypatch.setattr(grid_ai.claude_cli, "call", boom)
    grids = vision.build_grids(_fake_frames(tmp_path, 2), cfg, tmp_path / "g")
    with pytest.raises(ClaudeUnavailable):
        grid_ai.label_grids(grids, cfg)


# --------------------------------------------------------------------------- merge
def test_merge_fills_missing_with_cv():
    cv = [VisualFrame(t=0.0, kind="gameplay", action=3, letterbox=False),
          VisualFrame(t=3.0, kind="cutscene", action=1, letterbox=True),
          VisualFrame(t=6.0, kind="loading", action=0)]
    ai = [VisualFrame(t=3.0, kind="cutscene", title_text="ACT I", scene_tag="rooftop", from_ai=True)]
    out = vision.merge_frames(cv, ai)
    assert [f.t for f in out] == [0.0, 3.0, 6.0]
    assert out[1].from_ai and out[1].letterbox and out[1].title_text == "ACT I"
    assert not out[0].from_ai and not out[2].from_ai and out[2].kind == "loading"
