"""Metadata rules: game name, #Shorts + base hashtags, spoilers, lengths, templates."""
from __future__ import annotations

from clipper.models import Candidate
from clipper.scoring import build_meta, make_hashtags, make_title


def _cand(track="gameplay", events=(), judge=None, tag=None, dur=30.0):
    return Candidate(id="s-0-10", chunk_idx=0, start_s=0, peak_s=6, end_s=dur, track=track,
                     events=list(events), judge=judge, scene_tag=tag)


def test_template_titles_from_events(cfg):
    m = build_meta(_cand(events=["wanted level up to 5 stars", "gunfire / explosion bursts"]), cfg)
    assert m.title == "5-star chase in GTA 6" and m.hook == "5 stars in 30 seconds"
    assert m.source == "template"
    assert build_meta(_cand(events=["mission passed"]), cfg).title == "Mission passed! (GTA 6)"
    assert build_meta(_cand(events=["big audio spike", "flash on screen"]), cfg).title == "Huge explosion in GTA 6"
    assert build_meta(_cand(events=["WASTED"]), cfg).hook == "WASTED"
    plain = build_meta(_cand(), cfg)
    assert "GTA 6" in plain.title and plain.hook


def test_title_always_has_game_name(cfg):
    m = build_meta(_cand(judge={"rating": 8, "title": "Insane police chase", "description": "d",
                                "hashtags": [], "hook": "h", "spoiler": False, "reason": "r"}), cfg)
    assert m.title == "Insane police chase (GTA 6)"
    assert m.source == "judge"
    m2 = build_meta(_cand(judge={"rating": 8, "title": "GTA 6 chaos", "hook": "x"}), cfg)
    assert m2.title == "GTA 6 chaos"


def test_hashtags_include_shorts_and_base_dedup_cap(cfg):
    tags = ["gaming", "#GTA6", "#shorts", "#Funny Moments", "#a", "#b", "#c", "#d", "#e", "#f"]
    m = build_meta(_cand(judge={"rating": 7, "title": "x", "hashtags": tags, "hook": "y"}), cfg)
    assert m.hashtags[0] == "#Shorts"
    for b in cfg.publish.base_hashtags:
        assert b in m.hashtags
    low = [t.lower() for t in m.hashtags]
    assert len(low) == len(set(low)) and len(m.hashtags) <= 8
    assert "#gaming" in m.hashtags and "#FunnyMoments" in m.hashtags
    assert all(t.startswith("#") and " " not in t for t in m.hashtags)
    assert "#Shorts" in build_meta(_cand(), cfg).hashtags


def test_cutscene_is_spoiler(cfg):
    m = build_meta(_cand(track="cutscene", events=["letterbox cutscene"]), cfg)
    assert m.spoiler is True and m.title.startswith("[Spoiler] ") and "GTA 6" in m.title
    j = build_meta(_cand(track="cutscene", judge={"rating": 8, "title": "Lucia's big moment in GTA 6",
                                                  "hook": "No way", "spoiler": False}), cfg)
    assert j.spoiler is True and "spoiler" in j.title.lower()
    already = build_meta(_cand(track="cutscene", judge={"rating": 8, "title": "SPOILER: ending of GTA 6"}), cfg)
    assert already.title == "SPOILER: ending of GTA 6"
    g = build_meta(_cand(judge={"rating": 8, "title": "Story twist", "spoiler": True}), cfg)
    assert g.spoiler and "spoiler" in g.title.lower()


def test_title_and_hook_lengths(cfg):
    long = "An unbelievably long and rambling description of a chase " * 5
    m = build_meta(_cand(track="cutscene", judge={"rating": 9, "title": long, "hook": "x" * 80}), cfg)
    assert len(m.title) <= 100 and "GTA 6" in m.title and "spoiler" in m.title.lower()
    assert len(m.hook) <= 40
    t = make_title("Huge GTA 6 " + "word " * 40, "GTA 6", False)
    assert len(t) <= 100 and "GTA 6" in t
    t2 = make_title("word " * 40 + "GTA 6", "GTA 6", True)
    assert len(t2) <= 100 and "GTA 6" in t2 and t2.startswith("[Spoiler] ")


def test_make_hashtags_helper():
    assert make_hashtags(["#x", "x", "#Y z"], ["#Shorts", "#GTA6"]) == ["#Shorts", "#GTA6", "#x", "#Yz"]
    assert make_hashtags([f"#t{i}" for i in range(20)], ["#GTA6"])[:2] == ["#Shorts", "#GTA6"]
    assert len(make_hashtags([f"#t{i}" for i in range(20)], ["#GTA6"])) == 8
