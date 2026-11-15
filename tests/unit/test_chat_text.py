"""Chat text heuristics: hype, clip requests, sanitization."""
from __future__ import annotations

import pytest

from clipper.chat.scoring_text import clip_request, hype_score, sanitize
from clipper.config import load_config


@pytest.fixture
def cc():
    return load_config().chat


def test_hype_words_need_word_boundaries(cc):
    assert hype_score("omg that was insane", cc) > hype_score("hello there", cc) == 0
    assert hype_score("showtime", cc) == 0           # contains 'w' but not as a word
    assert hype_score("no way", cc) > 0              # multi-word
    assert hype_score("W", cc) > 0


def test_emotes_elongation_caps_and_cap(cc):
    assert hype_score("🔥🔥", cc) > 0
    base = hype_score("lmao", cc)
    assert hype_score("lmaooooo", cc) > base
    assert hype_score("THIS IS SO GOOD", cc) > hype_score("this is so good", cc)
    assert hype_score("OMG INSANE LMAOOOO 🔥🔥🔥 WTF", cc) == 2.0
    assert hype_score("", cc) == 0


def test_clip_request(cc):
    assert clip_request("CLIP THAT!!", cc)
    assert clip_request("someone clip it", cc)
    assert not clip_request("nice clipboard", cc)


def test_sanitize(cc):
    s = sanitize("@SomeUser check https://evil.example/x?y=1 or mail me bob@mail.com call +1 (555) 123-4567 wow", cc)
    assert "@" not in s and "http" not in s and "mail.com" not in s and "555" not in s
    assert "wow" in s
    assert sanitize("   ", cc) == "" and sanitize("@a @b", cc) == ""
    assert len(sanitize("x" * 500, cc)) == int(cc.sample_max_chars)
