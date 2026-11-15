"""Hook text: render one short line (max 2 lines) to a transparent full-width PNG with Pillow.

The PNG is uploaded to the GPU once and overlaid on the blurred background above the fg frame.
"""
from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

FONT_DIR = Path("C:/Windows/Fonts")
# Bold fonts tried (in order) when cfg.production.font_path is missing.
FALLBACK_FONTS = ("arialbd.ttf", "segoeuib.ttf", "seguibl.ttf", "impact.ttf", "verdanab.ttf",
                  "tahomabd.ttf", "calibrib.ttf", "arial.ttf", "segoeui.ttf")


def _p(cfg, key, default):
    prod = cfg.get("production", {}) if hasattr(cfg, "get") else {}
    v = prod.get(key, default) if hasattr(prod, "get") else default
    return default if v is None else v


def find_font(font_path: str | None) -> Path:
    if font_path and Path(font_path).is_file():
        return Path(font_path)
    for name in FALLBACK_FONTS:
        p = FONT_DIR / name
        if p.is_file():
            return p
    for p in sorted(FONT_DIR.glob("*.ttf")):
        return p
    raise FileNotFoundError("no TTF font found (production.font_path / C:/Windows/Fonts)")


def clean_text(text: str, max_chars: int) -> str:
    """Strip emoji/symbols the font can't draw, collapse whitespace, cap length at a word boundary."""
    keep = []
    for ch in text or "":
        cat = unicodedata.category(ch)
        if cat in ("So", "Cs", "Co", "Cn") or ord(ch) > 0xFFFF or ch in "\u200d\ufe0f\ufe0e":
            continue
        keep.append(" " if cat.startswith("Z") or ch in "\r\n\t" else ch)
    s = " ".join("".join(keep).split())
    if max_chars and len(s) > max_chars:
        cut = s[:max_chars].rstrip()
        if " " in cut and cut.rfind(" ") > max_chars // 2:
            cut = cut[:cut.rfind(" ")]
        s = cut.rstrip(" ,.;:-") + "…"
    return s


def _wrap(words: list[str], font, draw, max_w: int, stroke: int, max_lines: int) -> list[str] | None:
    """Greedy word wrap; None if it needs more than max_lines or a single word is too wide."""
    def width(s: str) -> int:
        l, _, r, _ = draw.textbbox((0, 0), s, font=font, stroke_width=stroke)
        return r - l

    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if width(trial) <= max_w:
            cur = trial
            continue
        if not cur or width(w) > max_w:
            return None
        lines.append(cur)
        cur = w
        if len(lines) >= max_lines:
            return None
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        return None
    # balance 2 lines: move words down while the split gets more even
    if len(lines) == 2:
        a, b = lines[0].split(), lines[1].split()
        while len(a) > 1:
            na, nb = a[:-1], [a[-1]] + b
            if width(" ".join(nb)) > max_w or width(" ".join(na)) < width(" ".join(nb)):
                break
            a, b = na, nb
        lines = [" ".join(a), " ".join(b)]
    return lines


def layout_hook(text: str, cfg) -> tuple[list[str], ImageFont.FreeTypeFont, int]:
    """Pick the largest font size where the text fits in <= max lines within the margins."""
    canvas_w = int(_p(cfg, "width", 1080))
    margin = int(_p(cfg, "hook_margin_px", 60))
    max_lines = int(_p(cfg, "hook_max_lines", 2))
    size0 = int(_p(cfg, "hook_font_size", 72))
    min_size = int(_p(cfg, "hook_min_font_size", 40))
    font_file = find_font(_p(cfg, "font_path", None))
    draw = ImageDraw.Draw(Image.new("L", (8, 8)))
    words = text.split()
    max_w = canvas_w - 2 * margin
    size = size0
    while True:
        font = ImageFont.truetype(str(font_file), size)
        stroke = max(2, round(size * float(_p(cfg, "hook_stroke_ratio", 0.11))))
        lines = _wrap(words, font, draw, max_w, stroke, max_lines)
        if lines is not None:
            return lines, font, stroke
        if size <= min_size:
            break
        size = max(min_size, size - 4)
    # still too long at the minimum size: hard-wrap by characters and drop the overflow
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        l, _, r, _ = draw.textbbox((0, 0), trial + "…", font=font, stroke_width=stroke)
        if r - l <= max_w:
            cur = trial
        else:
            lines.append(cur or w)
            cur = "" if not cur else w
    if cur:
        lines.append(cur)
    lines = [ln for ln in lines if ln][:max_lines]
    if len(lines) == max_lines and " ".join(lines) != " ".join(words):
        lines[-1] = lines[-1].rstrip(" ,.;:-") + "…"
    return lines, font, stroke


def render_hook(text: str, cfg, out_path: str | Path) -> Path | None:
    """Render `text` (white, thick black stroke, centred) to a transparent PNG `width` px wide.

    Returns the PNG path, or None if the text is empty after cleaning (skip the overlay branch).
    PNG dimensions are even (needed for yuva420p).
    """
    text = clean_text(text, int(_p(cfg, "hook_max_chars", 40) or 0))
    if not text:
        return None
    canvas_w = int(_p(cfg, "width", 1080))
    lines, font, stroke = layout_hook(text, cfg)
    draw = ImageDraw.Draw(Image.new("L", (8, 8)))
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + stroke * 2
    gap = int(line_h * 0.08)
    pad = stroke + 4
    h = pad * 2 + line_h * len(lines) + gap * (len(lines) - 1)
    h += h % 2
    img = Image.new("RGBA", (canvas_w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = pad
    for ln in lines:
        l, _, r, _ = draw.textbbox((0, 0), ln, font=font, stroke_width=stroke)
        x = (canvas_w - (r - l)) // 2 - l
        d.text((x, y + stroke), ln, font=font, fill=(255, 255, 255, 255),
               stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        y += line_h + gap
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    log.debug("hook rendered: %r -> %d lines, size %d, %dx%d", text, len(lines), font.size, canvas_w, h)
    return out_path
