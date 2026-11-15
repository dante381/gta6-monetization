"""E. Clip production: candidate window -> 1080x1920 Short with hook text and music swap (GPU only)."""
from .produce import ProductionError, build_command, produce_clip

__all__ = ["produce_clip", "build_command", "ProductionError"]
