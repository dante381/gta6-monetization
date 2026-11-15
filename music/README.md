# Music folder

Put royalty-free tracks here. Clip production uses them to replace in-game radio, and as a bed under ducked cutscenes.

- **Only use tracks you may use in monetized videos without credit.** In the YouTube Audio Library (YouTube Studio > Audio Library), filter by the license type "No attribution required" and download from there.
- Supported formats: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac` (also `.aac`, `.opus`).
- Put files directly in this folder; subfolders are ignored.
- Each clip picks a track and a start offset deterministically from its clip id, so re-rendering a clip gives the same music. Tracks shorter than the clip are looped.
- With no files here, radio sections are just muted, and a warning is logged once.
- Volume and fades are set in `config/settings.yaml` under `production:` (`music_volume`, `audio_fade_s`, `duck_volume`, `duck_scene_types`).
