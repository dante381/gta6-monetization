# GTA 6 Stream Auto Clipper

Turns a live GTA 6 stream from a PS5 into published YouTube Shorts with no manual review. You go live on the PS5 and start one script; the rest is automatic.

```
PS5 -> YouTube Live -> Recorder (5 min chunks) -> Analysis (audio + Claude CLI)
    -> Scoring (0-100) -> Clip production (9:16, hook text, music) -> Publish queue -> YouTube Shorts
```

A clip reaches the publish queue about 5 to 7 minutes after the moment happens.

- **Input:** a live YouTube stream, no capture card, no commentary.
- **Output:** vertical 1080x1920 Shorts of 20 to 45 seconds, scored, titled and scheduled automatically.
- **Signals:** game video and audio (plus live chat activity). Highlights from gameplay and cutscenes.
- **AI:** the [Claude CLI](https://claude.com/claude-code) (`claude -p`) reads frame grids, judges clips and writes titles and hook text. Audio is measured with plain numpy signal processing.

## Requirements

- Windows 11 laptop, always plugged in with sleep disabled while streaming
- NVIDIA GPU with current drivers and an ffmpeg build with CUDA, NVDEC and NVENC. All video work is GPU only; there is no CPU fallback.
- Python 3, ffmpeg, yt-dlp (`scripts/setup.ps1` installs these inside the project folder)
- Claude CLI, signed in
- A Google Cloud project with the YouTube Data API and a Desktop OAuth client saved as `client_secret.json` in the project root
- Royalty-free tracks in `music/` (see [`music/README.md`](music/README.md))

## Quick start

```powershell
# one-time setup; everything is installed inside the project folder, nothing goes to C:
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

# pre-stream checks: GPU encoder, tools, Claude login, secrets
python -m clipper check

# one-time YouTube sign-in
python -m clipper auth

# during a stream: starts and restarts recorder, worker and chat collector
python -m clipper supervise
```

The first one or two streams should run in test mode (everything runs except uploads) so scoring can be tuned. See *Calibration* in the plan.

## Commands

`python -m clipper <command>`

| Command | Purpose |
| --- | --- |
| `check` | Pre-stream checks (GPU, tools, Claude login, secrets) |
| `supervise` | The one you run. Starts and restarts the recorder and worker |
| `record` | Recorder process (normally started by the supervisor) |
| `work` | Processing worker (normally started by the supervisor) |
| `chat` | Live-chat collector (normally started by the supervisor) |
| `auth` | One-time YouTube sign-in |
| `publish` | Publish what is still queued, then exit. Honours `PAUSE` and test mode |
| `report` | Post-stream report |
| `calibrate` | Export or summarize the calibration review |
| `health` | Show problems now; `--watch` prints one line per new problem |
| `window` | Launch-day run window: `status`, `install`, `remove`, `rearm` |
| `autostart` | Register, unregister or check the Windows scheduled task |

## Publishing rules

| Final score | Action |
| --- | --- |
| 75 or more | Queued for publishing |
| 50 to 74 | Uploaded as unlisted, kept as a backup |
| Under 50 | Deleted |

Starting values, tuned during calibration. Daily cap of 10 Shorts, one every 2 to 3 hours, with a separate cap for cutscene clips.

**Kill switch:** create a file named `PAUSE` in the project folder to stop all uploads at once. Delete it to resume.

## Configuration

- `config/settings.yaml`: tunable settings (scoring weights, thresholds, caps, production, music)
- `config/locked.yaml`: values the repair agent may not change

## Reliability

- Recorder and worker are separate processes, so a processing crash never stops recording.
- A supervisor restarts crashed or hung processes (heartbeat every 30 s, hung after 2 min).
- State is persisted, so a restart resumes instead of redoing work.
- Everything runs in one Windows job object capped at 2 GB RAM.
- If the same error crashes 3 times in 10 minutes, a Claude CLI repair agent attempts a fix (at most 3 per stream). Fixes are git commits and must pass the tests.

## Repository layout

| Path | Contents |
| --- | --- |
| `clipper/` | Application code: `recorder/`, `analysis/`, `scoring/`, `production/`, `publish/`, `chat/`, `supervisor/` |
| `prompts/` | Claude prompts and JSON output schemas |
| `config/` | Settings |
| `scripts/` | `setup.ps1`, `make_music.py` (generates an original royalty-free music bed) |
| `site/` | Homepage and privacy policy for the Google OAuth consent screen (GitHub Pages) |
| `tests/` | Test suite and a fake Claude CLI |
| `docs/CONTRACTS.md` | Interface contracts between modules and the hard rules for contributors |
| `GTA6_Stream_Auto_Clipper_Plan.md` | Full spec: design, scoring signals, risks, build phases |

## Tests

```powershell
python -m pytest
```

## Documentation

- [Plan and spec](GTA6_Stream_Auto_Clipper_Plan.md): start here for how and why
- [Build contracts](docs/CONTRACTS.md): read before changing any module

## Secrets

`client_secret.json`, `token.json`, `secrets/`, `data/` and downloaded tools are git-ignored. The repair agent is blocked from reading the YouTube token.

## License

[MIT](LICENSE). Covers the code only; it grants no rights to any game footage the tool records or to the music you put in `music/`.

## Notes

- Monetization of no-commentary gameplay and cutscene clips is not guaranteed; see *Risks and limits* in the plan.
- Until the YouTube API Services audit passes, API uploads are locked to private.
