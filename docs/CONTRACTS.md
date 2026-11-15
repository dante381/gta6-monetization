# Build contracts (read before writing any module)

Spec: `GTA6_Stream_Auto_Clipper_Plan.md`. This file pins the interfaces between modules so they can be built in parallel.

## Hard rules
1. **Storage:** nothing is written or downloaded outside `E:\Development\GTA6 Monetization`.
   - Use `.tmp/` for scratch.
   - Subprocesses get `clipper.env.subprocess_env()`.
   - Never `pip install` or `winget`. If a package is missing, report it instead.
2. **Shell:** run shell commands with `run_in_background: true` and wait for the completion notice. This is the user's preference.
3. **GPU only for video:** decode, scale, overlay and encode run on CUDA/NVDEC/NVENC (`h264_nvenc`).
   - No libx264, swscale-on-full-frames or CPU fallback in pipeline code.
   - On GPU failure, raise or let `clipper.ffmpeg.GpuUnavailable` propagate; `run_gpu` waits and retries.
   - Audio filters run on the CPU, which is fine.
   - Test fixtures may use libx264.
4. **RAM cap 2 GB for everything combined.**
   - The worker runs stages strictly one at a time and never runs a claude call and a GPU ffmpeg job at the same time.
   - Measured peaks: render 816 MB, analysis decode 335 MB, claude -p 477 MB.
   - Stream frames and audio in pieces.
5. **Heartbeat honesty:** long waits pass an `on_tick` callback (both `clipper.ffmpeg.run` and `clipper.claude_cli.call` accept one). Never sleep longer than about 5 s without ticking.
6. **Frozen core:** do not edit these files. Request changes in your final report.
   - `clipper/{paths,env,config,models,state,log,heartbeat,ffmpeg,claude_cli,crash}.py`
   - `config/locked.yaml`
   - `requirements.txt`
   - `tests/conftest.py`
   - `tests/fakes/fake_claude.py`
   - `prompts/schemas/*.json`

   You may **add** keys to `config/settings.yaml` under your own section only. Say so in your report.
7. **Tests:**
   - Use pytest under `tests/unit/test_<area>_*.py`.
   - Run them with `.venv/Scripts/python.exe -m pytest tests/unit/test_<area>_*.py`.
   - Claude is always faked (CLAUDE_BIN is preset by conftest; use `claude_scenario` to script answers).
   - Real GPU and ffmpeg are allowed (mark `@pytest.mark.gpu`).
   - Network tests are marked `network` and skipped unless `-m network`.

## Core you can use
| Module | What |
|---|---|
| `clipper.paths` | `ROOT`, `DATA.chunks/frames/grids/work/ready/backup/logs/heartbeat/samples/reports/claude_work/state_db`, `MUSIC`, `HUD_REFS`, `CLIENT_SECRET`, `TOKEN`, `VENV_PY` |
| `clipper.config` | `load_config(overrides=)` gives attribute-dict `cfg`; `cfg.locked.*` holds the protected values; `paused(cfg)` checks the PAUSE kill switch |
| `clipper.models` | `Chunk, AudioFeatures, VisualFrame, MotionSeries, VisualFeatures, Candidate, ClipMeta, ClipSidecar, QueueItem` |
| `clipper.state.StateDB` | chunks, candidates, seen scenes, queue, upload log, kv, hold flags (`publish_hold`, `keep_chunks`) |
| `clipper.ffmpeg` | `run`, `run_gpu`, `popen`, `probe`, `duration`, `grab_frames`, `kill_tree`, `GPU_DECODE`, `GpuUnavailable`, `FfmpegError` |
| `clipper.claude_cli` | `call(prompt, schema="grid" or "judge" or dict or None, effort, add_dirs, on_tick)`, `run_raw`, `ClaudeUnavailable`, `ClaudeError`, `load_prompt`, `load_schema` |
| `clipper.log` | `setup_logging(name)`, `event(kind, **f)` (writes `data/logs/clips.jsonl`), `attention(msg)` |
| `clipper.heartbeat` | `Heartbeat(name, interval).tick()`, `age(name)`, `read`, `clear` |
| `clipper.crash` | `install_crash_handler(name)`, `read_report(name)` gives `{signature, type, message, traceback, ts}` |

**Time base:**
- *Stream seconds:* chunk N covers `[chunk.start_s, chunk.start_s + duration_s)`.
- Analysis arrays and `VisualFrame.t` use chunk-relative seconds.
- `Candidate.start_s / peak_s / end_s / radio_ranges` use stream seconds.

## Spike findings (verified on this laptop: GTX 1650 Max-Q, ffmpeg master build 2026-11-14)
- **YouTube live HLS has separate video and audio playlists.**
  - `yt-dlp -g -f "bv*[height<=720]+ba/b[height<=720]"` prints 1 or 2 URLs.
  - With 2 URLs: `ffmpeg -i V -i A -map 0:v:0 -map 1:a:0 -c copy -f segment -segment_time 300 -segment_format mpegts -reset_timestamps 1 -segment_start_number N -segment_list list.csv -segment_list_type csv chunk_%05d.ts`. This works.
  - The CSV gets one line per **closed** segment: `name,start,end`.
  - Segments have `start_time` of about 1.4 s. ffmpeg `-ss` on input is relative to the file start, so it doesn't matter.
  - yt-dlp needs deno. `tools/deno` is on PATH via `subprocess_env()`. Pass `--cache-dir .cache/yt-dlp`.
  - Use `.venv/Scripts/yt-dlp.exe` (or `python -m yt_dlp`).
  - Resolving takes about 20–60 s.
- **9:16 render graph that works.** overlay_cuda on this build is picky:
  - yuva420p can only overlay onto yuv420p.
  - The output of an overlay_cuda can be the *main* input of another overlay_cuda, but it **cannot be fed to the overlay pad or to scale_cuda** ("Frame size or format changed").
  - Its output is 1088 wide, so crop with the `h264_metadata` bitstream filter.

  ```
  ffmpeg -hwaccel cuda -hwaccel_output_format cuda -c:v h264_cuvid -crop 0x0x{L}x{R} -ss S -t D -i SRC   # bg: centre 9:16 crop by the decoder
         -hwaccel cuda -hwaccel_output_format cuda -ss S -t D -i SRC                                     # fg
         -i hook.png                                                                                     # single frame, NO -loop
    -filter_complex "[2:v]format=yuva420p,hwupload_cuda[hk];
                     [0:v]scale_cuda=54:96:format=yuv420p,scale_cuda=1080:1920:interp_algo=bicubic[bg];
                     [bg][hk]overlay_cuda=0:220[bgh];
                     [1:v]scale_cuda=1080:608:format=yuv420p[fg];
                     [bgh][fg]overlay_cuda=0:656[v]"
    -map "[v]" -map 1:a -c:v h264_nvenc -preset p5 -cq 23 -c:a aac -bsf:v h264_metadata=crop_right=8 out.mp4
  ```
  - For 1280×720 input: L=437, R=438 (crop width 405). In general: crop width `round(h*9/16)`, L=(w-cw)//2, R=w-cw-L.
  - Probe the output width after encoding. If it isn't 1080, adjust `crop_right` = coded width minus 1080.
  - A 45 s clip renders in about 6 s.
  - The hook PNG (full canvas width, transparent) sits above the fg, so the hook is drawn on the bg area.
  - Audio is processed separately (CPU filters) and muxed in the same command or a second copy pass.
- **Analysis decode (one GPU decode, two outputs):**
  ```
  ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i CHUNK -filter_complex
   "[0:v]split=2[a][b];[a]fps=1/3,scale_cuda=320:180,hwdownload,format=nv12[fa];[b]fps=2,scale_cuda=64:36,hwdownload,format=nv12[fb]"
   -map "[fa]" -f rawvideo -pix_fmt rgb24 frames.rgb -map "[fb]" -f rawvideo -pix_fmt gray motion.gray
  ```
  - A 50 s input takes about 1.4 s.
  - Frame i of `fa` is at t = i*3 s, frame j of `fb` is at t = j*0.5 s.
  - Stream through pipes or temp files in `DATA.frames`; don't hold the whole chunk in RAM.
- **claude -p:**
  - `--output-format json` returns an envelope; `structured_output` holds the schema-valid dict.
  - A grid call at low effort takes about 10 s and costs about $0.03.
  - `claude_cli.call` already handles all of this.
  - **Grid prompt contract:** the prompt must contain a line `TIMESTAMPS: 0,3,6,...` listing every tile's timestamp (the fake relies on it), plus the absolute grid image paths.

## Module APIs (what the worker will call)
**A. Recorder:** `clipper/recorder/`
- `stream.py`
  - `live_status(url, cfg) -> dict` returns `{"status": "is_live"|"is_upcoming"|"was_live"|"not_live"|"post_live"|"error", "video_id", "title"}`.
  - `resolve_hls(url, cfg) -> list[str]` returns 1 or 2 URLs.
  - `wait_until_live(url, cfg, hb) -> dict` polls every `cfg.stream.live_poll_s`.
- `recorder.py`
  - `class Recorder(cfg, state, url)` with `.run()`.
  - `main(argv)` takes `--url`, falling back to `cfg.stream.url`, plus `--chunk-seconds` for tests.
  - `stream_id = f"{video_id}-{YYYYMMDD}"`, kept in `state.set_current_stream`. Resume reuses the existing id and continues from `last_chunk_idx + 1`.
  - Chunks go to `DATA.chunks/<stream_id>/chunk_{idx:05d}.ts`.
  - On each closed segment: `state.add_chunk(Chunk(idx, path, start_s=previous end, duration_s=ffprobe duration, stream_id))`, plus `log.event("chunk_recorded")`.
  - Stall or exit: re-resolve and restart with the next index. Stream end: close the last partial chunk, register it, set kv `recorder.done=True`, exit 0.
  - Heartbeat name: `recorder`.

**B. Audio:** `clipper/analysis/audio.py`
- `analyze_audio(chunk, cfg, state, on_tick=None) -> AudioFeatures`
  - ffmpeg decodes audio to a 16 kHz mono f32le pipe, read in blocks.
  - Rolling baseline across chunks in `state.kv` key `audio.baseline`.

**C. Vision:** `clipper/analysis/vision.py`, `grid_ai.py`, `prompts/grid.md`
- `analyze_video(chunk, cfg, on_tick=None, hud_refs=True) -> VisualFeatures`
  - First a GPU decode producing frames, grids and motion.
  - Then **one** Claude grid call. On `ClaudeUnavailable`: frames are filled from CV only, `from_ai=False`, `ai_ok=False`, and `log.event("claude_unavailable")`.
  - Small frames are saved as JPEGs to `DATA.frames/<stream_id>/<idx>/`, grids to `DATA.grids/...`.

**D. Scoring:** `clipper/scoring/`, `prompts/judge.md`
- `score_chunk(chunk, audio, visual, cfg, state, on_tick=None) -> list[Candidate]`. It covers:
  - candidates and track assignment
  - signals and penalties
  - the judge for signal score ≥ `cfg.locked.thresholds.judge_gate`, using `ffmpeg.grab_frames` plus `claude_cli.call(schema="judge", effort=cfg.claude.effort_judge)`
  - the final score, route and template metadata
  - `state.save_candidate`, `state.mark_scene`, and `log.event("candidate")`
- Routes: drop when < `backup`, backup when below `publish`, publish otherwise.
- Titles always include `cfg.publish.game_name`. The hashtags include `#Shorts`. Spoiler clips (cutscene track) get "spoiler" in the title.

**E. Production:** `clipper/production/`
- `produce_clip(cand, chunks, cfg, on_tick=None) -> ClipSidecar`
  - `chunks` are the Chunk objects covering `[start_s, end_s]`, from `state.chunks_covering`. Use the concat demuxer when there are 2.
  - Output goes to `DATA.ready` (route publish) or `DATA.backup`: `<cand.id>.mp4` plus `<cand.id>.json` (the `ClipSidecar.to_dict()`).
  - Audio:
    - radio ranges are muted and replaced by a track from `paths.MUSIC` (mp3/wav/m4a), with fades
    - cutscene tracks keep their audio, unless `scene_tag` is in `cfg.production.duck_scene_types`, in which case the game audio is ducked under music
    - with no music files, radio sections are just muted
  - `log.event("clip_produced")`.

**F. Publishing:** `clipper/publish/`, `site/`
- `auth.py`
  - `get_credentials(interactive=False)` raises `AuthRequired` when the token is missing or expired.
  - `main()` is the browser flow, using `paths.CLIENT_SECRET` and `paths.TOKEN`.
- `queue.py`: `enqueue(sidecar, state, cfg)`.
- `publish_tick(cfg, state, now=None, youtube=None) -> list[dict]` is called by the worker every loop. It:
  - honours PAUSE, `test_mode` (`would_upload` events), `state.publish_hold()`, caps, slots and the daily upload limit (Pacific day)
  - uploads backups after scheduled posts
  - on `AuthRequired`: `set_publish_hold(True, "auth")` plus `log.attention`
- The `youtube` param is injectable for tests.
- Uploads are resumable with `MediaFileUpload` and `videos.insert`.

**G. Supervisor:** `clipper/supervisor/`, `prompts/repair.md`
- `main(argv)` runs as `python -m clipper supervise [--url URL]`. It:
  - creates the job object with a 2 GB limit (`cfg.locked.memory_limit_mb`)
  - spawns `[VENV_PY, "-m", "clipper", "record", "--url", URL]` and `[VENV_PY, "-m", "clipper", "work"]` inside it, suspended, assigned, then resumed
  - restarts with backoff and detects hangs via heartbeat age > `hang_s`
  - reads crash signatures from `crash.read_report("worker" or "recorder")` and runs the repair flow
  - keeps the laptop awake while running
- The recorder exits with code 0 when the stream ends: the supervisor doesn't restart it, lets the worker drain, then stops.
- `autostart.py`: `register(url) / unregister() / status()` via schtasks.

The worker (`clipper/worker.py`) and the CLI are integration work done by the lead; don't write them.
