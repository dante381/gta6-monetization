# GTA 6 Stream Auto Clipper: Plan

Nov 15, 2026 · @blah

## Overview and goals

A script on the laptop turns a live GTA 6 stream from the PS5 into published YouTube Shorts, with no manual review once the stream and script are running.

- **Input:** a live stream from the PS5 straight to YouTube, with no capture card and no commentary.
- **Output:** vertical 9:16 Shorts of 20 to 45 seconds, scored, titled and scheduled automatically.
- **Scope:** gameplay highlights plus cutscenes and title cards, since launch week rewards first-look footage.
- **Signals:** game video and game audio only. Chat and voice are left out.
- **Manual steps:** go live on the PS5 and start the script. Nothing else.

## End to end flow

```
PS5 (streams the game)
  -> YouTube Live (hosts the stream)
  -> Recorder (5 minute chunks; chunk deleted after use)
  -> Analysis (audio + Claude CLI)
  -> Scoring (two tracks, 0 to 100; under 50: deleted)
  -> Clip production (cut, 9:16, hook, music; 50 to 74: unlisted)
  -> Publish queue (daily cap, schedule; 75 or more)
  -> YouTube Shorts (scheduled posts)
```

You only go live on the PS5 and start the script; each clip reaches the publish queue about 5 to 7 minutes after the moment happens.

### Where the Claude CLI fits

Every step that needs AI uses the Claude CLI: reading the video, judging clips, writing text, and building the scripts. ffmpeg and Python handle recording, audio measurement and cutting.

- **Building:** Claude Code writes, tests and fixes the scripts on the laptop, inside the project folder.
- **Runtime judgement:** for every candidate clip, the script calls claude -p to rate it and write its title, description, hashtags and hook line.
- **Not the loop itself:** recording and cutting run in a Python script, not inside a Claude session. A Claude session can't watch video continuously, and running every 5 minute chunk through it would burn usage limits fast.
- **After the stream:** an optional claude -p call writes a short run report: clips made, scores, what was published.

## One time setup

Everything here is done once, before the first stream.

| Item | Purpose | Notes |
| --- | --- | --- |
| Python 3 | Runs the script | Plus numpy and OpenCV for audio and frame analysis |
| ffmpeg | Records, cuts, crops, mixes audio | Free |
| yt-dlp | Reads the live stream from YouTube | Update it often; YouTube changes break old versions |
| Audio analysis (numpy) | Measures loudness, sharp bangs and steady music from the sound itself | Signal processing only; no AI model, since the Claude CLI can't hear audio |
| Google Cloud project + YouTube Data API | Uploads and schedules Shorts | Apply for the API audit early; until it passes, uploads stay private |
| Claude CLI (Claude Code) | AI judge, titles and hook text at runtime; also used to build and maintain the scripts | Signed in on the laptop; each call counts toward your plan's usage limits |
| Royalty-free music folder | Replaces in-game radio in clips | Only tracks licensed for monetized use, e.g. YouTube Audio Library |
| HUD reference frames | Detects wanted stars, Wasted, Busted, Mission passed | Example frames from the first session, added to the Claude CLI prompt |

On the PS5: stream to YouTube at 720p, and keep the stream archive (VOD) on.

On the Windows laptop: install Python, ffmpeg and yt-dlp with winget, run the Claude CLI from PowerShell, and set the power plan so the laptop never sleeps while plugged in.

GPU: current NVIDIA drivers plus an ffmpeg build with CUDA, NVDEC and NVENC. In Windows graphics settings, set Python and ffmpeg to the NVIDIA GPU, so the laptop doesn't hand them to the built-in graphics. The startup check tests both before every stream.

### YouTube API setup

The API is free; uploads have their own default limit of 100 a day. Claude Code prepares the homepage and privacy policy pages and the upload login; the Google Cloud Console clicks and the audit form are yours.

| Step | Who | Details |
| --- | --- | --- |
| Create Google Cloud project, enable YouTube Data API v3 | You | No billing account needed |
| OAuth consent screen: External, scope youtube.upload, yourself as test user | You | APIs & Services > OAuth consent screen (Google Auth Platform) |
| Write homepage and privacy policy pages | Claude Code | index.html (what the app does) and privacy.html (uploads only to your own channel, token stored locally, no data shared), in a site/ folder |
| Publish the pages on GitHub Pages | Claude Code, with your GitHub login | Public repo <username>.github.io, pushed with the gh CLI after you sign in once |
| Fill in Branding: app name, support email, homepage URL, privacy policy URL, authorized domain <username>.github.io | You | Required before the app can switch to production |
| Audience > Publishing status > Publish app | You | In Testing mode the login token expires after 7 days; in production it doesn't |
| Create Desktop OAuth client, save as client_secret.json in the project folder | You | Claude Code adds it to .gitignore and blocks the repair agent from reading it |
| First sign-in from the upload script | You, once | Click Advanced > Go to app (unsafe) on the unverified app screen; the refresh token is saved locally |
| Apply for the YouTube API Services audit | You | Until approved, API uploads are locked to private; apply early |

## Recording

The script connects to the live stream once and saves it as consecutive 5 minute chunks, so each part of the stream is downloaded exactly once.

1. Start the script with the stream URL (or the channel's /live URL, so it waits until the stream goes live).
2. yt-dlp gets the stream's live playlist; ffmpeg copies it into 5 minute files without re-encoding.
3. When a chunk closes, it goes to the analysis queue.
4. If the connection drops or the stream link expires (after a few hours), the script fetches a fresh link and carries on with the next chunk number.
5. When the stream ends, the last partial chunk is processed and the script stops.

**Chunk deletion:** a chunk is deleted after its clips are cut AND the next chunk has been processed, so highlights that cross a chunk boundary aren't lost. At most about 10 minutes of footage sits on disk.

**Data use:** about 1.5 to 2 GB per hour at 720p, 3 to 4 GB at 1080p. The laptop's download shares the connection with the PS5's upload, so 720p is the safer default.

**Delay:** a clip is ready roughly 5 to 7 minutes after the moment happens.

## Highlight detection and scoring

Each candidate moment gets a score from 0 to 100, using only game audio and video, measured against the stream's own recent average rather than fixed levels.

First, each moment is sorted into one of two tracks: **cutscene** if black letterbox bars are present, otherwise **gameplay**.

### How the Claude CLI reads the video

All visual understanding is done by the Claude CLI; audio is measured with plain signal processing.

1. For each 5 minute chunk, ffmpeg saves one frame every 3 seconds (about 100 frames), small and timestamped.
2. The frames are tiled into 5 grid images of 20 frames each.
3. One claude -p call per chunk reads the grids and returns JSON per timestamp: cutscene or gameplay, HUD events, on-screen title text, menus or loading screens, and how much action is on screen.
4. That JSON feeds the visual signals below. Letterbox bars and motion are also measured directly as a cheap cross-check.

That's about 12 calls per hour of stream, plus judge calls.

### Gameplay track

| Signal | What it measures | Weight |
| --- | --- | --- |
| Audio spike | Sudden jump above the game's recent loudness | 30 |
| On-screen action | Motion and scene-change rate, screen shake, flashes | 25 |
| HUD events | Wanted stars appearing or rising; Wasted, Busted, Mission passed text (read by the Claude CLI) | 20 |
| Sound type | Sharp bursts in the waveform (gunfire, explosions, crashes) | 15 |
| Duration | Action sustained for several seconds, not a single bang | 10 |

### Cutscene track

| Signal | What it measures | Weight |
| --- | --- | --- |
| Title cards and text | Large centred text: mission titles, location names, credits (read by the Claude CLI) | 30 |
| Spectacle | Wide shots, big motion, fast cuts, explosions | 25 |
| Newness | First appearance of a scene type or location scores higher | 25 |
| Music swell | Rising score in the soundtrack | 20 |

Long cutscenes are trimmed to their strongest 30 to 45 seconds.

### Penalties (both tracks)

- Menus, map, pause and loading screens: static frames, little motion.
- In-game radio: steady, tonal sound over a long stretch is flagged as music so it doesn't count as action.
- Repeats: moments too close to an already selected clip.

### Optional AI judge

Clips scoring 60 or more are passed to the Claude CLI in headless mode (claude -p) with the paths of 3 or 4 saved frames and the detected events, returning JSON. Claude rates them 1 to 10 for "would a stranger watch this", which filters false spikes, and writes the title and the on-screen hook line. Final score = 70% signal score + 30% judge rating.

## Clip production

Each selected moment becomes a finished vertical Short in four steps, all done with ffmpeg on the GPU (see Compute, memory and recovery).

1. **Cut:** 20 to 45 seconds, starting a few seconds before the peak so the setup is kept. A moment crossing a chunk boundary is cut from both chunks joined.
2. **Vertical 9:16 at 1080 x 1920:** the full gameplay frame sits in the middle over a blurred, zoomed copy of itself, so the HUD and minimap stay visible. This is the chosen layout.
3. **Hook text:** one short line at the top, written by the Claude CLI from the frames and events (e.g. "5 stars in 30 seconds"). This adds original input to clips with no commentary.
4. **Audio:**
   - Clips without radio music keep the original game audio.
   - Where audio analysis flags in-game radio, the game audio is muted for that part and a royalty-free track from the local folder plays instead.
   - Cutscenes keep their audio by default. If a scene type keeps getting claimed, its score is ducked under a royalty-free track.

Finished clips go to a ready folder with a small JSON file holding the score, track, title, description and hashtags.

## Compute, memory and recovery

All video work runs on the GPU only, the whole pipeline is held to 2 GB of RAM, crashes restart automatically, and a Claude CLI agent fixes code that keeps failing.

### GPU only

| Step | Runs on | How |
| --- | --- | --- |
| Recording | Neither (copy only) | ffmpeg copies the stream as is; no decoding, almost no CPU |
| Frame grids for Claude | GPU | Hardware decode, GPU scaling; only the small finished frames come back to RAM |
| Cutting and 9:16 layout | GPU | Hardware decode; the blur is made by shrinking and re-enlarging the frame on the GPU, then overlaying the full frame |
| Hook text | GPU | The text is drawn once as a small image, then overlaid on the GPU |
| Encoding | GPU | NVIDIA NVENC hardware encoder |
| Audio measurement, music swap | CPU | Too light to need the GPU: a few seconds per chunk |

- **No CPU fallback.** If the GPU encoder or decoder isn't available, the job waits and retries; it never silently switches to the CPU.
- **GPU memory:** no limit set. A startup check confirms the hardware encoder works before the stream starts.

### 2 GB RAM cap

- The recorder, the processing worker, every ffmpeg run and every Claude CLI call run inside one Windows job object with a hard 2 GB memory limit, so their combined use can't go above it.
- To stay well below the limit, chunks are processed one at a time, frames are handled in small batches, and audio is read in pieces instead of all at once.
- The supervisor (below) runs outside the limit, but it's a small script using about 50 MB.

### Restart mechanism

- **Separate recorder and worker.** A processing crash never stops the recording; new chunks keep landing while the worker restarts.
- **Supervisor:** watches both processes and restarts any that exits, with a short wait that grows after repeated crashes.
- **Heartbeat:** each process updates a heartbeat file every 30 seconds; if it goes silent for 2 minutes it counts as hung, gets killed and restarted.
- **Resume, not redo:** a state file records which chunks are processed and what's in the publish queue, so a restart carries on where it stopped.
- **Autostart:** the supervisor is registered with Windows Task Scheduler, so it starts again after a reboot.

### Claude CLI repair agent

If the same error causes 3 crashes within 10 minutes, the supervisor launches a Claude CLI agent to fix it.

1. Publishing pauses, and chunks are kept instead of deleted, so nothing is lost while the fix happens.
2. The agent gets the error, the log tail and the project folder, and works only inside that folder.
3. It fixes the code and must pass the test suite, run on a saved sample chunk, before its fix is kept.
4. Every fix is a git commit, so a bad change can be rolled back.
5. The supervisor restarts the worker with the fix, then processes the backlog of kept chunks.

**Limits on the agent:** it can't read the YouTube login token, change the threshold, cap or kill switch, or install software. It gets at most 3 fix attempts per stream; after that, recording continues, publishing stays paused, and the problem is written to the log for you.

### Claude CLI model and effort

Every Claude CLI call uses Sonnet 5.5 (claude-sonnet-5-5), set explicitly on each call.

| Task | Effort | Why |
| --- | --- | --- |
| Reading frame grids | Low | Spotting HUD events and cutscenes needs speed, not deep reasoning |
| Judging clips, writing titles and hooks | Medium | Taste and wording matter, and calls are few |
| Repair agent | High | Debugging unattended code needs care, and it runs rarely |

## Automatic publishing

Clips above the threshold are uploaded through the YouTube Data API as scheduled posts, best first, within a daily cap. Starting values, to be tuned during calibration:

| Final score | Action |
| --- | --- |
| 75 or more | Queued for publishing |
| 50 to 74 | Uploaded as unlisted, kept as a backup |
| Under 50 | Deleted |

- **Daily cap:** 10 Shorts in total, with a separate cap of 2 or 3 for cutscene clips, so the channel isn't only story footage.
- **Schedule:** one Short every 2 to 3 hours. Clips left over from a long stream roll into the next day's queue, highest scores first.
- **Metadata:** title, description and hashtags from the Claude CLI, always including #Shorts and the game name. Story clips get "spoiler" in the title.
- **Upload quota:** the API is free, and uploads have their own default limit of 100 a day, so 10 a day fits with no quota increase.
- **Kill switch:** creating a file named PAUSE in the project folder stops all uploads at once; deleting it resumes them.
- **Log:** every clip, score and upload result is written to a log file for later tuning.

## Calibration

The first one or two streams run in test mode: everything runs except uploads, so the scoring can be tuned before it publishes on its own.

1. Save example frames of the HUD (wanted stars, Wasted, Busted, Mission passed) from the first session and add them to the Claude CLI prompt.
2. Run a full stream in test mode; all clips land in the ready folder with their scores.
3. Watch the top 20 and bottom 20 clips once, and mark which ones you'd actually post.
4. Give those marks to Claude Code, which adjusts the weights, penalties and threshold to match.
5. Repeat once if needed, then switch test mode off. From then on there's no review step.

## Risks and limits

| Risk | Impact | Mitigation |
| --- | --- | --- |
| YouTube API audit not passed | Uploads stuck as private; no public auto-posting | Apply as soon as the Google project exists |
| Reused content policy | No-commentary clips and cutscenes may be refused monetization | Hook text, varied editing, daily cap |
| Content ID claims | Revenue on that clip goes to the claimant, or the clip is blocked in some regions | Radio replacement; expect some claims on cutscenes |
| Wrong picks with no review | Dull clips, spoilers, or a stray notification on screen get posted | Calibration, AI judge, kill switch |
| Cutscene saturation | Many channels post the same scenes in launch week | Separate cutscene cap; favour gameplay |
| Stream link expiry or yt-dlp breaking | Recording stops mid-stream | Auto reconnect; keep yt-dlp updated |
| Laptop or network load | PS5 stream stutters, or processing falls behind | Record at 720p; small frame grids; no local AI models |
| Claude CLI usage limits | Judge calls fail on long streams | One grid call per chunk; judge only clips scoring 60 or more; fall back to audio and motion scores if limits are hit |
| Laptop sleeping or closing | Script stops | Disable sleep while streaming |

## Build phases and open decisions

Build in five phases with Claude Code, each tested on a recorded stream before moving on.

1. **Recorder:** live stream to 5 minute chunks, reconnects, chunk cleanup. Test on any live YouTube stream.
2. **Analysis:** audio measurement, GPU frame grids, the claude -p call per chunk, and both scoring tracks.
3. **Clip production:** cutting, vertical layout, hook text, radio replacement.
4. **Publishing:** homepage and privacy pages on GitHub Pages, OAuth login, scheduled uploads, caps, kill switch, log; then the supervisor, the 2 GB limit and the Claude CLI repair agent.
5. **Calibration:** test mode on the first one or two GTA 6 streams, then tuning.

Phases 1 to 3 can be ready before launch day and tested on other gameplay; HUD and cutscene tuning can only happen after launch.

### Decisions

- [x] Laptop OS: Windows
- [x] Stream quality: 720p
- [x] Publish threshold and daily cap: 75 and 10 a day
- [x] Vertical layout: blurred background
- [x] Source for royalty-free music: YouTube Audio Library (use tracks marked no attribution required)
- [x] Typical stream length: designed for 2 to 4 hour streams. That's 24 to 48 chunks and more strong clips than the cap of 10, so the queue carries the extras into the next days.
- [x] GPU make: NVIDIA (exact model still useful to set encoder quality)
