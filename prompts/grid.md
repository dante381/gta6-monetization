You are labelling frames from a live stream of the video game {{GAME}} so that an automatic clipper can find highlights. Answer with JSON that matches the provided schema and nothing else.

## The images

There are {{N_GRIDS}} grid image(s). Open every one with the Read tool before answering:

{{GRID_LIST}}

Each grid has {{COLS}} columns and {{ROWS}} rows, up to {{PER_GRID}} tiles. Read the tiles left to right, then top to bottom. Every tile is one frame of the stream, one frame every {{INTERVAL}} seconds. The yellow label `t=<seconds>` in the black strip directly above a tile is that frame's timestamp. The label strip is not part of the game picture. Grey cells with no label are empty padding: skip them.

Return exactly one entry in `frames` for every timestamp below, in this order. `t` is the number from the tile's label.

TIMESTAMPS: {{TIMESTAMPS}}

## What to fill in for each tile

- `kind`: what the screen shows.
  - `gameplay`: the player is in control. A HUD is usually visible (minimap or radar in a bottom corner, wanted stars, money or weapon at the top right), or the camera follows the player character or vehicle.
  - `cutscene`: a story scene, usually with no HUD, cinematic camera, often black letterbox bars at the top and bottom.
  - `menu`: pause menu, settings, phone or character screen, store or any other full-screen UI.
  - `loading`: loading screen, black or almost black frame, splash art with a spinner or progress text.
  - `map`: the full-screen world map.
  - `other`: anything else (stream overlay, "be right back" or starting screen, desktop, a different game screen).
- `hud_events`: HUD events visible in this tile (empty list if none). Only use these values:
  - `wanted_up`: the wanted level is rising: stars flashing, or more stars than the previous tile.
  - `wanted_stars`: wanted stars shown (police are after the player), steady.
  - `wasted`: big centred "WASTED" text (the player died).
  - `busted`: big centred "BUSTED" text (the player was arrested).
  - `mission_passed`: "MISSION PASSED" or a similar success banner.
  - `mission_failed`: "MISSION FAILED" or a similar failure banner.
  - `notification`: a pop-up message, help text box, phone text, or big on-screen message other than the ones above.
  - `other`: another clearly visible HUD event.
- `wanted_stars`: how many wanted stars are filled in (0 to 5). In {{GAME}} they sit at the top right. Use 0 when there are none or you cannot tell.
- `title_text`: large centred text such as a mission title, a location or area name, a chapter card or credits. Copy the text as written (short). Use null when there is none. Do not copy small HUD text, subtitles or stream overlays.
- `action`: how much is happening on screen, 0 to 10.
  - 0: static menu or loading screen.
  - 2–3: walking or standing, calm driving, dialogue.
  - 5–6: fast driving, a chase starting, a fight.
  - 8–10: shootout, explosion, crash, police chase with stars, big stunt.
- `scene_tag`: a short 2–4 word reusable label for the place or situation, lowercase, for example "city street chase", "beach", "strip club interior", "gas station", "highway drive", "loading screen", "pause menu". Give the same place or situation the same tag every time, so that repeats can be spotted. Use null only when the tile is unreadable.

{{HUD_REFS}}

## Rules

- Look at every tile. Tiles are small, so judge from shapes, colours and layout; do not guess text you cannot read.
- Neighbouring tiles are {{INTERVAL}} seconds apart. Use them for context, for example to spot rising wanted stars or a cutscene that continues across tiles.
- If this does not look like {{GAME}} (for example test footage or another game), still label honestly: use the closest `kind`, report only events you can actually see, and describe the scene in `scene_tag`.
- Output one entry per timestamp, with every field present.
