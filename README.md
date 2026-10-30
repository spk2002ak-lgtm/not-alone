# Not Alone — the AI working partner for print-on-demand sellers

"You're not in it alone anymore."

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026),
indie track. A seller uploads a photo of a shirt, and Not Alone cuts the
printed graphic out, writes the listing in the seller's own voice, and
grades it before anything ships — while remembering the seller across
sessions.

## What it does

1. **Persistent seller memory** (`src/memory.py`) — a SQLite seller profile
   (shop, niche, platform, voice notes, products) injected into the
   Responses API `instructions` at session start, with `previous_response_id`
   chaining inside the session and write-back at session end. The API gives
   continuity within a session; the database gives identity across sessions.
2. **SAM 3.1 vision** (`src/sam.py`, `src/decode_cutout.py`) — a text prompt
   ("the printed graphic") segments the design from a photo, and the mask
   is decoded and composited into a transparent PNG cutout.
3. **Muse Spark listing agent** (`src/listing.py`) — writes a title,
   description, and tags in the seller's voice from the stored profile. It
   never invents prices, sales figures, materials, sizes, or products the
   seller has not confirmed.
4. **Verifier agent** (`src/verifier.py`) — grades every listing before it
   ships: a model grade plus deterministic local checks (invented-price,
   voice, completeness, product-consistency rules). A provable violation
   can't be outvoted by a model pass. Each flag names the rule, the
   evidence, and the fix.
5. **One-screen UI** (`src/app.py`, `src/index.html`) — upload a shirt
   photo → cutout preview → listing kit → save to memory. Stdlib only,
   built to be screen-recorded.

## Status (Oct 30, 2026)

All five pieces are built and live-tested against the Meta Model API on
free credits — memory injection proven, chaining proven, SAM cutout
verified against the source photo, listing drafted in the seller's voice,
verifier passing it with zero flags. Total API spend for the whole build
so far is pennies (about 6.4 cents); a full happy-path run costs roughly
a dime. The demo video and submission land Nov 2–4.

## Quickstart

```bash
pip install -r requirements.txt
python3 src/app.py          # serves http://127.0.0.1:8901/
```

Live Meta Model API calls in the demo environment go through a small
local helper (base URL `https://api.meta.ai/v1`, Bearer auth). The call
sites are single functions — `listing.call_responses` and the SAM call
inside `sam.cut_graphic` — so pointing them at your own client and key
is a one-function change. Models used: **Muse Spark 1.3** (Responses API)
and **SAM 3.1**.

## Layout

- `src/memory.py` — seller profile + session store
- `src/sam.py` — SAM 3.1 cutout (prompt → mask → transparent PNG)
- `src/decode_cutout.py` — SSE parsing, mask decoding, compositing
- `src/listing.py` — listing agent
- `src/verifier.py` — verifier agent
- `src/app.py`, `src/index.html` — one-screen UI
- `vendor/meta_sam_parser/` — the official SAM 3 segmentation-protocol
  decoder (dependency-free, vendored so a clone runs as-is)
- `runs/` — demo artifacts: Day-1 and live cutouts, the listing draft +
  grade JSON, raw SAM streams
- `test-inputs/` — the shirt photo used for the first cutout test

## Not in this repo

API keys, `.env` files, and the SQLite database (`*.db`) are gitignored
and never committed. Nothing here can spend money or identify an account
on its own.
