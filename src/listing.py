"""Muse Spark listing agent for Not Alone.

Writes a Printify listing in the seller's own voice, using the SQLite
seller profile (memory.py) as context.

Flow:
  1. build_listing_prompt(): profile -> memory.render_instructions()
     plus a listing task (title, description, tags) as the Responses input.
  2. call_responses(): the single API-call site. It shells out to the
     meta-model-api connector helper (spk2002ak test key via the
     connector); the verifier module reuses this instead of adding a
     second call path.
  3. draft_listing(): prompt + call + parse into title, description,
     tags, response id, and usage.

Honesty rule, stated in both the profile instructions and the listing
prompt: never invent prices, sales figures, or products the seller has
not confirmed. This module writes listing text only — it never publishes
or lists anything on Printify.
"""

import json
import os
import subprocess
import sys

try:
    from memory import render_instructions
except ImportError:  # imported from elsewhere; src/ holds memory.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from memory import render_instructions

MODEL = "muse-spark-1.3"

_HELPER = os.path.expanduser(
    "~/workspace/custom-connectors/custom.meta-model-api/bin/meta_model.py")
_CREDENTIALS_DIR = "/opt/muse/skills/custom-connector-setup/bin"


def build_listing_prompt(profile, product_title, design_notes=None):
    """Return (instructions, input_text) for a listing draft.

    Pure assembly — no API call — so prompt shape can be tested free.
    """
    instructions = render_instructions(profile)
    lines = [
        "Write a Printify listing for the product \"%s\"." % product_title,
        "Return exactly three sections, in this order, with these labels:",
        "TITLE: one listing title.",
        "DESCRIPTION: the listing description, in the seller's voice.",
        "TAGS: a comma-separated list of search tags.",
    ]
    if design_notes:
        lines.append("Design notes: %s." % design_notes)
    lines.append(
        "Do not invent a price, sales figures, materials, sizes, or any"
        " product the seller has not confirmed. If a detail is not in the"
        " profile or the design notes, leave it out rather than guessing.")
    return instructions, "\n".join(lines)


def call_responses(input_text, instructions, previous_response_id=None):
    """Call the Responses API via the connector helper.

    Returns (text, response_id, usage). This is the module's only API
    call site; the verifier reuses it.
    """
    if not os.path.exists(_HELPER):
        raise RuntimeError("meta-model-api helper not found: %s" % _HELPER)
    env = dict(os.environ)
    env["PYTHONPATH"] = _CREDENTIALS_DIR + os.pathsep + env.get(
        "PYTHONPATH", "")
    cmd = [sys.executable, _HELPER, "responses",
           "--input", input_text, "--instructions", instructions]
    if previous_response_id:
        cmd += ["--previous-response-id", previous_response_id]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env=env, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError("helper failed: %s" % proc.stderr.strip()[-500:])
    try:
        result = json.loads(proc.stdout)
    except ValueError:
        raise RuntimeError("helper returned non-JSON output")
    if result.get("status") != 200:
        raise RuntimeError("API status %s" % result.get("status"))
    return result["text"], result["id"], result.get("usage", {})


def parse_listing_text(text):
    """Parse TITLE/DESCRIPTION/TAGS sections out of model output.

    Falls back gracefully: missing labels mean the whole text becomes
    the description and the title/tags stay empty for the caller to fill.
    """
    sections = {"title": "", "description": "", "tags": []}
    upper = text.upper()
    title_at = upper.find("TITLE:")
    desc_at = upper.find("DESCRIPTION:")
    tags_at = upper.find("TAGS:")
    if title_at < 0 or desc_at < 0 or tags_at < 0:
        sections["description"] = text.strip()
        return sections
    title = text[title_at + len("TITLE:"):desc_at].strip()
    if title.startswith(('"', "'")) and title.endswith(('"', "'")) \
            and len(title) >= 2:
        title = title[1:-1]
    sections["title"] = " ".join(title.split())
    sections["description"] = text[desc_at + len("DESCRIPTION:"):tags_at].strip()
    raw_tags = text[tags_at + len("TAGS:"):].strip()
    sections["tags"] = [t.strip().lstrip("#") for t in raw_tags.split(",")
                        if t.strip()]
    return sections


def draft_listing(profile, product_title, design_notes=None,
                  previous_response_id=None):
    """Draft a listing in the seller's voice.

    Returns a dict with title, description, tags, response_id, usage,
    and the raw model text.
    """
    instructions, input_text = build_listing_prompt(
        profile, product_title, design_notes)
    text, response_id, usage = call_responses(
        input_text, instructions, previous_response_id)
    listing = parse_listing_text(text)
    listing["response_id"] = response_id
    listing["usage"] = usage
    listing["raw_text"] = text
    return listing
