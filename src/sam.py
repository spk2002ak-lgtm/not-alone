"""SAM 3.1 cut module for Not Alone.

Cuts "the printed graphic" from a photo to transparency via text-prompt
segmentation. Shapes verified from official docs Oct 26 (dev.meta.ai)
and proven live Oct 30 (Day-1 run on the Hose Water photo).

Flow:
  1. cut_graphic(): shells out to the meta-model-api connector helper
     (spk2002ak test key via the connector; no key is stored here),
     which POSTs model sam-3.1 with input_text (one concept) +
     input_image and saves the raw SSE stream.
  2. The box + base85 mask tokens are parsed from the returned output
     text and decoded with the vendored meta-sam-parser package.
  3. Transparency is composited with Pillow (decode_cutout.py pipeline).

This module cuts images only — it never publishes or lists anything
on Printify. SAM pricing: $2.50 per 1,000 images ($0.0025 per cut).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import decode_cutout
except ImportError:  # imported from elsewhere; src/ holds decode_cutout.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import decode_cutout

_HELPER = os.path.expanduser(
    "~/workspace/custom-connectors/custom.meta-model-api/bin/meta_model.py")
_CREDENTIALS_DIR = "/opt/muse/skills/custom-connector-setup/bin"
_BASE = Path(__file__).resolve().parent.parent


def cut_graphic(photo_path, prompt="the printed graphic",
                out_path=None, sse_path=None):
    """Cut the prompted graphic from a photo to a transparent PNG.

    Runs one live SAM call via the connector helper, decodes the mask,
    and composites the cutout. Returns a dict with the cutout path, the
    saved SSE path, the box, the mask dims/encoding, the mask coverage
    percentage, the API usage, and the completed response id.
    """
    photo = Path(photo_path)
    if not photo.exists():
        raise ValueError("photo not found: %s" % photo)
    if out_path is None:
        out_path = _BASE / "runs" / ("%s-cutout.png" % photo.stem)
    if sse_path is None:
        sse_path = _BASE / "runs" / ("sam-%s.sse" % photo.stem)
    if not os.path.exists(_HELPER):
        raise RuntimeError("meta-model-api helper not found: %s" % _HELPER)
    env = dict(os.environ)
    env["PYTHONPATH"] = _CREDENTIALS_DIR + os.pathsep + env.get(
        "PYTHONPATH", "")
    cmd = [sys.executable, _HELPER, "sam",
           "--image", str(photo), "--prompt", prompt,
           "--out", str(sse_path)]
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
    box, mask = decode_cutout.parse_tokens(result.get("output_text", ""))
    raster = decode_cutout.decode_raster(mask)
    cutout = decode_cutout.composite_cutout(photo, box, mask, raster)
    cutout.save(out_path)
    return {
        "cutout": str(out_path),
        "sse": str(sse_path),
        "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "frame_w": box[4], "frame_h": box[5]},
        "mask": {"width": mask["width"], "height": mask["height"],
                 "encoding": mask["encoding"]},
        "coverage": round(decode_cutout.mask_coverage(raster), 1),
        "usage": result.get("usage", {}),
        "completed_id": result.get("completed_id"),
    }
