#!/usr/bin/env python3
"""Decode a SAM 3.1 mask line from a saved SSE stream and composite the
transparent cutout onto the source photo.

The pipeline is importable so sam.py can reuse it for live cuts:
  output_text_from_sse() -> parse_tokens() -> decode_raster()
  -> composite_cutout()

Usage (standalone, same as before):
  decode_cutout.py --sse runs/sam-first-run.sse \
      --photo test-inputs/hose-water-shirt-photo.jpg \
      --out runs/hose-water-cutout.png --qa runs/hose-water-cutout-qa.png
"""
import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "vendor"))
sys.path.insert(0, str(_HERE))
from meta_sam_parser import SegmentationMask, decode_mask_to_raster

BOX_RE = re.compile(
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>")
MASK_RE = re.compile(r"<\|mask;x=(\d+);y=(\d+);data=([^|]+)\|>")


def output_text_from_sse(path):
    """Join the response.output_text.delta events of a saved SSE stream."""
    deltas = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                ev = json.loads(payload)
            except ValueError:
                continue
            if ev.get("type") == "response.output_text.delta":
                deltas.append(ev.get("delta", ""))
    return "".join(deltas)


def parse_tokens(text):
    """Extract the box and mask tokens from SAM output text.

    Returns (box, mask) where box is (x1, y1, x2, y2, frame_w, frame_h)
    and mask is {"width", "height", "encoding", "payload"}. The mask
    raster is box-local: width == x2 - x1 + 1, height == y2 - y1 + 1.
    Raises ValueError when either token is missing.
    """
    box = BOX_RE.search(text)
    mask = MASK_RE.search(text)
    if not box or not mask:
        raise ValueError("no box/mask token found in output text")
    x1, y1, x2, y2, fw, fh = (int(v) for v in box.groups())
    height_s, width_s, rest = mask.group(3).split(",", 2)
    marker, payload = rest[0], rest
    encoding = "lossless" if marker == "~" else "one_bit"
    return (x1, y1, x2, y2, fw, fh), {
        "width": int(width_s), "height": int(height_s),
        "encoding": encoding, "payload": payload}


def decode_raster(mask):
    """Decode a parsed mask token into row-major bytes of 0/1."""
    raster = decode_mask_to_raster(SegmentationMask(
        encoding=mask["encoding"], payload=mask["payload"],
        width=mask["width"], height=mask["height"]))
    if len(raster) != mask["width"] * mask["height"]:
        raise ValueError("decoded raster size does not match mask dims")
    return raster


def mask_coverage(raster):
    """Share of mask pixels set, as a percentage (0-100)."""
    if not raster:
        return 0.0
    return 100.0 * sum(raster) / len(raster)


def composite_cutout(photo_path, box, mask, raster):
    """Composite the transparent cutout: photo pixels where mask is set.

    box is the parse_tokens box tuple; mask holds the raster dims.
    Returns a Pillow RGBA image the size of the source photo.
    """
    from PIL import Image
    x1, y1, _x2, _y2, fw, fh = box
    width, height = mask["width"], mask["height"]
    photo = Image.open(photo_path).convert("RGBA")
    if photo.size != (fw, fh):
        raise ValueError("photo size %s != frame size %s" % (
            photo.size, (fw, fh)))
    cutout = Image.new("RGBA", photo.size, (0, 0, 0, 0))
    px_photo = photo.load()
    px_out = cutout.load()
    for row in range(height):
        base = row * width
        y = y1 + row
        if y >= fh:
            break
        for col in range(width):
            if raster[base + col]:
                x = x1 + col
                if x < fw:
                    px_out[x, y] = px_photo[x, y]
    return cutout


def cutout_from_sse(sse_path, photo_path, out_path, qa_path=None):
    """Full pipeline for a saved SSE stream. Returns a result dict."""
    text = output_text_from_sse(sse_path)
    box, mask = parse_tokens(text)
    raster = decode_raster(mask)
    cutout = composite_cutout(photo_path, box, mask, raster)
    cutout.save(out_path)
    if qa_path:
        from PIL import Image
        qa = Image.new("RGBA", cutout.size, (255, 0, 255, 255))
        qa.alpha_composite(cutout)
        qa.convert("RGB").save(qa_path)
    return {
        "cutout": str(out_path),
        "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "frame_w": box[4], "frame_h": box[5]},
        "mask": {"width": mask["width"], "height": mask["height"],
                 "encoding": mask["encoding"]},
        "coverage": round(mask_coverage(raster), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", required=True)
    parser.add_argument("--photo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--qa", required=True)
    args = parser.parse_args()

    result = cutout_from_sse(args.sse, args.photo, args.out, args.qa)
    box = result["box"]
    mask = result["mask"]
    print("box: x1=%d y1=%d x2=%d y2=%d frame=%dx%d" % (
        box["x1"], box["y1"], box["x2"], box["y2"],
        box["frame_w"], box["frame_h"]))
    print("mask: %dx%d %s, coverage %.1f%%" % (
        mask["width"], mask["height"], mask["encoding"],
        result["coverage"]))
    print("wrote", args.out, "and", args.qa)


if __name__ == "__main__":
    main()
