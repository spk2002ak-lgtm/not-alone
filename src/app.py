"""Not Alone one-screen demo UI.

Stdlib only (http.server). Serves one page with four stages:
  1. upload a shirt photo
  2. cutout preview
  3. listing kit (title, description, tags)
  4. save to memory

Wiring (reuses the existing modules, no duplicated API paths):
  - memory: profile load + product save (SQLite, $0)
  - verifier.local_checks: deterministic grade, $0, used by the save path
  - listing.draft_listing: live Muse Spark draft, behind the Draft button
  - verifier.grade_listing: live model grade, behind the Grade button
  - sam.cut_graphic: live SAM cut, behind the Cut button; the saved
    Day-1 cutout stays available via "Show saved cutout"

Demo artifacts served with zero API calls:
  - /demo/photo  -> test-inputs/hose-water-shirt-photo.jpg
  - /demo/cutout -> runs/hose-water-cutout.png
  - /api/state  -> profile + latest runs/listing-*.json draft

Saved kits: POST /api/save writes runs/listing-<slug>.json with the
draft + grade, and records the product in memory, so later grades read
the saved draft instead of re-typed text.

Run:  python3 src/app.py [port]   (default port 8901, localhost only)
Env:  NOT_ALONE_DB overrides the SQLite path (tests use a temp DB).
      NOT_ALONE_PORT overrides the default port.

This server never publishes or lists anything on Printify, never mints
keys, and never touches the stusmethod account. Live buttons call the
Meta Model API through the existing connector helper only when clicked.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import memory
from verifier import local_checks

BASE = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("NOT_ALONE_DB", memory.DB_PATH))
DEFAULT_PORT = int(os.environ.get("NOT_ALONE_PORT", "8901"))

PHOTO = BASE / "test-inputs" / "hose-water-shirt-photo.jpg"
CUTOUT = BASE / "runs" / "hose-water-cutout.png"
RUNS = BASE / "runs"


def slug(text):
    """URL/file-safe slug from a title."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "draft"


def _connect():
    return memory.init_db(DB_PATH)


def load_profile():
    """First seller profile in the DB, or None when the DB is empty."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id FROM sellers ORDER BY id LIMIT 1").fetchone()
        if row is None:
            return None
        return memory.get_profile(conn, row["id"])
    finally:
        conn.close()


def latest_draft():
    """Most recently saved runs/listing-*.json, or None."""
    files = sorted(RUNS.glob("listing-*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        with open(files[0]) as fh:
            data = json.load(fh)
        data["_file"] = files[0].name
        return data
    except (ValueError, OSError):
        return None


def grade_locally(draft, profile):
    """Free deterministic grade; the save path never calls the API."""
    flags = local_checks(draft, profile or {})
    return {
        "verdict": "NEEDS_FIX" if flags else "PASS",
        "source": "local_checks",
        "flags": flags,
        "note": "Full model grade runs on demand via the Grade button.",
    }


def save_kit(draft, profile):
    """Write runs/listing-<slug>.json and record the product in memory.

    Returns (filename, grade, product_titles). Safe to call twice: the
    product row is added only when its title is not already stored.
    """
    title = (draft.get("title") or "").strip()
    grade = grade_locally(draft, profile)
    record = {
        "title": title,
        "description": draft.get("description") or "",
        "tags": draft.get("tags") or [],
        "product": draft.get("product") or title,
        "grade": grade,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "source": "not-alone-ui",
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    filename = "listing-%s.json" % slug(title)
    with open(RUNS / filename, "w") as fh:
        json.dump(record, fh, indent=2)
    titles = []
    if profile is not None and title:
        conn = _connect()
        try:
            existing = [p["title"] for p in profile.get("products", [])]
            if title not in existing:
                memory.add_product(conn, profile["id"], title,
                                   profile.get("niche"))
            titles = [p["title"] for p in
                      memory.get_profile(conn, profile["id"])["products"]]
        finally:
            conn.close()
    return filename, grade, titles


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output readable
        pass

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(404, "not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode())
        except ValueError:
            return None

    def do_GET(self):
        if self.path == "/":
            self._send_file(SRC / "index.html", "text/html; charset=utf-8")
        elif self.path == "/demo/photo":
            self._send_file(PHOTO, "image/jpeg")
        elif self.path == "/demo/cutout":
            self._send_file(CUTOUT, "image/png")
        elif self.path == "/api/live-cutout":
            self._send_file(RUNS / "live-cutout.png", "image/png")
        elif self.path == "/api/state":
            profile = load_profile()
            self._send_json({
                "profile": profile,
                "draft": latest_draft(),
                "photo": "/demo/photo",
                "cutout": "/demo/cutout",
            })
        else:
            self.send_error(404, "not found")

    def do_POST(self):
        body = self._read_json()
        if body is None:
            self._send_json({"ok": False, "error": "invalid JSON"}, 400)
            return
        profile = load_profile()

        if self.path == "/api/local-grade":
            draft = body.get("draft") or body
            self._send_json({"ok": True,
                             "grade": grade_locally(draft, profile)})
        elif self.path == "/api/save":
            draft = body.get("draft") or body
            if isinstance(draft.get("tags"), str):
                draft["tags"] = [t.strip() for t in
                                 draft["tags"].split(",") if t.strip()]
            filename, grade, titles = save_kit(draft, profile)
            self._send_json({"ok": True, "file": filename, "grade": grade,
                             "memory_products": titles})
        elif self.path == "/api/draft":
            # LIVE call, only when the Draft button is clicked.
            from listing import draft_listing
            if profile is None:
                self._send_json({"ok": False,
                                 "error": "no seller profile"}, 400)
                return
            try:
                draft = draft_listing(profile, body.get("product_title", ""),
                                      body.get("design_notes"))
            except Exception as exc:  # surface, never invent a draft
                self._send_json({"ok": False, "error": str(exc)}, 502)
                return
            self._send_json({"ok": True, "draft": draft})
        elif self.path == "/api/grade":
            # LIVE call, only when the Grade button is clicked.
            from verifier import grade_listing
            if profile is None:
                self._send_json({"ok": False,
                                 "error": "no seller profile"}, 400)
                return
            try:
                grade = grade_listing(body.get("draft") or body, profile)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, 502)
                return
            self._send_json({"ok": True, "grade": grade})
        elif self.path == "/api/cutout":
            # LIVE call, only when the Cut button is clicked.
            from sam import cut_graphic
            try:
                result = cut_graphic(PHOTO,
                                     out_path=RUNS / "live-cutout.png",
                                     sse_path=RUNS / "sam-live.sse")
            except Exception as exc:  # surface, never fake a cutout
                self._send_json({"ok": False, "error": str(exc)}, 502)
                return
            self._send_json({"ok": True, "cutout": "/api/live-cutout",
                             "box": result["box"],
                             "coverage": result["coverage"],
                             "file": "live-cutout.png"})
        else:
            self.send_error(404, "not found")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    port = int(argv[0]) if argv else DEFAULT_PORT
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("Not Alone demo UI: http://127.0.0.1:%d/ (Ctrl-C to stop)" % port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
