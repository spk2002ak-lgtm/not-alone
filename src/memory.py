"""Persistent seller memory for Not Alone.

SQLite seller profile, injected at session start. No API calls here.

Tables:
  sellers(id, name, shop_url, niche, platform, voice_notes, created_at,
          updated_at)
  products(id, seller_id, title, niche, created_at)
  sessions(id, seller_id, previous_response_id, started_at)

Session flow (see build plan):
  1. start: load profile -> render_instructions() -> Responses `instructions`
  2. within session: chain turns with `previous_response_id`
  3. end: save the last response id on the session; update_profile()
     with anything the seller confirmed (niche, voice, products)
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "not_alone.db"

# Only these seller columns may be changed through update_profile().
_PROFILE_FIELDS = {"name", "shop_url", "niche", "platform", "voice_notes"}


def init_db(path=DB_PATH):
    """Create the seller-profile tables if they do not exist yet."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sellers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            shop_url TEXT,
            niche TEXT,
            platform TEXT,
            voice_notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            seller_id INTEGER REFERENCES sellers(id),
            title TEXT,
            niche TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            seller_id INTEGER REFERENCES sellers(id),
            previous_response_id TEXT,
            started_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    return conn


def get_or_create_seller(conn, name, **fields):
    """Return the seller id for `name`, creating the row on first use."""
    row = conn.execute(
        "SELECT id FROM sellers WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    clean = {k: v for k, v in fields.items() if k in _PROFILE_FIELDS}
    cols = ", ".join(["name"] + list(clean))
    placeholders = ", ".join("?" * (1 + len(clean)))
    cur = conn.execute(
        "INSERT INTO sellers (%s) VALUES (%s)" % (cols, placeholders),
        (name, *clean.values()))
    conn.commit()
    return cur.lastrowid


def get_profile(conn, seller_id):
    """Return the seller row + products for session-start injection."""
    seller = conn.execute(
        "SELECT * FROM sellers WHERE id = ?", (seller_id,)).fetchone()
    if seller is None:
        return None
    products = conn.execute(
        "SELECT title, niche FROM products WHERE seller_id = ?"
        " ORDER BY id", (seller_id,)).fetchall()
    profile = dict(seller)
    profile["products"] = [dict(p) for p in products]
    return profile


def update_profile(conn, seller_id, **fields):
    """Update allowlisted seller profile fields after a session."""
    clean = {k: v for k, v in fields.items() if k in _PROFILE_FIELDS}
    if not clean:
        return 0
    assignments = ", ".join("%s = ?" % k for k in clean)
    cur = conn.execute(
        "UPDATE sellers SET %s, updated_at = CURRENT_TIMESTAMP"
        " WHERE id = ?" % assignments, (*clean.values(), seller_id))
    conn.commit()
    return cur.rowcount


def add_product(conn, seller_id, title, niche=None):
    """Record a product under the seller; returns the product id."""
    cur = conn.execute(
        "INSERT INTO products (seller_id, title, niche) VALUES (?, ?, ?)",
        (seller_id, title, niche))
    conn.commit()
    return cur.lastrowid


def _sentence(value):
    """Return `value` with exactly one trailing period."""
    return value.rstrip().rstrip(".") + "."


def render_instructions(profile):
    """Render a seller profile into Responses-API `instructions` text.

    This is the session-start injection: everything the model needs to
    sound like this seller's partner, without resending it every turn
    (within-session memory comes from `previous_response_id` chaining).
    """
    lines = [
        "You are Not Alone, a working partner for print-on-demand sellers.",
        "Seller: %s." % profile["name"],
    ]
    if profile.get("shop_url"):
        lines.append("Shop: %s" % _sentence(profile["shop_url"]))
    if profile.get("niche"):
        lines.append("Niche: %s" % _sentence(profile["niche"]))
    if profile.get("platform"):
        lines.append("Sells on: %s" % _sentence(profile["platform"]))
    if profile.get("voice_notes"):
        lines.append("Voice notes: %s" % _sentence(profile["voice_notes"]))
    if profile.get("products"):
        titles = ", ".join(p["title"] for p in profile["products"])
        lines.append("Existing products: %s" % _sentence(titles))
    lines.append(
        "Write in the seller's voice. Never invent prices, sales figures,"
        " or products the seller has not confirmed.")
    return " ".join(lines)


def start_session(conn, seller_id):
    """Open a session row; returns the session id."""
    cur = conn.execute(
        "INSERT INTO sessions (seller_id) VALUES (?)", (seller_id,))
    conn.commit()
    return cur.lastrowid


def save_session_response(conn, session_id, response_id):
    """Remember the latest response id so the next turn can chain to it."""
    conn.execute(
        "UPDATE sessions SET previous_response_id = ? WHERE id = ?",
        (response_id, session_id))
    conn.commit()


def get_last_response(conn, session_id):
    """Return the session's latest response id, or None for a fresh chat."""
    row = conn.execute(
        "SELECT previous_response_id FROM sessions WHERE id = ?",
        (session_id,)).fetchone()
    return row["previous_response_id"] if row else None
