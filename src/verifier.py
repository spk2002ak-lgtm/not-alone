"""Verifier agent for Not Alone.

Grades a draft listing against the seller's profile and a small rule
set before it reaches the seller. BenchTruth-style: every flag names
the violated rule, the evidence in the draft, and a concrete fix.

Flow:
  1. build_verifier_prompt(): profile -> memory.render_instructions()
     plus the draft and the rule list as the Responses input.
  2. Call path: listing.call_responses() — the single API-call site,
     reused here instead of adding a second call path. One model call
     per grade.
  3. grade_listing(): deterministic local checks (free) plus the model
     verdict, merged into PASS / NEEDS_FIX plus flag cards.

Local checks catch what a pattern can prove: invented prices, empty
fields, "merch" where the profile says Merchandise, and a draft that
drifts from its own product. The model call judges what a pattern
cannot: voice match and any other invented detail. This module grades
text only — it never publishes or lists anything on Printify.
"""

import json
import os
import re
import sys

try:
    from listing import call_responses
    from memory import render_instructions
except ImportError:  # imported from elsewhere; src/ holds both modules
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from listing import call_responses
    from memory import render_instructions

# Rule ids are stable: flag cards and callers refer to them by id.
DEFAULT_RULES = [
    {
        "id": "no-invented-details",
        "statement": "No invented prices, sales figures, materials, sizes,"
        " or products the seller has not confirmed. Every concrete"
        " detail must come from the profile or the design notes.",
    },
    {
        "id": "voice-match",
        "statement": "Hook first, funny then useful. Say Merchandise,"
        " never merch. Fair prices, no hype and no guru-speak.",
    },
    {
        "id": "completeness",
        "statement": "Title, description, and tags must all be present"
        " and non-empty.",
    },
    {
        "id": "product-consistency",
        "statement": "The listing must actually be about the product it"
        " claims: title, description, and tags must agree with the"
        " product title.",
    },
]

_PRICE_RE = re.compile(r"\$\s?\d[\d,]*(\.\d{1,2})?")
_MERCH_RE = re.compile(r"\bmerch\b", re.IGNORECASE)
_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "your", "you",
    "our", "are", "was", "were", "has", "have", "been", "into", "over",
    "under", "about", "after", "before", "design",
}


def _draft_text(draft):
    """Join the draft's fields into one searchable string."""
    parts = [draft.get("title", ""), draft.get("description", "")]
    tags = draft.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    parts.extend(tags)
    return "\n".join(p for p in parts if p)


def _significant_words(text):
    """Lowercased content words (length > 3, stopwords removed)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def _flag(rule, evidence, fix):
    """One BenchTruth-style flag card."""
    return {"rule": rule, "evidence": evidence, "fix": fix}


def local_checks(draft, profile):
    """Deterministic, $0 checks. Returns a list of flag cards.

    Pure function — no API call — so it is tested free, including the
    negative test: a draft with an invented price must be flagged.
    """
    flags = []
    title = (draft.get("title") or "").strip()
    description = (draft.get("description") or "").strip()
    tags = draft.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tags = [t.strip() for t in tags if t.strip()]

    # Completeness: every field present and non-empty.
    for field, value in (("title", title), ("description", description),
                         ("tags", tags)):
        if not value:
            flags.append(_flag(
                "completeness",
                "%s is missing or empty." % field,
                "Fill in the %s before this listing ships." % field))

    # Invented prices: no price is confirmed in the profile, so any
    # dollar amount in the draft is an invention until the seller
    # confirms it.
    text = _draft_text(draft)
    price = _PRICE_RE.search(text)
    if price:
        flags.append(_flag(
            "no-invented-details",
            "Draft states a price: %r." % price.group(0),
            "Remove the price, or confirm it with the seller first and"
            " add it to the profile."))

    # Voice: "merch" where the profile says Merchandise. Gated on the
    # profile so a seller without that rule is not false-flagged.
    voice_notes = (profile.get("voice_notes") or "").lower()
    if "merchandise" in voice_notes:
        merch = _MERCH_RE.search(text)
        if merch:
            flags.append(_flag(
                "voice-match",
                "Draft says %r; the profile says Merchandise, never"
                " merch." % merch.group(0),
                "Replace it with Merchandise."))

    # Product consistency: the description or tags must share at least
    # one significant word with the claimed product (the draft's
    # explicit product, else its own title).
    reference = draft.get("product") or title
    ref_words = _significant_words(reference)
    if ref_words and (description or tags):
        body_words = _significant_words(description + " " + " ".join(tags))
        if not (ref_words & body_words):
            flags.append(_flag(
                "product-consistency",
                "No significant word from %r appears in the description"
                " or tags." % reference,
                "Rewrite the description and tags so they describe this"
                " product, not another one."))
    return flags


def build_verifier_prompt(profile, draft, rules=None):
    """Return (instructions, input_text) for a grade.

    Pure assembly — no API call — so prompt shape can be tested free.
    """
    rules = rules if rules is not None else DEFAULT_RULES
    instructions = render_instructions(profile)
    rule_lines = "\n".join(
        "- %s: %s" % (r["id"], r["statement"]) for r in rules)
    input_text = "\n".join([
        "Grade this Printify listing draft against the rules below.",
        "DRAFT TITLE: %s" % (draft.get("title") or ""),
        "DRAFT DESCRIPTION: %s" % (draft.get("description") or ""),
        "DRAFT TAGS: %s" % ", ".join(draft.get("tags") or []),
        "RULES:",
        rule_lines,
        "Return ONLY JSON, no other text, in exactly this shape:",
        '{"verdict": "PASS" or "NEEDS_FIX", "flags": [{"rule": rule id,'
        ' "evidence": what in the draft violates it, "fix": the concrete'
        ' fix}]}.',
        "Every flag must name the violated rule, the evidence, and a"
        " concrete fix. An empty flags list means PASS.",
    ])
    return instructions, input_text


def parse_verdict_text(text):
    """Parse model output into (verdict, flags).

    Accepts raw JSON or a fenced ```json block. Raises ValueError on
    anything else, so the caller can fall back honestly instead of
    inventing a verdict.
    """
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        data = json.loads(candidate)
    except ValueError:
        raise ValueError("model output was not valid JSON")
    verdict = data.get("verdict")
    flags = data.get("flags")
    if verdict not in ("PASS", "NEEDS_FIX") or not isinstance(flags, list):
        raise ValueError("model output lacks verdict/flags shape")
    clean = []
    for flag in flags:
        if not isinstance(flag, dict):
            continue
        clean.append({
            "rule": str(flag.get("rule", "unruled")),
            "evidence": str(flag.get("evidence", "")),
            "fix": str(flag.get("fix", "")),
        })
    return verdict, clean


def _merge_flags(model_flags, local_flags):
    """Combine flag cards, dropping exact duplicates."""
    merged = list(model_flags)
    seen = {(f["rule"], f["evidence"]) for f in merged}
    for flag in local_flags:
        key = (flag["rule"], flag["evidence"])
        if key not in seen:
            seen.add(key)
            merged.append(flag)
    return merged


def grade_listing(draft, profile, rules=None, previous_response_id=None):
    """Grade a draft listing. One model call; local checks are free.

    Returns a dict with verdict (PASS / NEEDS_FIX), flags (flag cards),
    response_id, and usage. Local-check flags always stand: a provable
    violation cannot be outvoted by a model PASS.
    """
    instructions, input_text = build_verifier_prompt(profile, draft, rules)
    text, response_id, usage = call_responses(
        input_text, instructions, previous_response_id)
    local = local_checks(draft, profile)
    try:
        model_verdict, model_flags = parse_verdict_text(text)
    except ValueError:
        model_verdict, model_flags = None, [
            _flag("verdict-parse",
                  "Model output was not valid JSON, so no model grade"
                  " could be read.",
                  "Re-run the grade; if it repeats, inspect the raw"
                  " model output.")]
    flags = _merge_flags(model_flags, local)
    if model_verdict == "NEEDS_FIX" and not model_flags:
        flags.append(_flag(
            "model-verdict",
            "Model returned NEEDS_FIX without flag details.",
            "Review the draft by hand or re-run the grade."))
    verdict = "PASS" if not flags and model_verdict == "PASS" \
        else "NEEDS_FIX"
    return {
        "verdict": verdict,
        "flags": flags,
        "response_id": response_id,
        "usage": usage,
    }
