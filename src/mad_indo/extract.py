"""Label extraction with an audit trail.

Lesson from Rahmawati & Salsabila (2026): a fallback that picks labels by a fixed
priority order silently mislabels neutral answers whose rationale mentions other
labels ("not clearly positive or negative"). Here the label comes from the JSON
field whenever possible, and any fallback picks the EARLIEST label mention.
Every output records which path was used, so errors can be audited.
"""
from __future__ import annotations

import json
import re

SYNONYMS = {
    # SmSA
    "positif": "positive", "negatif": "negative", "netral": "neutral",
    # EmoT
    "sedih": "sadness", "kesedihan": "sadness", "marah": "anger", "kemarahan": "anger",
    "cinta": "love", "takut": "fear", "ketakutan": "fear", "senang": "happy",
    "bahagia": "happy", "gembira": "happy", "happiness": "happy", "sad": "sadness",
    "angry": "anger",
}


def normalise_label(value, labels):
    if not isinstance(value, str):
        return None
    v = value.strip().strip('"\'.').lower()
    v = SYNONYMS.get(v, v)
    return v if v in labels else None


def _mentions(text, labels):
    """Return label mentions as (position, label), using labels and their synonyms."""
    found = []
    vocab = {l: l for l in labels}
    vocab.update({k: v for k, v in SYNONYMS.items() if v in labels})
    for word, lab in vocab.items():
        for m in re.finditer(rf"\b{re.escape(word)}\b", text.lower()):
            found.append((m.start(), lab))
    return sorted(found)


def extract_label(raw: str, labels) -> dict:
    """Returns {'label', 'rationale', 'path', 'mentions_other'}.

    path: 'json' | 'json_embedded' | 'fallback_first_mention' | 'unknown'
    """
    raw = raw or ""
    parsed, path = None, None
    try:
        parsed, path = json.loads(raw), "json"
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            try:
                parsed, path = json.loads(m.group(0)), "json_embedded"
            except json.JSONDecodeError:
                parsed = None

    label, rationale = None, ""
    if isinstance(parsed, dict):
        label = normalise_label(parsed.get("label"), labels)
        rationale = str(parsed.get("rationale", ""))
    if label is None:
        mentions = _mentions(raw, labels)
        if mentions:
            label, path = mentions[0][1], "fallback_first_mention"
        else:
            label, path = "unknown", "unknown"
        rationale = rationale or raw[:500]

    others = {lab for _, lab in _mentions(rationale, labels)} - {label}
    return {"label": label, "rationale": rationale, "path": path,
            "mentions_other": sorted(others)}
