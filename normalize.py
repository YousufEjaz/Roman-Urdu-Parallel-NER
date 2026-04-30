"""
normalize.py
────────────
Spelling variant normalizer for Roman Urdu.
Loads the Mendeley variant map and provides a normalize()
function that maps any spelling variant to its canonical form.

This runs BEFORE the NER model at inference time:
  raw input → normalize() → NER model → entity tags

Usage:
    from normalize import normalize_token, normalize_sentence
"""

import os
import json

VARIANTS_PATH = os.path.join("data", "roman", "variants.json")

# Lazy-loaded — only reads file on first call
_variant_map = None


def _load_map():
    global _variant_map
    if _variant_map is not None:
        return _variant_map

    if not os.path.exists(VARIANTS_PATH):
        raise FileNotFoundError(
            f"variants.json not found at {VARIANTS_PATH}\n"
            f"Run build_variants.py first."
        )

    with open(VARIANTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    _variant_map = data.get("variant_to_canonical", {})
    print(f"  Normalization map loaded: "
          f"{len(_variant_map)} variant forms")
    return _variant_map


def normalize_token(token):
    """
    Maps a single Roman Urdu token to its canonical spelling.
    If not found in the map, returns the token unchanged (lowercased).

    Examples:
        normalize_token("kehna")  → "kahna"
        normalize_token("nhi")    → "nahi"
        normalize_token("karachi") → "karachi"  (not in map, pass through)
    """
    variant_map = _load_map()
    token_clean = token.lower().strip()
    return variant_map.get(token_clean, token_clean)


def normalize_sentence(tokens):
    """
    Normalizes a list of Roman Urdu tokens.
    Input:  ["pakistan", "ki", "janib", "se", "yasir", ...]
    Output: ["pakistan", "ki", "janib", "se", "yasir", ...]
            with any variant spellings mapped to canonical
    """
    return [normalize_token(t) for t in tokens]


def normalize_tagged_sentence(token_tag_list):
    """
    Normalizes a list of (token, tag) tuples.
    Tags are preserved unchanged.
    Input:  [("pakistan", "B-LOC"), ("ki", "O"), ...]
    Output: same structure with normalized tokens
    """
    return [(normalize_token(w), t) for w, t in token_tag_list]


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("── Normalization Test ──\n")

    test_variants = [
        # (input_variant, expected_canonical)
        ("kehna",   "?"),
        ("krna",    "?"),
        ("nhi",     "?"),
        ("nahin",   "?"),
        ("kia",     "?"),
        ("keya",    "?"),
        ("kya",     "?"),
        ("main",    "?"),
        ("mn",      "?"),
        ("he",      "?"),
        ("hy",      "?"),
        ("gya",     "?"),
        ("or",      "?"),
        ("hein",    "?"),
        # Should pass through unchanged (proper nouns)
        ("pakistan",  "?"),
        ("lahore",    "?"),
        ("imran",     "?"),
    ]

    print(f"  {'Input':<20} → Canonical")
    print("  " + "─" * 40)
    for variant, _ in test_variants:
        canonical = normalize_token(variant)
        changed = " ← normalized" if canonical != variant else ""
        print(f"  {variant:<20} → {canonical}{changed}")

    print("\n── Tagged sentence test ──\n")
    tagged = [
        ("pakistan",  "B-LOC"),
        ("ki",        "O"),
        ("hukumat",   "O"),
        ("ne",        "O"),
        ("kehna",     "O"),    # variant — should normalize
        ("hai",       "O"),
        ("ke",        "O"),
        ("imran",     "B-PER"),
        ("khan",      "I-PER"),
        ("nhi",       "O"),    # variant — should normalize
        ("aaye",      "O"),
    ]

    normalized = normalize_tagged_sentence(tagged)
    print(f"  {'Original':<20} {'Tag':<10} {'Normalized':<20}")
    print("  " + "─" * 55)
    for (orig, tag), (norm, _) in zip(tagged, normalized):
        changed = " ←" if norm != orig else ""
        print(f"  {orig:<20} {tag:<10} {norm:<20}{changed}")