"""
build_variants.py
─────────────────
Reads the Mendeley Roman Urdu spelling variations dataset.
Does three things:
  1. Builds a variant → canonical spelling map
  2. Cross-checks our current transliterate.py dictionaries
     against the Mendeley canonical forms
  3. Saves the normalization map to data/roman/variants.json
     for use at inference time

Run from project root:
    python build_variants.py
"""

import os
import json
import pandas as pd
from collections import defaultdict


# ──────────────────────────────────────────
# CONFIG — update filename if yours differs
# ──────────────────────────────────────────

MENDELEY_PATH = os.path.join("data", "raw", "roman_urdu_variations.xlsx")
MENDELEY_CSV  = os.path.join("data", "raw", "roman_urdu_variations.csv")
OUTPUT_PATH   = os.path.join("data", "roman", "variants.json")


# ──────────────────────────────────────────
# STEP 1 — Load Mendeley dataset
# ──────────────────────────────────────────

def load_mendeley():
    """
    Tries xlsx first, falls back to csv.
    Returns a pandas DataFrame.
    """
    if os.path.exists(MENDELEY_PATH):
        print(f"  Loading Excel: {MENDELEY_PATH}")
        df = pd.read_excel(MENDELEY_PATH)
    elif os.path.exists(MENDELEY_CSV):
        print(f"  Loading CSV: {MENDELEY_CSV}")
        df = pd.read_csv(MENDELEY_CSV)
    else:
        # Try to find any xlsx or csv in data/raw
        for f in os.listdir(os.path.join("data", "raw")):
            if f.endswith(".xlsx") or f.endswith(".csv"):
                path = os.path.join("data", "raw", f)
                print(f"  Found: {path}")
                if f.endswith(".xlsx"):
                    df = pd.read_excel(path)
                else:
                    df = pd.read_csv(path)
                print(f"  Loaded with shape: {df.shape}")
                return df
        raise FileNotFoundError(
            "No xlsx or csv found in data/raw/\n"
            "Please rename your Mendeley file to:\n"
            "  data/raw/roman_urdu_variations.xlsx"
        )

    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


# ──────────────────────────────────────────
# STEP 2 — Parse and build variant map
# ──────────────────────────────────────────

def build_variant_map(df):
    """
    Builds two structures:
      variant_to_canonical : dict  {variant_spelling → canonical}
      canonical_to_variants: dict  {canonical → [all_variants]}
      word_to_english      : dict  {canonical → english_meaning}
    """
    # Normalise column names — lowercase, strip spaces
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"\n  Columns after normalisation: {list(df.columns)}")

    # Find the canonical and variant columns
    # Mendeley uses: var-1..var-5, Common, English Translated
    variant_cols = [c for c in df.columns
                    if c.startswith("var") or c.startswith("var-")]
    common_col   = next((c for c in df.columns
                         if "common" in c), None)
    english_col  = next((c for c in df.columns
                         if "english" in c or "translat" in c), None)

    if common_col is None:
        raise ValueError(
            f"Could not find 'Common' column. "
            f"Available: {list(df.columns)}"
        )

    print(f"  Variant columns : {variant_cols}")
    print(f"  Canonical column: {common_col}")
    print(f"  English column  : {english_col}")

    variant_to_canonical  = {}
    canonical_to_variants = defaultdict(set)
    word_to_english       = {}

    skipped = 0
    for _, row in df.iterrows():
        canonical = str(row[common_col]).strip().lower()

        # Skip empty or NaN canonical
        if canonical in ("nan", "", "none"):
            skipped += 1
            continue

        # Record english meaning
        if english_col:
            eng = str(row[english_col]).strip()
            if eng not in ("nan", "", "none"):
                word_to_english[canonical] = eng

        # Collect all non-empty variants
        variants = set()
        variants.add(canonical)  # canonical maps to itself

        for col in variant_cols:
            v = str(row[col]).strip().lower()
            if v not in ("nan", "", "none") and v:
                variants.add(v)

        # Build mappings
        for v in variants:
            variant_to_canonical[v]   = canonical
            canonical_to_variants[canonical].add(v)

    # Convert sets to sorted lists for JSON serialisation
    canonical_to_variants = {
        k: sorted(v) for k, v in canonical_to_variants.items()
    }

    print(f"\n  Words processed     : {len(df) - skipped}")
    print(f"  Skipped (empty)     : {skipped}")
    print(f"  Total variant forms : {len(variant_to_canonical)}")
    print(f"  Unique canonicals   : {len(canonical_to_variants)}")

    return variant_to_canonical, canonical_to_variants, word_to_english


# ──────────────────────────────────────────
# STEP 3 — Cross-check our dictionaries
# ──────────────────────────────────────────

def cross_check_our_dicts(variant_to_canonical, word_to_english):
    """
    Imports our current transliterate.py dictionaries and
    checks each entry against the Mendeley canonical forms.

    Reports:
      - Entries we have that MATCH Mendeley canonical ✓
      - Entries we have that CONFLICT with Mendeley ✗
      - Mendeley words NOT in our dict (expansion opportunities)
    """
    # Import our dictionaries
    from transliterate import FUNCTION_WORDS, NAMED_ENTITY_DICT

    our_words = {}
    our_words.update(FUNCTION_WORDS)
    our_words.update(NAMED_ENTITY_DICT)

    # Only check Roman Urdu → Roman Urdu mappings
    # (our dicts map Urdu script → Roman, Mendeley is Roman→Roman)
    # So we check our OUTPUT values against Mendeley canonical

    matches    = []
    conflicts  = []
    not_found  = []

    for urdu_word, our_roman in our_words.items():
        our_roman_clean = our_roman.lower().strip()

        if our_roman_clean in variant_to_canonical:
            mendeley_canonical = variant_to_canonical[our_roman_clean]
            if mendeley_canonical == our_roman_clean:
                matches.append((urdu_word, our_roman_clean, "✓"))
            else:
                # Our form is a variant, Mendeley prefers different form
                conflicts.append((
                    urdu_word,
                    our_roman_clean,
                    mendeley_canonical
                ))
        else:
            not_found.append((urdu_word, our_roman_clean))

    print(f"\n  ── Dictionary Cross-Check ──")
    print(f"  Our dict entries     : {len(our_words)}")
    print(f"  Match Mendeley       : {len(matches)}")
    print(f"  Conflict with Mendeley: {len(conflicts)}")
    print(f"  Not in Mendeley      : {len(not_found)}")

    if conflicts:
        print(f"\n  ── Conflicts (our form → Mendeley prefers) ──")
        print(f"  {'Urdu':<20} {'Ours':<20} {'Mendeley':<20}")
        print("  " + "─" * 60)
        for urdu, ours, mend in conflicts[:30]:
            print(f"  {str(urdu):<20} {ours:<20} {mend:<20}")

    # Show expansion opportunities — high frequency Mendeley words
    # we don't have yet
    print(f"\n  ── Top expansion opportunities ──")
    print(f"  (Mendeley words not in our dict, sample of 100)")
    print(f"  {'Canonical':<20} {'English':<20}")
    print("  " + "─" * 40)
    shown = 0
    for canonical, english in word_to_english.items():
        if canonical not in [r.lower() for r in our_words.values()]:
            print(f"  {canonical:<20} {english:<20}")
            shown += 1
            if shown >= 100:
                break

    return conflicts


# ──────────────────────────────────────────
# STEP 4 — Apply corrections from conflicts
# ──────────────────────────────────────────

def build_correction_map(conflicts):
    """
    From the conflict list, builds a map:
      our_form → mendeley_preferred_form
    This will be applied during normalization.
    """
    correction_map = {}
    for urdu, ours, mendeley in conflicts:
        correction_map[ours] = mendeley
    return correction_map


# ──────────────────────────────────────────
# STEP 5 — Save normalization data
# ──────────────────────────────────────────

def save_normalization_data(
        variant_to_canonical,
        canonical_to_variants,
        word_to_english,
        correction_map):

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    output = {
        "variant_to_canonical":   variant_to_canonical,
        "canonical_to_variants":  canonical_to_variants,
        "word_to_english":        word_to_english,
        "our_dict_corrections":   correction_map,
        "metadata": {
            "total_variants":     len(variant_to_canonical),
            "total_canonicals":   len(canonical_to_variants),
            "total_corrections":  len(correction_map),
        }
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved → {OUTPUT_PATH}")


# ──────────────────────────────────────────
# STEP 6 — Preview the normalization map
# ──────────────────────────────────────────

def preview_normalization(variant_to_canonical, canonical_to_variants):
    """
    Shows examples of how the normalizer will work.
    """
    print(f"\n  ── Normalization Preview ──")
    print(f"  (any of these inputs → canonical output)")
    print()

    # Show a few high-value examples
    examples = [
        "kahna", "kehna", "karna", "krna",
        "hai", "he", "hy",
        "aur", "or",
        "mein", "main", "mn",
        "nahi", "nhi", "nahin",
        "kya", "kia", "keya",
        "bhi", "bi",
        "hain", "hein",
        "tha", "tha",
        "gaya", "gya",
    ]

    print(f"  {'Input variant':<20} → Canonical")
    print("  " + "─" * 40)
    for v in examples:
        canonical = variant_to_canonical.get(v, "(not in Mendeley)")
        print(f"  {v:<20} → {canonical}")


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

def main():
    print("=" * 55)
    print("STEP 1 — Loading Mendeley dataset")
    print("=" * 55)
    df = load_mendeley()

    print("\n" + "=" * 55)
    print("STEP 2 — Building variant map")
    print("=" * 55)
    variant_to_canonical, canonical_to_variants, word_to_english = \
        build_variant_map(df)

    print("\n" + "=" * 55)
    print("STEP 3 — Cross-checking our dictionaries")
    print("=" * 55)
    conflicts = cross_check_our_dicts(variant_to_canonical, word_to_english)

    print("\n" + "=" * 55)
    print("STEP 4 — Building correction map")
    print("=" * 55)
    correction_map = build_correction_map(conflicts)
    print(f"  {len(correction_map)} corrections identified")

    print("\n" + "=" * 55)
    print("STEP 5 — Saving normalization data")
    print("=" * 55)
    save_normalization_data(
        variant_to_canonical,
        canonical_to_variants,
        word_to_english,
        correction_map
    )

    print("\n" + "=" * 55)
    print("STEP 6 — Normalization preview")
    print("=" * 55)
    preview_normalization(variant_to_canonical, canonical_to_variants)

    print("\n" + "=" * 55)
    print("DONE")
    print("=" * 55)
    print(f"""
  Next steps:
  1. Review conflicts printed above
  2. Run: python normalize.py  (to test normalization)
  3. Run: python build_roman_dataset.py  (rebuild with corrections)
    """)


if __name__ == "__main__":
    main()