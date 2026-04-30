import re
import json
import os
from collections import Counter
from data.load_data import load_uner, load_wikiann, split_uner

# ──────────────────────────────────────────
# 1. Label Normalization
# Maps both UNER and WikiANN tags to 5 unified labels:
# PER, LOC, ORG, MISC, O
# ──────────────────────────────────────────

UNER_TAG_MAP = {
    "B-PERSON":       "B-PER",
    "I-PERSON":       "I-PER",
    "B-LOCATION":     "B-LOC",
    "I-LOCATION":     "I-LOC",
    "B-ORGANIZATION": "B-ORG",
    "I-ORGANIZATION": "I-ORG",
    "B-DESIGNATION":  "B-MISC",
    "I-DESIGNATION":  "I-MISC",
    "B-DATE":         "B-MISC",
    "I-DATE":         "I-MISC",
    "B-NUMBER":       "B-MISC",
    "I-NUMBER":       "I-MISC",
    "B-TIME":         "B-MISC",
    "I-TIME":         "I-MISC",
    "O":              "O",
}

WIKIANN_TAG_MAP = {
    "B-PER": "B-PER",
    "I-PER": "I-PER",
    "B-LOC": "B-LOC",
    "I-LOC": "I-LOC",
    "B-ORG": "B-ORG",
    "I-ORG": "I-ORG",
    "O":     "O",
}

def normalize_tags(sentences, tag_map):
    normalized = []
    unknown_tags = set()
    for sent in sentences:
        new_sent = []
        for word, tag in sent:
            mapped = tag_map.get(tag)
            if mapped is None:
                unknown_tags.add(tag)
                mapped = "O"
            new_sent.append((word, mapped))
        normalized.append(new_sent)
    if unknown_tags:
        print(f"  Warning — unmapped tags defaulted to O: {unknown_tags}")
    return normalized


# ──────────────────────────────────────────
# 2. Text Cleaning
# Removes URLs, hashtags, mentions, special
# chars that carry no entity information
# ──────────────────────────────────────────

# Roman Urdu phonetic normalization map
# Covers the most common spelling variants
PHONETIC_MAP = {
    "aa": "a",
    "ee": "i",
    "oo": "u",
    "kh": "k",
    "gh": "g",
    "ph": "f",
    "wh": "w",
    "ain": "an",
    "ck": "k",
}

def normalize_roman_urdu(word):
    """
    Applies phonetic normalization to Roman Urdu tokens only.
    Skips Urdu script words (they contain unicode chars above 0x0600).
    """
    # Check if word contains Urdu/Arabic unicode characters
    if any('\u0600' <= c <= '\u06FF' for c in word):
        return word  # leave Urdu script words untouched

    lowered = word.lower()
    for variant, canonical in PHONETIC_MAP.items():
        lowered = lowered.replace(variant, canonical)
    return lowered


def clean_token(word):
    """
    Cleans a single token:
    - Removes URLs
    - Removes hashtag/mention symbols
    - Strips punctuation attached to entity words
      but preserves the word itself
    """
    # Remove full URLs
    if re.match(r'https?://\S+|www\.\S+', word):
        return None  # signal to drop this token

    # Remove hashtag and mention symbols
    word = re.sub(r'^[#@]', '', word)

    # Remove leading/trailing punctuation that isn't part of the word
    # but keep internal punctuation (e.g. hyphenated names)
    word = re.sub(r'^[^\w\u0600-\u06FF]+', '', word)
    word = re.sub(r'[^\w\u0600-\u06FF]+$', '', word)

    # Drop empty tokens after cleaning
    if not word:
        return None

    return word


def clean_sentence(sentence):
    """
    Cleans all tokens in a sentence.
    Drops tokens that clean_token marks as None.
    Applies Roman Urdu phonetic normalization.
    """
    cleaned = []
    for word, tag in sentence:
        word = clean_token(word)
        if word is None:
            continue
        word = normalize_roman_urdu(word)
        cleaned.append((word, tag))
    return cleaned


def clean_dataset(sentences):
    cleaned = [clean_sentence(s) for s in sentences]
    # Remove sentences that became empty after cleaning
    cleaned = [s for s in cleaned if len(s) > 0]
    return cleaned


# ──────────────────────────────────────────
# 3. Sequence Length Analysis
# Needed for cache-aware batch scheduling
# later in the parallel pipeline
# ──────────────────────────────────────────

def compute_lengths(sentences):
    return [len(s) for s in sentences]

def length_stats(lengths, name=""):
    print(f"\n── Length Stats: {name} ──")
    print(f"  Min length    : {min(lengths)}")
    print(f"  Max length    : {max(lengths)}")
    print(f"  Avg length    : {sum(lengths)/len(lengths):.1f}")
    # Distribution buckets
    buckets = {"1-10": 0, "11-30": 0, "31-50": 0, "51+": 0}
    for l in lengths:
        if l <= 10:   buckets["1-10"] += 1
        elif l <= 30: buckets["11-30"] += 1
        elif l <= 50: buckets["31-50"] += 1
        else:         buckets["51+"] += 1
    print(f"  Length distribution:")
    for bucket, count in buckets.items():
        print(f"    {bucket:10s}: {count} sentences")


# ──────────────────────────────────────────
# 4. Save to Disk
# Saves as JSON — one file per split
# Format: list of sentences, each sentence
# is a list of [word, tag] pairs
# ──────────────────────────────────────────

def save_split(sentences, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(sentences)} sentences → {path}")


def load_split(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert inner lists back to tuples
    return [[tuple(pair) for pair in sent] for sent in data]


# ──────────────────────────────────────────
# 5. Label Inventory
# Builds the master list of all unique labels
# This will be needed by the fine-tuning script
# ──────────────────────────────────────────

def build_label_list(splits):
    all_tags = set()
    for split in splits:
        for sent in split:
            for _, tag in sent:
                all_tags.add(tag)
    # Sort with O first, then alphabetically
    label_list = ["O"] + sorted(t for t in all_tags if t != "O")
    return label_list


# ──────────────────────────────────────────
# 6. Final Summary
# ──────────────────────────────────────────

def print_final_summary(train, val, test, name):
    total = len(train) + len(val) + len(test)
    all_tags = [tag for split in [train, val, test]
                    for sent in split
                    for _, tag in sent]
    counts = Counter(all_tags)
    print(f"\n══ {name} — Final Summary ══")
    print(f"  Train / Val / Test : {len(train)} / {len(val)} / {len(test)}")
    print(f"  Total sentences    : {total}")
    print(f"  Total tokens       : {len(all_tags)}")
    print(f"  Label counts:")
    for tag, count in sorted(counts.items()):
        print(f"    {tag:15s}: {count}")


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("STEP 1 — Loading raw datasets")
    print("=" * 55)

    uner_raw             = load_uner()
    wiki_train_raw, wiki_val_raw, wiki_test_raw = load_wikiann()
    uner_train_raw, uner_val_raw, uner_test_raw = split_uner(uner_raw)


    print("\n" + "=" * 55)
    print("STEP 2 — Normalizing tags to unified label set")
    print("=" * 55)

    uner_train = normalize_tags(uner_train_raw, UNER_TAG_MAP)
    uner_val   = normalize_tags(uner_val_raw,   UNER_TAG_MAP)
    uner_test  = normalize_tags(uner_test_raw,  UNER_TAG_MAP)
    print("  UNER tags normalized")

    wiki_train = normalize_tags(wiki_train_raw, WIKIANN_TAG_MAP)
    wiki_val   = normalize_tags(wiki_val_raw,   WIKIANN_TAG_MAP)
    wiki_test  = normalize_tags(wiki_test_raw,  WIKIANN_TAG_MAP)
    print("  WikiANN tags normalized")


    print("\n" + "=" * 55)
    print("STEP 3 — Cleaning tokens")
    print("=" * 55)

    uner_train = clean_dataset(uner_train)
    uner_val   = clean_dataset(uner_val)
    uner_test  = clean_dataset(uner_test)
    print(f"  UNER cleaned  — Train: {len(uner_train)}, "
          f"Val: {len(uner_val)}, Test: {len(uner_test)}")

    wiki_train = clean_dataset(wiki_train)
    wiki_val   = clean_dataset(wiki_val)
    wiki_test  = clean_dataset(wiki_test)
    print(f"  WikiANN cleaned — Train: {len(wiki_train)}, "
          f"Val: {len(wiki_val)}, Test: {len(wiki_test)}")


    print("\n" + "=" * 55)
    print("STEP 4 — Sequence length analysis")
    print("=" * 55)

    length_stats(compute_lengths(uner_train),  "UNER Train")
    length_stats(compute_lengths(wiki_train),  "WikiANN Train")


    print("\n" + "=" * 55)
    print("STEP 5 — Saving preprocessed splits to disk")
    print("=" * 55)

    save_split(uner_train, "data/processed/uner_train.json")
    save_split(uner_val,   "data/processed/uner_val.json")
    save_split(uner_test,  "data/processed/uner_test.json")

    save_split(wiki_train, "data/processed/wiki_train.json")
    save_split(wiki_val,   "data/processed/wiki_val.json")
    save_split(wiki_test,  "data/processed/wiki_test.json")


    print("\n" + "=" * 55)
    print("STEP 6 — Building unified label list")
    print("=" * 55)

    label_list = build_label_list([
        uner_train, uner_val, uner_test,
        wiki_train, wiki_val, wiki_test
    ])

    label_path = "data/processed/labels.json"
    with open(label_path, "w") as f:
        json.dump(label_list, f, indent=2)

    print(f"  Labels: {label_list}")
    print(f"  Saved  → {label_path}")


    print("\n" + "=" * 55)
    print("FINAL SUMMARIES")
    print("=" * 55)

    print_final_summary(uner_train, uner_val, uner_test, "UNER")
    print_final_summary(wiki_train, wiki_val, wiki_test, "WikiANN")

    print("\n✓ Preprocessing complete. data/processed/ is ready.")