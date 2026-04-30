"""
build_roman_dataset.py
──────────────────────
Reads the raw UNER dataset, parses its XML-tag format,
transliterates every token from Urdu script to Roman Urdu,
and saves train/val/test splits as JSON files ready for
fine-tuning in the next pipeline stage.

Run from project root:
    python build_roman_dataset.py
"""

import re
import os
import json
import random
from collections import Counter
from transliterate import transliterate_word, NAMED_ENTITY_DICT


# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────

RAW_UNER_PATH  = os.path.join("data", "raw", "uner.txt")
OUTPUT_DIR     = os.path.join("data", "roman")
RANDOM_SEED    = 42
TRAIN_RATIO    = 0.80
VAL_RATIO      = 0.10
TEST_RATIO     = 0.10

# Unified label set — maps UNER tags to 5-class scheme
TAG_MAP = {
    "PERSON":       "PER",
    "LOCATION":     "LOC",
    "ORGANIZATION": "ORG",
    "DESIGNATION":  "MISC",
    "DATE":         "MISC",
    "NUMBER":       "MISC",
    "TIME":         "MISC",
    "O":            "O",
}


# ──────────────────────────────────────────
# STEP 1 — PARSE RAW UNER FILE
# Format: inline XML tags
# <LOCATION>برطانیہ</LOCATION> کے سابق ...
# ──────────────────────────────────────────

def parse_uner(filepath):
    """
    Parses the raw UNER .txt file.
    Returns list of sentences.
    Each sentence is a list of (urdu_word, bio_tag) tuples.
    """
    sentences = []

    # Try encodings in order
    lines = None
    for enc in ["utf-16", "utf-16-le", "utf-8-sig", "utf-8"]:
        try:
            with open(filepath, encoding=enc, errors="replace") as f:
                lines = f.readlines()
            print(f"  Opened with encoding: {enc}")
            break
        except Exception:
            continue

    if lines is None:
        raise RuntimeError(f"Could not open file: {filepath}")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        tokens = []
        remaining = line

        while remaining:
            # Try to match an XML entity tag first
            match = re.match(r'<(\w+)>(.*?)</\1>', remaining)
            if match:
                raw_tag   = match.group(1).upper()
                entity_text = match.group(2).strip()
                words = entity_text.split()

                for i, word in enumerate(words):
                    # Map to unified label set
                    unified = TAG_MAP.get(raw_tag, "MISC")
                    bio     = f"B-{unified}" if i == 0 else f"I-{unified}"
                    tokens.append((word, bio))

                remaining = remaining[match.end():].lstrip()

            else:
                # Plain word — outside any entity tag
                word_match = re.match(r'(\S+)', remaining)
                if word_match:
                    word = word_match.group(1)
                    tokens.append((word, "O"))
                    remaining = remaining[word_match.end():].lstrip()
                else:
                    break

        if tokens:
            sentences.append(tokens)

    return sentences


# ──────────────────────────────────────────
# STEP 2 — TRANSLITERATE
# Converts each Urdu token to Roman Urdu.
# Tags are preserved unchanged.
# ──────────────────────────────────────────

def transliterate_dataset(sentences):
    """
    Applies transliteration to every token in every sentence.
    Returns same structure with Roman Urdu words.
    """
    roman_sentences = []

    for sent in sentences:
        roman_sent = []
        for word, tag in sent:
            roman = transliterate_word(word)

            # Safety check — if transliteration produced
            # empty string, keep original word
            if not roman or not roman.strip():
                roman = word

            # Lowercase for consistency with social media Roman Urdu
            roman = roman.lower().strip()

            roman_sent.append((roman, tag))

        # Only keep sentences that have at least one token
        if roman_sent:
            roman_sentences.append(roman_sent)

    return roman_sentences


# ──────────────────────────────────────────
# STEP 3 — SPLIT
# 80 / 10 / 10 split with fixed seed
# ──────────────────────────────────────────

def split_dataset(sentences, seed=RANDOM_SEED):
    data = sentences[:]
    random.seed(seed)
    random.shuffle(data)

    n       = len(data)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train = data[:n_train]
    val   = data[n_train : n_train + n_val]
    test  = data[n_train + n_val :]

    return train, val, test


# ──────────────────────────────────────────
# STEP 4 — SAVE
# JSON format: list of sentences,
# each sentence is list of [word, tag] pairs
# ──────────────────────────────────────────

def save_split(sentences, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert tuples to lists for JSON serialisation
    serialisable = [[list(pair) for pair in sent]
                    for sent in sentences]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(sentences):>5} sentences → {path}")


# ──────────────────────────────────────────
# STEP 5 — VERIFY
# Prints samples and stats so you can
# visually confirm quality before training
# ──────────────────────────────────────────

def verify_output(train, val, test):
    all_splits = train + val + test

    # Tag distribution
    all_tags = [tag for sent in all_splits for _, tag in sent]
    counts   = Counter(all_tags)

    print(f"\n  Total sentences  : {len(all_splits)}")
    print(f"  Total tokens     : {len(all_tags)}")
    print(f"  Entity tokens    : {sum(v for k,v in counts.items() if k != 'O')}")
    print(f"\n  Tag distribution:")
    for tag in sorted(counts):
        bar = "█" * (counts[tag] // 100)
        print(f"    {tag:<12} {counts[tag]:>6}  {bar}")

    # Entity quality sample — show 20 entity tokens
    print(f"\n  ── Entity token sample (20 random) ──")
    entity_tokens = [(w,t) for sent in all_splits
                          for w,t in sent if t != "O"]
    random.seed(0)
    sample = random.sample(entity_tokens,
                           min(20, len(entity_tokens)))
    print(f"  {'Roman Urdu':<25} Tag")
    print("  " + "─" * 40)
    for word, tag in sorted(sample, key=lambda x: x[1]):
        print(f"  {word:<25} {tag}")

    # Sentence sample — show 5 full sentences
    print(f"\n  ── Full sentence samples (5) ──")
    for i, sent in enumerate(all_splits[:5]):
        print(f"\n  Sentence {i+1}:")
        for word, tag in sent:
            marker = f"[{tag}]" if tag != "O" else ""
            print(f"    {word:<25} {marker}")


def check_label_file(train, val, test):
    """Saves the label list for use by fine-tuning script."""
    all_tags = set()
    for split in [train, val, test]:
        for sent in split:
            for _, tag in sent:
                all_tags.add(tag)

    # O first, then sorted
    label_list = ["O"] + sorted(t for t in all_tags if t != "O")

    label_path = os.path.join(OUTPUT_DIR, "labels.json")
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(label_list, f, indent=2)

    print(f"\n  Label list: {label_list}")
    print(f"  Saved → {label_path}")
    return label_list


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

def main():
    print("=" * 55)
    print("STEP 1 — Parsing raw UNER file")
    print("=" * 55)

    if not os.path.exists(RAW_UNER_PATH):
        raise FileNotFoundError(
            f"UNER file not found at: {RAW_UNER_PATH}\n"
            f"Make sure uner.txt is in data/raw/"
        )

    urdu_sentences = parse_uner(RAW_UNER_PATH)
    print(f"  Parsed {len(urdu_sentences)} sentences from UNER")


    print("\n" + "=" * 55)
    print("STEP 2 — Transliterating to Roman Urdu")
    print("=" * 55)

    roman_sentences = transliterate_dataset(urdu_sentences)
    print(f"  Transliterated {len(roman_sentences)} sentences")


    print("\n" + "=" * 55)
    print("STEP 3 — Splitting 80 / 10 / 10")
    print("=" * 55)

    train, val, test = split_dataset(roman_sentences)
    print(f"  Train : {len(train)}")
    print(f"  Val   : {len(val)}")
    print(f"  Test  : {len(test)}")


    print("\n" + "=" * 55)
    print("STEP 4 — Saving to data/roman/")
    print("=" * 55)

    save_split(train, os.path.join(OUTPUT_DIR, "train.json"))
    save_split(val,   os.path.join(OUTPUT_DIR, "val.json"))
    save_split(test,  os.path.join(OUTPUT_DIR, "test.json"))


    print("\n" + "=" * 55)
    print("STEP 5 — Saving label list")
    print("=" * 55)

    labels = check_label_file(train, val, test)


    print("\n" + "=" * 55)
    print("STEP 6 — Quality verification")
    print("=" * 55)

    verify_output(train, val, test)


    print("\n" + "=" * 55)
    print("DONE")
    print("=" * 55)
    print(f"""
  Output files:
    data/roman/train.json   ({len(train)} sentences)
    data/roman/val.json     ({len(val)} sentences)
    data/roman/test.json    ({len(test)} sentences)
    data/roman/labels.json  ({len(labels)} labels)

  These files are ready for fine-tuning.
  Next step: finetune.py
    """)


if __name__ == "__main__":
    main()