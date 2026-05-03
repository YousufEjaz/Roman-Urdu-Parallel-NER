# check_coverage.py — updated to work on raw UNER
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transliterate import NAMED_ENTITY_DICT, transliterate_word
from collections import Counter

def parse_uner_raw(filepath="data/raw/uner.txt"):
    entity_tokens = []
    for enc in ["utf-16", "utf-16-le", "utf-8-sig"]:
        try:
            with open(filepath, encoding=enc, errors="replace") as f:
                lines = f.readlines()
            break
        except:
            continue

    for line in lines:
        line = line.strip()
        if not line:
            continue
        remaining = line
        while remaining:
            match = re.match(r'<(\w+)>(.*?)</\1>', remaining)
            if match:
                tag  = match.group(1).upper()
                text = match.group(2).strip()
                for i, word in enumerate(text.split()):
                    bio = f"B-{tag}" if i == 0 else f"I-{tag}"
                    entity_tokens.append((word, bio))
                remaining = remaining[match.end():].lstrip()
            else:
                word_match = re.match(r'(\S+)', remaining)
                if word_match:
                    remaining = remaining[word_match.end():].lstrip()
                else:
                    break
    return entity_tokens


def check_coverage():
    entity_tokens = parse_uner_raw()

    total   = len(entity_tokens)
    unique  = len(set(w for w,t in entity_tokens))
    hits    = [(w,t) for w,t in entity_tokens if w in NAMED_ENTITY_DICT]
    fallback = [(w,t) for w,t in entity_tokens if w not in NAMED_ENTITY_DICT]

    print(f"Total entity tokens   : {total}")
    print(f"Unique entity tokens  : {unique}")
    print(f"Dictionary entries    : {len(NAMED_ENTITY_DICT)}")
    print(f"Dictionary hits       : {len(hits)} ({100*len(hits)/total:.1f}%)")
    print(f"Fallback              : {len(fallback)} ({100*len(fallback)/total:.1f}%)")

    fallback_counts = Counter(w for w,t in fallback)
    print(f"\n── Top 30 remaining uncovered ──")
    print(f"  {'Urdu':<25} {'Count':<8} {'Auto Roman':<25} Tag")
    print("  " + "─"*65)
    for word, count in fallback_counts.most_common(30):
        roman = transliterate_word(word)
        tag   = next(t for w,t in fallback if w==word)
        print(f"  {word:<25} {count:<8} {roman:<25} {tag}")

if __name__ == "__main__":
    check_coverage()