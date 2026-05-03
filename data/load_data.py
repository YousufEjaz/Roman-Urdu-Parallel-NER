import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from collections import Counter

# ──────────────────────────────────────────
# 1. Load UNER dataset (inline XML tag format)
# Format: <TAG>word</TAG> rest of sentence
# ──────────────────────────────────────────
def load_uner(filepath="data/raw/uner.txt"):
    sentences = []

    # Try different encodings — UNER files vary
    for enc in ["utf-16", "utf-16-le", "utf-8-sig", "utf-8"]:
        try:
            with open(filepath, encoding=enc, errors="replace") as f:
                lines = f.readlines()
            print(f"File opened with encoding: {enc}")
            break
        except Exception as e:
            print(f"Encoding {enc} failed: {e}")
            continue

    for line in lines:
        line = line.strip()
        if not line:
            continue

        tokens_tags = []
        # We'll process the line character by character using regex
        # Pattern matches either <TAG>word</TAG> or plain words
        pattern = r'<(\w+)>(.*?)</\1>|(\S+)'
        pos = 0
        remaining = line

        while remaining:
            match = re.match(r'<(\w+)>(.*?)</\1>', remaining)
            if match:
                tag = match.group(1).upper()
                # Entity span — may contain multiple words
                entity_text = match.group(2).strip()
                words = entity_text.split()
                for i, word in enumerate(words):
                    bio_tag = f"B-{tag}" if i == 0 else f"I-{tag}"
                    tokens_tags.append((word, bio_tag))
                remaining = remaining[match.end():].lstrip()
            else:
                # Plain word (outside entity)
                word_match = re.match(r'(\S+)', remaining)
                if word_match:
                    tokens_tags.append((word_match.group(1), "O"))
                    remaining = remaining[word_match.end():].lstrip()
                else:
                    break

        if tokens_tags:
            sentences.append(tokens_tags)

    print(f"UNER loaded: {len(sentences)} sentences")
    return sentences


# ──────────────────────────────────────────
# 2. Load WikiANN Urdu dataset
# ──────────────────────────────────────────
def load_wikiann():
    print("Downloading WikiANN Urdu dataset...")
    dataset = load_dataset("wikiann", "ur")
    label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def extract_sentences(split):
        sentences = []
        for item in dataset[split]:
            tokens = item["tokens"]
            tags = [label_names[t] for t in item["ner_tags"]]
            sentences.append(list(zip(tokens, tags)))
        return sentences

    train = extract_sentences("train")
    val   = extract_sentences("validation")
    test  = extract_sentences("test")

    print(f"WikiANN — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


# ──────────────────────────────────────────
# 3. Split UNER into train/val/test (80/10/10)
# ──────────────────────────────────────────
def split_uner(sentences):
    train, temp = train_test_split(sentences, test_size=0.2, random_state=42)
    val,   test = train_test_split(temp,      test_size=0.5, random_state=42)
    print(f"UNER split — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


# ──────────────────────────────────────────
# 4. Basic statistics
# ──────────────────────────────────────────
def print_stats(sentences, name="Dataset"):
    all_tags = [tag for sent in sentences for _, tag in sent]
    tag_counts = Counter(all_tags)
    entity_tokens = [t for t in all_tags if t != "O"]

    print(f"\n── {name} Stats ──")
    print(f"  Total sentences : {len(sentences)}")
    print(f"  Total tokens    : {len(all_tags)}")
    print(f"  Entity tokens   : {len(entity_tokens)}")
    print(f"  Tag distribution:")
    for tag, count in sorted(tag_counts.items()):
        print(f"    {tag:25s} {count}")


# ──────────────────────────────────────────
# 5. Preview a few parsed sentences
# ──────────────────────────────────────────
def preview(sentences, n=3):
    print(f"\n── First {n} parsed sentences ──")
    for i, sent in enumerate(sentences[:n]):
        print(f"\nSentence {i+1}:")
        for word, tag in sent:
            print(f"  {word:30s} {tag}")


# ──────────────────────────────────────────
# Run
# ──────────────────────────────────────────
if __name__ == "__main__":
    uner_sentences = load_uner()
    preview(uner_sentences)
    train, val, test = split_uner(uner_sentences)
    print_stats(train, "UNER Train")
    print_stats(test,  "UNER Test")

    wiki_train, wiki_val, wiki_test = load_wikiann()
    print_stats(wiki_test, "WikiANN Test")