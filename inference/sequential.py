"""
inference/sequential.py
───────────────────────
Sequential single-threaded NER inference baseline.
This is Configuration 1 from the proposal — the reference
point against which all parallel configurations are compared.

Measures:
  - Total execution time
  - Per-document average latency (ms)
  - Throughput (documents per second)
  - Memory usage
  - Per-class F1, Precision, Recall

Run from project root:
    python inference/sequential.py

IMPORTANT: Run this BEFORE parallel scripts.
All speedup calculations divide into this baseline.
"""

import os
import sys
import json
import time
import torch
import psutil
import numpy as np
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
)
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# ── Make sure project root is on path ────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────

MODEL_DIR    = os.path.join("model", "saved")
DATA_PATH    = os.path.join("data",  "roman", "test.json")
LABELS_PATH  = os.path.join("data",  "roman", "labels.json")
RESULTS_DIR  = os.path.join("results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "sequential_results.json")

MAX_LEN      = 128

# Force single thread — this is the sequential baseline
# Both Python-level and OpenMP/MKL threads set to 1
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"

DEVICE = torch.device("cpu")


# ──────────────────────────────────────────
# STEP 1 — Load model and data
# ──────────────────────────────────────────

def load_model_and_tokenizer():
    print(f"  Loading model from: {MODEL_DIR}")

    # Check model exists — if fine-tuning still running,
    # fall back to base model for pipeline testing
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Model not found at {MODEL_DIR}\n"
            f"Run finetune.py first."
        )

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    labels    = load_labels()
    label2id  = {l: i for i, l in enumerate(labels)}
    id2label  = {i: l for i, l in enumerate(labels)}

    model = DistilBertForTokenClassification.from_pretrained(
        MODEL_DIR,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(DEVICE)
    model.eval()

    print(f"  Model loaded. Labels: {labels}")
    return model, tokenizer, labels, id2label, label2id


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_test_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    sentences = [[tuple(p) for p in sent] for sent in raw]
    print(f"  Test sentences loaded: {len(sentences)}")
    return sentences


# ──────────────────────────────────────────
# STEP 2 — Single document inference
# This is the atomic unit of work reused
# across all three pipeline configurations
# ──────────────────────────────────────────

def infer_single(model, tokenizer, words, id2label):
    """
    Runs NER inference on a single sentence (list of words).
    Returns list of predicted tag strings, aligned to input words.
    """
    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        max_length=MAX_LEN,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    input_ids      = tokenized["input_ids"].to(DEVICE)
    attention_mask = tokenized["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    logits      = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze(0)
    word_ids    = tokenized.word_ids(batch_index=0)

    # Align predictions back to original words
    # Take prediction of first subword for each word
    aligned_preds = []
    seen_word_ids = set()

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id in seen_word_ids:
            continue
        seen_word_ids.add(word_id)
        aligned_preds.append(id2label[predictions[token_idx].item()])

    # Pad or truncate to match word count
    while len(aligned_preds) < len(words):
        aligned_preds.append("O")
    aligned_preds = aligned_preds[:len(words)]

    return aligned_preds


# ──────────────────────────────────────────
# STEP 3 — Sequential inference loop
# ──────────────────────────────────────────

def run_sequential(model, tokenizer, sentences, id2label):
    """
    Processes all sentences one at a time, sequentially.
    Records per-document latency for statistical analysis.
    Returns predictions and timing data.
    """
    all_predictions = []
    latencies       = []   # per-document latency in ms

    total_start = time.perf_counter()

    for i, sentence in enumerate(sentences):
        words = [w for w, _ in sentence]

        doc_start = time.perf_counter()
        preds     = infer_single(model, tokenizer, words, id2label)
        doc_end   = time.perf_counter()

        latency_ms = (doc_end - doc_start) * 1000
        latencies.append(latency_ms)
        all_predictions.append(preds)

        # Progress every 100 docs
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - total_start
            print(f"    Processed {i+1}/{len(sentences)} | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"Avg latency: {np.mean(latencies):.1f}ms",
                  end="\r")

    total_end  = time.perf_counter()
    total_time = total_end - total_start

    print(f"\n    Done. Total time: {total_time:.2f}s")
    return all_predictions, latencies, total_time


# ──────────────────────────────────────────
# STEP 4 — Compute NER metrics
# ──────────────────────────────────────────

def compute_metrics(sentences, all_predictions):
    """
    Computes F1, Precision, Recall using seqeval.
    Compares predictions against ground truth tags.
    """
    true_tags = [[t for _, t in sent] for sent in sentences]
    pred_tags = all_predictions

    f1        = f1_score(true_tags, pred_tags)
    precision = precision_score(true_tags, pred_tags)
    recall    = recall_score(true_tags, pred_tags)
    report    = classification_report(true_tags, pred_tags, digits=4)

    return f1, precision, recall, report


# ──────────────────────────────────────────
# STEP 5 — Timing statistics
# ──────────────────────────────────────────

def compute_timing_stats(latencies, total_time, num_docs):
    """
    Computes all timing metrics needed for PDC comparison.
    These exact values become the denominator for speedup
    calculations in the parallel scripts.
    """
    latencies_arr = np.array(latencies)

    stats = {
        "total_time_s"       : round(total_time, 4),
        "num_documents"      : num_docs,
        "throughput_docs_s"  : round(num_docs / total_time, 4),
        "latency_mean_ms"    : round(float(np.mean(latencies_arr)),  4),
        "latency_median_ms"  : round(float(np.median(latencies_arr)),4),
        "latency_std_ms"     : round(float(np.std(latencies_arr)),   4),
        "latency_min_ms"     : round(float(np.min(latencies_arr)),   4),
        "latency_max_ms"     : round(float(np.max(latencies_arr)),   4),
        "latency_p95_ms"     : round(float(np.percentile(latencies_arr, 95)), 4),
        "latency_p99_ms"     : round(float(np.percentile(latencies_arr, 99)), 4),
    }
    return stats


# ──────────────────────────────────────────
# STEP 6 — Memory measurement
# ──────────────────────────────────────────

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / 1e6, 2)


# ──────────────────────────────────────────
# STEP 7 — Print and save results
# ──────────────────────────────────────────

def print_results(timing_stats, f1, precision, recall, report,
                  mem_before, mem_after):

    print("\n" + "=" * 55)
    print("SEQUENTIAL BASELINE RESULTS")
    print("=" * 55)

    print(f"\n  ── Timing ──")
    print(f"  Total time        : {timing_stats['total_time_s']:.2f} s")
    print(f"  Documents         : {timing_stats['num_documents']}")
    print(f"  Throughput        : "
          f"{timing_stats['throughput_docs_s']:.2f} docs/sec")
    print(f"  Avg latency       : "
          f"{timing_stats['latency_mean_ms']:.2f} ms/doc")
    print(f"  Median latency    : "
          f"{timing_stats['latency_median_ms']:.2f} ms/doc")
    print(f"  Std deviation     : "
          f"{timing_stats['latency_std_ms']:.2f} ms")
    print(f"  P95 latency       : "
          f"{timing_stats['latency_p95_ms']:.2f} ms")
    print(f"  P99 latency       : "
          f"{timing_stats['latency_p99_ms']:.2f} ms")

    print(f"\n  ── Memory ──")
    print(f"  Before inference  : {mem_before} MB")
    print(f"  After inference   : {mem_after} MB")
    print(f"  Delta             : {mem_after - mem_before:.2f} MB")

    print(f"\n  ── NER Accuracy ──")
    print(f"  F1 Score          : {f1:.4f}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"\n── Per-class Report ──\n")
    print(report)


def save_results(timing_stats, f1, precision, recall,
                 report, mem_before, mem_after):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {
        "config"          : "sequential",
        "num_threads"     : 1,
        "timing"          : timing_stats,
        "memory_before_mb": mem_before,
        "memory_after_mb" : mem_after,
        "ner_metrics"     : {
            "f1"         : round(f1, 4),
            "precision"  : round(precision, 4),
            "recall"     : round(recall, 4),
            "report"     : report,
        }
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved → {RESULTS_PATH}")
    print(f"  NOTE: Keep this file — parallel scripts")
    print(f"        read it to compute speedup ratios.")


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

def main():
    print("=" * 55)
    print("SEQUENTIAL INFERENCE BASELINE")
    print("(OMP_NUM_THREADS=1, single process)")
    print("=" * 55)

    print("\n" + "=" * 55)
    print("STEP 1 — Loading model and data")
    print("=" * 55)

    model, tokenizer, labels, id2label, label2id = \
        load_model_and_tokenizer()
    sentences = load_test_data()

    print(f"\n  Threads (PyTorch) : {torch.get_num_threads()}")
    print(f"  OMP_NUM_THREADS   : "
          f"{os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"  Device            : {DEVICE}")


    print("\n" + "=" * 55)
    print("STEP 2 — Measuring memory before inference")
    print("=" * 55)

    mem_before = get_memory_mb()
    print(f"  Memory usage: {mem_before} MB")


    print("\n" + "=" * 55)
    print("STEP 3 — Running sequential inference")
    print("=" * 55)
    print(f"  Processing {len(sentences)} documents one at a time...")
    print(f"  Each document through full NER pipeline:")
    print(f"  tokenize → forward pass → decode\n")

    all_predictions, latencies, total_time = run_sequential(
        model, tokenizer, sentences, id2label
    )


    print("\n" + "=" * 55)
    print("STEP 4 — Computing timing statistics")
    print("=" * 55)

    timing_stats = compute_timing_stats(
        latencies, total_time, len(sentences)
    )
    mem_after = get_memory_mb()


    print("\n" + "=" * 55)
    print("STEP 5 — Computing NER accuracy")
    print("=" * 55)

    f1, precision, recall, report = compute_metrics(
        sentences, all_predictions
    )


    print_results(
        timing_stats, f1, precision, recall, report,
        mem_before, mem_after
    )

    save_results(
        timing_stats, f1, precision, recall, report,
        mem_before, mem_after
    )

    print("\n" + "=" * 55)
    print("BASELINE COMPLETE")
    print("=" * 55)
    print(f"""
  Sequential baseline recorded.

  Key numbers for your report:
    Total time   : {timing_stats['total_time_s']} s
    Throughput   : {timing_stats['throughput_docs_s']} docs/sec
    Avg latency  : {timing_stats['latency_mean_ms']} ms/doc
    Test F1      : {round(f1, 4)}

  Next steps:
    python inference/naive_parallel.py    (2 workers, then 4)
    python inference/cache_aware.py       (2 workers, then 4)
    """)


if __name__ == "__main__":
    main()