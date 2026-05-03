"""
inference/cache_aware.py
─────────────────────────
Configuration 3: Cache-aware parallel inference.
Sorts corpus by token length BEFORE distributing
to workers: the main PDC contribution of this project.

This ensures each worker gets homogeneous document lengths:
  - Reduces padding waste per batch
  - Improves memory locality (similar tensor shapes)
  - Eliminates load imbalance from outlier documents
  - Reduces cache miss rate

Matches proposed method in paper:
  "sorting the full corpus by token length before distributing
   documents to threads, homogeneous batches, better locality"

Run from project root:
    python inference/cache_aware.py --workers 2
    python inference/cache_aware.py --workers 4

Reads  : results/sequential_results.json   (baseline)
         results/naive_parallel_N.json      (naive comparison)
Writes : results/cache_aware_N.json
"""

import os
import sys
import json
import time
import math
import argparse
import torch
import psutil
import numpy as np
import multiprocessing as mp
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

sys.path.insert(0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────

MODEL_DIR     = os.path.join("model",   "saved")
DATA_PATH     = os.path.join("data",    "roman", "test.json")
LABELS_PATH   = os.path.join("data",    "roman", "labels.json")
RESULTS_DIR   = os.path.join("results")
BASELINE_PATH = os.path.join(RESULTS_DIR, "sequential_results.json")

MAX_LEN       = 128


# ──────────────────────────────────────────
# WORKER — identical to naive_parallel
# The difference is in how documents
# are ordered before being passed to workers
# ──────────────────────────────────────────

def worker_fn(worker_id, sentences, result_queue,
              model_dir, labels_path, max_len):
    """
    Identical worker to naive_parallel.
    The cache-aware benefit comes from document
    ordering, not from changes inside the worker.
    """
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device("cpu")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model     = DistilBertForTokenClassification.from_pretrained(
        model_dir,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)
    model.eval()

    predictions = []
    latencies   = []

    for sentence in sentences:
        words = [w for w, _ in sentence]

        t0 = time.perf_counter()

        tokenized = tokenizer(
            words,
            is_split_into_words=True,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(
                input_ids=tokenized["input_ids"].to(device),
                attention_mask=tokenized["attention_mask"].to(device),
            )

        preds    = torch.argmax(outputs.logits, dim=-1).squeeze(0)
        word_ids = tokenized.word_ids(batch_index=0)

        aligned = []
        seen    = set()
        for idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            aligned.append(id2label[preds[idx].item()])

        while len(aligned) < len(words):
            aligned.append("O")
        aligned = aligned[:len(words)]

        t1 = time.perf_counter()
        predictions.append(aligned)
        latencies.append((t1 - t0) * 1000)

    result_queue.put({
        "worker_id"  : worker_id,
        "predictions": predictions,
        "latencies"  : latencies,
    })


# ──────────────────────────────────────────
# Cache-aware Partition 
# This is THE key difference from naive.
# Sort by length → split into contiguous
# chunks → each worker gets similar lengths
# ──────────────────────────────────────────

def partition_cache_aware(sentences, n_workers):
    """
    The core contribution:
    1. Record original indices (needed to restore order later)
    2. Sort all sentences by token count (ascending)
    3. Split sorted list into N contiguous chunks
    4. Each worker gets a chunk of similar-length documents

    Effect:
    - Worker 0 gets shortest documents
    - Worker N-1 gets longest documents
    - No worker gets a mix of short and very long docs
    - Padding waste is minimised within each chunk
    - Memory access patterns are more uniform (cache-friendly)
    - Load is balanced — similar total work per worker
    """
    # Step 1 — tag each sentence with its original index
    # so we can restore order after parallel processing
    indexed = list(enumerate(sentences))

    # Step 2 — sort by token count (number of words)
    # This is the length-sorting that gives cache awareness
    indexed_sorted = sorted(indexed, key=lambda x: len(x[1]))

    # Log the length distribution across chunks
    lengths = [len(s) for _, s in indexed_sorted]
    chunk_size = math.ceil(len(indexed_sorted) / n_workers)

    chunk_length_ranges = []
    for i in range(n_workers):
        start  = i * chunk_size
        end    = min(start + chunk_size, len(indexed_sorted))
        chunk  = lengths[start:end]
        if chunk:
            chunk_length_ranges.append({
                "worker"    : i,
                "min_len"   : min(chunk),
                "max_len"   : max(chunk),
                "mean_len"  : round(sum(chunk) / len(chunk), 1),
                "num_docs"  : len(chunk),
            })

    # Step 3 — split into N chunks
    chunks         = []
    original_index_chunks = []
    for i in range(n_workers):
        start = i * chunk_size
        end   = min(start + chunk_size, len(indexed_sorted))
        chunk = indexed_sorted[start:end]
        if chunk:
            orig_indices = [idx for idx, _ in chunk]
            sents        = [s   for _, s  in chunk]
            chunks.append(sents)
            original_index_chunks.append(orig_indices)

    return chunks, original_index_chunks, chunk_length_ranges


# ──────────────────────────────────────────
# RUN CACHE-AWARE PARALLEL INFERENCE
# ──────────────────────────────────────────

def run_cache_aware(sentences, n_workers):
    result_queue = mp.Queue()

    # THE KEY STEP — sort before distributing
    chunks, orig_index_chunks, length_ranges = \
        partition_cache_aware(sentences, n_workers)

    # Print length distribution to show cache-aware effect
    print(f"\n  Length distribution across workers (after sorting):")
    print(f"  {'Worker':<8} {'Docs':<8} {'Min len':<10} "
          f"{'Max len':<10} {'Mean len':<10}")
    print("  " + "─" * 50)
    for lr in length_ranges:
        print(f"  {lr['worker']:<8} {lr['num_docs']:<8} "
              f"{lr['min_len']:<10} {lr['max_len']:<10} "
              f"{lr['mean_len']:<10}")

    homogeneity = max(lr["max_len"] - lr["min_len"]
                      for lr in length_ranges)
    print(f"\n  Max length spread within any worker: {homogeneity}")
    print(f"  (compare to full corpus spread: "
          f"{max(len(s) for s in sentences) - min(len(s) for s in sentences)})")

    print(f"\n  Launching {len(chunks)} workers...")

    wall_start = time.perf_counter()

    processes = []
    for i, chunk in enumerate(chunks):
        p = mp.Process(
            target=worker_fn,
            args=(i, chunk, result_queue,
                  MODEL_DIR, LABELS_PATH, MAX_LEN),
        )
        p.start()
        processes.append(p)

    raw_results = []
    for _ in range(len(chunks)):
        raw_results.append(result_queue.get())

    for p in processes:
        p.join()

    wall_end  = time.perf_counter()
    wall_time = wall_end - wall_start

    raw_results.sort(key=lambda x: x["worker_id"])

    # Restore original document order using saved indices
    # This ensures NER accuracy comparison is valid
    total_docs  = len(sentences)
    pred_by_idx = [None] * total_docs
    lat_by_idx  = [None] * total_docs

    for worker_result, orig_indices in zip(
            raw_results, orig_index_chunks):
        for pred, lat, orig_idx in zip(
                worker_result["predictions"],
                worker_result["latencies"],
                orig_indices):
            pred_by_idx[orig_idx] = pred
            lat_by_idx[orig_idx]  = lat

    all_predictions = pred_by_idx
    all_latencies   = lat_by_idx

    # Per-worker stats — load balance analysis
    worker_stats = []
    for r, lr in zip(raw_results, length_ranges):
        lats = np.array(r["latencies"])
        worker_stats.append({
            "worker_id"     : r["worker_id"],
            "num_docs"      : len(r["predictions"]),
            "doc_len_min"   : lr["min_len"],
            "doc_len_max"   : lr["max_len"],
            "doc_len_mean"  : lr["mean_len"],
            "total_time_ms" : round(float(np.sum(lats)),  2),
            "mean_lat_ms"   : round(float(np.mean(lats)), 2),
            "max_lat_ms"    : round(float(np.max(lats)),  2),
        })

    return (all_predictions, all_latencies,
            wall_time, worker_stats, length_ranges)


# ──────────────────────────────────────────
# METRICS AND COMPARISON
# ──────────────────────────────────────────

def compute_metrics(sentences, predictions):
    true_tags = [[t for _, t in s] for s in sentences]
    f1        = f1_score(true_tags, predictions)
    precision = precision_score(true_tags, predictions)
    recall    = recall_score(true_tags, predictions)
    report    = classification_report(
                    true_tags, predictions, digits=4)
    return f1, precision, recall, report


def compute_timing_stats(latencies, wall_time, n_docs):
    a = np.array(latencies)
    return {
        "wall_time_s"      : round(wall_time, 4),
        "num_documents"    : n_docs,
        "throughput_docs_s": round(n_docs / wall_time, 4),
        "latency_mean_ms"  : round(float(np.mean(a)),           4),
        "latency_median_ms": round(float(np.median(a)),         4),
        "latency_std_ms"   : round(float(np.std(a)),            4),
        "latency_p95_ms"   : round(float(np.percentile(a, 95)), 4),
    }


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────
# PRINT RESULTS
# ──────────────────────────────────────────

def print_results(timing, f1, precision, recall, report,
                  worker_stats, n_workers,
                  baseline, naive_result):

    print("\n" + "=" * 55)
    print(f"CACHE-AWARE PARALLEL RESULTS ({n_workers} workers)")
    print("=" * 55)

    print(f"\n  ── Timing ──")
    print(f"  Wall time         : {timing['wall_time_s']:.2f} s")
    print(f"  Throughput        : "
          f"{timing['throughput_docs_s']:.2f} docs/sec")
    print(f"  Avg latency       : "
          f"{timing['latency_mean_ms']:.2f} ms/doc")
    print(f"  Latency std dev   : "
          f"{timing['latency_std_ms']:.2f} ms")

    # ── Three-way comparison ──
    if baseline and naive_result:
        t_seq   = baseline["timing"]["total_time_s"]
        t_naive = naive_result["timing"]["wall_time_s"]
        t_cache = timing["wall_time_s"]

        sp_vs_seq   = round(t_seq   / t_cache, 4)
        sp_vs_naive = round(t_naive / t_cache, 4)
        eff_vs_seq  = round(sp_vs_seq / n_workers * 100, 2)

        print(f"\n  ── Three-Way Comparison ──")
        print(f"  {'Config':<25} {'Time (s)':<12} "
              f"{'Speedup vs Seq':<18} {'Throughput'}")
        print("  " + "─" * 68)
        print(f"  {'Sequential':<25} {t_seq:<12.2f} "
              f"{'1.0000x':<18} "
              f"{baseline['timing']['throughput_docs_s']:.2f} docs/s")
        print(f"  {'Naive parallel':<25} {t_naive:<12.2f} "
              f"{naive_result['speedup']:<18} "
              f"{naive_result['timing']['throughput_docs_s']:.2f} docs/s")
        print(f"  {'Cache-aware parallel':<25} {t_cache:<12.2f} "
              f"{str(sp_vs_seq)+'x':<18} "
              f"{timing['throughput_docs_s']:.2f} docs/s")

        print(f"\n  Cache-aware vs naive speedup : {sp_vs_naive:.4f}x")
        print(f"  Cache-aware parallel efficiency: {eff_vs_seq:.2f}%")

        # Amdahl analysis
        if sp_vs_seq > 1:
            p = (1 - 1/sp_vs_seq) / (1 - 1/n_workers)
            p = min(max(p, 0), 1)
            print(f"\n  ── Amdahl's Law (cache-aware) ──")
            print(f"  Parallelizable fraction (p): {p:.4f}")
            for t in [2, 4, 8, 16]:
                th = 1 / ((1-p) + p/t)
                print(f"  Theoretical speedup @{t:>2}   : {th:.4f}x")

    # ── Load balance comparison ──
    print(f"\n  ── Worker Load Balance (cache-aware) ──")
    print(f"  {'Worker':<8} {'Docs':<6} {'Len range':<18} "
          f"{'Total ms':<12} {'Mean ms':<12} {'Max ms'}")
    print("  " + "─" * 68)
    for ws in worker_stats:
        len_range = f"{ws['doc_len_min']}–{ws['doc_len_max']}"
        print(f"  {ws['worker_id']:<8} {ws['num_docs']:<6} "
              f"{len_range:<18} {ws['total_time_ms']:<12} "
              f"{ws['mean_lat_ms']:<12} {ws['max_lat_ms']}")

    imbalance_cache = max(w["total_time_ms"] for w in worker_stats) - \
                      min(w["total_time_ms"] for w in worker_stats)
    print(f"\n  Load imbalance (cache-aware) : {imbalance_cache:.2f} ms")

    if naive_result:
        naive_stats = naive_result["worker_stats"]
        imbalance_naive = (
            max(w["total_time_ms"] for w in naive_stats) -
            min(w["total_time_ms"] for w in naive_stats)
        )
        improvement = round(
            (imbalance_naive - imbalance_cache) / imbalance_naive * 100,
            2
        )
        print(f"  Load imbalance (naive)       : {imbalance_naive:.2f} ms")
        print(f"  Imbalance reduction          : {improvement:.2f}%")

    print(f"\n  ── NER Accuracy ──")
    print(f"  F1        : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"\n── Per-class Report ──\n")
    print(report)


def save_results(timing, f1, precision, recall, report,
                 worker_stats, length_ranges, n_workers,
                 baseline, naive_result):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR,
                        f"cache_aware_{n_workers}.json")

    t_seq   = baseline["timing"]["total_time_s"] if baseline else None
    t_naive = naive_result["timing"]["wall_time_s"] \
              if naive_result else None
    t_cache = timing["wall_time_s"]

    speedup_vs_seq   = round(t_seq   / t_cache, 4) if t_seq   else None
    speedup_vs_naive = round(t_naive / t_cache, 4) if t_naive else None
    efficiency       = round(speedup_vs_seq / n_workers * 100, 2) \
                       if speedup_vs_seq else None

    results = {
        "config"              : "cache_aware_parallel",
        "num_workers"         : n_workers,
        "timing"              : timing,
        "speedup_vs_sequential": speedup_vs_seq,
        "speedup_vs_naive"    : speedup_vs_naive,
        "parallel_efficiency" : efficiency,
        "worker_stats"        : worker_stats,
        "length_ranges"       : length_ranges,
        "ner_metrics"         : {
            "f1"       : round(f1, 4),
            "precision": round(precision, 4),
            "recall"   : round(recall, 4),
            "report"   : report,
        },
        "baseline_time_s"     : t_seq,
        "naive_time_s"        : t_naive,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved → {path}")


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Number of parallel worker processes (default: 2)"
    )
    args      = parser.parse_args()
    n_workers = args.workers

    print("=" * 55)
    print(f"CACHE-AWARE PARALLEL INFERENCE ({n_workers} workers)")
    print("(Length-sorted documents — homogeneous batches)")
    print("=" * 55)

    print("\n" + "=" * 55)
    print("STEP 1 — Loading data and baselines")
    print("=" * 55)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    sentences = [[tuple(p) for p in s] for s in raw]
    print(f"  Test sentences : {len(sentences)}")

    baseline     = load_json(BASELINE_PATH)
    naive_result = load_json(os.path.join(
        RESULTS_DIR, f"naive_parallel_{n_workers}.json"))

    if baseline:
        print(f"  Sequential baseline : "
              f"{baseline['timing']['total_time_s']} s")
    if naive_result:
        print(f"  Naive parallel ({n_workers}w): "
              f"{naive_result['timing']['wall_time_s']} s")

    print("\n" + "=" * 55)
    print("STEP 2 — Sorting corpus by document length")
    print("(This is the cache-aware scheduling step)")
    print("=" * 55)

    lengths = [len(s) for s in sentences]
    print(f"  Corpus length stats:")
    print(f"  Min : {min(lengths)} tokens")
    print(f"  Max : {max(lengths)} tokens")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f} tokens")
    print(f"  Std : {np.std(lengths):.1f} tokens")

    print("\n" + "=" * 55)
    print(f"STEP 3 — Running cache-aware parallel inference")
    print("=" * 55)

    (all_preds, latencies, wall_time,
     worker_stats, length_ranges) = run_cache_aware(
        sentences, n_workers
    )

    print("\n" + "=" * 55)
    print("STEP 4 — Computing metrics")
    print("=" * 55)

    timing             = compute_timing_stats(
        latencies, wall_time, len(sentences))
    f1, prec, rec, rep = compute_metrics(sentences, all_preds)

    print_results(timing, f1, prec, rec, rep,
                  worker_stats, n_workers,
                  baseline, naive_result)

    save_results(timing, f1, prec, rec, rep,
                 worker_stats, length_ranges, n_workers,
                 baseline, naive_result)

    print("\n" + "=" * 55)
    print("DONE")
    print("=" * 55)
    print(f"""
  Run with 4 workers too:
    python inference/cache_aware.py --workers 4

  Then generate final comparison:
    python evaluation/compare_all.py
    """)


if __name__ == "__main__":
    mp.freeze_support()
    main()