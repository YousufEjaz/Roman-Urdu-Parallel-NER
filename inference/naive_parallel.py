"""
inference/naive_parallel.py
────────────────────────────
Configuration 2: Naive parallel inference.
Splits corpus across N worker processes with NO regard
for document length ordering.

Matches proposal description:
  "documents assigned to threads in sequential order
   without regard for document length"

Run from project root:
    python inference/naive_parallel.py --workers 2
    python inference/naive_parallel.py --workers 4

Reads  : results/sequential_results.json  (baseline)
Writes : results/naive_parallel_N.json
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────

MODEL_DIR       = os.path.join("model",   "saved")
DATA_PATH       = os.path.join("data",    "roman", "test.json")
LABELS_PATH     = os.path.join("data",    "roman", "labels.json")
RESULTS_DIR     = os.path.join("results")
BASELINE_PATH   = os.path.join(RESULTS_DIR, "sequential_results.json")

MAX_LEN         = 128


# ──────────────────────────────────────────
# WORKER FUNCTION
# Runs in a separate process.
# Each worker loads its own model instance
# — required because PyTorch models are not
# safely shareable across processes.
# ──────────────────────────────────────────

def worker_fn(worker_id, sentences, result_queue,
              model_dir, labels_path, max_len):
    """
    Worker process: loads model, runs inference on
    its assigned chunk, returns predictions + timings.
    """
    # Each worker uses exactly 1 thread
    # Total parallelism = N processes × 1 thread each
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    device = torch.device("cpu")

    # Load labels
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    # Load model — each worker has its own copy
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model     = DistilBertForTokenClassification.from_pretrained(
        model_dir,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)
    model.eval()

    # Run inference on assigned sentences
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
# PARTITION — naive (no ordering)
# Documents split into N chunks in their
# original order — no length sorting
# ──────────────────────────────────────────

def partition_naive(sentences, n_workers):
    """
    Splits sentences into N equal chunks
    preserving original document order.
    This is the naive strategy — no length
    awareness, creates load imbalance.
    """
    chunk_size = math.ceil(len(sentences) / n_workers)
    chunks     = []
    for i in range(n_workers):
        start = i * chunk_size
        end   = min(start + chunk_size, len(sentences))
        chunk = sentences[start:end]
        if chunk:
            chunks.append(chunk)
    return chunks


# ──────────────────────────────────────────
# RUN PARALLEL INFERENCE
# ──────────────────────────────────────────

def run_parallel(sentences, n_workers):
    """
    Launches N worker processes, collects results,
    measures wall-clock time.
    """
    result_queue = mp.Queue()
    chunks       = partition_naive(sentences, n_workers)

    print(f"\n  Launching {len(chunks)} workers...")
    for i, chunk in enumerate(chunks):
        print(f"  Worker {i}: {len(chunk)} documents "
              f"(indices {i * math.ceil(len(sentences)/n_workers)}"
              f"–{i * math.ceil(len(sentences)/n_workers) + len(chunk) - 1})")

    # Start all workers
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

    # Collect results as workers finish
    raw_results = []
    for _ in range(len(chunks)):
        raw_results.append(result_queue.get())

    # Wait for all processes to exit
    for p in processes:
        p.join()

    wall_end  = time.perf_counter()
    wall_time = wall_end - wall_start

    # Sort by worker_id to restore document order
    raw_results.sort(key=lambda x: x["worker_id"])

    # Flatten predictions and latencies
    all_predictions = []
    all_latencies   = []
    for r in raw_results:
        all_predictions.extend(r["predictions"])
        all_latencies.extend(r["latencies"])

    # Per-worker latency stats (load imbalance analysis)
    worker_stats = []
    for r in raw_results:
        lats = np.array(r["latencies"])
        worker_stats.append({
            "worker_id"     : r["worker_id"],
            "num_docs"      : len(r["predictions"]),
            "total_time_ms" : round(float(np.sum(lats)),  2),
            "mean_lat_ms"   : round(float(np.mean(lats)), 2),
            "max_lat_ms"    : round(float(np.max(lats)),  2),
        })

    return all_predictions, all_latencies, wall_time, worker_stats


# ──────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────

def compute_metrics(sentences, predictions):
    true_tags = [[t for _, t in s] for s in sentences]
    f1        = f1_score(true_tags, predictions)
    precision = precision_score(true_tags, predictions)
    recall    = recall_score(true_tags, predictions)
    report    = classification_report(true_tags, predictions, digits=4)
    return f1, precision, recall, report


def compute_timing_stats(latencies, wall_time, n_docs):
    a = np.array(latencies)
    return {
        "wall_time_s"      : round(wall_time, 4),
        "num_documents"    : n_docs,
        "throughput_docs_s": round(n_docs / wall_time, 4),
        "latency_mean_ms"  : round(float(np.mean(a)),              4),
        "latency_median_ms": round(float(np.median(a)),            4),
        "latency_std_ms"   : round(float(np.std(a)),               4),
        "latency_p95_ms"   : round(float(np.percentile(a, 95)),    4),
    }


def load_baseline():
    if not os.path.exists(BASELINE_PATH):
        print(f"  Warning: baseline not found at {BASELINE_PATH}")
        return None
    with open(BASELINE_PATH, "r") as f:
        return json.load(f)


def compute_speedup(wall_time, baseline):
    if baseline is None:
        return None, None
    t_seq      = baseline["timing"]["total_time_s"]
    speedup    = round(t_seq / wall_time, 4)
    efficiency = round(speedup / n_workers_global * 100, 2)
    return speedup, efficiency


# Global for speedup calculation
n_workers_global = 1


# ──────────────────────────────────────────
# PRINT AND SAVE
# ──────────────────────────────────────────

def print_results(timing, f1, precision, recall, report,
                  worker_stats, speedup, efficiency,
                  n_workers, baseline):

    print("\n" + "=" * 55)
    print(f"NAIVE PARALLEL RESULTS ({n_workers} workers)")
    print("=" * 55)

    print(f"\n  ── Timing ──")
    print(f"  Wall time         : {timing['wall_time_s']:.2f} s")
    print(f"  Documents         : {timing['num_documents']}")
    print(f"  Throughput        : "
          f"{timing['throughput_docs_s']:.2f} docs/sec")
    print(f"  Avg latency       : "
          f"{timing['latency_mean_ms']:.2f} ms/doc")

    if baseline:
        t_seq = baseline["timing"]["total_time_s"]
        print(f"\n  ── Speedup vs Sequential ──")
        print(f"  Sequential time   : {t_seq:.2f} s")
        print(f"  Parallel time     : {timing['wall_time_s']:.2f} s")
        print(f"  Speedup           : {speedup:.4f}x")
        print(f"  Parallel efficiency: {efficiency:.2f}%")

        # Amdahl's Law analysis
        if speedup and speedup > 1:
            p = (1 - 1/speedup) / (1 - 1/n_workers)
            p = min(max(p, 0), 1)
            print(f"\n  ── Amdahl's Law ──")
            print(f"  Parallelizable fraction (p): {p:.4f}")
            print(f"  Theoretical max speedup    : "
                  f"{1/(1-p):.4f}x (infinite threads)")
            for t in [2, 4, 8]:
                theoretical = 1 / ((1-p) + p/t)
                print(f"  Theoretical speedup @{t:>2}    : "
                      f"{theoretical:.4f}x")

    print(f"\n  ── Worker Load Balance ──")
    print(f"  {'Worker':<8} {'Docs':<8} {'Total ms':<12} "
          f"{'Mean ms':<12} {'Max ms':<10}")
    print("  " + "─" * 52)
    for ws in worker_stats:
        print(f"  {ws['worker_id']:<8} {ws['num_docs']:<8} "
              f"{ws['total_time_ms']:<12} "
              f"{ws['mean_lat_ms']:<12} "
              f"{ws['max_lat_ms']:<10}")

    imbalance = max(w["total_time_ms"] for w in worker_stats) - \
                min(w["total_time_ms"] for w in worker_stats)
    print(f"\n  Load imbalance    : {imbalance:.2f} ms")
    print(f"  (difference between slowest and fastest worker)")
    print(f"  NOTE: Cache-aware parallel will reduce this.")

    print(f"\n  ── NER Accuracy ──")
    print(f"  F1 Score          : {f1:.4f}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"\n── Per-class Report ──\n")
    print(report)


def save_results(timing, f1, precision, recall, report,
                 worker_stats, speedup, efficiency,
                 n_workers, baseline):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR,
                        f"naive_parallel_{n_workers}.json")
    results = {
        "config"           : "naive_parallel",
        "num_workers"      : n_workers,
        "timing"           : timing,
        "speedup"          : speedup,
        "parallel_efficiency": efficiency,
        "worker_stats"     : worker_stats,
        "ner_metrics"      : {
            "f1"       : round(f1, 4),
            "precision": round(precision, 4),
            "recall"   : round(recall, 4),
            "report"   : report,
        },
        "baseline_time_s"  : baseline["timing"]["total_time_s"]
                             if baseline else None,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved → {path}")


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

def main():
    global n_workers_global

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Number of parallel worker processes (default: 2)"
    )
    args      = parser.parse_args()
    n_workers = args.workers
    n_workers_global = n_workers

    print("=" * 55)
    print(f"NAIVE PARALLEL INFERENCE ({n_workers} workers)")
    print("(No length ordering — documents in original order)")
    print("=" * 55)

    # Load data
    print("\n" + "=" * 55)
    print("STEP 1 — Loading data and baseline")
    print("=" * 55)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    sentences = [[tuple(p) for p in s] for s in raw]
    print(f"  Test sentences : {len(sentences)}")

    baseline = load_baseline()
    if baseline:
        print(f"  Baseline time  : "
              f"{baseline['timing']['total_time_s']} s")
    else:
        print("  No baseline found — speedup not computed")

    # Run
    print("\n" + "=" * 55)
    print(f"STEP 2 — Running naive parallel inference")
    print("=" * 55)
    print(f"  Strategy: split corpus into {n_workers} equal chunks")
    print(f"  No sorting — documents in original order")

    mem_before = psutil.Process(os.getpid()).memory_info().rss / 1e6

    all_preds, latencies, wall_time, worker_stats = run_parallel(
        sentences, n_workers
    )

    mem_after = psutil.Process(os.getpid()).memory_info().rss / 1e6

    # Metrics
    print("\n" + "=" * 55)
    print("STEP 3 — Computing metrics")
    print("=" * 55)

    timing             = compute_timing_stats(latencies, wall_time,
                                              len(sentences))
    f1, prec, rec, rep = compute_metrics(sentences, all_preds)
    speedup, efficiency = compute_speedup(wall_time, baseline)

    print_results(timing, f1, prec, rec, rep,
                  worker_stats, speedup, efficiency,
                  n_workers, baseline)

    save_results(timing, f1, prec, rec, rep,
                 worker_stats, speedup, efficiency,
                 n_workers, baseline)

    print("\n" + "=" * 55)
    print("DONE")
    print("=" * 55)
    print(f"""
  Run again with more workers:
    python inference/naive_parallel.py --workers 4

  Then run cache-aware:
    python inference/cache_aware.py --workers 2
    python inference/cache_aware.py --workers 4
    """)


if __name__ == "__main__":
    mp.freeze_support()   # required on Windows
    main()