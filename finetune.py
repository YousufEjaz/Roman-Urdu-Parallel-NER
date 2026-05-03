"""
finetune.py
───────────
Fine-tunes multilingual DistilBERT on the Roman Urdu NER dataset.
  - Model  : distilbert-base-multilingual-cased
  - Task   : Token classification (NER)
  - Data   : data/roman/{train,val,test}.json
  - Metrics: F1, Precision, Recall per entity class
  - Output : model/saved/

Run from project root:
    python finetune.py
"""

import os
import json
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from collections import defaultdict


# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────

DATA_DIR       = os.path.join("data", "roman")
MODEL_SAVE_DIR = os.path.join("model", "saved")
LOG_DIR        = os.path.join("results")

MODEL_NAME     = "distilbert-base-multilingual-cased"

# Training hyperparameters
# Kept conservative for CPU training on weak system
MAX_LEN        = 128       # max token sequence length
BATCH_SIZE     = 16        # reduce to 8 if you run out of memory
EPOCHS         = 10        # early stopping will cut this short
LEARNING_RATE  = 5e-5
WARMUP_STEPS   = 50
WEIGHT_DECAY   = 0.01

# Early stopping
PATIENCE       = 3         # stop if val F1 doesn't improve for 3 epochs

# Force CPU — this project runs on CPU only
DEVICE = torch.device("cpu")


# ──────────────────────────────────────────
# STEP 1 — Load data
# ──────────────────────────────────────────

def load_json_split(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Each item is a list of [word, tag] pairs
    sentences = [[tuple(pair) for pair in sent] for sent in raw]
    return sentences


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_data():
    print("  Loading splits...")
    train = load_json_split(os.path.join(DATA_DIR, "train.json"))
    val   = load_json_split(os.path.join(DATA_DIR, "val.json"))
    test  = load_json_split(os.path.join(DATA_DIR, "test.json"))
    labels = load_labels(os.path.join(DATA_DIR, "labels.json"))

    print(f"  Train sentences : {len(train)}")
    print(f"  Val sentences   : {len(val)}")
    print(f"  Test sentences  : {len(test)}")
    print(f"  Labels          : {labels}")

    return train, val, test, labels


# ──────────────────────────────────────────
# STEP 2 — Dataset class
# Handles subword tokenization alignment
# ──────────────────────────────────────────

class NERDataset(Dataset):
    """
    Converts (word, tag) sentence lists into
    DistilBERT input tensors with correct label
    alignment for subword tokenization.

    Key challenge: DistilBERT splits words into
    subword pieces. We label the first subword
    with the real tag and mark continuation
    subwords with -100 (ignored in loss).
    """

    def __init__(self, sentences, tokenizer, label2id, max_len=MAX_LEN):
        self.sentences  = sentences
        self.tokenizer  = tokenizer
        self.label2id   = label2id
        self.max_len    = max_len
        self.encodings  = []
        self._encode_all()

    def _encode_all(self):
        for sentence in self.sentences:
            words = [w for w, _ in sentence]
            tags  = [t for _, t in sentence]
            encoding = self._encode_sentence(words, tags)
            self.encodings.append(encoding)

    def _encode_sentence(self, words, tags):
        """
        Tokenizes words and aligns NER tags to subword tokens.
        Returns dict with input_ids, attention_mask, labels.
        """
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        word_ids = tokenized.word_ids(batch_index=0)
        labels   = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens [CLS], [SEP], [PAD] → ignore
                labels.append(-100)
            elif word_id != prev_word_id:
                # First subword of a word → real label
                labels.append(self.label2id[tags[word_id]])
            else:
                # Continuation subword → ignore in loss
                labels.append(-100)
            prev_word_id = word_id

        return {
            "input_ids"     : tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels"        : torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


# ──────────────────────────────────────────
# STEP 3 — Evaluation
# ──────────────────────────────────────────

def evaluate(model, dataloader, id2label):
    """
    Runs model on a dataloader and computes
    F1, Precision, Recall using seqeval.
    Returns (f1, precision, recall, report_str).
    """
    model.eval()
    all_true  = []
    all_pred  = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits     = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Collect predictions and true labels
            # Skip positions marked -100 (special/pad tokens)
            for pred_seq, label_seq in zip(predictions, labels):
                true_tags = []
                pred_tags = []
                for p, l in zip(pred_seq, label_seq):
                    if l.item() == -100:
                        continue
                    true_tags.append(id2label[l.item()])
                    pred_tags.append(id2label[p.item()])
                all_true.append(true_tags)
                all_pred.append(pred_tags)

    f1        = f1_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred)
    recall    = recall_score(all_true, all_pred)
    report    = classification_report(all_true, all_pred, digits=4)

    return f1, precision, recall, report


# ──────────────────────────────────────────
# STEP 4 — Training loop
# ──────────────────────────────────────────

def train(model, train_loader, val_loader, id2label, num_epochs=EPOCHS):
    """
    Full training loop with:
    - Linear warmup scheduler
    - Per-epoch validation F1
    - Early stopping based on val F1
    - Best model checkpoint saving
    """
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * num_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    best_val_f1     = -1.0
    patience_count  = 0
    history         = []

    print(f"\n  Training on {DEVICE}")
    print(f"  Epochs         : {num_epochs}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  Learning rate  : {LEARNING_RATE}")
    print(f"  Early stopping : patience={PATIENCE}\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss   = 0.0
        epoch_start  = time.time()
        num_batches  = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            # Gradient clipping — stabilizes training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss  += loss.item()
            num_batches += 1

            # Progress indicator every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg = epoch_loss / num_batches
                print(f"    Epoch {epoch} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg:.4f}", end="\r")

        # ── Epoch-end evaluation ──
        epoch_time = time.time() - epoch_start
        avg_loss   = epoch_loss / num_batches

        val_f1, val_prec, val_rec, _ = evaluate(
            model, val_loader, id2label
        )

        print(f"\n  Epoch {epoch:>2}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Prec: {val_prec:.4f} | "
              f"Rec: {val_rec:.4f} | "
              f"Time: {epoch_time:.1f}s")

        history.append({
            "epoch"    : epoch,
            "loss"     : avg_loss,
            "val_f1"   : val_f1,
            "val_prec" : val_prec,
            "val_rec"  : val_rec,
        })

        # ── Save best model checkpoint ──
        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            model.save_pretrained(MODEL_SAVE_DIR)
            print(f"  ✓ New best model saved "
                  f"(val F1 = {best_val_f1:.4f})")
        else:
            patience_count += 1
            print(f"  No improvement. "
                  f"Patience: {patience_count}/{PATIENCE}")

        # ── Early stopping ──
        if patience_count >= PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            print(f"  Best val F1: {best_val_f1:.4f}")
            break

    return history, best_val_f1


# ──────────────────────────────────────────
# STEP 5 — Final test evaluation
# ──────────────────────────────────────────

def final_evaluation(test_loader, id2label, label2id, labels):
    """
    Loads the best saved model and evaluates on test set.
    Reports F1, Precision, Recall per entity class.
    This is the NLP contribution result for the report.
    """
    print("\n  Loading best saved model for test evaluation...")
    model = DistilBertForTokenClassification.from_pretrained(
        MODEL_SAVE_DIR,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(DEVICE)

    f1, precision, recall, report = evaluate(
        model, test_loader, id2label
    )

    print("\n" + "=" * 55)
    print("TEST SET RESULTS")
    print("=" * 55)
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print("\n── Per-class Report ──\n")
    print(report)

    return f1, precision, recall, report


# ──────────────────────────────────────────
# STEP 6 — Save training history and results
# ──────────────────────────────────────────

def save_results(history, test_f1, test_prec, test_rec, report, labels, id2label):
    os.makedirs(LOG_DIR, exist_ok=True)

    results = {
        "model"         : MODEL_NAME,
        "max_len"       : MAX_LEN,
        "batch_size"    : BATCH_SIZE,
        "learning_rate" : LEARNING_RATE,
        "test_f1"       : test_f1,
        "test_precision": test_prec,
        "test_recall"   : test_rec,
        "per_class_report": report,
        "training_history": history,
    }

    path = os.path.join(LOG_DIR, "finetune_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved → {path}")

    # Also save tokenizer alongside model
    # (needed by inference pipeline)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    # Save label maps separately for inference scripts
    label_map_path = os.path.join(MODEL_SAVE_DIR, "label_map.json")
    label_map = {
        "labels": labels,
        "id2label": {str(k): v for k, v in id2label.items()},
        "label2id": {v: k for k, v in id2label.items()},
    }
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"  Label map saved → {label_map_path}")
    print(f"  Model directory → {MODEL_SAVE_DIR}/")


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

def main():
    print("=" * 55)
    print("STEP 1 — Loading data")
    print("=" * 55)

    train_sents, val_sents, test_sents, labels = load_all_data()

    # Build label ↔ id mappings
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}

    print(f"\n  label2id: {label2id}")


    print("\n" + "=" * 55)
    print("STEP 2 — Loading tokenizer and model")
    print("=" * 55)

    print(f"  Model: {MODEL_NAME}")
    print("  Downloading tokenizer (first run only)...")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # Save tokenizer into model dir immediately
    # so inference scripts can find it
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print("  Tokenizer saved.")

    model = DistilBertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")


    print("\n" + "=" * 55)
    print("STEP 3 — Building datasets and dataloaders")
    print("=" * 55)

    train_dataset = NERDataset(train_sents, tokenizer, label2id)
    val_dataset   = NERDataset(val_sents,   tokenizer, label2id)
    test_dataset  = NERDataset(test_sents,  tokenizer, label2id)

    print(f"  Train dataset size : {len(train_dataset)}")
    print(f"  Val dataset size   : {len(val_dataset)}")
    print(f"  Test dataset size  : {len(test_dataset)}")

    # num_workers=0 required on Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")


    print("\n" + "=" * 55)
    print("STEP 4 — Fine-tuning")
    print("=" * 55)
    

    train_start = time.time()
    history, best_val_f1 = train(
        model, train_loader, val_loader, id2label
    )
    train_time = time.time() - train_start

    print(f"\n  Total training time : {train_time/60:.1f} minutes")
    print(f"  Best val F1         : {best_val_f1:.4f}")


    print("\n" + "=" * 55)
    print("STEP 5 — Final test evaluation")
    print("=" * 55)

    test_f1, test_prec, test_rec, report = final_evaluation(
        test_loader, id2label, label2id, labels
    )


    print("\n" + "=" * 55)
    print("STEP 6 — Saving results")
    print("=" * 55)

    save_results(
        history,
        test_f1,
        test_prec,
        test_rec,
        report,
        labels,
        id2label,
    )


    print("\n" + "=" * 55)
    print("FINE-TUNING COMPLETE")
    print("=" * 55)
    print(f"""
  Model saved to  : {MODEL_SAVE_DIR}/
  Results saved to: {LOG_DIR}/finetune_results.json

  Summary:
    Best Val F1  : {best_val_f1:.4f}
    Test F1      : {test_f1:.4f}
    Test Prec    : {test_prec:.4f}
    Test Recall  : {test_rec:.4f}

  Next step: inference/sequential.py
    """)


if __name__ == "__main__":
    main()
